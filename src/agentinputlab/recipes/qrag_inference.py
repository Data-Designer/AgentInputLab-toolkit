"""Native Q-RAG inference under a distinct, trained policy identity."""

from __future__ import annotations

import gc
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from agentinputlab.algorithms.acquire.qrag.common_support import (
    LocalQRAGFrozenPolicyRuntime,
    QRAGCommonSupport,
    QRAGCommonSupportConfig,
    _source_import,
    _unused_vllm_import_compatibility,
)
from agentinputlab.contracts import ExternalDependencyError, TraceEvent, TracePhase, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .qrag import (
    QRAG_UPSTREAM_COMMIT,
    RELEASE,
    SOURCE_FILES,
    QRAGFitControls,
    QRAGTrainedPolicyArtifact,
    _file,
    _sha,
    validate_qrag_initial_model,
    validate_qrag_training_source,
)
from .recomp_extractive import _LOCK


def _read(path):
    import json

    return json.loads(Path(path).read_text())


def _checkpoint_path(artifact, selection):
    if selection == "final":
        return _file(artifact.checkpoint, "final checkpoint")
    if selection != "native_best":
        raise ValidationError("Q-RAG checkpoint selection must be native_best or final")
    terminal = _read(artifact.root / "fit-terminal.json")
    events = [
        e
        for e in terminal["checkpoint_storage_events"]
        if Path(e["logical_path"]).name == "model_best.pt"
    ]
    if not events:
        raise ValidationError("Q-RAG fit did not save a native best checkpoint")
    event = events[-1]
    step = event["native_update_step"]
    if type(step) is not int or not 1 <= step <= len(terminal["updates"]):
        raise ValidationError("Q-RAG native best checkpoint update is invalid")
    expected = f"update-{step:08d}.pt"
    if Path(event["immutable_snapshot_path"]).name != expected:
        raise ValidationError("Q-RAG native best snapshot does not match its update")
    # Resolve within the artifact, not the historical absolute training path.
    path = _file(artifact.root / "native-checkpoints" / expected, "native best checkpoint")
    if not path.is_relative_to(artifact.root):
        raise ValidationError("Q-RAG checkpoint escapes its trained artifact")
    return path


def _stamps(paths):
    try:
        rows = []
        for p in paths:
            if p.is_symlink() or not p.is_file():
                raise ValidationError("Q-RAG trained deployment file changed or is symlinked")
            s = p.stat()
            rows.append((str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
        return tuple(rows)
    except OSError as exc:
        raise ValidationError("Q-RAG trained deployment file changed or is missing") from exc


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ExternalDependencyError(
            "Q-RAG inference requires pip install 'agentinputlab[qrag-fit]'"
        ) from exc
    return torch


def _load_inference_states(torch, agent, checkpoint):
    """Load native module state and alpha; never restore optimizer or scheduler."""
    names = ("critic", "policy", "v_net_target", "action_embed_target")
    if not isinstance(checkpoint, Mapping) or not set(names) <= checkpoint.keys():
        raise ValidationError("Q-RAG trained checkpoint lacks native inference modules")
    for name in names:
        values, module = checkpoint[name], getattr(agent, name)
        expected = module.state_dict()
        if not isinstance(values, Mapping) or set(values) != set(expected):
            raise ValidationError(f"Q-RAG trained state keys differ: {name}")
        for key, tensor in values.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype
                or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
            ):
                raise ValidationError(
                    f"Q-RAG trained tensor shape/dtype/value differs: {name}.{key}"
                )
        module.load_state_dict(values, strict=True)
        module.eval()
        module.requires_grad_(False)
    if checkpoint.get("random_policy") != {}:
        raise ValidationError("Q-RAG native random policy must have no parameter state")
    alpha = checkpoint.get("alpha")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValidationError("Q-RAG trained policy alpha is invalid")
    agent.alpha = alpha
    agent.random_policy.eval()


class QRAGTrainedPolicyRuntime:
    """The upstream QAEnv/PQN selector with hash-bound fitted native weights.

    A best-checkpoint checksum must be supplied from its retained fit evidence.
    This prevents an unsealed best snapshot from silently becoming a new policy.
    Inputs contain only the question and retrieved documents, never test labels.
    """

    def __init__(
        self,
        artifact,
        *,
        checkout,
        initial_checkpoint,
        initial_manifest,
        initial_manifest_sha256,
        checkpoint_sha256,
        selection="native_best",
        device="cpu",
        source_and_asset_acknowledged=False,
    ):
        if source_and_asset_acknowledged is not True:
            raise ValidationError("Q-RAG deployment requires mounted-source/model acknowledgement")
        if not isinstance(artifact, QRAGTrainedPolicyArtifact):
            raise ValidationError("Q-RAG deployment requires a typed trained artifact")
        if not isinstance(checkpoint_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", checkpoint_sha256
        ):
            raise ValidationError("Q-RAG selected checkpoint requires its expected SHA-256")
        if device not in {"cpu", "cuda:0"}:
            raise ValidationError("Q-RAG deployment requires CPU or one logical CUDA device")
        verified = QRAGTrainedPolicyArtifact.load(artifact.root)
        if verified.artifact_id != artifact.artifact_id:
            raise ValidationError("Q-RAG trained deployment artifact identity differs")
        self.source_root = validate_qrag_training_source(checkout)
        self.initial_root, _ = validate_qrag_initial_model(
            initial_checkpoint, initial_manifest, initial_manifest_sha256
        )
        self.plan = _read(verified.root / "fit-plan.json")
        if (
            self.plan["source_commit"] != QRAG_UPSTREAM_COMMIT
            or self.plan["source_files"] != SOURCE_FILES
            or self.plan["initial_manifest_sha256"] != initial_manifest_sha256
        ):
            raise ValidationError("Q-RAG fitted source/initial model binding differs")
        self.controls = QRAGFitControls(**self.plan["controls"])
        self.checkpoint = _checkpoint_path(verified, selection)
        if _sha(self.checkpoint) != checkpoint_sha256:
            raise ValidationError("Q-RAG selected trained checkpoint checksum differs")
        self.device, self._runtime = device, None
        self.artifact_id, self.selection = verified.artifact_id, selection
        self._paths = [
            verified.root / n for n in ("artifact.json", "fit-plan.json", "fit-terminal.json")
        ]
        self._paths += [self.checkpoint, _file(initial_manifest, "initial manifest")]
        self._paths += sorted(p for p in self.initial_root.rglob("*") if p.is_file())
        self._paths += sorted(
            p
            for p in self.source_root.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
            and ".git" not in p.parts
        )
        self._identity = _stamps(self._paths)
        self._binding = dict(
            runtime="qrag_native_trained_policy.v1",
            release_id=RELEASE,
            trained_artifact_id=self.artifact_id,
            checkpoint_selection=selection,
            checkpoint_sha256=checkpoint_sha256,
            initial_manifest_sha256=initial_manifest_sha256,
            source_commit=QRAG_UPSTREAM_COMMIT,
            safe_weights_only_load=True,
            strict_state_dict_load=True,
            optimizer_state_loaded=False,
            training_allowed=False,
            paper_result_parity_claimed=False,
        )
        self.deployment_id = content_hash(self._binding)
        self._parameters = self._parameter_binding()

    def _parameter_binding(self):
        return (
            self.source_root,
            self.initial_root,
            self.checkpoint,
            self.controls,
            self.device,
            self.artifact_id,
            self.selection,
            self.deployment_id,
            content_hash(self._binding),
        )

    def _verify_binding(self):
        if self._parameter_binding() != self._parameters or _stamps(self._paths) != self._identity:
            raise ValidationError("Q-RAG trained deployment files changed; no silent substitution")

    def _load(self):
        self._verify_binding()
        if self._runtime is not None:
            return self._runtime
        torch = _torch()
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        c = self.controls
        with _LOCK, _unused_vllm_import_compatibility(), _source_import(self.source_root):
            from envs.qa_env import QAEnv
            from envs.text_env import AbsolutePositionProcessor
            from rl.agents.pqn import PQN
            from rl.feedback.feedback import AFeedbackModel, DummyFeedbackModel

            for module_name in (
                "envs.qa_env",
                "envs.text_env",
                "rl.agents.pqn",
                "rl.feedback.feedback",
            ):
                import sys

                module = sys.modules[module_name]
                if not Path(module.__file__).resolve().is_relative_to(self.source_root):
                    raise ValidationError("Q-RAG native module resolved outside its pinned source")

            with initialize_config_dir(
                config_dir=str(self.source_root / "configs"), version_base=None
            ):
                cfg = compose(
                    config_name="training",
                    overrides=[
                        "envs=hotpotqa",
                        "algo=pqn_e5_hotpotqa",
                        f"seed={c.seed}",
                        f"steps_count={c.steps_count}",
                        f"accumulate_grads={c.accumulate_grads}",
                        f"algo.pqn.scheduler.warmup={c.warmup_steps}",
                    ],
                )
            cfg.algo.model.model_name, cfg.algo.model.revision = str(self.initial_root), None
            OmegaConf.resolve(cfg)
            with torch.device(self.device):
                agent = PQN(cfg.algo)
            # Newly trained snapshots contain no OmegaConf pickle globals. Do not
            # invoke upstream PQN.load, add pickle allowlists or load the optimizer.
            if torch.serialization.get_unsafe_globals_in_checkpoint(self.checkpoint):
                raise ValidationError("Q-RAG trained checkpoint contains nondefault pickle globals")
            checkpoint = torch.load(
                self.checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            _load_inference_states(torch, agent, checkpoint)
            del checkpoint

            class Feedback(DummyFeedbackModel):
                def reset(self, obs, info):
                    AFeedbackModel.reset(self, obs, info)

            env = QAEnv(
                dataset=None,
                max_steps=2,
                positions_processor=AbsolutePositionProcessor(),
                action_embed_length=int(cfg.max_action_length),
                max_action_length_in_memory=int(
                    cfg.algo.pqn.hyperparams.max_action_length_in_memory
                ),
                feedback_model=Feedback(),
                separator=" [SEP] ",
                sort_by_index=False,
            )
        gc.collect()
        self._verify_binding()
        self._runtime = (torch, agent, env)
        return self._runtime

    def select(self, request):
        torch = _torch()

        with _LOCK, torch.device(self.device), torch.no_grad():
            precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("high")
            try:
                return LocalQRAGFrozenPolicyRuntime.select(self, request)
            finally:
                torch.set_float32_matmul_precision(precision)

    def fingerprint(self):
        self._verify_binding()
        return freeze_mapping({**self._binding, "deployment_id": self.deployment_id})


class QRAGTrainedPolicyIntervention(QRAGCommonSupport):
    """Common-support document placement bound to one fitted runtime."""

    def __init__(self, runtime, *, config=None):
        if not isinstance(runtime, QRAGTrainedPolicyRuntime):
            raise ValidationError("Q-RAG trained intervention requires its native runtime")
        settings = (
            QRAGCommonSupportConfig(**config)
            if isinstance(config, Mapping)
            else config or QRAGCommonSupportConfig()
        )
        if not isinstance(settings, QRAGCommonSupportConfig):
            raise ValidationError("Q-RAG trained intervention needs typed controls")
        super().__init__(config=settings, release_id=RELEASE)
        self.runtime = runtime

    def propose(self, state, context):
        if context.external_runtime is not None and context.external_runtime is not self.runtime:
            raise ValidationError("Q-RAG trained runtime must not be substituted")
        patch = super().propose(state, replace(context, external_runtime=self.runtime))
        binding = dict(self.runtime.fingerprint())
        operations = tuple(
            replace(op, rationale="select shared evidence with the native fitted Q-RAG policy")
            for op in patch.operations
        )
        return replace(
            patch,
            operations=operations,
            trace=replace(
                patch.trace,
                events=(
                    TraceEvent(TracePhase.PREPARE, "qrag_native_trained_policy", binding),
                    *patch.trace.events,
                ),
            ),
            implementation_identity="native_qrag_pqn_with_trained_artifact",
            metadata={
                **patch.metadata,
                **binding,
                "common_support_adaptation": "trained_value_policy_over_shared_candidate_pool",
            },
        )


__all__ = ["QRAGTrainedPolicyIntervention", "QRAGTrainedPolicyRuntime"]
