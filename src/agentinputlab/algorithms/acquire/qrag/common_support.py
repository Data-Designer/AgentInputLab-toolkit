"""Released Q-RAG value policy adapted to the shared agent arenas.

The upstream worker remains the paper-shaped selector/reader path.  This module
provides a separate common-support intervention: the shared retriever builds an
ordered candidate pool, the released frozen PQN selects two documents with the
official QAEnv/PQN loop, and the unified downstream agent answers from those
real documents.  No gold answer, supporting-fact label, reward, training step,
or upstream reader is available to the intervention.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ContextBlock,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .bridge import (
    QRAG_CHECKPOINT_CONFIG_SHA256,
    QRAG_CHECKPOINT_FILE_SHA256,
    QRAG_CHECKPOINT_REPOSITORY,
    QRAG_CHECKPOINT_REVISION,
    QRAG_EMBEDDER_MODEL_SHA256,
    QRAG_EMBEDDER_REPOSITORY,
    QRAG_EMBEDDER_REVISION,
    QRAG_SOURCE_MANIFEST_SHA256,
    QRAG_UPSTREAM_COMMIT,
)
from .native_axis import (
    QRAG_SINGLE_CANDIDATE_AXIS_COMPAT,
    install_single_candidate_axis_compat,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RECEIPT = "agentinputlab-source-receipt.json"
_EMBEDDER_MANIFEST = "agentinputlab-manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalDependencyError(f"Q-RAG artifact JSON is invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise ExternalDependencyError(f"Q-RAG artifact JSON must be an object: {path}")
    return value


def _real_directory(value: Path, *, label: str) -> Path:
    path = value.expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise ValidationError(f"Q-RAG {label} must be a real directory")
    return path


def _real_file(value: Path, *, label: str) -> Path:
    path = value.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValidationError(f"Q-RAG {label} must be a real file")
    return path


def _safe_file(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ExternalDependencyError("Q-RAG artifact path is invalid")
    path = root / relative
    if path.is_symlink():
        raise ExternalDependencyError("Q-RAG artifact symlinks are forbidden")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ExternalDependencyError("Q-RAG artifact path escapes its root or is missing") from exc
    if not resolved.is_file():
        raise ExternalDependencyError("Q-RAG artifact path is not a file")
    return resolved


@contextmanager
def _source_import(root: Path):
    value = str(root)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)
        else:  # pragma: no cover - defensive against third-party path mutation
            with suppress(ValueError):
                sys.path.remove(value)


@contextmanager
def _unused_vllm_import_compatibility():
    """Allow the released dummy-feedback path without a worker-local vLLM.

    Q-RAG's ``rl.feedback`` package imports its optional local reader at module
    import time, even though the common-support runtime uses only the released
    ``DummyFeedbackModel``. Keep target serving isolated in the external vLLM
    process and fail closed if this unused reader is ever constructed.
    """

    try:
        available = importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        available = "vllm" in sys.modules
    if available:
        yield
        return

    class _UnavailableVLLM:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "Q-RAG common-support execution must not instantiate its unused vLLM reader"
            )

    module = types.ModuleType("vllm")
    module.LLM = _UnavailableVLLM
    module.SamplingParams = _UnavailableVLLM
    sys.modules["vllm"] = module
    try:
        yield
    finally:
        if sys.modules.get("vllm") is module:
            del sys.modules["vllm"]


@dataclass(frozen=True, slots=True)
class QRAGPolicyRequest:
    question: str
    candidates: tuple[RetrievalResult, ...]
    seed: int
    max_steps: int = 2

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.candidates:
            raise ValidationError("Q-RAG policy request requires question and candidates")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError("Q-RAG policy seed must be non-negative")
        if self.max_steps != 2:
            raise ValidationError("released Q-RAG policy requires exactly two selection steps")
        object.__setattr__(self, "candidates", tuple(self.candidates))


@runtime_checkable
class QRAGCommonSupportRuntime(Protocol):
    def select(self, request: QRAGPolicyRequest) -> Mapping[str, Any]:
        """Select two candidate indices with the released value policy."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return checksum-closed source, checkpoint, and embedder identity."""


@dataclass(slots=True)
class LocalQRAGFrozenPolicyRuntime:  # pragma: no cover - L20 GPU acceptance
    """Lazy inference-only runtime around the pinned upstream QAEnv and PQN."""

    source_root: Path
    checkpoint_root: Path
    embedder_root: Path
    compatibility_manifest_path: Path
    compatibility_manifest_sha256: str
    embedder_manifest_sha256: str
    device: str = "cuda"
    worker_id: str = "qrag-common-support-local"
    single_candidate_axis_compat: bool = False
    _runtime: tuple[Any, Any, Any] | None = field(init=False, default=None, repr=False)
    _verified: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.single_candidate_axis_compat, bool):
            raise ValidationError("Q-RAG single_candidate_axis_compat must be boolean")
        self.source_root = _real_directory(self.source_root, label="source_root")
        self.checkpoint_root = _real_directory(self.checkpoint_root, label="checkpoint_root")
        self.embedder_root = _real_directory(self.embedder_root, label="embedder_root")
        self.compatibility_manifest_path = _real_file(
            self.compatibility_manifest_path,
            label="compatibility_manifest_path",
        )
        for label, value in (
            ("compatibility_manifest_sha256", self.compatibility_manifest_sha256),
            ("embedder_manifest_sha256", self.embedder_manifest_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValidationError(f"Q-RAG {label} must be a lowercase SHA-256")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValidationError("Q-RAG device must not be empty")

    def _verify_artifacts(self) -> Mapping[str, Any]:
        manifest = _json(self.compatibility_manifest_path)
        if content_hash(manifest) != self.compatibility_manifest_sha256:
            raise ExternalDependencyError("Q-RAG compatibility manifest checksum mismatch")
        source = manifest.get("source")
        checkpoint = manifest.get("checkpoint")
        embedder = manifest.get("embedder")
        if not all(isinstance(row, Mapping) for row in (source, checkpoint, embedder)):
            raise ExternalDependencyError("Q-RAG compatibility manifest sections are invalid")
        assert isinstance(source, Mapping)
        assert isinstance(checkpoint, Mapping)
        assert isinstance(embedder, Mapping)
        if (
            source.get("commit") != QRAG_UPSTREAM_COMMIT
            or source.get("content_manifest_sha256") != QRAG_SOURCE_MANIFEST_SHA256
            or checkpoint.get("repository") != QRAG_CHECKPOINT_REPOSITORY
            or checkpoint.get("revision") != QRAG_CHECKPOINT_REVISION
            or embedder.get("repository") != QRAG_EMBEDDER_REPOSITORY
            or embedder.get("revision") != QRAG_EMBEDDER_REVISION
            or embedder.get("model_safetensors_sha256") != QRAG_EMBEDDER_MODEL_SHA256
        ):
            raise ExternalDependencyError("Q-RAG compatibility manifest identity drifted")
        required = source.get("required_files")
        if not isinstance(required, Mapping) or not required:
            raise ExternalDependencyError("Q-RAG source required-file manifest is invalid")
        for relative, expected in required.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ExternalDependencyError("Q-RAG source required-file row is invalid")
            if _sha256_file(_safe_file(self.source_root, relative)) != expected:
                raise ExternalDependencyError(f"Q-RAG source hash mismatch: {relative}")
        self._verify_source_transport(source)

        model_path = _safe_file(self.checkpoint_root, "model_best.pt")
        config_path = _safe_file(self.checkpoint_root, "config.yaml")
        if model_path.stat().st_size != checkpoint.get("model_best_size"):
            raise ExternalDependencyError("Q-RAG checkpoint size mismatch")
        if _sha256_file(model_path) != QRAG_CHECKPOINT_FILE_SHA256:
            raise ExternalDependencyError("Q-RAG checkpoint checksum mismatch")
        if _sha256_file(config_path) != QRAG_CHECKPOINT_CONFIG_SHA256:
            raise ExternalDependencyError("Q-RAG checkpoint configuration checksum mismatch")
        self._verify_embedder_manifest()
        self._verified = True
        return manifest

    def _verify_source_transport(self, source: Mapping[str, Any]) -> None:
        git_dir = self.source_root / ".git"
        if git_dir.exists():
            completed = subprocess.run(
                ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if completed.returncode or completed.stdout.strip() != QRAG_UPSTREAM_COMMIT:
                raise ExternalDependencyError("Q-RAG source checkout commit mismatch")
            return
        receipt = _json(_safe_file(self.source_root, _SOURCE_RECEIPT))
        expected = {
            "repository": source.get("repository"),
            "commit": source.get("commit"),
            "tree": source.get("tree"),
            "transport": "jsdelivr_commit_snapshot",
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ExternalDependencyError("Q-RAG source snapshot identity mismatch")
        self._verify_file_closure(
            self.source_root,
            receipt.get("files"),
            excluded={_SOURCE_RECEIPT},
            label="source snapshot",
        )

    def _verify_embedder_manifest(self) -> None:
        path = _safe_file(self.embedder_root, _EMBEDDER_MANIFEST)
        manifest = _json(path)
        if content_hash(manifest) != self.embedder_manifest_sha256:
            raise ExternalDependencyError("Q-RAG embedder manifest checksum mismatch")
        if (
            manifest.get("repository") != QRAG_EMBEDDER_REPOSITORY
            or manifest.get("revision") != QRAG_EMBEDDER_REVISION
        ):
            raise ExternalDependencyError("Q-RAG embedder repository/revision mismatch")
        if _sha256_file(_safe_file(self.embedder_root, "model.safetensors")) != (
            QRAG_EMBEDDER_MODEL_SHA256
        ):
            raise ExternalDependencyError("Q-RAG embedder weights checksum mismatch")
        self._verify_file_closure(
            self.embedder_root,
            manifest.get("files"),
            excluded={_EMBEDDER_MANIFEST},
            label="embedder snapshot",
        )

    @staticmethod
    def _verify_file_closure(
        root: Path,
        rows: Any,
        *,
        excluded: set[str],
        label: str,
    ) -> None:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
            raise ExternalDependencyError(f"Q-RAG {label} file receipt is invalid")
        declared: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size"}:
                raise ExternalDependencyError(f"Q-RAG {label} file row is invalid")
            relative, digest, size = row.get("path"), row.get("sha256"), row.get("size")
            if (
                not isinstance(relative, str)
                or relative in excluded
                or relative in declared
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ExternalDependencyError(f"Q-RAG {label} file row is invalid")
            path = _safe_file(root, relative)
            if path.stat().st_size != size or _sha256_file(path) != digest:
                raise ExternalDependencyError(f"Q-RAG {label} file checksum mismatch: {relative}")
            declared.add(relative)
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
            and str(path.relative_to(root)) not in excluded
            and ".cache" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".lock", ".incomplete"}
        }
        if actual != declared:
            raise ExternalDependencyError(f"Q-RAG {label} file closure mismatch")

    def _load(self) -> tuple[Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self._verified:
            self._verify_artifacts()
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        try:
            import torch
            from omegaconf import OmegaConf
            from omegaconf.base import ContainerMetadata, Metadata
            from omegaconf.listconfig import ListConfig
            from omegaconf.nodes import AnyNode
        except ImportError as exc:
            raise ExternalDependencyError(
                "Q-RAG local runtime requires torch, numpy, omegaconf, hydra-core, "
                "rotary-embedding-torch, sortedcontainers, nltk, and transformers"
            ) from exc
        with _unused_vllm_import_compatibility(), _source_import(self.source_root):
            try:
                from envs.qa_env import QAEnv
                from envs.text_env import AbsolutePositionProcessor
                from rl.agents.pqn import PQN
                from rl.feedback.feedback import AFeedbackModel, DummyFeedbackModel
            except ImportError as exc:
                raise ExternalDependencyError("unable to import pinned Q-RAG source") from exc

        class RunnableDummyFeedbackModel(DummyFeedbackModel):
            """Repair the upstream missing abstract reset without changing feedback semantics."""

            def reset(self, obs: Any, info: Any) -> None:
                AFeedbackModel.reset(self, obs, info)

        cfg = OmegaConf.load(self.checkpoint_root / "config.yaml")
        cfg.algo.model.model_name = str(self.embedder_root)
        cfg.algo.model.revision = None
        OmegaConf.resolve(cfg)
        torch.set_default_device(self.device)
        torch.set_float32_matmul_precision("high")
        with _source_import(self.source_root):
            agent = PQN(cfg.algo)
        # The released checkpoint contains an OmegaConf ListConfig under its
        # scheduler state. Keep weights-only loading and scope the smallest
        # inert constructor allowlist needed to deserialize that container;
        # arbitrary upstream Python globals remain rejected.
        safe_checkpoint_globals = [
            ListConfig,
            ContainerMetadata,
            Metadata,
            Any,
            list,
            defaultdict,
            dict,
            int,
            AnyNode,
        ]
        with torch.serialization.safe_globals(safe_checkpoint_globals):
            checkpoint = torch.load(
                self.checkpoint_root / "model_best.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        required = {
            "critic": agent.critic,
            "policy": agent.policy,
            "v_net_target": agent.v_net_target,
            "action_embed_target": agent.action_embed_target,
        }
        if not isinstance(checkpoint, Mapping) or not set(required).issubset(checkpoint):
            raise ExternalDependencyError("Q-RAG checkpoint lacks inference state dictionaries")
        for name, module in required.items():
            module.load_state_dict(checkpoint[name], strict=True)
            module.eval()
        agent.random_policy.eval()
        if self.single_candidate_axis_compat:
            install_single_candidate_axis_compat(agent.policy, self.source_root)
        del checkpoint
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        env = QAEnv(
            dataset=None,
            max_steps=2,
            positions_processor=AbsolutePositionProcessor(),
            action_embed_length=int(cfg.max_action_length),
            max_action_length_in_memory=int(cfg.algo.pqn.hyperparams.max_action_length_in_memory),
            feedback_model=RunnableDummyFeedbackModel(),
            separator=" [SEP] ",
            sort_by_index=False,
        )
        self._runtime = (torch, agent, env)
        return self._runtime

    def select(self, request: QRAGPolicyRequest) -> Mapping[str, Any]:
        if not isinstance(request, QRAGPolicyRequest):
            raise TypeError("Q-RAG runtime requires QRAGPolicyRequest")
        torch, agent, env = self._load()
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ExternalDependencyError("Q-RAG runtime requires numpy") from exc
        random.seed(request.seed)
        np.random.seed(request.seed)
        torch.manual_seed(request.seed)
        if str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(request.seed)
            torch.cuda.synchronize()
        candidate_ids = tuple(row.document.doc_id for row in request.candidates)
        chunks = tuple(_policy_candidate_text(row) for row in request.candidates)
        sample = {
            "id": content_hash({"question": request.question, "candidates": candidate_ids}),
            "question": request.question,
            "answer": "",
            "chunks": chunks,
            "sf_idx": [],
        }
        started = time.perf_counter()
        state = env.reset(sample)
        embeds, embeds_target = env.get_extra_embeds(
            agent.action_tokenizer,
            agent.critic.action_embed,
            agent.action_embed_target,
        )
        steps: list[dict[str, Any]] = []
        done = False
        while not done and len(steps) < request.max_steps:
            available = tuple(sorted(int(index) for index in state.available_ids))
            embeds = env.update_embeds(embeds, agent.critic.action_embed)
            embeds_target = env.update_embeds(embeds_target, agent.action_embed_target)
            action, q_values, _ = agent.select_action(
                state,
                embeds["rope"],
                embeds_target["rope"],
                random=False,
                evaluate=True,
            )
            q_row = q_values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if q_row.numel() != len(candidate_ids) or action not in available:
                raise AlgorithmOutputError("Q-RAG policy returned an invalid candidate action")
            selected_q = float(q_row[action].item())
            if not math.isfinite(selected_q):
                raise AlgorithmOutputError("Q-RAG policy returned a non-finite selected value")
            state, _, _, done = env.step(action)
            selected = tuple(int(index) for index in state.item_ids)
            steps.append(
                {
                    "index": len(steps),
                    "available_indices": available,
                    "q_values": tuple(float(q_row[index].item()) for index in available),
                    "action_index": int(action),
                    "selected_q_value": selected_q,
                    "selected_indices": selected,
                    "selected_document_ids": tuple(candidate_ids[index] for index in selected),
                }
            )
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        selected = tuple(int(index) for index in state.item_ids)
        expected_steps = min(2, len(candidate_ids))
        if len(steps) != expected_steps or len(selected) != expected_steps:
            raise AlgorithmOutputError("Q-RAG policy did not complete its fixed selection horizon")
        return {
            "status": "ok",
            "trajectory": tuple(steps),
            "trajectory_sha256": content_hash(steps),
            "selected_indices": selected,
            "selected_document_ids": tuple(candidate_ids[index] for index in selected),
            "candidate_order_sha256": content_hash(candidate_ids),
            "candidate_text_sha256": content_hash(chunks),
            "usage": {
                "model_calls": 0,
                "retrieval_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_calls": 2 + 2 * len(steps),
                # CPU process time is recorded by the runner; do not also
                # mislabel the CPU policy's wall time as accelerator usage.
                "accelerator_seconds": 0.0 if self.device.split(":", 1)[0] == "cpu" else elapsed,
            },
            "policy_steps": len(steps),
            "attestation": {
                "official_candidate_builder_called": True,
                "official_pqn_select_action_called": True,
                "official_qa_env_step_called": True,
                "safe_weights_only_load": True,
                "strict_state_dict_load": True,
                "checkpoint_eval_mode": True,
                "dummy_feedback_used": True,
                "gold_answer_used": False,
                "supporting_fact_labels_used": False,
                "reward_used": False,
                "training_called": False,
                "stop_action_used": False,
                "seed_applied": True,
            },
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "runtime": "qrag_released_pqn_common_support.v1",
                "source_repository": "griver/Q-RAG",
                "source_commit": QRAG_UPSTREAM_COMMIT,
                "source_manifest_sha256": QRAG_SOURCE_MANIFEST_SHA256,
                "checkpoint_repository": QRAG_CHECKPOINT_REPOSITORY,
                "checkpoint_revision": QRAG_CHECKPOINT_REVISION,
                "checkpoint_file_sha256": QRAG_CHECKPOINT_FILE_SHA256,
                "checkpoint_config_sha256": QRAG_CHECKPOINT_CONFIG_SHA256,
                "embedder_repository": QRAG_EMBEDDER_REPOSITORY,
                "embedder_revision": QRAG_EMBEDDER_REVISION,
                "embedder_model_sha256": QRAG_EMBEDDER_MODEL_SHA256,
                "embedder_manifest_sha256": self.embedder_manifest_sha256,
                "compatibility_manifest_sha256": self.compatibility_manifest_sha256,
                "safe_weights_only_load": True,
                "strict_state_dict_load": True,
                "optimizer_state_loaded": False,
                "training_allowed": False,
                "paper_result_parity_claimed": False,
                **({"native_tensor_axis_adapter": QRAG_SINGLE_CANDIDATE_AXIS_COMPAT}
                   if self.single_candidate_axis_compat else {}),
            }
        )


@dataclass(frozen=True, slots=True)
class QRAGCommonSupportConfig:
    candidate_top_k: int = 6
    max_select_actions: int = 2
    allow_candidate_exhaustion: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_candidate_exhaustion, bool):
            raise ValidationError("Q-RAG allow_candidate_exhaustion must be boolean")
        minimum = 1 if self.allow_candidate_exhaustion else 2
        if (isinstance(self.candidate_top_k, bool)
                or not isinstance(self.candidate_top_k, int)
                or self.candidate_top_k < minimum):
            raise ValidationError(f"Q-RAG common support requires at least {minimum} candidates")
        if self.max_select_actions != 2:
            raise ValidationError("released Q-RAG policy requires two selection actions")


@dataclass(slots=True)
class QRAGCommonSupport:
    """Select real shared-corpus documents with the released Q-RAG PQN."""

    config: QRAGCommonSupportConfig = QRAGCommonSupportConfig()
    method_id: str = "qrag"
    release_id: str = "acquire.qrag.common_support"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("Q-RAG requires a non-empty query")
        if state.budget.max_retrieval_calls < 1:
            raise BudgetExceededError("Q-RAG has no retrieval budget")
        retriever = _retriever(context.retriever)
        runtime = _runtime(context.external_runtime)
        results = _validated_results(
            retriever.retrieve(state.query, top_k=self.config.candidate_top_k),
            top_k=self.config.candidate_top_k,
        )
        if not results:
            raise AlgorithmOutputError("Q-RAG requires at least one real retrieved candidate")
        if len(results) < 2 and not self.config.allow_candidate_exhaustion:
            raise AlgorithmOutputError("Q-RAG requires at least two retrieved candidates")
        candidate_ids = tuple(row.document.doc_id for row in results)
        response = runtime.select(
            QRAGPolicyRequest(
                question=state.query,
                candidates=results,
                seed=context.seed,
                max_steps=self.config.max_select_actions,
            )
        )
        trajectory, selected, usage, attestation = _validate_response(response, candidate_ids)
        blocks = tuple(
            _real_context(results[index], rank) for rank, index in enumerate(selected, 1)
        )
        runtime_identity = dict(runtime.fingerprint())
        events = [
            TraceEvent(
                TracePhase.PREPARE,
                "qrag_common_support_candidate_pool",
                {
                    "candidate_document_ids": candidate_ids,
                    "candidate_order_sha256": content_hash(candidate_ids),
                    "retriever": dict(retriever.fingerprint()),
                    "runtime": runtime_identity,
                    "selection_budget": 2,
                    **({"candidate_exhaustion_enabled": True,
                        "effective_selection_steps": min(2, len(results))}
                       if self.config.allow_candidate_exhaustion else {}),
                    "upstream_reader_used": False,
                    "paper_result_parity_claimed": False,
                },
            )
        ]
        events.extend(
            TraceEvent(TracePhase.OBSERVE, "qrag_value_policy_step", dict(step))
            for step in trajectory
        )
        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "qrag_selected_real_context",
                {
                    "selected_indices": selected,
                    "selected_document_ids": tuple(candidate_ids[index] for index in selected),
                    "trajectory_sha256": content_hash(trajectory),
                    "gold_answer_used": False,
                    "supporting_fact_labels_used": False,
                    "reward_used": False,
                },
            )
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    blocks,
                    target="qrag_value_selected_real_documents",
                    rationale="select shared-corpus evidence with the released frozen Q-RAG PQN",
                ),
            ),
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                events=tuple(events),
                counters={
                    "retrieval_calls": 1,
                    "embedding_calls": usage["embedding_calls"],
                    "policy_steps": len(trajectory),
                    "selected_documents": len(blocks),
                    "accelerator_seconds": usage["accelerator_seconds"],
                },
            ),
            implementation_identity=f"thin_external_wrapper:griver/Q-RAG@{QRAG_UPSTREAM_COMMIT}",
            profile_kind="controlled_adapter",
            metadata={
                "common_support_adaptation": "released_value_policy_over_shared_candidate_pool",
                "selected_indices": selected,
                "selected_document_ids": tuple(candidate_ids[index] for index in selected),
                "runtime_attestation": attestation,
                "upstream_reader_used": False,
                "real_documents_are_evidence": True,
                "gold_answer_used": False,
                "supporting_fact_labels_used": False,
                "reward_used": False,
                "training_called": False,
                "upstream_checkpoint_parity_claimed": False,
                "paper_result_parity_claimed": False,
            },
        )


def _policy_candidate_text(result: RetrievalResult) -> str:
    title = result.document.title.strip()
    text = result.document.text.strip()
    return f"Title: {title}\n{text}" if title else text


def _retriever(value: object) -> Retriever:
    if value is None or not hasattr(value, "retrieve") or not hasattr(value, "fingerprint"):
        raise ExternalDependencyError("Q-RAG common support requires a Retriever")
    return value  # type: ignore[return-value]


def _runtime(value: object) -> QRAGCommonSupportRuntime:
    if not isinstance(value, QRAGCommonSupportRuntime):
        raise ExternalDependencyError(
            "Q-RAG common support requires InterventionContext.external_runtime"
        )
    return value


def _validated_results(
    values: Sequence[RetrievalResult], *, top_k: int
) -> tuple[RetrievalResult, ...]:
    try:
        results = tuple(values)
    except TypeError as exc:
        raise AlgorithmOutputError("Q-RAG retriever returned a non-sequence") from exc
    if not all(isinstance(row, RetrievalResult) for row in results) or len(results) > top_k:
        raise AlgorithmOutputError("Q-RAG retriever returned invalid results")
    ids = tuple(row.document.doc_id for row in results)
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise AlgorithmOutputError("Q-RAG retrieval IDs must be non-empty and unique")
    if tuple(row.rank for row in results) != tuple(range(1, len(results) + 1)):
        raise AlgorithmOutputError("Q-RAG retrieval ranks must be contiguous and one-based")
    return results


def _validate_response(
    value: Mapping[str, Any],
    candidate_ids: tuple[str, ...],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[int, ...],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    if not isinstance(value, Mapping) or value.get("status") != "ok":
        raise AlgorithmOutputError("Q-RAG policy runtime returned a failed response")
    raw_steps = value.get("trajectory")
    raw_selected = value.get("selected_indices")
    usage = value.get("usage")
    attestation = value.get("attestation")
    if (
        isinstance(raw_steps, (str, bytes))
        or not isinstance(raw_steps, Sequence)
        or isinstance(raw_selected, (str, bytes))
        or not isinstance(raw_selected, Sequence)
        or not isinstance(usage, Mapping)
        or not isinstance(attestation, Mapping)
    ):
        raise AlgorithmOutputError("Q-RAG policy response schema is invalid")
    selected = tuple(raw_selected)
    if (
        len(selected) != min(2, len(candidate_ids))
        or any(isinstance(index, bool) or not isinstance(index, int) for index in selected)
        or len(selected) != len(set(selected))
        or any(index < 0 or index >= len(candidate_ids) for index in selected)
    ):
        raise AlgorithmOutputError("Q-RAG policy selected indices are invalid")
    steps: list[Mapping[str, Any]] = []
    history: tuple[int, ...] = ()
    for step_index, row in enumerate(raw_steps):
        if not isinstance(row, Mapping):
            raise AlgorithmOutputError("Q-RAG policy trajectory row is invalid")
        available = tuple(row.get("available_indices", ()))
        q_values = tuple(row.get("q_values", ()))
        action = row.get("action_index")
        current = tuple(row.get("selected_indices", ()))
        ids = tuple(row.get("selected_document_ids", ()))
        expected_available = tuple(
            index for index in range(len(candidate_ids)) if index not in history
        )
        if (
            row.get("index") != step_index
            or available != expected_available
            or len(q_values) != len(available)
            or not all(
                isinstance(score, (int, float)) and math.isfinite(score) for score in q_values
            )
            or action not in available
            or current != (*history, action)
            or ids != tuple(candidate_ids[index] for index in current)
            or row.get("selected_q_value") != q_values[available.index(action)]
        ):
            raise AlgorithmOutputError("Q-RAG policy trajectory is inconsistent")
        history = current
        steps.append(freeze_mapping(dict(row)))
    if tuple(history) != selected or tuple(raw_steps) == ():
        raise AlgorithmOutputError("Q-RAG policy trajectory disagrees with final selection")
    if value.get("selected_document_ids") != tuple(candidate_ids[index] for index in selected):
        raise AlgorithmOutputError("Q-RAG selected document IDs mismatch")
    if value.get("candidate_order_sha256") != content_hash(candidate_ids):
        raise AlgorithmOutputError("Q-RAG candidate order checksum mismatch")
    if value.get("trajectory_sha256") != content_hash(raw_steps):
        raise AlgorithmOutputError("Q-RAG trajectory checksum mismatch")
    if (
        value.get("policy_steps") != len(steps)
        or usage.get("embedding_calls") != 2 + 2 * len(steps)
        or any(
            usage.get(field) != 0
            for field in ("model_calls", "retrieval_calls", "input_tokens", "output_tokens")
        )
        or not isinstance(usage.get("accelerator_seconds"), (int, float))
        or usage["accelerator_seconds"] < 0
    ):
        raise AlgorithmOutputError("Q-RAG usage receipt is invalid")
    required_true = {
        "official_candidate_builder_called",
        "official_pqn_select_action_called",
        "official_qa_env_step_called",
        "safe_weights_only_load",
        "strict_state_dict_load",
        "checkpoint_eval_mode",
        "dummy_feedback_used",
        "seed_applied",
    }
    required_false = {
        "gold_answer_used",
        "supporting_fact_labels_used",
        "reward_used",
        "training_called",
        "stop_action_used",
    }
    if any(attestation.get(key) is not True for key in required_true) or any(
        attestation.get(key) is not False for key in required_false
    ):
        raise AlgorithmOutputError("Q-RAG runtime attestation failed")
    return tuple(steps), selected, freeze_mapping(dict(usage)), freeze_mapping(dict(attestation))


def _real_context(result: RetrievalResult, selected_rank: int) -> ContextBlock:
    document = result.document
    if not document.corpus_revision:
        raise AlgorithmOutputError("Q-RAG evidence requires a corpus revision")
    reserved = {
        "corpus_revision",
        "document_id",
        "evidence_kind",
        "qrag_candidate_rank",
        "qrag_selected_rank",
        "title",
    }
    if reserved & set(document.metadata):
        raise AlgorithmOutputError("Q-RAG document metadata uses reserved provenance keys")
    return ContextBlock(
        block_id=document.doc_id,
        text=document.text,
        source=document.corpus_revision,
        score=result.score,
        metadata={
            **dict(document.metadata),
            "document_id": document.doc_id,
            "title": document.title,
            "corpus_revision": document.corpus_revision,
            "qrag_candidate_rank": result.rank,
            "qrag_selected_rank": selected_rank,
            "evidence_kind": "real_corpus_document",
        },
    )


__all__ = [
    "LocalQRAGFrozenPolicyRuntime",
    "QRAGCommonSupport",
    "QRAGCommonSupportConfig",
    "QRAGCommonSupportRuntime",
    "QRAGPolicyRequest",
]
