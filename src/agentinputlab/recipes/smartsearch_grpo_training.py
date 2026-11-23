"""Public native SmartSearch GRPO execution, with explicit aggregate admission."""
from __future__ import annotations

import importlib
import json
import os
import secrets
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .qrag import _file, _sha
from .smartsearch_grpo_data import (
    SmartSearchGRPOControls,
    SmartSearchGRPODataArtifact,
    _native_grpo_config,
    verify_smartsearch_grpo_source,
)
from .smartsearch_grpo_meter import SmartSearchGRPOCostLedger, SmartSearchGRPOCostLimits
from .smartsearch_grpo_tools import (
    SmartSearchGRPORetrievalService,
    _new_output,
    _save,
    make_smartsearch_grpo_retrieval_server,
)
from .smartsearch_grpo_transport import (
    SmartSearchGRPOCoordinator,
    make_smartsearch_grpo_coordinator_server,
)
from .smartsearch_policy import MODEL_SCHEMA, _model

GRPO_TRAINING_PROTOCOL = "agentinputlab.smartsearch.native_grpo_fit.v1"
_RUNTIME_FILES = (
    "smartsearch_grpo_training.py", "smartsearch_grpo_worker.py", "smartsearch_grpo_transport.py",
    "smartsearch_grpo_meter.py", "smartsearch_grpo_prefix.py", "smartsearch_grpo_rollout.py",
    "smartsearch_grpo_tools.py", "smartsearch_grpo_data.py", "smartsearch_teacher.py",
    "smartsearch_sampling.py", "smartsearch_curriculum.py",
)


def _prepare_native_output_directories(root):
    """Create native checkpoint/rollout parents before any paid generation.

    The pinned reward manager opens its validation journal in append mode and
    does not create the journal's parent. The enclosing fit root is new and
    exclusively owned; existing native output (including symlinks) is refused.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValidationError("SmartSearch native output requires its own regular fit root")
    native = root / "native"
    native.mkdir()
    (native / "rollouts").mkdir()
    return native


@dataclass(frozen=True, slots=True)
class SmartSearchGRPOArtifact:
    root: Path
    artifact_id: str

    @property
    def checkpoint(self):
        return self.root / "exported-policy"

    @classmethod
    def load(cls, root):
        folder = Path(root)
        if folder.is_symlink() or not folder.is_dir():
            raise ValidationError("SmartSearch GRPO artifact must be a regular completed run directory")
        folder = folder.resolve()
        record = json.loads(_file(folder / "artifact.json", "native GRPO artifact").read_text())
        identity = record.pop("artifact_id", None)
        if record.get("protocol") != GRPO_TRAINING_PROTOCOL or identity != content_hash(record):
            raise ValidationError("SmartSearch native GRPO artifact identity differs")
        for name, key in (("fit-plan.json", "fit_plan_sha256"), ("fit-terminal.json", "fit_terminal_sha256"),
                          ("model-manifest.json", "model_manifest_sha256")):
            if _sha(_file(folder / name, "native GRPO closure")) != record.get(key):
                raise ValidationError("SmartSearch native GRPO artifact closure changed")
        _model(folder / "exported-policy", folder / "model-manifest.json", record["model_manifest_sha256"])
        terminal = json.loads((folder / "fit-terminal.json").read_text())
        if terminal.get("status") != "trained_native_grpo_candidate" or terminal.get("usage", {}).get("failed") is not False:
            raise ValidationError("SmartSearch native GRPO artifact is not a healthy completed fit")
        return cls(folder, identity)


def _references(data):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ExternalDependencyError("Install the benchmark parquet extra for native GRPO inputs") from exc
    result = {}
    manifest = json.loads((data.root / "manifest.json").read_text())
    for role, filename in (("train", "train.parquet"), ("validation", "validation.parquet")):
        if _sha(data.root / filename) != manifest["outputs"][filename]:
            raise ValidationError("SmartSearch GRPO reference parquet changed before native loading")
        rows = pq.read_table(data.root / filename).to_pylist()
        if _sha(data.root / filename) != manifest["outputs"][filename]:
            raise ValidationError("SmartSearch GRPO reference parquet changed during native loading")
        values = {}
        for row in rows:
            key = row["question"].strip()
            if key in values:
                raise ValidationError("SmartSearch native GRPO reference questions collide after native extraction")
            values[key] = row["reward_model"]["ground_truth"]
        result[role + "_references"] = values
    if set(result["train_references"]) & set(result["validation_references"]):
        raise ValidationError("SmartSearch GRPO training/development references overlap")
    return result


def _visible_actor_devices(controls):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",") if visible else []
    if len(devices) != controls.gpu_processes or any(not value.strip() or value != value.strip() for value in devices) or (
        len(set(devices)) != len(devices)
    ):
        raise ValidationError("Select exactly the admitted actor GPUs with CUDA_VISIBLE_DEVICES in a fresh process")
    return visible


class SmartSearchNativeGRPOTraining:
    """Run upstream native FSDP/vLLM/GRPO, not a replacement training loop.

    The operator supplies a frozen retriever and a separately metered raw-text
    helper provider. The helper model must run outside the selected actor GPU
    mesh. A fresh, dedicated native runtime is required; this owns a new local
    Ray cluster and never attaches to an existing user's cluster.
    """

    def __init__(self, *, controls: SmartSearchGRPOControls, cost_limits: SmartSearchGRPOCostLimits,
                 max_helper_rows=10000, max_retrieval_calls=10000, min_free_output_bytes=64 * 1024**3):
        if not isinstance(controls, SmartSearchGRPOControls) or not isinstance(cost_limits, SmartSearchGRPOCostLimits):
            raise ValidationError("SmartSearch native GRPO requires typed training and aggregate cost controls")
        if cost_limits.actor_ranks != controls.gpu_processes or any(type(v) is not int or v < 1
            for v in (max_helper_rows, max_retrieval_calls, min_free_output_bytes)):
            raise ValidationError("SmartSearch GRPO actor mesh and explicit runtime caps differ")
        self.controls, self.cost_limits = controls, cost_limits
        self.max_helper_rows, self.max_retrieval_calls = max_helper_rows, max_retrieval_calls
        self.min_free_output_bytes = min_free_output_bytes

    def fit(self, *, checkout, model_root, model_manifest, model_manifest_sha256, data_root,
            output_root, retriever, generate_helper, count_helper_input_tokens, helper_fingerprint_sha256,
            model_provenance, source_and_data_acknowledged=False):
        if source_and_data_acknowledged is not True or not isinstance(model_provenance, dict) or not model_provenance:
            raise ValidationError("SmartSearch native GRPO needs explicit model provenance and source/data acknowledgement")
        try:
            model_provenance = to_jsonable(model_provenance)
            json.dumps(model_provenance, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("SmartSearch native GRPO model provenance must be canonical JSON") from exc
        if any(name == "verl" or name.startswith("verl.") for name in sys.modules):
            raise ExternalDependencyError("Run native SmartSearch GRPO once in a fresh dedicated process")
        visible = _visible_actor_devices(self.controls)
        source, binding = verify_smartsearch_grpo_source(checkout)
        model, model_closure = _model(model_root, model_manifest, model_manifest_sha256)
        data = SmartSearchGRPODataArtifact.load(data_root)
        data_record = json.loads((data.root / "manifest.json").read_text())
        if data_record["source"] != binding:
            raise ValidationError("SmartSearch GRPO data was not prepared with the same native source binding")
        global_steps = self.controls.planned_steps(data.train_rows)
        # Native YAML is loaded below as well; this value is rechecked against
        # its actual ppo_epochs before any model initialization.
        planned_actor_steps = self.controls.planned_actor_optimizer_steps(data.train_rows, ppo_epochs=1)
        if planned_actor_steps > self.cost_limits.optimizer_steps_per_rank:
            raise ValidationError("SmartSearch GRPO aggregate optimizer admission is smaller than the native plan")
        references = _references(data)
        supplied = Path(output_root)
        output = supplied.resolve(strict=False)
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, model, data.root)):
            raise ValidationError("SmartSearch GRPO output overlaps a protected source/model/data input")
        parent = output.parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < self.min_free_output_bytes:
            raise ExternalDependencyError("SmartSearch native GRPO checkpoint capacity is insufficient")
        root = _new_output(supplied)
        plan = dict(protocol=GRPO_TRAINING_PROTOCOL, source=binding, controls=self.controls, cost_limits=self.cost_limits,
            model_manifest_sha256=model_manifest_sha256, model_revision=model_closure["revision"],
            model_provenance=model_provenance, data_artifact_id=data.artifact_id,
            planned_global_steps=global_steps, planned_actor_optimizer_steps=planned_actor_steps,
            actor_visible_devices=visible, helper_fingerprint_sha256=helper_fingerprint_sha256,
            max_helper_rows=self.max_helper_rows, max_retrieval_calls=self.max_retrieval_calls,
            no_automatic_neural_retry=True, native_loss_optimizer_rollout_reward_math_unchanged=True,
            physical_engine_initialization_and_checkpoint_recomputation_forwards_unmeasured=True,
            seed_scope="data/helper/PyTorch seed; native vLLM rank-derived seed retained",
            toolkit_modules={name: _sha(Path(__file__).with_name(name)) for name in _RUNTIME_FILES},
            toolkit_executor_sha256=_sha(Path(__file__)))
        _save(root, "fit-plan.json", plan)
        servers, threads, ray_module, ledger = [], [], None, None
        prior_path = list(sys.path)
        old_ensure = os.environ.get("ENSURE_CUDA_VISIBLE_DEVICES")
        try:
            native_output = _prepare_native_output_directories(root)
            ledger = SmartSearchGRPOCostLedger(output_root=root / "costs", limits=self.cost_limits)
            coordinator = SmartSearchGRPOCoordinator(ledger=ledger, generate=generate_helper,
                count_input_tokens=count_helper_input_tokens, model_fingerprint_sha256=helper_fingerprint_sha256)
            rpc_capability, tool_capability = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            def start_server(server):
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                try:
                    thread.start()
                except BaseException:
                    server.server_close()
                    raise
                servers.append(server)
                threads.append(thread)

            rpc = make_smartsearch_grpo_coordinator_server(coordinator, capability=rpc_capability)
            start_server(rpc)
            tools = SmartSearchGRPORetrievalService(retriever, output_root=root / "retrieval",
                                                   max_retrieval_calls=self.max_retrieval_calls)
            tool_server = make_smartsearch_grpo_retrieval_server(tools, capability=tool_capability)
            start_server(tool_server)
            descriptor = dict(checkout=str(source), output_root=str(root), seed=self.controls.seed,
                coordinator_endpoint=f"http://127.0.0.1:{rpc.server_port}/{rpc_capability}/rpc",
                helper_fingerprint_sha256=helper_fingerprint_sha256, max_helper_rows=self.max_helper_rows,
                cost_limits=to_jsonable(self.cost_limits), toolkit_modules=plan["toolkit_modules"], **references)
            config = _native_grpo_config(source, model=model, data=data, output_root=native_output,
                controls=self.controls,
                sandbox_url=f"http://127.0.0.1:{tool_server.server_port}/{tool_capability}",
                reward_module=source / "src/verl/utils/reward_score/re_call.py")
            actual_steps = self.controls.planned_actor_optimizer_steps(data.train_rows,
                ppo_epochs=config["actor_rollout_ref"]["actor"]["ppo_epochs"])
            if actual_steps != planned_actor_steps or config["actor_rollout_ref"]["actor"]["strategy"] != "fsdp":
                raise ExternalDependencyError("SmartSearch native actor plan/strategy differs from admitted FSDP execution")
            _save(root, "private-runtime.json", dict(descriptor=descriptor, native_config=config,
                contains_run_local_capabilities_not_for_publication=True))
            try:
                import ray
                import torch
                from omegaconf import OmegaConf
            except ImportError as exc:
                raise ExternalDependencyError("Install the pinned native SmartSearch Ray/vLLM training runtime") from exc
            if ray.is_initialized():
                raise ExternalDependencyError("SmartSearch GRPO cannot reuse an existing Ray cluster")
            if not torch.cuda.is_available() or torch.cuda.device_count() != self.controls.gpu_processes:
                raise ExternalDependencyError("SmartSearch native GRPO requires exactly its selected CUDA actor mesh")
            sys.path.insert(0, str(source / "src"))
            native_driver = importlib.import_module("verl.trainer.main_ppo")
            from .smartsearch_grpo_worker import native_grpo_task_runner
            task_type = native_grpo_task_runner(native_driver, descriptor)
            package_root = Path(__file__).resolve().parents[2]
            temporary = tempfile.mkdtemp(prefix="ail-smartsearch-ray-")
            _save(root, "ray-runtime.json", dict(temp_root=temporary, own_local_cluster=True, actor_gpu_count=self.controls.gpu_processes))
            os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = visible
            ray_module = ray
            ray.init(address="local", num_gpus=self.controls.gpu_processes, num_cpus=max(8, self.controls.gpu_processes * 4),
                include_dashboard=False, _temp_dir=temporary,
                runtime_env={"env_vars": dict(PYTHONPATH=str(source / "src") + os.pathsep + str(package_root),
                    PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false",
                    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ENSURE_CUDA_VISIBLE_DEVICES=visible,
                    NCCL_DEBUG="WARN", VLLM_LOGGING_LEVEL="WARN")})
            actor_type = ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0)(task_type)
            runner = actor_type.remote()
            result = ray.get(runner.run.remote(OmegaConf.create(config)))
            if result["global_steps"] != global_steps:
                raise ExternalDependencyError("SmartSearch native GRPO did not finish its admitted global-step plan")
            usage = coordinator.assert_healthy(expected_optimizer_steps=actual_steps,
                                                required_roles=("actor_rollout", "ref"))
            retrieval_usage = tools.assert_healthy()
            export = json.loads(_file(root / "safe-export.json", "trained actor export").read_text())
            if export.get("same_live_trained_actor") is not True or export.get("new_optimizer_steps") != 0:
                raise ExternalDependencyError("SmartSearch GRPO checkpoint is not its actual trained actor")
            manifest = dict(schema=MODEL_SCHEMA, revision=content_hash(dict(plan=plan, export=export)),
                files=export["files"], native_grpo=True, trained_actor_not_original_input=True)
            _save(root, "model-manifest.json", manifest)
            _model(root / "exported-policy", root / "model-manifest.json", _sha(root / "model-manifest.json"))
            # Confirm the mounted input snapshot was not used as an output path.
            _model(model, model_manifest, model_manifest_sha256)
            _save(root, "fit-terminal.json", dict(status="trained_native_grpo_candidate", usage=usage,
                retrieval_usage=retrieval_usage, native_driver_result=result, safe_export=export,
                full_sft_dpo_grpo_recipe_certification=False, heldout_acceptance_completed=False))
            artifact = dict(protocol=GRPO_TRAINING_PROTOCOL, fit_plan_sha256=_sha(root / "fit-plan.json"),
                fit_terminal_sha256=_sha(root / "fit-terminal.json"),
                model_manifest_sha256=_sha(root / "model-manifest.json"),
                source_commit=binding["source_commit"], data_artifact_id=data.artifact_id)
            _save(root, "artifact.json", {**artifact, "artifact_id": content_hash(artifact)})
            return SmartSearchGRPOArtifact.load(root)
        except BaseException as exc:
            if ledger is not None:
                ledger.fail("native-driver", error_type=type(exc).__name__)
            _save(root, "failed.json", dict(error_type=type(exc).__name__, error=str(exc),
                usage=ledger.snapshot() if ledger is not None else None,
                no_automatic_retry=True, accept_checkpoint=False))
            raise
        finally:
            if ray_module is not None:
                ray_module.shutdown()  # This newly owned cluster only.
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=10)
            sys.path[:] = prior_path
            if old_ensure is None:
                os.environ.pop("ENSURE_CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = old_ensure
