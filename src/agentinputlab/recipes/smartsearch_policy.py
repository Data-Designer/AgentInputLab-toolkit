"""Native full-SFT / LoRA-DPO policy stages for SmartSearch's curriculum.

The pinned LLaMA-Factory owns data collation, losses, optimizer, scheduler and
checkpoint export. This module binds data/source/models, mounts fresh paths,
observes native steps and retains failure costs. GRPO/full-pipeline completion
is deliberately not claimed by these policy-stage APIs.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .llmlingua2_training import _closed_files
from .qrag import _file, _sha
from .recomp_extractive import _LOCK, _scope
from .smartsearch_curriculum import _question_hash, verify_smartsearch_preparation_source
from .smartsearch_native_data import verify_smartsearch_native_data_source

POLICY_PROTOCOL = "agentinputlab.smartsearch.native_policy_stage.v1"
MODEL_SCHEMA = "agentinputlab.smartsearch.policy_snapshot.v1"
_CONFIGS = {
    "sft": ("LLaMA-Factory/examples/train_full/qwen_full_sft.yaml",
            "bc92534ece9249ebb54e02bfd6a7c22dd9424fa77a908a25e39936a68a91d471"),
    "dpo": ("LLaMA-Factory/examples/train_lora/qwen_lora_dpo.yaml",
            "1176f154ed14b5f9c0b9fec08424717689e92ca7ec2f47eb4712b0be0cc69394"),
    "merge": ("LLaMA-Factory/examples/merge_lora/qwen_lora_dpo.yaml",
              "c2176e4152139518bf75f371659b1e6970805307ff106e78dce7f93c39ea4cea"),
}
_ZERO3 = "LLaMA-Factory/examples/deepspeed/ds_z3_config.json"
_ZERO3_SHA = "6e2753c870561658babe45922114f14ad899d5e4cfb6deb38fe47935ee60dc6f"
_TRAIN_SOURCE = {
    "LLaMA-Factory/src/llamafactory/train/tuner.py": "2e7b0bbd3db765ab140be8539f36c4c72e9bf53572ff43e719aed944fcde113a",
    "LLaMA-Factory/src/llamafactory/train/sft/workflow.py": "eadb7829102c234d00464ca7895ea9bf33993d9503d7b713770012be4ef44735",
    "LLaMA-Factory/src/llamafactory/train/sft/trainer.py": "66e2c3dbbe0acda54090edb6e29a9e3fbc0bd7ed7aaf808bacac61d624e3172b",
    "LLaMA-Factory/src/llamafactory/train/dpo/workflow.py": "12e03b82dc61ac8c0149be9f2f6aef1919e4519d794e27ccf909970a740fc678",
    "LLaMA-Factory/src/llamafactory/train/dpo/trainer.py": "f1af018d741a5f3a4cde2d654ce027f6bb4396bf88f1328d273ab669b71ac1d0",
}


@dataclass(frozen=True, slots=True)
class SmartSearchPolicyFitControls:
    seed: int = 17
    max_optimizer_steps: int = 10000
    max_data_bytes: int = 16 * 1024 * 1024
    min_free_output_bytes: int = 32 * 1024**3
    gpu_processes: int = 1

    def __post_init__(self):
        for name in ("seed", "max_optimizer_steps", "max_data_bytes", "min_free_output_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValidationError("SmartSearch policy stage controls require positive integer caps")
        if type(self.gpu_processes) is not int or not 1 <= self.gpu_processes <= 4:
            raise ValidationError("SmartSearch policy fitting supports one to four operator-selected GPUs")


def _save(root, name, value):
    with (root / name).open("x", encoding="utf-8") as stream:
        json.dump(to_jsonable(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _source(checkout):
    binding = dict(verify_smartsearch_preparation_source(checkout))
    root = Path(binding["root"])
    # These exact configs are part of the pinned upstream Git checkout. Never
    # load caller-authored YAML with a Python object constructor.
    checks = dict(_CONFIGS.values()) | {_ZERO3: _ZERO3_SHA} | _TRAIN_SOURCE
    for relative, expected in checks.items():
        path = _file(root / relative, "SmartSearch native training config")
        if not path.is_relative_to(root) or _sha(path) != expected:
            raise ExternalDependencyError("SmartSearch native training configuration changed")
    binding["policy_configs"] = checks
    return root, binding


def _model(root, manifest, manifest_sha256):
    supplied = Path(root)
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValidationError("SmartSearch policy requires a mounted regular model snapshot")
    folder = supplied.resolve()
    config = json.loads(_file(folder / "config.json", "policy model config").read_text())
    tokenizer = json.loads(_file(folder / "tokenizer_config.json", "policy tokenizer config").read_text())
    if config.get("quantization_config"):
        raise ExternalDependencyError(
            "SmartSearch native full-SFT/LoRA-merge pipeline does not support a quantized base; "
            "no silent dequantization or replacement precision is applied"
        )
    if config.get("auto_map") or tokenizer.get("auto_map"):
        raise ExternalDependencyError("SmartSearch policy fitting does not execute model remote code")
    if not str(config.get("model_type", "")).startswith("qwen"):
        raise ExternalDependencyError(
            "SmartSearch's pinned policy configs use the Qwen template; another model family "
            "requires an explicitly defined training variant, not an implicit prompt-template change"
        )
    folder, closure = _closed_files(folder, manifest, manifest_sha256, MODEL_SCHEMA)
    names = closure["files"]
    if not any(name.endswith(".safetensors") for name in names) or any(
        name.endswith((".py", ".bin", ".pt", ".pkl", ".pickle")) for name in names
    ):
        raise ValidationError("SmartSearch policy snapshot must use safe, code-free safetensors")
    if "adapter_config.json" in names:
        raise ValidationError("SmartSearch full policy snapshot cannot be an implicit LoRA adapter")
    if not isinstance(closure.get("revision"), str) or closure["revision"].lower() in {"", "main", "latest", "head"}:
        raise ValidationError("SmartSearch policy snapshot requires an immutable revision")
    return folder, closure


def _curriculum(root, stage, byte_cap):
    folder = Path(root)
    if folder.is_symlink() or not folder.is_dir():
        raise ValidationError("SmartSearch policy curriculum root must be a regular directory")
    folder = folder.resolve()
    manifest = json.loads(_file(folder / "manifest.json", "curriculum manifest").read_text())
    artifact_id = manifest.pop("artifact_id", None)
    if artifact_id != content_hash(manifest) or manifest.get("schema") != "agentinputlab.smartsearch.native-curriculum-data.v1":
        raise ValidationError("SmartSearch curriculum artifact identity differs")
    if manifest.get("rollout_provenance", {}).get("fixture_only"):
        raise ValidationError("SmartSearch neural fitting cannot consume fixture-only rollouts")
    name = f"{stage}.json"
    path = _file(folder / name, "curriculum rows")
    if path.stat().st_size > byte_cap or _sha(path) != manifest.get("outputs", {}).get(name):
        raise ValidationError("SmartSearch native curriculum rows changed or exceed byte cap")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValidationError("SmartSearch native selection is empty; no replacement examples are manufactured")
    audit = manifest.get("split_audit", {})
    training_hashes = set(audit.get("train", {}).get("question_hashes", []))
    heldout_hashes = set(audit.get("validation", {}).get("question_hashes", [])) | set(
        audit.get("test", {}).get("question_hashes", []))
    if not training_hashes or not audit.get("validation", {}).get("case_ids") or training_hashes & heldout_hashes:
        raise ValidationError("SmartSearch curriculum lacks disjoint native train/validation/test audit")
    keys = ("instruction", "input", "output") if stage == "sft" else ("instruction", "input", "chosen", "rejected")
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in keys):
            raise ValidationError("SmartSearch native policy row is incomplete")
        if _question_hash(row["input"]) not in training_hashes:
            raise ValidationError("SmartSearch policy row is not a declared training question")
    return rows, manifest, artifact_id


def _native_config(source, stage, model, root, seed):
    try:
        import yaml
    except ImportError as exc:
        raise ExternalDependencyError("Install agentinputlab[yaml] to build native SmartSearch policy configs") from exc
    config = yaml.safe_load((source / _CONFIGS[stage][0]).read_text())
    if not isinstance(config, dict):
        raise ExternalDependencyError("SmartSearch native policy configuration must be a mapping")
    config.update(model_name_or_path=str(model), trust_remote_code=False)
    if stage == "merge":
        config.update(export_dir=str(root / "checkpoint"))
        return config
    # Only paths, explicit seed, cache/telemetry and create-only execution
    # controls differ. Native loss/rank/LR/batches/epochs/cutoff/ZeRO stay intact.
    config.update(
        dataset=f"mine_{stage}", dataset_dir=str(root / "data"),
        output_dir=str(root / "checkpoint"), deepspeed=str(source / _ZERO3),
        seed=seed, data_seed=seed, overwrite_cache=False, overwrite_output_dir=False,
        push_to_hub=False, report_to="none", resume_from_checkpoint=None,
    )
    return config


def _source_import(source, native_data_source=None):
    name = "llamafactory"
    prefix = source / "LLaMA-Factory/src"
    previous = list(sys.path)
    if any(n == name or n.startswith(name + ".") for n in sys.modules):
        raise ExternalDependencyError("Run each native SmartSearch policy stage in a fresh dedicated process")
    sys.path.insert(0, str(prefix))
    try:
        if native_data_source is not None:
            bound = verify_smartsearch_native_data_source(source, native_data_source["root"])
            if bound != native_data_source:
                raise ExternalDependencyError("SmartSearch native data supplement launch binding changed")
            package = importlib.import_module(name)
            if not Path(package.__file__).resolve().is_relative_to(prefix):
                raise ExternalDependencyError("SmartSearch native package was shadowed")
            # Mount this package only; never add an alternative framework root
            # to sys.path or the original package's general search path.
            data = Path(bound["root"]) / "data"
            spec = importlib.util.spec_from_file_location(
                "llamafactory.data", data / "__init__.py", submodule_search_locations=[str(data)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            package.data = module
            spec.loader.exec_module(module)
        module = importlib.import_module("llamafactory.train.tuner")
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise ExternalDependencyError("SmartSearch native trainer was shadowed by another installation")
        return module
    finally:
        sys.path[:] = previous


def _observer(counters, max_steps):
    from transformers import TrainerCallback

    class Observer(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.max_steps < 1 or state.max_steps > max_steps:
                raise ValidationError("SmartSearch native optimizer plan exceeds the caller's step cap")
            counters["planned_optimizer_steps"] = state.max_steps
            model = kwargs.get("model")
            if model is None:
                raise ExternalDependencyError("SmartSearch native trainer did not expose its policy")

            def before(module, args, kwargs):
                counters["forward_attempts"] += 1
                inputs = kwargs.get("input_ids")
                if inputs is None and args:
                    inputs = args[0]
                counters["input_token_positions"] += int(inputs.numel()) if hasattr(inputs, "numel") else 0

            def after(module, args, output):
                counters["forward_responses"] += 1

            self.before = model.register_forward_pre_hook(before, with_kwargs=True)
            self.after = model.register_forward_hook(after)

        def on_step_end(self, args, state, control, **kwargs):
            counters["optimizer_steps"] = state.global_step
            if state.global_step > max_steps:
                raise ValidationError("SmartSearch native optimizer exceeded its declared step cap")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                record = {}
                for key, value in logs.items():
                    if isinstance(value, float) and not math.isfinite(value):
                        counters["nonfinite_native_metrics"] += 1
                        record[key] = repr(value)
                    elif isinstance(value, (float, int, str, bool)):
                        record[key] = value
                counters["native_logs"].append(record)

        def on_train_end(self, args, state, control, **kwargs):
            self.before.remove()
            self.after.remove()

    return Observer()


@contextmanager
def _environment(root, seed):
    overrides = {"HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled"}
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        with _scope(root, seed):
            os.environ.update(overrides)
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True, slots=True)
class SmartSearchPolicyStageArtifact:
    root: Path
    artifact_id: str
    stage: str

    @property
    def checkpoint(self):
        return self.root / "checkpoint"

    @classmethod
    def load(cls, root):
        supplied = Path(root)
        if supplied.is_symlink() or not supplied.is_dir():
            raise ValidationError("SmartSearch policy artifact root is invalid")
        folder = supplied.resolve()
        record = json.loads(_file(folder / "artifact.json", "policy artifact").read_text())
        identity = record.pop("artifact_id", None)
        if identity != content_hash(record) or record.get("protocol") != POLICY_PROTOCOL or record.get("stage") not in {"sft", "dpo", "merged_dpo"}:
            raise ValidationError("SmartSearch policy artifact identity differs")
        for key, name in (("fit_plan_sha256", "fit-plan.json"), ("fit_terminal_sha256", "fit-terminal.json")):
            if _sha(_file(folder / name, "policy receipt")) != record.get(key):
                raise ValidationError("SmartSearch policy artifact plan/terminal changed")
        terminal = json.loads((folder / "fit-terminal.json").read_text())
        if record["stage"] == "merged_dpo":
            plan = json.loads((folder / "fit-plan.json").read_text())
            if terminal.get("status") != "merged_candidate" or not plan.get("dpo_artifact_id"):
                raise ValidationError("SmartSearch merged policy has no fitted adapter binding")
        elif terminal.get("status") != "trained_candidate" or terminal.get("usage", {}).get("optimizer_steps", 0) < 1:
            raise ValidationError("SmartSearch policy artifact has no completed optimizer step")
        files = record.get("checkpoint_files")
        if not isinstance(files, dict) or not files or not any(n.endswith(".safetensors") for n in files):
            raise ValidationError("SmartSearch policy artifact has no safe checkpoint closure")
        for name, digest in files.items():
            if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                raise ValidationError("SmartSearch policy checkpoint file path is unsafe")
            if _sha(_file(folder / "checkpoint" / name, "policy checkpoint")) != digest:
                raise ValidationError("SmartSearch policy checkpoint bytes changed")
        return cls(folder, identity, record["stage"])

    def materialize_snapshot(self, output_root):
        """Copy final SFT weights/tokenizer only; never pickle/optimizer state.

        DPO is still an adapter and needs native merge before becoming a policy.
        This explicit copy has a storage cost; the original training log and
        checkpoints remain untouched.
        """
        import shutil

        current = self.load(self.root)
        if current.stage == "dpo":
            raise ValidationError("SmartSearch LoRA-DPO adapter must be natively merged before policy export")
        record = json.loads((current.root / "artifact.json").read_text())
        root = Path(output_root)
        if root.exists() or root.is_symlink():
            raise ValidationError("SmartSearch policy snapshot must be create-only")
        root = root.resolve(strict=False)
        if root.is_relative_to(current.root) or current.root.is_relative_to(root):
            raise ValidationError("SmartSearch policy snapshot must not overlap the training artifact")
        files = record["checkpoint_files"]
        if not {"config.json", "tokenizer_config.json"} <= files.keys():
            raise ValidationError("SmartSearch policy checkpoint lacks model/tokenizer configuration")
        for name in files:
            if Path(name).suffix not in {".safetensors", ".json", ".model", ".txt", ".jinja"}:
                raise ValidationError("SmartSearch policy export refuses executable/binary payloads")
        # Keep native training reports in the training artifact, not in the
        # inference snapshot. The latter closes only model/tokenizer payloads.
        hf_metadata = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt"}
        files = {name: digest for name, digest in files.items() if name in hf_metadata or
                 name.endswith((".safetensors", ".safetensors.index.json", ".model", ".jinja"))}
        parent = root.parent
        while not parent.exists():
            parent = parent.parent
        required = sum((current.checkpoint / name).stat().st_size for name in files)
        if shutil.disk_usage(parent).free < required:
            raise ExternalDependencyError("SmartSearch policy snapshot copy lacks output capacity")
        root.mkdir(parents=True, exist_ok=False)
        (root / "model").mkdir()
        for name, digest in files.items():
            shutil.copyfile(current.checkpoint / name, root / "model" / name)
            if _sha(root / "model" / name) != digest:
                raise ValidationError("SmartSearch policy changed while copying its safe closure")
        manifest = {"schema": MODEL_SCHEMA, "revision": "artifact:" + current.artifact_id,
                    "files": files, "training_artifact_id": current.artifact_id,
                    "toolkit_exporter_sha256": _sha(Path(__file__))}
        _save(root, "model-manifest.json", manifest)
        return {"model_root": str(root / "model"), "model_manifest": str(root / "model-manifest.json"),
                "model_manifest_sha256": _sha(root / "model-manifest.json")}


class SmartSearchNativePolicyTraining:
    """Execute original LLaMA-Factory SFT/DPO; not the complete GRPO recipe."""

    def __init__(self, controls=None):
        self.controls = SmartSearchPolicyFitControls(**controls) if isinstance(controls, Mapping) else controls or SmartSearchPolicyFitControls()
        if not isinstance(self.controls, SmartSearchPolicyFitControls):
            raise ValidationError("SmartSearch policy fitting requires typed controls")

    def fit(
        self, *, stage, checkout, model_root, model_manifest, model_manifest_sha256,
        curriculum_root, output_root, source_and_data_acknowledged=False, native_data_root=None,
    ):
        if source_and_data_acknowledged is not True or stage not in {"sft", "dpo"}:
            raise ValidationError("SmartSearch policy stage needs explicit source/data acknowledgement and sft/dpo")
        source, source_binding = _source(checkout)
        if native_data_root is None:
            raise ExternalDependencyError("Prepare SmartSearch's pinned native data source and pass native_data_root")
        native_data_source = verify_smartsearch_native_data_source(source, native_data_root)
        model, closure = _model(model_root, model_manifest, model_manifest_sha256)
        rows, curriculum, curriculum_id = _curriculum(curriculum_root, stage, self.controls.max_data_bytes)
        root = Path(output_root)
        if root.exists() or root.is_symlink():
            raise ValidationError("SmartSearch policy output must be create-only; no automatic retry")
        root = root.resolve(strict=False)
        protected = (source, model, Path(curriculum_root).resolve())
        if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
            raise ValidationError("SmartSearch policy output must not overlap mounted assets")
        import shutil

        parent = root.parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < self.controls.min_free_output_bytes:
            raise ExternalDependencyError("SmartSearch native policy output capacity is insufficient")
        config = _native_config(source, stage, model, root, self.controls.seed)
        root.mkdir(parents=True, exist_ok=False)
        (root / "data").mkdir()
        _save(root / "data", f"{stage}.json", rows)
        dataset = {"file_name": f"{stage}.json", "columns": {"prompt": "instruction", "query": "input"}}
        if stage == "sft":
            dataset["columns"]["response"] = "output"
        else:
            dataset["ranking"] = True
            dataset["columns"].update(chosen="chosen", rejected="rejected")
        _save(root / "data", "dataset_info.json", {f"mine_{stage}": dataset})
        versions = {}
        for name in ("torch", "transformers", "trl", "datasets", "peft", "accelerate", "deepspeed"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        plan = {"protocol": POLICY_PROTOCOL, "stage": stage, "source": source_binding,
                "native_data_source": native_data_source,
                "controls": self.controls, "native_config": config, "model_manifest_sha256": model_manifest_sha256,
                "model_revision": closure["revision"], "curriculum_artifact_id": curriculum_id,
                "split_audit": curriculum["split_audit"], "training_only": True,
                "native_loss_optimizer_scheduler_unchanged": True, "dependency_versions": versions,
                "toolkit_policy_sha256": _sha(Path(__file__)),
                "toolkit_worker_sha256": _sha(Path(__file__).with_name("smartsearch_policy_worker.py"))}
        _save(root, "fit-plan.json", plan)
        counters = {"rank_receipts": {}, "rank_usage_complete": False,
                    "forward_attempts": None, "forward_responses": None,
                    "optimizer_steps": None, "nonfinite_native_metrics": None}
        started = time.monotonic()
        try:
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       f"--nproc_per_node={self.controls.gpu_processes}", "--max_restarts=0", "--module",
                       "agentinputlab.recipes.smartsearch_policy_worker", str(root),
                       _sha(root / "fit-plan.json")]
            _save(root, "launch.json", {"argv": command, "no_automatic_retry": True,
                                        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
            with (root / "native.log").open("xb") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            _collect_rank_usage(root, self.controls.gpu_processes, counters)
            if process.returncode:
                raise ExternalDependencyError(f"SmartSearch native distributed trainer exited {process.returncode}; see native.log")
            if not counters["rank_usage_complete"] or any(
                value.get("status") != "completed" for value in counters["rank_receipts"].values()
            ):
                raise ExternalDependencyError("SmartSearch native rank receipts are incomplete")
            if not counters["distributed_optimizer_step_agreement"]:
                raise ExternalDependencyError("SmartSearch distributed ranks disagree on native optimizer steps")
            checkpoint = root / "checkpoint"
            tensors = [p for p in checkpoint.iterdir() if p.name.endswith(".safetensors")]
            if counters["optimizer_steps"] < 1 or not tensors or any(p.is_symlink() for p in tensors):
                raise ExternalDependencyError("SmartSearch native policy did not produce a trained safe checkpoint")
            if counters["nonfinite_native_metrics"]:
                raise ExternalDependencyError("SmartSearch native training recorded a non-finite loss")
            files = {p.name: _sha(p) for p in checkpoint.iterdir()
                     if p.is_file() and p.suffix in {".safetensors", ".json", ".model", ".txt", ".jinja"}}
            terminal = {"status": "trained_candidate", "stage": stage, "usage": counters,
                        "elapsed_seconds": time.monotonic() - started, "checkpoint_files": files,
                        "heldout_evaluation_completed": False, "full_sft_dpo_grpo_completed": False}
            _save(root, "fit-terminal.json", terminal)
            artifact = {"protocol": POLICY_PROTOCOL, "stage": stage,
                        "fit_plan_sha256": _sha(root / "fit-plan.json"),
                        "fit_terminal_sha256": _sha(root / "fit-terminal.json"),
                        "checkpoint_files": files, "base_model_manifest_sha256": model_manifest_sha256,
                        "source_commit": source_binding["source_commit"], "curriculum_artifact_id": curriculum_id}
            artifact["artifact_id"] = content_hash(artifact)
            _save(root, "artifact.json", artifact)
            return SmartSearchPolicyStageArtifact.load(root)
        except Exception as exc:
            _collect_rank_usage(root, self.controls.gpu_processes, counters)
            _save(root, "failed.json", {"error_type": type(exc).__name__, "error": str(exc),
                                        "usage": counters, "elapsed_seconds": time.monotonic() - started,
                                        "no_automatic_retry": True})
            raise

    def merge(
        self, *, checkout, model_root, model_manifest, model_manifest_sha256,
        dpo_artifact_root, output_root, source_and_data_acknowledged=False, native_data_root=None,
    ):
        """Native CPU LoRA merge with the exact SFT base; no new training."""
        if source_and_data_acknowledged is not True:
            raise ValidationError("SmartSearch native merge requires explicit source/data acknowledgement")
        source, source_binding = _source(checkout)
        if native_data_root is None:
            raise ExternalDependencyError("Prepare SmartSearch's pinned native data source and pass native_data_root")
        native_data_source = verify_smartsearch_native_data_source(source, native_data_root)
        model, closure = _model(model_root, model_manifest, model_manifest_sha256)
        adapter = SmartSearchPolicyStageArtifact.load(dpo_artifact_root)
        record = json.loads((adapter.root / "artifact.json").read_text())
        if adapter.stage != "dpo" or record["base_model_manifest_sha256"] != model_manifest_sha256:
            raise ValidationError("SmartSearch DPO adapter and native merge base do not match")
        root = Path(output_root)
        if root.exists() or root.is_symlink():
            raise ValidationError("SmartSearch native merge must use a new output directory")
        root = root.resolve(strict=False)
        if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (source, model, adapter.root)):
            raise ValidationError("SmartSearch native merge output overlaps a protected input")
        config = _native_config(source, "merge", model, root, self.controls.seed)
        config["adapter_name_or_path"] = str(adapter.checkpoint)
        root.mkdir(parents=True, exist_ok=False)
        controls = to_jsonable(self.controls)
        controls["gpu_processes"] = 1  # original merge export_device=cpu
        plan = {"protocol": POLICY_PROTOCOL, "operation": "merge", "source": source_binding,
                "native_data_source": native_data_source,
                "controls": controls, "native_config": config, "model_revision": closure["revision"],
                "model_manifest_sha256": model_manifest_sha256, "dpo_artifact_id": adapter.artifact_id,
                "toolkit_policy_sha256": _sha(Path(__file__)),
                "toolkit_worker_sha256": _sha(Path(__file__).with_name("smartsearch_policy_worker.py"))}
        _save(root, "fit-plan.json", plan)
        usage = {"rank_receipts": {}, "rank_usage_complete": False, "forward_attempts": None,
                 "forward_responses": None, "optimizer_steps": None}
        started = time.monotonic()
        try:
            command = [sys.executable, "-m", "agentinputlab.recipes.smartsearch_policy_worker",
                       str(root), _sha(root / "fit-plan.json")]
            _save(root, "launch.json", {"argv": command, "no_automatic_retry": True, "export_device": "cpu"})
            with (root / "native.log").open("xb") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            _collect_rank_usage(root, 1, usage)
            if process.returncode or not usage["rank_usage_complete"] or usage["rank_receipts"]["0"]["status"] != "completed":
                raise ExternalDependencyError("SmartSearch native LoRA merge did not complete; see retained native logs")
            checkpoint = root / "checkpoint"
            files = {p.name: _sha(p) for p in checkpoint.iterdir()
                     if p.is_file() and not p.is_symlink() and p.suffix in {".safetensors", ".json", ".model", ".txt", ".jinja"}}
            if not any(name.endswith(".safetensors") for name in files) or "config.json" not in files:
                raise ExternalDependencyError("SmartSearch native merge did not export safe policy weights")
            _save(root, "fit-terminal.json", {"status": "merged_candidate", "usage": usage,
                                              "elapsed_seconds": time.monotonic() - started,
                                              "full_sft_dpo_grpo_completed": False})
            artifact = {"protocol": POLICY_PROTOCOL, "stage": "merged_dpo",
                        "fit_plan_sha256": _sha(root / "fit-plan.json"),
                        "fit_terminal_sha256": _sha(root / "fit-terminal.json"),
                        "checkpoint_files": files, "base_model_manifest_sha256": model_manifest_sha256,
                        "source_commit": source_binding["source_commit"], "dpo_artifact_id": adapter.artifact_id}
            artifact["artifact_id"] = content_hash(artifact)
            _save(root, "artifact.json", artifact)
            return SmartSearchPolicyStageArtifact.load(root)
        except Exception as exc:
            _collect_rank_usage(root, 1, usage)
            _save(root, "failed.json", {"error_type": type(exc).__name__, "error": str(exc), "usage": usage,
                                        "elapsed_seconds": time.monotonic() - started, "no_automatic_retry": True})
            raise


def _collect_rank_usage(root, world_size, counters):
    records = {}
    errors = []
    expected_plan = _sha(root / "fit-plan.json")
    for rank in range(world_size):
        path = root / f"rank-{rank}-terminal.json"
        if path.is_file() and not path.is_symlink():
            try:
                row = json.loads(path.read_text())
                if row.get("rank") != rank or row.get("plan_sha256") != expected_plan or row.get("status") not in {"completed", "failed"}:
                    raise ValueError("rank/plan/status binding differs")
                usage = row["usage"]
                for name in ("forward_attempts", "forward_responses", "input_token_positions", "nonfinite_native_metrics", "optimizer_steps"):
                    if type(usage.get(name)) is not int or usage[name] < 0:
                        raise ValueError("rank usage is not a measured nonnegative integer")
                records[str(rank)] = row
            except (KeyError, TypeError, ValueError) as exc:
                errors.append({"rank": rank, "error": str(exc)})
    counters["rank_receipts"] = records
    counters["rank_parse_errors"] = errors
    counters["rank_usage_complete"] = len(records) == world_size
    if counters["rank_usage_complete"]:
        usages = [row["usage"] for row in records.values()]
        for key in ("forward_attempts", "forward_responses", "input_token_positions", "nonfinite_native_metrics"):
            counters[key] = sum(row[key] for row in usages)
        steps = {row["optimizer_steps"] for row in usages}
        counters["optimizer_steps"] = min(steps)
        counters["distributed_optimizer_step_agreement"] = len(steps) == 1


def run_native_policy_rank(output_root, plan_sha256):
    """Distributed worker body; invoked once per rank by native torchrun."""
    root = Path(output_root).resolve()
    path = _file(root / "fit-plan.json", "policy launch plan")
    if _sha(path) != plan_sha256:
        raise ValidationError("SmartSearch distributed policy plan identity differs")
    plan = json.loads(path.read_text())
    if _sha(Path(__file__)) != plan.get("toolkit_policy_sha256") or _sha(
        Path(__file__).with_name("smartsearch_policy_worker.py")
    ) != plan.get("toolkit_worker_sha256"):
        raise ValidationError("SmartSearch child toolkit bytes differ from the launcher")
    controls = SmartSearchPolicyFitControls(**plan["controls"])
    merge = plan.get("operation") == "merge"
    rank = 0 if merge else int(os.environ.get("RANK", "-1"))
    if not merge and (not 0 <= rank < controls.gpu_processes or int(os.environ.get("WORLD_SIZE", "0")) != controls.gpu_processes):
        raise ValidationError("SmartSearch policy rank must be launched by the bound distributed runtime")
    rank_root = root / f"rank-{rank}-workspace"
    rank_root.mkdir(exist_ok=False)
    counters = {"forward_attempts": 0, "forward_responses": 0, "input_token_positions": 0,
                "forward_observation_scope": "primary_policy_after_on_train_begin",
                "unobserved_reference_forward_calls": None,
                "optimizer_steps": 0, "planned_optimizer_steps": None, "native_logs": [],
                "nonfinite_native_metrics": 0}
    started = time.monotonic()
    try:
        source = Path(plan["source"]["root"])
        for relative, digest in plan["source"]["policy_configs"].items():
            if _sha(_file(source / relative, "native worker source")) != digest:
                raise ExternalDependencyError("SmartSearch distributed native source/config changed")
        with _LOCK, _environment(rank_root, controls.seed), (root / f"native-rank-{rank}.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
            trainer = _source_import(source, plan.get("native_data_source"))
            if merge:
                trainer.export_model(args=plan["native_config"])
            else:
                trainer.run_exp(args=plan["native_config"], callbacks=[_observer(counters, controls.max_optimizer_steps)])
        _save(root, f"rank-{rank}-terminal.json", {"status": "completed", "rank": rank,
                                                 "plan_sha256": plan_sha256, "usage": counters,
                                                 "elapsed_seconds": time.monotonic() - started})
    except Exception as exc:
        _save(root, f"rank-{rank}-terminal.json", {"status": "failed", "rank": rank,
                                                 "plan_sha256": plan_sha256, "usage": counters,
                                                 "error_type": type(exc).__name__, "error": str(exc),
                                                 "elapsed_seconds": time.monotonic() - started})
        raise
