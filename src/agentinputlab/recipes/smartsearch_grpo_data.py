"""Pinned native SmartSearch GRPO data/configuration, with explicit split roles."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .qrag import _file, _sha
from .smartsearch import SMARTSEARCH_TOOL_SCHEMA
from .smartsearch_curriculum import (
    SmartSearchTrainingCase,
    _question_hash,
    validate_smartsearch_training_splits,
)
from .smartsearch_grpo_tools import SMARTSEARCH_GRPO_ENVIRONMENT
from .smartsearch_sampling import verify_smartsearch_sampling_source

GRPO_DATA_PROTOCOL = "agentinputlab.smartsearch.native_grpo_data.v1"
_SOURCE = {
    "data/prepare_dataset.py": "abf1dc41c410e8716c02b8651c9ef3fb3bbba6d52b50ae86db8325ecf5154554",
    "scripts/train/train.sh": "07903f49e46bf18a0d764b6e92c1430d0a651523751fee653dec112379c66d72",
    "src/verl/trainer/main_ppo.py": "6eace6c898923fe2e25081f473801c9439a709eb02d1968944f1f2cd2602f258",
    "src/verl/trainer/config/ppo_trainer.yaml": "8a239cda92c40d520df0344857b5f51d6550254c449d28841e1de3f5b3feb1d7",
    "src/verl/utils/dataset/rl_dataset.py": "06a599f81f7288fdd183111b15207c37e0c32e2d9399c8e9289edba637bf5f26",
    "src/verl/workers/rollout/vllm_rollout/generate_new_prompts.py": "bc1f7d2a16d599e7f180406c9cd356034f1bb5ce0e97b5fb09d6264a11511615",
    "src/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py": "064aa3d6682149620c5b55239379199130941dc05d6fc5ed13026987b2867453",
    "src/verl/workers/reward_manager/re_call.py": "0b239aaa9ee0c21b693359569e3f091bc650d537d7cf2a63e478c8fb4d976a7b",
    "src/verl/trainer/ppo/ray_trainer.py": "465e1b53dfa9f0bfcdb597957cef64ea18941e5e75a2af17a48c0ecfd00ab3e2",
    "src/verl/workers/actor/dp_actor.py": "09923f0891e2ca913e1eb860bfa7e62537b10b1f4527b9647784f7063ba86a8d",
    "src/verl/workers/fsdp_workers.py": "65aa287db53a3308d5b579f47cf557ab8333231e26625ed62bbb65ca90153833",
    "src/verl/utils/checkpoint/fsdp_checkpoint_manager.py": "640556d9c0801602818fcfd52aa6d2d066b49bf30f30eb0a81c08efde2495f4a",
}


def verify_smartsearch_grpo_source(checkout):
    root, binding = verify_smartsearch_sampling_source(checkout)
    for name, expected in _SOURCE.items():
        path = _file(root / name, "SmartSearch native GRPO source")
        if not path.is_relative_to(root) or _sha(path) != expected:
            raise ExternalDependencyError("SmartSearch native GRPO source/configuration changed")
    return root, {**binding, "grpo_source": dict(_SOURCE)}


def _save(root, name, value):
    encoded = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    with (root / name).open("x", encoding="utf-8") as stream:
        stream.write(encoded)


@dataclass(frozen=True, slots=True)
class SmartSearchGRPOExample:
    case: SmartSearchTrainingCase
    golden_answers: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.case, SmartSearchTrainingCase) or self.case.split not in {"train", "validation"}:
            raise ValidationError("SmartSearch GRPO examples must declare training/development case roles")
        answers = self.golden_answers
        if not isinstance(answers, (tuple, list)) or not answers or any(
            not isinstance(value, str) or not value.strip() for value in answers
        ):
            raise ValidationError("SmartSearch GRPO examples require nonempty native reference answers")
        object.__setattr__(self, "golden_answers", tuple(answers))


@dataclass(frozen=True, slots=True)
class SmartSearchGRPODataArtifact:
    root: Path
    artifact_id: str
    train_rows: int
    validation_rows: int

    @classmethod
    def load(cls, root, *, max_rows=10000, max_file_bytes=64 * 1024 * 1024):
        if any(type(n) is not int or n < 1 for n in (max_rows, max_file_bytes)):
            raise ValidationError("SmartSearch GRPO artifact read limits must be positive integers")
        folder = Path(root)
        if folder.is_symlink() or not folder.is_dir():
            raise ValidationError("SmartSearch GRPO data artifact must be a regular directory")
        folder = folder.resolve()
        record = json.loads(_file(folder / "manifest.json", "GRPO data manifest").read_text())
        identity = record.pop("artifact_id", None)
        if record.get("schema") != GRPO_DATA_PROTOCOL or identity != content_hash(record):
            raise ValidationError("SmartSearch GRPO data artifact identity differs")
        if set(record.get("outputs", {})) != {"train.parquet", "validation.parquet"}:
            raise ValidationError("SmartSearch GRPO data artifact has an incomplete closure")
        for name, digest in record["outputs"].items():
            path = _file(folder / name, "GRPO native data")
            if path.stat().st_size > max_file_bytes or _sha(path) != digest:
                raise ValidationError("SmartSearch GRPO native data bytes changed")
        for key in ("train_rows", "validation_rows"):
            if type(record.get(key)) is not int or record[key] < 1:
                raise ValidationError("SmartSearch GRPO data has an empty declared split")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ExternalDependencyError("Install agentinputlab[benchmarks] to verify native GRPO data") from exc
        for split in ("train", "validation"):
            path = folder / f"{split}.parquet"
            declared = record[split + "_rows"]
            if pq.read_metadata(path).num_rows != declared or declared > max_rows:
                raise ValidationError("SmartSearch GRPO parquet row count differs or exceeds its read limit")
            rows = pq.read_table(path).to_pylist()
            hashes = []
            for row in rows:
                if set(row) != {"data_source", "question", "ability", "reward_model", "extra_info"} or (
                    row["data_source"] != "re_call" or row["ability"] != "re_call"
                ) or not isinstance(row["question"], str) or not row["question"].strip():
                    raise ValidationError("SmartSearch GRPO native logical fields differ")
                reward, extra = row["reward_model"], row["extra_info"]
                if not isinstance(reward, dict) or set(reward) != {"style", "ground_truth"} or (
                    reward["style"] != "rule" or not isinstance(reward["ground_truth"], list)
                    or not reward["ground_truth"] or any(not isinstance(v, str) or not v.strip() for v in reward["ground_truth"])
                ) or not isinstance(extra, dict) or set(extra) != {"env", "func_schemas"} or (
                    extra["env"] != SMARTSEARCH_GRPO_ENVIRONMENT
                    or json.loads(extra["func_schemas"]) != to_jsonable(SMARTSEARCH_TOOL_SCHEMA)
                ):
                    raise ValidationError("SmartSearch GRPO reference/tool field contract differs")
                hashes.append(_question_hash(row["question"]))
            audit = record.get("split_audit", {}).get(split, {})
            if hashes != audit.get("question_hashes") or len(set(hashes)) != declared:
                raise ValidationError("SmartSearch GRPO native questions differ from their declared split")
        if set(record["split_audit"]["train"]["question_hashes"]) & set(
            record["split_audit"]["validation"]["question_hashes"]
        ):
            raise ValidationError("SmartSearch GRPO data artifact splits overlap")
        return cls(folder, identity, record["train_rows"], record["validation_rows"])


def prepare_smartsearch_grpo_data(
    *, checkout, output_root, train: Sequence[SmartSearchGRPOExample],
    validation: Sequence[SmartSearchGRPOExample], data_provenance: Mapping,
    test: Sequence[SmartSearchTrainingCase] = (), max_rows: int = 10000,
    max_input_bytes: int = 64 * 1024 * 1024,
):
    """Write exact native logical rows without upstream's test-to-dev sampling.

    Corpus/retrieval is a separate frozen input. References remain in reward_model
    and never enter the question field. The environment is replaced with the
    fixed non-executable retrieval identifier. No model is loaded or fit here.
    """
    if any(type(n) is not int or n < 1 for n in (max_rows, max_input_bytes)) or (
        not isinstance(data_provenance, Mapping) or not data_provenance
    ):
        raise ValidationError("SmartSearch GRPO preparation needs explicit limits and provenance")
    for rows in (train, validation):
        if not isinstance(rows, (tuple, list)) or not rows or len(rows) > max_rows or any(
            not isinstance(row, SmartSearchGRPOExample) for row in rows
        ):
            raise ValidationError("SmartSearch GRPO preparation split exceeds its explicit row bound")
    train, validation = tuple(train), tuple(validation)
    data_provenance = to_jsonable(data_provenance)
    audit = validate_smartsearch_training_splits([r.case for r in train], [r.case for r in validation], test)
    encoded = json.dumps(to_jsonable({"train": train, "validation": validation,
                                     "provenance": data_provenance}), allow_nan=False).encode()
    if len(encoded) > max_input_bytes:
        raise ValidationError("SmartSearch GRPO data inputs exceed the explicit byte bound")
    source, binding = verify_smartsearch_grpo_source(checkout)
    root = Path(output_root)
    if root.is_symlink() or root.exists():
        raise ValidationError("SmartSearch GRPO preparation must be create-only")
    root = root.resolve(strict=False)
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValidationError("SmartSearch GRPO preparation output must not overlap native source")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ExternalDependencyError("Install agentinputlab[benchmarks] for native GRPO parquet data") from exc
    root.mkdir(parents=True, exist_ok=False)
    _save(root, "claim.json", {"source": binding, "split_audit": audit,
          "inputs_sha256": content_hash({"train": train, "validation": validation}),
          "data_provenance": data_provenance, "model_calls": 0, "optimizer_steps": 0})
    try:
        schema_text = json.dumps(to_jsonable(SMARTSEARCH_TOOL_SCHEMA), indent=4)
        for name, rows in (("train", train), ("validation", validation)):
            native_rows = [{"data_source": "re_call", "question": row.case.question,
                 "ability": "re_call", "reward_model": {"style": "rule", "ground_truth": list(row.golden_answers)},
                 "extra_info": {"env": SMARTSEARCH_GRPO_ENVIRONMENT, "func_schemas": schema_text}}
                for row in rows]
            pq.write_table(pa.Table.from_pylist(native_rows), root / f"{name}.parquet", compression="snappy")
            if pq.read_table(root / f"{name}.parquet").to_pylist() != native_rows:
                raise ExternalDependencyError("SmartSearch GRPO parquet roundtrip changed native rows")
        manifest = {"schema": GRPO_DATA_PROTOCOL, "source": binding, "split_audit": audit,
                    "train_rows": len(train), "validation_rows": len(validation),
                    "outputs": {name: _sha(root / name) for name in ("train.parquet", "validation.parquet")},
                    "data_provenance": data_provenance, "model_calls": 0, "optimizer_steps": 0,
                    "native_logical_fields_unchanged": True, "fixed_environment_identifier_adapter": True,
                    "no_native_test_to_dev_downsampling": True}
        _save(root, "manifest.json", {**manifest, "artifact_id": content_hash(manifest)})
        return SmartSearchGRPODataArtifact.load(root, max_rows=max_rows, max_file_bytes=max_input_bytes * 2 + 1024**2)
    except Exception as exc:
        _save(root, "failed.json", {"error_type": type(exc).__name__, "error": str(exc),
                                    "model_calls": 0, "optimizer_steps": 0})
        raise


@dataclass(frozen=True, slots=True)
class SmartSearchGRPOControls:
    train_batch_size: int = 256
    ppo_mini_batch_size: int = 64
    rollout_samples: int = 8
    total_epochs: int = 2
    gpu_processes: int = 4
    gpu_memory_utilization: float = .8
    max_optimizer_steps: int = 10000
    seed: int = 17
    bounded_training_variant: bool = False

    def __post_init__(self):
        names = ("train_batch_size", "ppo_mini_batch_size", "rollout_samples", "total_epochs",
                 "gpu_processes", "max_optimizer_steps")
        if any(type(getattr(self, n)) is not int or getattr(self, n) < 1 for n in names) or (
            type(self.seed) is not int or self.seed < 0 or type(self.bounded_training_variant) is not bool
        ):
            raise ValidationError("SmartSearch GRPO controls require explicit positive integer bounds")
        if not 1 <= self.gpu_processes <= 4 or self.rollout_samples < 2 or (
            self.ppo_mini_batch_size > self.train_batch_size or self.train_batch_size % self.ppo_mini_batch_size
        ) or (self.train_batch_size * self.rollout_samples) % self.gpu_processes or (
            (self.ppo_mini_batch_size * self.rollout_samples) % self.gpu_processes
        ):
            raise ValidationError("SmartSearch GRPO native batch/sample/GPU divisibility differs")
        utilization = self.gpu_memory_utilization
        if isinstance(utilization, bool) or not isinstance(utilization, (float, int)) or (
            not math.isfinite(utilization) or not 0 < utilization <= .95
        ):
            raise ValidationError("SmartSearch GRPO GPU cache allowance is invalid")
        if (self.train_batch_size, self.ppo_mini_batch_size, self.rollout_samples, self.total_epochs) != (
            256, 64, 8, 2
        ) and not self.bounded_training_variant:
            raise ValidationError("Changing native GRPO batch/n/epochs requires an explicit labeled training variant")

    def planned_steps(self, train_rows):
        """Native dataloader/global steps, not the number of actor updates."""
        if type(train_rows) is not int or train_rows < self.train_batch_size:
            raise ValidationError("SmartSearch native drop_last would produce an empty GRPO training loader")
        steps = train_rows // self.train_batch_size * self.total_epochs
        if steps > self.max_optimizer_steps:
            raise ValidationError("SmartSearch native optimizer plan exceeds the declared cap; it is not truncated")
        return steps

    def planned_actor_optimizer_steps(self, train_rows, *, ppo_epochs):
        """Synchronized actor updates per rank, before any neural work.

        Native FSDP normalizes the PPO minibatch by rollout_n/world_size;
        those factors cancel when dividing each rank's local training batch.
        Do not sum synchronized rank counters into extra distributed updates.
        The executor must still meter actual optimizer calls on every rank.
        """
        if type(ppo_epochs) is not int or ppo_epochs < 1:
            raise ValidationError("SmartSearch native PPO epochs must be a positive integer")
        global_steps = self.planned_steps(train_rows)
        steps = global_steps * (self.train_batch_size // self.ppo_mini_batch_size) * ppo_epochs
        if steps > self.max_optimizer_steps:
            raise ValidationError("SmartSearch native actor optimizer plan exceeds the declared cap; it is not truncated")
        return steps


def _native_grpo_config(source, *, model, data, output_root, controls, sandbox_url, reward_module):
    try:
        import yaml
    except ImportError as exc:
        raise ExternalDependencyError("Install agentinputlab[yaml] to build native GRPO config") from exc
    if not isinstance(sandbox_url, str):
        raise ValidationError("SmartSearch GRPO fixed tool URL must be operator-owned loopback HTTP")
    try:
        url = urlsplit(sandbox_url)
        port = url.port
    except ValueError as exc:
        raise ValidationError("SmartSearch GRPO fixed tool URL is malformed") from exc
    if url.scheme != "http" or url.hostname != "127.0.0.1" or not port or (
        url.username or url.password or url.query or url.fragment
    ):
        raise ValidationError("SmartSearch GRPO fixed tool URL must be operator-owned loopback HTTP")
    controls.planned_steps(data.train_rows)
    config = yaml.safe_load((Path(source) / "src/verl/trainer/config/ppo_trainer.yaml").read_text())
    config["data"].update(train_files=str(data.root / "train.parquet"), val_files=str(data.root / "validation.parquet"),
        prompt_key="question", train_batch_size=controls.train_batch_size, max_prompt_length=8192,
        max_response_length=8192, use_re_call=True, prompt_template_name="re_call_template_sys",
        search_url=sandbox_url, filter_overlong_prompts=True, trust_remote_code=False, seed=controls.seed,
        cache_dir=str(Path(output_root) / "data-cache"))
    policy = config["actor_rollout_ref"]
    policy["model"].update(path=str(model), enable_gradient_checkpointing=True, use_remove_padding=True)
    actor = policy["actor"]
    controls.planned_actor_optimizer_steps(data.train_rows, ppo_epochs=actor["ppo_epochs"])
    actor["optim"]["lr"] = 1e-6
    actor.update(ppo_mini_batch_size=controls.ppo_mini_batch_size, use_dynamic_bsz=True,
                 ppo_max_token_len_per_gpu=32768, use_kl_loss=True, kl_loss_coef=.001, kl_loss_type="low_var_kl")
    actor["fsdp_config"].update(param_offload=False, optimizer_offload=False)
    # Preserve native shards plus its HF config/tokenizer metadata. The native
    # checkpoint manager does NOT export HF weights; the live actor observer
    # separately gathers the actual trained state into safe HF tensors.
    actor["checkpoint"]["contents"] = [*actor["checkpoint"]["contents"], "hf_model"]
    policy["rollout"].update(log_prob_max_token_len_per_gpu=65536, tensor_model_parallel_size=1,
        name="vllm_with_tool", gpu_memory_utilization=controls.gpu_memory_utilization,
        n=controls.rollout_samples, max_turns=5, sandbox_url=sandbox_url, enforce_eager=False, free_cache_engine=False)
    policy["ref"]["log_prob_max_token_len_per_gpu"] = 65536
    policy["ref"]["fsdp_config"]["param_offload"] = True
    config["algorithm"].update(adv_estimator="grpo")
    config["algorithm"]["kl_ctrl"]["kl_coef"] = .001
    config["reward_model"]["reward_manager"] = "re_call"
    config["custom_reward_function"].update(path=str(reward_module), name="compute_score")
    config["trainer"].update(critic_warmup=0, logger=["console"], project_name="agentinputlab-smartsearch",
        experiment_name="native-grpo", n_gpus_per_node=controls.gpu_processes, nnodes=1,
        save_freq=10, test_freq=10, total_epochs=controls.total_epochs, resume_mode="disable",
        resume_from_path=None, default_hdfs_dir=None, default_local_dir=str(output_root),
        rollout_save_path=str(Path(output_root) / "rollouts"))
    return config
