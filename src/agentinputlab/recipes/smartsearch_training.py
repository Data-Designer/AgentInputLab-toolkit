"""Stage-oriented public SmartSearch training recipe and verified artifact chain.

Native policy and GRPO runtimes differ, so stages run in explicit dedicated
processes. No implicit retry, fabricated empty selection, or unverified resume
is performed. Training-chain closure is not benchmark acceptance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash

from .qrag import _file, _sha
from .smartsearch_curriculum import SmartSearchCurriculumPreparation
from .smartsearch_grpo_data import (
    SmartSearchGRPOControls,
    SmartSearchGRPODataArtifact,
    prepare_smartsearch_grpo_data,
    verify_smartsearch_grpo_source,
)
from .smartsearch_grpo_meter import SmartSearchGRPOCostLimits
from .smartsearch_grpo_tools import _new_output, _save
from .smartsearch_grpo_training import SmartSearchGRPOArtifact, SmartSearchNativeGRPOTraining
from .smartsearch_policy import (
    SmartSearchNativePolicyTraining,
    SmartSearchPolicyFitControls,
    SmartSearchPolicyStageArtifact,
    _curriculum,
    _model,
    _source,
)
from .smartsearch_sampling import SmartSearchPolicySampler, SmartSearchSamplingControls
from .smartsearch_teacher import SmartSearchProcessTeacher, SmartSearchTeacherControls

TRAINING_RECIPE_PROTOCOL = "agentinputlab.smartsearch.native_training_chain.v1"
_HF_POLICY_METADATA = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt"}
# An accepted older native export contains this inert Ollama descriptor. The
# native HF/SFT/DPO/GRPO loaders never read it; its hash remains in the snapshot
# manifest, but it is not falsely attributed to the trained checkpoint payload.
_NON_INFERENCE_AUXILIARY_FILES = {"Modelfile"}


def _read(path):
    return json.loads(_file(Path(path), "SmartSearch recipe receipt").read_text())


def _split_identity(audit):
    """Compare actual case IDs/questions, not incidental stage provenance."""
    if not isinstance(audit, dict) or set(audit) != {"train", "validation", "test"}:
        raise ValidationError("SmartSearch native chain needs explicit train/development/test split bindings")
    result, all_ids, all_questions = {}, set(), set()
    for role, item in audit.items():
        if not isinstance(item, dict):
            raise ValidationError("SmartSearch native split binding must be an object")
        ids, questions = item.get("case_ids"), item.get("question_hashes")
        if not isinstance(ids, (list, tuple)) or not isinstance(questions, (list, tuple)) or (
            len(ids) != len(questions) or (role != "test" and not ids)
            or any(not isinstance(v, str) or not v for v in ids)
            or any(not isinstance(v, str) or len(v) != 64 or set(v) - set("0123456789abcdef") for v in questions)
            or len(set(ids)) != len(ids) or len(set(questions)) != len(questions)
            or all_ids & set(ids) or all_questions & set(questions)
        ):
            raise ValidationError("SmartSearch native chain split IDs/questions duplicate, overlap or differ")
        result[role] = dict(sorted(zip(ids, questions, strict=True)))
        all_ids.update(ids)
        all_questions.update(questions)
    return result


def _snapshot_from_stage(stage, root, manifest, digest):
    model, record = _model(root, manifest, digest)
    checkpoint = _read(stage.root / "artifact.json")["checkpoint_files"]
    required = {name for name in checkpoint if name in _HF_POLICY_METADATA or
                name.endswith((".safetensors", ".safetensors.index.json", ".model", ".jinja"))}
    identifiers = [record[key] for key in ("training_artifact_id", "native_merged_dpo_artifact_id") if key in record]
    if not identifiers or any(identity != stage.artifact_id for identity in identifiers) or (
        not required <= record["files"].keys()
    ) or any(
        checkpoint.get(name) != value for name, value in record["files"].items()
        if name not in _NON_INFERENCE_AUXILIARY_FILES
    ):
        raise ValidationError("SmartSearch policy snapshot is not the same trained stage's safe file export")
    return model, record


def _verify_chain(references):
    if not isinstance(references, dict) or set(references) != {
        "checkout", "sft_root", "sft_snapshot", "initial_curriculum_root", "dpo_root",
        "dpo_curriculum_root", "merged_root", "merged_snapshot", "grpo_root", "grpo_data_root",
    }:
        raise ValidationError("SmartSearch native training chain reference set is incomplete")
    source, binding = verify_smartsearch_grpo_source(references["checkout"])
    _source(source)
    policy = {name: SmartSearchPolicyStageArtifact.load(references[name + "_root"])
              for name in ("sft", "dpo", "merged")}
    if [policy[n].stage for n in ("sft", "dpo", "merged")] != ["sft", "dpo", "merged_dpo"]:
        raise ValidationError("SmartSearch training chain must contain SFT, DPO and merged-DPO stages in order")
    grpo = SmartSearchGRPOArtifact.load(references["grpo_root"])
    records = {name: _read(artifact.root / "artifact.json") for name, artifact in {**policy, "grpo": grpo}.items()}
    plans = {name: _read(artifact.root / "fit-plan.json") for name, artifact in {**policy, "grpo": grpo}.items()}
    if any(record.get("source_commit") != binding["source_commit"] for record in records.values()):
        raise ValidationError("SmartSearch training chain mixes different upstream source revisions")
    seeds = [plan.get("controls", {}).get("seed") for plan in plans.values()]
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 1:
        raise ValidationError("SmartSearch native chain stage seeds differ")
    for name in ("sft_snapshot", "merged_snapshot"):
        item = references[name]
        if not isinstance(item, dict) or set(item) != {"model_root", "model_manifest", "model_manifest_sha256"}:
            raise ValidationError("SmartSearch chain snapshots require explicit model/manifest/hash bindings")
    sft_snapshot, merged_snapshot = references["sft_snapshot"], references["merged_snapshot"]
    sft_model, sft_record = _snapshot_from_stage(policy["sft"], sft_snapshot["model_root"],
        sft_snapshot["model_manifest"], sft_snapshot["model_manifest_sha256"])
    merged_model, merged_record = _snapshot_from_stage(policy["merged"], merged_snapshot["model_root"],
        merged_snapshot["model_manifest"], merged_snapshot["model_manifest_sha256"])
    if any(records[name].get("base_model_manifest_sha256") != sft_snapshot["model_manifest_sha256"]
           for name in ("dpo", "merged")) or records["merged"].get("dpo_artifact_id") != policy["dpo"].artifact_id:
        raise ValidationError("SmartSearch DPO/merge does not use the same fitted SFT policy and adapter")
    if plans["grpo"].get("model_manifest_sha256") != merged_snapshot["model_manifest_sha256"] or (
        plans["grpo"].get("model_provenance", {}).get("merged_dpo_artifact_id") != policy["merged"].artifact_id
        or plans["grpo"].get("model_revision") != merged_record["revision"]
    ):
        raise ValidationError("SmartSearch GRPO is not based on the bound merged DPO policy")
    _, initial, initial_id = _curriculum(references["initial_curriculum_root"], "sft", 64 * 1024**2)
    _, dpo_data, dpo_id = _curriculum(references["dpo_curriculum_root"], "dpo", 64 * 1024**2)
    if records["sft"].get("curriculum_artifact_id") != initial_id or records["dpo"].get("curriculum_artifact_id") != dpo_id:
        raise ValidationError("SmartSearch policy fitting used a different native curriculum artifact")
    data = SmartSearchGRPODataArtifact.load(references["grpo_data_root"])
    data_record = _read(data.root / "manifest.json")
    if records["grpo"].get("data_artifact_id") != data.artifact_id or data_record["source"] != binding:
        raise ValidationError("SmartSearch GRPO data/source artifact differs from the training chain")
    identities = [_split_identity(value["split_audit"]) for value in
                  (initial, dpo_data, plans["sft"], plans["dpo"], data_record)]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValidationError("SmartSearch SFT/DPO/GRPO stages use different training/development/test collections")
    terminal = _read(grpo.root / "fit-terminal.json")
    usage = terminal.get("usage", {})
    if usage.get("failed") is not False or usage.get("pending_generation_tickets") != [] or (
        usage.get("synchronized_optimizer_steps") != plans["grpo"].get("planned_actor_optimizer_steps")
        or type(usage.get("synchronized_optimizer_steps")) is not int
        or usage["synchronized_optimizer_steps"] < 1
        or terminal.get("native_driver_result", {}).get("global_steps") != plans["grpo"].get("planned_global_steps")
    ):
        raise ValidationError("SmartSearch native GRPO has unresolved costs or an incomplete optimizer plan")
    summary = dict(source_commit=binding["source_commit"], split_identity=identities[0],
        artifacts={name: record["artifact_id"] for name, record in records.items()},
        initial_curriculum_artifact_id=initial_id, dpo_curriculum_artifact_id=dpo_id,
        grpo_data_artifact_id=data.artifact_id,
        receipt_files={name: dict(artifact_sha256=_sha(artifact.root / "artifact.json"),
            plan_sha256=_sha(artifact.root / "fit-plan.json"), terminal_sha256=_sha(artifact.root / "fit-terminal.json"))
            for name, artifact in {**policy, "grpo": grpo}.items()},
        sft_snapshot_manifest_sha256=sft_snapshot["model_manifest_sha256"],
        merged_snapshot_manifest_sha256=merged_snapshot["model_manifest_sha256"],
        learned_grpo_manifest_sha256=_sha(grpo.root / "model-manifest.json"),
        separately_hashed_non_inference_auxiliary_files={name: sorted(set(record["files"]) & _NON_INFERENCE_AUXILIARY_FILES)
            for name, record in (("sft_snapshot", sft_record), ("merged_snapshot", merged_record))},
        native_optimizer_and_safe_export_chain_closed=True,
        sampling_teacher_provenance_and_heldout_acceptance_separate=True,
        full_training_recipe_compatibility_certified=False, new_model_calls=0, new_optimizer_steps=0)
    protected = [source, sft_model, merged_model, data.root, *[artifact.root for artifact in policy.values()], grpo.root,
                 Path(references["initial_curriculum_root"]).resolve(), Path(references["dpo_curriculum_root"]).resolve()]
    return summary, protected


@dataclass(frozen=True, slots=True)
class SmartSearchTrainingChainArtifact:
    root: Path
    artifact_id: str

    @classmethod
    def load(cls, root):
        supplied = Path(root)
        if supplied.is_symlink() or not supplied.is_dir():
            raise ValidationError("SmartSearch training-chain artifact must be a regular directory")
        folder = supplied.resolve()
        record = _read(folder / "artifact.json")
        identity = record.pop("artifact_id", None)
        if record.get("protocol") != TRAINING_RECIPE_PROTOCOL or identity != content_hash(record):
            raise ValidationError("SmartSearch training-chain artifact identity differs")
        summary, _ = _verify_chain(record.get("references"))
        if summary != record.get("chain"):
            raise ValidationError("SmartSearch training-chain input artifacts changed")
        return cls(folder, identity)


class SmartSearchNativeTrainingRecipe:
    """Concrete native stage API; use each stage's dedicated dependency runtime.

    `fit(stage=...)` executes the actual native SFT, DPO, merge or GRPO API.
    `assemble` verifies an already completed optimizer/checkpoint chain without
    re-running it. Neither operation declares all benchmark/model combinations
    accepted, and no stage is automatically retried or skipped on failure.
    """

    release_id = "training.smartsearch.sft_dpo_grpo"

    def __init__(self, *, policy_controls=None, sampling_controls=None, teacher_controls=None,
                 grpo_controls=None, grpo_cost_limits=None):
        controls = (("policy", policy_controls if policy_controls is not None else SmartSearchPolicyFitControls(), SmartSearchPolicyFitControls),
                    ("sampling", sampling_controls if sampling_controls is not None else SmartSearchSamplingControls(), SmartSearchSamplingControls),
                    ("teacher", teacher_controls if teacher_controls is not None else SmartSearchTeacherControls(), SmartSearchTeacherControls),
                    ("grpo", grpo_controls if grpo_controls is not None else SmartSearchGRPOControls(), SmartSearchGRPOControls))
        if any(not isinstance(value, kind) for _, value, kind in controls) or (
            grpo_cost_limits is not None and not isinstance(grpo_cost_limits, SmartSearchGRPOCostLimits)
        ):
            raise ValidationError("SmartSearch native recipe requires typed per-stage controls and aggregate GRPO limits")
        if len({value.seed for _, value, _ in controls}) != 1:
            raise ValidationError("SmartSearch native recipe stage seeds differ; declare a separate explicit variant")
        for name, value, _ in controls:
            setattr(self, name + "_controls", value)
        self.grpo_cost_limits = grpo_cost_limits

    def sample(self, *, model, retriever, **inputs):
        return SmartSearchPolicySampler(model, retriever, controls=self.sampling_controls).sample(**inputs)

    def annotate(self, *, generate, model_fingerprint_sha256, **inputs):
        return SmartSearchProcessTeacher(generate, model_fingerprint_sha256=model_fingerprint_sha256,
            controls=self.teacher_controls).annotate(**inputs)

    def prepare_curriculum(self, **inputs):
        return SmartSearchCurriculumPreparation().prepare(**inputs)

    def prepare_grpo_data(self, **inputs):
        return prepare_smartsearch_grpo_data(**inputs)

    def inference(self, *, training_chain_root, model, retriever, controls=None):
        """Bind a completed fitted policy; never use the train/development sampler for tests."""
        from .smartsearch_inference import SmartSearchTrainedPolicyInference

        return SmartSearchTrainedPolicyInference(training_chain_root=training_chain_root,
            model=model, retriever=retriever, controls=controls)

    def fit(self, *, stage, **inputs):
        if not isinstance(stage, str):
            raise ValidationError("SmartSearch native stage must be an explicit string")
        if stage in {"sft", "dpo"}:
            return SmartSearchNativePolicyTraining(self.policy_controls).fit(stage=stage, **inputs)
        if stage == "merged_dpo":
            return SmartSearchNativePolicyTraining(self.policy_controls).merge(**inputs)
        if stage == "grpo":
            if self.grpo_cost_limits is None:
                raise ValidationError("SmartSearch GRPO requires explicit aggregate cost limits before execution")
            return SmartSearchNativeGRPOTraining(controls=self.grpo_controls,
                cost_limits=self.grpo_cost_limits).fit(**inputs)
        raise ValidationError("Run an explicit native stage: sft, dpo, merged_dpo or grpo; no implicit all-stage retry")

    def assemble(self, *, output_root, **references):
        summary, protected = _verify_chain(references)
        output = Path(output_root).resolve(strict=False)
        if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
            raise ValidationError("SmartSearch training-chain receipt overlaps protected stage inputs")
        root = _new_output(output_root)
        references = {key: ({name: (str(Path(value).resolve()) if name != "model_manifest_sha256" else value)
                              for name, value in item.items()} if isinstance(item, dict)
                             else str(Path(item).resolve())) for key, item in references.items()}
        record = dict(protocol=TRAINING_RECIPE_PROTOCOL, references=references, chain=summary,
            no_model_or_optimizer_invocation=True, checkpoint_weights_not_copied=True,
            relative_relocation_requires_explicit_rebinding=True,
            not_benchmark_or_full_recipe_acceptance=True, toolkit_recipe_sha256=_sha(Path(__file__)))
        identity = content_hash(record)
        _save(root, "artifact.json", {**record, "artifact_id": identity})
        return SmartSearchTrainingChainArtifact(root, identity)
