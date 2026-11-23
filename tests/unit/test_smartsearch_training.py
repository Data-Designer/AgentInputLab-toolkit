"""Recipe orchestration/closure fixtures; these do not run neural training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes import (
    SmartSearchGRPOControls,
    SmartSearchGRPOCostLimits,
    SmartSearchNativeTrainingRecipe,
    SmartSearchPolicyFitControls,
    SmartSearchSamplingControls,
    SmartSearchTrainingChainArtifact,
)
from agentinputlab.recipes import smartsearch_training as module


def test_explicit_native_stages_delegate_once_without_parameter_changes(monkeypatch):
    policy, grpo = Mock(), Mock()
    monkeypatch.setattr(module, "SmartSearchNativePolicyTraining", Mock(return_value=policy))
    monkeypatch.setattr(module, "SmartSearchNativeGRPOTraining", Mock(return_value=grpo))
    limits = SmartSearchGRPOCostLimits(10, 81920, 20, 40960, 2, 8)
    recipe = SmartSearchNativeTrainingRecipe(grpo_controls=SmartSearchGRPOControls(gpu_processes=2),
                                            grpo_cost_limits=limits)
    for stage in ("sft", "dpo"):
        assert recipe.fit(stage=stage, checkout="pinned-source", output_root="fresh") is policy.fit.return_value
        policy.fit.assert_called_with(stage=stage, checkout="pinned-source", output_root="fresh")
    assert recipe.fit(stage="merged_dpo", model_root="same-base") is policy.merge.return_value
    policy.merge.assert_called_once_with(model_root="same-base")
    assert recipe.fit(stage="grpo", data_root="frozen-data") is grpo.fit.return_value
    grpo.fit.assert_called_once_with(data_root="frozen-data")


def test_registered_public_factory_preserves_native_training_profile():
    from agentinputlab.api import create_intervention
    recipe = create_intervention("training.smartsearch.sft_dpo_grpo")
    assert isinstance(recipe, SmartSearchNativeTrainingRecipe)
    assert recipe.release_id == "training.smartsearch.sft_dpo_grpo"


def test_no_retry_after_native_exception(monkeypatch):
    native = Mock()
    native.fit.side_effect = ExternalDependencyError("retained paid native failure")
    monkeypatch.setattr(module, "SmartSearchNativePolicyTraining", Mock(return_value=native))
    with pytest.raises(ExternalDependencyError, match="retained paid"):
        SmartSearchNativeTrainingRecipe().fit(stage="dpo", output_root="one-attempt")
    native.fit.assert_called_once()


@pytest.mark.parametrize("stage", ["all", "retry", "sft_then_dpo", [], None])
def test_invalid_or_implicit_stage_does_not_start_a_trainer(stage, monkeypatch):
    native = Mock()
    monkeypatch.setattr(module, "SmartSearchNativePolicyTraining", native)
    with pytest.raises(ValidationError):
        SmartSearchNativeTrainingRecipe().fit(stage=stage)
    native.assert_not_called()


def test_grpo_requires_explicit_aggregate_costs_before_execution(monkeypatch):
    native = Mock()
    monkeypatch.setattr(module, "SmartSearchNativeGRPOTraining", native)
    with pytest.raises(ValidationError, match="aggregate cost limits"):
        SmartSearchNativeTrainingRecipe().fit(stage="grpo")
    native.assert_not_called()


@pytest.mark.parametrize("values", [dict(policy_controls={}), dict(sampling_controls=False),
    dict(teacher_controls=0), dict(grpo_controls=[]), dict(grpo_cost_limits={})])
def test_controls_are_typed_even_when_empty_or_false(values):
    with pytest.raises(ValidationError, match="typed"):
        SmartSearchNativeTrainingRecipe(**values)


def test_stage_seed_change_is_not_silent():
    with pytest.raises(ValidationError, match="seeds differ"):
        SmartSearchNativeTrainingRecipe(policy_controls=SmartSearchPolicyFitControls(seed=18),
            sampling_controls=SmartSearchSamplingControls(seed=17))


def test_sampling_teacher_and_preparation_are_concrete_native_apis(monkeypatch):
    sampler, teacher, curriculum = Mock(), Mock(), Mock()
    sampler_factory, teacher_factory = Mock(return_value=sampler), Mock(return_value=teacher)
    monkeypatch.setattr(module, "SmartSearchPolicySampler", sampler_factory)
    monkeypatch.setattr(module, "SmartSearchProcessTeacher", teacher_factory)
    monkeypatch.setattr(module, "SmartSearchCurriculumPreparation", Mock(return_value=curriculum))
    prepare_grpo = Mock()
    monkeypatch.setattr(module, "prepare_smartsearch_grpo_data", prepare_grpo)
    recipe = SmartSearchNativeTrainingRecipe()
    recipe.sample(model="provider", retriever="frozen-corpus", case="declared-train")
    sampler_factory.assert_called_once_with("provider", "frozen-corpus", controls=recipe.sampling_controls)
    sampler.sample.assert_called_once_with(case="declared-train")
    recipe.annotate(generate="teacher", model_fingerprint_sha256="a" * 64, refine=False)
    teacher_factory.assert_called_once_with("teacher", model_fingerprint_sha256="a" * 64,
                                           controls=recipe.teacher_controls)
    teacher.annotate.assert_called_once_with(refine=False)
    recipe.prepare_curriculum(checkout="source")
    curriculum.prepare.assert_called_once_with(checkout="source")
    recipe.prepare_grpo_data(train="declared-train", validation="declared-dev")
    prepare_grpo.assert_called_once_with(train="declared-train", validation="declared-dev")


def split():
    return {role: dict(case_ids=ids, question_hashes=questions) for role, ids, questions in (
        ("train", ["t"], ["1" * 64]), ("validation", ["d"], ["2" * 64]), ("test", [], []))}


@pytest.fixture
def chain(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ("source", "sft", "dpo", "merged", "grpo", "grpo-data",
        "sft-model", "merged-model", "initial-curriculum", "dpo-curriculum")}
    for root in roots.values():
        root.mkdir()
    source = dict(source_commit="bound-source", grpo_source={"original.py": "a" * 64})
    files = {"model.safetensors": "3" * 64}
    policy = {name: SimpleNamespace(root=roots[name], artifact_id=name + "-id", stage=stage)
        for name, stage in (("sft", "sft"), ("dpo", "dpo"), ("merged", "merged_dpo"))}
    grpo = SimpleNamespace(root=roots["grpo"], artifact_id="grpo-id")
    plans = {}
    for name in ("sft", "dpo", "merged", "grpo"):
        record = dict(artifact_id=name + "-id", source_commit="bound-source", checkpoint_files=files)
        if name in {"dpo", "merged"}:
            record["base_model_manifest_sha256"] = "a" * 64
        if name == "merged":
            record["dpo_artifact_id"] = "dpo-id"
        if name in {"sft", "dpo"}:
            record["curriculum_artifact_id"] = name + "-curriculum-id"
        if name == "grpo":
            record["data_artifact_id"] = "grpo-data-id"
        (roots[name] / "artifact.json").write_text(json.dumps(record))
        plans[name] = dict(controls=dict(seed=17), split_audit=split())
        if name == "grpo":
            plans[name].update(model_manifest_sha256="b" * 64, model_revision="merged-revision",
                model_provenance=dict(merged_dpo_artifact_id="merged-id"),
                planned_global_steps=4, planned_actor_optimizer_steps=8)
        (roots[name] / "fit-plan.json").write_text(json.dumps(plans[name]))
        terminal = dict(usage=dict(failed=False, pending_generation_tickets=[], synchronized_optimizer_steps=8),
                        native_driver_result=dict(global_steps=4))
        (roots[name] / "fit-terminal.json").write_text(json.dumps(terminal))
    data_record = dict(source=source, split_audit=split())
    (roots["grpo-data"] / "manifest.json").write_text(json.dumps(data_record))
    (roots["grpo"] / "model-manifest.json").write_text('{}')
    curricula = {name: dict(split_audit=split()) for name in ("sft", "dpo")}
    monkeypatch.setattr(module, "verify_smartsearch_grpo_source", lambda _: (roots["source"], source))
    monkeypatch.setattr(module, "_source", lambda _: (roots["source"], source))
    monkeypatch.setattr(module.SmartSearchPolicyStageArtifact, "load", lambda path: policy[Path(path).name])
    monkeypatch.setattr(module.SmartSearchGRPOArtifact, "load", lambda path: grpo)
    monkeypatch.setattr(module.SmartSearchGRPODataArtifact, "load", lambda path: SimpleNamespace(
        root=roots["grpo-data"], artifact_id="grpo-data-id"))
    monkeypatch.setattr(module, "_curriculum", lambda path, stage, cap: ([{}], curricula[stage], stage + "-curriculum-id"))
    def model(path, _manifest, _digest):
        stage = "sft" if Path(path).name == "sft-model" else "merged"
        return roots[stage + "-model"], dict(files=files, training_artifact_id=stage + "-id", revision=stage + "-revision")
    monkeypatch.setattr(module, "_model", model)
    references = dict(checkout=roots["source"], sft_root=roots["sft"], dpo_root=roots["dpo"],
        merged_root=roots["merged"], grpo_root=roots["grpo"], grpo_data_root=roots["grpo-data"],
        initial_curriculum_root=roots["initial-curriculum"], dpo_curriculum_root=roots["dpo-curriculum"])
    for stage, digest in (("sft", "a"), ("merged", "b")):
        references[stage + "_snapshot"] = dict(model_root=roots[stage + "-model"],
            model_manifest=tmp_path / (stage + "-manifest.json"), model_manifest_sha256=digest * 64)
    return roots, references, curricula


def mutate_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_chain_closes_without_copying_or_fitting_and_reloads_after_cwd_change(chain, tmp_path, monkeypatch):
    _, references, _ = chain
    result = SmartSearchNativeTrainingRecipe().assemble(output_root=tmp_path / "chain", **references)
    record = json.loads((result.root / "artifact.json").read_text())
    assert record["chain"]["native_optimizer_and_safe_export_chain_closed"] is True
    assert record["chain"]["full_training_recipe_compatibility_certified"] is False
    assert record["no_model_or_optimizer_invocation"] is True
    assert sorted(p.name for p in result.root.iterdir()) == ["artifact.json"]
    monkeypatch.chdir(tmp_path.parent)
    assert SmartSearchTrainingChainArtifact.load(result.root).artifact_id == result.artifact_id


@pytest.mark.parametrize("kind", ["dpo_base", "merged_adapter", "grpo_base", "grpo_source", "seed", "unresolved_costs", "steps"])
def test_mixed_or_incomplete_chain_is_rejected_before_receipt_creation(chain, tmp_path, kind):
    roots, references, _ = chain
    if kind == "dpo_base":
        mutate_json(roots["dpo"] / "artifact.json", lambda d: d.update(base_model_manifest_sha256="c" * 64))
    elif kind == "merged_adapter":
        mutate_json(roots["merged"] / "artifact.json", lambda d: d.update(dpo_artifact_id="different-adapter"))
    elif kind == "grpo_base":
        mutate_json(roots["grpo"] / "fit-plan.json", lambda d: d.update(model_manifest_sha256="c" * 64))
    elif kind == "grpo_source":
        mutate_json(roots["grpo"] / "artifact.json", lambda d: d.update(source_commit="other-source"))
    elif kind == "seed":
        mutate_json(roots["dpo"] / "fit-plan.json", lambda d: d.update(controls=dict(seed=18)))
    elif kind == "unresolved_costs":
        mutate_json(roots["grpo"] / "fit-terminal.json", lambda d: d["usage"].update(pending_generation_tickets=["paid-unknown"]))
    else:
        mutate_json(roots["grpo"] / "fit-terminal.json", lambda d: d["usage"].update(synchronized_optimizer_steps=7))
    with pytest.raises(ValidationError):
        SmartSearchNativeTrainingRecipe().assemble(output_root=tmp_path / "not-created", **references)
    assert not (tmp_path / "not-created").exists()


def test_changed_training_collection_is_not_a_new_stage_of_same_recipe(chain, tmp_path):
    _, references, curricula = chain
    curricula["dpo"]["split_audit"]["train"]["question_hashes"] = ["9" * 64]
    with pytest.raises(ValidationError, match="different training"):
        SmartSearchNativeTrainingRecipe().assemble(output_root=tmp_path / "not-created", **references)


def test_snapshot_identifier_and_tensor_closure_are_both_required(chain, monkeypatch):
    _, references, _ = chain
    monkeypatch.setattr(module, "_model", lambda *_: (Path("unused"), dict(
        files={"model.safetensors": "changed"}, training_artifact_id="sft-id", revision="x")))
    with pytest.raises(ValidationError, match="safe file export"):
        module._verify_chain(references)


def test_every_trained_policy_shard_must_be_in_snapshot(chain):
    roots, references, _ = chain
    mutate_json(roots["sft"] / "artifact.json", lambda d: d["checkpoint_files"].update(
        {"model-00002-of-00002.safetensors": "9" * 64}))
    with pytest.raises(ValidationError, match="safe file export"):
        module._verify_chain(references)


def test_only_explicit_non_inference_auxiliary_descriptor_is_allowed(chain, monkeypatch):
    _, references, _ = chain
    native = module._model
    def model(*args):
        root, record = native(*args)
        record["files"]["Modelfile"] = "f" * 64
        return root, record
    monkeypatch.setattr(module, "_model", model)
    summary, _ = module._verify_chain(references)
    assert summary["separately_hashed_non_inference_auxiliary_files"]["merged_snapshot"] == ["Modelfile"]
    def unsafe_model(*args):
        root, record = native(*args)
        record["files"]["unbound_config.json"] = "f" * 64
        return root, record
    monkeypatch.setattr(module, "_model", unsafe_model)
    with pytest.raises(ValidationError, match="safe file export"):
        module._verify_chain(references)


def test_protected_input_overlap_and_create_only_receipts(chain, tmp_path):
    roots, references, _ = chain
    recipe = SmartSearchNativeTrainingRecipe()
    with pytest.raises(ValidationError, match="overlaps"):
        recipe.assemble(output_root=roots["grpo"] / "overwrite", **references)
    recipe.assemble(output_root=tmp_path / "receipt", **references)
    with pytest.raises(ValidationError):
        recipe.assemble(output_root=tmp_path / "receipt", **references)


def test_saved_chain_detects_later_stage_receipt_changes(chain, tmp_path):
    roots, references, _ = chain
    artifact = SmartSearchNativeTrainingRecipe().assemble(output_root=tmp_path / "receipt", **references)
    mutate_json(roots["dpo"] / "fit-terminal.json", lambda d: d.update(extra_observation="changed"))
    with pytest.raises(ValidationError, match="input artifacts changed"):
        SmartSearchTrainingChainArtifact.load(artifact.root)


def test_split_binding_rejects_train_development_overlap():
    audit = split()
    audit["validation"] = copy.deepcopy(audit["train"])
    with pytest.raises(ValidationError, match="overlap"):
        module._split_identity(audit)
