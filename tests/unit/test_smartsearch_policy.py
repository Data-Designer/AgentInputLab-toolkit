"""Offline policy-stage contracts; fixture tensors are NOT real fitted weights."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import (
    SmartSearchNativePolicyTraining,
    SmartSearchPolicyFitControls,
    SmartSearchPolicyStageArtifact,
)
from agentinputlab.recipes.qrag import _sha
from agentinputlab.recipes.smartsearch_policy import (
    POLICY_PROTOCOL,
    _collect_rank_usage,
    _curriculum,
    _environment,
    _model,
    _native_config,
    _observer,
    _source_import,
)


def curriculum(tmp_path, stage="sft", fixture=False):
    root = tmp_path / "data"
    root.mkdir()
    row = {"instruction": "Fixture instruction", "input": "Fixture train question"}
    row.update(output="Fixture response") if stage == "sft" else row.update(chosen="Fixture positive", rejected="Fixture negative")
    path = root / f"{stage}.json"
    path.write_text(json.dumps([row]))
    manifest = {"schema": "agentinputlab.smartsearch.native-curriculum-data.v1",
                "rollout_provenance": {"fixture_only": fixture, "test_fixture_protocol_only": True},
                "outputs": {path.name: _sha(path)},
                "split_audit": {"train": {"question_hashes": [content_hash("fixture train question")]},
                                "validation": {"case_ids": ["dev"], "question_hashes": ["different"]}}}
    manifest["artifact_id"] = content_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def test_native_curriculum_binding_and_declared_training_questions(tmp_path):
    root, manifest = curriculum(tmp_path)
    rows, parsed, identity = _curriculum(root, "sft", 10000)
    assert identity == manifest["artifact_id"] and rows[0]["input"] == "Fixture train question"
    assert parsed["split_audit"] == manifest["split_audit"]
    (root / "sft.json").write_text("[]")
    with pytest.raises(ValidationError, match="changed"):
        _curriculum(root, "sft", 10000)


def test_fixture_curriculum_cannot_become_neural_fit_input(tmp_path):
    root, _ = curriculum(tmp_path, fixture=True)
    with pytest.raises(ValidationError, match="fixture-only"):
        _curriculum(root, "sft", 10000)


def test_empty_native_selection_is_not_replaced(tmp_path):
    root, manifest = curriculum(tmp_path)
    (root / "sft.json").write_text("[]")
    manifest.pop("artifact_id")
    manifest["outputs"]["sft.json"] = _sha(root / "sft.json")
    manifest["artifact_id"] = content_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="empty"):
        _curriculum(root, "sft", 10000)


@pytest.mark.parametrize("quantization", [{"quant_method": "fp8"}, {"quant_method": "mxfp4"}])
def test_quantized_policy_is_not_silently_dequantized_or_loaded(tmp_path, quantization):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"quantization_config": quantization}))
    (root / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ExternalDependencyError, match="quantized base"):
        _model(root, "absent-weight-manifest", "0" * 64)


def test_remote_code_rejected_before_model_weight_access(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "custom.Model"}}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ExternalDependencyError, match="remote code"):
        _model(tmp_path, "absent", "0" * 64)


def test_native_qwen_template_is_not_silently_applied_to_gemma(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ExternalDependencyError, match="Qwen template"):
        _model(tmp_path, "absent", "0" * 64)


def test_native_config_keeps_math_and_changes_execution_fields_only(tmp_path):
    native = {"stage": "dpo", "finetuning_type": "lora", "pref_beta": 0.1, "pref_loss": "sigmoid",
              "lora_rank": 8, "lora_target": "all", "learning_rate": 7e-6, "num_train_epochs": 3.0,
              "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
              "cutoff_len": 10000, "bf16": True, "template": "qwen"}
    # Exact YAML parsing is checked separately against the mounted native source.
    source = tmp_path / "source"
    path = source / "LLaMA-Factory/examples/train_lora/qwen_lora_dpo.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("explicit fixture YAML input")
    yaml = SimpleNamespace(safe_load=lambda text: dict(native))
    with patch.dict(sys.modules, {"yaml": yaml}):
        result = _native_config(source, "dpo", tmp_path / "model", tmp_path / "out", 17)
    assert all(result[key] == value for key, value in native.items())
    assert result["trust_remote_code"] is False and result["overwrite_output_dir"] is False
    assert result["seed"] == result["data_seed"] == 17


def test_native_observer_caps_before_forward_and_preserves_unknown_reference_costs():
    counters = {"native_logs": [], "nonfinite_native_metrics": 0}
    with patch.dict(sys.modules, {"transformers": SimpleNamespace(TrainerCallback=object)}):
        observer = _observer(counters, 3)
    with pytest.raises(ValidationError, match="step cap"):
        observer.on_train_begin(None, SimpleNamespace(max_steps=4), None)
    observer.on_log(None, None, None, logs={"loss": float("nan"), "learning_rate": 7e-6})
    assert counters["nonfinite_native_metrics"] == 1
    assert counters["native_logs"][0]["loss"] == "nan"
    json.dumps(counters, allow_nan=False)


def test_stage_requires_explicit_acknowledgement():
    with pytest.raises(ValidationError, match="acknowledgement"):
        SmartSearchNativePolicyTraining().fit(stage="sft", checkout="absent", model_root="absent",
                                             model_manifest="absent", model_manifest_sha256="0" * 64,
                                             curriculum_root="absent", output_root="absent")
    for key, value in (("seed", True), ("max_optimizer_steps", 0), ("max_data_bytes", False),
                       ("gpu_processes", 5), ("gpu_processes", False)):
        with pytest.raises(ValidationError):
            SmartSearchPolicyFitControls(**{key: value})


def test_training_import_cannot_shadow_an_existing_framework(tmp_path):
    with patch.dict(sys.modules, {"llamafactory": SimpleNamespace(__file__="fixture")}), pytest.raises(ExternalDependencyError, match="dedicated process"):
        _source_import(tmp_path)


def test_execution_environment_is_restored_after_failure(tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def scope(root, seed):
        yield

    monkeypatch.setenv("WANDB_MODE", "operator-original")
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    with patch("agentinputlab.recipes.smartsearch_policy._scope", scope), pytest.raises(RuntimeError), _environment(tmp_path, 17):
        import os
        assert os.environ["WANDB_MODE"] == "disabled"
        raise RuntimeError("fixture failure")
    import os
    assert os.environ["WANDB_MODE"] == "operator-original"
    assert "HF_DATASETS_OFFLINE" not in os.environ


def test_missing_rank_metering_is_unknown_not_zero(tmp_path):
    (tmp_path / "fit-plan.json").write_text("{}")
    counters = {"forward_attempts": None, "forward_responses": None, "optimizer_steps": None}
    _collect_rank_usage(tmp_path, 2, counters)
    assert not counters["rank_usage_complete"]
    assert counters["forward_attempts"] is None


def test_native_rank_receipt_plan_binding_and_step_agreement(tmp_path):
    plan = tmp_path / "fit-plan.json"
    plan.write_text("{}")
    usage = dict(forward_attempts=6, forward_responses=6, input_token_positions=100,
                 nonfinite_native_metrics=0, optimizer_steps=3)
    for rank in range(2):
        (tmp_path / f"rank-{rank}-terminal.json").write_text(json.dumps(
            dict(rank=rank, plan_sha256=_sha(plan), status="completed", usage=usage)))
    counters = {}
    _collect_rank_usage(tmp_path, 2, counters)
    assert counters["rank_usage_complete"]
    assert counters["forward_attempts"] == 12
    assert counters["optimizer_steps"] == 3  # distributed steps are not summed
    assert counters["distributed_optimizer_step_agreement"]
    changed = json.loads((tmp_path / "rank-1-terminal.json").read_text())
    changed["plan_sha256"] = "0" * 64
    (tmp_path / "rank-1-terminal.json").write_text(json.dumps(changed))
    counters = {"forward_attempts": None}
    _collect_rank_usage(tmp_path, 2, counters)
    assert not counters["rank_usage_complete"] and counters["forward_attempts"] is None
    assert counters["rank_parse_errors"]


def test_fixture_artifact_hash_binding_and_safe_snapshot_copy(tmp_path):
    root = tmp_path / "artifact"
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    # Explicit non-neural fixture: the protocol test does not load these bytes
    # as weights or certify the fake checkpoint as a real trained model.
    for name, body in {"model.safetensors": "fixture-not-real-weights", "config.json": "{}",
                       "tokenizer_config.json": "{}", "trainer_state.json": "{}",
                       "train_results.json": "{}", "all_results.json": "{}"}.items():
        (checkpoint / name).write_text(body)
    (root / "fit-plan.json").write_text("{}")
    (root / "fit-terminal.json").write_text(json.dumps({"status": "trained_candidate", "usage": {"optimizer_steps": 3}}))
    record = dict(protocol=POLICY_PROTOCOL, stage="sft", fixture_only=True,
                  fit_plan_sha256=_sha(root / "fit-plan.json"),
                  fit_terminal_sha256=_sha(root / "fit-terminal.json"),
                  checkpoint_files={p.name: _sha(p) for p in checkpoint.iterdir()})
    record["artifact_id"] = content_hash(record)
    (root / "artifact.json").write_text(json.dumps(record))
    artifact = SmartSearchPolicyStageArtifact.load(root)
    snapshot = artifact.materialize_snapshot(tmp_path / "export")
    assert _sha(root / "checkpoint/model.safetensors") == _sha(
        Path(snapshot["model_root"]) / "model.safetensors")
    assert not (Path(snapshot["model_root"]) / "trainer_state.json").exists()
    assert not (Path(snapshot["model_root"]) / "train_results.json").exists()
    exported = json.loads(Path(snapshot["model_manifest"]).read_text())
    assert set(exported["files"]) == {"model.safetensors", "config.json", "tokenizer_config.json"}
    assert exported["toolkit_exporter_sha256"]
    with pytest.raises(ValidationError, match="create-only"):
        artifact.materialize_snapshot(tmp_path / "export")
    (checkpoint / "model.safetensors").write_text("changed fixture")
    with pytest.raises(ValidationError, match="bytes changed"):
        SmartSearchPolicyStageArtifact.load(root)
