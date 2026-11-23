"""Offline schema fixtures only; no native training/real acceptance is implied."""

import json
from dataclasses import replace

import pytest

from agentinputlab.api import create_intervention
from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import LLMLingua2FitControls, LLMLingua2TrainedArtifact
from agentinputlab.recipes.llmlingua2_training import (
    _new_root,
    _sha,
    audit_llmlingua_training_data,
    validate_llmlingua_initial_checkpoint,
)


def records(tmp_path):
    path = tmp_path / "training.json"
    path.write_text(
        json.dumps(
            dict(origin=[f"original chunk {i}" for i in range(5)], labels=[[True, False, True]] * 5)
        )
    )
    return path


def checkpoint(tmp_path):
    root = tmp_path / "bert-base-multilingual-cased"
    root.mkdir()
    values = {
        "model.safetensors": "fixture-not-weights",
        "tokenizer.json": "{}",
        "config.json": json.dumps(dict(model_type="bert", id2label={"0": "drop", "1": "keep"})),
        "tokenizer_config.json": "{}",
    }
    for name, value in values.items():
        (root / name).write_text(value)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="agentinputlab.llmlingua2.token_checkpoint.v1",
                files={p.name: _sha(p) for p in root.iterdir()},
            )
        )
    )
    return root, manifest


def test_native_split_and_bounded_step_plan(tmp_path):
    p = records(tmp_path)
    before = p.read_bytes()
    data, audit = audit_llmlingua_training_data(p, LLMLingua2FitControls(epochs=1, batch_size=4))
    assert audit["planned_optimizer_steps"] == 1
    assert len(audit["native_train_indices"]) == 4 and len(audit["native_validation_indices"]) == 1
    assert set(audit["native_train_indices"]).isdisjoint(audit["native_validation_indices"])
    assert data["labels"][0] == [True, False, True] and p.read_bytes() == before
    with pytest.raises(ValidationError, match="step cap"):
        audit_llmlingua_training_data(
            p, LLMLingua2FitControls(epochs=2, batch_size=1, max_optimizer_steps=1)
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("epochs", 0),
        ("batch_size", False),
        ("seed", -1),
        ("max_optimizer_steps", 0),
        ("max_data_bytes", 0),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("learning_rate", True),
        ("device", "cuda:7"),
    ],
)
def test_invalid_controls(name, value):
    with pytest.raises(ValidationError):
        replace(LLMLingua2FitControls(), **{name: value})


@pytest.mark.parametrize(
    "mutation", ["short", "duplicate", "empty", "numeric_label", "no_label", "mismatch"]
)
def test_bad_native_data_rejected(tmp_path, mutation):
    p = records(tmp_path)
    d = json.loads(p.read_text())
    if mutation == "short":
        d["origin"].pop()
        d["labels"].pop()
    if mutation == "duplicate":
        d["origin"][1] = d["origin"][0].upper()
    if mutation == "empty":
        d["origin"][0] = " "
    if mutation == "numeric_label":
        d["labels"][0] = [1, 0]
    if mutation == "no_label":
        d["labels"][0] = []
    if mutation == "mismatch":
        d["labels"].pop()
    p.write_text(json.dumps(d))
    with pytest.raises(ValidationError):
        audit_llmlingua_training_data(p, LLMLingua2FitControls())


def test_data_byte_limit_and_symlink(tmp_path):
    p = records(tmp_path)
    with pytest.raises(ValidationError, match="byte cap"):
        audit_llmlingua_training_data(p, LLMLingua2FitControls(max_data_bytes=1))
    link = tmp_path / "link.json"
    link.symlink_to(p)
    with pytest.raises(ValidationError, match="non-symlink"):
        audit_llmlingua_training_data(link, LLMLingua2FitControls())


def test_checkpoint_closure_and_remote_code_rejected(tmp_path):
    root, manifest = checkpoint(tmp_path)
    assert validate_llmlingua_initial_checkpoint(root, manifest, _sha(manifest))[0] == root
    (root / "extra.py").write_text("pass")
    with pytest.raises(ValidationError, match="extra or missing"):
        validate_llmlingua_initial_checkpoint(root, manifest, _sha(manifest))
    (root / "extra.py").unlink()
    (root / "config.json").write_text(
        json.dumps(dict(model_type="bert", id2label={"0": "d", "1": "k"}, auto_map={"x": "y"}))
    )
    d = json.loads(manifest.read_text())
    d["files"]["config.json"] = _sha(root / "config.json")
    manifest.write_text(json.dumps(d))
    with pytest.raises(ValidationError, match="standard"):
        validate_llmlingua_initial_checkpoint(root, manifest, _sha(manifest))


def test_native_default_label_mapping_can_be_omitted_but_not_changed(tmp_path):
    root, manifest = checkpoint(tmp_path)
    config = {"model_type": "bert", "architectures": ["BertForTokenClassification"]}
    for label_count in (2, 3):
        config["num_labels"] = label_count
        (root / "config.json").write_text(json.dumps(config))
        meta = json.loads(manifest.read_text())
        meta["files"]["config.json"] = _sha(root / "config.json")
        manifest.write_text(json.dumps(meta))
        if label_count == 2:
            assert validate_llmlingua_initial_checkpoint(root, manifest, _sha(manifest))[0] == root
        else:
            with pytest.raises(ValidationError, match="standard"):
                validate_llmlingua_initial_checkpoint(root, manifest, _sha(manifest))


def test_root_and_untrained_artifact_are_not_reused(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ValidationError, match="create-only"):
        _new_root(root)
    with pytest.raises(ValidationError, match="mutate"):
        _new_root(root / "child", root)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValidationError, match="directory"):
        LLMLingua2TrainedArtifact.load(link)


def test_public_factory_does_not_run_training_without_acknowledgement():
    method = create_intervention(
        "training.llmlingua2.distillation", controls=dict(epochs=1, batch_size=4)
    )
    assert method.controls.batch_size == 4
    with pytest.raises(ValidationError, match="acknowledgement"):
        method.fit(
            checkout="missing",
            initial_checkpoint="missing",
            initial_checkpoint_manifest="missing",
            initial_manifest_sha256="0" * 64,
            training_file="missing",
            data_provenance="missing",
            data_provenance_sha256="0" * 64,
            nlp_root="missing",
            nlp_manifest="missing",
            nlp_manifest_sha256="0" * 64,
            output_root="missing",
        )
