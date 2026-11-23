"""Offline training-plan and artifact boundaries; no simulated training success."""

import json
from dataclasses import replace

import pytest

from agentinputlab.api import create_intervention
from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import RECOMPExtractiveFitControls, RECOMPTrainedEncoderArtifact
from agentinputlab.recipes.recomp_extractive import audit_recomp_training_data


def row(query="train"):
    return {
        "question": query,
        "retrieved_docs": [
            {"text": "positive", "em": 1},
            {"text": "negative", "em": 0},
            {"text": "tie", "em": 1},
        ],
        "dpr_instance": {"query": query, "positive_ctxs": [0], "negative_ctxs": [1, 2]},
    }


def files(tmp_path, train=None, dev=None):
    a, b = tmp_path / "train.json", tmp_path / "dev.json"
    a.write_text(json.dumps([row()] if train is None else train))
    b.write_text(json.dumps([row("dev")] if dev is None else dev))
    return a, b


def test_uses_exact_native_EM_pair_predicate_and_bounded_steps(tmp_path):
    a, b = files(tmp_path)
    controls = RECOMPExtractiveFitControls(epochs=2, batch_size=64)
    result = audit_recomp_training_data(a, b, controls)
    assert result["files"]["train"]["native_triplets"] == 1  # tie is not a negative
    assert result["planned_optimizer_steps"] == 2
    assert result["dev_loaded_by_native_script_but_not_used_for_optimization_or_validation"]


def test_bounded_cap_does_not_truncate_input_or_change_epochs(tmp_path):
    a, b = files(tmp_path)
    before = a.read_bytes()
    with pytest.raises(ValidationError, match="optimizer plan"):
        audit_recomp_training_data(
            a, b, RECOMPExtractiveFitControls(epochs=2, max_training_steps=1)
        )
    with pytest.raises(ValidationError, match="byte cap"):
        audit_recomp_training_data(a, b, RECOMPExtractiveFitControls(max_data_bytes=1))
    assert a.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("batch_size", True),
        ("warmup_steps", -1),
        ("learning_rate", float("nan")),
        ("learning_rate", True),
        ("max_training_steps", 0),
        ("seed", -1),
    ],
)
def test_invalid_controls(field, value):
    with pytest.raises(ValidationError):
        replace(RECOMPExtractiveFitControls(), **{field: value})


def test_train_dev_same_file_or_normalized_query_overlap_rejected(tmp_path):
    a, b = files(tmp_path, dev=[row(" TRAIN  ")])
    with pytest.raises(ValidationError, match="overlap"):
        audit_recomp_training_data(a, b, RECOMPExtractiveFitControls())
    with pytest.raises(ValidationError, match="differ"):
        audit_recomp_training_data(a, a, RECOMPExtractiveFitControls())


@pytest.mark.parametrize("change", ["index", "text", "score", "query", "triplets", "schema"])
def test_bad_native_records_fail_before_training(tmp_path, change):
    v = row()
    if change == "index":
        v["dpr_instance"]["negative_ctxs"] = [-1]
    elif change == "text":
        v["retrieved_docs"][0]["text"] = None
    elif change == "score":
        v["retrieved_docs"][0]["em"] = "1"
    elif change == "query":
        v["dpr_instance"]["query"] = " "
    elif change == "triplets":
        v["dpr_instance"]["positive_ctxs"] = []
    else:
        v.pop("dpr_instance")
    a, b = files(tmp_path, train=[v])
    with pytest.raises(ValidationError):
        audit_recomp_training_data(a, b, RECOMPExtractiveFitControls())


def test_symlink_data_rejected(tmp_path):
    a, b = files(tmp_path)
    link = tmp_path / "linked.json"
    link.symlink_to(a)
    with pytest.raises(ValidationError, match="non-symlink"):
        audit_recomp_training_data(link, b, RECOMPExtractiveFitControls())


def test_original_blank_document_sentinel_is_preserved_in_native_training(tmp_path):
    original = row()
    original["retrieved_docs"][0]["text"] = "   "
    a, b = files(tmp_path, train=[original])
    before = a.read_bytes()
    audit = audit_recomp_training_data(a, b, RECOMPExtractiveFitControls())
    assert audit["files"]["train"]["native_triplets"] == 1
    assert a.read_bytes() == before


def test_registered_factory_accepts_data_only_controls_and_ack_required_before_source():
    recipe = create_intervention(
        "training.recomp.extractive", controls={"epochs": 1, "warmup_steps": 0}
    )
    assert recipe.controls.epochs == 1 and recipe.controls.warmup_steps == 0
    with pytest.raises(ValidationError, match="acknowledgement"):
        recipe.fit(
            checkout="absent",
            initial_checkpoint="absent",
            initial_checkpoint_manifest="absent",
            train_file="absent",
            dev_file="absent",
            output_root="absent",
        )


def test_candidate_loader_does_not_treat_a_raw_native_checkpoint_as_verified(tmp_path):
    with pytest.raises(ValidationError):
        RECOMPTrainedEncoderArtifact.load(tmp_path)
