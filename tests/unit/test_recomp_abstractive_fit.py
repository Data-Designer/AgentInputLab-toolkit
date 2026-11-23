"""Synthetic schema/plan fixtures only; no claim of simulated model training."""

import hashlib
import json
from dataclasses import replace

import pytest

from agentinputlab.api import create_intervention
from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import RECOMPAbstractiveFitControls, RECOMPTrainedSeq2SeqArtifact
from agentinputlab.recipes.recomp_abstractive import (
    _NLTK_REQUIRED,
    audit_recomp_seq2seq_data,
    validate_recomp_nltk_assets,
)


def files(tmp_path, *, blank=False, duplicate=False, array=False):
    train = [
        dict(question="train", passages="original passage", summary=" " if blank else "target")
    ]
    if duplicate:
        train.append(dict(train[0]))
    dev = [dict(question="dev", passages="dev passage", summary="dev target")]
    paths = []
    for name, rows in (("train", train), ("dev", dev)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rows) if array else "\n".join(json.dumps(row) for row in rows))
        paths.append(path)
    return paths


@pytest.mark.parametrize("array", [False, True])
def test_native_formats_blank_targets_and_duplicate_queries_are_not_filtered(tmp_path, array):
    train, dev = files(tmp_path, blank=True, duplicate=True, array=array)
    before = train.read_bytes()
    audit = audit_recomp_seq2seq_data(train, dev, RECOMPAbstractiveFitControls(max_steps=1))
    assert audit["files"]["train"]["rows"] == 2
    assert audit["files"]["train"]["blank_summary_rows"] == 2
    assert audit["planned_optimizer_steps"] == 1
    assert train.read_bytes() == before


def test_epoch_and_explicit_step_plans_cannot_silently_exceed_cap(tmp_path):
    train, dev = files(tmp_path)
    c = RECOMPAbstractiveFitControls(epochs=2, batch_size=1, max_optimizer_steps=1)
    with pytest.raises(ValidationError, match="step cap"):
        audit_recomp_seq2seq_data(train, dev, c)
    with pytest.raises(ValidationError, match="step cap"):
        audit_recomp_seq2seq_data(train, dev, replace(c, max_steps=3))
    assert (
        audit_recomp_seq2seq_data(train, dev, replace(c, max_steps=1))["planned_optimizer_steps"]
        == 1
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("epochs", 0),
        ("epochs", float("nan")),
        ("epochs", True),
        ("batch_size", False),
        ("gradient_accumulation_steps", 0),
        ("max_target_length", 513),
        ("device", "cuda:1"),
        ("seed", -1),
        ("learning_rate", float("inf")),
        ("max_steps", 0),
        ("max_steps", True),
    ],
)
def test_invalid_controls(name, value):
    with pytest.raises(ValidationError):
        replace(RECOMPAbstractiveFitControls(), **{name: value})


@pytest.mark.parametrize(
    "bad",
    [
        None,
        3,
        {},
        dict(question="", passages="p", summary="s"),
        dict(question="q", passages="p", summary=None),
    ],
)
def test_malformed_rows_fail_before_training(tmp_path, bad):
    train, dev = files(tmp_path)
    train.write_text(json.dumps([bad]))
    with pytest.raises(ValidationError):
        audit_recomp_seq2seq_data(train, dev, RECOMPAbstractiveFitControls())


def test_split_overlap_and_symlink_and_byte_cap(tmp_path):
    train, dev = files(tmp_path)
    with pytest.raises(ValidationError, match="differ"):
        audit_recomp_seq2seq_data(train, train, RECOMPAbstractiveFitControls())
    dev.write_text(json.dumps(dict(question="  TRAIN ", passages="p", summary="s")))
    with pytest.raises(ValidationError, match="overlap"):
        audit_recomp_seq2seq_data(train, dev, RECOMPAbstractiveFitControls())
    with pytest.raises(ValidationError, match="byte cap"):
        audit_recomp_seq2seq_data(train, dev, RECOMPAbstractiveFitControls(max_data_bytes=1))
    link = tmp_path / "link.json"
    link.symlink_to(train)
    with pytest.raises(ValidationError, match="non-symlink"):
        audit_recomp_seq2seq_data(link, dev, RECOMPAbstractiveFitControls())


def resource_fixture(tmp_path):
    folder = tmp_path / "nltk"
    hashes = {}
    for relative in _NLTK_REQUIRED:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture-not-a-real-NLTK-resource")
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = tmp_path / "nltk.json"
    manifest.write_text(
        json.dumps(dict(schema="agentinputlab.recomp.seq2seq.nltk.v1", files=hashes))
    )
    return folder, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_nltk_requires_pinned_resources_not_an_empty_import_bypass(tmp_path):
    folder, manifest, sha = resource_fixture(tmp_path)
    assert validate_recomp_nltk_assets(folder, manifest, sha) == folder
    (folder / next(iter(_NLTK_REQUIRED))).write_text("changed")
    with pytest.raises(ValidationError, match="hash"):
        validate_recomp_nltk_assets(folder, manifest, sha)


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "tokenizers/../alias", "./alias"])
def test_nltk_manifest_path_escape_rejected(tmp_path, relative):
    folder, manifest, _ = resource_fixture(tmp_path)
    metadata = json.loads(manifest.read_text())
    metadata["files"][relative] = "0" * 64
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError, match="canonical"):
        validate_recomp_nltk_assets(
            folder, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()
        )


def test_factory_requires_ack_and_raw_checkpoint_is_not_verified(tmp_path):
    recipe = create_intervention("training.recomp.abstractive", controls=dict(max_steps=1))
    assert recipe.controls.max_steps == 1
    with pytest.raises(ValidationError, match="acknowledgement"):
        recipe.fit(
            checkout="missing",
            initial_checkpoint="missing",
            initial_checkpoint_manifest="missing",
            train_file="missing",
            dev_file="missing",
            nltk_root="missing",
            nltk_manifest="missing",
            nltk_manifest_sha256="0" * 64,
            output_root="missing",
        )
    with pytest.raises(ValidationError):
        RECOMPTrainedSeq2SeqArtifact.load(tmp_path)
