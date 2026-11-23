"""Synthetic schema/provenance fixtures; these are not real model acceptance."""

import hashlib
import json
from dataclasses import replace

import pytest

from agentinputlab.api import create_intervention
from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import AdaptiveRAGFitControls, AdaptiveRAGTrainedRouterArtifact
from agentinputlab.recipes.adaptive_rag import (
    _align_t5_initial_keys,
    _new_output_root,
    audit_adaptive_training_data,
    rebuild_adaptive_silver,
    validate_adaptive_router_checkpoint,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_native_t5_tied_aliases_can_expand_deduplicated_safe_keys():
    initial = {"shared.weight": object(), "encoder.block.weight": object()}
    expected = set(initial) | {
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    }
    expanded, aliases = _align_t5_initial_keys(initial, expected, tie_word_embeddings=True)
    assert len(aliases) == 3 and set(expanded) == expected
    assert all(expanded[k] is initial["shared.weight"] for k in aliases)
    with pytest.raises(ValidationError, match="namespace"):
        _align_t5_initial_keys(initial, expected, tie_word_embeddings=False)
    with pytest.raises(ValidationError, match="namespace"):
        _align_t5_initial_keys(initial, expected | {"invented.weight"}, tie_word_embeddings=True)
    with pytest.raises(ValidationError, match="namespace"):
        _align_t5_initial_keys(initial, {"shared.weight"}, tie_word_embeddings=True)


def test_output_does_not_mutate_source_assets_or_reuse_existing_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValidationError, match="create-only"):
        _new_output_root(source)
    with pytest.raises(ValidationError, match="mutate"):
        _new_output_root(source / "nested", source)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValidationError, match="mutate"):
        _new_output_root(alias / "nested", source)
    assert not (source / "nested").exists()
    created = _new_output_root(tmp_path / "new", source)
    assert created.is_dir()


def records(tmp_path):
    paths = []
    for split in ("train", "dev"):
        path = tmp_path / f"{split}.json"
        path.write_text(
            json.dumps(
                [
                    dict(
                        id=f"{split}{i}",
                        question=f"{split} question {i}",
                        answer=label,
                        dataset_name="fixture",
                    )
                    for i, label in enumerate(("A", "B", "C"))
                ]
            )
        )
        paths.append(path)
    return paths


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    for name, value in {
        "config.json": '{"model_type":"t5"}',
        "tokenizer.json": "{}",
        "model.safetensors": "fixture-not-weights",
    }.items():
        (root / name).write_text(value)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="agentinputlab.adaptive-rag.router_checkpoint.v1",
                files={p.name: sha(p) for p in root.iterdir()},
            )
        )
    )
    return root, manifest


def test_native_records_and_nontruncated_step_plan(tmp_path):
    train, dev = records(tmp_path)
    before = train.read_bytes()
    controls = AdaptiveRAGFitControls(epochs=2, batch_size=2, gradient_accumulation_steps=2)
    result = audit_adaptive_training_data(train, dev, controls)
    assert result["minimum_planned_optimizer_steps"] == 2
    assert result["files"]["train"]["class_counts"] == {"A": 1, "B": 1, "C": 1}
    assert train.read_bytes() == before
    with pytest.raises(ValidationError, match="step cap"):
        audit_adaptive_training_data(train, dev, replace(controls, max_optimizer_steps=1))
    assert (
        audit_adaptive_training_data(train, dev, replace(controls, max_steps=1))[
            "minimum_planned_optimizer_steps"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("epochs", 0),
        ("epochs", 1.5),
        ("batch_size", False),
        ("seed", -1),
        ("max_steps", 0),
        ("max_steps", True),
        ("max_answer_length", 513),
        ("device", "cuda:1"),
        ("doc_stride", 384),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("weight_decay", -1),
    ],
)
def test_invalid_controls(name, value):
    with pytest.raises(ValidationError):
        replace(AdaptiveRAGFitControls(), **{name: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("question", None),
        ("answer", "D"),
        ("answer", 0),
        ("dataset_name", ""),
        ("question", "q\0"),
    ],
)
def test_reject_malformed_native_records(tmp_path, field, value):
    train, dev = records(tmp_path)
    rows = json.loads(train.read_text())
    rows[0][field] = value
    train.write_text(json.dumps(rows))
    with pytest.raises(ValidationError):
        audit_adaptive_training_data(train, dev, AdaptiveRAGFitControls())


def test_split_ids_and_questions_are_both_disjoint(tmp_path):
    train, dev = records(tmp_path)
    rows = json.loads(dev.read_text())
    rows[0]["question"] = "  TRAIN   question 0 "
    dev.write_text(json.dumps(rows))
    with pytest.raises(ValidationError, match="overlap"):
        audit_adaptive_training_data(train, dev, AdaptiveRAGFitControls())
    rows[0]["question"] = "dev query"
    rows[0]["id"] = "train0"
    dev.write_text(json.dumps(rows))
    with pytest.raises(ValidationError, match="overlap"):
        audit_adaptive_training_data(train, dev, AdaptiveRAGFitControls())


def test_duplicate_ids_and_same_file_byte_cap_and_symlink_rejected(tmp_path):
    train, dev = records(tmp_path)
    with pytest.raises(ValidationError, match="differ"):
        audit_adaptive_training_data(train, train, AdaptiveRAGFitControls())
    with pytest.raises(ValidationError, match="byte cap"):
        audit_adaptive_training_data(train, dev, AdaptiveRAGFitControls(max_data_bytes=1))
    link = tmp_path / "linked.json"
    link.symlink_to(train)
    with pytest.raises(ValidationError, match="non-symlink"):
        audit_adaptive_training_data(link, dev, AdaptiveRAGFitControls())
    rows = json.loads(train.read_text())
    rows.append(dict(rows[0]))
    train.write_text(json.dumps(rows))
    with pytest.raises(ValidationError, match="unique"):
        audit_adaptive_training_data(train, dev, AdaptiveRAGFitControls())


def test_safe_checkpoint_is_closed_and_hash_pinned(tmp_path):
    root, manifest = checkpoint(tmp_path)
    assert validate_adaptive_router_checkpoint(root, manifest, sha(manifest))[0] == root
    (root / "model.safetensors").write_text("changed")
    with pytest.raises(ValidationError, match="hash"):
        validate_adaptive_router_checkpoint(root, manifest, sha(manifest))


@pytest.mark.parametrize("name", ["../outside", "model.bin", "custom.py"])
def test_unsafe_checkpoint_entries_are_not_loaded(tmp_path, name):
    root, manifest = checkpoint(tmp_path)
    metadata = json.loads(manifest.read_text())
    metadata["files"][name] = "0" * 64
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError):
        validate_adaptive_router_checkpoint(root, manifest, sha(manifest))


def test_factory_requires_ack_and_raw_checkpoint_is_not_trained_artifact(tmp_path):
    recipe = create_intervention("acquire.adaptive_rag.rebuilt_offline", controls=dict(max_steps=1))
    assert recipe.controls.max_steps == 1
    with pytest.raises(ValidationError, match="acknowledgement"):
        recipe.fit(
            checkout="missing",
            initial_checkpoint="missing",
            initial_checkpoint_manifest="missing",
            initial_manifest_sha256="0" * 64,
            train_file="missing",
            dev_file="missing",
            data_provenance="missing",
            data_provenance_sha256="0" * 64,
            nltk_root="missing",
            nltk_manifest="missing",
            nltk_manifest_sha256="0" * 64,
            output_root="missing",
        )
    with pytest.raises(ValidationError):
        AdaptiveRAGTrainedRouterArtifact.load(tmp_path)
    with pytest.raises(ValidationError, match="acknowledgement"):
        rebuild_adaptive_silver(
            checkout="missing",
            original_file="missing",
            zero_file="missing",
            single_file="missing",
            multi_file="missing",
            dataset_name="fixture",
            teacher_revision="fixture-revision",
            output_root="missing",
        )
