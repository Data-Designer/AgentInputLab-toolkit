"""Synthetic feature API contracts, not native embedding or benchmark evidence."""

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import R2AGNativeFeatureEncoder, R2AGRetrievalDocument
from agentinputlab.recipes import r2ag_features as module


def documents(count=10):
    return [R2AGRetrievalDocument(str(i), f"Title {i}", f"Document {i}") for i in range(count)]


@pytest.mark.parametrize(
    "field,value", [("document_id", ""), ("text", " "), ("title", "bad\x00title")]
)
def test_visible_document_text_is_strict(field, value):
    with pytest.raises(ValidationError):
        replace(documents()[0], **{field: value})


@pytest.mark.parametrize("count", [0, 9, 101])
def test_candidate_bounds_checked_without_neural_runtime(count):
    with pytest.raises(ValidationError, match="10-100"):
        module._validate_inputs("q", documents(count))


def test_duplicate_ids_and_implicit_large_candidates_rejected():
    rows = documents()
    with pytest.raises(ValidationError, match="unique"):
        module._validate_inputs("q", [*rows[:-1], rows[0]])
    with pytest.raises(ValidationError, match="4 MiB"):
        module._validate_inputs("q", [replace(d, text="x" * (512 * 1024)) for d in rows])


def test_native_input_contains_no_answer_or_support_labels():
    row = module._native_documents("visible question", documents())
    assert set(row) == {"question", "context", "supporting_facts"}
    assert row["supporting_facts"] == [] and row["question"] == "visible question"
    assert row["context"][0] == ["Title 0", ["Document 0"], {"document_id": "0"}]


def fixture(tmp_path, monkeypatch, *, fail=False, corrupt=False):
    source, encoder_root = tmp_path / "source", tmp_path / "encoder"
    source.mkdir()
    encoder_root.mkdir()
    encoder = SimpleNamespace(forward=lambda features: object())

    def extract(rows, model):
        assert len(rows) == 1 and rows[0]["supporting_facts"] == []
        assert "answer" not in rows[0]
        model.forward(
            {"attention_mask": SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 29))}
        )
        if fail:
            raise RuntimeError("native synthetic failure")
        row = rows[0]
        ranked = [[*c, (1 - i * 0.01, 0.1, 0.2)] for i, c in enumerate(reversed(row["context"]))]
        if corrupt:
            ranked[0][1] = ["changed"]
        return [dict(row, context=ranked)]

    runtime = R2AGNativeFeatureEncoder.__new__(R2AGNativeFeatureEncoder)
    runtime.source, runtime.encoder = source, encoder_root
    runtime._verify = lambda: None
    runtime.fingerprint = lambda: {"synthetic_fixture": True}
    runtime._load = lambda: (
        SimpleNamespace(no_grad=nullcontext),
        SimpleNamespace(get_dual_sim_hotpotqa=extract),
        encoder,
    )
    monkeypatch.setattr(module, "_scope", lambda *a: nullcontext())
    return runtime


def test_single_native_feature_attempt_retains_rank_identity_and_cost(tmp_path, monkeypatch):
    runtime = fixture(tmp_path, monkeypatch)
    output = tmp_path / "attempt"
    result = runtime.encode(query="q", documents=documents(12), output_root=output)
    assert len(result) == 12 and result[0].document_id == "11"
    assert result[0].text == "Document 11" and result[0].features == (1.0, 0.1, 0.2)
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "features_prepared" and terminal["reader_documents"] == 10
    assert terminal["embedding_forward_attempts"] == terminal["embedding_forward_responses"] == 1
    assert terminal["input_tokens"] == 29 and terminal["input_token_unknown_attempts"] == 0
    assert terminal["new_generator_calls"] == terminal["new_optimizer_steps"] == 0
    claim = json.loads((output / "claim.json").read_text())
    assert not claim["support_labels_supplied"] and not claim["reference_answers_supplied"]
    with pytest.raises(ValidationError, match="create-only"):
        runtime.encode(query="q", documents=documents(12), output_root=output)


@pytest.mark.parametrize("flag", ["fail", "corrupt"])
def test_failure_is_saved_and_does_not_become_success(tmp_path, monkeypatch, flag):
    runtime = fixture(tmp_path, monkeypatch, **{flag: True})
    output = tmp_path / "attempt"
    with pytest.raises((RuntimeError, ValidationError)):
        runtime.encode(query="q", documents=documents(), output_root=output)
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "failed_terminal" and not terminal["automatic_retry"]
    assert terminal["embedding_forward_attempts"] == terminal["embedding_forward_responses"] == 1
    assert not (output / "features.json").exists()


def test_acknowledgement_before_asset_or_optional_import(tmp_path):
    with pytest.raises(ValidationError, match="acknowledgement"):
        R2AGNativeFeatureEncoder(
            checkout=tmp_path,
            encoder_root=tmp_path,
            encoder_manifest=tmp_path / "missing",
            encoder_manifest_sha256="0" * 64,
        )
