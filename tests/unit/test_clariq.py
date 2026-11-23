from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks.clariq import ClariQDevBundle, ClariQPrediction, _topic_queries
from agentinputlab.contracts.errors import ArtifactError, ValidationError


@pytest.mark.parametrize("need", [0, 5, True, "2", 2.0])
def test_need_labels_are_strict(need):
    with pytest.raises(ValidationError):
        ClariQPrediction(need, ("Q00001",))


def test_rankings_reject_empty_and_duplicate_ids():
    for ids in ((), ("",), ("Q00001", "Q00001")):
        with pytest.raises(ValidationError):
            ClariQPrediction(2, ids)


def test_local_source_terms_gate_precedes_file_access(tmp_path):
    with pytest.raises(ArtifactError, match="local-use terms"):
        ClariQDevBundle(tmp_path)


def test_wrong_checksum_is_rejected(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/dev.tsv").write_text("not the official source")
    with pytest.raises(ArtifactError, match="checksum"):
        ClariQDevBundle(tmp_path, accept_license_review=True)


def test_per_topic_aggregation_rejects_conflicting_labels():
    row = {"topic_id": "101", "initial_request": "user query", "clarification_need": "2"}
    assert _topic_queries([row, row]) == {"101": "user query"}
    with pytest.raises(ArtifactError, match="conflicting"):
        _topic_queries([row, {**row, "clarification_need": "3"}])


def test_visible_examples_do_not_contain_gold_or_answers():
    bundle = object.__new__(ClariQDevBundle)
    bundle._topics = {"101": "visible user request"}
    bundle._rows = [{"facet_desc": "hidden intent", "answer": "hidden answer"}]
    example = bundle.examples()[0]
    visible = json.dumps(example.visible_state().canonical_dict())
    assert "visible user request" in visible
    assert "hidden intent" not in visible and "hidden answer" not in visible
    assert not example.evaluator_gold
    with pytest.raises(ValidationError):
        bundle.examples("test")


def test_question_ranking_uses_bank_not_reference_labels():
    bundle = object.__new__(ClariQDevBundle)
    bundle._bank = {"Q00001": "", "Q00002": "apple", "Q00003": "banana"}
    bundle._retriever = InMemoryBM25Retriever(documents=tuple(
        Document(doc_id=key, text=text) for key, text in bundle._bank.items()
    ))
    assert bundle.rank_questions("apple", top_k=2) == ("Q00002", "Q00001")
    for query, k in (("", 1), ("x", 0), ("x", True), ("x", 4)):
        with pytest.raises(ValidationError):
            bundle.rank_questions(query, top_k=k)


@pytest.fixture
def official_bundle():
    source = os.getenv("AIL_CLARIQ_SOURCE_ROOT")
    if not source:
        pytest.skip("requires three caller-mounted pinned ClariQ files; no test downloads")
    return ClariQDevBundle(Path(source), accept_license_review=True)


def test_native_official_perfect_labels_and_ranked_questions(official_bundle, tmp_path):
    topics = [example.example_id for example in official_bundle.examples()][:2]
    predictions = {}
    for topic in topics:
        rows = [row for row in official_bundle._rows if row["topic_id"] == topic]
        predictions[topic] = ClariQPrediction(
            int(rows[0]["clarification_need"]),
            tuple(sorted({row["question_id"] for row in rows})),
        )
    result = official_bundle.evaluate(predictions, topic_ids=topics, output=tmp_path / "native")
    assert result["metrics"]["weighted_f1"] == 1.0
    assert result["predicted_topics"] == 2
    assert result["evaluation_scope"] == "fixed_dev_compatibility_subset"
    assert not result["document_retrieval_utility_evaluated"]


def test_native_official_missing_predictions_are_not_dropped(official_bundle, tmp_path):
    topics = [example.example_id for example in official_bundle.examples()][:2]
    result = official_bundle.evaluate({}, topic_ids=topics, output=tmp_path / "missing")
    assert result["missing_predictions"] == topics
    assert set(result["metrics"].values()) == {0.0}


def test_native_official_foreign_topics_and_questions_fail_before_output(official_bundle, tmp_path):
    topic = official_bundle.examples()[0].example_id
    bad = ClariQPrediction(1, ("not-a-bank-id",))
    with pytest.raises(ValidationError, match="outside the source bank"):
        official_bundle.evaluate({topic: bad}, topic_ids=[topic], output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValidationError):
        official_bundle.evaluate({}, topic_ids=[topic, topic], output=tmp_path / "bad")
