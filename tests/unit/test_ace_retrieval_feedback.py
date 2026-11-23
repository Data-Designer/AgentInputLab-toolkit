from __future__ import annotations

import json

import pytest

from agentinputlab.backends import Document
from agentinputlab.backends.retrieval import RetrievalResult
from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.retrieval import RetrievalPrediction
from agentinputlab.contracts import ContextBlock, Message, ValidationError
from agentinputlab.recipes import ACEOnlineRetrievalFeedback, ACERetrievalFeedback


class Adapter:
    benchmark_id = "fixture.native_ir"
    pairing_mode = "stateless_score_twice"

    def __init__(self):
        self.cases = (
            BenchmarkExample(
                "train-id",
                "find alpha",
                instruction="Preserve negation.",
                visible_context=(ContextBlock("source context", "visible-doc"),),
                visible_messages=(Message("user", "prior evidence"),),
                evaluator_gold={"secret-qrel-doc": 1},
            ),
            BenchmarkExample("validation-id", "find beta", evaluator_gold={"another-secret": 1}),
        )
        self.calls = []
        self.revision = "native-fixture-v1"

    def examples(self, split):
        return self.cases

    def fingerprint(self):
        return {"benchmark": self.benchmark_id, "revision": self.revision}

    def score(self, example, prediction):
        assert any(example is e for e in self.cases)
        assert isinstance(prediction, RetrievalPrediction)
        self.calls.append((example.example_id, list(prediction.scores)))
        return {
            "ndcg_at_10": (0.6 if example.example_id == "train-id" else 0.2)
            if prediction.scores
            else 0.0
        }


class Retriever:
    retriever_id = "fixture.rank_preserving"

    def __init__(self):
        self.calls = []
        self.revision = "corpus-v1"

    def fingerprint(self):
        return {"corpus": self.revision}

    def retrieve(self, query, *, top_k):
        self.calls.append((query, top_k))
        return (
            RetrievalResult(Document("rank-first", "alpha"), 0.0, 1),
            RetrievalResult(Document("rank-second", "beta"), 10.0, 2),
        )


def make(**kwargs):
    a, r = Adapter(), Retriever()
    return (
        ACERetrievalFeedback(
            adapter=a, retriever=r, train=a.cases[:1], validation=a.cases[1:], **kwargs
        ),
        a,
        r,
    )


def test_inputs_keep_visible_task_context_and_never_forward_qrels():
    p, a, r = make()
    train = p.fit_examples("train")[0]
    text = json.dumps(train.as_dict()["input"])
    assert all(
        s in text for s in ("find alpha", "Preserve negation", "source context", "prior evidence")
    )
    assert "secret-qrel-doc" not in text and "another-secret" not in text
    assert train.expected_output == "train-id" and train.split == "train"
    p.validate_examples([train.as_dict()], [p.fit_examples("validation")[0].as_dict()])
    assert len(a.calls) == 2 and not r.calls  # native empty-ranking preflight only


def test_native_rank_order_metric_and_binary_vs_continuous_feedback():
    p, a, r = make()
    assert p.answer_is_correct('{"query":" alpha "}', "train-id") is True
    assert p.evaluate_accuracy(
        ['{"query":" alpha "}', '{"query":"beta"}'], ["train-id", "validation-id"]
    ) == pytest.approx(0.4)
    assert r.calls == [(" alpha ", 15), ("beta", 15)]
    assert a.calls[-1][1] == ["rank-first", "rank-second"]
    assert len(p.events) == 2 and p.events[0]["score"] == 0.6


@pytest.mark.parametrize(
    "text",
    [
        "alpha",
        '{"answer":"alpha"}',
        '{"query":""}',
        '{"query":"alpha","query":"beta"}',
        '```json\n{"query":"alpha"}\n```',
    ],
)
def test_invalid_training_query_gets_zero_without_retrieval_or_model_retry(text):
    p, _, r = make()
    assert p.answer_is_correct(text, "train-id") is False
    assert not r.calls and p.events[-1]["status"] == "invalid_query_training_feedback"


def test_fence_option_is_explicit_and_fingerprinted():
    p, _, r = make(allow_json_fence=True)
    assert p.score_query('```json\n{"query":"  alpha  "}\n```', "train-id") == 0.6
    assert r.calls == [("  alpha  ", 15)] and p.fingerprint()["allow_json_fence"] is True


def test_unknown_or_heldout_target_and_modified_fit_input_are_rejected():
    p, _, r = make()
    with pytest.raises(ValidationError, match="outside"):
        p.score_query('{"query":"alpha"}', "heldout-id")
    example = p.fit_examples("train")[0].as_dict()
    example["input"]["question"] = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        p.validate_examples([example], [p.fit_examples("validation")[0].as_dict()])
    with pytest.raises(ValidationError):
        p.fit_examples("test")
    assert not r.calls


@pytest.mark.parametrize("which", ["dataset", "retriever"])
def test_fingerprint_drift_is_not_silently_zeroed(which):
    p, a, r = make()
    (a if which == "dataset" else r).revision = "changed"
    with pytest.raises(ValidationError, match="changed"):
        p.score_query('{"query":"alpha"}', "train-id")


def test_retrieval_or_scorer_failure_is_not_silently_zeroed():
    p, a, r = make()

    def broken(*args, **kwargs):
        raise OSError("corpus unavailable")

    r.retrieve = broken
    with pytest.raises(OSError, match="corpus unavailable"):
        p.score_query('{"query":"alpha"}', "train-id")
    p, a, _ = make()
    a.score = broken
    with pytest.raises(OSError):
        p.score_query('{"query":"alpha"}', "train-id")


def test_invalid_native_ranks_and_missing_metric_are_errors():
    p, _, r = make()
    r.retrieve = lambda *a, **k: (
        RetrievalResult(Document("duplicate", "x"), 1.0, 1),
        RetrievalResult(Document("duplicate", "x"), 0.0, 2),
    )
    with pytest.raises(ValidationError, match="duplicate"):
        p.score_query('{"query":"alpha"}', "train-id")
    with pytest.raises(ValidationError, match="metric"):
        make(metric="not-a-native-metric")


def test_split_ids_and_actual_inputs_must_be_disjoint():
    a, r = Adapter(), Retriever()
    with pytest.raises(ValidationError, match="IDs"):
        ACERetrievalFeedback(adapter=a, retriever=r, train=a.cases[:1], validation=a.cases[:1])
    a.cases = (BenchmarkExample("one", "same query"), BenchmarkExample("two", "same query"))
    with pytest.raises(ValidationError, match="inputs"):
        ACERetrievalFeedback(adapter=a, retriever=r, train=a.cases[:1], validation=a.cases[1:])


def test_mean_contract_and_public_export():
    p, _, _ = make()
    assert p.evaluate_accuracy([], []) == 0.0
    with pytest.raises(ValidationError):
        p.evaluate_accuracy(['{"query":"alpha"}'], [])
    assert (
        p.fingerprint()["validation_selection"]
        == "mean_declared_native_metric_not_text_exact_match"
    )


@pytest.mark.parametrize("count", [1, 2])
def test_native_online_stream_is_ordered_test_feedback_not_offline_groups(count):
    a, r = Adapter(), Retriever()
    p = ACEOnlineRetrievalFeedback(adapter=a, retriever=r, stream=a.cases[:count])
    rows = [row.as_dict() for row in p.stream_examples()]
    assert [row["example_id"] for row in rows] == [e.example_id for e in a.cases[:count]]
    assert all(row["split"] == "test" for row in rows)
    assert "secret-qrel-doc" not in json.dumps([row["input"] for row in rows])
    assert p.fingerprint()["transductive_feedback_not_offline_or_heldout"]
    p.validate_online_examples(rows)
    assert p.score_query('{"query":"alpha"}', "train-id") == 0.6
    assert not p.answer_is_correct('```json\n{"query":"alpha"}\n```', "train-id")
    assert r.calls == [("alpha", 15)]
    with pytest.raises(ValidationError, match="offline"):
        p.fit_examples("train")
    with pytest.raises(ValidationError, match="offline"):
        p.validate_examples([], [])
    rows[0]["expected_output"] = "different"
    with pytest.raises(ValidationError, match="frozen stream"):
        p.validate_online_examples(rows)


def test_online_native_feedback_matches_offline_metric_and_preserves_drift_errors():
    offline, a, r = make()
    online = ACEOnlineRetrievalFeedback(adapter=a, retriever=r, stream=a.cases)
    predictions = ['{"query":"alpha"}', '{"query":"beta"}']
    targets = [e.example_id for e in a.cases]
    assert online.evaluate_accuracy(predictions, targets) == offline.evaluate_accuracy(
        predictions, targets
    )
    with pytest.raises(ValidationError, match="frozen stream"):
        online.validate_online_examples(list(reversed(online.stream_examples())))
    r.revision = "changed"
    with pytest.raises(ValidationError, match="changed"):
        online.score_query(predictions[0], targets[0])


def test_online_processor_rejects_empty_duplicate_foreign_and_missing_native_metric():
    a, r = Adapter(), Retriever()
    for stream in ([], [a.cases[0], a.cases[0]]):
        with pytest.raises(ValidationError):
            ACEOnlineRetrievalFeedback(adapter=a, retriever=r, stream=stream)
    with pytest.raises(ValidationError, match="metric"):
        ACEOnlineRetrievalFeedback(adapter=a, retriever=r, stream=a.cases, metric="not_native")
    p = ACEOnlineRetrievalFeedback(adapter=a, retriever=r, stream=a.cases)
    with pytest.raises(ValidationError, match="outside"):
        p.score_query('{"query":"alpha"}', "unbound-test-id")
