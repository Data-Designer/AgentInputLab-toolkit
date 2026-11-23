"""File-binding fixtures, not real model acceptance."""

import hashlib
import json

import pytest

from agentinputlab.contracts import ValidationError
from agentinputlab.online_configuration import OnlineConfig
from agentinputlab.recipes.ace_online_files import create_online_feedback
from tests.unit.test_online_configuration import stub_native, value


def inputs(tmp_path, monkeypatch, kind):
    if kind == "bfcl":
        from tests.unit.test_bfcl_fit_configuration import inputs as base

        source, feedback, adapter = base(tmp_path, monkeypatch)
        for field in ("train_ids", "validation_ids", "heldout_ids"):
            source.pop(field)
        feedback.pop("partition_label")
        ids = ["case-1", "case-0"]
    else:
        from tests.unit.test_retrieval_fit_configuration import inputs as base

        source, feedback, adapter = base(tmp_path, monkeypatch)
        source.pop("groups")
        source.pop("partition_label")
        source["retrieval_limit_per_case"] = 7
        ids = ["validation-id", "train-id"]
    source.update(benchmark=adapter.benchmark_id, stream=dict(split="test", ids=ids))
    save(source, feedback)
    return source, feedback, adapter


def save(source, feedback):
    from pathlib import Path

    path = Path(feedback["source_input"]["path"])
    path.write_text(json.dumps(source))
    feedback["source_input"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", ["bfcl", "retrieval"])
def test_original_processors_order_and_scoring(tmp_path, monkeypatch, kind):
    source, feedback, adapter = inputs(tmp_path, monkeypatch, kind)
    p = create_online_feedback(**feedback, benchmark=adapter.benchmark_id)
    rows = p.stream_examples()
    assert [r.example_id for r in rows] == source["stream"]["ids"]
    assert all(r.split == "test" for r in rows)
    assert "secret" not in json.dumps([dict(r.input) for r in rows])
    p.validate_online_examples(rows)
    with pytest.raises(ValidationError):
        p.validate_online_examples(reversed(rows))
    if kind == "bfcl":
        assert p.score_answer('[{"lookup":{"x":3}}]', "case-0") == 1.0
        assert adapter.calls[-1] == ("case-0", [{"lookup": '{"x":3}'}])
    else:
        assert p.score_query('{"query":"alpha"}', "train-id") == 0.6
        assert adapter.calls[-1] == ("train-id", ["d1", "d2"])
    with pytest.raises(ValidationError):
        p.fit_examples("train")


@pytest.mark.parametrize("kind", ["bfcl", "retrieval"])
@pytest.mark.parametrize("change", ["duplicates", "empty", "unknown", "offline", "benchmark"])
def test_online_does_not_infer_or_relabel_offline_groups(tmp_path, monkeypatch, kind, change):
    source, feedback, adapter = inputs(tmp_path, monkeypatch, kind)
    if change == "duplicates":
        source["stream"]["ids"] *= 2
    elif change == "empty":
        source["stream"]["ids"] = []
    elif change == "unknown":
        source["stream"]["ids"] = ["absent"]
    elif change == "offline":
        source["train_ids"] = source["stream"]["ids"]
    else:
        source["benchmark"] = "another.dataset"
    save(source, feedback)
    with pytest.raises(ValidationError):
        create_online_feedback(**feedback, benchmark=adapter.benchmark_id)
    assert not adapter.calls


@pytest.mark.parametrize("kind", ["bfcl", "retrieval"])
def test_file_execute_selects_native_online_processor(tmp_path, monkeypatch, kind):
    source, feedback, adapter = inputs(tmp_path, monkeypatch, kind)
    v = value()
    v.pop("stream")
    v.update(benchmark=adapter.benchmark_id, feedback=feedback)
    calls = stub_native(monkeypatch)
    OnlineConfig.from_dict(v).execute(tmp_path / "out")
    p = calls[0][1]["data_processor"]
    assert [r.example_id for r in p.stream_examples()] == source["stream"]["ids"]
    assert type(p).__name__ == (
        "ACEOnlineBFCLFeedback" if kind == "bfcl" else "BudgetedOnlineRetrievalFeedback"
    )
    if kind == "bfcl":
        v["feedback"]["model_id"] = "other"
        with pytest.raises(ValidationError, match="target deployment"):
            OnlineConfig.from_dict(v)


def test_retrieval_quota_is_explicit_and_cached_queries_are_not_charged(tmp_path, monkeypatch):
    source, feedback, adapter = inputs(tmp_path, monkeypatch, "retrieval")
    source["retrieval_limit_per_case"] = 1
    save(source, feedback)
    p = create_online_feedback(**feedback, benchmark=adapter.benchmark_id)
    assert p.score_query('{"query":"alpha"}', "train-id") == 0.6
    assert p.score_query('{"query":"alpha"}', "train-id") == 0.6
    assert p.retrieval_attempts["train-id"] == 1
    before = list(adapter.calls)
    with pytest.raises(ValidationError, match="before IO"):
        p.score_query('{"query":"beta"}', "train-id")
    assert adapter.calls == before
    assert p.score_query('{"query":"beta"}', "validation-id") == 0.2
    source.pop("retrieval_limit_per_case")
    save(source, feedback)
    with pytest.raises(ValidationError, match="explicit nonnegative"):
        create_online_feedback(**feedback, benchmark=adapter.benchmark_id)


def test_failed_retrieval_consumes_allowance(tmp_path, monkeypatch):
    source, feedback, adapter = inputs(tmp_path, monkeypatch, "retrieval")
    source["retrieval_limit_per_case"] = 1
    save(source, feedback)
    p = create_online_feedback(**feedback, benchmark=adapter.benchmark_id)

    def fail(*a, **kw):
        raise OSError("fixture unavailable")

    monkeypatch.setattr(type(p.retriever), "retrieve", fail)
    with pytest.raises(OSError):
        p.score_query('{"query":"alpha"}', "train-id")
    assert p.retrieval_attempts["train-id"] == 1
    with pytest.raises(ValidationError, match="before IO"):
        p.score_query('{"query":"alpha"}', "train-id")


def test_concurrent_same_query_consumes_only_one_retrieval(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    source, feedback, adapter = inputs(tmp_path, monkeypatch, "retrieval")
    source["retrieval_limit_per_case"] = 1
    save(source, feedback)
    p = create_online_feedback(**feedback, benchmark=adapter.benchmark_id)
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(lambda _: p.score_query('{"query":"alpha"}', "train-id"), range(8)))
    assert scores == [0.6] * 8
    assert p.retrieval_attempts["train-id"] == 1
    assert sum(bool(ids) for _, ids in adapter.calls) == 1
