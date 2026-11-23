"""Synthetic cache boundary fixtures, not real-model acceptance evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from agentinputlab.algorithms.decompose.plan_and_budget import (
    CachedPlanBudgetRecord,
    PlanAndBudgetCache,
    allocate_tokens,
)
from agentinputlab.api import create_intervention
from agentinputlab.contracts import ArtifactError, FrozenAgentState, InterventionContext


def row():
    return {"id": "fixed", "query": "Synthetic question", "steps": ["Step A", "Step B"],
            "credits": [20, 60], "level": 3, "reference": "PRIVATE_REFERENCE",
            "solution": "PRIVATE_SOLUTION", "answer": "PRIVATE_ANSWER"}


def load(tmp_path, value, **kwargs):
    path = tmp_path / "cache.json"
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return PlanAndBudgetCache.from_native_json(
        path, expected_sha256=hashlib.sha256(raw).hexdigest(), **kwargs
    )


def test_native_envelope_is_pinned_immutable_and_solver_only(tmp_path):
    cache = load(tmp_path, {"last_question_total": 1.5, "stats": [row()]})
    assert cache.verification == "caller_sha256_verified"
    record = cache.records["fixed"]
    assert record.credits == (20.0, 60.0)
    assert "cached_credit_sum_not_100" in record.audit_warnings
    for name in ("reference", "solution", "answer", "gold"):
        assert not hasattr(record, name)
    with pytest.raises(TypeError):
        cache.records["new"] = record
    with pytest.raises(FrozenInstanceError):
        cache.artifact_sha256 = "a" * 64
    method = create_intervention("plan.plan_and_budget.native_cached_soft_budget",
                                 records=cache, total_word_budget=10)
    patch = method.propose(FrozenAgentState(query=record.query, task_id="fixed"),
                           InterventionContext())
    assert patch.metadata["cache_artifact_sha256"] == cache.artifact_sha256
    assert patch.metadata["cache_verification"] == "caller_sha256_verified"
    assert patch.metadata["historical_cache_generation_cost"] == "unknown_not_free"
    assert not patch.metadata["hard_decoder_cap"]
    assert patch.metadata["allocation_tie_policy"] == "stable_source_index"
    assert not patch.metadata["numpy_tie_order_parity_claimed"]
    assert "PRIVATE_" not in str(patch)
    assert patch.trace.counters["optimizer_model_calls"] == 0
    with pytest.raises(ArtifactError, match="mismatch"):
        method.propose(FrozenAgentState(query="Another question", task_id="fixed"),
                       InterventionContext())


def test_direct_construction_does_not_claim_file_verification():
    record = CachedPlanBudgetRecord("fixed", "Question", ("A",), (100,), 3)
    values = {"fixed": record}
    cache = PlanAndBudgetCache(values, "a" * 64)
    values.clear()
    assert list(cache.records) == ["fixed"]
    assert cache.verification == "caller_declared_digest"
    method = create_intervention("plan.plan_and_budget.native_cached_soft_budget",
                                 records=cache, total_word_budget=10)
    patch = method.propose(FrozenAgentState(query="Question", task_id="fixed"),
                           InterventionContext())
    assert patch.metadata["cache_verification"] == "caller_declared_digest"


def test_equal_weight_remainders_use_disclosed_stable_source_indices():
    allocation = allocate_tokens((15, 15, 20, 10, 40), 512, "constant")
    assert allocation == (77, 76, 103, 51, 205)
    assert sum(allocation) == 512


@pytest.mark.parametrize("mutate", [
    lambda r: {**r, "id": None},
    lambda r: {**r, "id": " "},
    lambda r: {**r, "query": 7},
    lambda r: {**r, "query": ""},
    lambda r: {**r, "steps": []},
    lambda r: {**r, "steps": [""]},
    lambda r: {**r, "steps": [42, "B"]},
    lambda r: {**r, "credits": [True, 99]},
    lambda r: {**r, "credits": [0, 0]},
    lambda r: {**r, "credits": [-1, 101]},
    lambda r: {**r, "credits": ["20", 80]},
    lambda r: {**r, "credits": [100]},
    lambda r: {**r, "level": True},
    lambda r: {**r, "level": 1.5},
    lambda r: {**r, "level": 6},
    lambda r: {**r, "unexpected": "field"},
])
def test_malformed_native_rows_rejected(tmp_path, mutate):
    with pytest.raises(ArtifactError):
        load(tmp_path, {"last_question_total": 0, "stats": [mutate(row())]})


@pytest.mark.parametrize("value", [
    {"stats": [row()]},
    {"last_question_total": 0, "stats": []},
    {"last_question_total": 0, "stats": [row(), row()]},
    {"last_question_total": 0, "stats": "not a list"},
    {"last_question_total": 0, "stats": [None]},
    {"last_question_total": -1, "stats": [row()]},
    {"last_question_total": True, "stats": [row()]},
    {"last_question_total": 0, "stats": [row()], "extra": 1},
])
def test_malformed_envelope_rejected(tmp_path, value):
    with pytest.raises(ArtifactError):
        load(tmp_path, value)


def test_file_boundaries_reject_changed_oversized_duplicate_and_symlink(tmp_path):
    raw = json.dumps({"last_question_total": 0, "stats": [row()]}).encode()
    path = tmp_path / "cache.json"
    path.write_bytes(raw)
    pin = hashlib.sha256(raw).hexdigest()
    for kwargs in ({"expected_sha256": "a" * 64},
                   {"expected_sha256": pin, "max_bytes": 4}):
        with pytest.raises(ArtifactError):
            PlanAndBudgetCache.from_native_json(path, **kwargs)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ArtifactError):
        PlanAndBudgetCache.from_native_json(link, expected_sha256=pin)
    raw = b'{"last_question_total":0,"stats":[],"stats":[]}'
    path.write_bytes(raw)
    with pytest.raises(ArtifactError, match="unique"):
        PlanAndBudgetCache.from_native_json(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_nonfinite_values_rejected(tmp_path):
    for value in (float("nan"), float("inf"), -float("inf")):
        bad = {**row(), "credits": [value, 60]}
        with pytest.raises(ArtifactError, match="finite"):
            load(tmp_path, {"last_question_total": 0, "stats": [bad]})
