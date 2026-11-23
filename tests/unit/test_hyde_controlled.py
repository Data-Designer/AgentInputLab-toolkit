"""Boundary fixtures for the dependency-locked HyDE public profile, not live evals."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from agentinputlab.algorithms.acquire import HyDEConfig, HyDEControlled, HyDEControlledBindings
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab.backends.llm import GenerationResponse, ReplayBackend, Usage
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.contracts.errors import (
    AlgorithmOutputError,
    BudgetExceededError,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.runner import InterventionRunner


class Embedder:
    embedder_id = "controlled-boundary-fixture"

    def __init__(self):
        self.identity = {"model": self.embedder_id, "revision": "fixture-v1"}
        self.calls = 0

    def fingerprint(self):
        return self.identity

    def embed(self, texts):
        self.calls += 1
        return tuple((float(i), 1.0) for i in range(len(texts)))


class Searcher:
    searcher_id = "controlled-boundary-index"

    def __init__(self):
        self.identity = {"index": self.searcher_id, "corpus": "fixture-v1"}
        self.calls = 0
        self.last_top_k = None

    def fingerprint(self):
        return self.identity

    def search(self, vector, *, top_k):
        self.calls += 1
        self.last_top_k = top_k
        return (
            RetrievalResult(Document("d1", "Corpus evidence.", corpus_revision="fixture-v1"), 1, 1),
        )


def setup(*, responses=None, config=None, budget=None):
    config = config or HyDEConfig(num_hypotheses=2, max_tokens=32, top_k=1)
    budget = budget or ExecutionBudget()
    backend = ReplayBackend(responses or ["First hypothesis", "Second hypothesis"])
    context = InterventionContext(
        backend=backend, dense_embedder=Embedder(), vector_searcher=Searcher(), seed=17
    )
    bindings = HyDEControlledBindings.from_context(context, budget=budget, config=config)
    return context, bindings, FrozenAgentState(query="Fixed fixture query", budget=budget)


def test_public_factory_and_runner_keep_real_instrumented_dependencies():
    response = GenerationResponse("hypothesis", usage=Usage(input_tokens=3, output_tokens=2))
    context, bindings, state = setup(responses=[response, response])
    method = create_intervention("acquire.hyde.controlled", bindings=bindings)
    result = run_intervention(method, state, context)
    record = result["record"]
    assert record["usage"]["model_calls"] == 2
    assert record["usage"]["embedding_calls"] == 1
    assert record["usage"]["retrieval_calls"] == 1
    assert record["usage"]["output_tokens"] == 4
    assert result["benchmark_score"] is None
    assert record["patch"]["metadata"]["controlled_binding_sha256"] == bindings.binding_hash
    assert not record["patch"]["metadata"]["weight_files_verified_by_binding"]
    assert context.vector_searcher.last_top_k == 1


def test_binding_capture_performs_no_runtime_operations_and_detaches_nested_identity():
    context, bindings, _ = setup()
    before = bindings.binding_hash
    assert context.backend._calls == context.dense_embedder.calls == context.vector_searcher.calls == 0
    context.dense_embedder.identity["revision"] = "different"
    assert bindings.embedder["revision"] == "fixture-v1"
    assert bindings.binding_hash == before
    with pytest.raises(TypeError):
        bindings.embedder["revision"] = "mutable"
    with pytest.raises(FrozenInstanceError):
        bindings.config = HyDEConfig(num_hypotheses=1)


@pytest.mark.parametrize("dependency", ["backend", "dense_embedder", "vector_searcher"])
def test_changed_dependency_rejected_before_model_call(dependency):
    context, bindings, state = setup()
    if dependency == "backend":
        changed = ReplayBackend(["unused"], model_id="different")
    else:
        changed = Embedder() if dependency == "dense_embedder" else Searcher()
        changed.identity["revision"] = "different"
    with pytest.raises(ValidationError, match="fingerprint differs"):
        HyDEControlled(bindings).propose(state, replace(context, **{dependency: changed}))
    assert context.backend._calls == 0


def test_changed_budget_rejected_before_model_call():
    context, bindings, state = setup()
    with pytest.raises(ValidationError, match="budget differs"):
        HyDEControlled(bindings).propose(
            replace(state, budget=ExecutionBudget(max_output_tokens=8192)), context
        )
    assert context.backend._calls == 0


def test_identity_change_during_execution_rejected_without_hiding_consumed_calls():
    context, bindings, state = setup()
    original = context.vector_searcher.search

    def drifting_search(vector, *, top_k):
        result = original(vector, top_k=top_k)
        context.vector_searcher.identity["corpus"] = "changed-during-run"
        return result

    context.vector_searcher.search = drifting_search
    with pytest.raises(ValidationError, match="searcher fingerprint differs"):
        HyDEControlled(bindings).propose(state, context)
    assert context.backend._calls == 2
    assert context.vector_searcher.calls == 1


def test_all_empty_outputs_remain_failures_not_no_op():
    context, bindings, state = setup(responses=["", ""])
    with pytest.raises(AlgorithmOutputError, match="no usable"):
        HyDEControlled(bindings).propose(state, context)
    assert context.backend._calls == 2


def test_runner_budget_is_not_bypassed_by_locked_profile():
    context, bindings, state = setup(budget=ExecutionBudget(max_model_calls=1))
    with pytest.raises(BudgetExceededError):
        InterventionRunner().run(HyDEControlled(bindings), state, context)
    assert context.backend._calls == 1


def test_data_only_recipe_is_round_trippable_and_rejects_unknown_fields():
    context, bindings, state = setup()
    method = create_intervention("acquire.hyde.controlled", bindings=to_jsonable(bindings))
    assert method.bindings == bindings
    assert method.propose(state, context).release_id == "acquire.hyde.controlled"
    with pytest.raises(ValidationError, match="unknown fields"):
        HyDEControlled({**to_jsonable(bindings), "silent_fallback": True})
    with pytest.raises(ValidationError, match="invalid setting fields"):
        HyDEControlled({**to_jsonable(bindings), "config": {"unexpected": True}})


@pytest.mark.parametrize("bad", [None, {}, {"model": float("nan")}, {"model": object()}])
def test_invalid_fingerprint_is_not_a_valid_lock(bad):
    _, bindings, _ = setup()
    with pytest.raises(ValidationError, match="fingerprint"):
        replace(bindings, generator=bad)


@pytest.mark.parametrize("dependency", ["backend", "dense_embedder", "vector_searcher"])
def test_missing_runtime_dependency_fails_before_dispatch(dependency):
    context, bindings, state = setup()
    with pytest.raises(ExternalDependencyError):
        HyDEControlled(bindings).propose(state, replace(context, **{dependency: None}))
    assert context.backend._calls == 0


def test_factorial_label_is_explicit_and_does_not_claim_paper_parity():
    context, bindings, state = setup()
    patch = HyDEControlled(bindings, profile_kind="factorial_adapter").propose(state, context)
    assert patch.profile_kind == "factorial_adapter"
    assert not patch.metadata["original_paper_result_parity_claimed"]
    with pytest.raises(ValidationError, match="profile_kind"):
        HyDEControlled(bindings, profile_kind="official_native")


@pytest.mark.parametrize("field", ["num_hypotheses", "max_tokens", "top_k"])
@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "2"])
def test_config_rejects_invalid_integer_settings(field, bad):
    with pytest.raises(ValidationError):
        HyDEConfig(**{field: bad})


@pytest.mark.parametrize("bad", [True, float("inf"), float("nan"), -1, "0.7"])
def test_config_rejects_invalid_temperature(bad):
    with pytest.raises(ValidationError):
        HyDEConfig(temperature=bad)
