"""Synthetic boundary tests for the public controlled QueryGym constructor."""
from dataclasses import FrozenInstanceError, replace

import pytest

from agentinputlab.algorithms.reformulate.query2doc import (
    QUERYGYM_REVISION,
    Query2DocControlledLive,
    Query2DocLiveBindings,
    Query2DocLiveConfig,
    QueryGymRawResult,
    compose_sparse_expansion,
)
from agentinputlab.api import create_intervention
from agentinputlab.backends.llm import GenerationRequest, GenerationResponse, ReplayBackend, Usage
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import ExecutionBudget, FrozenAgentState, InterventionContext, PatchOp
from agentinputlab.contracts.errors import (
    BudgetExceededError,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.runner import InterventionRunner


class Runtime:
    def __init__(self, backend):
        self.backend = backend
        self.calls = 0
        self.revision = QUERYGYM_REVISION

    def fingerprint(self):
        return {"backend": self.backend.fingerprint(), "revision": self.revision,
                "source_closure_sha256": "a"*64, "test_fixture": True}

    def reformulate(self, request):
        self.calls += 1
        response = self.backend.generate(GenerationRequest(
            prompt=request.query, max_tokens=request.max_tokens,
            temperature=request.temperature, seed=request.seed))
        return QueryGymRawResult(
            reformulated=compose_sparse_expansion(request.query, response.text) if response.text else "",
            generated_text=response.text, prompt_hash="fixture-prompt", fallback=not bool(response.text),
            usage={"model_calls": 1, "retrieval_calls": 0, "input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens})


def setup(*, text="A hypothetical passage.", budget=None):
    backend = ReplayBackend([GenerationResponse(text, usage=Usage(input_tokens=7, output_tokens=4))])
    runtime = Runtime(backend)
    retriever = InMemoryBM25Retriever([Document("doc", "Real corpus text.", corpus_revision="fixture-v1")])
    context = InterventionContext(backend=backend, retriever=retriever, external_runtime=runtime, seed=17)
    state = FrozenAgentState(query="Fixed query", task_id="fixture", budget=budget or ExecutionBudget())
    bindings = Query2DocLiveBindings.from_context(context, budget=state.budget)
    return context, state, bindings


def test_public_factory_runs_native_bridge_with_observed_usage():
    context, state, bindings = setup()
    method = create_intervention("rewrite.query2doc.controlled_live", bindings=bindings)
    record = InterventionRunner().run(method, state, context)
    assert record.patch.release_id == "rewrite.query2doc.controlled_live"
    assert record.patch.apply(state).query == compose_sparse_expansion(state.query, "A hypothetical passage.")
    assert record.usage.model_calls == 1
    assert record.usage.input_tokens == 7 and record.usage.output_tokens == 4
    assert record.usage.unpriced_model_calls == 1
    assert record.usage.retrieval_calls == 0
    assert record.patch.metadata["native_source_release"] == "rewrite.query2doc.querygym_live"
    assert not record.patch.metadata["downstream_retrieval_performed"]
    assert context.external_runtime.calls == context.backend._calls == 1


def test_capture_only_and_json_builder_recipe_preserve_binding():
    context, state, bindings = setup()
    assert context.external_runtime.calls == context.backend._calls == 0
    method = create_intervention("rewrite.query2doc.controlled_live", bindings=to_jsonable(bindings))
    assert method.bindings.binding_hash == bindings.binding_hash
    assert method.propose(state, context).metadata["controlled_binding_sha256"] == bindings.binding_hash
    with pytest.raises(TypeError):
        bindings.generator["model_id"] = "mutated"
    with pytest.raises(FrozenInstanceError):
        bindings.budget = ExecutionBudget(max_model_calls=10)


@pytest.mark.parametrize("field", ["backend", "retriever", "external_runtime"])
def test_missing_dependency_fails_before_model_call(field):
    context, state, bindings = setup()
    with pytest.raises(ExternalDependencyError):
        Query2DocControlledLive(bindings).propose(state, replace(context, **{field: None}))
    assert context.backend._calls == context.external_runtime.calls == 0


@pytest.mark.parametrize("field", ["backend", "retriever", "external_runtime"])
def test_wrong_dependency_fails_before_model_call(field):
    context, state, bindings = setup()
    if field == "backend":
        other = ReplayBackend(["unused"], model_id="wrong-reader")
    elif field == "retriever":
        other = InMemoryBM25Retriever([Document("other", "Other text.", corpus_revision="different")])
    else:
        other = Runtime(context.backend)
        other.revision = "moving-source"
    with pytest.raises(ValidationError, match="fingerprint differs"):
        Query2DocControlledLive(bindings).propose(state, replace(context, **{field: other}))
    assert context.backend._calls == context.external_runtime.calls == 0


def test_budget_change_fails_before_execution_and_runner_enforces_remaining_budget():
    context, state, bindings = setup()
    with pytest.raises(ValidationError, match="budget differs"):
        Query2DocControlledLive(bindings).propose(
            replace(state, budget=ExecutionBudget(max_output_tokens=8192)), context)
    context, state, bindings = setup(budget=ExecutionBudget(max_model_calls=0))
    with pytest.raises(BudgetExceededError):
        InterventionRunner().run(Query2DocControlledLive(bindings), state, context)
    assert context.external_runtime.calls == context.backend._calls == 0


def test_native_empty_output_fallback_remains_explicit_and_costed():
    context, state, bindings = setup(text="")
    record = InterventionRunner().run(Query2DocControlledLive(bindings), state, context)
    assert record.patch.operations[0].op is PatchOp.NO_OP
    assert record.patch.metadata["fallback"]
    assert record.patch.trace.counters["fallbacks"] == 1
    assert record.usage.model_calls == 1


def test_runtime_source_and_generator_cannot_be_silently_aliased():
    _, _, bindings = setup()
    with pytest.raises(ValidationError, match="pinned QueryGym revision"):
        replace(bindings, runtime={**bindings.runtime, "revision": "moving"})
    with pytest.raises(ValidationError, match="source-closure"):
        replace(bindings, runtime={**bindings.runtime, "source_closure_sha256": ""})
    with pytest.raises(ValidationError, match="different generator"):
        replace(bindings, runtime={**bindings.runtime, "backend": {"model_id": "wrong"}})


def test_dependency_drift_after_native_call_is_not_silently_accepted():
    context, state, bindings = setup()
    original = context.external_runtime.reformulate

    def drifting(request):
        result = original(request)
        context.external_runtime.revision = "changed-during-execution"
        return result

    context.external_runtime.reformulate = drifting
    with pytest.raises(ValidationError, match="runtime fingerprint differs"):
        Query2DocControlledLive(bindings).propose(state, context)
    assert context.external_runtime.calls == context.backend._calls == 1


def test_unknown_json_recipe_fields_cannot_change_behavior():
    _, _, bindings = setup()
    with pytest.raises(ValidationError, match="unknown fields"):
        Query2DocControlledLive({**to_jsonable(bindings), "retry": True})
    with pytest.raises(ValidationError, match="invalid settings"):
        Query2DocControlledLive({**to_jsonable(bindings), "config": {"retry": True}})


@pytest.mark.parametrize("options", [
    {"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": 3.5},
    {"temperature": float("nan")}, {"temperature": float("inf")}, {"temperature": True},
    {"mode": "unknown"}, {"mode": None}, {"mode": "few_shot"},
    {"demonstration_ids": "abc"}, {"mode": "few_shot", "demonstration_ids": ["x", "x"]},
])
def test_invalid_controls_rejected_without_runtime(options):
    with pytest.raises(ValidationError):
        Query2DocLiveConfig(**options)


def test_factorial_label_does_not_claim_paper_result_parity():
    context, state, bindings = setup()
    patch = Query2DocControlledLive(bindings, profile_kind="factorial_adapter").propose(state, context)
    assert patch.profile_kind == "factorial_adapter"
    assert not patch.metadata["original_paper_result_parity_claimed"]
    with pytest.raises(ValidationError, match="profile_kind"):
        Query2DocControlledLive(bindings, profile_kind="official_native")
