from __future__ import annotations

import pytest

from agentinputlab.algorithms.decompose.plan_and_solve import (
    PLAN_AND_SOLVE_PLAN_TRIGGER,
    PLAN_AND_SOLVE_TRIGGER,
    PlanAndSolvePlanComponent,
    render_plan_component_prompt,
)
from agentinputlab.api import ToolkitService, create_intervention
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, ModelCapability, Usage
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CapabilityError,
    ContextBlock,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    Message,
    ValidationError,
)
from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.experiments.builders import RegistryInterventionBuilder
from agentinputlab.experiments.drivers.static_production import (
    StaticArmBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RID = "plan.plan_and_solve.plan_component"
TEXT = "1. Identify the relevant evidence.\n2. Compare the facts before answering."


def backend(text=TEXT):
    return CallableBackend(
        lambda _: GenerationResponse(
            text=text,
            model="fixture",
            finish_reason="stop",
            usage=Usage(input_tokens=21, output_tokens=13, cost_usd=0.003),
        ),
        model_id="fixture",
        revision="fixture-v1",
        callable_id="ps-planning-fixture",
    )


def test_fixed_source_clause_excludes_solve_clause():
    assert PLAN_AND_SOLVE_TRIGGER.split(" Then,", 1)[0] == PLAN_AND_SOLVE_PLAN_TRIGGER
    assert (
        render_plan_component_prompt("Question") == "Q:Question\nA: " + PLAN_AND_SOLVE_PLAN_TRIGGER
    )


def test_one_generation_query_only_preserves_reader_state_and_budget():
    b = backend()
    state = FrozenAgentState(
        query="Visible question",
        instruction="Reader instruction",
        task_id="fixed",
        context=(ContextBlock(text="Reader evidence"),),
        messages=(Message(role="user", content="Reader history"),),
        metadata={"gold_answer": "Hidden label"},
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096),
    )
    method = create_intervention(RID)
    patch = method.propose(state, InterventionContext(backend=b, seed=17))
    updated = patch.apply(state)
    assert len(b.calls) == 1
    assert b.calls[0].max_tokens == 512 and b.calls[0].seed == 17
    assert b.calls[0].stop == () and b.calls[0].response_format is None
    for hidden in ("Reader instruction", "Reader evidence", "Reader history", "Hidden label"):
        assert hidden not in b.calls[0].prompt
    assert updated.query == state.query and updated.context == state.context
    assert updated.messages == state.messages and updated.budget == state.budget
    assert updated.instruction == "Reader instruction\n\nSuggested plan:\n" + TEXT
    assert patch.trace.counters["optimizer_model_calls"] == 1
    assert patch.trace.counters["cost_usd"] == 0.003
    assert patch.metadata["full_system"] is False
    assert patch.metadata["gold_fields_visible_to_optimizer"] is False
    assert patch.metadata["solver_executed"] is False
    assert patch.metadata["paper_result_parity_claimed"] is False


@pytest.mark.parametrize("text", ["", "  \n "])
def test_empty_final_paid_failure_is_not_retried(text):
    b = backend(text)
    monitor = RunMonitor()
    with pytest.raises(AlgorithmOutputError):
        InterventionRunner(monitor=monitor).run(
            PlanAndSolvePlanComponent(), FrozenAgentState(query="q"), InterventionContext(backend=b)
        )
    assert len(b.calls) == 1
    usage = monitor.list_snapshots()[0].usage
    assert (usage.model_calls, usage.input_tokens, usage.output_tokens) == (1, 21, 13)


def test_opaque_text_is_not_rewritten_to_make_a_better_plan():
    text = "The answer might be 42.\nThis is an unverified suggestion."
    patch = PlanAndSolvePlanComponent().propose(
        FrozenAgentState(query="q"), InterventionContext(backend=backend(text))
    )
    assert patch.metadata["plan"] == text
    assert patch.metadata["plan_quality_judged"] is False
    assert patch.metadata["answer_free_plan_guaranteed"] is False


def test_cap_obeys_smaller_remaining_budget():
    b = backend()
    PlanAndSolvePlanComponent().propose(
        FrozenAgentState(
            query="q", budget=ExecutionBudget(max_output_tokens=37, max_model_calls=1)
        ),
        InterventionContext(backend=b),
    )
    assert b.calls[0].max_tokens == 37


@pytest.mark.parametrize(
    "values",
    [
        {"max_tokens": 0},
        {"max_tokens": True},
        {"temperature": True},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"profile_kind": "official_native"},
    ],
)
def test_invalid_constructor_rejected(values):
    with pytest.raises(ValidationError):
        PlanAndSolvePlanComponent(**values)


def test_missing_dependency_and_budget_preflight():
    state = FrozenAgentState(query="q")
    with pytest.raises(ExternalDependencyError):
        PlanAndSolvePlanComponent().propose(state, InterventionContext())
    b = backend()
    b.capabilities = frozenset({ModelCapability.CHAT})
    with pytest.raises(CapabilityError):
        PlanAndSolvePlanComponent().propose(state, InterventionContext(backend=b))
    for budget in (ExecutionBudget(max_model_calls=0), ExecutionBudget(max_output_tokens=0)):
        with pytest.raises(BudgetExceededError):
            PlanAndSolvePlanComponent().propose(
                FrozenAgentState(query="q", budget=budget), InterventionContext(backend=b)
            )
    assert b.calls == []


def test_registry_builder_constructs_json_recipe_and_factorial_identity():
    builder = RegistryInterventionBuilder()
    parameters = {
        "schema_version": 1,
        "release_id": RID,
        "constructor_kwargs": {"max_tokens": 512, "profile_kind": "factorial_adapter"},
    }
    assert not builder.preflight(parameters)
    result = builder.build(
        StaticArmBuildRequest(
            arm_id="planning",
            release_id=RID,
            family=SemanticFamily.DECOMPOSE,
            profile_kind=ProfileKind.FACTORIAL_ADAPTER,
            model_slot="fixture",
            seed=17,
            builder_parameters=parameters,
            arm_parameters={},
            runtime=StaticRuntimeComponents(backend=backend()),
        )
    )
    patch = result.propose(FrozenAgentState(query="q"), InterventionContext(backend=backend()))
    assert patch.profile_kind == ProfileKind.FACTORIAL_ADAPTER


def test_operator_configured_sdk_service_uses_same_python_entry():
    b = backend()
    service = ToolkitService(
        factories={RID: lambda: create_intervention(RID)},
        context_factory=lambda seed: InterventionContext(backend=b, seed=seed),
    )
    output = service.execute({"release_id": RID, "state": {"query": "q"}, "seed": 17})
    assert output["scored"] is False and output["benchmark_score"] is None
    assert output["state"]["instruction"] == "Suggested plan:\n" + TEXT
    assert len(b.calls) == 1
