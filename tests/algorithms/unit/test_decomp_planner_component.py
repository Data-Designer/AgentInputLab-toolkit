from __future__ import annotations

from dataclasses import replace

import pytest

from agentinputlab.algorithms.decompose.decomp import (
    DECOMP_PLANNER_MODE,
    DECOMP_PLANNER_RELEASE,
    DECOMP_PLANNER_SCHEMA,
    DECOMP_REVISION,
    DecompPlannerComponent,
    DecompPlannerRequest,
)
from agentinputlab.api import create_intervention
from agentinputlab.backends.llm import CallableBackend
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.integrations.external_workers import DecompPlannerLiveRuntime
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner


class Runtime:
    def __init__(self, backend, **changes):
        self.backend, self.changes, self.calls = backend, changes, []

    def fingerprint(self):
        return {
            "backend_binding_mode": DECOMP_PLANNER_MODE,
            "upstream_revision": DECOMP_REVISION,
            "backend": self.backend.fingerprint(),
            "component_only": True,
        }

    def run(self, request):
        self.calls.append(request)
        selected = {
            "question": "(select) [qa] Find the source fact",
            "next_command": "execute",
            "score": 0.2,
        }
        return {
            "schema_version": DECOMP_PLANNER_SCHEMA,
            "profile_id": DECOMP_PLANNER_RELEASE,
            "case_id": request.case_id,
            "upstream_revision": DECOMP_REVISION,
            "paragraph_ids": request.paragraph_ids,
            "context_hash": request.context_hash,
            "component_only": True,
            "router_executed": False,
            "qa_executed": False,
            "selected": selected,
            "candidates": [selected],
            "native_artifacts": {
                "config_artifact_id": "fixture.config",
                "backend_binding_mode": DECOMP_PLANNER_MODE,
            },
            "usage": {
                "model_calls": 1,
                "retrieval_calls": 0,
                "input_tokens": 37,
                "output_tokens": 9,
            },
            **self.changes,
        }


def fixture(**changes):
    backend = CallableBackend(
        lambda _: pytest.fail("host generate must not replace native completion"),
        revision="fixture-v1",
        callable_id="native-planner-fixture",
    )
    runtime = Runtime(backend, **changes)
    context = InterventionContext(backend=backend, external_runtime=runtime, seed=17)
    state = FrozenAgentState(
        query="visible question",
        task_id="case",
        instruction="reader only",
        context=(ContextBlock(block_id="p1", text="visible passage"),),
        metadata={"gold": "hidden"},
    )
    return runtime, context, state


def test_factory_and_one_native_step_preserve_reader_input():
    runtime, context, state = fixture()
    method = create_intervention(DECOMP_PLANNER_RELEASE, config_artifact_id="fixture.config")
    patch = method.propose(state, context)
    updated = patch.apply(state)
    assert isinstance(runtime.calls[0], DecompPlannerRequest)
    assert (
        runtime.calls[0].max_steps == 1 and runtime.calls[0].schema_version == DECOMP_PLANNER_SCHEMA
    )
    assert runtime.calls[0].paras == ("visible passage",)
    assert "hidden" not in str(runtime.calls[0]) and "reader only" not in str(runtime.calls[0])
    assert updated.query == state.query and updated.context == state.context
    assert updated.instruction.startswith("reader only\n\nNative DecomP next subquestion")
    assert patch.metadata["full_system"] is False and patch.metadata["router_executed"] is False
    assert patch.metadata["qa_executed"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"component_only": False},
        {"final_answer": "forbidden"},
        {"router_executed": True},
        {"qa_executed": True},
        {"context_hash": "0" * 64},
        {"schema_version": "agentinputlab.decomp.worker.v2"},
        {"candidates": []},
        {"native_artifacts": {"config_artifact_id": "other"}},
    ],
)
def test_full_or_mismatched_outputs_rejected(changes):
    _, context, state = fixture(**changes)
    with pytest.raises(AlgorithmOutputError):
        DecompPlannerComponent(config_artifact_id="fixture.config").propose(state, context)


def test_unusable_native_end_marker_keeps_paid_usage():
    selected = {"question": "[EOQ]", "next_command": "[EOQ]", "score": 0}
    runtime, context, state = fixture(selected=selected, candidates=[selected])
    monitor = RunMonitor()
    with pytest.raises(AlgorithmOutputError):
        InterventionRunner(monitor=monitor).run(
            DecompPlannerComponent(config_artifact_id="fixture.config"), state, context
        )
    assert len(runtime.calls) == 1
    usage = monitor.list_snapshots()[0].usage
    assert (usage.model_calls, usage.input_tokens, usage.output_tokens) == (1, 37, 9)


def test_native_runtime_distinguishes_schema_and_command_flag():
    assert DecompPlannerLiveRuntime.request_type is DecompPlannerRequest
    assert DecompPlannerLiveRuntime.worker_flags == ("--planner-component",)
    assert DecompPlannerLiveRuntime.worker_id != "agentinputlab.decomp.controlled-live.v1"


def test_budget_and_request_boundary_rejected_before_worker():
    runtime, context, state = fixture()
    with pytest.raises(BudgetExceededError):
        DecompPlannerComponent(config_artifact_id="fixture.config").propose(
            replace(state, budget=ExecutionBudget(max_model_calls=0)), context
        )
    assert not runtime.calls
    DecompPlannerComponent(config_artifact_id="fixture.config").propose(state, context)
    with pytest.raises(ValidationError):
        replace(runtime.calls[0], max_steps=2)


def test_finite_native_input_budget_requires_an_explicit_bound_and_retains_it():
    runtime, context, state = fixture()
    state = replace(state, budget=ExecutionBudget(max_context_tokens=262144))
    method = DecompPlannerComponent(config_artifact_id="fixture.config")
    with pytest.raises(ValidationError, match="pre-call estimators"):
        method.propose(state, context)
    assert runtime.calls == []
    original = runtime.fingerprint
    runtime.fingerprint = lambda: {**original(), "per_call_input_token_bound": 40960}
    method.propose(state, context)
    assert runtime.calls[0].execution_budget.max_context_tokens == 262144
    assert runtime.calls[0].execution_budget.max_output_tokens == 4096


def test_insufficient_native_input_reservation_does_not_start_a_worker():
    runtime, context, state = fixture()
    original = runtime.fingerprint
    runtime.fingerprint = lambda: {**original(), "per_call_input_token_bound": 40960}
    state = replace(state, budget=ExecutionBudget(max_context_tokens=40959))
    with pytest.raises(BudgetExceededError, match="input budget"):
        DecompPlannerComponent(config_artifact_id="fixture.config").propose(state, context)
    assert runtime.calls == []
