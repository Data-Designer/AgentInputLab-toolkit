"""Controlled planning-clause component, separate from full PS prompting.

The fixed first sentence of PS 201 requests a plan. One generation is attached
as advisory input for the caller's unchanged downstream reader/environment.
No solver, arithmetic answer extraction, tool action or plan-quality judge is
executed here. This is a declared component adaptation, not upstream parity.
"""

from __future__ import annotations

import math

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CapabilityError,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash

from .ps_prompt import (
    PLAN_AND_SOLVE_PROMPT_ID,
    PLAN_AND_SOLVE_TRIGGER,
    PLAN_AND_SOLVE_UPSTREAM_REVISION,
    render_plan_and_solve_prompt,
)

PLAN_AND_SOLVE_PLAN_TRIGGER = (
    "Let's first understand the problem and devise a plan to solve the problem."
)
PLAN_AND_SOLVE_PLAN_CONTRACT = "ps201.first-clause.raw-advisory-plan.v1"


def render_plan_component_prompt(question: str) -> str:
    return render_plan_and_solve_prompt(question, trigger=PLAN_AND_SOLVE_PLAN_TRIGGER)


class PlanAndSolvePlanComponent:
    """Generate an evaluator-blind planning suggestion in one counted call.

    Nonempty generated text is preserved verbatim apart from outer whitespace.
    It is not asserted to be an answer-free plan or independently verified.
    Empty output remains a paid failure; there is no fallback or second call.
    """

    method_id = "plan_and_solve"
    release_id = "plan.plan_and_solve.plan_component"
    family = SemanticFamily.DECOMPOSE

    def __init__(self, *, max_tokens=512, temperature=0.0, profile_kind="controlled_adapter"):
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValidationError("PS planning max_tokens must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValidationError("PS planning temperature must be finite and non-negative")
        if profile_kind not in {"controlled_adapter", "factorial_adapter"}:
            raise ValidationError("PS planning is a controlled/factorial component")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.profile_kind = profile_kind

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("PS planning requires a non-empty visible query")
        if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
            raise BudgetExceededError("PS planning has no generation budget")
        backend = context.backend
        if backend is None:
            raise ExternalDependencyError("PS planning requires an injected LLMBackend")
        if (
            not isinstance(backend, LLMBackend)
            or ModelCapability.COMPLETION not in backend.capabilities
        ):
            raise CapabilityError("PS planning requires completion capability")
        prompt = render_plan_component_prompt(state.query)
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=min(self.max_tokens, state.budget.max_output_tokens),
            temperature=self.temperature,
            seed=context.seed,
            metadata={
                "method_id": self.method_id,
                "release_id": self.release_id,
                "stage": "planning_component",
                "source_prompt_id": PLAN_AND_SOLVE_PROMPT_ID,
                "component_contract": PLAN_AND_SOLVE_PLAN_CONTRACT,
            },
        )
        response = backend.generate(request)
        if not isinstance(response.text, str) or not response.text.strip():
            raise AlgorithmOutputError("PS planning returned no final advisory text")
        plan = response.text.strip()
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "fixed_planning_clause",
                    {
                        "component_contract": PLAN_AND_SOLVE_PLAN_CONTRACT,
                        "source_prompt_id": PLAN_AND_SOLVE_PROMPT_ID,
                        "source_full_trigger_hash": content_hash(PLAN_AND_SOLVE_TRIGGER),
                        "planner_trigger": PLAN_AND_SOLVE_PLAN_TRIGGER,
                        "rendered_prompt_hash": content_hash(prompt),
                        "generation_request_hash": request.request_hash,
                        "upstream_revision_for_audit": PLAN_AND_SOLVE_UPSTREAM_REVISION,
                        "optimizer_input_projection": "query_only",
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "raw_advisory_plan",
                    {
                        "plan": plan,
                        "raw_response_hash": content_hash(response.text),
                        "finish_reason": response.finish_reason,
                        "downstream_execution_required": True,
                        "plan_quality_judged": False,
                        "solver_executed": False,
                    },
                ),
            ),
            counters={
                "model_calls": 1,
                "optimizer_model_calls": 1,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                **(
                    {"cost_usd": response.usage.cost_usd}
                    if response.usage.cost_usd is not None
                    else {}
                ),
            },
            warnings=("planning_clause_component_not_full_ps_pipeline",),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.APPEND_INSTRUCTION,
                    value="Suggested plan:\n" + plan,
                    target="instruction",
                    rationale="one-call PS planning-clause advisory component",
                ),
            ),
            trace=trace,
            implementation_identity=f"controlled_ps201_planning_clause@{PLAN_AND_SOLVE_UPSTREAM_REVISION}",
            profile_kind=self.profile_kind,
            metadata={
                "component_only": True,
                "component_contract": PLAN_AND_SOLVE_PLAN_CONTRACT,
                "full_system": False,
                "source_prompt_id": PLAN_AND_SOLVE_PROMPT_ID,
                "plan": plan,
                "plan_quality_judged": False,
                "response_finish_reason": response.finish_reason,
                "raw_response_preserved_except_outer_whitespace": True,
                "answer_free_plan_guaranteed": False,
                "optimizer_input_projection": "query_only",
                "gold_fields_visible_to_optimizer": False,
                "downstream_execution_required": True,
                "solver_executed": False,
                "tool_actions_executed": False,
                "paper_result_parity_claimed": False,
            },
        )


__all__ = [
    "PLAN_AND_SOLVE_PLAN_CONTRACT",
    "PLAN_AND_SOLVE_PLAN_TRIGGER",
    "PlanAndSolvePlanComponent",
    "render_plan_component_prompt",
]
