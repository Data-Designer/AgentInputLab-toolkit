"""Clean-room paper-spec implementation of the Plan-and-Solve PS prompt.

The authors' repository has no source-code license.  This module therefore
implements the short prompt intervention described in the ACL 2023 paper; it
does not vendor their experiment harness or claim historical native execution.
"""

from __future__ import annotations

from agentinputlab.contracts import (
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
)
from agentinputlab.contracts.serialization import content_hash

PLAN_AND_SOLVE_UPSTREAM_REVISION = "9d5a176f5b4de1011d426009ab6398dc76046e88"
PLAN_AND_SOLVE_PROMPT_ID = 201
PLAN_AND_SOLVE_TRIGGER = (
    "Let's first understand the problem and devise a plan to solve the problem. "
    "Then, let's carry out the plan to solve the problem step by step."
)
PLAN_AND_SOLVE_PLUS_PROMPT_ID = 302
PLAN_AND_SOLVE_PLUS_TRIGGER = (
    "Let's first understand the problem, extract relevant variables and their corresponding "
    "numerals, and devise a complete plan. Then, let's carry out the plan, calculate "
    "intermediate variables (pay attention to correct numerical calculation and commonsense), "
    "solve the problem step by step, and show the answer."
)


def render_plan_and_solve_prompt(
    question: str,
    *,
    trigger: str = PLAN_AND_SOLVE_TRIGGER,
) -> str:
    """Render the paper/repository PS input serialization for golden tests."""

    return f"Q:{question}\nA: {trigger}"


class PlanAndSolvePSPrompt:
    """Append the PS instruction to the frozen downstream-agent input."""

    method_id = "plan_and_solve"
    release_id = "plan.plan_and_solve.ps_prompt"
    family = SemanticFamily.DECOMPOSE
    prompt_id = PLAN_AND_SOLVE_PROMPT_ID
    trigger = PLAN_AND_SOLVE_TRIGGER
    variant = "ps"

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        rendered = render_plan_and_solve_prompt(state.query, trigger=self.trigger)
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "paper_prompt_spec",
                    {
                        "prompt_id": self.prompt_id,
                        "variant": self.variant,
                        "trigger": self.trigger,
                        "rendered_prompt": rendered,
                        "rendered_prompt_hash": content_hash(rendered),
                        "upstream_revision_for_audit": PLAN_AND_SOLVE_UPSTREAM_REVISION,
                        "historical_backend": "text-davinci-003",
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "append_plan_and_solve_instruction",
                    {
                        "instruction": self.trigger,
                        "downstream_execution_required": True,
                    },
                ),
            ),
            counters={"optimizer_model_calls": 0},
            warnings=(
                "clean_room_paper_spec_not_upstream_source_port",
                "modern_backend_results_are_controlled_model_transfer",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.APPEND_INSTRUCTION,
                    value=self.trigger,
                    target="instruction",
                    rationale=f"ACL 2023 Plan-and-Solve {self.variant.upper()} prompt instruction",
                ),
            ),
            trace=trace,
            implementation_identity=f"paper_reimplementation:acl2023:{self.variant}_prompt_{self.prompt_id}",
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "clean_room_paper_spec",
                "prompt_id": self.prompt_id,
                "variant": self.variant,
                "historical_native_available": False,
            },
        )


class PlanAndSolvePSPlusPrompt(PlanAndSolvePSPrompt):
    """Append the source-fixed PS+ 302 instruction; the caller executes the LLM.

    This input intervention does not execute the original arithmetic answer-
    extraction pass or replace a benchmark's parser/scorer. Downstream model
    transfer and task-specific output handling remain explicit caller choices.
    """

    release_id = "plan.plan_and_solve.ps_plus_prompt"
    prompt_id = PLAN_AND_SOLVE_PLUS_PROMPT_ID
    trigger = PLAN_AND_SOLVE_PLUS_TRIGGER
    variant = "ps_plus"
