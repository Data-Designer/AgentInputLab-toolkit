from __future__ import annotations

import unittest

from agentinputlab.algorithms.decompose.plan_and_solve import (
    PLAN_AND_SOLVE_PLUS_PROMPT_ID,
    PLAN_AND_SOLVE_PLUS_TRIGGER,
    PLAN_AND_SOLVE_PROMPT_ID,
    PLAN_AND_SOLVE_TRIGGER,
    PlanAndSolvePSPlusPrompt,
    PlanAndSolvePSPrompt,
    render_plan_and_solve_prompt,
)
from agentinputlab.api import create_intervention
from agentinputlab.contracts import FrozenAgentState, InterventionContext, PatchOp


class PlanAndSolveTests(unittest.TestCase):
    def test_ps_plus_source_fixed_302_golden(self) -> None:
        self.assertEqual(PLAN_AND_SOLVE_PLUS_PROMPT_ID, 302)
        self.assertEqual(
            PLAN_AND_SOLVE_PLUS_TRIGGER,
            "Let's first understand the problem, extract relevant variables and their "
            "corresponding numerals, and devise a complete plan. Then, let's carry out "
            "the plan, calculate intermediate variables (pay attention to correct "
            "numerical calculation and commonsense), solve the problem step by step, "
            "and show the answer.",
        )

    def test_ps_plus_public_factory_and_frozen_state(self) -> None:
        state = FrozenAgentState(query="What is 2+2?", instruction="Return valid JSON.")
        method = create_intervention("plan.plan_and_solve.ps_plus_prompt")
        self.assertIsInstance(method, PlanAndSolvePSPlusPrompt)
        patch = method.propose(state, InterventionContext())
        updated = patch.apply(state)
        self.assertEqual(updated.instruction, "Return valid JSON.\n\n" + PLAN_AND_SOLVE_PLUS_TRIGGER)
        self.assertEqual(updated.query, state.query)
        self.assertEqual(updated.context, state.context)
        self.assertEqual(updated.budget, state.budget)
        self.assertEqual(state.instruction, "Return valid JSON.")
        self.assertEqual(patch.trace.counters["optimizer_model_calls"], 0)
        self.assertEqual(patch.metadata["variant"], "ps_plus")
        self.assertEqual(patch.metadata["prompt_id"], 302)
        self.assertFalse(patch.metadata["historical_native_available"])
        self.assertIn("ps_plus_prompt_302", patch.implementation_identity)
        self.assertEqual(
            patch.trace.events[0].payload["rendered_prompt"],
            render_plan_and_solve_prompt(state.query, trigger=PLAN_AND_SOLVE_PLUS_TRIGGER),
        )

    def test_paper_prompt_serialization(self) -> None:
        self.assertEqual(
            render_plan_and_solve_prompt("What is 2+2?"),
            f"Q:What is 2+2?\nA: {PLAN_AND_SOLVE_TRIGGER}",
        )

    def test_ps_prompt_appends_instruction_without_optimizer_call(self) -> None:
        state = FrozenAgentState(query="What is 2+2?", instruction="Be accurate.")
        patch = PlanAndSolvePSPrompt().propose(state, InterventionContext())
        updated = patch.apply(state)
        self.assertEqual(patch.operations[0].op, PatchOp.APPEND_INSTRUCTION)
        self.assertEqual(
            updated.instruction,
            f"Be accurate.\n\n{PLAN_AND_SOLVE_TRIGGER}",
        )
        self.assertEqual(patch.trace.counters["optimizer_model_calls"], 0)
        self.assertEqual(patch.metadata["prompt_id"], PLAN_AND_SOLVE_PROMPT_ID)
        self.assertEqual(patch.metadata["implementation_origin"], "clean_room_paper_spec")


if __name__ == "__main__":
    unittest.main()
