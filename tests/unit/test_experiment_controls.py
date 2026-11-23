from __future__ import annotations

import unittest

from agentinputlab.contracts import (
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    SemanticFamily,
)
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.experiments import CostMatchedNoOpIntervention


class _Delegate:
    method_id = "fixture_optimizer"
    release_id = "fixture.optimizer.v1"
    family = SemanticFamily.REFORMULATE

    def propose(self, state, context):
        del context
        patch = InputPatch.no_op(
            state,
            method_id=self.method_id,
            release_id=self.release_id,
        )
        return InputPatch(
            method_id=patch.method_id,
            release_id=patch.release_id,
            source_state_hash=patch.source_state_hash,
            operations=patch.operations,
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                counters={"model_calls": 1, "input_tokens": 10, "output_tokens": 4},
            ),
            implementation_identity="fixture",
            profile_kind="controlled_adapter",
        )


class ExperimentControlTests(unittest.TestCase):
    def test_cost_shadow_discards_patch_and_retains_usage_counters(self) -> None:
        state = FrozenAgentState(task_id="task", query="original")
        shadow = CostMatchedNoOpIntervention(_Delegate())
        patch = shadow.propose(state, InterventionContext(seed=3))
        self.assertEqual(patch.operations[0].op, PatchOp.NO_OP)
        self.assertEqual(patch.apply(state).state_hash, state.state_hash)
        self.assertEqual(patch.trace.counters["model_calls"], 1)
        self.assertEqual(patch.trace.counters["output_tokens"], 4)
        self.assertTrue(patch.metadata["visible_input_unchanged"])
        self.assertEqual(patch.metadata["delegate_release_id"], "fixture.optimizer.v1")

    def test_cost_shadow_rejects_a_noop_delegate(self) -> None:
        delegate = _Delegate()
        delegate.family = SemanticFamily.NO_OP
        with self.assertRaisesRegex(ValidationError, "real intervention"):
            CostMatchedNoOpIntervention(delegate)


if __name__ == "__main__":
    unittest.main()
