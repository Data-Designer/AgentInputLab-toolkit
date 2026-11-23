from __future__ import annotations

import unittest

from agentinputlab import FrozenAgentState, InterventionContext
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.contracts import SemanticFamily
from agentinputlab.metrics import intent_token_recall, summarize_pairs
from agentinputlab.runner import InterventionRunner, RunStatus
from agentinputlab.selectors import (
    NoOpIntervention,
    RuleSelector,
    SelectionFeatures,
)


class SelectorMetricBenchmarkTests(unittest.TestCase):
    def test_rule_selector_can_abstain(self) -> None:
        decision = RuleSelector().select(
            FrozenAgentState(query="hello"),
            SelectionFeatures(ambiguity_score=0.3, retrieval_need_score=0.2),
        )
        self.assertEqual(decision.family, SemanticFamily.NO_OP)

    def test_rule_selector_uses_margin(self) -> None:
        decision = RuleSelector(threshold=0.5, margin=0.1).select(
            FrozenAgentState(query="q"),
            SelectionFeatures(ambiguity_score=0.9, multi_step_score=0.5),
        )
        self.assertEqual(decision.family, SemanticFamily.CLARIFY)

    def test_rule_selector_honors_per_example_eligibility(self) -> None:
        decision = RuleSelector(threshold=0.5, margin=0.1).select(
            FrozenAgentState(query="q"),
            SelectionFeatures(ambiguity_score=0.95, retrieval_need_score=0.8),
            eligible_families=(SemanticFamily.ACQUIRE,),
        )
        self.assertEqual(decision.family, SemanticFamily.ACQUIRE)
        self.assertEqual(decision.selection_score, 0.8)

    def test_noop_runner_does_not_invoke_dependencies(self) -> None:
        state = FrozenAgentState(query="q")
        record = InterventionRunner().run(
            NoOpIntervention(),
            state,
            InterventionContext(),
        )
        self.assertEqual(record.status, RunStatus.NO_OP)
        self.assertEqual(record.input_state_hash, record.output_state_hash)

    def test_gold_is_not_projected_into_visible_state(self) -> None:
        example = BenchmarkExample(
            example_id="x",
            query="question",
            visible_metadata={"domain": "tiny"},
            evaluator_gold={"answer": "secret"},
        )
        state = example.visible_state()
        serialized = str(state.canonical_dict())
        self.assertNotIn("secret", serialized)
        self.assertNotIn("answer", serialized)

    def test_paired_metrics_and_intent_recall(self) -> None:
        summary = summarize_pairs((0.0, 1.0, 0.5), (1.0, 0.5, 0.5))
        self.assertAlmostEqual(summary.win_rate, 1 / 3)
        self.assertAlmostEqual(summary.harm_rate, 1 / 3)
        self.assertEqual(intent_token_recall("find red book", "find the red book now"), 1.0)


if __name__ == "__main__":
    unittest.main()
