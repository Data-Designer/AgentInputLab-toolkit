from __future__ import annotations

import unittest

from agentinputlab.contracts import (
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    Message,
)
from agentinputlab.contracts.serialization import canonical_json
from agentinputlab.selectors.features import (
    VisibleFeatureConfig,
    VisibleStateFeatureExtractor,
)


class VisibleStateFeatureExtractorTests(unittest.TestCase):
    def test_is_deterministic_and_bound_to_the_input_state(self) -> None:
        state = FrozenAgentState(
            query="Compare the latest studies, then cite the best source.",
            instruction="Use the supplied context.",
            context=(ContextBlock("A short visible source."),),
            messages=(Message("user", "Please be precise."),),
            tool_names=("search",),
            budget=ExecutionBudget(max_model_calls=3),
        )
        extractor = VisibleStateFeatureExtractor()
        first = extractor.extract(state)
        second = extractor.extract(state)

        self.assertEqual(first, second)
        self.assertEqual(first.state_hash, state.state_hash)
        self.assertEqual(first.provenance["extractor_hash"], extractor.extractor_hash)
        self.assertEqual(first.provenance["schema_id"], extractor.feature_schema_id)
        self.assertEqual(first.provenance["schema_hash"], extractor.feature_schema_hash)
        self.assertGreater(first.features.multi_step_score, 0.0)
        self.assertGreater(first.features.retrieval_need_score, 0.0)

    def test_task_id_and_metadata_are_not_features_or_provenance(self) -> None:
        extractor = VisibleStateFeatureExtractor()
        first = extractor.extract(
            FrozenAgentState(
                query="Find recent evidence.",
                task_id="benchmark-secret-a",
                metadata={"benchmark_id": "hidden-a", "gold_answer": "secret-a"},
            )
        )
        second = extractor.extract(
            FrozenAgentState(
                query="Find recent evidence.",
                task_id="benchmark-secret-b",
                metadata={"benchmark_id": "hidden-b", "gold_answer": "secret-b"},
            )
        )

        self.assertEqual(first.features, second.features)
        self.assertEqual(first.provenance, second.provenance)
        serialized = canonical_json(first.provenance)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("benchmark-secret", serialized)
        self.assertNotIn("hidden-a", serialized)

    def test_context_pressure_uses_frozen_soft_limit(self) -> None:
        extractor = VisibleStateFeatureExtractor(VisibleFeatureConfig(context_soft_limit_tokens=4))
        observation = extractor.extract(
            FrozenAgentState(
                query="one two",
                context=(ContextBlock("three four five"),),
            )
        )
        self.assertEqual(observation.features.context_pressure, 1.0)

    def test_instruction_only_and_cjk_signals_are_supported(self) -> None:
        observation = VisibleStateFeatureExtractor().extract(
            FrozenAgentState(
                query="",
                instruction="首先查找最新论文\uff0c然后比较证据\u3002",
            )
        )
        self.assertGreater(observation.features.multi_step_score, 0.0)
        self.assertGreater(observation.features.retrieval_need_score, 0.0)

    def test_invalid_config_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            VisibleFeatureConfig(context_soft_limit_tokens=0)


if __name__ == "__main__":
    unittest.main()
