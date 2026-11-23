from __future__ import annotations

import unittest
from dataclasses import replace

from agentinputlab.contracts import FrozenAgentState, SemanticFamily
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.selectors import SelectionFeatures
from agentinputlab.selectors.outcome_table import (
    EvaluatedArmOutcome,
    EvaluatedArmStatus,
    UtilityDefinition,
    build_selector_training_table,
)
from agentinputlab.selectors.runtime import FeatureObservation

_BACKEND = content_hash({"backend": "fixture"})
_PROFILE = content_hash({"profile": "controlled"})
_RELEASES = {
    SemanticFamily.ACQUIRE: "acquire.hyde.paper_reimplementation",
    SemanticFamily.DECOMPOSE: "plan.plan_and_solve.ps_prompt",
}


def _state_observation(query: str) -> tuple[FrozenAgentState, FeatureObservation]:
    state = FrozenAgentState(query=query)
    observation = FeatureObservation(
        SelectionFeatures(retrieval_need_score=0.8),
        state.state_hash,
        "fixture.visible",
        "v1",
        provenance={"schema_hash": content_hash({"schema": 1})},
    )
    return state, observation


def _row(
    state: FrozenAgentState,
    *,
    family: SemanticFamily,
    release_id: str,
    seed: int,
    value: float | None,
    split: str = "train",
    group_id: str = "group-a",
    status: EvaluatedArmStatus = EvaluatedArmStatus.SUCCEEDED,
) -> EvaluatedArmOutcome:
    return EvaluatedArmOutcome(
        example_id="source-a",
        group_id=group_id,
        state_hash=state.state_hash,
        split=split,
        benchmark_id="fixture.benchmark",
        family=family,
        release_id=release_id,
        seed=seed,
        backend_hash=_BACKEND,
        comparison_profile_hash=_PROFILE,
        status=status,
        metrics={} if value is None else {"valid_success": value},
        cost=0.1 if family is not SemanticFamily.NO_OP else 0.0,
        risk=0.2 if family is not SemanticFamily.NO_OP else 0.0,
    )


class SelectorOutcomeTableTests(unittest.TestCase):
    def test_aggregates_replicates_and_does_not_impute_structural_actions(self) -> None:
        state, observation = _state_observation("Find evidence.")
        rows = (
            _row(
                state, family=SemanticFamily.NO_OP, release_id="baseline.no_op", seed=1, value=0.2
            ),
            _row(
                state, family=SemanticFamily.NO_OP, release_id="baseline.no_op", seed=2, value=0.4
            ),
            _row(
                state,
                family=SemanticFamily.ACQUIRE,
                release_id=_RELEASES[SemanticFamily.ACQUIRE],
                seed=1,
                value=0.8,
            ),
            _row(
                state,
                family=SemanticFamily.ACQUIRE,
                release_id=_RELEASES[SemanticFamily.ACQUIRE],
                seed=2,
                value=1.0,
            ),
            _row(
                state,
                family=SemanticFamily.DECOMPOSE,
                release_id=_RELEASES[SemanticFamily.DECOMPOSE],
                seed=1,
                value=None,
                status=EvaluatedArmStatus.STRUCTURAL_INELIGIBLE,
            ),
            _row(
                state,
                family=SemanticFamily.DECOMPOSE,
                release_id=_RELEASES[SemanticFamily.DECOMPOSE],
                seed=2,
                value=None,
                status=EvaluatedArmStatus.STRUCTURAL_INELIGIBLE,
            ),
        )
        table = build_selector_training_table(
            rows,
            {state.state_hash: observation},
            utility_definition=UtilityDefinition("valid_success"),
            representative_by_family=_RELEASES,
        )

        self.assertEqual(len(table.examples), 1)
        example = table.examples[0]
        self.assertAlmostEqual(example.utilities[SemanticFamily.NO_OP], 0.3)
        self.assertAlmostEqual(example.utilities[SemanticFamily.ACQUIRE], 0.9)
        self.assertNotIn(SemanticFamily.DECOMPOSE, example.utilities)
        self.assertEqual(len(table.table_hash), 64)
        reordered = build_selector_training_table(
            tuple(reversed(rows)),
            {state.state_hash: observation},
            utility_definition=UtilityDefinition("valid_success"),
            representative_by_family=_RELEASES,
        )
        self.assertEqual(reordered.table_hash, table.table_hash)

    def test_algorithm_failure_uses_preregistered_utility(self) -> None:
        state, observation = _state_observation("Find evidence.")
        rows = (
            _row(
                state, family=SemanticFamily.NO_OP, release_id="baseline.no_op", seed=1, value=0.4
            ),
            _row(
                state,
                family=SemanticFamily.ACQUIRE,
                release_id=_RELEASES[SemanticFamily.ACQUIRE],
                seed=1,
                value=None,
                status=EvaluatedArmStatus.ALGORITHM_FAILURE,
            ),
        )
        table = build_selector_training_table(
            rows,
            {state.state_hash: observation},
            utility_definition=UtilityDefinition(
                "valid_success",
                algorithm_failure_utility=-1.0,
            ),
            representative_by_family=_RELEASES,
        )
        self.assertEqual(table.examples[0].utilities[SemanticFamily.ACQUIRE], -1.0)

    def test_infrastructure_missing_and_missing_noop_fail_closed(self) -> None:
        state, observation = _state_observation("Find evidence.")
        treatment = _row(
            state,
            family=SemanticFamily.ACQUIRE,
            release_id=_RELEASES[SemanticFamily.ACQUIRE],
            seed=1,
            value=0.8,
        )
        with self.assertRaisesRegex(ValueError, "explicit NO_OP"):
            build_selector_training_table(
                (treatment,),
                {state.state_hash: observation},
                utility_definition=UtilityDefinition("valid_success"),
                representative_by_family=_RELEASES,
            )
        missing = replace(
            treatment,
            status=EvaluatedArmStatus.INFRASTRUCTURE_MISSING,
            metrics={},
        )
        with self.assertRaisesRegex(ValueError, "infrastructure-missing"):
            build_selector_training_table(
                (missing,),
                {state.state_hash: observation},
                utility_definition=UtilityDefinition("valid_success"),
                representative_by_family=_RELEASES,
            )

    def test_group_cross_split_is_rejected_even_with_distinct_examples(self) -> None:
        first_state, first_observation = _state_observation("First state.")
        second_state, second_observation = _state_observation("Second state.")
        rows = (
            _row(
                first_state,
                family=SemanticFamily.NO_OP,
                release_id="baseline.no_op",
                seed=1,
                value=0.2,
                split="train",
                group_id="shared-template",
            ),
            replace(
                _row(
                    second_state,
                    family=SemanticFamily.NO_OP,
                    release_id="baseline.no_op",
                    seed=1,
                    value=0.2,
                    split="calibration",
                    group_id="shared-template",
                ),
                example_id="source-b",
            ),
        )
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            build_selector_training_table(
                rows,
                {
                    first_state.state_hash: first_observation,
                    second_state.state_hash: second_observation,
                },
                utility_definition=UtilityDefinition("valid_success"),
                representative_by_family=_RELEASES,
            )

    def test_seed_and_representative_drift_are_rejected(self) -> None:
        state, observation = _state_observation("Find evidence.")
        baseline = _row(
            state,
            family=SemanticFamily.NO_OP,
            release_id="baseline.no_op",
            seed=1,
            value=0.2,
        )
        treatment = _row(
            state,
            family=SemanticFamily.ACQUIRE,
            release_id=_RELEASES[SemanticFamily.ACQUIRE],
            seed=2,
            value=0.8,
        )
        with self.assertRaisesRegex(ValueError, "seed set differs"):
            build_selector_training_table(
                (baseline, treatment),
                {state.state_hash: observation},
                utility_definition=UtilityDefinition("valid_success"),
                representative_by_family=_RELEASES,
            )
        with self.assertRaisesRegex(ValueError, "frozen representative"):
            build_selector_training_table(
                (baseline, replace(treatment, seed=1, release_id="acquire.other")),
                {state.state_hash: observation},
                utility_definition=UtilityDefinition("valid_success"),
                representative_by_family=_RELEASES,
            )


if __name__ == "__main__":
    unittest.main()
