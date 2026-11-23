from __future__ import annotations

import math
import unittest

from agentinputlab.contracts import FrozenAgentState, SemanticFamily
from agentinputlab.selectors import (
    RidgeUtilitySelector,
    SelectionFeatures,
    SelectorExample,
    calibrate_abstention,
    evaluate_selector,
    fit_ridge_selector,
    validate_selector_split_ids,
)


def example(index: int, *, split: str) -> SelectorExample:
    retrieval = index / 10
    multi_step = 1.0 - retrieval
    no_op = 0.4
    return SelectorExample(
        example_id=f"{split}-{index}",
        features=SelectionFeatures(
            multi_step_score=multi_step,
            retrieval_need_score=retrieval,
        ),
        utilities={
            SemanticFamily.NO_OP: no_op,
            SemanticFamily.ACQUIRE: no_op + retrieval - 0.25,
            SemanticFamily.DECOMPOSE: no_op + multi_step - 0.25,
        },
        split=split,
    )


class LearnedSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.releases = {
            SemanticFamily.ACQUIRE: "acquire.fixture",
            SemanticFamily.DECOMPOSE: "decompose.fixture",
        }

    def test_fit_select_calibrate_and_external_evaluate(self) -> None:
        selector = fit_ridge_selector(
            tuple(example(index, split="train") for index in range(11)),
            release_by_family=self.releases,
        )
        calibrated = calibrate_abstention(
            selector,
            tuple(example(index, split="calibration") for index in range(1, 10, 2)),
            gain_grid=(0.0, 0.2, 0.5),
            margin_grid=(0.0, 0.1),
            max_harm_rate=0.0,
        )
        evaluation = evaluate_selector(
            calibrated.selector,
            tuple(example(index, split="external_test") for index in range(11)),
            required_split="external_test",
        )
        self.assertEqual(calibrated.candidates_considered, 6)
        self.assertEqual(evaluation.harm_rate, 0.0)
        self.assertGreater(evaluation.mean_gain, 0.0)
        self.assertGreater(evaluation.coverage, 0.0)
        self.assertLess(evaluation.oracle_regret, 0.1)

    def test_training_and_evaluation_splits_fail_closed(self) -> None:
        train = tuple(example(index, split="train") for index in range(7))
        selector = fit_ridge_selector(train, release_by_family=self.releases)
        with self.assertRaisesRegex(ValueError, "split='train'"):
            fit_ridge_selector(
                (example(1, split="external_test"),),
                release_by_family=self.releases,
            )
        with self.assertRaisesRegex(ValueError, "must not be train"):
            evaluate_selector(selector, train, required_split="train")

    def test_feature_domain_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            SelectionFeatures(ambiguity_score=1.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            SelectionFeatures(ambiguity_score=math.nan)

    def test_zero_gain_tie_abstains_and_coefficients_are_frozen(self) -> None:
        source = {SemanticFamily.ACQUIRE: [0.0] * 6}
        selector = RidgeUtilitySelector(
            source,
            {SemanticFamily.ACQUIRE: "acquire.fixture"},
        )
        source[SemanticFamily.ACQUIRE][0] = 100.0
        decision = selector.select(
            FrozenAgentState(query="q"),
            SelectionFeatures(),
        )
        self.assertEqual(decision.family, SemanticFamily.NO_OP)
        self.assertEqual(len(selector.model_hash), 64)

    def test_evaluation_masks_ineligible_action_before_selection(self) -> None:
        selector = RidgeUtilitySelector(
            {SemanticFamily.ACQUIRE: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            {SemanticFamily.ACQUIRE: "acquire.fixture"},
        )
        row = SelectorExample(
            "external-1",
            SelectionFeatures(),
            {SemanticFamily.NO_OP: 0.0},
            "external_test",
        )
        evaluation = evaluate_selector(selector, (row,), required_split="external_test")
        self.assertEqual(evaluation.coverage, 0.0)
        self.assertEqual(evaluation.mean_gain, 0.0)

    def test_direct_construction_requires_valid_family_release_mapping(self) -> None:
        coefficients = {SemanticFamily.ACQUIRE: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
        with self.assertRaisesRegex(ValueError, "missing release mapping"):
            RidgeUtilitySelector(coefficients, {})
        with self.assertRaisesRegex(ValueError, "intervention-family"):
            RidgeUtilitySelector(coefficients, {"acquire": "acquire.fixture"})  # type: ignore[dict-item]

    def test_decision_exposes_score_not_probability_claim(self) -> None:
        selector = RidgeUtilitySelector(
            {SemanticFamily.ACQUIRE: (2.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            {SemanticFamily.ACQUIRE: "acquire.fixture"},
        )
        decision = selector.select(FrozenAgentState(query="q"), SelectionFeatures())
        self.assertEqual(decision.selection_score, 2.0)
        self.assertEqual(decision.confidence, decision.selection_score)

    def test_split_validator_rejects_overlap_and_binds_order_invariant_ids(self) -> None:
        train = tuple(example(index, split="train") for index in range(3))
        calibration = tuple(example(index, split="calibration") for index in range(3))
        external = tuple(example(index, split="external_test") for index in range(3))
        first = validate_selector_split_ids(
            {
                "train": train,
                "calibration": calibration,
                "external_test": external,
            }
        )
        second = validate_selector_split_ids(
            {
                "external_test": tuple(reversed(external)),
                "calibration": tuple(reversed(calibration)),
                "train": tuple(reversed(train)),
            }
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.provenance_hash), 64)
        self.assertEqual(first.split_sizes["train"], 3)
        with self.assertRaises((AttributeError, TypeError)):
            first.id_set_digests["train"] = "0" * 64  # type: ignore[index]

        overlapping_test = (
            SelectorExample(
                example_id=train[0].example_id,
                features=external[0].features,
                utilities=external[0].utilities,
                split="external_test",
            ),
            *external[1:],
        )
        with self.assertRaisesRegex(ValueError, "overlaps splits"):
            validate_selector_split_ids(
                {
                    "train": train,
                    "calibration": calibration,
                    "external_test": overlapping_test,
                }
            )


if __name__ == "__main__":
    unittest.main()
