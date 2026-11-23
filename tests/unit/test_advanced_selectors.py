from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from dataclasses import replace

from agentinputlab.contracts import FrozenAgentState, SemanticFamily
from agentinputlab.selectors import (
    CostRiskAwareSelector,
    FrozenGBDTOutcomeModel,
    FrozenRegressionTree,
    GBDTTLearnerSelector,
    KNNUtilitySelector,
    LogisticTLearnerSelector,
    RidgeTLearnerSelector,
    RidgeUtilitySelector,
    RuleSelector,
    SelectionFeatures,
    Selector,
    SelectorExample,
    SelectorKind,
    calibrate_cost_risk_guardrails,
    calibrate_selector_abstention,
    fit_cost_risk_selector,
    fit_gbdt_t_learner,
    fit_knn_selector,
    fit_logistic_t_learner,
    fit_ridge_t_learner,
    get_selector_card,
    list_selector_cards,
)

_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
_RELEASES = {
    SemanticFamily.ACQUIRE: "acquire.hyde.paper_reimplementation",
    SemanticFamily.DECOMPOSE: "plan.plan_and_solve.ps_prompt",
}


def utility_example(index: int, *, split: str) -> SelectorExample:
    retrieval = index / 10
    multi_step = 1.0 - retrieval
    return SelectorExample(
        example_id=f"{split}-{index}",
        features=SelectionFeatures(
            retrieval_need_score=retrieval,
            multi_step_score=multi_step,
        ),
        utilities={
            SemanticFamily.NO_OP: 0.2,
            SemanticFamily.ACQUIRE: 0.1 + 0.9 * retrieval,
            SemanticFamily.DECOMPOSE: 1.0 - 0.8 * retrieval,
        },
        split=split,
        costs={
            SemanticFamily.ACQUIRE: 10.0,
            SemanticFamily.DECOMPOSE: 1.0,
        },
        risks={
            SemanticFamily.ACQUIRE: 0.9,
            SemanticFamily.DECOMPOSE: 0.1,
        },
    )


def binary_example(index: int, *, split: str) -> SelectorExample:
    retrieval = index / 20
    multi_step = 1.0 - retrieval
    return SelectorExample(
        example_id=f"binary-{split}-{index}",
        features=SelectionFeatures(
            retrieval_need_score=retrieval,
            multi_step_score=multi_step,
        ),
        utilities={
            SemanticFamily.NO_OP: 0.0,
            SemanticFamily.ACQUIRE: float(retrieval >= 0.55),
            SemanticFamily.DECOMPOSE: float(multi_step >= 0.55),
        },
        split=split,
    )


def manual_gbdt() -> GBDTTLearnerSelector:
    leaf = FrozenRegressionTree(
        children_left=(-1,),
        children_right=(-1,),
        features=(-2,),
        thresholds=(-2.0,),
        values=(0.0,),
    )
    acquire = FrozenRegressionTree(
        children_left=(1, -1, -1),
        children_right=(2, -1, -1),
        features=(2, -2, -2),
        thresholds=(0.5, -2.0, -2.0),
        values=(0.0, 0.1, 0.9),
    )
    decompose = FrozenRegressionTree(
        children_left=(1, -1, -1),
        children_right=(2, -1, -1),
        features=(2, -2, -2),
        thresholds=(0.5, -2.0, -2.0),
        values=(0.0, 0.8, 0.1),
    )
    return GBDTTLearnerSelector(
        outcome_models={
            SemanticFamily.NO_OP: FrozenGBDTOutcomeModel(0.2, 1.0, (leaf,)),
            SemanticFamily.ACQUIRE: FrozenGBDTOutcomeModel(0.0, 1.0, (acquire,)),
            SemanticFamily.DECOMPOSE: FrozenGBDTOutcomeModel(0.0, 1.0, (decompose,)),
        },
        release_by_family=_RELEASES,
        sklearn_version="fixture",
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        random_state=0,
    )


class KNNSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = tuple(utility_example(index, split="train") for index in range(11))

    def test_fit_select_eligibility_noop_and_identity_independence(self) -> None:
        selector = fit_knn_selector(self.train, release_by_family=_RELEASES, k=3)
        features = SelectionFeatures(retrieval_need_score=0.9, multi_step_score=0.1)
        first = selector.select(FrozenAgentState(query="one"), features)
        second = selector.select(FrozenAgentState(query="different hidden identity"), features)
        self.assertIs(first.family, SemanticFamily.ACQUIRE)
        self.assertEqual(first, second)
        eligible = selector.select(
            FrozenAgentState(query="q"),
            features,
            eligible_families=(SemanticFamily.DECOMPOSE,),
        )
        self.assertIs(eligible.family, SemanticFamily.DECOMPOSE)
        blocked = selector.select(
            FrozenAgentState(query="q"),
            features,
            eligible_families=(),
        )
        self.assertIs(blocked.family, SemanticFamily.NO_OP)

    def test_fit_is_order_stable_frozen_and_train_only(self) -> None:
        first = fit_knn_selector(self.train, release_by_family=_RELEASES, k=3)
        second = fit_knn_selector(tuple(reversed(self.train)), release_by_family=_RELEASES, k=3)
        self.assertEqual(first.model_hash, second.model_hash)
        self.assertEqual(len(first.model_hash), 64)
        with self.assertRaises((AttributeError, TypeError)):
            first.training_rows[0].gains[SemanticFamily.ACQUIRE] = 99.0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "split='train'"):
            fit_knn_selector(
                (utility_example(1, split="calibration"),),
                release_by_family=_RELEASES,
            )

    def test_generic_calibration_is_calibration_only(self) -> None:
        selector = fit_knn_selector(self.train, release_by_family=_RELEASES, k=3)
        calibration = tuple(
            utility_example(index, split="calibration") for index in range(1, 10, 2)
        )
        result = calibrate_selector_abstention(
            selector,
            calibration,
            gain_grid=(0.0, 0.2),
            margin_grid=(0.0, 0.1),
            max_harm_rate=0.25,
        )
        self.assertEqual(result.candidates_considered, 4)
        self.assertIsInstance(result.selector, KNNUtilitySelector)
        with self.assertRaisesRegex(ValueError, "split='calibration'"):
            calibrate_selector_abstention(
                selector,
                self.train,
                gain_grid=(0.0,),
                margin_grid=(0.0,),
                max_harm_rate=1.0,
            )


class TLearnerTests(unittest.TestCase):
    def test_ridge_t_learner_is_a_distinct_potential_outcome_model(self) -> None:
        train = tuple(utility_example(index, split="train") for index in range(11))
        selector = fit_ridge_t_learner(train, release_by_family=_RELEASES)
        decision = selector.select(
            FrozenAgentState(query="q"),
            SelectionFeatures(retrieval_need_score=0.9, multi_step_score=0.1),
        )
        self.assertIsInstance(selector, RidgeTLearnerSelector)
        self.assertIn(SemanticFamily.NO_OP, selector.outcome_coefficients)
        self.assertIs(decision.family, SemanticFamily.ACQUIRE)
        self.assertEqual(len(selector.model_hash), 64)
        with self.assertRaisesRegex(ValueError, "split='train'"):
            fit_ridge_t_learner(
                (utility_example(1, split="external_test"),),
                release_by_family=_RELEASES,
            )

    def test_logistic_t_learner_handles_binary_success_and_eligibility(self) -> None:
        train = tuple(binary_example(index, split="train") for index in range(21))
        selector = fit_logistic_t_learner(
            train,
            release_by_family=_RELEASES,
            iterations=800,
        )
        features = SelectionFeatures(retrieval_need_score=0.9, multi_step_score=0.1)
        decision = selector.select(FrozenAgentState(query="q"), features)
        self.assertIsInstance(selector, LogisticTLearnerSelector)
        self.assertIs(decision.family, SemanticFamily.ACQUIRE)
        masked = selector.select(
            FrozenAgentState(query="q"),
            features,
            eligible_families=(),
        )
        self.assertIs(masked.family, SemanticFamily.NO_OP)
        with self.assertRaisesRegex(ValueError, r"probabilities in \[0, 1\]"):
            fit_logistic_t_learner(
                (
                    SelectorExample(
                        "bad",
                        SelectionFeatures(),
                        {
                            SemanticFamily.NO_OP: 0.0,
                            SemanticFamily.ACQUIRE: 2.0,
                            SemanticFamily.DECOMPOSE: 0.0,
                        },
                        "train",
                    ),
                ),
                release_by_family=_RELEASES,
            )


class GBDTSelectorTests(unittest.TestCase):
    def test_frozen_artifact_roundtrip_hash_eligibility_and_abstention(self) -> None:
        selector = manual_gbdt()
        artifact = json.loads(json.dumps(selector.to_artifact()))
        restored = GBDTTLearnerSelector.from_artifact(artifact)
        self.assertEqual(selector.model_hash, restored.model_hash)
        self.assertEqual(selector.to_artifact(), restored.to_artifact())
        high_retrieval = SelectionFeatures(retrieval_need_score=0.9)
        self.assertIs(
            restored.select(FrozenAgentState(query="q"), high_retrieval).family,
            SemanticFamily.ACQUIRE,
        )
        self.assertIs(
            restored.select(
                FrozenAgentState(query="q"),
                high_retrieval,
                eligible_families=(SemanticFamily.DECOMPOSE,),
            ).family,
            SemanticFamily.NO_OP,
        )
        self.assertIs(
            replace(restored, min_gain=1.0)
            .select(FrozenAgentState(query="q"), high_retrieval)
            .family,
            SemanticFamily.NO_OP,
        )

    def test_invalid_artifact_and_frozen_tree_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid GBDT selector artifact"):
            GBDTTLearnerSelector.from_artifact({"schema_version": 99})
        selector = manual_gbdt()
        with self.assertRaises((AttributeError, TypeError)):
            selector.outcome_models[SemanticFamily.NO_OP] = object()  # type: ignore[index]

    def test_artifact_decoder_rejects_coercion_and_noncanonical_json(self) -> None:
        source = manual_gbdt().to_artifact()
        corruptions = []

        float_integer = copy.deepcopy(source)
        float_integer["fit"]["n_estimators"] = 1.9
        corruptions.append(float_integer)

        boolean_integer = copy.deepcopy(source)
        boolean_integer["fit"]["random_state"] = True
        corruptions.append(boolean_integer)

        boolean_child = copy.deepcopy(source)
        boolean_child["outcome_models"]["no_op"]["trees"][0]["children_left"] = [True]
        corruptions.append(boolean_child)

        string_number = copy.deepcopy(source)
        string_number["outcome_models"]["no_op"]["trees"][0]["thresholds"] = ["-2.0"]
        corruptions.append(string_number)

        tuple_array = copy.deepcopy(source)
        tuple_array["outcome_models"]["no_op"]["trees"][0]["values"] = (0.0,)
        corruptions.append(tuple_array)

        coerced_release = copy.deepcopy(source)
        coerced_release["release_by_family"]["acquire"] = 123
        corruptions.append(coerced_release)

        extra_key = copy.deepcopy(source)
        extra_key["unexpected"] = None
        corruptions.append(extra_key)

        for artifact in corruptions:
            with (
                self.subTest(artifact=artifact),
                self.assertRaisesRegex(
                    ValueError,
                    "^invalid GBDT selector artifact$",
                ),
            ):
                GBDTTLearnerSelector.from_artifact(artifact)

    def test_tree_decoder_rejects_disconnected_and_wrong_json_types(self) -> None:
        malformed = {
            "children_left": [1, -1, -1, -1],
            "children_right": [2, -1, -1, -1],
            "features": [0, -2, -2, -2],
            "thresholds": [0.5, -2.0, -2.0, -2.0],
            "values": [0.0, 0.0, 1.0, 2.0],
        }
        with self.assertRaisesRegex(ValueError, "invalid frozen regression tree artifact"):
            FrozenRegressionTree.from_artifact(malformed)
        wrong_array = manual_gbdt().to_artifact()["outcome_models"]["no_op"]["trees"][0]
        wrong_array["children_left"] = (False,)
        with self.assertRaisesRegex(ValueError, "invalid frozen regression tree artifact"):
            FrozenRegressionTree.from_artifact(wrong_array)

    @unittest.skipUnless(_HAS_SKLEARN, "install agentinputlab[selectors]")
    def test_optional_sklearn_fit_detaches_and_serializes(self) -> None:
        train = tuple(utility_example(index, split="train") for index in range(11))
        selector = fit_gbdt_t_learner(
            train,
            release_by_family=_RELEASES,
            n_estimators=5,
            max_depth=1,
            min_samples_leaf=1,
        )
        restored = GBDTTLearnerSelector.from_artifact(selector.to_artifact())
        self.assertEqual(selector.model_hash, restored.model_hash)
        self.assertEqual(selector.train_example_count, len(train))
        self.assertEqual(len(selector.train_id_set_digest), 64)
        self.assertIsInstance(restored, Selector)

    @unittest.skipIf(_HAS_SKLEARN, "missing-dependency branch only")
    def test_optional_fit_has_actionable_missing_dependency_error(self) -> None:
        train = tuple(utility_example(index, split="train") for index in range(3))
        with self.assertRaisesRegex(ImportError, r"agentinputlab\[selectors\]"):
            fit_gbdt_t_learner(train, release_by_family=_RELEASES)


class CostRiskPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = RidgeUtilitySelector(
            {
                SemanticFamily.ACQUIRE: (0.8, 0.0, 0.0, 0.0, 0.0, 0.0),
                SemanticFamily.DECOMPOSE: (0.7, 0.0, 0.0, 0.0, 0.0, 0.0),
            },
            _RELEASES,
        )
        self.train = tuple(utility_example(index, split="train") for index in range(4))

    def test_wrapper_is_a_guardrail_policy_not_an_independent_predictor(self) -> None:
        selector = fit_cost_risk_selector(
            self.base,
            self.train,
            release_by_family=_RELEASES,
            max_cost=5.0,
            max_risk=0.5,
        )
        decision = selector.select(FrozenAgentState(query="q"), SelectionFeatures())
        self.assertIsInstance(selector, CostRiskAwareSelector)
        self.assertIs(selector.base_selector, self.base)
        self.assertIs(decision.family, SemanticFamily.DECOMPOSE)
        self.assertEqual(len(selector.model_hash), 64)
        all_unsafe = replace(selector, max_cost=0.0, max_risk=0.0)
        self.assertIs(
            all_unsafe.select(FrozenAgentState(query="q"), SelectionFeatures()).family,
            SemanticFamily.NO_OP,
        )

    def test_wrapper_respects_base_abstention_and_training_boundary(self) -> None:
        abstaining = replace(self.base, min_gain=1.0)
        selector = fit_cost_risk_selector(
            abstaining,
            self.train,
            release_by_family=_RELEASES,
            max_cost=20.0,
            max_risk=1.0,
        )
        self.assertIs(
            selector.select(FrozenAgentState(query="q"), SelectionFeatures()).family,
            SemanticFamily.NO_OP,
        )
        with self.assertRaisesRegex(ValueError, "split='train'"):
            fit_cost_risk_selector(
                self.base,
                (utility_example(1, split="calibration"),),
                release_by_family=_RELEASES,
                max_cost=20.0,
                max_risk=1.0,
            )

    def test_wrapper_rejects_generic_rule_scores_as_utility_gains(self) -> None:
        with self.assertRaisesRegex(ValueError, "UtilityGainProvider"):
            fit_cost_risk_selector(
                RuleSelector(),
                self.train,
                release_by_family=_RELEASES,
                max_cost=20.0,
                max_risk=1.0,
            )

    def test_all_guardrails_are_frozen_on_calibration_only(self) -> None:
        selector = fit_cost_risk_selector(
            self.base,
            self.train,
            release_by_family=_RELEASES,
            max_cost=20.0,
            max_risk=1.0,
        )
        calibration = tuple(utility_example(index, split="calibration") for index in range(4))
        result = calibrate_cost_risk_guardrails(
            selector,
            calibration,
            max_cost_grid=(5.0, 20.0),
            max_risk_grid=(0.5, 1.0),
            gain_grid=(0.0,),
            margin_grid=(0.0,),
            max_harm_rate=1.0,
        )
        self.assertEqual(result.candidates_considered, 4)
        self.assertIsInstance(result.selector, CostRiskAwareSelector)

    def test_guardrail_grids_are_canonical_and_ties_prefer_stricter_caps(self) -> None:
        selector = fit_cost_risk_selector(
            replace(self.base, min_gain=1.0),
            self.train,
            release_by_family=_RELEASES,
            max_cost=20.0,
            max_risk=1.0,
        )
        calibration = tuple(utility_example(index, split="calibration") for index in range(4))
        result = calibrate_cost_risk_guardrails(
            selector,
            calibration,
            max_cost_grid=(20.0, 5.0, 20.0),
            max_risk_grid=(1.0, 0.5, 1.0),
            gain_grid=(0.0, 0.0),
            margin_grid=(0.0, 0.0),
            max_harm_rate=1.0,
        )
        self.assertEqual(result.candidates_considered, 4)
        self.assertEqual(result.selector.max_cost, 5.0)
        self.assertEqual(result.selector.max_risk, 0.5)
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            calibrate_cost_risk_guardrails(
                selector,
                calibration,
                max_cost_grid=(True,),
                max_risk_grid=(0.5,),
                gain_grid=(0.0,),
                margin_grid=(0.0,),
                max_harm_rate=1.0,
            )


class PublicContractTests(unittest.TestCase):
    def test_public_catalog_has_stable_predictor_and_guardrail_metadata(self) -> None:
        cards = list_selector_cards()
        self.assertGreaterEqual(len(cards), 5)
        self.assertEqual(len({card.selector_id for card in cards}), len(cards))
        self.assertEqual(
            get_selector_card("gbdt_t_learner.v1").optional_dependency,
            "scikit-learn",
        )
        self.assertIs(
            get_selector_card("cost_risk_guardrail.v1").kind,
            SelectorKind.GUARDRAIL,
        )
        self.assertGreaterEqual(len(list_selector_cards(kind=SelectorKind.PREDICTOR)), 5)

    def test_five_public_types_are_frozen_hashable_selectors(self) -> None:
        train = tuple(utility_example(index, split="train") for index in range(11))
        selectors = (
            RuleSelector(),
            fit_ridge_t_learner(train, release_by_family=_RELEASES),
            manual_gbdt(),
            fit_knn_selector(train, release_by_family=_RELEASES),
            fit_cost_risk_selector(
                RidgeUtilitySelector(
                    {
                        SemanticFamily.ACQUIRE: (0.8, 0, 0, 0, 0, 0),
                        SemanticFamily.DECOMPOSE: (0.7, 0, 0, 0, 0, 0),
                    },
                    _RELEASES,
                ),
                train,
                release_by_family=_RELEASES,
                max_cost=20.0,
                max_risk=1.0,
            ),
        )
        self.assertTrue(all(isinstance(selector, Selector) for selector in selectors))
        self.assertTrue(all(len(selector.model_hash) == 64 for selector in selectors))

    def test_selector_example_rejects_invalid_cost_and_risk_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, "costs"):
            SelectorExample(
                "bad-cost",
                SelectionFeatures(),
                {SemanticFamily.NO_OP: 0.0},
                "train",
                costs={SemanticFamily.ACQUIRE: -1.0},
            )
        with self.assertRaisesRegex(ValueError, "risks"):
            SelectorExample(
                "bad-risk",
                SelectionFeatures(),
                {SemanticFamily.NO_OP: 0.0},
                "train",
                risks={SemanticFamily.ACQUIRE: 1.1},
            )


if __name__ == "__main__":
    unittest.main()
