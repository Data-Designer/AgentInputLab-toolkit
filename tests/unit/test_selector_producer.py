from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace

from agentinputlab.contracts import SemanticFamily
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.selectors import (
    EvaluatedArmOutcome,
    EvaluatedArmStatus,
    FeatureObservation,
    SelectionFeatures,
    SelectorArtifactStage,
    SelectorArtifactStore,
    SelectorDiscoveryFreeze,
    SelectorDiscoverySource,
    UtilityDefinition,
    build_selector_policy_manifest,
    build_selector_training_table,
    calibrate_selector_bundle,
    create_fitted_selector_bundle,
    freeze_selector_bundle,
)

_ACQUIRE = "acquire.hyde.paper_reimplementation"
_RELEASES = {SemanticFamily.ACQUIRE: _ACQUIRE}
_BACKEND = content_hash({"backend": "producer-fixture"})
_PROFILE = content_hash({"profile": "controlled"})
_EXTRACTOR_HASH = content_hash({"extractor": "fixture-v1"})
_SCHEMA_HASH = content_hash({"schema": "family_defect_features.v1"})
_FULL_RELEASES = {
    SemanticFamily.REFORMULATE: "rewrite.rrr.prompt_external",
    SemanticFamily.DECOMPOSE: "plan.plan_and_solve.ps_prompt",
    SemanticFamily.ACQUIRE: _ACQUIRE,
    SemanticFamily.CLARIFY: "clarify.ask_or_assume.intent_monitor_component",
    SemanticFamily.CURATE: "curate.llmlingua2.xlm_roberta_official",
}


def _discovery_freeze() -> SelectorDiscoveryFreeze:
    observations: list[FeatureObservation] = []
    outcomes: list[EvaluatedArmOutcome] = []
    for index in range(10):
        split = "train" if index < 6 else "calibration"
        state_hash = content_hash({"visible-state": index})
        features = SelectionFeatures(
            ambiguity_score=(index % 3) / 2,
            multi_step_score=(index % 4) / 3,
            retrieval_need_score=index / 9,
            context_pressure=(9 - index) / 9,
            reformulation_need_score=(index % 2),
        )
        observations.append(
            FeatureObservation(
                features=features,
                state_hash=state_hash,
                extractor_id="visible_state_fixture",
                extractor_revision="v1",
                provenance={
                    "extractor_hash": _EXTRACTOR_HASH,
                    "schema_id": "family_defect_features.v1",
                    "schema_hash": _SCHEMA_HASH,
                },
            )
        )
        baseline = 0.2 if index % 2 == 0 else 0.4
        treatment = 0.9 if index >= 3 else 0.1
        common = {
            "example_id": f"source-{index}",
            "group_id": "group_"
            + content_hash({"benchmark_id": "fixture.benchmark", "group_id": f"group-{index}"}),
            "state_hash": state_hash,
            "split": split,
            "benchmark_id": "fixture.benchmark",
            "seed": 7,
            "backend_hash": _BACKEND,
            "comparison_profile_hash": _PROFILE,
            "status": EvaluatedArmStatus.SUCCEEDED,
        }
        outcomes.extend(
            (
                EvaluatedArmOutcome(
                    **common,
                    family=SemanticFamily.NO_OP,
                    release_id="baseline.no_op",
                    metrics={"valid_success": baseline},
                ),
                EvaluatedArmOutcome(
                    **common,
                    family=SemanticFamily.ACQUIRE,
                    release_id=_ACQUIRE,
                    metrics={"valid_success": treatment},
                    cost=0.25 + index / 100,
                    risk=0.1 + index / 100,
                ),
            )
        )
    utility = UtilityDefinition("valid_success")
    table = build_selector_training_table(
        outcomes,
        {item.state_hash: item for item in observations},
        utility_definition=utility,
        representative_by_family=_RELEASES,
    )
    return SelectorDiscoveryFreeze(
        revision="producer-fixture-v1",
        source_runs=(
            SelectorDiscoverySource(
                study_id="producer.fixture",
                identity_hash=content_hash({"run": "fixture"}),
                manifest_hash=content_hash({"manifest": "fixture"}),
                plan_hash=content_hash({"plan": "fixture"}),
                result_set_hash=content_hash({"results": "fixture"}),
            ),
        ),
        observations=tuple(observations),
        outcomes=tuple(outcomes),
        training_table=table,
        utility_definition=utility,
        representative_by_family=_RELEASES,
        extractor_id="visible_state_fixture",
        extractor_revision="v1",
        extractor_hash=_EXTRACTOR_HASH,
        feature_schema_id="family_defect_features.v1",
        feature_schema_hash=_SCHEMA_HASH,
        comparison_profile_hash=_PROFILE,
    )


def _full_five_discovery_freeze() -> SelectorDiscoveryFreeze:
    observations: list[FeatureObservation] = []
    outcomes: list[EvaluatedArmOutcome] = []
    ordered_families = tuple(sorted(_FULL_RELEASES, key=lambda family: family.value))
    for index in range(8):
        split = "train" if index < 4 else "calibration"
        state_hash = content_hash({"full-five-visible-state": index})
        observations.append(
            FeatureObservation(
                features=SelectionFeatures(
                    ambiguity_score=index / 8,
                    multi_step_score=(index % 3) / 2,
                    retrieval_need_score=(8 - index) / 8,
                    context_pressure=float(index % 2),
                    reformulation_need_score=(index % 4) / 3,
                ),
                state_hash=state_hash,
                extractor_id="visible_state_fixture",
                extractor_revision="v1",
                provenance={
                    "extractor_hash": _EXTRACTOR_HASH,
                    "schema_id": "family_defect_features.v1",
                    "schema_hash": _SCHEMA_HASH,
                },
            )
        )
        common = {
            "example_id": f"full-source-{index}",
            "group_id": "group_"
            + content_hash(
                {
                    "benchmark_id": "fixture.full-five",
                    "group_id": f"full-group-{index}",
                }
            ),
            "state_hash": state_hash,
            "split": split,
            "benchmark_id": "fixture.full-five",
            "seed": 7,
            "backend_hash": _BACKEND,
            "comparison_profile_hash": _PROFILE,
            "status": EvaluatedArmStatus.SUCCEEDED,
        }
        outcomes.append(
            EvaluatedArmOutcome(
                **common,
                family=SemanticFamily.NO_OP,
                release_id="baseline.no_op",
                metrics={"valid_success": 0.2},
            )
        )
        for family_index, family in enumerate(ordered_families):
            outcomes.append(
                EvaluatedArmOutcome(
                    **common,
                    family=family,
                    release_id=_FULL_RELEASES[family],
                    metrics={"valid_success": 0.3 + 0.1 * family_index},
                )
            )
    utility = UtilityDefinition("valid_success")
    table = build_selector_training_table(
        outcomes,
        {item.state_hash: item for item in observations},
        utility_definition=utility,
        representative_by_family=_FULL_RELEASES,
    )
    return SelectorDiscoveryFreeze(
        revision="full-five-producer-v1",
        source_runs=(
            SelectorDiscoverySource(
                study_id="producer.full-five",
                identity_hash=content_hash({"run": "full-five"}),
                manifest_hash=content_hash({"manifest": "full-five"}),
                plan_hash=content_hash({"plan": "full-five"}),
                result_set_hash=content_hash({"results": "full-five"}),
            ),
        ),
        observations=tuple(observations),
        outcomes=tuple(outcomes),
        training_table=table,
        utility_definition=utility,
        representative_by_family=_FULL_RELEASES,
        extractor_id="visible_state_fixture",
        extractor_revision="v1",
        extractor_hash=_EXTRACTOR_HASH,
        feature_schema_id="family_defect_features.v1",
        feature_schema_hash=_SCHEMA_HASH,
        comparison_profile_hash=_PROFILE,
    )


_FIT_CONFIGS = {
    "rule": {"threshold": 0.5, "margin": 0.0},
    "ridge_utility": {"ridge": 0.01},
    "ridge_t_learner": {"ridge": 0.01},
    "logistic_t_learner": {
        "l2": 0.01,
        "learning_rate": 0.1,
        "iterations": 30,
    },
    "gbdt_t_learner": {
        "n_estimators": 2,
        "learning_rate": 0.1,
        "max_depth": 1,
        "min_samples_leaf": 1,
        "random_state": 0,
    },
    "knn_utility": {"k": 2, "distance_power": 2.0},
    "cost_risk_guardrail": {
        "max_cost": 1.0,
        "max_risk": 1.0,
        "cost_weight": 0.0,
        "risk_weight": 0.0,
    },
}

_CARDS = {
    "rule": "rule.v1",
    "ridge_utility": "ridge_utility.v1",
    "ridge_t_learner": "ridge_t_learner.v1",
    "logistic_t_learner": "logistic_t_learner.v1",
    "gbdt_t_learner": "gbdt_t_learner.v1",
    "knn_utility": "knn_utility.v1",
    "cost_risk_guardrail": "cost_risk_guardrail.v1",
}


def _calibration_config(kind: str) -> dict[str, object]:
    if kind == "rule":
        return {
            "threshold_grid": [0.0, 0.5, 1.0],
            "margin_grid": [0.0],
            "max_harm_rate": 1.0,
        }
    if kind == "cost_risk_guardrail":
        return {
            "max_cost_grid": [0.0, 1.0],
            "max_risk_grid": [0.0, 1.0],
            "gain_grid": [-1.0, 0.0],
            "margin_grid": [0.0],
            "max_harm_rate": 1.0,
        }
    return {
        "gain_grid": [-1.0, 0.0],
        "margin_grid": [0.0],
        "max_harm_rate": 1.0,
    }


class SelectorProducerTests(unittest.TestCase):
    def test_policy_manifest_is_derived_from_frozen_full_five_calibration(self) -> None:
        discovery = _full_five_discovery_freeze()
        fitted = create_fitted_selector_bundle(
            discovery,
            selector_kind="ridge_utility",
            revision="policy-fit-v1",
            fit_config=_FIT_CONFIGS["ridge_utility"],
        )
        calibrated, _summary = calibrate_selector_bundle(
            fitted,
            discovery,
            revision="policy-cal-v1",
            calibration_config=_calibration_config("ridge_utility"),
        )
        frozen = freeze_selector_bundle(calibrated, revision="policy-frozen-v1")

        manifest = build_selector_policy_manifest(frozen, discovery)

        self.assertEqual(manifest["stage"], "frozen")
        self.assertEqual(
            manifest["locked_best_eligible_global"]["selected_family"],
            SemanticFamily.REFORMULATE.value,
        )
        self.assertEqual(
            manifest["locked_best_eligible_global"]["selected_release_id"],
            _FULL_RELEASES[SemanticFamily.REFORMULATE],
        )
        self.assertEqual(
            manifest["discovery"]["sha256"],
            discovery.discovery_hash,
        )
        expected_members = {
            split: tuple(
                sorted(
                    {outcome.group_id for outcome in discovery.outcomes if outcome.split == split}
                )
            )
            for split in ("train", "calibration")
        }
        self.assertEqual(
            manifest["group_partition"]["hash_scheme"],
            "benchmark_group.content_hash.v1",
        )
        for split, members in expected_members.items():
            partition = manifest["group_partition"]["splits"][split]
            self.assertEqual(partition["member_hashes"], members)
            self.assertEqual(
                partition["commitment_sha256"],
                discovery.training_table.group_split_digests[split],
            )
        with self.assertRaisesRegex(ValidationError, "requires a frozen"):
            build_selector_policy_manifest(
                replace(frozen, stage=SelectorArtifactStage.CALIBRATED),
                discovery,
            )

    def test_all_seven_selectors_fit_calibrate_freeze_and_store(self) -> None:
        discovery = _discovery_freeze()
        base = create_fitted_selector_bundle(
            discovery,
            selector_kind="ridge_utility",
            revision="base-fit-v1",
            fit_config=_FIT_CONFIGS["ridge_utility"],
        )
        kinds = list(_FIT_CONFIGS)
        if importlib.util.find_spec("sklearn") is None:
            kinds.remove("gbdt_t_learner")

        with tempfile.TemporaryDirectory() as directory:
            store = SelectorArtifactStore(directory)
            for kind in kinds:
                with self.subTest(kind=kind):
                    fitted = create_fitted_selector_bundle(
                        discovery,
                        selector_kind=kind,
                        revision=f"{kind}-fit-v1",
                        fit_config=_FIT_CONFIGS[kind],
                        base_selector_bundle=(base if kind == "cost_risk_guardrail" else None),
                    )
                    self.assertEqual(fitted.selector_card_id, _CARDS[kind])
                    self.assertEqual(fitted.stage, SelectorArtifactStage.FITTED)
                    fitted_identity = store.save(fitted)

                    calibrated, summary = calibrate_selector_bundle(
                        fitted,
                        discovery,
                        revision=f"{kind}-cal-v1",
                        calibration_config=_calibration_config(kind),
                    )
                    self.assertGreaterEqual(summary.candidates_considered, 1)
                    self.assertEqual(
                        summary.receipt["predecessor_bundle_hash"],
                        fitted.bundle_hash,
                    )
                    self.assertEqual(
                        summary.receipt["discovery_hash"],
                        discovery.discovery_hash,
                    )
                    calibrated_identity = store.save(
                        calibrated,
                        previous_identity=fitted_identity,
                    )

                    frozen = freeze_selector_bundle(
                        calibrated,
                        revision=f"{kind}-frozen-v1",
                    )
                    frozen_identity = store.save(
                        frozen,
                        previous_identity=calibrated_identity,
                    )
                    _selector, restored = store.load(
                        frozen_identity,
                        require_stage=SelectorArtifactStage.FROZEN,
                    )
                    self.assertEqual(restored.model_payload, calibrated.model_payload)
                    self.assertEqual(
                        restored.calibration_receipt,
                        calibrated.calibration_receipt,
                    )

    def test_fit_config_is_fail_closed_and_guardrail_requires_base(self) -> None:
        discovery = _discovery_freeze()
        with self.assertRaisesRegex(ValidationError, "fields differ"):
            create_fitted_selector_bundle(
                discovery,
                selector_kind="ridge_utility",
                revision="bad-fit-v1",
                fit_config={"ridge": 0.01, "unexpected": True},
            )
        with self.assertRaisesRegex(ValidationError, "SelectorArtifactBundle"):
            create_fitted_selector_bundle(
                discovery,
                selector_kind="cost_risk_guardrail",
                revision="bad-guard-v1",
                fit_config=_FIT_CONFIGS["cost_risk_guardrail"],
            )
        naked = create_fitted_selector_bundle(
            discovery,
            selector_kind="ridge_utility",
            revision="naked-base-v1",
            fit_config=_FIT_CONFIGS["ridge_utility"],
        ).selector()
        with self.assertRaisesRegex(ValidationError, "not a naked selector"):
            create_fitted_selector_bundle(
                discovery,
                selector_kind="cost_risk_guardrail",
                revision="bad-naked-guard-v1",
                fit_config=_FIT_CONFIGS["cost_risk_guardrail"],
                base_selector_bundle=naked,  # type: ignore[arg-type]
            )

    def test_guardrail_rejects_unrelated_base_and_preserves_exact_lineage(self) -> None:
        discovery = _discovery_freeze()
        base = create_fitted_selector_bundle(
            discovery,
            selector_kind="ridge_utility",
            revision="base-fit-v1",
            fit_config=_FIT_CONFIGS["ridge_utility"],
        )
        unrelated_discovery = replace(
            discovery,
            revision="unrelated-discovery-v1",
        )
        with self.assertRaisesRegex(
            ValidationError,
            "unrelated SelectorDiscoveryFreeze",
        ):
            create_fitted_selector_bundle(
                unrelated_discovery,
                selector_kind="cost_risk_guardrail",
                revision="guardrail-wrong-base-v1",
                fit_config=_FIT_CONFIGS["cost_risk_guardrail"],
                base_selector_bundle=base,
            )

        fitted = create_fitted_selector_bundle(
            discovery,
            selector_kind="cost_risk_guardrail",
            revision="guardrail-fit-v1",
            fit_config=_FIT_CONFIGS["cost_risk_guardrail"],
            base_selector_bundle=base,
        )
        self.assertEqual(
            fitted.fit_lineage["base_selector_bundle"],
            {
                "selector_id": base.selector_card_id,
                "revision": base.revision,
                "sha256": base.bundle_hash,
                "stage": base.stage.value,
                "model_hash": base.model_hash,
            },
        )
        self.assertEqual(
            fitted.fit_lineage["discovery"],
            {
                "revision": discovery.revision,
                "sha256": discovery.discovery_hash,
            },
        )

        calibrated, _summary = calibrate_selector_bundle(
            fitted,
            discovery,
            revision="guardrail-cal-v1",
            calibration_config=_calibration_config("cost_risk_guardrail"),
        )
        frozen = freeze_selector_bundle(
            calibrated,
            revision="guardrail-frozen-v1",
        )
        self.assertEqual(calibrated.fit_lineage, fitted.fit_lineage)
        self.assertEqual(frozen.fit_lineage, fitted.fit_lineage)

        with tempfile.TemporaryDirectory() as directory:
            store = SelectorArtifactStore(directory)
            fitted_identity = store.save(fitted)
            calibrated_identity = store.save(
                calibrated,
                previous_identity=fitted_identity,
            )
            frozen_identity = store.save(
                frozen,
                previous_identity=calibrated_identity,
            )
            _selector, restored = store.load(frozen_identity)
        self.assertEqual(restored.fit_lineage, fitted.fit_lineage)

    def test_calibration_rejects_a_different_discovery_binding(self) -> None:
        discovery = _discovery_freeze()
        fitted = create_fitted_selector_bundle(
            discovery,
            selector_kind="ridge_utility",
            revision="fit-v1",
            fit_config=_FIT_CONFIGS["ridge_utility"],
        )
        misbound = replace(
            fitted,
            extractor_hash=content_hash({"extractor": "changed"}),
        )
        with self.assertRaisesRegex(ValidationError, "differs from"):
            calibrate_selector_bundle(
                misbound,
                discovery,
                revision="cal-v1",
                calibration_config=_calibration_config("ridge_utility"),
            )


if __name__ == "__main__":
    unittest.main()
