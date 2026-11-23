from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace

from agentinputlab.contracts import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import canonical_json, content_hash
from agentinputlab.selectors import (
    CostRiskAwareSelector,
    FrozenGBDTOutcomeModel,
    FrozenRegressionTree,
    GBDTTLearnerSelector,
    KNNTrainingRow,
    KNNUtilitySelector,
    LogisticTLearnerSelector,
    RidgeTLearnerSelector,
    RidgeUtilitySelector,
    RuleSelector,
)
from agentinputlab.selectors.artifacts import (
    SelectorArtifactBundle,
    SelectorArtifactStage,
    SelectorArtifactStore,
    selector_card_id,
    selector_from_artifact,
    selector_model_hash,
    selector_to_artifact,
)

_ACQUIRE = "acquire.hyde.paper_reimplementation"
_RELEASES = {SemanticFamily.ACQUIRE: _ACQUIRE}
_HASHES = {
    "extractor": content_hash({"extractor": "v1"}),
    "schema": content_hash({"schema": "v1"}),
    "table": content_hash({"table": "train"}),
    "split": content_hash({"splits": "v1"}),
    "profile": content_hash({"profile": "controlled"}),
}


def _manual_gbdt() -> GBDTTLearnerSelector:
    no_op_tree = FrozenRegressionTree(
        children_left=(-1,),
        children_right=(-1,),
        features=(-2,),
        thresholds=(-2.0,),
        values=(0.2,),
    )
    acquire_tree = FrozenRegressionTree(
        children_left=(1, -1, -1),
        children_right=(2, -1, -1),
        features=(2, -2, -2),
        thresholds=(0.5, -2.0, -2.0),
        values=(0.0, 0.1, 0.9),
    )
    return GBDTTLearnerSelector(
        outcome_models={
            SemanticFamily.NO_OP: FrozenGBDTOutcomeModel(0.0, 1.0, (no_op_tree,)),
            SemanticFamily.ACQUIRE: FrozenGBDTOutcomeModel(
                0.0,
                1.0,
                (acquire_tree,),
            ),
        },
        release_by_family=_RELEASES,
        sklearn_version="fixture",
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        random_state=0,
    )


def _selectors():
    ridge = RidgeUtilitySelector(
        {SemanticFamily.ACQUIRE: (0.1, 0.0, 0.0, 1.0, 0.0, 0.0)},
        _RELEASES,
    )
    ridge_t = RidgeTLearnerSelector(
        {
            SemanticFamily.NO_OP: (0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            SemanticFamily.ACQUIRE: (0.1, 0.0, 0.0, 1.0, 0.0, 0.0),
        },
        _RELEASES,
    )
    logistic = LogisticTLearnerSelector(
        {
            SemanticFamily.NO_OP: (0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            SemanticFamily.ACQUIRE: (0.1, 0.0, 0.0, 1.0, 0.0, 0.0),
        },
        _RELEASES,
        l2=0.01,
        learning_rate=0.2,
        iterations=10,
    )
    knn = KNNUtilitySelector(
        (
            KNNTrainingRow(
                (0.0, 0.0, 1.0, 0.0, 0.0),
                {SemanticFamily.ACQUIRE: 0.8},
            ),
        ),
        _RELEASES,
        k=1,
    )
    guardrail = CostRiskAwareSelector(
        ridge,
        _RELEASES,
        {SemanticFamily.ACQUIRE: 1.0},
        {SemanticFamily.ACQUIRE: 0.1},
        max_cost=2.0,
        max_risk=0.5,
    )
    return (
        RuleSelector(),
        ridge,
        ridge_t,
        logistic,
        _manual_gbdt(),
        knn,
        guardrail,
    )


def _bundle(
    selector,
    *,
    stage: SelectorArtifactStage,
    revision: str,
    receipt=None,
) -> SelectorArtifactBundle:
    return SelectorArtifactBundle.create(
        selector,
        stage=stage,
        revision=revision,
        extractor_id="visible_state_heuristic",
        extractor_revision="v1",
        extractor_hash=_HASHES["extractor"],
        feature_schema_id="family_defect_features.v1",
        feature_schema_hash=_HASHES["schema"],
        representative_by_family=(
            _RELEASES if not hasattr(selector, "release_by_family") else selector.release_by_family
        ),
        training_table_sha256=_HASHES["table"],
        split_provenance_hash=_HASHES["split"],
        comparison_profile_hash=_HASHES["profile"],
        calibration_receipt=receipt,
    )


class SelectorCodecTests(unittest.TestCase):
    def test_all_seven_public_selectors_roundtrip_without_pickle(self) -> None:
        for selector in _selectors():
            with self.subTest(selector=type(selector).__name__):
                artifact = selector_to_artifact(selector)
                wire = json.loads(json.dumps(artifact, allow_nan=False))
                restored = selector_from_artifact(wire)
                self.assertEqual(selector_card_id(restored), selector_card_id(selector))
                self.assertEqual(
                    selector_model_hash(restored),
                    selector_model_hash(selector),
                )
                self.assertNotIn("pickle", json.dumps(wire).casefold())

    def test_codec_rejects_extra_fields_coercion_and_unknown_kind(self) -> None:
        artifact = selector_to_artifact(RuleSelector())
        extra = copy.deepcopy(artifact)
        extra["payload"]["extra"] = 1
        boolean = copy.deepcopy(artifact)
        boolean["payload"]["threshold"] = True
        unknown = copy.deepcopy(artifact)
        unknown["selector_kind"] = "python_pickle"
        for corrupt in (extra, boolean, unknown):
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                selector_from_artifact(corrupt)

    def test_bundle_binds_model_extractor_schema_and_representatives(self) -> None:
        selector = _selectors()[1]
        bundle = _bundle(
            selector,
            stage=SelectorArtifactStage.FITTED,
            revision="fit-v1",
        )
        restored = SelectorArtifactBundle.from_artifact(
            json.loads(canonical_json(bundle.to_artifact()))
        )
        self.assertEqual(restored.bundle_hash, bundle.bundle_hash)
        self.assertEqual(
            selector_model_hash(restored.selector()),
            selector_model_hash(selector),
        )
        with self.assertRaisesRegex(ValueError, "model hash"):
            replace(bundle, model_hash="0" * 64)


class SelectorArtifactStoreTests(unittest.TestCase):
    def test_append_only_lifecycle_requires_exact_predecessors(self) -> None:
        selector = _selectors()[1]
        fitted = _bundle(
            selector,
            stage=SelectorArtifactStage.FITTED,
            revision="fit-v1",
        )
        calibrated = _bundle(
            selector,
            stage=SelectorArtifactStage.CALIBRATED,
            revision="cal-v1",
            receipt={"gain_grid": [0.0, 0.1], "chosen_gain": 0.1},
        )
        frozen = _bundle(
            selector,
            stage=SelectorArtifactStage.FROZEN,
            revision="frozen-v1",
            receipt={"gain_grid": [0.0, 0.1], "chosen_gain": 0.1},
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SelectorArtifactStore(directory)
            fit_id = store.save(fitted)
            with self.assertRaisesRegex(ArtifactError, "requires its exact predecessor"):
                store.save(calibrated)
            cal_id = store.save(calibrated, previous_identity=fit_id)
            frozen_id = store.save(frozen, previous_identity=cal_id)
            restored, restored_bundle = store.load(
                frozen_id,
                require_stage=SelectorArtifactStage.FROZEN,
            )
            self.assertEqual(restored_bundle, frozen)
            self.assertEqual(
                selector_model_hash(restored),
                selector_model_hash(selector),
            )
            self.assertEqual(store.save(fitted), fit_id)

    def test_same_revision_cannot_be_rebound_to_different_content(self) -> None:
        first = _bundle(
            RuleSelector(threshold=0.6),
            stage=SelectorArtifactStage.FITTED,
            revision="fit-v1",
        )
        second = _bundle(
            RuleSelector(threshold=0.7),
            stage=SelectorArtifactStage.FITTED,
            revision="fit-v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SelectorArtifactStore(directory)
            store.save(first)
            with self.assertRaisesRegex(ArtifactError, "append-only"):
                store.save(second)

    def test_freeze_cannot_change_calibrated_policy(self) -> None:
        fitted = _bundle(
            RuleSelector(threshold=0.6),
            stage=SelectorArtifactStage.FITTED,
            revision="fit-v1",
        )
        calibrated = _bundle(
            RuleSelector(threshold=0.7),
            stage=SelectorArtifactStage.CALIBRATED,
            revision="cal-v1",
            receipt={"chosen_threshold": 0.7},
        )
        changed = _bundle(
            RuleSelector(threshold=0.8),
            stage=SelectorArtifactStage.FROZEN,
            revision="frozen-v1",
            receipt={"chosen_threshold": 0.7},
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SelectorArtifactStore(directory)
            fit_id = store.save(fitted)
            cal_id = store.save(calibrated, previous_identity=fit_id)
            with self.assertRaisesRegex(ArtifactError, "cannot change"):
                store.save(changed, previous_identity=cal_id)

    def test_calibration_cannot_replace_fitted_model_parameters(self) -> None:
        fitted_selector = RidgeUtilitySelector(
            {SemanticFamily.ACQUIRE: (0.1, 0.0, 0.0, 1.0, 0.0, 0.0)},
            _RELEASES,
        )
        replaced_selector = RidgeUtilitySelector(
            {SemanticFamily.ACQUIRE: (0.9, 0.0, 0.0, 1.0, 0.0, 0.0)},
            _RELEASES,
            min_gain=0.2,
        )
        fitted = _bundle(
            fitted_selector,
            stage=SelectorArtifactStage.FITTED,
            revision="fit-v1",
        )
        replaced = _bundle(
            replaced_selector,
            stage=SelectorArtifactStage.CALIBRATED,
            revision="cal-replaced-v1",
            receipt={"chosen_gain": 0.2},
        )
        allowed = _bundle(
            replace(fitted_selector, min_gain=0.2, margin=0.1),
            stage=SelectorArtifactStage.CALIBRATED,
            revision="cal-allowed-v1",
            receipt={"chosen_gain": 0.2, "chosen_margin": 0.1},
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SelectorArtifactStore(directory)
            fit_id = store.save(fitted)
            with self.assertRaisesRegex(ArtifactError, "fitted model parameters"):
                store.save(replaced, previous_identity=fit_id)
            stored_id = store.save(allowed, previous_identity=fit_id)
            _selector, restored = store.load(
                stored_id,
                require_stage=SelectorArtifactStage.CALIBRATED,
            )
            self.assertEqual(restored, allowed)


if __name__ == "__main__":
    unittest.main()
