from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentinputlab.contracts import (
    FrozenAgentState,
    ProfileKind,
    SemanticFamily,
)
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import canonical_json, content_hash, to_jsonable
from agentinputlab.experiments import (
    ExperimentArmSpec,
    ExperimentArtifactStore,
    ExperimentBenchmarkTarget,
    ExperimentCellResult,
    ExperimentManifest,
    ExperimentStage,
    build_experiment_plan,
)
from agentinputlab.selectors import (
    SelectorDiscoveryFreeze,
    SelectorDiscoveryStore,
    UtilityDefinition,
    VisibleStateFeatureExtractor,
    build_selector_discovery_freeze,
)
from agentinputlab.selectors.discovery import _normalized_selector_arm

_RELEASE = "acquire.hyde.paper_reimplementation"
_DEPENDENCIES = {
    "backend": content_hash({"backend": "fixture"}),
    "scorer": content_hash({"scorer": "fixture"}),
}


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        schema_version=1,
        study_id="selector.discovery.fixture",
        stage=ExperimentStage.DISCOVERY,
        estimand_profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id="fixture.benchmark",
                split="discovery_train",
                unit_ids=("train-unit",),
                external=True,
                source_revision="fixture-v1",
            ),
            ExperimentBenchmarkTarget(
                cell_id="fixture.benchmark",
                split="discovery_calibration",
                unit_ids=("calibration-unit",),
                external=True,
                source_revision="fixture-v1",
            ),
        ),
        arms=(
            ExperimentArmSpec(
                arm_id="no_op",
                release_id="baseline.no_op",
                family=SemanticFamily.NO_OP,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
            ),
            ExperimentArmSpec(
                arm_id="acquire",
                release_id=_RELEASE,
                family=SemanticFamily.ACQUIRE,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
            ),
        ),
        seeds=(7,),
        metadata={"selector_training_allowed": True},
    )


def _result_payload(
    cell,
    *,
    plan_hash: str,
    inject_untrusted_selector_fields: bool = False,
    tamper_state_after_receipt: bool = False,
) -> dict[str, object]:
    state = FrozenAgentState(
        query=f"Find latest evidence for {cell.unit_id}.",
        task_id=f"raw-secret-{cell.unit_id}",
        metadata={"gold_answer": f"gold-secret-{cell.unit_id}"},
    )
    extractor = VisibleStateFeatureExtractor()
    observation = extractor.extract(state)
    features = observation.features
    selector_state = state.canonical_dict()
    if tamper_state_after_receipt:
        selector_state["query"] = f"post-action state for {cell.unit_id}"
    selector_record: dict[str, object] = {
        "schema_version": 1,
        "state": selector_state,
        "group_id": f"template-{cell.unit_id}",
        "eligible_families": ["acquire"],
    }
    if inject_untrusted_selector_fields:
        selector_record.update(
            {
                "selector_split": "calibration",
                "backend_hash": content_hash({"forged": "backend"}),
                "comparison_profile_hash": content_hash({"forged": "profile"}),
                "arm_resources": {
                    "acquire": {"cost": 0.0, "risk": 0.0},
                },
            }
        )
    result = ExperimentCellResult(
        cell_id=cell.cell_id,
        status="succeeded",
        payload={
            "observed_dependency_hashes": {
                arm_id: {
                    **_DEPENDENCIES,
                    "official_runtime": content_hash({"split": cell.split}),
                }
                for arm_id in cell.arm_ids
            },
            "arms": [
                {
                    "arm_id": "no_op",
                    "status": "succeeded",
                    "metrics": {"valid_success": 0.2},
                    "usage": {
                        "unpriced_model_calls": 0,
                        "total_cost_usd": 0.02,
                    },
                },
                {
                    "arm_id": "acquire",
                    "status": "succeeded",
                    "metrics": {"valid_success": 0.8},
                    "usage": {
                        "unpriced_model_calls": 0,
                        "total_cost_usd": 0.12,
                    },
                },
            ],
            "pre_action": {
                "state_hash": observation.state_hash,
                "feature_hash": observation.feature_hash,
                "feature_schema_id": extractor.feature_schema_id,
                "feature_schema_hash": extractor.feature_schema_hash,
                "provenance_hash": content_hash(observation.provenance),
                "defect_scores": {
                    "reformulate": features.reformulation_need_score,
                    "decompose": features.multi_step_score,
                    "acquire": features.retrieval_need_score,
                    "clarify": features.ambiguity_score,
                    "curate": features.context_pressure,
                },
            },
            "selector_record": selector_record,
        },
    )
    return {
        "schema_version": 1,
        "study_id": cell.study_id,
        "manifest_hash": cell.manifest_hash,
        "plan_hash": plan_hash,
        "cell": {**to_jsonable(cell), "cell_id": cell.cell_id},
        "result": result.to_dict(),
        "finished_at_utc": "2026-01-01T00:00:00Z",
        "claims": {
            "official_parity_verified": False,
            "confirmatory_candidate_record": False,
            "paper_claim_authorized": False,
            "resource_receipt_count": len(result.resource_receipts),
        },
    }


def _write_run(
    root: Path,
    *,
    inject_untrusted_selector_fields: bool = False,
    tamper_state_after_receipt: bool = False,
) -> Path:
    manifest = _manifest()
    plan = build_experiment_plan(
        manifest,
        {
            "fixture.benchmark:discovery_train": ("train-unit",),
            "fixture.benchmark:discovery_calibration": ("calibration-unit",),
        },
    )
    store = ExperimentArtifactStore(
        root,
        manifest=manifest,
        plan=plan,
        driver_fingerprint={"driver": "selector-fixture-v1"},
    )
    store.initialize(resume=False)
    for cell in plan.cells:
        store.write_cell_result(
            cell.cell_id,
            _result_payload(
                cell,
                plan_hash=plan.plan_hash,
                inject_untrusted_selector_fields=inject_untrusted_selector_fields,
                tamper_state_after_receipt=tamper_state_after_receipt,
            ),
        )
    return store.run_dir


class SelectorDiscoveryProducerTests(unittest.TestCase):
    def test_algorithm_failure_is_a_strict_adapter_hard_stop(self) -> None:
        status, metrics, cost, hard_harm = _normalized_selector_arm(
            {
                "status": "failed",
                "failure": {"scope": "algorithm", "category": "runtime_error"},
                "usage": {"unpriced_model_calls": 0, "total_cost_usd": 0.1},
            },
            arm_id="acquire",
        )
        self.assertEqual(status.value, "algorithm_failure")
        self.assertEqual(metrics, {})
        self.assertEqual(cost, 0.1)
        self.assertTrue(hard_harm)

    def test_verified_run_converts_and_roundtrips_without_raw_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = build_selector_discovery_freeze(
                (_write_run(root / "runs"),),
                revision="discovery-v1",
                extractor=VisibleStateFeatureExtractor(),
                utility_definition=UtilityDefinition("valid_success"),
                representative_by_family={SemanticFamily.ACQUIRE: _RELEASE},
            )
            self.assertEqual(len(freeze.observations), 2)
            self.assertEqual(len(freeze.outcomes), 4)
            self.assertEqual(
                {row.split for row in freeze.training_table.examples},
                {"train", "calibration"},
            )
            serialized = canonical_json(freeze.to_artifact())
            self.assertNotIn("raw-secret", serialized)
            self.assertNotIn("gold-secret", serialized)
            self.assertNotIn("Find latest evidence", serialized)

            store = SelectorDiscoveryStore(root / "selector-artifacts")
            identity = store.save(freeze)
            self.assertEqual(store.load(identity), freeze)
            self.assertEqual(store.save(freeze), identity)

    def test_artifact_tampering_and_revision_rebinding_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = build_selector_discovery_freeze(
                (_write_run(root / "runs"),),
                revision="discovery-v1",
                extractor=VisibleStateFeatureExtractor(),
                utility_definition=UtilityDefinition("valid_success"),
                representative_by_family={SemanticFamily.ACQUIRE: _RELEASE},
            )
            artifact = json.loads(canonical_json(freeze.to_artifact()))
            artifact["outcomes"][0]["metrics"]["valid_success"] = 0.99
            with self.assertRaisesRegex(ValueError, "outcome hash"):
                SelectorDiscoveryFreeze.from_artifact(artifact)

            store = SelectorDiscoveryStore(root / "selector-artifacts")
            store.save(freeze)
            changed = build_selector_discovery_freeze(
                (_write_run(root / "other-runs"),),
                revision="discovery-v1",
                extractor=VisibleStateFeatureExtractor(),
                utility_definition=UtilityDefinition(
                    "valid_success",
                    algorithm_failure_utility=-1.0,
                ),
                representative_by_family={SemanticFamily.ACQUIRE: _RELEASE},
            )
            with self.assertRaisesRegex(ArtifactError, "different content"):
                store.save(changed)

    def test_split_resources_and_identities_cannot_be_driver_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                ValueError,
                "unknown fields",
            ):
                build_selector_discovery_freeze(
                    (
                        _write_run(
                            root / "untrusted-fields",
                            inject_untrusted_selector_fields=True,
                        ),
                    ),
                    revision="untrusted-fields",
                    extractor=VisibleStateFeatureExtractor(),
                    utility_definition=UtilityDefinition("valid_success"),
                    representative_by_family={SemanticFamily.ACQUIRE: _RELEASE},
                )

    def test_pre_action_state_receipt_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                ArtifactError,
                "does not authenticate selector_record.state",
            ):
                build_selector_discovery_freeze(
                    (
                        _write_run(
                            root / "tampered-state",
                            tamper_state_after_receipt=True,
                        ),
                    ),
                    revision="tampered-state",
                    extractor=VisibleStateFeatureExtractor(),
                    utility_definition=UtilityDefinition("valid_success"),
                    representative_by_family={SemanticFamily.ACQUIRE: _RELEASE},
                )


if __name__ == "__main__":
    unittest.main()
