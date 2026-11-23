from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentinputlab.benchmarks import AtomicReplayArmReceipt, AtomicReplayReceipt
from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments import transfer_analysis as transfer
from agentinputlab.experiments.bindings import (
    FrozenBindingEntry,
    FrozenBindingReceipt,
    build_guarded_driver_fingerprint,
)
from agentinputlab.experiments.schema import (
    ExperimentArmSpec,
    ExperimentBenchmarkTarget,
    ExperimentManifest,
    ExperimentPrerequisite,
    ExperimentStage,
    PrerequisiteKind,
)

_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)
_REPRESENTATIVE_HASH = "e" * 64
_DEPENDENCIES = {
    "agent_model": "1" * 64,
    "official_runtime": "2" * 64,
    "corpus": "3" * 64,
    "preprocessing": "4" * 64,
    "index": "5" * 64,
    "embedder": "6" * 64,
    "retriever": "7" * 64,
    "reranker": "8" * 64,
    "reader": "9" * 64,
    "scorer": "a" * 64,
}
_TRANSFER_MODEL_HASHES = {
    model_slot: f"{index:x}" * 64
    for index, model_slot in enumerate(transfer._TRANSFER_MODEL_SLOTS, start=10)
}


def _arms() -> tuple[ExperimentArmSpec, ...]:
    return (
        ExperimentArmSpec(
            arm_id="no_op",
            release_id="baseline.no_op",
            family=SemanticFamily.NO_OP,
            profile_kind=ProfileKind.CONTROLLED_ADAPTER,
            parameters={"role": "paired_baseline"},
        ),
        *(
            ExperimentArmSpec(
                arm_id=f"{family.value}_representative",
                release_id=f"method.{family.value}.rank1",
                family=family,
                profile_kind=ProfileKind.CONTROLLED_ADAPTER,
                parameters={
                    "representative_rank": 1,
                    "representative_manifest_hash": _REPRESENTATIVE_HASH,
                },
            )
            for family in _FAMILIES
        ),
    )


def _file_prerequisite(name: str) -> ExperimentPrerequisite:
    return ExperimentPrerequisite(
        kind=PrerequisiteKind.FILE_ENV,
        name=name,
        description=f"Fixture {name}",
        sha256_env=f"{name}_SHA256",
    )


def _heldout_manifest() -> ExperimentManifest:
    arms = _arms()
    return ExperimentManifest(
        schema_version=1,
        study_id="robustness.heldout_model_transfer",
        stage=ExperimentStage.TRANSFER,
        estimand_profile_kind=ProfileKind.CONTROLLED_ADAPTER,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id="state_bench.common_support",
                split="heldout_model_transfer",
                unit_inventory_env="AILAB_HELDOUT_MODEL_UNITS",
                unit_inventory_sha256_env="AILAB_HELDOUT_MODEL_UNITS_SHA256",
                external=True,
                source_revision=transfer._STATEBENCH_REVISION,
            ),
        ),
        arms=arms,
        seeds=(17,),
        model_slots=transfer._TRANSFER_MODEL_SLOTS,
        driver_entrypoint=transfer._EXTERNAL_DRIVER,
        prerequisites=(
            ExperimentPrerequisite(
                kind=PrerequisiteKind.DIRECTORY_ENV,
                name="AILAB_STATEBENCH_CHECKOUT",
                description="Fixture checkout",
            ),
            _file_prerequisite("AILAB_REPRESENTATIVE_MANIFEST"),
            _file_prerequisite("AILAB_EXTERNAL_ARENA_DRIVER_CONFIG"),
            _file_prerequisite("AILAB_MODEL_CONFIG"),
            _file_prerequisite("AILAB_RETRIEVER_CONFIG"),
            _file_prerequisite("AILAB_HELDOUT_MODEL_FREEZE_RECEIPT"),
        ),
        metadata={
            "representative_manifest_hash": _REPRESENTATIVE_HASH,
            "model_used_for_tuning": False,
            "group_relation": "prespecified_subset_of_common_support_lockbox",
            "requires_atomic_external_replay": True,
            "confirmatory_claimed_before_execution": False,
            "checkout_expected_revisions": {
                "AILAB_STATEBENCH_CHECKOUT": transfer._STATEBENCH_REVISION,
            },
            "experiment_census": {
                "protocol": "agentinputlab.experiment_census.v1",
                "campaign_kind": "heldout_model_transfer",
                "independent_group_unit": "benchmark_cell:unit_id",
                "independent_group_target": 200,
                "target_group_counts": {
                    "state_bench.common_support:heldout_model_transfer": 200,
                },
                "expected_arm_ids": [arm.arm_id for arm in arms],
                "expected_model_slots": list(transfer._TRANSFER_MODEL_SLOTS),
                "expected_seeds": [17],
                "expected_plan_cells": 800,
                "expected_agent_episodes": 4_800,
                "episode_formula": (
                    "sum(target_groups * target_arms * models * continuation_seeds)"
                ),
                "seed_semantics": ("one_primary_continuation_seed_per_group_model_arm"),
            },
        },
    )


def _binding_entry(
    prerequisite: ExperimentPrerequisite,
    *,
    manifest: ExperimentManifest,
) -> FrozenBindingEntry:
    if prerequisite.kind is PrerequisiteKind.DIRECTORY_ENV:
        revision = manifest.metadata["checkout_expected_revisions"][prerequisite.name]
        identity = {
            "identity_mode": "clean_git_checkout",
            "resolved_path_sha256": "b" * 64,
            "git_head": revision,
            "git_tree": "c" * 40,
            "git_clean": True,
            "expected_revision": revision,
        }
    else:
        digest = (
            "d" * 64
            if prerequisite.name == "AILAB_EXTERNAL_ARENA_DRIVER_CONFIG"
            else content_hash({"fixture-binding": prerequisite.name})
        )
        identity = {
            "identity_mode": "file_sha256",
            "resolved_path_sha256": "b" * 64,
            "content_sha256": digest,
            "size_bytes": 100,
            "checksum_environment": prerequisite.sha256_env,
        }
    return FrozenBindingEntry(
        name=prerequisite.name,
        kind=prerequisite.kind.value,
        identity=identity,
    )


def _operator(manifest: ExperimentManifest) -> dict[str, object]:
    return {
        "implementation_id": transfer._EXTERNAL_DRIVER_IMPLEMENTATION,
        "schema_version": 3,
        "driver_id": "fixture.external",
        "driver_revision_sha256": "f" * 64,
        "config_sha256": "d" * 64,
        "baseline_arm_id": "no_op",
        "benchmark_targets": {
            f"{target.cell_id}:{target.split}": {
                "benchmark_id": target.cell_id,
                "split": target.split,
                "source_revision": target.source_revision,
            }
            for target in manifest.benchmarks
        },
        "arms": {
            arm.arm_id: {
                "release_id": arm.release_id,
                "family": arm.family.value,
                "profile_kind": arm.profile_kind.value,
                "arm_kind": arm.arm_kind.value,
            }
            for arm in manifest.arms
        },
        "resolved_policy_bindings": {},
        "resolved_gate_bindings": {},
    }


def _run_identity(
    manifest: ExperimentManifest,
    *,
    operator: dict[str, object] | None = None,
) -> dict[str, object]:
    receipt = FrozenBindingReceipt(
        manifest_hash=manifest.manifest_hash,
        bindings=tuple(
            _binding_entry(prerequisite, manifest=manifest)
            for prerequisite in manifest.prerequisites
        ),
    )
    return {
        "driver_fingerprint": build_guarded_driver_fingerprint(
            operator or _operator(manifest),
            receipt,
        )
    }


def _atomic_pair(
    *,
    treatment_arm_id: str,
    group_index: int,
    dependencies: dict[str, str],
) -> dict[str, object]:
    origin = content_hash({"origin": group_index})
    replay = content_hash({"replay": group_index})
    receipt = AtomicReplayReceipt(
        benchmark_id="state_bench.common_support",
        checkpoint_id=f"checkpoint-{group_index:04d}",
        source_revision=transfer._STATEBENCH_REVISION,
        bridge_id="fixture.bridge",
        bridge_revision="b" * 64,
        scorer_id="fixture.scorer",
        scorer_revision="c" * 64,
        origin_state_hash=origin,
        replay_identity=replay,
        seed=17,
        locked_dependency_hashes=dependencies,
        arms=(
            AtomicReplayArmReceipt(
                role="baseline",
                arm_id="no_op",
                origin_state_hash=origin,
                replay_identity=replay,
                seed=17,
                locked_dependency_hashes=dependencies,
                runtime_instance_id=content_hash(
                    {"runtime": group_index, "arm": "no_op", "pair": treatment_arm_id}
                ),
                completed=True,
            ),
            AtomicReplayArmReceipt(
                role="treatment",
                arm_id=treatment_arm_id,
                origin_state_hash=origin,
                replay_identity=replay,
                seed=17,
                locked_dependency_hashes=dependencies,
                runtime_instance_id=content_hash(
                    {
                        "runtime": group_index,
                        "arm": treatment_arm_id,
                        "pair": treatment_arm_id,
                    }
                ),
                completed=True,
            ),
        ),
    )
    baseline_consistency = content_hash({"baseline": group_index, "pair_invariant": True})
    return {
        "treatment_arm_id": treatment_arm_id,
        "request_sha256": content_hash({"request": group_index, "treatment": treatment_arm_id}),
        "receipt": to_jsonable(receipt),
        "receipt_sha256": content_hash(receipt),
        "baseline_consistency_sha256": baseline_consistency,
        "metric_deltas": {},
        "resource_receipt_sha256s": [],
    }


def _records(
    manifest: ExperimentManifest,
    *,
    groups: int = 200,
    dependencies: dict[str, str] | None = None,
) -> tuple[dict[str, object], ...]:
    base_dependencies = dict(_DEPENDENCIES if dependencies is None else dependencies)
    arm_ids = tuple(arm.arm_id for arm in manifest.arms)
    records: list[dict[str, object]] = []
    for group_index in range(groups):
        for model_slot in transfer._TRANSFER_MODEL_SLOTS:
            resolved_dependencies = {
                **base_dependencies,
                "agent_model": _TRANSFER_MODEL_HASHES[model_slot],
            }
            unit_id = f"shared-{group_index:04d}"
            origin = content_hash({"origin": group_index})
            replay = content_hash({"replay": group_index})
            pairs = [
                _atomic_pair(
                    treatment_arm_id=arm_id,
                    group_index=group_index,
                    dependencies=resolved_dependencies,
                )
                for arm_id in arm_ids[1:]
            ]
            baseline_consistency = pairs[0]["baseline_consistency_sha256"]
            records.append(
                {
                    "cell": {
                        "benchmark_id": "state_bench.common_support",
                        "split": "heldout_model_transfer",
                        "unit_id": unit_id,
                        "model_slot": model_slot,
                        "seed": 17,
                        "arm_ids": list(arm_ids),
                    },
                    "result": {
                        "status": "succeeded",
                        "payload": {
                            "benchmark_id": "state_bench.common_support",
                            "split": "heldout_model_transfer",
                            "unit_id": unit_id,
                            "model_slot": model_slot,
                            "seed": 17,
                            "baseline_arm_id": "no_op",
                            "planned_arm_ids": list(arm_ids),
                            "checkpoint_identity": {
                                "identity_sha256": content_hash({"checkpoint": group_index}),
                                "group_sha256": content_hash({"group": group_index}),
                                "visible_state_sha256": content_hash({"visible": group_index}),
                                "origin_state_sha256": origin,
                                "replay_identity_sha256": replay,
                                "source_revision": transfer._STATEBENCH_REVISION,
                            },
                            "pre_action": {},
                            "arms": [
                                {
                                    "arm_id": arm.arm_id,
                                    "status": "succeeded",
                                    "metrics": {
                                        "valid_success": (
                                            0.2
                                            if arm.family is SemanticFamily.NO_OP
                                            else 0.25 + 0.01 * _FAMILIES.index(arm.family)
                                        )
                                    },
                                }
                                for arm in manifest.arms
                            ],
                            "atomic_pairs": pairs,
                            "baseline_consistency_sha256": baseline_consistency,
                            "observed_dependency_hashes": resolved_dependencies,
                            "not_applicable_dependency_slots": [],
                            "target_source_revision": transfer._STATEBENCH_REVISION,
                            "atomic_replay_contract": ("agentinputlab.atomic_external_pair.v1"),
                        },
                    },
                }
            )
    return tuple(records)


def _primary_metric_records(
    manifest: ExperimentManifest,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    arm_ids = [arm.arm_id for arm in manifest.arms]
    for group_index in range(200):
        for model_slot in ("anchor_a", "anchor_b"):
            records.append(
                {
                    "cell": {
                        "benchmark_id": "state_bench.common_support",
                        "split": "common_support_lockbox",
                        "unit_id": f"shared-{group_index:04d}",
                        "model_slot": model_slot,
                        "seed": 17,
                        "arm_ids": arm_ids,
                    },
                    "result": {
                        "status": "succeeded",
                        "payload": {
                            "arms": [
                                {
                                    "arm_id": arm.arm_id,
                                    "status": "succeeded",
                                    "metrics": {
                                        "valid_success": (
                                            0.2
                                            if arm.family is SemanticFamily.NO_OP
                                            else 0.23 + 0.01 * _FAMILIES.index(arm.family)
                                        )
                                    },
                                }
                                for arm in manifest.arms
                            ]
                        },
                    },
                }
            )
    return tuple(records)


class HeldoutModelTransferAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "heldout"
        self.manifest = _heldout_manifest()
        self.records = _records(self.manifest)
        self.identity = _run_identity(self.manifest)

    def _patch_run(self, *, records=None, identity=None):
        return (
            patch.object(transfer, "_load_manifest", return_value=self.manifest),
            patch.object(
                transfer,
                "load_verified_cell_results",
                return_value=self.records if records is None else records,
            ),
            patch.object(
                transfer,
                "_load_run_identity",
                return_value=self.identity if identity is None else identity,
            ),
        )

    def test_exact_verified_run_emits_five_family_effects_and_receipts(self) -> None:
        manifest_patch, records_patch, identity_patch = self._patch_run()
        with manifest_patch, records_patch, identity_patch:
            analysis = transfer.analyze_heldout_model_transfer(
                self.root,
                metric="valid_success",
                bootstrap_samples=100,
                random_seed=7,
            )

        self.assertEqual(
            analysis["analysis_protocol"],
            transfer.HELDOUT_MODEL_TRANSFER_PROTOCOL,
        )
        self.assertEqual(analysis["independent_groups"], 200)
        self.assertEqual(analysis["verified_agent_episodes"], 4_800)
        self.assertEqual(analysis["model_slots"], list(transfer._TRANSFER_MODEL_SLOTS))
        self.assertEqual(
            set(analysis["family_summaries"]),
            {family.value for family in _FAMILIES},
        )
        self.assertEqual(
            set(analysis["family_summaries_by_model"]),
            set(transfer._TRANSFER_MODEL_SLOTS),
        )
        self.assertTrue(analysis["runtime_receipts"]["dependency_and_source_receipts_verified"])
        self.assertEqual(
            analysis["runtime_receipts"]["atomic_pair_receipts"],
            4_000,
        )
        self.assertIs(analysis["experiments_executed_by_analysis"], False)
        self.assertIs(analysis["paper_claim_authorized"], False)

    def test_wrong_cell_count_fails_closed(self) -> None:
        manifest_patch, records_patch, identity_patch = self._patch_run(records=self.records[:-1])
        with (
            manifest_patch,
            records_patch,
            identity_patch,
            self.assertRaisesRegex(ArtifactError, "799 verified cells"),
        ):
            transfer.validate_heldout_model_transfer_run(self.root)

    def test_mixed_operator_arm_identity_fails_closed(self) -> None:
        mixed_operator = copy.deepcopy(_operator(self.manifest))
        mixed_operator["arms"]["acquire_representative"]["release_id"] = "method.acquire.another"
        mixed_identity = _run_identity(self.manifest, operator=mixed_operator)
        manifest_patch, records_patch, identity_patch = self._patch_run(identity=mixed_identity)
        with (
            manifest_patch,
            records_patch,
            identity_patch,
            self.assertRaisesRegex(ArtifactError, "mixed or invalid"),
        ):
            transfer.validate_heldout_model_transfer_run(self.root)

    def test_missing_dependency_receipt_fails_closed(self) -> None:
        records = list(self.records)
        records[0] = copy.deepcopy(records[0])
        del records[0]["result"]["payload"]["observed_dependency_hashes"]
        manifest_patch, records_patch, identity_patch = self._patch_run(records=tuple(records))
        with (
            manifest_patch,
            records_patch,
            identity_patch,
            self.assertRaisesRegex(ArtifactError, "dependency receipt"),
        ):
            transfer.validate_heldout_model_transfer_run(self.root)

    def test_missing_source_receipt_fails_closed(self) -> None:
        records = list(self.records)
        records[0] = copy.deepcopy(records[0])
        del records[0]["result"]["payload"]["checkpoint_identity"]
        manifest_patch, records_patch, identity_patch = self._patch_run(records=tuple(records))
        with (
            manifest_patch,
            records_patch,
            identity_patch,
            self.assertRaisesRegex(ArtifactError, "source receipt"),
        ):
            transfer.validate_heldout_model_transfer_run(self.root)

    def test_optional_primary_run_reports_only_matched_interaction_boundary(
        self,
    ) -> None:
        manifest_patch, records_patch, identity_patch = self._patch_run()
        with manifest_patch, records_patch, identity_patch:
            verified_transfer = transfer._load_and_validate_heldout_run(self.root)
        primary_manifest = replace(
            self.manifest,
            study_id="primary.common_support",
            stage=ExperimentStage.CONFIRMATORY,
            model_slots=("anchor_a", "anchor_b"),
        )
        primary_dependencies = {
            f"state_bench.common_support::{model_slot}": {
                **_DEPENDENCIES,
                "agent_model": digest,
            }
            for model_slot, digest in (
                ("anchor_a", "e" * 64),
                ("anchor_b", "f" * 64),
            )
        }
        verified_primary = transfer._VerifiedTransferRun(
            root=(Path(self.temporary.name) / "primary").resolve(),
            manifest=primary_manifest,
            records=_primary_metric_records(primary_manifest),
            clusters=frozenset(),
            representative_hash=_REPRESENTATIVE_HASH,
            baseline_arm_id="no_op",
            arms_by_family=verified_transfer.arms_by_family,
            binding_receipt=verified_transfer.binding_receipt,
            runtime_receipts={
                "dependency_receipt_strata": primary_dependencies,
            },
        )
        with (
            patch.object(
                transfer,
                "_load_and_validate_heldout_run",
                return_value=verified_transfer,
            ),
            patch.object(
                transfer,
                "_load_and_validate_primary_run",
                return_value=verified_primary,
            ),
        ):
            analysis = transfer.analyze_heldout_model_transfer(
                self.root,
                metric="valid_success",
                primary_common_support_run_dir=verified_primary.root,
                bootstrap_samples=100,
                random_seed=13,
            )

        boundary = analysis["primary_common_support_boundary"]
        self.assertEqual(
            boundary["comparison_kind"],
            "matched_transfer_minus_primary_interaction_boundary",
        )
        self.assertEqual(boundary["matched_underlying_groups"], 200)
        self.assertTrue(boundary["dependency_boundary"]["boundary_verified"])
        self.assertIs(boundary["paper_claim_authorized"], False)
        self.assertIn("not a pooled population effect", boundary["scope_note"])


if __name__ == "__main__":
    unittest.main()
