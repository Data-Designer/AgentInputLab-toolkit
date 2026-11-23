from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments import (
    ExperimentBenchmarkTarget,
    ExperimentCell,
)
from agentinputlab.experiments import stateful_analysis as stateful
from agentinputlab.experiments.drivers.external_arena import (
    ExternalArenaExperimentDriver,
)
from tests.unit import test_external_arena_driver as arena_fixture


def _fixture_run(tmp_path: Path) -> tuple[stateful._VerifiedStatefulRun, list[dict]]:
    _reset_arena_fixture()
    config_path, config_sha256 = arena_fixture._write_config(
        tmp_path,
        arena_fixture._config(),
    )
    driver = ExternalArenaExperimentDriver.from_file(
        config_path,
        expected_sha256=config_sha256,
    )
    manifest = arena_fixture._manifest()
    units = ("checkpoint-1", "checkpoint-2")
    manifest = replace(
        manifest,
        benchmarks=(
            ExperimentBenchmarkTarget(
                cell_id="tau3.banking_knowledge",
                split="base",
                unit_ids=units,
                external=True,
                source_revision=arena_fixture._SOURCE_REVISION,
            ),
        ),
        driver_entrypoint=("agentinputlab.experiments.drivers.external_arena:factory"),
    )
    assert driver.preflight(manifest) == ()
    fingerprint = to_jsonable(driver.fingerprint())
    target = fingerprint["benchmark_targets"]["tau3.banking_knowledge:base"]
    gate = _gate_receipt(
        manifest_hash=manifest.manifest_hash,
        driver_config_sha256=config_sha256,
        bridge=target["bridge_lock"],
    )
    fingerprint["resolved_gate_bindings"] = {"tau3.banking_knowledge:base": gate}

    records: list[dict] = []
    for unit_id in units:
        cell = ExperimentCell(
            study_id=manifest.study_id,
            benchmark_id="tau3.banking_knowledge",
            split="base",
            unit_id=unit_id,
            model_slot="anchor_a",
            seed=17,
            arm_ids=tuple(arm.arm_id for arm in manifest.arms),
            manifest_hash=manifest.manifest_hash,
        )
        result = driver.execute_cell(manifest, cell).to_dict()
        result["payload"]["external_gate_receipt"] = copy.deepcopy(gate)
        records.append(
            {
                "cell": {**to_jsonable(cell), "cell_id": cell.cell_id},
                "result": result,
            }
        )
    run = stateful._VerifiedStatefulRun(
        root=tmp_path / "verified-run",
        manifest=manifest,
        records=tuple(records),
        verification={
            "manifest_hash": manifest.manifest_hash,
            "plan_hash": "b" * 64,
            "driver_hash": "c" * 64,
            "comparison_profile_hash": "d" * 64,
        },
        driver_fingerprint=fingerprint,
    )
    return run, records


def _gate_receipt(
    *,
    manifest_hash: str,
    driver_config_sha256: str,
    bridge: dict,
) -> dict:
    scope_sha256 = "e" * 64
    return {
        "file_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "scope_sha256": scope_sha256,
        "benchmark_id": "tau3.banking_knowledge",
        "split": "base",
        "source_revision": arena_fixture._SOURCE_REVISION,
        "manifest_hash": manifest_hash,
        "driver_config_sha256": driver_config_sha256,
        "bridge": {
            "bridge_id": bridge["bridge_id"],
            "bridge_revision": bridge["bridge_revision"],
        },
        "scorer": {
            "scorer_id": bridge["scorer_id"],
            "scorer_revision": bridge["scorer_revision"],
        },
        "approval": {
            "approved": True,
            "approver_id": "fixture-review-board",
            "approval_revision": "3" * 64,
            "approved_at": "2026-07-27T00:00:00Z",
            "scope_sha256": scope_sha256,
        },
        "verification": {
            "atomic_replay_contract": "agentinputlab.atomic_external_pair.v1",
            "reset_mode": "atomic_restore",
            "reset_verified": True,
            "reset_receipt_sha256": "4" * 64,
            "no_op_parity_verified": True,
            "no_op_parity_receipt_sha256": "5" * 64,
        },
        "unit_inventory_sha256": "6" * 64,
    }


def _reset_arena_fixture() -> None:
    arena_fixture.PAIR_REQUESTS.clear()
    arena_fixture.DRIFT_BASELINE = False
    arena_fixture.BRIDGE_ID_OVERRIDE = None
    arena_fixture.BUILDER_BLOCKER = None
    arena_fixture.SELECTOR_INPUT_DRIFT = False
    arena_fixture.INVALID_OUTCOME_MODE = None


class StatefulAnalysisTests(unittest.TestCase):
    def test_stateful_report_keeps_suite_estimand_relative_to_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run, _records = _fixture_run(tmp_path)
            with patch.object(
                stateful,
                "_load_verified_stateful_run",
                return_value=run,
            ):
                report = stateful.analyze_stateful_arena_replication(
                    (tmp_path / "verified-run",),
                    bootstrap_samples=100,
                    random_seed=7,
                )

        self.assertIs(report["experiments_executed_by_analysis"], False)
        self.assertIs(report["paper_claim_authorized"], False)
        self.assertIs(report["raw_metric_pooling_performed"], False)
        suite = report["suites"]["tau3.banking_knowledge"]
        self.assertIs(suite["gate_receipts"]["approved"], True)
        self.assertIs(suite["dependency_receipts"]["balanced"], True)
        self.assertIs(
            suite["balance_receipts"]["same_origin_no_op_verified"],
            True,
        )
        self.assertEqual(
            suite["treatments"]["fixture.rewrite.v1"]["mean_effect_vs_no_op"],
            0.5,
        )
        self.assertEqual(
            suite["treatments"]["fixture.acquire.v1"]["mean_effect_vs_no_op"],
            0.25,
        )
        cross_suite = report["cross_suite_robustness"]
        self.assertIs(cross_suite["raw_metric_pooling_performed"], False)
        self.assertNotIn("grand_mean", cross_suite)
        self.assertEqual(
            cross_suite["by_treatment"]["fixture.rewrite.v1"]["normalized_win_minus_harm_by_suite"],
            {"tau3.banking_knowledge": 1.0},
        )

    def test_record_validator_rejects_missing_atomic_or_dependency_receipts(
        self,
    ) -> None:
        for field, message in (
            ("atomic_pairs", "atomic pair"),
            ("observed_dependency_hashes", "dependency receipt"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp_dir:
                run, records = _fixture_run(Path(tmp_dir))
                record = copy.deepcopy(records[0])
                del record["result"]["payload"][field]

                with self.assertRaisesRegex(ArtifactError, message):
                    stateful.validate_external_arena_record_receipts(
                        run.manifest,
                        record,
                        metric="valid_success",
                    )

    def test_stateful_report_rejects_missing_gate_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run, records = _fixture_run(tmp_path)
            records = copy.deepcopy(records)
            for record in records:
                del record["result"]["payload"]["external_gate_receipt"]
            run = replace(run, records=tuple(records))

            with (
                patch.object(
                    stateful,
                    "_load_verified_stateful_run",
                    return_value=run,
                ),
                self.assertRaisesRegex(ArtifactError, "gate receipt"),
            ):
                stateful.analyze_stateful_arena_replication(
                    (tmp_path / "verified-run",),
                    bootstrap_samples=100,
                )

    def test_stateful_report_rejects_raw_metric_pooling_request(self) -> None:
        with self.assertRaisesRegex(ValidationError, "forbids pooling"):
            stateful.analyze_stateful_arena_replication(
                (Path("/not/read"),),
                pool_raw_metrics=True,
            )

    def test_record_validator_rejects_mixed_official_scorer_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run, records = _fixture_run(Path(tmp_dir))
            record = copy.deepcopy(records[0])
            treatment = record["result"]["payload"]["arms"][1]
            composition = treatment["valid_success_composition"]
            composition["scorer_id"] = "another.official-scorer"
            composition["composition_receipt_sha256"] = content_hash(
                {
                    key: value
                    for key, value in composition.items()
                    if key != "composition_receipt_sha256"
                }
            )

            with self.assertRaisesRegex(ArtifactError, "identity differs across arms"):
                stateful.validate_external_arena_record_receipts(
                    run.manifest,
                    record,
                    metric="valid_success",
                )

    def test_record_validator_rejects_mixed_outcome_contract_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run, records = _fixture_run(Path(tmp_dir))
            record = copy.deepcopy(records[0])
            treatment = record["result"]["payload"]["arms"][1]
            composition = treatment["valid_success_composition"]
            composition["outcome_contract_sha256"] = "f" * 64
            composition["composition_receipt_sha256"] = content_hash(
                {
                    key: value
                    for key, value in composition.items()
                    if key != "composition_receipt_sha256"
                }
            )

            with self.assertRaisesRegex(
                ArtifactError,
                "outcome contract identity differs",
            ):
                stateful.validate_external_arena_record_receipts(
                    run.manifest,
                    record,
                    metric="valid_success",
                )
