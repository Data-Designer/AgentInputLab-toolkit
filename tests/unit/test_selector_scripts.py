from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agentinputlab.selectors import (
    SelectorArtifactStore,
    SelectorDiscoveryStore,
    calibrate_selector_bundle,
    create_fitted_selector_bundle,
    freeze_selector_bundle,
)
from tests.unit.test_selector_discovery import _write_run
from tests.unit.test_selector_producer import (
    _FIT_CONFIGS,
    _calibration_config,
    _full_five_discovery_freeze,
)

_ROOT = Path(__file__).resolve().parents[2]


def _run_script(name: str, *arguments: str) -> dict[str, object]:
    environment = dict(os.environ)
    source_root = str(_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not current else f"{source_root}{os.pathsep}{current}"
    )
    completed = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "selectors" / name), *arguments],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AssertionError(f"{name} failed with exit {completed.returncode}:\n{completed.stderr}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise AssertionError(f"{name} did not emit one JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class SelectorProducerScriptTests(unittest.TestCase):
    def test_frozen_policy_manifest_script_uses_exact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = _full_five_discovery_freeze()
            discovery_store = SelectorDiscoveryStore(root / "discovery-store")
            discovery_identity = discovery_store.save(discovery)
            fitted = create_fitted_selector_bundle(
                discovery,
                selector_kind="ridge_utility",
                revision="script-policy-fit-v1",
                fit_config=_FIT_CONFIGS["ridge_utility"],
            )
            calibrated, _summary = calibrate_selector_bundle(
                fitted,
                discovery,
                revision="script-policy-cal-v1",
                calibration_config=_calibration_config("ridge_utility"),
            )
            frozen = freeze_selector_bundle(
                calibrated,
                revision="script-policy-frozen-v1",
            )
            selector_store = SelectorArtifactStore(root / "selector-store")
            fitted_identity = selector_store.save(fitted)
            calibrated_identity = selector_store.save(
                calibrated,
                previous_identity=fitted_identity,
            )
            frozen_identity = selector_store.save(
                frozen,
                previous_identity=calibrated_identity,
            )
            destination = root / "selector-policy.json"

            result = _run_script(
                "build_policy_manifest.py",
                "--discovery-store",
                str(discovery_store.root),
                "--discovery-revision",
                discovery_identity.revision,
                "--discovery-sha256",
                discovery_identity.sha256,
                "--selector-store",
                str(selector_store.root),
                "--frozen-selector-id",
                frozen_identity.selector_id,
                "--frozen-revision",
                frozen_identity.revision,
                "--frozen-sha256",
                frozen_identity.sha256,
                "--output",
                str(destination),
            )

            self.assertEqual(result["stage"], "frozen")
            self.assertEqual(len(result["sha256"]), 64)
            self.assertTrue(destination.is_file())
            self.assertFalse(result["experiments_executed"])

    def test_export_train_calibrate_freeze_command_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _write_run(root / "runs")
            discovery_store = root / "discovery-store"
            selector_store = root / "selector-store"
            export_config = root / "export.json"
            fit_config = root / "fit.json"
            guardrail_fit_config = root / "guardrail-fit.json"
            calibration_config = root / "calibration.json"
            _write_json(
                export_config,
                {
                    "schema_version": 1,
                    "revision": "script-discovery-v1",
                    "feature_extractor": {
                        "extractor_id": "visible_state_heuristic",
                        "extractor_revision": "v1",
                        "schema_id": "family_defect_features.v1",
                        "tokenization_revision": "unicode_word_cjk_v1",
                        "context_soft_limit_tokens": 8192,
                    },
                    "utility_definition": {
                        "utility_metric": "valid_success",
                        "algorithm_failure_utility": 0.0,
                        "aggregate": "mean",
                    },
                    "representative_by_family": {"acquire": "acquire.hyde.paper_reimplementation"},
                },
            )
            _write_json(
                fit_config,
                {
                    "schema_version": 1,
                    "selector_kind": "ridge_utility",
                    "fit": {"ridge": 0.01},
                },
            )
            _write_json(
                guardrail_fit_config,
                {
                    "schema_version": 1,
                    "selector_kind": "cost_risk_guardrail",
                    "fit": {
                        "max_cost": 1.0,
                        "max_risk": 1.0,
                        "cost_weight": 0.0,
                        "risk_weight": 0.0,
                    },
                },
            )
            _write_json(
                calibration_config,
                {
                    "schema_version": 1,
                    "calibration": {
                        "gain_grid": [-1.0, 0.0],
                        "margin_grid": [0.0],
                        "max_harm_rate": 1.0,
                    },
                },
            )

            discovery = _run_script(
                "export_discovery.py",
                "--run-dir",
                str(run_dir),
                "--config",
                str(export_config),
                "--store",
                str(discovery_store),
            )
            discovery_identity = discovery["identity"]
            self.assertIsInstance(discovery_identity, dict)
            fitted = _run_script(
                "train_selector.py",
                "--discovery-store",
                str(discovery_store),
                "--discovery-revision",
                discovery_identity["revision"],
                "--discovery-sha256",
                discovery_identity["sha256"],
                "--selector-store",
                str(selector_store),
                "--config",
                str(fit_config),
                "--revision",
                "script-fit-v1",
            )
            fitted_identity = fitted["identity"]
            self.assertIsInstance(fitted_identity, dict)
            guardrail = _run_script(
                "train_selector.py",
                "--discovery-store",
                str(discovery_store),
                "--discovery-revision",
                discovery_identity["revision"],
                "--discovery-sha256",
                discovery_identity["sha256"],
                "--selector-store",
                str(selector_store),
                "--config",
                str(guardrail_fit_config),
                "--revision",
                "script-guardrail-fit-v1",
                "--base-selector-store",
                str(selector_store),
                "--base-selector-id",
                fitted_identity["selector_id"],
                "--base-revision",
                fitted_identity["revision"],
                "--base-sha256",
                fitted_identity["sha256"],
            )
            guardrail_artifact = json.loads(Path(guardrail["path"]).read_text())
            self.assertEqual(
                guardrail_artifact["fit_lineage"]["base_selector_bundle"]["sha256"],
                fitted_identity["sha256"],
            )
            self.assertEqual(
                guardrail_artifact["fit_lineage"]["discovery"]["sha256"],
                discovery_identity["sha256"],
            )
            calibrated = _run_script(
                "calibrate_selector.py",
                "--discovery-store",
                str(discovery_store),
                "--discovery-revision",
                discovery_identity["revision"],
                "--discovery-sha256",
                discovery_identity["sha256"],
                "--selector-store",
                str(selector_store),
                "--fitted-selector-id",
                fitted_identity["selector_id"],
                "--fitted-revision",
                fitted_identity["revision"],
                "--fitted-sha256",
                fitted_identity["sha256"],
                "--config",
                str(calibration_config),
                "--revision",
                "script-cal-v1",
            )
            calibrated_identity = calibrated["identity"]
            self.assertIsInstance(calibrated_identity, dict)
            frozen = _run_script(
                "freeze_selector.py",
                "--selector-store",
                str(selector_store),
                "--calibrated-selector-id",
                calibrated_identity["selector_id"],
                "--calibrated-revision",
                calibrated_identity["revision"],
                "--calibrated-sha256",
                calibrated_identity["sha256"],
                "--revision",
                "script-frozen-v1",
            )

            self.assertEqual(fitted["stage"], "fitted")
            self.assertEqual(calibrated["stage"], "calibrated")
            self.assertEqual(frozen["stage"], "frozen")
            self.assertFalse(discovery["experiments_executed"])
            self.assertFalse(fitted["experiments_executed"])
            self.assertFalse(calibrated["experiments_executed"])
            self.assertFalse(frozen["experiments_executed"])
            self.assertTrue(Path(frozen["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
