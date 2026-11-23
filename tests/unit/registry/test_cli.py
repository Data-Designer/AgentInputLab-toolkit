"""CLI behavior tests that require no model, network, or optional package."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from agentinputlab.cli.main import main

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "manifests" / "algorithms" / "registry.yaml"


class CliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--registry", str(SOURCE), *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_list_core_acquire_methods_as_json(self) -> None:
        code, stdout, stderr = self.invoke(
            "list",
            "--family",
            "acquire",
            "--tier",
            "core",
            "--json",
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(len(payload), 5)
        self.assertTrue(all(item["semantic_family"] == "acquire" for item in payload))

    def test_list_must_ship_releases(self) -> None:
        code, stdout, stderr = self.invoke(
            "list",
            "--releases",
            "--delivery-tier",
            "must_ship",
            "--json",
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(len(payload), 12)
        for item in payload:
            self.assertIn("entrypoint_available", item)
            self.assertIn("runnable", item)
            self.assertIn("availability", item)
            self.assertIn("blocker", item)

    def test_release_inventory_exposes_entrypoint_and_blocker_counts(self) -> None:
        code, stdout, stderr = self.invoke("list", "--releases", "--json")
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(sum(item["entrypoint_available"] for item in payload), 86)
        self.assertEqual(sum(item["availability"] == "locked" for item in payload), 1)
        self.assertTrue(all(item["blocker"] for item in payload if not item["runnable"]))

    def test_release_table_exposes_availability_columns(self) -> None:
        code, stdout, stderr = self.invoke("list", "--releases")
        self.assertEqual(code, 0, stderr)
        heading = stdout.splitlines()[0]
        self.assertIn("ENTRYPOINT", heading)
        self.assertIn("RUNNABLE", heading)
        self.assertIn("AVAILABILITY", heading)
        self.assertIn("BLOCKER", heading)

    def test_show_release(self) -> None:
        code, stdout, stderr = self.invoke("show", "rewrite.query2doc.official_cached")
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["method"]["method_id"], "rewrite.query2doc")
        self.assertEqual(
            payload["release"]["id"],
            "rewrite.query2doc.official_cached",
        )

    def test_doctor_checks_snapshot(self) -> None:
        code, stdout, stderr = self.invoke(
            "doctor",
            "--source",
            str(SOURCE),
            "--json",
        )
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["snapshot_synchronized"])
        self.assertEqual(payload["method_count"], 27)
        self.assertEqual(payload["entrypoint_count"], 86)
        self.assertEqual(payload["locked_count"], 1)

    def test_offline_smoke_command(self) -> None:
        code, stdout, stderr = self.invoke("smoke", "--json")
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["offline"])
        self.assertEqual(payload["top_document_id"], "fr")

    def test_selector_catalog_and_visible_features(self) -> None:
        code, stdout, stderr = self.invoke("selector", "list", "--json")
        self.assertEqual(code, 0, stderr)
        catalog = json.loads(stdout)
        self.assertEqual(len(catalog), 7)
        self.assertGreaterEqual(
            sum(item["kind"] == "predictor" for item in catalog),
            5,
        )

        code, stdout, stderr = self.invoke(
            "selector",
            "features",
            "--query",
            "Please verify the latest source and then compare it.",
            "--tool-name",
            "search",
        )
        self.assertEqual(code, 0, stderr)
        features = json.loads(stdout)
        self.assertEqual(features["access_domains"], ["visible_pre_action"])
        self.assertFalse(features["raw_text_persisted"])
        self.assertNotIn("Please verify", json.dumps(features))


if __name__ == "__main__":
    unittest.main()
