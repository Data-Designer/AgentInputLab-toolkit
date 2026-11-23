from __future__ import annotations

import json
import unittest
from pathlib import Path

from agentinputlab.benchmarks import BenchmarkRegistry


class BenchmarkManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.payload = json.loads(
            (root / "manifests/benchmarks/registry.json").read_text(encoding="utf-8")
        )

    def test_inventory_has_exactly_31_unique_cells(self) -> None:
        cells = self.payload["cells"]
        self.assertEqual(self.payload["cell_count"], 31)
        self.assertEqual(len(cells), 31)
        self.assertEqual(len({cell["cell_id"] for cell in cells}), 31)

    def test_all_runtime_families_have_eligible_cells(self) -> None:
        families = {family for cell in self.payload["cells"] for family in cell["families"]}
        self.assertEqual(
            families,
            {"reformulate", "decompose", "acquire", "clarify", "curate"},
        )

    def test_no_cell_uses_old_tool_family(self) -> None:
        serialized = json.dumps(self.payload)
        self.assertNotIn("select_tools", serialized.lower())

    def test_packaged_registry_is_queryable_by_family(self) -> None:
        registry = BenchmarkRegistry.default()
        self.assertEqual(len(registry), 31)
        self.assertGreater(len(registry.list(family="clarify")), 0)

    def test_tooltalk_matches_actual_provider_identity(self) -> None:
        from workers.official_arenas.providers.tooltalk import ToolTalkProvider

        cell = BenchmarkRegistry.default().get(ToolTalkProvider.benchmark_id)
        self.assertEqual(cell.source_revision, ToolTalkProvider.source_revision)
        self.assertEqual(cell.family_support, "checkpoint_conditional")
        self.assertIn("official_success", cell.adapter_metrics)
        self.assertIn("adapter_hard_stops", cell.adapter_metrics)

    def test_interactive_arenas_declare_conditional_support_and_metric_layers(self) -> None:
        registry = BenchmarkRegistry.default()
        adapter_metrics = {
            "tau3.banking_knowledge": ("official_reward", "cost"),
            "appworld_ul.common_support": ("task_success", "cost"),
            "state_bench.common_support": (
                "task_completion_pass",
                "state_requirements_met",
                "task_requirements_met",
                "ux_score",
                "cost_usd",
                "turns",
                "tool_calls",
            ),
            "toolsandbox.common_support": (
                "milestone_similarity",
                "minefield_similarity",
                "official_similarity",
                "turn_count",
                "cost",
            ),
        }
        for cell_id, expected_adapter_metrics in adapter_metrics.items():
            with self.subTest(cell_id=cell_id):
                cell = registry.get(cell_id)
                self.assertEqual(cell.family_support, "checkpoint_conditional")
                self.assertEqual(cell.adapter_metrics, expected_adapter_metrics)
                self.assertEqual(cell.paired_metrics, ("harm_rate",))
                self.assertEqual(cell.study_metrics, ("valid_success",))
                self.assertEqual(
                    set(cell.metrics),
                    {*cell.adapter_metrics, *cell.paired_metrics, *cell.study_metrics},
                )

    def test_ale_is_a_four_family_external_transfer_cell(self) -> None:
        cell = BenchmarkRegistry.default().get("ale.external_transfer")
        self.assertEqual(
            {family.value for family in cell.families},
            {"reformulate", "decompose", "acquire", "curate"},
        )
        self.assertEqual(cell.family_support, "checkpoint_conditional")
        self.assertNotIn("clarify", {family.value for family in cell.families})

    def test_retail_is_distinct_and_does_not_imply_acquisition_readiness(self) -> None:
        registry = BenchmarkRegistry.default()
        retail = registry.get("tau3.retail")
        banking = registry.get("tau3.banking_knowledge")
        self.assertEqual(retail.source_revision, banking.source_revision)
        self.assertEqual(retail.task, "retail")
        self.assertEqual({family.value for family in retail.families},
                         {"reformulate", "decompose", "clarify", "curate"})
        self.assertEqual(retail.adapter_metrics, ("official_reward", "terminal_success", "cost"))
        self.assertEqual(retail.paired_metrics, ("harm_rate",))
        self.assertEqual(retail.study_metrics, ("valid_success",))
        self.assertEqual(retail.family_support, "checkpoint_conditional")
        self.assertIn("acquisition remains unqualified", retail.eligibility_gate)

    def test_packaged_and_manifest_registry_remain_identical(self) -> None:
        root = Path(__file__).resolve().parents[2]
        packaged = json.loads((root / "src/agentinputlab/benchmarks/data/registry.json").read_text())
        self.assertEqual(packaged, self.payload)


if __name__ == "__main__":
    unittest.main()
