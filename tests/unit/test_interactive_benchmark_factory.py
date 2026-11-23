from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentinputlab.benchmarks import (
    create_interactive_benchmark_adapter,
    interactive_benchmark_specs,
    structural_full_block_checkpoint_ids,
)
from agentinputlab.contracts.errors import ValidationError


class InteractiveBenchmarkFactoryTests(unittest.TestCase):
    def test_inventory_is_registry_locked_and_not_pairing_attested(self) -> None:
        specs = interactive_benchmark_specs()

        self.assertEqual(
            tuple(spec.cell_id for spec in specs),
            (
                "tau3.banking_knowledge",
                "appworld_ul.common_support",
                "state_bench.common_support",
                "toolsandbox.common_support",
            ),
        )
        self.assertTrue(all(spec.structural_five_family_candidate for spec in specs))
        self.assertTrue(all(not spec.confirmatory_pairing_ready for spec in specs))
        self.assertTrue(all(spec.pairing_mode == "unsupported_external_replay" for spec in specs))

    def test_unknown_cell_and_arena_specific_arguments_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown interactive benchmark"):
            create_interactive_benchmark_adapter(
                "unknown.arena",
                checkpoints=(),
                evaluation_bridge=object(),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValidationError, "requires a prepared source bundle"):
            create_interactive_benchmark_adapter(
                "tau3.banking_knowledge",
                checkpoints=(),
                evaluation_bridge=object(),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValidationError, "only valid for tau3"):
            create_interactive_benchmark_adapter(
                "state_bench.common_support",
                checkpoints=(),
                evaluation_bridge=object(),  # type: ignore[arg-type]
                bundle=object(),  # type: ignore[arg-type]
            )

    @patch("agentinputlab.benchmarks.interactive.ToolSandboxCommonSupportAdapter")
    def test_toolsandbox_factory_forwards_optional_inventory(self, adapter_type: MagicMock) -> None:
        checkpoint = object()
        bridge = object()
        inventory = object()
        expected = object()
        adapter_type.return_value = expected

        result = create_interactive_benchmark_adapter(
            "toolsandbox.common_support",
            checkpoints=(checkpoint,),  # type: ignore[arg-type]
            evaluation_bridge=bridge,  # type: ignore[arg-type]
            inventory=inventory,  # type: ignore[arg-type]
        )

        adapter_type.assert_called_once_with(
            checkpoints=(checkpoint,),
            evaluation_bridge=bridge,
            inventory=inventory,
        )
        self.assertIs(result, expected)

    def test_structural_selection_is_explicitly_only_the_first_gate(self) -> None:
        examples = (
            SimpleNamespace(example_id="full"),
            SimpleNamespace(example_id="partial"),
        )
        adapter = MagicMock()
        adapter.examples.return_value = examples
        adapter.supports_full_block.side_effect = lambda checkpoint_id: checkpoint_id == "full"

        selected = structural_full_block_checkpoint_ids(adapter, "base")

        self.assertEqual(selected, ("full",))
        adapter.examples.assert_called_once_with("base")


if __name__ == "__main__":
    unittest.main()
