from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agentinputlab.benchmarks import (
    ALE_BENCHMARK_ID,
    ALE_OFFICIAL_SOURCE_REVISION,
    ALECheckpoint,
    ALEEvaluationResult,
    ALEExternalTransferAdapter,
    CallerAssertedALEEvaluationBridge,
    build_ale_official_command,
)
from agentinputlab.contracts import ContextBlock, Message, SemanticFamily, ValidationError


def _checkpoint(*, with_context: bool = True) -> ALECheckpoint:
    query = "Prepare the audited quarterly revenue workbook and memo."
    return ALECheckpoint(
        checkpoint_id="ale-fixture-1",
        task_id="business_finance/fixture",
        domain="business_finance",
        query=query,
        instruction=query,
        visible_messages=(Message("user", query),),
        visible_tool_names=("shell", "computer"),
        visible_read_tool_names=("shell",),
        visible_context=(
            (
                ContextBlock("Visible workbook instructions", block_id="instructions")
                if with_context
                else None
            ),
        )
        if with_context
        else (),
        run_handle="external-run-fixture",
        task_source_sha256="1" * 64,
        environment_snapshot_sha256="2" * 64,
        grader_sha256="3" * 64,
        agent_harness_sha256="4" * 64,
        provider_identity_sha256="5" * 64,
        image_identity_sha256="6" * 64,
    )


class ALEBenchmarkTests(unittest.TestCase):
    def test_four_family_external_transfer_never_claims_clarification(self) -> None:
        checkpoint = _checkpoint()
        support = checkpoint.family_eligibility
        self.assertEqual(
            set(support.eligible_families),
            {
                SemanticFamily.REFORMULATE,
                SemanticFamily.DECOMPOSE,
                SemanticFamily.ACQUIRE,
                SemanticFamily.CURATE,
            },
        )
        self.assertFalse(support.eligibility[SemanticFamily.CLARIFY.value])

    def test_adapter_keeps_hidden_run_bindings_out_of_visible_state(self) -> None:
        bridge = CallerAssertedALEEvaluationBridge(
            bridge_id="ale-fixture",
            bridge_revision="a" * 64,
            supports_fresh_environment=True,
            supports_hidden_reference_staging=True,
            callback=lambda _payload, _checkpoint, _prediction: ALEEvaluationResult(
                task_score=0.75,
                cost_usd=1.25,
                wall_seconds=90,
            ),
        )
        adapter = ALEExternalTransferAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=bridge,
        )
        example = adapter.examples("public")[0]
        visible = example.visible_state().canonical_dict()
        self.assertEqual(adapter.benchmark_id, ALE_BENCHMARK_ID)
        self.assertNotIn("run_handle", str(visible))
        self.assertNotIn("grader_sha256", str(visible))
        self.assertEqual(
            dict(adapter.score(example, {"official_run_handle": "external-run-fixture"})),
            {"task_score": 0.75, "cost_usd": 1.25, "wall_seconds": 90.0},
        )
        self.assertFalse(adapter.supports_full_block(example.example_id))

    def test_bridge_refuses_missing_official_runtime_boundaries(self) -> None:
        with self.assertRaisesRegex(ValidationError, "fresh environments"):
            CallerAssertedALEEvaluationBridge(
                bridge_id="ale-invalid",
                bridge_revision="a" * 64,
                supports_fresh_environment=False,
                supports_hidden_reference_staging=True,
                callback=lambda *_: ALEEvaluationResult(1, 0, 1),
            )

    def test_native_command_keeps_ale_cli_as_execution_owner(self) -> None:
        checkout = Path("/tmp/ale-pinned")
        with (
            patch(
                "agentinputlab.benchmarks.ale.validate_ale_checkout",
                return_value=checkout,
            ),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "is_symlink", return_value=False),
        ):
            command = build_ale_official_command(
                checkout=checkout,
                experiment_config="/tmp/ale-experiment.yaml",
                dry_run=True,
            )
        self.assertEqual(
            command[:7],
            (
                "uv",
                "run",
                "python",
                "-m",
                "ale_run",
                "run",
                str(Path("/tmp/ale-experiment.yaml").resolve()),
            ),
        )
        self.assertEqual(command[-1], "--dry-run")

    def test_source_revision_is_frozen(self) -> None:
        checkpoint = _checkpoint(with_context=False)
        self.assertEqual(checkpoint.source_revision, ALE_OFFICIAL_SOURCE_REVISION)
        self.assertNotIn(SemanticFamily.CURATE, checkpoint.family_eligibility.eligible_families)


if __name__ == "__main__":
    unittest.main()
