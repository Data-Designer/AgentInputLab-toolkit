from __future__ import annotations

import hashlib
import unittest
from dataclasses import FrozenInstanceError
from importlib import resources

from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.toolsandbox import (
    OFFICIAL_BASE_SCENARIO_COUNT,
    OFFICIAL_SCENARIO_MEMBERSHIP_ASSET,
    OFFICIAL_SCENARIO_MEMBERSHIP_SHA256,
    OFFICIAL_SOURCE_REVISION,
    OFFICIAL_VARIANT_SCENARIO_COUNT,
    CallerAssertedToolSandboxEvaluationBridge,
    ToolSandboxCheckpoint,
    ToolSandboxCommonSupportAdapter,
    ToolSandboxEvaluationResult,
    ToolSandboxScenarioInventory,
)
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import canonical_json
from agentinputlab.contracts.state import Message


def _checkpoint(**overrides: object) -> ToolSandboxCheckpoint:
    values: dict[str, object] = {
        "checkpoint_id": "scenario-1@turn-3",
        "scenario_name": "find_temperature",
        "query": "What will the weather be after I confirm the city?",
        "instruction": "Complete the user's task.",
        "visible_messages": (
            Message(role="user", content="Can you check tomorrow's weather?"),
            Message(role="assistant", content="Which city should I use?"),
            Message(
                role="user",
                content="What will the weather be after I confirm the city?",
            ),
        ),
        "tool_names": ("search_weather_around_lat_lon", "end_conversation"),
        "visible_evidence_tool_names": ("search_weather_around_lat_lon",),
        "runtime_snapshot_id": "a" * 64,
        "replay_identity": "b" * 64,
        "snapshot_restore_available": True,
        "user_simulator_continuation_available": True,
    }
    values.update(overrides)
    return ToolSandboxCheckpoint(**values)  # type: ignore[arg-type]


def _bridge(
    result: ToolSandboxEvaluationResult | dict[str, float] | None = None,
) -> CallerAssertedToolSandboxEvaluationBridge:
    output = result or ToolSandboxEvaluationResult(
        milestone_similarity=0.75,
        minefield_similarity=0.0,
        similarity=0.75,
        turn_count=4,
        cost=0.03,
    )
    return CallerAssertedToolSandboxEvaluationBridge(
        bridge_id="official-runtime-wrapper.v1",
        bridge_revision="c" * 64,
        cost_basis="provider-reported USD",
        supports_snapshot_restore=True,
        supports_user_simulator_replay=True,
        callback=lambda checkpoint, prediction: output,  # type: ignore[return-value]
    )


class ToolSandboxTests(unittest.TestCase):
    def test_inventory_is_locked_to_the_audited_official_source(self) -> None:
        inventory = ToolSandboxScenarioInventory()
        assert inventory.source_revision == OFFICIAL_SOURCE_REVISION
        assert inventory.membership_revision == OFFICIAL_SCENARIO_MEMBERSHIP_SHA256
        assert inventory.base_scenario_count == OFFICIAL_BASE_SCENARIO_COUNT == 129
        assert inventory.variant_scenario_count == OFFICIAL_VARIANT_SCENARIO_COUNT == 1032
        assert len(inventory.base_scenario_names) == 129
        assert len(inventory.scenario_names) == 1032
        assert len(inventory.variant_suffixes) == 8
        assert inventory.contains("find_temperature")
        assert inventory.contains("find_temperature_3_distraction_tools_tool_name_scrambled")
        with self.assertRaisesRegex(ValidationError, "base-scenario count"):
            ToolSandboxScenarioInventory(base_scenario_count=128)

    def test_scenario_membership_asset_is_hash_locked(self) -> None:
        payload = (
            resources.files("agentinputlab.benchmarks")
            .joinpath("data", OFFICIAL_SCENARIO_MEMBERSHIP_ASSET)
            .read_bytes()
        )
        assert hashlib.sha256(payload).hexdigest() == OFFICIAL_SCENARIO_MEMBERSHIP_SHA256

    def test_checkpoint_accepts_official_variant_and_rejects_invented_name(self) -> None:
        variant = _checkpoint(
            scenario_name="find_temperature_3_distraction_tools_arg_type_scrambled"
        )
        assert variant.scenario_name.endswith("_arg_type_scrambled")
        with self.assertRaisesRegex(ValidationError, "pinned official inventory"):
            _checkpoint(scenario_name="invented_weather_scenario")

    def test_full_checkpoint_is_conditionally_eligible_for_all_five_families(self) -> None:
        support = _checkpoint().family_eligibility
        assert support.supports_full_block
        assert support.eligible_families == (
            SemanticFamily.REFORMULATE,
            SemanticFamily.DECOMPOSE,
            SemanticFamily.ACQUIRE,
            SemanticFamily.CLARIFY,
            SemanticFamily.CURATE,
        )
        assert support.provenance_status == "caller_asserted"
        assert support.support_status == "structural_candidate"

    def test_restore_user_simulator_evidence_and_history_gates_fail_closed(self) -> None:
        no_restore = _checkpoint(snapshot_restore_available=False).family_eligibility
        assert not any(no_restore.eligibility.values())

        no_simulator = _checkpoint(user_simulator_continuation_available=False).family_eligibility
        assert no_simulator.eligibility["reformulate"]
        assert not no_simulator.eligibility["clarify"]

        no_search = _checkpoint(
            tool_names=("send_message_with_phone_number", "end_conversation"),
            visible_evidence_tool_names=(),
        ).family_eligibility
        assert not no_search.eligibility["acquire"]

        short_history = _checkpoint(
            visible_messages=(
                Message(
                    role="user",
                    content="What will the weather be after I confirm the city?",
                ),
            )
        ).family_eligibility
        assert not short_history.eligibility["curate"]

    def test_checkpoint_requires_latest_visible_user_query_and_source_pin(self) -> None:
        with self.assertRaisesRegex(ValidationError, "latest visible user message"):
            _checkpoint(query="hidden evaluator request")
        with self.assertRaisesRegex(ValidationError, "source revision"):
            _checkpoint(source_revision="d" * 40)
        with self.assertRaisesRegex(ValidationError, "audited subset"):
            _checkpoint(visible_evidence_tool_names=("not_visible",))

    def test_checkpoint_is_immutable(self) -> None:
        checkpoint = _checkpoint()
        rendered = repr(checkpoint)
        assert checkpoint.runtime_snapshot_id not in rendered
        assert checkpoint.replay_identity not in rendered
        with self.assertRaises(FrozenInstanceError):
            checkpoint.query = "mutated"  # type: ignore[misc]

    def test_official_similarity_follows_minefield_nullification(self) -> None:
        result = ToolSandboxEvaluationResult(
            milestone_similarity=1.0,
            minefield_similarity=0.25,
            similarity=0.0,
            turn_count=5,
            cost=0,
        )
        assert result.metrics == {
            "cost": 0.0,
            "milestone_similarity": 1.0,
            "minefield_similarity": 0.25,
            "official_similarity": 0.0,
            "turn_count": 5.0,
        }
        with self.assertRaisesRegex(ValidationError, "nullification"):
            ToolSandboxEvaluationResult(
                milestone_similarity=1.0,
                minefield_similarity=0.25,
                similarity=1.0,
                turn_count=5,
                cost=0,
            )

    def test_adapter_exposes_visible_state_and_typed_official_metrics(self) -> None:
        adapter = ToolSandboxCommonSupportAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(),
        )
        assert adapter.benchmark_id == "toolsandbox.common_support"
        assert adapter.adapter_metric_names == (
            "milestone_similarity",
            "minefield_similarity",
            "official_similarity",
            "turn_count",
            "cost",
        )
        example = adapter.examples("external_checkpoints")[0]
        assert example.visible_state().task_id.startswith("ailcp_")
        assert example.visible_state().task_id != example.example_id
        assert "scenario-1@turn-3" not in canonical_json(example.visible_state())
        assert _checkpoint().scenario_name not in canonical_json(example.visible_state())
        assert example.evaluator_gold == {}
        assert example.visible_metadata["source_revision"] == OFFICIAL_SOURCE_REVISION
        assert adapter.supports_full_block(example.example_id)
        assert adapter.score(example, {"trajectory": "opaque"}) == {
            "cost": 0.03,
            "milestone_similarity": 0.75,
            "minefield_similarity": 0.0,
            "official_similarity": 0.75,
            "turn_count": 4.0,
        }
        assert adapter.evaluation_provenance["official_runtime_verified"] is False
        assert adapter.evaluation_provenance["pairing_mode"] == ("unsupported_external_replay")

    def test_adapter_rejects_bare_metric_mapping_and_foreign_examples(self) -> None:
        adapter = ToolSandboxCommonSupportAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(
                {
                    "milestone_similarity": 1.0,
                    "minefield_similarity": 0.0,
                }
            ),
        )
        example = adapter.examples("external_checkpoints")[0]
        with self.assertRaisesRegex(ValidationError, "ToolSandboxEvaluationResult"):
            adapter.score(example, object())
        with self.assertRaisesRegex(ValidationError, "does not belong"):
            adapter.score(BenchmarkExample(example_id=example.example_id, query="x"), object())
        with self.assertRaisesRegex(ValidationError, "exposes split"):
            adapter.examples("test")

    def test_adapter_rejects_duplicate_experimental_units(self) -> None:
        with self.assertRaisesRegex(ValidationError, "experimental units"):
            ToolSandboxCommonSupportAdapter(
                checkpoints=(
                    _checkpoint(),
                    _checkpoint(checkpoint_id="scenario-1@turn-3-duplicate"),
                ),
                evaluation_bridge=_bridge(),
            )

    def test_bare_callable_is_not_accepted_as_a_bridge(self) -> None:
        with self.assertRaisesRegex(TypeError, "CallerAsserted"):
            ToolSandboxCommonSupportAdapter(
                checkpoints=(_checkpoint(),),
                evaluation_bridge=lambda checkpoint, prediction: {},  # type: ignore[arg-type]
            )

    def test_bridge_cannot_understate_checkpoint_runtime_capability(self) -> None:
        cases = (
            (False, True, "snapshot restoration"),
            (True, False, "user-simulator continuation"),
        )
        for restore, simulator, match in cases:
            with self.subTest(restore=restore, simulator=simulator):
                bridge = CallerAssertedToolSandboxEvaluationBridge(
                    bridge_id="limited.v1",
                    bridge_revision="d" * 64,
                    cost_basis="provider-reported USD",
                    supports_snapshot_restore=restore,
                    supports_user_simulator_replay=simulator,
                    callback=lambda checkpoint, prediction: ToolSandboxEvaluationResult(
                        milestone_similarity=1,
                        minefield_similarity=0,
                        similarity=1,
                        turn_count=1,
                        cost=0,
                    ),
                )
                with self.assertRaisesRegex(ValidationError, match):
                    ToolSandboxCommonSupportAdapter(
                        checkpoints=(_checkpoint(),),
                        evaluation_bridge=bridge,
                    )

    def test_result_rejects_invalid_metrics(self) -> None:
        cases = (
            ("milestone_similarity", float("nan")),
            ("minefield_similarity", -0.1),
            ("similarity", 1.1),
            ("turn_count", -1),
            ("cost", -0.01),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                values: dict[str, float | int] = {
                    "milestone_similarity": 1.0,
                    "minefield_similarity": 0.0,
                    "similarity": 1.0,
                    "turn_count": 1,
                    "cost": 0.0,
                }
                values[field] = value
                with self.assertRaises(ValidationError):
                    ToolSandboxEvaluationResult(**values)  # type: ignore[arg-type]
