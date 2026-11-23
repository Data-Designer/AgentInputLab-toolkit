from __future__ import annotations

import dataclasses
import unittest
from typing import Any

from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.runtime import BenchmarkBatchRunner
from agentinputlab.benchmarks.statebench import (
    STATEBENCH_BENCHMARK_ID,
    STATEBENCH_EXTERNAL_SPLIT,
    STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS,
    STATEBENCH_OFFICIAL_ENVIRONMENT_CATALOG_REVISIONS,
    STATEBENCH_OFFICIAL_MAX_AGENT_TURNS,
    STATEBENCH_OFFICIAL_PROTOCOL_ID,
    STATEBENCH_OFFICIAL_PROTOCOL_REVISION,
    STATEBENCH_OFFICIAL_RUNS_PER_TASK,
    STATEBENCH_OFFICIAL_SIMULATOR_PROMPT_REVISIONS,
    STATEBENCH_OFFICIAL_SOURCE_REVISION,
    STATEBENCH_OFFICIAL_SPLIT_REVISIONS,
    STATEBENCH_OFFICIAL_TASK_CATALOG_REVISIONS,
    STATEBENCH_OFFICIAL_TASK_IDS,
    STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT,
    STATEBENCH_OFFICIAL_TASK_MEMBERSHIP_REVISION,
    STATEBENCH_OFFICIAL_TOOL_CATALOG_REVISIONS,
    STATEBENCH_OFFICIAL_TOOL_NAMES,
    CallerAssertedStateBenchEvaluationBridge,
    StateBenchCheckpoint,
    StateBenchCheckpointAdapter,
    StateBenchEvaluationResult,
)
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import canonical_json
from agentinputlab.contracts.state import ContextBlock, FrozenAgentState, Message

_TASK_IDS = {
    "customer_support": "7-return_restocking_waived",
    "shopping_assistant": "4-brand_bundle_missed",
    "travel": "4-strategy_cancel_rebook_cheaper_user_baits_change_state",
}


def _checkpoint(
    *,
    domain: str = "travel",
    turn_index: int = 0,
    run_index: int = 1,
    full: bool = False,
    capabilities: dict[str, bool] | None = None,
    checkpoint_id: str | None = None,
    task_id: str | None = None,
) -> StateBenchCheckpoint:
    query = "Please help me change this booking."
    messages = (
        (
            Message(role="user", content="I need help with my trip."),
            Message(role="assistant", content="What would you like to change?"),
            Message(role="user", content=query),
        )
        if full
        else (Message(role="user", content=query),)
    )
    if capabilities is None:
        capabilities = {
            "continuation_available": True,
            (
                "fresh_task_reset_available" if turn_index == 0 else "mid_dialogue_clone_available"
            ): True,
        }
        if full:
            capabilities["clarification_roundtrip_available"] = True
    return StateBenchCheckpoint(
        checkpoint_id=checkpoint_id or f"opaque-checkpoint-{domain}-{turn_index}",
        task_id=task_id or _TASK_IDS[domain],
        domain=domain,
        query=query,
        instruction="Use the official domain tools to help the user.",
        visible_messages=messages,
        visible_context=(
            (
                ContextBlock(text="Visible policy excerpt.", block_id="policy"),
                ContextBlock(text="Visible booking summary.", block_id="booking"),
            )
            if full
            else ()
        ),
        tool_names=STATEBENCH_OFFICIAL_TOOL_NAMES[domain],
        turn_index=turn_index,
        run_index=run_index,
        source_revision=STATEBENCH_OFFICIAL_SOURCE_REVISION,
        protocol_id=STATEBENCH_OFFICIAL_PROTOCOL_ID,
        protocol_revision=STATEBENCH_OFFICIAL_PROTOCOL_REVISION,
        task_catalog_revision=STATEBENCH_OFFICIAL_TASK_CATALOG_REVISIONS[domain],
        environment_catalog_revision=(STATEBENCH_OFFICIAL_ENVIRONMENT_CATALOG_REVISIONS[domain]),
        split_revision=STATEBENCH_OFFICIAL_SPLIT_REVISIONS[domain],
        tool_catalog_revision=STATEBENCH_OFFICIAL_TOOL_CATALOG_REVISIONS[domain],
        simulator_prompt_revision=(STATEBENCH_OFFICIAL_SIMULATOR_PROMPT_REVISIONS[domain]),
        runtime_snapshot_id="1" * 64,
        initial_environment_revision="2" * 64,
        active_environment_revision="3" * 64,
        simulator_session_identity="4" * 64,
        replay_identity="5" * 64,
        continuation_identity="6" * 64,
        structural_capabilities=capabilities,
    )


def _result() -> StateBenchEvaluationResult:
    return StateBenchEvaluationResult(
        task_completion_pass=1,
        state_requirements_met=1,
        task_requirements_met=1,
        ux_score=4.5,
        cost_usd=0.125,
        turns=4,
        tool_calls=6,
    )


def _bridge(
    callback: Any = None,
    *,
    fresh: bool = True,
    mid: bool = True,
    simulator: bool = True,
) -> CallerAssertedStateBenchEvaluationBridge:
    return CallerAssertedStateBenchEvaluationBridge(
        bridge_id="fixture.state-bench-runtime",
        bridge_revision="7" * 64,
        callback=callback or (lambda _checkpoint, _prediction: _result()),
        supports_fresh_task_reset=fresh,
        supports_mid_dialogue_clone=mid,
        supports_simulator_replay=simulator,
        preserves_domain_tools=True,
        preserves_user_simulator=True,
        preserves_scorer=True,
        preserves_turn_limit=True,
    )


class StateBenchBenchmarkTests(unittest.TestCase):
    def test_pinned_source_audit_has_450_tasks_and_three_domains(self) -> None:
        self.assertEqual(STATEBENCH_BENCHMARK_ID, "state_bench.common_support")
        self.assertEqual(STATEBENCH_EXTERNAL_SPLIT, "external_checkpoints")
        self.assertEqual(sum(STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS.values()), 450)
        self.assertEqual(
            STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS,
            {"customer_support": 150, "shopping_assistant": 150, "travel": 150},
        )
        self.assertEqual(
            {domain: len(task_ids) for domain, task_ids in STATEBENCH_OFFICIAL_TASK_IDS.items()},
            STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS,
        )
        self.assertEqual(
            {
                split: {domain: len(task_ids) for domain, task_ids in task_ids_by_domain.items()}
                for split, task_ids_by_domain in STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT.items()
            },
            {
                "test": {"customer_support": 50, "shopping_assistant": 50, "travel": 50},
                "train": {"customer_support": 100, "shopping_assistant": 100, "travel": 100},
            },
        )
        for domain, task_ids in STATEBENCH_OFFICIAL_TASK_IDS.items():
            train_ids = STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT["train"][domain]
            test_ids = STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT["test"][domain]
            self.assertTrue(train_ids.isdisjoint(test_ids))
            self.assertEqual(train_ids | test_ids, task_ids)
        self.assertEqual(
            STATEBENCH_OFFICIAL_TASK_MEMBERSHIP_REVISION,
            "1410974ccd3ea80e1d20425ba4b11539a0883eb2029b3b1023fa8627ed1b2870",
        )
        for domain, task_id in _TASK_IDS.items():
            self.assertIn(task_id, STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT["test"][domain])
        self.assertEqual(
            {domain: len(names) for domain, names in STATEBENCH_OFFICIAL_TOOL_NAMES.items()},
            {"customer_support": 11, "shopping_assistant": 18, "travel": 17},
        )
        self.assertEqual(STATEBENCH_OFFICIAL_RUNS_PER_TASK, 5)
        self.assertEqual(STATEBENCH_OFFICIAL_MAX_AGENT_TURNS, 15)
        for revisions in (
            STATEBENCH_OFFICIAL_TASK_CATALOG_REVISIONS,
            STATEBENCH_OFFICIAL_ENVIRONMENT_CATALOG_REVISIONS,
            STATEBENCH_OFFICIAL_SPLIT_REVISIONS,
            STATEBENCH_OFFICIAL_TOOL_CATALOG_REVISIONS,
            STATEBENCH_OFFICIAL_SIMULATOR_PROMPT_REVISIONS,
        ):
            self.assertEqual(set(revisions), set(STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS))
            self.assertTrue(all(len(value) == 64 for value in revisions.values()))

    def test_minimal_turn_zero_checkpoint_supports_three_families(self) -> None:
        checkpoint = _checkpoint()
        support = checkpoint.family_eligibility

        self.assertTrue(checkpoint.replay_ready)
        self.assertEqual(
            support.eligible_families,
            (
                SemanticFamily.REFORMULATE,
                SemanticFamily.DECOMPOSE,
                SemanticFamily.ACQUIRE,
            ),
        )
        self.assertFalse(support.supports_full_block)
        self.assertEqual(support.provenance_status, "caller_asserted")
        self.assertEqual(support.support_status, "structural_candidate")

    def test_full_mid_dialogue_checkpoint_supports_all_five_families(self) -> None:
        checkpoint = _checkpoint(turn_index=3, full=True)

        self.assertTrue(checkpoint.family_eligibility.supports_full_block)
        self.assertEqual(
            checkpoint.family_eligibility.eligible_families,
            (
                SemanticFamily.REFORMULATE,
                SemanticFamily.DECOMPOSE,
                SemanticFamily.ACQUIRE,
                SemanticFamily.CLARIFY,
                SemanticFamily.CURATE,
            ),
        )

    def test_last_official_turn_cannot_claim_clarification_round_trip(self) -> None:
        checkpoint = _checkpoint(turn_index=14, full=True)

        self.assertFalse(checkpoint.family_eligibility.eligibility[SemanticFamily.CLARIFY.value])
        self.assertTrue(checkpoint.family_eligibility.eligibility[SemanticFamily.CURATE.value])

    def test_unknown_or_gold_derived_capabilities_are_rejected(self) -> None:
        checkpoint = _checkpoint()
        for name in ("task_requirements_met", "hidden_defect", "gold_clarify"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValidationError,
                    "gold, defect labels, and task requirements",
                ),
            ):
                dataclasses.replace(
                    checkpoint,
                    structural_capabilities={name: True},
                )

    def test_checkpoint_rejects_source_protocol_and_tool_drift(self) -> None:
        checkpoint = _checkpoint()
        invalid_values = (
            {"source_revision": "8" * 40},
            {"protocol_revision": "8" * 64},
            {"tool_names": checkpoint.tool_names[:-1]},
            {"run_index": 6},
            {"turn_index": 15},
            {"task_id": "151-not-active"},
            {"task_id": "150-not_an_official_task"},
            {"task_id": "1-cancel_economy_domestic"},
            {"task_id": _TASK_IDS["customer_support"]},
        )
        for changes in invalid_values:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                dataclasses.replace(checkpoint, **changes)

    def test_checkpoint_requires_query_to_match_latest_visible_user_message(self) -> None:
        checkpoint = _checkpoint()

        with self.assertRaisesRegex(ValidationError, "latest visible user message"):
            dataclasses.replace(checkpoint, query="Different query")

    def test_adapter_fails_closed_without_reset_clone_or_continuation(self) -> None:
        cases = (
            (
                _checkpoint(
                    capabilities={
                        "continuation_available": True,
                        "fresh_task_reset_available": False,
                    }
                ),
                _bridge(),
                "lacks a continuation plus reset/clone",
            ),
            (
                _checkpoint(turn_index=3),
                _bridge(mid=False),
                "requires bridge clone support",
            ),
            (
                _checkpoint(),
                _bridge(simulator=False),
                "remaining user turn requires bridge simulator replay",
            ),
        )
        for checkpoint, bridge, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    ValidationError,
                    message,
                ),
            ):
                StateBenchCheckpointAdapter(
                    checkpoints=(checkpoint,),
                    evaluation_bridge=bridge,
                )

    def test_final_agent_turn_does_not_require_another_simulator_response(self) -> None:
        adapter = StateBenchCheckpointAdapter(
            checkpoints=(_checkpoint(turn_index=14),),
            evaluation_bridge=_bridge(simulator=False),
        )

        self.assertEqual(len(adapter.examples(STATEBENCH_EXTERNAL_SPLIT)), 1)

    def test_adapter_rejects_duplicate_experimental_unit_under_another_id(self) -> None:
        first = _checkpoint(checkpoint_id="checkpoint-a")
        duplicate = dataclasses.replace(first, checkpoint_id="checkpoint-b")

        with self.assertRaisesRegex(
            ValidationError,
            r"experimental units \(domain, task_id, run_index, turn_index\) must be unique",
        ):
            StateBenchCheckpointAdapter(
                checkpoints=(first, duplicate),
                evaluation_bridge=_bridge(),
            )

    def test_checkpoint_repr_omits_protected_task_and_runtime_identities(self) -> None:
        checkpoint = _checkpoint()
        rendered = repr(checkpoint)

        self.assertIn(checkpoint.checkpoint_id, rendered)
        for field_name in (
            "task_id",
            "runtime_snapshot_id",
            "initial_environment_revision",
            "active_environment_revision",
            "simulator_session_identity",
            "replay_identity",
            "continuation_identity",
        ):
            with self.subTest(field_name=field_name):
                self.assertNotIn(field_name, rendered)
                self.assertNotIn(str(getattr(checkpoint, field_name)), rendered)

    def test_bare_callable_is_not_an_evaluation_bridge(self) -> None:
        with self.assertRaisesRegex(TypeError, "StateBenchEvaluationBridge"):
            StateBenchCheckpointAdapter(
                checkpoints=(_checkpoint(),),
                evaluation_bridge=lambda *_args: _result(),  # type: ignore[arg-type]
            )

    def test_visible_state_excludes_task_gold_and_opaque_runtime_bindings(self) -> None:
        calls: list[tuple[StateBenchCheckpoint, Any]] = []

        def evaluate(
            checkpoint: StateBenchCheckpoint,
            prediction: Any,
        ) -> StateBenchEvaluationResult:
            calls.append((checkpoint, prediction))
            self.assertEqual(checkpoint.task_id, _TASK_IDS["travel"])
            return _result()

        checkpoint = _checkpoint(turn_index=3, full=True)
        adapter = StateBenchCheckpointAdapter(
            checkpoints=(checkpoint,),
            evaluation_bridge=_bridge(evaluate),
        )
        seen: list[FrozenAgentState] = []

        def predict(state: FrozenAgentState) -> dict[str, str]:
            seen.append(state)
            serialized = canonical_json(state)
            for hidden in (
                checkpoint.checkpoint_id,
                checkpoint.task_id,
                checkpoint.runtime_snapshot_id,
                checkpoint.initial_environment_revision,
                checkpoint.active_environment_revision,
                checkpoint.simulator_session_identity,
                checkpoint.replay_identity,
                checkpoint.continuation_identity,
                "state_requirements",
                "task_requirements",
                "user_simulator",
            ):
                self.assertNotIn(hidden, serialized)
            return {"intervention": "visible-only patch"}

        record = BenchmarkBatchRunner().run(
            adapter,
            STATEBENCH_EXTERNAL_SPLIT,
            predict,
        )

        self.assertEqual(record.aggregate_metrics["task_completion_pass"], 1.0)
        self.assertEqual(record.aggregate_metrics["ux_score"], 4.5)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(adapter.supports_full_block(checkpoint.checkpoint_id))
        self.assertEqual(
            adapter.evaluation_provenance["source_revision"], checkpoint.source_revision
        )
        self.assertFalse(adapter.evaluation_provenance["official_runtime_verified"])

    def test_adapter_rejects_wrong_split_and_foreign_example(self) -> None:
        adapter = StateBenchCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(),
        )
        with self.assertRaisesRegex(ValidationError, "external_checkpoints"):
            adapter.examples("test")
        with self.assertRaisesRegex(ValidationError, "does not belong"):
            adapter.score(
                BenchmarkExample(example_id="foreign", query="visible"),
                {"prediction": "x"},
            )

    def test_bridge_rejects_bare_metric_mapping(self) -> None:
        bridge = _bridge(
            lambda _checkpoint, _prediction: {
                "task_completion_pass": 1,
                "ux_score": 5,
            }
        )
        adapter = StateBenchCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=bridge,
        )

        with self.assertRaisesRegex(ValidationError, "bare metric mappings"):
            adapter.score(adapter.examples(STATEBENCH_EXTERNAL_SPLIT)[0], {})

    def test_bridge_cannot_claim_or_drop_official_dependencies(self) -> None:
        bridge = _bridge()
        for changes in (
            {"source_revision": "8" * 40},
            {"protocol_revision": "8" * 64},
            {"preserves_domain_tools": False},
            {"preserves_user_simulator": False},
            {"preserves_scorer": False},
            {"preserves_turn_limit": False},
            {"cost_basis": "estimated_tokens"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                dataclasses.replace(bridge, **changes)

    def test_evaluation_result_enforces_official_metric_relations(self) -> None:
        invalid = (
            {"task_completion_pass": 1, "state_requirements_met": 0},
            {"task_completion_pass": True},
            {"ux_score": 5.1},
            {"cost_usd": float("nan")},
            {"turns": 0},
            {"turns": 16},
            {"tool_calls": -1},
        )
        baseline = {
            "task_completion_pass": 1,
            "state_requirements_met": 1,
            "task_requirements_met": 1,
            "ux_score": 4.5,
            "cost_usd": 0.1,
            "turns": 4,
            "tool_calls": 6,
        }
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                StateBenchEvaluationResult(**(baseline | changes))

    def test_public_mappings_and_adapter_metadata_are_immutable(self) -> None:
        with self.assertRaises(TypeError):
            STATEBENCH_OFFICIAL_DOMAIN_TASK_COUNTS["travel"] = 0  # type: ignore[index]
        with self.assertRaises(TypeError):
            STATEBENCH_OFFICIAL_TASK_IDS["travel"] = frozenset()  # type: ignore[index]
        with self.assertRaises(AttributeError):
            STATEBENCH_OFFICIAL_TASK_IDS["travel"].add("other")  # type: ignore[attr-defined]
        with self.assertRaises(TypeError):
            STATEBENCH_OFFICIAL_TASK_IDS_BY_SPLIT["test"]["travel"] = (  # type: ignore[index]
                frozenset()
            )

        adapter = StateBenchCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(),
        )
        example = adapter.examples(STATEBENCH_EXTERNAL_SPLIT)[0]
        with self.assertRaises(TypeError):
            example.visible_metadata["domain"] = "other"  # type: ignore[index]
        with self.assertRaises(TypeError):
            adapter.evaluation_provenance["bridge_id"] = "other"  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
