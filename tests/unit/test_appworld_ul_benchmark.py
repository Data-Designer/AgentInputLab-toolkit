from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

from agentinputlab.benchmarks.appworld_ul import (
    APPWORLD_UL_PROJECT_URL,
    APPWORLD_UPSTREAM_REPOSITORY,
    AUDITED_APPWORLD_UPSTREAM_COMMIT,
    AppWorldULCheckpoint,
    AppWorldULCheckpointAdapter,
    AppWorldULEvaluationResult,
    CallerAssertedAppWorldULEvaluationBridge,
)
from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.runtime import BenchmarkBatchRunner
from agentinputlab.benchmarks.tau3 import Tau3EvaluationBridge
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import canonical_json
from agentinputlab.contracts.state import ContextBlock, FrozenAgentState, Message


def _checkpoint(
    *,
    full: bool = True,
    restore: bool = True,
    split: str = "test",
) -> AppWorldULCheckpoint:
    messages = (
        (
            Message(role="user", content="I need help buying an item."),
            Message(role="assistant", content="Which constraints should I use?"),
            Message(role="user", content="Buy a suitable option without overspending."),
        )
        if full
        else (Message(role="user", content="Buy a suitable option without overspending."),)
    )
    return AppWorldULCheckpoint(
        checkpoint_id=f"checkpoint-{split}",
        task_id=f"protected-task-{split}",
        split=split,
        query="Buy a suitable option without overspending.",
        instruction="Interact with the user and the visible AppWorld APIs.",
        visible_messages=messages,
        visible_context=(
            (
                ContextBlock(text="Visible preference A.", block_id="preference-a"),
                ContextBlock(text="Visible preference B.", block_id="preference-b"),
            )
            if full
            else ()
        ),
        tool_names=(
            ("api_docs.search", "amazon.search_products", "amazon.purchase")
            if full
            else ("amazon.purchase",)
        ),
        visible_read_tool_names=(("api_docs.search", "amazon.search_products") if full else ()),
        runtime_state_handle=f"opaque-state-handle-{split}",
        runtime_state_identity="1" * 64,
        user_simulator_identity="2" * 64,
        data_revision="3" * 64,
        tool_catalog_revision="4" * 64,
        evaluator_identity="5" * 64,
        replay_identity="6" * 64,
        structural_capabilities=(
            {
                "clarification_roundtrip_available": True,
                "state_restore_available": restore,
            }
            if full
            else {"state_restore_available": restore}
        ),
    )


def _bridge(
    callback: Any,
    *,
    supports_state_restore: bool = True,
    supports_user_simulator_replay: bool = True,
) -> CallerAssertedAppWorldULEvaluationBridge:
    return CallerAssertedAppWorldULEvaluationBridge(
        bridge_id="fixture.appworld-ul.external",
        bridge_revision="7" * 64,
        cost_basis="fixture_total_cost_units",
        supports_state_restore=supports_state_restore,
        supports_user_simulator_replay=supports_user_simulator_replay,
        callback=callback,
    )


class AppWorldULBenchmarkTests(unittest.TestCase):
    def test_bridge_has_nominal_arena_identity(self) -> None:
        bridge = _bridge(
            lambda *_: AppWorldULEvaluationResult(task_success=1.0, cost=0.0),
        )

        self.assertFalse(isinstance(bridge, Tau3EvaluationBridge))

    def test_source_pin_and_benchmark_identity_are_explicit(self) -> None:
        adapter = AppWorldULCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(
                lambda _gold, _checkpoint, _prediction: AppWorldULEvaluationResult(
                    task_success=1.0,
                    cost=0.0,
                )
            ),
        )

        self.assertEqual(adapter.benchmark_id, "appworld_ul.common_support")
        self.assertEqual(
            adapter.source_provenance,
            {
                "benchmark_id": "appworld_ul.common_support",
                "project_url": APPWORLD_UL_PROJECT_URL,
                "repository_url": APPWORLD_UPSTREAM_REPOSITORY,
                "upstream_commit": AUDITED_APPWORLD_UPSTREAM_COMMIT,
            },
        )

    def test_partial_checkpoint_supports_only_visible_families(self) -> None:
        support = _checkpoint(full=False, restore=False).family_eligibility

        self.assertEqual(
            support.eligible_families,
            (SemanticFamily.REFORMULATE, SemanticFamily.DECOMPOSE),
        )
        self.assertFalse(support.supports_full_block)
        self.assertEqual(support.provenance_status, "caller_asserted")
        self.assertEqual(support.support_status, "structural_candidate")

    def test_full_checkpoint_supports_five_families_and_paired_replay(self) -> None:
        checkpoint = _checkpoint(full=True, restore=True)

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
        self.assertTrue(checkpoint.family_eligibility.supports_full_block)
        self.assertTrue(checkpoint.structural_replay_candidate)

    def test_full_family_support_does_not_imply_restorable_paired_replay(self) -> None:
        checkpoint = _checkpoint(full=True, restore=False)

        self.assertTrue(checkpoint.family_eligibility.supports_full_block)
        self.assertFalse(checkpoint.structural_replay_candidate)

    def test_clarification_checkpoint_requires_bridge_simulator_replay(self) -> None:
        with self.assertRaisesRegex(ValidationError, "user-simulator replay support"):
            AppWorldULCheckpointAdapter(
                checkpoints=(_checkpoint(full=True, restore=False),),
                evaluation_bridge=_bridge(
                    lambda _gold, _checkpoint, _prediction: AppWorldULEvaluationResult(
                        task_success=1.0,
                        cost=0.0,
                    ),
                    supports_state_restore=True,
                    supports_user_simulator_replay=False,
                ),
            )

    def test_checkpoint_and_bridge_state_restore_capabilities_must_match(self) -> None:
        with self.assertRaisesRegex(ValidationError, "advertises state restoration"):
            AppWorldULCheckpointAdapter(
                checkpoints=(_checkpoint(full=True, restore=True),),
                evaluation_bridge=_bridge(
                    lambda _gold, _checkpoint, _prediction: AppWorldULEvaluationResult(
                        task_success=1.0,
                        cost=0.0,
                    ),
                    supports_state_restore=False,
                ),
            )

    def test_bridge_capability_declarations_must_be_boolean(self) -> None:
        def callback(
            _gold: Any,
            _checkpoint: Any,
            _prediction: Any,
        ) -> AppWorldULEvaluationResult:
            return AppWorldULEvaluationResult(
                task_success=1.0,
                cost=0.0,
            )

        for field_name in (
            "supports_state_restore",
            "supports_user_simulator_replay",
        ):
            kwargs = {
                "supports_state_restore": True,
                "supports_user_simulator_replay": True,
            }
            kwargs[field_name] = 1
            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(ValidationError, "must be boolean"),
            ):
                _bridge(callback, **kwargs)  # type: ignore[arg-type]

    def test_full_block_requires_bridge_state_restore_even_if_checkpoint_does_not(self) -> None:
        with self.assertRaisesRegex(ValidationError, "full-block checkpoint"):
            AppWorldULCheckpointAdapter(
                checkpoints=(_checkpoint(full=True, restore=False),),
                evaluation_bridge=_bridge(
                    lambda _gold, _checkpoint, _prediction: AppWorldULEvaluationResult(
                        task_success=1.0,
                        cost=0.0,
                    ),
                    supports_state_restore=False,
                ),
            )

    def test_task_and_defect_labels_cannot_drive_family_eligibility(self) -> None:
        checkpoint = _checkpoint()

        for forbidden in (
            "ambiguity_type",
            "clarification_opportunity",
            "infeasible_instruction",
            "hidden_defect_label",
        ):
            with (
                self.subTest(forbidden=forbidden),
                self.assertRaisesRegex(
                    ValidationError,
                    "task types and defect labels are not eligibility inputs",
                ),
            ):
                replace(checkpoint, structural_capabilities={forbidden: True})

    def test_read_tools_must_be_visible_tools(self) -> None:
        checkpoint = _checkpoint()

        with self.assertRaisesRegex(ValidationError, "must be a subset"):
            replace(checkpoint, visible_read_tool_names=("hidden_database.search",))

    def test_checkpoint_is_locked_to_the_audited_upstream_commit(self) -> None:
        checkpoint = _checkpoint()

        with self.assertRaisesRegex(ValidationError, "audited upstream commit"):
            replace(checkpoint, upstream_commit="a" * 40)

    def test_checkpoint_requires_bound_runtime_identities(self) -> None:
        checkpoint = _checkpoint()

        for field_name in (
            "runtime_state_identity",
            "user_simulator_identity",
            "data_revision",
            "tool_catalog_revision",
            "evaluator_identity",
            "replay_identity",
        ):
            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(ValidationError, "lowercase SHA-256"),
            ):
                replace(checkpoint, **{field_name: "not-a-hash"})

    def test_query_must_match_latest_visible_user_message(self) -> None:
        checkpoint = _checkpoint()

        with self.assertRaisesRegex(ValidationError, "latest visible user message"):
            replace(checkpoint, query="A hidden replacement query.")

    def test_evaluator_only_checkpoint_bindings_are_omitted_from_repr(self) -> None:
        rendered = repr(_checkpoint())

        for protected in (
            "protected-task-test",
            "opaque-state-handle-test",
            "runtime_state_identity",
            "user_simulator_identity",
            "evaluator_identity",
            "replay_identity",
            "1" * 64,
            "5" * 64,
            "6" * 64,
        ):
            self.assertNotIn(protected, rendered)

    def test_adapter_never_exposes_runtime_or_evaluator_bindings_to_predictor(self) -> None:
        evaluator_calls: list[tuple[Any, ...]] = []

        def evaluate(
            payload: Any,
            checkpoint: AppWorldULCheckpoint,
            prediction: Any,
        ) -> AppWorldULEvaluationResult:
            evaluator_calls.append((payload, checkpoint, prediction))
            self.assertEqual(payload["source_task_id"], "protected-task-test")
            self.assertEqual(payload["runtime_state_handle"], "opaque-state-handle-test")
            self.assertEqual(payload["evaluator_identity"], "5" * 64)
            return AppWorldULEvaluationResult(
                task_success=1.0,
                cost=2.5,
            )

        adapter = AppWorldULCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(evaluate),
        )
        seen: list[FrozenAgentState] = []

        def predict(state: FrozenAgentState) -> dict[str, str]:
            seen.append(state)
            serialized = canonical_json(state)
            for hidden in (
                "checkpoint-test",
                "protected-task-test",
                "opaque-state-handle-test",
                "runtime_state_identity",
                "user_simulator_identity",
                "evaluator_identity",
                "replay_identity",
                "1" * 64,
                "5" * 64,
                "6" * 64,
            ):
                self.assertNotIn(hidden, serialized)
            return {"action": "continue through the official runtime"}

        record = BenchmarkBatchRunner().run(adapter, "test", predict)

        self.assertEqual(record.example_count, 1)
        self.assertEqual(
            record.aggregate_metrics,
            {"cost": 2.5, "task_success": 1.0},
        )
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(evaluator_calls), 1)

    def test_evaluator_payload_and_provenance_are_immutable(self) -> None:
        adapter = AppWorldULCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(
                lambda _gold, _checkpoint, _prediction: AppWorldULEvaluationResult(
                    task_success=0.0,
                    cost=0.0,
                )
            ),
        )
        example = adapter.examples("test")[0]

        with self.assertRaises(TypeError):
            example.evaluator_gold["source_task_id"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            adapter.evaluation_provenance["bridge_id"] = "changed"  # type: ignore[index]

    def test_adapter_rejects_foreign_example_even_with_matching_id(self) -> None:
        adapter = AppWorldULCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=_bridge(
                lambda _gold, _checkpoint, _prediction: AppWorldULEvaluationResult(
                    task_success=1.0,
                    cost=0.0,
                )
            ),
        )
        example = adapter.examples("test")[0]
        forged = BenchmarkExample(
            example_id=example.example_id,
            query=example.query,
            instruction=example.instruction,
        )

        with self.assertRaisesRegex(ValidationError, "does not belong"):
            adapter.score(forged, {"action": "forged"})

    def test_duplicate_protected_experimental_unit_fails_closed(self) -> None:
        checkpoint = _checkpoint()
        duplicate = replace(
            checkpoint,
            checkpoint_id="checkpoint-duplicate",
            split="dev",
        )

        with self.assertRaisesRegex(ValidationError, "protected experimental units"):
            AppWorldULCheckpointAdapter(
                checkpoints=(checkpoint, duplicate),
                evaluation_bridge=_bridge(
                    lambda *_: AppWorldULEvaluationResult(task_success=1.0, cost=0.0)
                ),
            )

    def test_unknown_split_and_checkpoint_fail_closed(self) -> None:
        adapter = AppWorldULCheckpointAdapter(
            checkpoints=(_checkpoint(split="dev"),),
            evaluation_bridge=_bridge(
                lambda _gold, _checkpoint, _prediction: AppWorldULEvaluationResult(
                    task_success=1.0,
                    cost=0.0,
                )
            ),
        )

        self.assertEqual(adapter.available_splits, ("dev",))
        with self.assertRaisesRegex(ValidationError, "does not expose split"):
            adapter.examples("test")
        with self.assertRaisesRegex(ValidationError, "unknown AppWorld-UL checkpoint"):
            adapter.supports_full_block("missing")

    def test_bridge_rejects_bare_metric_mapping(self) -> None:
        bridge = _bridge(
            lambda _gold, _checkpoint, _prediction: {
                "cost": 0.0,
                "task_success": 1.0,
            }
        )
        adapter = AppWorldULCheckpointAdapter(
            checkpoints=(_checkpoint(),),
            evaluation_bridge=bridge,
        )

        with self.assertRaisesRegex(
            ValidationError,
            "bare metric mappings are not accepted",
        ):
            adapter.score(adapter.examples("test")[0], {"action": "done"})

    def test_current_bridge_cannot_claim_official_runtime_verification(self) -> None:
        class FalseOfficialBridge:
            appworld_ul_bridge_contract = "appworld_ul_evaluation_bridge_v1"
            benchmark_id = "appworld_ul.common_support"
            source_revision = AUDITED_APPWORLD_UPSTREAM_COMMIT
            bridge_id = "fixture.false-official"
            bridge_revision = "8" * 64
            cost_basis = "fixture_cost"
            supports_state_restore = True
            supports_user_simulator_replay = True
            provenance_status = "caller_asserted"
            evaluation_scope = "external_bridge"
            official_runtime_verified = True

            def evaluate(
                self,
                _payload: Any,
                _checkpoint: Any,
                _prediction: Any,
            ) -> AppWorldULEvaluationResult:
                return AppWorldULEvaluationResult(
                    task_success=1.0,
                    cost=0.0,
                )

        with self.assertRaisesRegex(ValidationError, "cannot claim official"):
            AppWorldULCheckpointAdapter(
                checkpoints=(_checkpoint(),),
                evaluation_bridge=FalseOfficialBridge(),
            )

    def test_task_metric_is_bounded_and_scenario_metric_is_not_fabricated(self) -> None:
        result = AppWorldULEvaluationResult(
            task_success=1,
            cost=3,
        )

        self.assertEqual(
            result.metrics,
            {"cost": 3.0, "task_success": 1.0},
        )
        self.assertNotIn("scenario_goal_completion", result.metrics)
        for value in (-0.1, 0.5, 1.1, float("inf"), True):
            with (
                self.subTest(value=value),
                self.assertRaises(ValidationError),
            ):
                AppWorldULEvaluationResult(
                    task_success=value,  # type: ignore[arg-type]
                    cost=0.0,
                )
        with self.assertRaisesRegex(ValidationError, "cost must be non-negative"):
            AppWorldULEvaluationResult(task_success=1.0, cost=-1.0)


if __name__ == "__main__":
    unittest.main()
