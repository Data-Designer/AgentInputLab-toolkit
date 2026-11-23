from __future__ import annotations

import unittest

from agentinputlab.algorithms.decompose.plan_and_budget import (
    CachedPlanBudgetRecord,
    DecaySchedule,
    PlanAndBudgetCachedSoftBudget,
    PlanAndBudgetRegeneratedSoftBudget,
    allocate_tokens,
    load_cached_record,
)
from agentinputlab.backends.llm import (
    CallableBackend,
    GenerationResponse,
    ModelCapability,
    Usage,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
)


class PlanAndBudgetTests(unittest.TestCase):
    def test_all_upstream_schedules_have_deterministic_allocations(self) -> None:
        credits = (50, 30, 20)
        expected = {
            DecaySchedule.CONSTANT: (5, 3, 2),
            DecaySchedule.LINEAR: (7, 3, 0),
            DecaySchedule.POLYNOMIAL: (8, 2, 0),
            DecaySchedule.EXPONENTIAL: (6, 3, 1),
            DecaySchedule.COSINE: (8, 2, 0),
        }
        for schedule, allocation in expected.items():
            with self.subTest(schedule=schedule):
                self.assertEqual(
                    allocate_tokens(credits, 10, schedule),
                    allocation,
                )
                self.assertEqual(sum(allocation), 10)

    def test_cached_anomalies_are_preserved_and_gold_is_physically_removed(self) -> None:
        record = load_cached_record(
            {
                "id": "472",
                "query": "task",
                "steps": ["a", "b", "c", "d", "e"],
                "credits": [15, 15, 20, 20, 10],
                "level": 3,
                "reference": "secret reference",
                "answer": "secret answer",
            }
        )
        self.assertEqual(sum(record.credits), 80)
        self.assertIn("cached_credit_sum_not_100", record.audit_warnings)
        self.assertIn("gold_fields_removed_from_solver_view", record.audit_warnings)
        self.assertFalse(hasattr(record, "reference"))
        self.assertFalse(hasattr(record, "answer"))

    def test_loader_rejects_misaligned_steps_and_credits(self) -> None:
        with self.assertRaises(ArtifactError):
            load_cached_record(
                {
                    "id": "bad",
                    "query": "q",
                    "steps": ["a", "b"],
                    "credits": [100],
                    "level": 1,
                }
            )

    def test_profile_materializes_soft_words_without_gold_or_hard_cap(self) -> None:
        record = CachedPlanBudgetRecord(
            record_id="x",
            query="Solve this",
            steps=("First step.", "Second step."),
            credits=(60, 40),
            level=3,
        )
        algorithm = PlanAndBudgetCachedSoftBudget(
            {"x": record},
            total_word_budget=10,
            schedule="constant",
        )
        state = FrozenAgentState(query="Solve this", task_id="x")
        patch = algorithm.propose(state, InterventionContext())
        updated = patch.apply(state)
        self.assertIn("up to 6 words", updated.instruction)
        self.assertIn("up to 4 words", updated.instruction)
        self.assertNotIn("Reference", updated.instruction)
        self.assertEqual(patch.metadata["budget_unit"], "soft_words")
        self.assertFalse(patch.metadata["hard_decoder_cap"])
        self.assertTrue(patch.metadata["gold_fields_removed"])

    def test_regenerated_profile_preserves_two_stage_upstream_protocol(self) -> None:
        responses = iter(
            (
                GenerationResponse(
                    text=(
                        "1. Identify the relevant evidence.\nHint: Read the paper.\n\n"
                        "2. Synthesize the answer.\nHint: Keep the response concise."
                    ),
                    usage=Usage(input_tokens=30, output_tokens=24),
                ),
                GenerationResponse(
                    text=(
                        '{"problem":{"evaluated_level":3},'
                        '"1":{"evaluated_level":2,"credit":40},'
                        '"2":{"evaluated_level":3,"credit":60}}'
                    ),
                    usage=Usage(input_tokens=40, output_tokens=18),
                ),
            )
        )
        backend = CallableBackend(
            lambda _request: next(responses),
            revision="fixture-v1",
            callable_id="plan-budget-two-stage-fixture",
            capabilities=frozenset(
                {
                    ModelCapability.CHAT,
                    ModelCapability.COMPLETION,
                    ModelCapability.JSON_OBJECT,
                    ModelCapability.SEED,
                }
            ),
        )
        algorithm = PlanAndBudgetRegeneratedSoftBudget(total_word_budget=10)
        state = FrozenAgentState(
            query="What does the paper conclude?",
            task_id="qasper-1",
            budget=ExecutionBudget(max_model_calls=3, max_output_tokens=512),
        )
        patch = algorithm.propose(state, InterventionContext(backend=backend, seed=17))
        updated = patch.apply(state)

        self.assertEqual([request.seed for request in backend.calls], [17, 18])
        self.assertIsNone(backend.calls[0].response_format)
        self.assertEqual(backend.calls[1].response_format, {"type": "json_object"})
        self.assertIn("up to 4 words", updated.instruction)
        self.assertIn("up to 6 words", updated.instruction)
        self.assertEqual(patch.profile_kind, "controlled_adapter")
        self.assertEqual(patch.trace.counters["model_calls"], 2)
        self.assertEqual(patch.trace.counters["input_tokens"], 70)
        self.assertEqual(patch.trace.counters["output_tokens"], 42)
        self.assertFalse(patch.metadata["paper_result_parity_claimed"])

    def test_regenerated_profile_preserves_non_100_credit_mass_with_warning(self) -> None:
        responses = iter(
            (
                "1. First.\nHint: A.\n2. Second.\nHint: B.",
                (
                    '{"problem":{"evaluated_level":3},'
                    '"1":{"evaluated_level":2,"credit":40},'
                    '"2":{"evaluated_level":3,"credit":50}}'
                ),
            )
        )
        backend = CallableBackend(
            lambda _request: next(responses),
            revision="fixture-v1",
            callable_id="plan-budget-invalid-credit-fixture",
            capabilities=frozenset(
                {
                    ModelCapability.COMPLETION,
                    ModelCapability.JSON_OBJECT,
                    ModelCapability.SEED,
                }
            ),
        )
        patch = PlanAndBudgetRegeneratedSoftBudget(total_word_budget=10).propose(
            FrozenAgentState(query="question"),
            InterventionContext(backend=backend),
        )
        self.assertIn("generated_credit_sum_not_100", patch.trace.warnings)
        self.assertEqual(patch.trace.counters["allocated_soft_words"], 10)

    def test_regenerated_profile_uses_dynamic_json_schema_when_available(self) -> None:
        responses = iter(
            (
                "1. First.\nHint: A.\n2. Second.\nHint: B.\n3. Third.\nHint: C.",
                (
                    '{"problem":{"evaluated_level":3},'
                    '"1":{"evaluated_level":2,"credit":30},'
                    '"2":{"evaluated_level":3,"credit":30},'
                    '"3":{"evaluated_level":4,"credit":40}}'
                ),
            )
        )
        backend = CallableBackend(
            lambda _request: next(responses),
            revision="fixture-v1",
            callable_id="plan-budget-json-schema-fixture",
            capabilities=frozenset(
                {
                    ModelCapability.COMPLETION,
                    ModelCapability.JSON_SCHEMA,
                    ModelCapability.SEED,
                }
            ),
        )
        PlanAndBudgetRegeneratedSoftBudget().propose(
            FrozenAgentState(query="question"),
            InterventionContext(backend=backend),
        )

        response_format = backend.calls[1].response_format
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["required"], ("problem", "1", "2", "3"))
        self.assertFalse(schema["additionalProperties"])

    def test_regenerated_profile_rejects_non_positive_credit(self) -> None:
        responses = iter(
            (
                "1. First.\nHint: A.\n2. Second.\nHint: B.",
                (
                    '{"problem":{"evaluated_level":3},'
                    '"1":{"evaluated_level":2,"credit":0},'
                    '"2":{"evaluated_level":3,"credit":100}}'
                ),
            )
        )
        backend = CallableBackend(
            lambda _request: next(responses),
            revision="fixture-v1",
            callable_id="plan-budget-zero-credit-fixture",
            capabilities=frozenset(
                {
                    ModelCapability.COMPLETION,
                    ModelCapability.JSON_OBJECT,
                    ModelCapability.SEED,
                }
            ),
        )
        with self.assertRaises(AlgorithmOutputError):
            PlanAndBudgetRegeneratedSoftBudget().propose(
                FrozenAgentState(query="question"),
                InterventionContext(backend=backend),
            )

    def test_regenerated_profile_requires_two_model_calls(self) -> None:
        backend = CallableBackend(
            lambda _request: "unused",
            revision="fixture-v1",
            callable_id="plan-budget-budget-fixture",
            capabilities=frozenset(
                {
                    ModelCapability.COMPLETION,
                    ModelCapability.JSON_OBJECT,
                    ModelCapability.SEED,
                }
            ),
        )
        state = FrozenAgentState(
            query="question",
            budget=ExecutionBudget(max_model_calls=1),
        )
        with self.assertRaises(BudgetExceededError):
            PlanAndBudgetRegeneratedSoftBudget().propose(
                state,
                InterventionContext(backend=backend),
            )
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
