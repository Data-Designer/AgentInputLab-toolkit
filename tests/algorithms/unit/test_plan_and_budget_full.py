"""Offline contract fixtures, not model/benchmark acceptance outcomes."""
from __future__ import annotations

import unittest

from agentinputlab.algorithms.decompose.plan_and_budget import (
    CachedPlanBudgetRecord,
    PlanAndBudgetCache,
    PlanAndBudgetFull,
)
from agentinputlab.api import create_intervention
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, ModelCapability, Usage
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    CapabilityError,
    ContextBlock,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    Message,
    ValidationError,
)
from agentinputlab.integrations.full_architecture.plan_and_budget import (
    PAB_NATIVE_PROMPT_MODE,
    PAB_SOURCE_HASHES,
    PlanAndBudgetNativePromptRuntime,
    PlanBudgetPromptRequest,
)
from agentinputlab.runner import CompositeInterventionRunner

REVISION = "2ca2d90c6d3d630ec2b11cc22e0f56751f7f2b7e"


class PromptFixture:
    def __init__(self):
        self.requests = []

    def fingerprint(self):
        return {"binding_mode": PAB_NATIVE_PROMPT_MODE, "upstream_revision": REVISION}

    def run(self, request):
        self.requests.append(request)
        return {
            "messages": [{"role": "system", "content": request.instruction or "fixture"},
                         {"role": "user", "content": request.record.query + request.reference}],
            "allocations": (140, 210), "soft_word_budget": 350,
            "identity": self.fingerprint(),
            "usage": {"model_calls": 0, "retrieval_calls": 0,
                      "input_tokens": 0, "output_tokens": 0},
        }


def record():
    return CachedPlanBudgetRecord("case-1", "Question?", ("First", "Second"), (40, 60), 3)


def state(**kwargs):
    return FrozenAgentState(query="Question?", task_id="case-1", **kwargs)


def backend(*responses):
    iterator = iter(responses or (GenerationResponse("fixture answer", usage=Usage(12, 6)),))
    return CallableBackend(
        lambda _request: next(iterator), revision="fixture-v1", callable_id="pab-full-fixture",
        capabilities=frozenset({ModelCapability.CHAT, ModelCapability.COMPLETION,
                               ModelCapability.JSON_OBJECT, ModelCapability.SEED}),
    )


class PlanBudgetFullTests(unittest.TestCase):
    def test_registered_cached_composite_and_observed_zero_cost_prompt_boundary(self):
        method = create_intervention("plan.plan_and_budget.full", output_format="Answer text",
                                     records={"case-1": record()})
        runtime, model = PromptFixture(), backend()
        original = state(context=(ContextBlock("Visible only", block_id="doc-1"),),
                         metadata={"gold": "NOT VISIBLE"})
        result = CompositeInterventionRunner().run(
            method, original, InterventionContext(backend=model, external_runtime=runtime, seed=17)
        )
        self.assertIsInstance(method, PlanAndBudgetFull)
        self.assertEqual(result.result.final_answer, "fixture answer")
        self.assertEqual(result.result.evidence, original.context)
        self.assertEqual(result.result.source_state_hash, original.state_hash)
        self.assertEqual(result.usage.model_calls, 1)
        self.assertEqual((result.usage.input_tokens, result.usage.output_tokens), (12, 6))
        self.assertEqual(model.calls[0].seed, 17)
        self.assertEqual(model.calls[0].max_tokens, 512)
        self.assertEqual(runtime.requests[0].reference, "[doc-1] Visible only\nSupport ID: doc-1")
        self.assertNotIn("NOT VISIBLE", repr(model.calls))
        self.assertEqual(result.result.metadata["solver_calls"], 1)
        self.assertFalse(result.result.metadata["per_step_hard_token_cap"])

    def test_live_three_call_path_and_native_allocation_not_ported_allocation(self):
        model = backend(
            GenerationResponse("1. First.\nHint: A\n2. Second.\nHint: B", usage=Usage(10, 20)),
            GenerationResponse('{"problem":{"evaluated_level":3},'
                               '"1":{"evaluated_level":2,"credit":40},'
                               '"2":{"evaluated_level":3,"credit":60}}', usage=Usage(20, 15)),
            GenerationResponse("Unchanged final text\n", usage=Usage(30, 9)),
        )
        runtime = PromptFixture()
        result = CompositeInterventionRunner().run(
            PlanAndBudgetFull(output_format="Answer text"), state(),
            InterventionContext(backend=model, external_runtime=runtime, seed=17),
        )
        self.assertEqual([r.seed for r in model.calls], [17, 18, 19])
        self.assertEqual(result.usage.model_calls, 3)
        self.assertEqual((result.usage.input_tokens, result.usage.output_tokens), (60, 44))
        self.assertEqual(result.result.final_answer, "Unchanged final text\n")
        self.assertEqual(runtime.requests[0].record.credits, (40, 60))
        self.assertEqual(result.result.trace.events[-2].payload["total_soft_words"], 350)
        self.assertEqual(result.result.trace.counters["optimizer_model_calls"], 2)

    def test_cache_identity_and_misses_do_not_silently_regenerate(self):
        for task_id, query in (("absent", "Question?"), ("case-1", "Other")):
            with self.subTest(task_id=task_id, query=query):
                model = backend()
                with self.assertRaisesRegex(ArtifactError, "cached ID/query"):
                    PlanAndBudgetFull(output_format="text", records={"case-1": record()}).run(
                        FrozenAgentState(query=query, task_id=task_id),
                        InterventionContext(backend=model, external_runtime=PromptFixture()),
                    )
                self.assertEqual(len(model.calls), 0)

    def test_pinned_cache_provenance_is_retained(self):
        cache = PlanAndBudgetCache({"case-1": record()}, "a" * 64)
        result = PlanAndBudgetFull(output_format="text", records=cache).run(
            state(), InterventionContext(backend=backend(), external_runtime=PromptFixture())
        )
        self.assertEqual(result.metadata["cache_artifact_sha256"], "a" * 64)
        self.assertEqual(result.metadata["cache_verification"], "caller_declared_digest")

    def test_insufficient_total_budget_and_non_qa_inputs_fail_before_models(self):
        for original in (state(budget=ExecutionBudget(max_model_calls=2)),
                         state(budget=ExecutionBudget(max_output_tokens=2)),
                         state(messages=(Message("user", "history"),)),
                         state(tool_names=("search",))):
            with self.subTest(original=original):
                model = backend()
                with self.assertRaises((BudgetExceededError, CapabilityError)):
                    PlanAndBudgetFull(output_format="text").run(
                        original, InterventionContext(backend=model, external_runtime=PromptFixture())
                    )
                self.assertEqual(len(model.calls), 0)

    def test_dependencies_preflight_before_spending_live_preparation(self):
        for runtime in (None, object()):
            model = backend()
            with self.assertRaises(ExternalDependencyError):
                PlanAndBudgetFull(output_format="text").run(
                    state(), InterventionContext(backend=model, external_runtime=runtime)
                )
            self.assertEqual(len(model.calls), 0)
        with self.assertRaises(ArtifactError):
            PlanAndBudgetNativePromptRuntime(source_root="/nonexistent-source").fingerprint()

    def test_empty_final_is_one_paid_failure_no_retry(self):
        model = backend(GenerationResponse("", usage=Usage(12, 512)))
        runner = CompositeInterventionRunner()
        with self.assertRaisesRegex(AlgorithmOutputError, "empty final answer"):
            runner.run(PlanAndBudgetFull(output_format="text", records={"case-1": record()}),
                       state(), InterventionContext(backend=model, external_runtime=PromptFixture()))
        self.assertEqual(len(model.calls), 1)
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 1)
        self.assertEqual(snapshot.usage.output_tokens, 512)

    def test_invalid_configuration_and_source_request(self):
        for kwargs in ({"output_format": ""}, {"output_format": "text", "max_solver_tokens": 0},
                       {"output_format": "text", "temperature": float("nan")},
                       {"output_format": "text", "max_solver_tokens": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                PlanAndBudgetFull(**kwargs)
        with self.assertRaises(ValidationError):
            PlanBudgetPromptRequest(record(), "", "", "", "constant")
        self.assertEqual(set(PAB_SOURCE_HASHES), {
            "inference/_5_planned_local_weighted_model.py", "utils/utils.py",
        })

    def test_small_budget_reserves_solver_and_further_shrinks_cap(self):
        model = backend(GenerationResponse("x", usage=Usage(1, 1)))
        result = PlanAndBudgetFull(output_format="text", records={"case-1": record()}).run(
            state(budget=ExecutionBudget(max_output_tokens=7)),
            InterventionContext(backend=model, external_runtime=PromptFixture()),
        )
        self.assertEqual(model.calls[0].max_tokens, 7)
        self.assertEqual(result.metadata["decoder_output_cap"], 7)

    def test_visible_official_support_ids_are_preserved_without_metadata_gold(self):
        runtime = PromptFixture()
        PlanAndBudgetFull(output_format="text", records={"case-1": record()}).run(
            state(context=(ContextBlock("Visible paragraph", block_id="p1", source="Article",
                metadata={"official_support_id": "Article:0", "gold": "not injected"}),)),
            InterventionContext(backend=backend(), external_runtime=runtime),
        )
        self.assertEqual(runtime.requests[0].reference,
            "[p1] Visible paragraph\nSupport ID: Article:0\nSource: Article")
        self.assertNotIn("not injected", runtime.requests[0].reference)


if __name__ == "__main__":
    unittest.main()
