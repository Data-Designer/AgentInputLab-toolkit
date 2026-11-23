from __future__ import annotations

import unittest

from agentinputlab.algorithms.decompose.plan_and_budget import (
    PlanAndBudgetDecomposerComponent,
    PlanAndBudgetRegeneratedSoftBudget,
)
from agentinputlab.api import create_intervention
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, ModelCapability, Usage
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
    Message,
    ValidationError,
)
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

PLAN = "1. Locate the evidence.\nHint: Read the source.\n\n2. Combine it.\nHint: Answer the question."


def backend(text=PLAN, *, json_capability=True):
    responses = iter((text, '{"problem":{"evaluated_level":3},'
                      '"1":{"evaluated_level":2,"credit":40},'
                      '"2":{"evaluated_level":3,"credit":60}}'))
    return CallableBackend(lambda _: GenerationResponse(
        text=next(responses), usage=Usage(input_tokens=30, output_tokens=24)),
        revision="component-fixture-v1", callable_id="p-and-b-first-stage-fixture",
        capabilities=frozenset({ModelCapability.COMPLETION, ModelCapability.SEED}
            | ({ModelCapability.JSON_OBJECT} if json_capability else set())))


class PlanAndBudgetComponentTests(unittest.TestCase):
    def test_first_stage_request_is_identical_to_existing_controlled_full_profile(self):
        state = FrozenAgentState(query="Visible question", task_id="case",
            budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096))
        component_backend, full_backend = backend(), backend()
        component = PlanAndBudgetDecomposerComponent(decomposition_max_tokens=512)
        patch = component.propose(state, InterventionContext(backend=component_backend, seed=17))
        PlanAndBudgetRegeneratedSoftBudget(decomposition_max_tokens=512).propose(
            state, InterventionContext(backend=full_backend, seed=17))
        a, b = to_jsonable(component_backend.calls[0]), to_jsonable(full_backend.calls[0])
        # The auditable component release ID differs; no model-visible request field does.
        a["metadata"]["release_id"] = b["metadata"]["release_id"]
        self.assertEqual(a, b)
        self.assertEqual(len(component_backend.calls), 1)
        self.assertEqual(len(full_backend.calls), 2)
        self.assertFalse(patch.metadata["credit_assessment_executed"])
        self.assertFalse(patch.metadata["budget_allocation_executed"])
        self.assertFalse(patch.metadata["solver_executed"])
        self.assertNotIn("credits", patch.metadata)
        self.assertFalse(patch.metadata["paper_result_parity_claimed"])

    def test_one_call_completion_only_backend_and_original_reader_state_preserved(self):
        b = backend(json_capability=False)
        state = FrozenAgentState(query="Visible question", task_id="case",
            instruction="Preserve reader instruction", context=(ContextBlock(text="Reader context"),),
            messages=(Message(role="user", content="History for reader"),),
            budget=ExecutionBudget(max_model_calls=1, max_output_tokens=512))
        patch = PlanAndBudgetDecomposerComponent().propose(state, InterventionContext(backend=b))
        updated = patch.apply(state)
        self.assertEqual(updated.query, state.query)
        self.assertEqual(updated.context, state.context)
        self.assertEqual(updated.messages, state.messages)
        self.assertTrue(updated.instruction.startswith(state.instruction))
        self.assertIn("Suggested sub-questions:", updated.instruction)
        self.assertNotIn("Reader context", b.calls[0].prompt)
        self.assertNotIn("History for reader", b.calls[0].prompt)
        self.assertNotIn("Preserve reader instruction", b.calls[0].prompt)
        self.assertEqual(b.calls[0].max_tokens, 256)
        self.assertIsNone(b.calls[0].response_format)

    def test_source_parser_fallback_stays_explicit(self):
        patch = PlanAndBudgetDecomposerComponent().propose(FrozenAgentState(query="q"),
            InterventionContext(backend=backend("unstructured nonempty output")))
        self.assertTrue(patch.metadata["decomposition_fallback"])
        self.assertIn("decomposition_parse_fallback_to_direct_solve", patch.trace.warnings)
        self.assertFalse(patch.metadata["full_system"])

    def test_empty_or_excessive_steps_keep_paid_usage_on_failure(self):
        for text in ("", "\n".join(f"{i}. Step" for i in range(1, 7))):
            with self.subTest(text=text):
                b, monitor = backend(text), RunMonitor()
                with self.assertRaises(AlgorithmOutputError):
                    InterventionRunner(monitor=monitor).run(PlanAndBudgetDecomposerComponent(),
                        FrozenAgentState(query="q"), InterventionContext(backend=b))
                usage = monitor.list_snapshots()[0].usage
                self.assertEqual((usage.model_calls, usage.input_tokens, usage.output_tokens), (1, 30, 24))
                self.assertEqual(len(b.calls), 1)

    def test_invalid_controls_and_absent_budget_do_not_generate(self):
        for values in ({"domain": ""}, {"initial_level": True}, {"initial_level": 6},
                       {"decomposition_max_tokens": 0}, {"decomposition_max_tokens": True},
                       {"temperature": float("nan")}, {"temperature": -1}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                PlanAndBudgetDecomposerComponent(**values)
        b = backend()
        with self.assertRaises(BudgetExceededError):
            PlanAndBudgetDecomposerComponent().propose(FrozenAgentState(query="q",
                budget=ExecutionBudget(max_model_calls=0)), InterventionContext(backend=b))
        self.assertEqual(b.calls, [])

    def test_public_factory_resolves_exact_component_release(self):
        method = create_intervention("plan.plan_and_budget.decomposer_component",
                                     decomposition_max_tokens=512)
        self.assertIsInstance(method, PlanAndBudgetDecomposerComponent)
