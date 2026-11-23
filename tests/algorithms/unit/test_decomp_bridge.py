from __future__ import annotations

import unittest
from dataclasses import replace

from agentinputlab.algorithms.decompose.decomp import (
    DECOMP_REVISION,
    DecompFrontierEvent,
    DecompFullBridge,
    DecompGraphNode,
    DecompParticipantCall,
    DecompWorkerResponse,
    validate_decomp_full_response,
)
from agentinputlab.algorithms.decompose.decomp.worker_bridge import decomp_context_hash
from agentinputlab.contracts import (
    BudgetExceededError,
    ContextBlock,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.runner import CompositeInterventionRunner


def valid_response(
    case_id: str = "case",
    *,
    paragraph_ids: tuple[str, ...] = (),
    context_hash: str | None = None,
) -> DecompWorkerResponse:
    return DecompWorkerResponse(
        case_id=case_id,
        profile_id="plan.decomp.full",
        upstream_revision=DECOMP_REVISION,
        paragraph_ids=paragraph_ids,
        context_hash=context_hash or decomp_context_hash((), ()),
        nodes=(
            DecompGraphNode("root", "main question"),
            DecompGraphNode("sub", "subquestion", "subanswer", "qa", 1),
        ),
        dependency_edges=(("root", "sub"),),
        participant_calls=(
            DecompParticipantCall(0, "question_gen", "s0", "s1", "decompose"),
            DecompParticipantCall(1, "qa", "s1", "s2", "answer"),
        ),
        frontier=(
            DecompFrontierEvent(0, "s0", 0.0, "decompose", "pop"),
            DecompFrontierEvent(1, "s1", 1.0, "[EOQ]", "push"),
        ),
        reasoning_chain="main -> sub -> answer",
        final_answer="answer",
        termination="[EOQ]",
        usage={
            "model_calls": 2,
            "retrieval_calls": 0,
            "input_tokens": 20,
            "output_tokens": 8,
        },
    )


class FakeDecompRuntime:
    def __init__(self) -> None:
        self.request = None

    def run(self, request):
        self.request = request
        return valid_response(
            request.case_id,
            paragraph_ids=request.paragraph_ids,
            context_hash=request.context_hash,
        )

    def fingerprint(self):
        return {"worker_id": "decomp-fixture", "image_digest": "sha256:fixture"}


class DecompBridgeTests(unittest.TestCase):
    def test_full_bridge_preserves_worker_trace_and_identity(self) -> None:
        runtime = FakeDecompRuntime()
        state = FrozenAgentState(
            query="main question",
            task_id="case",
            context=(
                ContextBlock(text="Title||First paragraph.", block_id="p1"),
                ContextBlock(text="Title 2||Second paragraph.", block_id="p2"),
            ),
        )
        result = DecompFullBridge(runtime, config_artifact_id="fixture-config").run(
            state, InterventionContext()
        )
        self.assertEqual(result.metadata["full_system"], True)
        self.assertEqual(result.trace.counters["participant_calls"], 2)
        self.assertEqual(result.final_answer, "answer")
        self.assertEqual(result.evidence, state.context)
        self.assertEqual(runtime.request.paras, tuple(block.text for block in state.context))
        self.assertEqual(runtime.request.paragraph_ids, ("p1", "p2"))
        self.assertEqual(result.metadata["context_hash"], runtime.request.context_hash)
        self.assertIn("main -> sub -> answer", result.metadata["reasoning_chain"])

    def test_missing_worker_is_typed_and_has_no_simplified_fallback(self) -> None:
        with self.assertRaises(ExternalDependencyError):
            DecompFullBridge(None, config_artifact_id="fixture-config").run(
                FrozenAgentState(query="q"), InterventionContext()
            )

    def test_full_bridge_has_no_patch_propose_path(self) -> None:
        self.assertFalse(
            hasattr(DecompFullBridge(None, config_artifact_id="fixture-config"), "propose")
        )

    def test_context_requires_stable_paragraph_ids(self) -> None:
        with self.assertRaisesRegex(ValidationError, "block_id"):
            DecompFullBridge(FakeDecompRuntime(), config_artifact_id="fixture-config").run(
                FrozenAgentState(
                    query="q",
                    context=(ContextBlock(text="evidence without identity"),),
                ),
                InterventionContext(),
            )

    def test_bridge_rejects_worker_context_identity_mismatch(self) -> None:
        class MismatchedRuntime(FakeDecompRuntime):
            def __init__(self, field: str) -> None:
                super().__init__()
                self.field = field

            def run(self, request):
                response = super().run(request)
                if self.field == "paragraph_ids":
                    return replace(response, paragraph_ids=("different",))
                return replace(response, context_hash="0" * 64)

        state = FrozenAgentState(
            query="q",
            context=(ContextBlock(text="evidence", block_id="p1"),),
        )
        for field in ("paragraph_ids", "context_hash"):
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, field):
                DecompFullBridge(
                    MismatchedRuntime(field),
                    config_artifact_id="fixture-config",
                ).run(state, InterventionContext())

    def test_component_output_cannot_satisfy_full_profile(self) -> None:
        with self.assertRaises(ValidationError):
            validate_decomp_full_response(replace(valid_response(), component_only=True))

    def test_full_profile_requires_terminal_answer(self) -> None:
        with self.assertRaisesRegex(ValidationError, "terminal final answer"):
            validate_decomp_full_response(replace(valid_response(), final_answer=""))

    def test_cycle_is_rejected(self) -> None:
        response = replace(
            valid_response(),
            dependency_edges=(("root", "sub"), ("sub", "root")),
        )
        with self.assertRaises(ValidationError):
            validate_decomp_full_response(response)

    def test_unpinned_revision_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            validate_decomp_full_response(
                replace(valid_response(), upstream_revision="moving-main")
            )

    def test_controlled_runner_rejects_hidden_decomp_runtime(self) -> None:
        runtime = FakeDecompRuntime()
        with self.assertRaisesRegex(ValidationError, "runtime dependencies"):
            CompositeInterventionRunner().run(
                DecompFullBridge(runtime, config_artifact_id="fixture-config"),
                FrozenAgentState(query="q"),
                InterventionContext(),
            )

    def test_context_runtime_uses_common_monitored_usage_envelope(self) -> None:
        record = CompositeInterventionRunner().run(
            DecompFullBridge(None, config_artifact_id="fixture-config"),
            FrozenAgentState(query="q", task_id="case"),
            InterventionContext(external_runtime=FakeDecompRuntime()),
        )
        self.assertEqual(record.usage.model_calls, 2)
        self.assertEqual(record.usage.retrieval_calls, 0)
        self.assertEqual(record.usage.input_tokens, 20)
        self.assertEqual(record.usage.output_tokens, 8)
        self.assertEqual(record.usage.unpriced_model_calls, 2)
        self.assertEqual(record.result.final_answer, "answer")

    def test_model_budget_rejects_before_decomp_worker_side_effect(self) -> None:
        class CountingRuntime(FakeDecompRuntime):
            def __init__(self) -> None:
                self.calls = 0

            def run(self, request):
                self.calls += 1
                return super().run(request)

        runtime = CountingRuntime()
        with self.assertRaisesRegex(BudgetExceededError, "model_calls"):
            CompositeInterventionRunner().run(
                DecompFullBridge(None, config_artifact_id="fixture-config"),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_model_calls=0),
                ),
                InterventionContext(external_runtime=runtime),
            )
        self.assertEqual(runtime.calls, 0)


if __name__ == "__main__":
    unittest.main()
