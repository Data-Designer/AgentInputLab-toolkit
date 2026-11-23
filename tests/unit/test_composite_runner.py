from __future__ import annotations

import unittest
from dataclasses import dataclass

from agentinputlab.contracts import (
    AlgorithmOutputError,
    CompositeResult,
    ExecutionTrace,
    FrozenAgentState,
    InterventionContext,
    SemanticFamily,
    ValidationError,
)
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    MethodWorkerResponse,
)
from agentinputlab.observability import ResourceUsage
from agentinputlab.runner import CompositeInterventionRunner


@dataclass(slots=True)
class FixtureComposite:
    method_id: str = "fixture_composite"
    release_id: str = "fixture.composite.v1"
    family: SemanticFamily = SemanticFamily.COMPOSITE
    wrong_source_hash: bool = False

    def run(
        self,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> CompositeResult:
        del context
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash="0" * 64 if self.wrong_source_hash else state.state_hash,
            final_answer="fixture answer",
            evidence=(),
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                counters={"model_calls": 1, "input_tokens": 4, "output_tokens": 2},
            ),
            implementation_identity="fixture:v1",
            profile_kind="official_native",
        )


class CompositeRunnerTests(unittest.TestCase):
    def test_runner_preserves_terminal_result_and_reconciles_usage(self) -> None:
        state = FrozenAgentState(query="question")
        record = CompositeInterventionRunner().run(
            FixtureComposite(),
            state,
            InterventionContext(),
        )

        self.assertEqual(record.result.final_answer, "fixture answer")
        self.assertEqual(record.input_state_hash, state.state_hash)
        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual(record.usage.input_tokens, 4)
        self.assertEqual(record.usage.output_tokens, 2)

    def test_runner_rejects_wrong_source_state(self) -> None:
        with self.assertRaisesRegex(AlgorithmOutputError, "source_state_hash mismatch"):
            CompositeInterventionRunner().run(
                FixtureComposite(wrong_source_hash=True),
                FrozenAgentState(query="question"),
                InterventionContext(),
            )

    def test_method_worker_envelopes_validate_direct_construction(self) -> None:
        with self.assertRaisesRegex(ValidationError, "payload must be a mapping"):
            MethodWorkerRequest(
                method_id="fixture",
                release_id="fixture.v1",
                upstream_commit="fixture-commit",
                state_hash="a" * 64,
                payload=[],  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValidationError, "payload must be a mapping"):
            MethodWorkerResponse(
                request_id="request-id",
                upstream_commit="fixture-commit",
                protocol_version="worker.v1",
                payload=[],  # type: ignore[arg-type]
                usage=ResourceUsage(),
            )

    def test_composite_result_rejects_empty_terminal_answer(self) -> None:
        with self.assertRaisesRegex(ValidationError, "non-empty"):
            CompositeResult(
                method_id="fixture_composite",
                release_id="fixture.composite.v1",
                source_state_hash="a" * 64,
                final_answer="  ",
                evidence=(),
                trace=ExecutionTrace(
                    method_id="fixture_composite",
                    release_id="fixture.composite.v1",
                ),
                implementation_identity="fixture:v1",
                profile_kind="official_native",
            )


if __name__ == "__main__":
    unittest.main()
