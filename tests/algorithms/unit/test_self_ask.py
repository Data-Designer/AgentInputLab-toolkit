from __future__ import annotations

import unittest

from agentinputlab.algorithms.decompose.self_ask import (
    SelfAskConfig,
    SelfAskLoop,
    parse_final_answer,
    parse_followup_question,
)
from agentinputlab.backends.llm.callable import CallableBackend, ReplayBackend
from agentinputlab.backends.retrieval.base import Document
from agentinputlab.backends.retrieval.memory import InMemoryBM25Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    FrozenAgentState,
    InterventionContext,
)

DEMONSTRATION = (
    "Question: Demo?\nAre follow up questions needed here: No.\nSo the final answer is: Demo."
)


class SelfAskTests(unittest.TestCase):
    def test_marker_parsers_use_last_typed_marker(self) -> None:
        self.assertEqual(
            parse_followup_question("Yes.\nFollow up: Where was Ada born?"),
            "Where was Ada born?",
        )
        self.assertEqual(
            parse_final_answer("So the final answer is: London"),
            "London",
        )

    def test_full_loop_retrieves_and_feeds_intermediate_answer(self) -> None:
        backend = ReplayBackend(
            (
                "Yes.\nFollow up: Where was Ada born?",
                "So the final answer is: London",
            )
        )
        retriever = InMemoryBM25Retriever(
            (Document(doc_id="d1", text="London", corpus_revision="fixture"),)
        )
        state = FrozenAgentState(query="Where was Ada born and what city is that?")
        algorithm = SelfAskLoop(SelfAskConfig(demonstrations=DEMONSTRATION))
        patch = algorithm.propose(
            state,
            InterventionContext(backend=backend, retriever=retriever, seed=9),
        )
        self.assertEqual(patch.metadata["full_loop"], True)
        self.assertEqual(patch.trace.counters["followups"], 1)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 1)
        self.assertEqual(
            patch.trace.events[1].payload["retrieval_records"][0]["corpus_revision"],
            "fixture",
        )
        updated = patch.apply(state)
        self.assertEqual(updated.metadata["self_ask_final_answer"], "London")
        self.assertIn("Intermediate answer: London", updated.context[-1].text)

    def test_empty_search_uses_explicit_model_fallback(self) -> None:
        backend = ReplayBackend(
            (
                "Yes.\nFollow up: Missing fact?",
                "fallback fact",
                "So the final answer is: finished",
            )
        )
        retriever = InMemoryBM25Retriever(())
        result = SelfAskLoop(SelfAskConfig(demonstrations=DEMONSTRATION)).run(
            FrozenAgentState(query="question"),
            InterventionContext(backend=backend, retriever=retriever),
        )
        self.assertEqual(result.model_fallbacks, 1)
        self.assertEqual(result.steps[0].answer_source, "model_fallback")
        self.assertEqual(result.final_answer, "finished")

    def test_final_reply_termination_records_marker_or_first_line_without_request_changes(self) -> None:
        for final_reply, expected_termination in (
            ("London\nAdditional explanation.", "first_line_fallback"),
            ("So the final answer is: London", "final_answer_marker"),
        ):
            with self.subTest(termination=expected_termination):
                responses = iter(("No.", final_reply))
                backend = CallableBackend(
                    fn=lambda _request, responses=responses: next(responses),
                    revision="self-ask-termination-test-v1",
                    callable_id="tests.self_ask.termination",
                )
                state = FrozenAgentState(query="Where was Ada born?")
                patch = SelfAskLoop(SelfAskConfig(demonstrations=DEMONSTRATION)).propose(
                    state,
                    InterventionContext(
                        backend=backend, retriever=InMemoryBM25Retriever(()), seed=17
                    ),
                )
                self.assertEqual(patch.trace.events[-1].payload["termination"], expected_termination)
                self.assertEqual(patch.apply(state).metadata["self_ask_final_answer"], "London")
                self.assertEqual(patch.trace.counters["model_calls"], 2)
                self.assertEqual([request.seed for request in backend.calls], [17, 18])
                self.assertEqual([request.max_tokens for request in backend.calls], [256, 256])

    def test_direct_final_marker_is_recorded_without_an_extra_generation(self) -> None:
        backend = CallableBackend(
            fn=lambda _request: "No.\nSo the final answer is: London",
            revision="self-ask-direct-termination-test-v1",
            callable_id="tests.self_ask.direct_termination",
        )
        result = SelfAskLoop(SelfAskConfig(demonstrations=DEMONSTRATION)).run(
            FrozenAgentState(query="Where was Ada born?"),
            InterventionContext(backend=backend, retriever=InMemoryBM25Retriever(())),
        )
        self.assertEqual(result.termination, "final_answer_marker")
        self.assertEqual(result.final_answer, "London")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(len(backend.calls), 1)

    def test_each_model_call_uses_a_distinct_deterministic_seed(self) -> None:
        responses = iter(
            (
                "Yes.\nFollow up: Where was Ada born?",
                "So the final answer is: London",
            )
        )
        backend = CallableBackend(
            fn=lambda _request: next(responses),
            revision="self-ask-seed-test-v1",
            callable_id="tests.self_ask.seed_sequence",
        )
        retriever = InMemoryBM25Retriever(
            (Document(doc_id="d1", text="London", corpus_revision="fixture"),)
        )

        result = SelfAskLoop(SelfAskConfig(demonstrations=DEMONSTRATION)).run(
            FrozenAgentState(query="Where was Ada born?"),
            InterventionContext(backend=backend, retriever=retriever, seed=17),
        )

        self.assertEqual(result.model_calls, 2)
        self.assertEqual([request.seed for request in backend.calls], [17, 18])

    def test_followup_cap_is_budget_exhaustion_not_normal_termination(self) -> None:
        backend = ReplayBackend(("Yes.\nFollow up: another?",))
        retriever = InMemoryBM25Retriever((Document(doc_id="d", text="answer"),))
        algorithm = SelfAskLoop(SelfAskConfig(demonstrations=DEMONSTRATION, max_followups=0))
        with self.assertRaises(BudgetExceededError):
            algorithm.run(
                FrozenAgentState(query="question"),
                InterventionContext(backend=backend, retriever=retriever),
            )

    def test_empty_model_output_is_typed_as_arm_local_output_failure(self) -> None:
        backend = ReplayBackend(("",))
        retriever = InMemoryBM25Retriever(())

        with self.assertRaisesRegex(AlgorithmOutputError, "empty response"):
            SelfAskLoop(SelfAskConfig(demonstrations=DEMONSTRATION)).run(
                FrozenAgentState(query="question"),
                InterventionContext(backend=backend, retriever=retriever),
            )

    def test_query_specific_empty_search_is_not_dependency_unavailability(self) -> None:
        backend = ReplayBackend(("Yes.\nFollow up: Missing fact?",))
        retriever = InMemoryBM25Retriever(())
        algorithm = SelfAskLoop(
            SelfAskConfig(
                demonstrations=DEMONSTRATION,
                allow_model_search_fallback=False,
            )
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "search returned no result"):
            algorithm.run(
                FrozenAgentState(query="question"),
                InterventionContext(backend=backend, retriever=retriever),
            )


if __name__ == "__main__":
    unittest.main()
