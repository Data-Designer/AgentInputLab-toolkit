from __future__ import annotations

import hashlib
import unittest

from agentinputlab.algorithms.acquire import (
    IRCoTCommonSupport,
    IRCoTCommonSupportConfig,
)
from agentinputlab.backends.llm import CallableBackend
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
)

DEMONSTRATIONS = (
    "Q: Answer the following question by reasoning step-by-step.\n"
    "What city is the capital of France?\n"
    "A: France has Paris as its capital. So the answer is: Paris."
)


class RecordingRetriever:
    retriever_id = "recording-bm25"

    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int = 5):
        self.queries.append((query, top_k))
        index = len(self.queries)
        return (
            RetrievalResult(
                document=Document(
                    doc_id=f"d{index}",
                    title=f"Document {index}",
                    text=f"Evidence {index}",
                    corpus_revision="corpus-v1",
                ),
                score=1.0,
                rank=1,
            ),
        )

    def fingerprint(self):
        return {"retriever_id": self.retriever_id, "revision": "v1"}


def config(**kwargs):
    return IRCoTCommonSupportConfig(
        demonstrations=DEMONSTRATIONS,
        prompt_sha256=hashlib.sha256(DEMONSTRATIONS.encode()).hexdigest(),
        **kwargs,
    )


class IRCoTCommonSupportTests(unittest.TestCase):
    def test_interleaves_generation_and_retrieval_then_exposes_only_documents(self):
        responses = iter(
            (
                {
                    "text": "Paris is France's capital. More text is ignored.",
                    "usage": {"input_tokens": 20, "output_tokens": 7},
                },
                {
                    "text": "So the answer is: Paris.",
                    "usage": {"input_tokens": 24, "output_tokens": 6},
                },
            )
        )
        backend = CallableBackend(
            lambda _request: next(responses),
            callable_id="tests.ircot.common-support",
            revision="v1",
        )
        retriever = RecordingRetriever()
        state = FrozenAgentState(query="What is the capital of France?", task_id="q1")

        patch = IRCoTCommonSupport(config=config()).propose(
            state,
            InterventionContext(backend=backend, retriever=retriever, seed=17),
        )

        self.assertEqual([request.seed for request in backend.calls], [17, 18])
        self.assertEqual(retriever.queries[0], ("the capital of France?", 6))
        self.assertEqual(retriever.queries[1], ("Paris France's capital.", 6))
        self.assertEqual(patch.trace.counters["model_calls"], 2)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 2)
        self.assertEqual(patch.trace.counters["input_tokens"], 44)
        self.assertEqual(patch.trace.counters["output_tokens"], 13)
        updated = patch.apply(state)
        self.assertEqual([block.block_id for block in updated.context], ["d1", "d2"])
        self.assertNotIn("Paris is France's capital", " ".join(x.text for x in updated.context))
        self.assertFalse(patch.metadata["generated_answer_visible_to_downstream"])

    def test_reasoning_sentence_falls_back_to_original_question_for_retrieval(self):
        backend = CallableBackend(
            lambda _request: "Therefore we combine both observations.",
            callable_id="tests.ircot.reasoning-fallback",
            revision="v1",
        )
        retriever = RecordingRetriever()
        algorithm = IRCoTCommonSupport(config=config(max_steps=2))
        algorithm.propose(
            FrozenAgentState(query="Which observation is correct?"),
            InterventionContext(backend=backend, retriever=retriever),
        )
        self.assertEqual(retriever.queries[1][0], "observation correct?")

    def test_empty_initial_retrieval_is_arm_local_output_failure(self):
        class EmptyRetriever(RecordingRetriever):
            def retrieve(self, query: str, *, top_k: int = 5):
                self.queries.append((query, top_k))
                return ()

        with self.assertRaisesRegex(AlgorithmOutputError, "initial retrieval"):
            IRCoTCommonSupport(config=config()).propose(
                FrozenAgentState(query="question"),
                InterventionContext(
                    backend=CallableBackend(
                        lambda _request: "unused",
                        callable_id="tests.ircot.empty",
                        revision="v1",
                    ),
                    retriever=EmptyRetriever(),
                ),
            )

    def test_cumulative_usage_enforces_context_budget(self):
        backend = CallableBackend(
            lambda _request: {
                "text": "First factual sentence.",
                "usage": {"input_tokens": 11, "output_tokens": 1},
            },
            callable_id="tests.ircot.budget",
            revision="v1",
        )
        with self.assertRaisesRegex(Exception, "max_context_tokens"):
            IRCoTCommonSupport(config=config(max_steps=2)).propose(
                FrozenAgentState(
                    query="question",
                    budget=ExecutionBudget(max_context_tokens=20),
                ),
                InterventionContext(backend=backend, retriever=RecordingRetriever()),
            )


if __name__ == "__main__":
    unittest.main()
