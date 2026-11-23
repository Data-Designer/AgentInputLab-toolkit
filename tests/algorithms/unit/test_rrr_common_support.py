from __future__ import annotations

import unittest

from agentinputlab.algorithms.reformulate.rrr import (
    RRRCommonSupport,
    RRRCommonSupportConfig,
)
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab.backends.llm import GenerationResponse, ModelCapability, Usage
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import AlgorithmOutputError, FrozenAgentState, InterventionContext
from agentinputlab.contracts.serialization import content_hash


class SequenceBackend:
    model_id = "rrr-test-model"
    capabilities = frozenset({ModelCapability.COMPLETION, ModelCapability.SEED})

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(
            text=self.outputs.pop(0),
            model=self.model_id,
            usage=Usage(input_tokens=10, output_tokens=3),
        )

    def fingerprint(self):
        return {"backend": "rrr-test", "model_id": self.model_id}


class StaticRetriever:
    retriever_id = "rrr-test-retriever"

    def __init__(self) -> None:
        self.queries = []

    def retrieve(self, query: str, *, top_k: int = 5):
        self.queries.append((query, top_k))
        return (
            RetrievalResult(
                document=Document(
                    doc_id="doc-1",
                    title="Dune",
                    text="Frank Herbert wrote Dune.",
                    corpus_revision="fixture-v1",
                ),
                score=1.0,
                rank=1,
            ),
        )

    def fingerprint(self):
        return {"retriever": "rrr-test", "revision": "fixture-v1"}


def config() -> RRRCommonSupportConfig:
    rewrite = "Provide a better search query. Question: {query}\nQuery:"
    reader = "Passage: {background}\nQuestion: {query}\nThe answer is"
    return RRRCommonSupportConfig(
        rewrite_prompt=rewrite,
        reader_prompt=reader,
        prompt_source_sha256="a" * 64,
        rewrite_prompt_sha256=content_hash(rewrite),
        reader_prompt_sha256=content_hash(reader),
        top_k=5,
    )


class RRRCommonSupportTests(unittest.TestCase):
    def test_recommended_profile_uses_public_factory_and_metered_core_without_checkpoint(
        self,
    ) -> None:
        backend = SequenceBackend(["Query: Dune author**", "Frank Herbert**"])
        method = create_intervention("rewrite.rrr.common_support_prompt", config=config())
        result = run_intervention(
            method,
            FrozenAgentState(query="Who wrote Dune?"),
            InterventionContext(backend=backend, retriever=StaticRetriever(), seed=17),
        )
        self.assertEqual(method.release_id, "rewrite.rrr.common_support_prompt")
        self.assertEqual(result["state"]["query"], "Dune author")
        self.assertFalse(result["scored"])
        self.assertIsNone(result["benchmark_score"])
        record = result["record"]
        self.assertEqual(record["usage"]["model_calls"], 2)
        self.assertEqual(record["usage"]["retrieval_calls"], 1)
        self.assertEqual(record["patch"]["profile_kind"], "controlled_adapter")
        self.assertFalse(record["patch"]["metadata"]["paper_checkpoint_used"])
        self.assertFalse(record["patch"]["metadata"]["ppo_training_used"])
        self.assertEqual([r.max_tokens for r in backend.requests], [100, 300])

    def test_source_locked_stage_order_materializes_query_and_real_evidence(self) -> None:
        backend = SequenceBackend(["Query: Dune author**", "Frank Herbert**"])
        retriever = StaticRetriever()
        state = FrozenAgentState(query="Who wrote Dune?", task_id="rrr-common")

        patch = RRRCommonSupport(config()).propose(
            state,
            InterventionContext(backend=backend, retriever=retriever, seed=17),
        )
        updated = patch.apply(state)

        self.assertEqual(updated.query, "Dune author")
        self.assertEqual(tuple(block.block_id for block in updated.context), ("doc-1",))
        self.assertEqual(retriever.queries, [("Dune author", 5)])
        self.assertEqual([request.seed for request in backend.requests], [17, 18])
        self.assertEqual(patch.trace.counters["model_calls"], 2)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 1)
        self.assertFalse(patch.metadata["paper_checkpoint_used"])
        self.assertFalse(patch.metadata["ppo_training_used"])

    def test_rejects_multiple_queries_in_single_query_profile(self) -> None:
        backend = SequenceBackend(["first; second**", "unused"])
        with self.assertRaisesRegex(AlgorithmOutputError, "exactly one query"):
            RRRCommonSupport(config()).propose(
                FrozenAgentState(query="question"),
                InterventionContext(backend=backend, retriever=StaticRetriever(), seed=0),
            )

    def test_prompt_hashes_are_enforced(self) -> None:
        rewrite = "Question: {query}"
        reader = "{background}\n{query}"
        with self.assertRaisesRegex(Exception, "rewrite prompt hash"):
            RRRCommonSupportConfig(
                rewrite_prompt=rewrite,
                reader_prompt=reader,
                prompt_source_sha256="a" * 64,
                rewrite_prompt_sha256="b" * 64,
                reader_prompt_sha256=content_hash(reader),
            )
