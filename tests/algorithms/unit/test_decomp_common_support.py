from __future__ import annotations

import unittest

from agentinputlab.algorithms.decompose.decomp import (
    DecompCommonSupport,
    DecompCommonSupportConfig,
    parse_decomp_command,
)
from agentinputlab.backends.llm import GenerationResponse, ModelCapability, Usage
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
)
from agentinputlab.contracts.serialization import content_hash


class SequenceBackend:
    model_id = "decomp-test-model"
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
        return {"backend": "decomp-test", "model_id": self.model_id}


class StaticRetriever:
    retriever_id = "decomp-test-retriever"

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
        return {"retriever": "decomp-test", "revision": "fixture-v1"}


def config() -> DecompCommonSupportConfig:
    demonstrations = (
        "QC: Who wrote Dune?\n"
        "QS: (select) [retrieve_odqa] Who is the author of Dune?\n"
        'A: {"titles":["Dune"],"answer":["Frank Herbert"]}\n'
        "QS: (select) [multihop_titleqa] Titles: [Dune]. Question: Who wrote Dune?\n"
        'A: ["Frank Herbert"]\nQS: [EOQ]'
    )
    return DecompCommonSupportConfig(
        demonstrations=demonstrations,
        prompt_source_sha256="a" * 64,
        demonstrations_sha256=content_hash(demonstrations),
        max_steps=3,
        retrieval_top_k=2,
    )


class DecompCommonSupportTests(unittest.TestCase):
    def test_routes_retrieval_then_delegates_final_aggregation(self) -> None:
        backend = SequenceBackend(
            [
                "(select) [retrieve_odqa] Who is the author of Dune?",
                "Frank Herbert",
                "(select) [multihop_titleqa] Titles: [Dune]. Question: Who wrote Dune?",
            ]
        )
        retriever = StaticRetriever()
        state = FrozenAgentState(query="Who wrote Dune?", task_id="decomp-common")

        patch = DecompCommonSupport(config()).propose(
            state,
            InterventionContext(backend=backend, retriever=retriever, seed=17),
        )
        updated = patch.apply(state)

        self.assertEqual(retriever.queries, [("Who is the author of Dune?", 2)])
        self.assertEqual([request.seed for request in backend.requests], [17, 18, 19])
        self.assertEqual(tuple(block.block_id for block in updated.context)[:1], ("doc-1",))
        self.assertIn("aggregation_delegated_to_common_agent", updated.context[-1].text)
        self.assertEqual(patch.trace.counters["model_calls"], 3)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 1)
        self.assertFalse(patch.metadata["final_answer_injected"])
        self.assertTrue(patch.metadata["upstream_participant_routing_preserved"])

    def test_source_router_grammar_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(AlgorithmOutputError, "router grammar"):
            parse_decomp_command("(project) [python] unsafe")
        self.assertIsNone(parse_decomp_command("QS: [EOQ]"))

    def test_aggregation_before_retrieval_is_an_arm_local_algorithm_failure(self) -> None:
        backend = SequenceBackend(["(select) [multihop_titleqa] Question: Who wrote Dune?"])
        with self.assertRaisesRegex(AlgorithmOutputError, "before materializing"):
            DecompCommonSupport(config()).propose(
                FrozenAgentState(query="Who wrote Dune?", task_id="decomp-early-aggregate"),
                InterventionContext(
                    backend=backend,
                    retriever=StaticRetriever(),
                    seed=17,
                ),
            )

    def test_model_call_budget_is_enforced_across_handlers(self) -> None:
        backend = SequenceBackend(
            [
                "(select) [retrieve_odqa] Who is the author of Dune?",
                "Frank Herbert",
                "[EOQ]",
            ]
        )
        state = FrozenAgentState(
            query="Who wrote Dune?",
            budget=ExecutionBudget(max_model_calls=2),
        )
        with self.assertRaisesRegex(Exception, "max_model_calls"):
            DecompCommonSupport(config()).propose(
                state,
                InterventionContext(backend=backend, retriever=StaticRetriever(), seed=0),
            )


if __name__ == "__main__":
    unittest.main()
