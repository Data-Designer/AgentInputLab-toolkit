from __future__ import annotations

import unittest

from agentinputlab import FrozenAgentState, InterventionContext
from agentinputlab.algorithms.reformulate.query2doc import (
    OfficialQuery2DocRow,
    Query2DocCache,
    Query2DocFixtureCached,
)
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.runner import InterventionRunner, RunStatus


class MinimalOfflineLoopTests(unittest.TestCase):
    def test_query2doc_cache_to_shared_retriever(self) -> None:
        cache = Query2DocCache(
            [
                OfficialQuery2DocRow(
                    query_id="q1",
                    query="capital france",
                    pseudo_doc="Paris is the capital of France",
                    split="tiny",
                )
            ]
        )
        algorithm = Query2DocFixtureCached(cache, split="tiny")
        state = FrozenAgentState(task_id="q1", query="capital france")
        record = InterventionRunner().run(
            algorithm,
            state,
            InterventionContext(seed=0),
        )
        transformed = record.patch.apply(state)
        retriever = InMemoryBM25Retriever(
            [
                Document("fr", "Paris is the capital of France", corpus_revision="tiny-v1"),
                Document("de", "Berlin is the capital of Germany", corpus_revision="tiny-v1"),
            ]
        )
        results = retriever.retrieve(transformed.query, top_k=2)
        self.assertEqual(record.status, RunStatus.SUCCESS)
        self.assertEqual(results[0].document.doc_id, "fr")
        self.assertEqual(record.patch.metadata["pseudo_document_is_evidence"], False)


if __name__ == "__main__":
    unittest.main()
