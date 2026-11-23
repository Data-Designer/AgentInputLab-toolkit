from __future__ import annotations

import unittest

from agentinputlab.algorithms.acquire.hyde import HyDECleanRoom, HyDEConfig
from agentinputlab.backends.llm import (
    CallableBackend,
    GenerationResponse,
    ReplayBackend,
    Usage,
)
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.errors import BudgetExceededError
from agentinputlab.observability import ObservedRunStatus
from agentinputlab.runner import InterventionRunner


class FakeEmbedder:
    embedder_id = "fake-embedder"

    def __init__(self, vectors):
        self.vectors = vectors
        self.texts = None
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        self.texts = tuple(texts)
        return self.vectors

    def fingerprint(self):
        return {"embedder_id": self.embedder_id, "revision": "fixture-v1"}


class FakeSearcher:
    searcher_id = "fake-vector-index"

    def __init__(self):
        self.vector = None
        self.calls = 0

    def search(self, vector, *, top_k):
        self.calls += 1
        self.vector = tuple(vector)
        return (
            RetrievalResult(
                document=Document(
                    doc_id="doc-1",
                    text="Paris is the capital of France.",
                    title="France",
                    corpus_revision="tiny-corpus-v1",
                ),
                score=0.9,
                rank=1,
                features={"dense_score": 0.9},
            ),
        )[:top_k]

    def fingerprint(self):
        return {
            "searcher_id": self.searcher_id,
            "index_revision": "tiny-index-v1",
            "metric": "dot",
        }


class HyDETests(unittest.TestCase):
    def test_exact_mean_and_real_evidence_provenance(self):
        embedder = FakeEmbedder(((1.0, 3.0), (3.0, 5.0), (5.0, 7.0)))
        searcher = FakeSearcher()
        backend = ReplayBackend(["hypothesis one", "hypothesis two"])
        algorithm = HyDECleanRoom(
            embedder=embedder,
            searcher=searcher,
            generator=backend,
            config=HyDEConfig(num_hypotheses=2, top_k=1),
        )
        state = FrozenAgentState(query="What is the capital of France?")

        patch = algorithm.propose(state, InterventionContext(seed=17))

        self.assertEqual(searcher.vector, (3.0, 5.0))
        self.assertEqual(
            embedder.texts,
            (
                "What is the capital of France?",
                "hypothesis one",
                "hypothesis two",
            ),
        )
        materialized = patch.apply(state)
        self.assertEqual(materialized.context[0].block_id, "doc-1")
        self.assertEqual(
            materialized.context[0].metadata["evidence_kind"],
            "real_corpus_document",
        )
        self.assertFalse(patch.metadata["synthetic_documents_are_evidence"])
        hypothesis_event = patch.trace.events[1]
        self.assertEqual(
            hypothesis_event.payload["semantic_role"],
            "synthetic_query_surrogate",
        )

    def test_partial_empty_generations_are_audited(self):
        algorithm = HyDECleanRoom(
            embedder=FakeEmbedder(((1.0,), (3.0,))),
            searcher=FakeSearcher(),
            generator=ReplayBackend(["", "usable"]),
            config=HyDEConfig(num_hypotheses=2),
        )
        patch = algorithm.propose(
            FrozenAgentState(query="q"),
            InterventionContext(),
        )
        self.assertIn("empty_hypothesis_indices=[0]", patch.trace.warnings)

    def test_all_empty_generations_fail_instead_of_becoming_baseline(self):
        algorithm = HyDECleanRoom(
            embedder=FakeEmbedder(((1.0,),)),
            searcher=FakeSearcher(),
            generator=ReplayBackend([""]),
            config=HyDEConfig(num_hypotheses=1),
        )
        with self.assertRaises(AlgorithmOutputError):
            algorithm.propose(FrozenAgentState(query="q"), InterventionContext())

    def test_missing_generator_is_typed_unavailability(self):
        algorithm = HyDECleanRoom(
            embedder=FakeEmbedder(((1.0,), (2.0,))),
            searcher=FakeSearcher(),
        )
        with self.assertRaises(ExternalDependencyError):
            algorithm.propose(FrozenAgentState(query="q"), InterventionContext())

    def test_embedding_dimension_mismatch_fails_closed(self):
        algorithm = HyDECleanRoom(
            embedder=FakeEmbedder(((1.0, 2.0), (3.0,))),
            searcher=FakeSearcher(),
            generator=CallableBackend(
                lambda _: "hypothesis",
                revision="fixture-v1",
                callable_id="tests.hyde.hypothesis.v1",
            ),
            config=HyDEConfig(num_hypotheses=1),
        )
        with self.assertRaisesRegex(AlgorithmOutputError, "inconsistent dimensions"):
            algorithm.propose(FrozenAgentState(query="q"), InterventionContext())

    def test_document_metadata_cannot_override_provenance(self):
        class SpoofingSearcher(FakeSearcher):
            def search(self, vector, *, top_k):
                return (
                    RetrievalResult(
                        document=Document(
                            doc_id="real-id",
                            text="real text",
                            corpus_revision="real-corpus",
                            metadata={"document_id": "spoofed"},
                        ),
                        score=1.0,
                        rank=1,
                    ),
                )

        algorithm = HyDECleanRoom(
            embedder=FakeEmbedder(((1.0,), (2.0,))),
            searcher=SpoofingSearcher(),
            generator=ReplayBackend(["hypothesis"]),
            config=HyDEConfig(num_hypotheses=1),
        )
        with self.assertRaisesRegex(AlgorithmOutputError, "reserved provenance"):
            algorithm.propose(FrozenAgentState(query="q"), InterventionContext())

    def test_runner_rejects_renamed_hidden_generator_before_execution(self):
        backend = ReplayBackend(("hypothesis",))
        algorithm = HyDECleanRoom(
            embedder=FakeEmbedder(((1.0,), (2.0,))),
            searcher=FakeSearcher(),
            generator=backend,
            config=HyDEConfig(num_hypotheses=1),
        )

        with self.assertRaisesRegex(ValidationError, "generator"):
            InterventionRunner().run(
                algorithm,
                FrozenAgentState(query="q"),
                InterventionContext(),
            )
        self.assertEqual(backend._calls, 0)

    def test_runner_observes_hyde_generation_embedding_and_search(self):
        response = GenerationResponse(
            text="hypothesis",
            model="fixture",
            usage=Usage(input_tokens=11, output_tokens=4, cost_usd=0.02),
        )
        embedder = FakeEmbedder(((1.0,), (2.0,)))
        searcher = FakeSearcher()
        algorithm = HyDECleanRoom(config=HyDEConfig(num_hypotheses=1))

        record = InterventionRunner().run(
            algorithm,
            FrozenAgentState(query="q"),
            InterventionContext(
                backend=ReplayBackend((response,)),
                dense_embedder=embedder,
                vector_searcher=searcher,
            ),
        )

        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual(record.usage.embedding_calls, 1)
        self.assertEqual(record.usage.retrieval_calls, 1)
        self.assertEqual(record.usage.input_tokens, 11)
        self.assertEqual(record.usage.output_tokens, 4)
        self.assertAlmostEqual(record.usage.cost_usd, 0.02)
        self.assertEqual(embedder.calls, 1)
        self.assertEqual(searcher.calls, 1)

    def test_runner_records_embedding_failure_at_the_exact_stage(self):
        class FailingEmbedder(FakeEmbedder):
            def embed(self, texts):
                self.calls += 1
                raise RuntimeError("embedding unavailable")

        embedder = FailingEmbedder(())
        searcher = FakeSearcher()
        runner = InterventionRunner()
        with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
            runner.run(
                HyDECleanRoom(config=HyDEConfig(num_hypotheses=1)),
                FrozenAgentState(query="q"),
                InterventionContext(
                    backend=ReplayBackend(("hypothesis",)),
                    dense_embedder=embedder,
                    vector_searcher=searcher,
                ),
            )

        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.FAILED)
        self.assertEqual(snapshot.usage.embedding_calls, 1)
        self.assertEqual(snapshot.usage.retrieval_calls, 0)
        self.assertEqual(snapshot.failure.category, "embedding_error")
        self.assertEqual(snapshot.failure.stage, "embedding")
        self.assertEqual(embedder.calls, 1)
        self.assertEqual(searcher.calls, 0)

    def test_vector_search_budget_blocks_before_the_real_search(self):
        embedder = FakeEmbedder(((1.0,), (2.0,)))
        searcher = FakeSearcher()
        runner = InterventionRunner()
        with self.assertRaises(BudgetExceededError):
            runner.run(
                HyDECleanRoom(config=HyDEConfig(num_hypotheses=1)),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_retrieval_calls=0),
                ),
                InterventionContext(
                    backend=ReplayBackend(("hypothesis",)),
                    dense_embedder=embedder,
                    vector_searcher=searcher,
                ),
            )

        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(snapshot.usage.embedding_calls, 1)
        self.assertEqual(snapshot.usage.retrieval_calls, 0)
        self.assertEqual(searcher.calls, 0)

    def test_runner_records_vector_search_failure_at_the_exact_stage(self):
        class FailingSearcher(FakeSearcher):
            def search(self, vector, *, top_k):
                self.calls += 1
                raise RuntimeError("vector index unavailable")

        embedder = FakeEmbedder(((1.0,), (2.0,)))
        searcher = FailingSearcher()
        runner = InterventionRunner()
        with self.assertRaisesRegex(RuntimeError, "vector index unavailable"):
            runner.run(
                HyDECleanRoom(config=HyDEConfig(num_hypotheses=1)),
                FrozenAgentState(query="q"),
                InterventionContext(
                    backend=ReplayBackend(("hypothesis",)),
                    dense_embedder=embedder,
                    vector_searcher=searcher,
                ),
            )

        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.status, ObservedRunStatus.FAILED)
        self.assertEqual(snapshot.usage.embedding_calls, 1)
        self.assertEqual(snapshot.usage.retrieval_calls, 1)
        self.assertEqual(snapshot.failure.category, "retrieval_error")
        self.assertEqual(snapshot.failure.stage, "vector_search")
        self.assertEqual(searcher.calls, 1)


if __name__ == "__main__":
    unittest.main()
