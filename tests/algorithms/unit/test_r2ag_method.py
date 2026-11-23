from __future__ import annotations

import math
import unittest
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from agentinputlab.algorithms.acquire.r2ag import (
    R2AG_FEATURE_NAMES,
    R2AG_PROTOCOL,
    R2AGArtifactIdentity,
    R2AGControlledWorker,
    R2AGControls,
    R2AGFeatureSource,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ContextBlock,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.runner import CompositeInterventionRunner

_RUNTIME_DEPENDENCIES = {
    "torch": "2.1.2",
    "transformers": "4.40.2",
    "sentence-transformers": "2.6.1",
}


class FixtureEmbedder:
    embedder_id = "r2ag-fixture-embedder"

    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls = 0
        self.texts: tuple[str, ...] = ()

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls += 1
        self.texts = tuple(texts)
        return self.vectors

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "embedder_id": self.embedder_id,
            "repository": "fixture/r2ag-retriever",
            "revision": "retriever-v1",
        }


class R2AGFixtureWorker:
    worker_id = "r2ag-fixture-worker"

    def __init__(
        self,
        *,
        fixture_mode: bool = True,
        mutate_first_feature: bool = False,
        worker_num_k: int | None = None,
        output_tokens: int = 2,
        reverse_ids: bool = False,
    ) -> None:
        self.calls = 0
        self.request: Any = None
        self.fixture_mode = fixture_mode
        self.mutate_first_feature = mutate_first_feature
        self.worker_num_k = worker_num_k
        self.output_tokens = output_tokens
        self.reverse_ids = reverse_ids

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.call_controlled(
            SimpleNamespace(
                payload=payload["payload"],
                request_id=payload["request_id"],
                upstream_commit=payload["upstream_commit"],
                protocol_version=payload["protocol_version"],
            )
        )

    def call_controlled(self, request: Any) -> Mapping[str, Any]:
        self.calls += 1
        self.request = request
        request_features = request.payload["features"]
        if request_features is None:
            features = [
                {"rerank": 0.9, "neighbor": 0.2, "precedent": 1.0},
                {"rerank": 0.7, "neighbor": 0.3, "precedent": 0.4},
                {"rerank": 0.1, "neighbor": 0.1, "precedent": 0.2},
            ]
            feature_hash = content_hash(
                tuple(tuple(row[name] for name in R2AG_FEATURE_NAMES) for row in features)
            )
        else:
            features = [dict(row) for row in request_features]
            feature_hash = request.payload["feature_tensor_sha256"]
        if self.mutate_first_feature:
            features[0]["rerank"] -= 0.25
        artifacts = request.payload["artifacts"]
        num_k = self.worker_num_k or artifacts["checkpoint_num_documents"]
        candidate_ids = list(request.payload["candidate_document_ids"])
        if self.reverse_ids:
            candidate_ids.reverse()
        payload = {
            "candidate_document_ids": candidate_ids,
            "candidate_documents_sha256": request.payload["candidate_documents_sha256"],
            "candidate_order_sha256": request.payload["candidate_order_sha256"],
            "candidate_pool_fingerprint_sha256": request.payload[
                "candidate_pool_fingerprint_sha256"
            ],
            "artifacts": dict(artifacts),
            "artifact_identity_sha256": request.payload["artifact_identity_sha256"],
            "worker_config_num_k": num_k,
            "features": features,
            "feature_tensor_sha256": feature_hash,
            "feature_embedder_fingerprint_sha256": request.payload[
                "feature_embedder_fingerprint_sha256"
            ],
            "feature_artifact_fingerprint_sha256": request.payload[
                "feature_artifact_fingerprint_sha256"
            ],
            "reader_instruction_sha256": request.payload["reader_instruction_sha256"],
            "reader_input_sha256": request.payload["reader_input_sha256"],
            "r2former_output_sha256": "1" * 64,
            "projection_output_sha256": "2" * 64,
            "r_token_count": len(features),
            "r_token_positions": [11 + 10 * index for index in range(len(features))],
            "non_retrieval_unk_positions_modified": False,
            "runtime_attestation": _runtime_attestation(
                artifacts,
                num_k=num_k,
                fixture_mode=self.fixture_mode,
            ),
            "answer": "Paris",
        }
        embedding_calls = (
            1 if request.payload["feature_source"] == R2AGFeatureSource.WORKER.value else 0
        )
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": request.upstream_commit,
            "protocol_version": request.protocol_version,
            "payload": payload,
            "usage": {
                "model_calls": 1,
                "retrieval_calls": 0,
                "tool_calls": 0,
                "embedding_calls": embedding_calls,
                "input_tokens": 12,
                "output_tokens": self.output_tokens,
            },
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.worker_id,
            "image_digest": "sha256:r2ag-fixture",
            "source_revision": "61cfd1027c0101172fda701ec816bc18360b5bab",
            "runtime_dependencies": _RUNTIME_DEPENDENCIES,
        }


def _runtime_attestation(
    artifacts: Mapping[str, Any],
    *,
    num_k: int,
    fixture_mode: bool,
) -> Mapping[str, Any]:
    return {
        "fixture_mode": fixture_mode,
        "source_commit": "61cfd1027c0101172fda701ec816bc18360b5bab",
        "source_manifest_sha256": artifacts["source_manifest_sha256"],
        "compatibility_manifest_sha256": artifacts["compatibility_manifest_sha256"],
        "checkpoint_config_sha256": artifacts["checkpoint_config_sha256"],
        "rformer_weights_sha256": artifacts["rformer_weights_sha256"],
        "projection_weights_sha256": artifacts["projection_weights_sha256"],
        "base_model_manifest_sha256": artifacts["base_model_manifest_sha256"],
        "tokenizer_manifest_sha256": artifacts["tokenizer_manifest_sha256"],
        "feature_schema_sha256": artifacts["feature_schema_sha256"],
        "worker_config_num_k": num_k,
        "source_commit_verified": True,
        "source_files_verified": True,
        "source_checkout_clean": True,
        "checkpoint_hashes_verified": True,
        "rformer_called": True,
        "projection_called": True,
        "marker_replacement_called": True,
        "reader_generate_called": True,
        "only_marker_positions_modified": True,
        "gold_metadata_received": False,
        "training_called": False,
        "budget_checked_before_reader_call": True,
        "runtime_dependencies": _RUNTIME_DEPENDENCIES,
        "runtime_dependencies_verified": not fixture_mode,
        "execution_evidence": (
            "fixture_source_execution" if fixture_mode else "pinned_source_checkpoint_execution"
        ),
    }


def artifacts(*, num_k: int = 3) -> R2AGArtifactIdentity:
    return R2AGArtifactIdentity(
        checkpoint_repository="fixture/r2ag-checkpoint",
        checkpoint_revision="checkpoint-v1",
        checkpoint_config_sha256="1" * 64,
        rformer_weights_sha256="2" * 64,
        projection_weights_sha256="3" * 64,
        checkpoint_num_documents=num_k,
        base_model_repository="fixture/reader",
        base_model_revision="reader-v1",
        base_model_manifest_sha256="4" * 64,
        tokenizer_repository="fixture/tokenizer",
        tokenizer_revision="tokenizer-v1",
        tokenizer_manifest_sha256="5" * 64,
        feature_schema_revision="rrag-features-v1",
        feature_schema_sha256="6" * 64,
        source_manifest_sha256="7" * 64,
        compatibility_manifest_sha256="8" * 64,
    )


def r2ag_state(*, budget: ExecutionBudget | None = None) -> FrozenAgentState:
    return FrozenAgentState(
        query="What is the capital of France?",
        task_id="r2ag-local-1",
        context=(
            ContextBlock(
                "Paris is the capital of France.",
                block_id="d1",
                source="corpus",
                score=0.9,
                metadata={
                    "title": "France",
                    "corpus_revision": "corpus-v1",
                    "gold_answer": "Paris",
                    "is_supporting": True,
                },
            ),
            ContextBlock(
                "France is a country in Europe.",
                block_id="d2",
                source="corpus",
                score=0.7,
                metadata={
                    "title": "France geography",
                    "corpus_revision": "corpus-v1",
                    "reward": 1,
                },
            ),
            ContextBlock(
                "Berlin is the capital of Germany.",
                block_id="d3",
                source="corpus",
                score=0.1,
                metadata={
                    "title": "Germany",
                    "corpus_revision": "corpus-v1",
                    "supporting_facts": ["Germany"],
                },
            ),
        ),
        budget=budget or ExecutionBudget(),
    )


def candidate_pool(state: FrozenAgentState) -> Mapping[str, Any]:
    documents = tuple(
        {
            "document_id": block.block_id,
            "text": block.text,
            "title": block.metadata.get("title", ""),
            "source": block.source,
            "score": block.score,
            "corpus_revision": block.metadata.get("corpus_revision", ""),
        }
        for block in state.context
    )
    return {
        "retriever": {"repository": "fixture/retriever", "revision": "retriever-v1"},
        "corpus": {"repository": "fixture/corpus", "revision": "corpus-v1"},
        "index": {"algorithm": "flat", "revision": "index-v1"},
        "top_k": len(documents),
        "order": "rerank_desc",
        "content_sha256": content_hash(documents),
    }


def context(
    state: FrozenAgentState,
    *,
    worker: R2AGFixtureWorker | None = None,
    embedder: FixtureEmbedder | None = None,
    include_pool: bool = True,
) -> InterventionContext:
    return InterventionContext(
        external_runtime=worker,
        dense_embedder=embedder,
        run_config=({"candidate_pool_fingerprint": candidate_pool(state)} if include_pool else {}),
    )


def embedder_controls(*, max_output_tokens: int = 2048) -> R2AGControls:
    return R2AGControls(
        feature_source=R2AGFeatureSource.EMBEDDER,
        checkpoint_num_documents=3,
        max_output_tokens=max_output_tokens,
    )


def adapter(
    *,
    controls: R2AGControls | None = None,
    allow_fixture_worker: bool = True,
) -> R2AGControlledWorker:
    return R2AGControlledWorker(
        artifacts=artifacts(),
        controls=controls or embedder_controls(),
        allow_fixture_worker=allow_fixture_worker,
    )


def vectors() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0),
        (1.0, 0.0),
        (0.6, 0.8),
        (0.0, 1.0),
    )


class R2AGMethodTests(unittest.TestCase):
    def test_embedder_path_computes_features_and_sends_only_allowlisted_dto(self) -> None:
        state = r2ag_state()
        embedder = FixtureEmbedder(vectors())
        worker = R2AGFixtureWorker()

        result = adapter().run(state, context(state, worker=worker, embedder=embedder))

        self.assertEqual(embedder.calls, 1)
        self.assertEqual(worker.request.protocol_version, R2AG_PROTOCOL)
        request_features = worker.request.payload["features"]
        self.assertAlmostEqual(request_features[0]["rerank"], 1.0)
        self.assertAlmostEqual(request_features[1]["neighbor"], 0.7)
        expected_weight_first = math.exp(1.0) / (math.exp(1.0) + math.exp(0.6))
        expected_weight_second = 1.0 - expected_weight_first
        precedent_vector = (
            expected_weight_first + 0.6 * expected_weight_second,
            0.8 * expected_weight_second,
        )
        self.assertAlmostEqual(
            request_features[2]["precedent"],
            precedent_vector[1] / math.sqrt(precedent_vector[0] ** 2 + precedent_vector[1] ** 2),
        )
        candidate_documents = worker.request.payload["candidate_documents"]
        allowed = {
            "document_id",
            "text",
            "title",
            "source",
            "score",
            "corpus_revision",
        }
        self.assertTrue(all(set(row) == allowed for row in candidate_documents))
        serialized = repr(candidate_documents)
        for forbidden in ("gold_answer", "is_supporting", "reward", "supporting_facts"):
            self.assertNotIn(forbidden, serialized)
            self.assertNotIn(forbidden, repr(result.evidence))
        self.assertEqual(result.final_answer, "Paris")
        self.assertEqual(
            result.metadata["implementation_origin"],
            "controlled_source_informed_with_verified_worker",
        )
        self.assertFalse(result.metadata["paper_checkpoint_parity"])
        self.assertTrue(result.implementation_identity.startswith("controlled_source_informed:"))
        self.assertTrue(result.metadata["fixture_mode"])
        self.assertEqual(result.metadata["execution_evidence"], "fixture_source_execution")
        self.assertEqual(result.trace.counters["embedding_calls"], 1)

    def test_every_feature_source_requires_pool_and_exact_checkpoint_num_k(self) -> None:
        state = r2ag_state()
        worker = R2AGFixtureWorker()
        with self.assertRaisesRegex(ExternalDependencyError, "candidate_pool_fingerprint"):
            adapter().run(
                state,
                context(
                    state,
                    worker=worker,
                    embedder=FixtureEmbedder(vectors()),
                    include_pool=False,
                ),
            )
        self.assertEqual(worker.calls, 0)

        short = FrozenAgentState(query="question", context=state.context[:2])
        with self.assertRaisesRegex(ValidationError, "checkpoint_num_documents exactly"):
            adapter().run(short, InterventionContext())

        controls = R2AGControls(
            feature_source=R2AGFeatureSource.WORKER,
            checkpoint_num_documents=30,
        )
        self.assertEqual(controls.checkpoint_num_documents, 30)

    def test_pool_content_hash_and_worker_num_k_must_match(self) -> None:
        state = r2ag_state()
        bad_pool = dict(candidate_pool(state))
        bad_pool["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValidationError, "content hash mismatch"):
            adapter().run(
                state,
                InterventionContext(
                    external_runtime=R2AGFixtureWorker(),
                    dense_embedder=FixtureEmbedder(vectors()),
                    run_config={"candidate_pool_fingerprint": bad_pool},
                ),
            )

        with self.assertRaisesRegex(AlgorithmOutputError, "config.num_k"):
            adapter().run(
                state,
                context(
                    state,
                    worker=R2AGFixtureWorker(worker_num_k=2),
                    embedder=FixtureEmbedder(vectors()),
                ),
            )

        with self.assertRaisesRegex(AlgorithmOutputError, "candidate ordering"):
            adapter().run(
                state,
                context(
                    state,
                    worker=R2AGFixtureWorker(reverse_ids=True),
                    embedder=FixtureEmbedder(vectors()),
                ),
            )

    def test_nested_pool_and_feature_artifact_oracle_keys_never_reach_worker(self) -> None:
        state = r2ag_state()
        worker = R2AGFixtureWorker()
        bad_pool = dict(candidate_pool(state))
        bad_pool["retriever"] = {
            **dict(bad_pool["retriever"]),
            "gold_answer": "Paris",
        }
        with self.assertRaisesRegex(ValidationError, "forbidden metadata key"):
            adapter().run(
                state,
                InterventionContext(
                    external_runtime=worker,
                    dense_embedder=FixtureEmbedder(vectors()),
                    run_config={"candidate_pool_fingerprint": bad_pool},
                ),
            )
        self.assertEqual(worker.calls, 0)

        controls = R2AGControls(
            feature_source=R2AGFeatureSource.CANDIDATE_METADATA,
            checkpoint_num_documents=3,
        )
        with self.assertRaisesRegex(ValidationError, "forbidden metadata key"):
            adapter(controls=controls).run(
                state,
                InterventionContext(
                    external_runtime=worker,
                    run_config={
                        "candidate_pool_fingerprint": candidate_pool(state),
                        "r2ag_feature_artifact_fingerprint": {
                            "repository": "fixture/features",
                            "revision": "features-v1",
                            "sha256": "9" * 64,
                            "reward_label": 1,
                        },
                    },
                ),
            )
        self.assertEqual(worker.calls, 0)

    def test_fixture_attestation_is_rejected_by_production_host(self) -> None:
        state = r2ag_state()
        with self.assertRaisesRegex(AlgorithmOutputError, "rejects fixture"):
            adapter(allow_fixture_worker=False).run(
                state,
                context(
                    state,
                    worker=R2AGFixtureWorker(fixture_mode=True),
                    embedder=FixtureEmbedder(vectors()),
                ),
            )

    def test_worker_feature_source_is_executable_and_usage_checked(self) -> None:
        state = r2ag_state()
        controls = R2AGControls(
            feature_source=R2AGFeatureSource.WORKER,
            checkpoint_num_documents=3,
        )
        worker = R2AGFixtureWorker()
        result = adapter(controls=controls).run(state, context(state, worker=worker))
        self.assertEqual(result.metadata["feature_source"], "worker")
        self.assertEqual(result.trace.counters["embedding_calls"], 1)

        with self.assertRaisesRegex(AlgorithmOutputError, "output-token budget"):
            adapter(controls=embedder_controls(max_output_tokens=1)).run(
                state,
                context(
                    state,
                    worker=R2AGFixtureWorker(output_tokens=2),
                    embedder=FixtureEmbedder(vectors()),
                ),
            )

    def test_candidate_metadata_feature_source_preserves_formula_identity(self) -> None:
        state = r2ag_state()
        rows = ((0.9, 0.2, 1.0), (0.7, 0.3, 0.4), (0.1, 0.1, 0.2))
        state = FrozenAgentState(
            query=state.query,
            task_id=state.task_id,
            context=tuple(
                ContextBlock(
                    block.text,
                    block_id=block.block_id,
                    source=block.source,
                    score=block.score,
                    metadata={
                        **dict(block.metadata),
                        "rerank_score": row[0],
                        "rerank_nb_score": row[1],
                        "rerank_precedent_score": row[2],
                    },
                )
                for block, row in zip(state.context, rows, strict=True)
            ),
        )
        controls = R2AGControls(
            feature_source=R2AGFeatureSource.CANDIDATE_METADATA,
            checkpoint_num_documents=3,
        )
        worker = R2AGFixtureWorker()
        run_context = InterventionContext(
            external_runtime=worker,
            run_config={
                "candidate_pool_fingerprint": candidate_pool(state),
                "r2ag_feature_artifact_fingerprint": {
                    "repository": "fixture/features",
                    "revision": "features-v1",
                    "sha256": "9" * 64,
                },
            },
        )
        result = adapter(controls=controls).run(state, run_context)
        self.assertEqual(result.metadata["feature_source"], "candidate_metadata")
        self.assertEqual(result.trace.counters["embedding_calls"], 0)

    def test_composite_runner_enforces_pre_call_and_actual_usage_budgets(self) -> None:
        exhausted = r2ag_state(budget=ExecutionBudget(max_model_calls=0, max_output_tokens=8))
        worker = R2AGFixtureWorker()
        embedder = FixtureEmbedder(vectors())
        with self.assertRaisesRegex(BudgetExceededError, "model-call budget"):
            CompositeInterventionRunner().run(
                adapter(controls=embedder_controls(max_output_tokens=8)),
                exhausted,
                context(exhausted, worker=worker, embedder=embedder),
            )
        self.assertEqual(worker.calls, 0)
        self.assertEqual(embedder.calls, 0)

        state = r2ag_state(
            budget=ExecutionBudget(
                max_model_calls=1,
                max_output_tokens=8,
                max_context_tokens=64,
            )
        )
        record = CompositeInterventionRunner().run(
            adapter(controls=embedder_controls(max_output_tokens=8)),
            state,
            context(
                state,
                worker=R2AGFixtureWorker(),
                embedder=FixtureEmbedder(vectors()),
            ),
        )
        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual(record.usage.embedding_calls, 1)
        self.assertEqual(record.usage.input_tokens, 12)
        self.assertEqual(record.usage.output_tokens, 2)

    def test_zero_context_budget_fails_before_embedder_and_worker(self) -> None:
        state = r2ag_state(
            budget=ExecutionBudget(
                max_model_calls=1,
                max_output_tokens=8,
                max_context_tokens=0,
            )
        )
        worker = R2AGFixtureWorker()
        embedder = FixtureEmbedder(vectors())

        with self.assertRaisesRegex(BudgetExceededError, "context-token budget"):
            adapter(controls=embedder_controls(max_output_tokens=8)).run(
                state,
                context(state, worker=worker, embedder=embedder),
            )

        self.assertEqual(worker.calls, 0)
        self.assertEqual(embedder.calls, 0)

    def test_worker_runtime_version_identity_fails_before_worker_call(self) -> None:
        state = r2ag_state()
        worker = R2AGFixtureWorker()
        original_fingerprint = worker.fingerprint
        worker.fingerprint = lambda: {
            **dict(original_fingerprint()),
            "runtime_dependencies": {
                **_RUNTIME_DEPENDENCIES,
                "transformers": "4.41.0",
            },
        }

        with self.assertRaisesRegex(ExternalDependencyError, "dependency versions"):
            adapter().run(
                state,
                context(
                    state,
                    worker=worker,
                    embedder=FixtureEmbedder(vectors()),
                ),
            )

        self.assertEqual(worker.calls, 0)

    def test_artifact_identity_is_hash_bound_and_rejects_mutable_revisions(self) -> None:
        identity = artifacts()
        self.assertEqual(len(identity.identity_sha256), 64)
        self.assertEqual(identity.as_mapping()["checkpoint_num_documents"], 3)
        kwargs = dict(identity.as_mapping())
        kwargs["checkpoint_revision"] = "latest"
        with self.assertRaisesRegex(ValidationError, "immutable"):
            R2AGArtifactIdentity(**kwargs)


if __name__ == "__main__":
    unittest.main()
