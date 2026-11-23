from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping
from typing import Any

from agentinputlab.algorithms.reformulate.infocqr import (
    InfoCQRCache,
    InfoCQRMode,
    InfoCQRRecord,
    InfoCQRZeroShotReplay,
    conversation_view_hash,
)
from agentinputlab.algorithms.reformulate.reformer import (
    ReFormeRPaperConcat,
    ReFormeRRepoNative,
)
from agentinputlab.algorithms.reformulate.rrr import RRRPromptExternal
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    ContextBlock,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
)


class FakeMethodWorker:
    worker_id = "remaining-reformulate-fixture"

    def __init__(self, payload: Callable[[Any], Mapping[str, Any]]) -> None:
        self._payload = payload
        self.calls = 0
        self.last_request: Any = None

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError("tests use the typed call_controlled seam")

    def call_controlled(self, request: Any) -> Mapping[str, Any]:
        self.calls += 1
        self.last_request = request
        response_payload = dict(self._payload(request))
        if request.method_id == "rrr":
            for key in (
                "prompt_hash",
                "reader_prompt_hash",
                "search_artifact_hash",
                "rewriter_checkpoint_revision",
                "reader_checkpoint_revision",
                "backend_artifact_hash",
                "retriever_artifact_hash",
                "input_artifact_hash",
            ):
                response_payload[key] = request.payload[key]
            response_payload.update(
                source_manifest_sha256="f" * 64,
                trajectory_hash="e" * 64,
                execution_evidence="fake_checkout_plumbing",
                stage_order=["rewrite", "retrieve", "snippet", "reader"],
            )
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": request.upstream_commit,
            "protocol_version": request.protocol_version,
            "payload": response_payload,
            "usage": {
                "model_calls": 2,
                "retrieval_calls": 1,
                "input_tokens": 32,
                "output_tokens": 12,
            },
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.worker_id,
            "image_digest": "sha256:remaining-reformulate-fixture",
            "source_manifest_sha256": "f" * 64,
        }


class FingerprintedDependency:
    def __init__(self, identity: Mapping[str, Any]) -> None:
        self.identity = identity

    def fingerprint(self) -> Mapping[str, Any]:
        return dict(self.identity)


class RemainingReformulateTests(unittest.TestCase):
    def test_rrr_happy_path_materializes_rewrite_and_retrieved_context(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "queries": ["who is the author of Dune"],
                "reader_query": "Dune author?",
                "documents": [
                    {
                        "document_id": "d1",
                        "text": "Frank Herbert wrote Dune.",
                        "title": "Dune",
                        "corpus_revision": "corpus-v1",
                        "query_index": 0,
                        "rank": 1,
                        "score": 0.9,
                        "metadata": {},
                    }
                ],
                "answer": "Frank Herbert",
                "parser_mode": "upstream_semicolon_terminator_single",
            }
        )
        state = FrozenAgentState(query="Dune author?", task_id="rrr-1")

        patch = RRRPromptExternal(
            prompt_hash="a" * 64,
            reader_prompt_hash="b" * 64,
            search_artifact_hash="c" * 64,
            source_license_acknowledged=True,
            checkpoint_license_acknowledged=True,
        ).propose(
            state,
            InterventionContext(
                backend=FingerprintedDependency({"model": "reader-fixture"}),
                external_runtime=worker,
            ),
        )
        updated = patch.apply(state)

        self.assertEqual(updated.query, "who is the author of Dune")
        self.assertEqual(tuple(block.block_id for block in updated.context), ("d1",))
        self.assertEqual(patch.release_id, "rewrite.rrr.prompt_external")
        self.assertEqual(worker.calls, 1)

    def test_rrr_rejects_reader_query_not_emitted_by_parser(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "queries": ["parsed query"],
                "reader_query": "different query",
                "documents": [],
                "answer": "",
                "parser_mode": "upstream_semicolon_terminator_single",
            }
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "reader_query"):
            RRRPromptExternal(
                prompt_hash="a" * 64,
                reader_prompt_hash="b" * 64,
                search_artifact_hash="c" * 64,
                source_license_acknowledged=True,
                checkpoint_license_acknowledged=True,
            ).propose(
                FrozenAgentState(query="question"),
                InterventionContext(
                    backend=FingerprintedDependency({"model": "reader-fixture"}),
                    external_runtime=worker,
                ),
            )

    def test_reformer_paper_profile_preserves_q_concat_r_semantics(self) -> None:
        state = FrozenAgentState(
            query="original question",
            task_id="reformer-1",
            context=(
                ContextBlock("first", block_id="d1"),
                ContextBlock("second", block_id="d2"),
            ),
        )
        worker = FakeMethodWorker(
            lambda _request: {
                "pattern_library_hash": "a" * 64,
                "selected_pattern_id": "p1",
                "selected_pattern": "replace entity with canonical name",
                "reformulation": "canonical reformulation",
                "top_document_ids": ["d1", "d2"],
                "selection_parse_status": "valid_json",
                "fallback_first_pattern_id": "",
                "semantics": "q_concat_r",
                "pattern_mode": "automatic",
            }
        )

        patch = ReFormeRPaperConcat(
            pattern_artifact_hash="a" * 64,
            source_license_acknowledged=True,
        ).propose(
            state,
            InterventionContext(
                backend=FingerprintedDependency({"model": "fixture-model"}),
                retriever=FingerprintedDependency({"index": "fixture-index"}),
                external_runtime=worker,
            ),
        )

        self.assertEqual(
            patch.apply(state).query,
            "original question canonical reformulation",
        )
        self.assertEqual(patch.metadata["paper_code_semantics"], "q_concat_r")

    def test_reformer_rejects_pattern_library_identity_drift(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "pattern_library_hash": "b" * 64,
                "selected_pattern_id": "p1",
                "selected_pattern": "pattern",
                "reformulation": "rewrite",
                "top_document_ids": [],
                "selection_parse_status": "valid_json",
                "fallback_first_pattern_id": "",
                "semantics": "r_only",
                "pattern_mode": "automatic",
            }
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "library hash mismatch"):
            ReFormeRRepoNative(
                pattern_artifact_hash="a" * 64,
                source_license_acknowledged=True,
            ).propose(
                FrozenAgentState(query="question"),
                InterventionContext(
                    backend=FingerprintedDependency({"model": "fixture-model"}),
                    retriever=FingerprintedDependency({"index": "fixture-index"}),
                    external_runtime=worker,
                ),
            )

    def test_noassertion_rewriters_require_acknowledgement_before_worker_call(self) -> None:
        worker = FakeMethodWorker(lambda _request: {})
        state = FrozenAgentState(query="question")
        adapters = (
            RRRPromptExternal(
                prompt_hash="a" * 64,
                reader_prompt_hash="b" * 64,
                search_artifact_hash="c" * 64,
            ),
            ReFormeRRepoNative(pattern_artifact_hash="a" * 64),
        )

        for adapter in adapters:
            with (
                self.subTest(adapter=type(adapter).__name__),
                self.assertRaisesRegex(
                    ExternalDependencyError,
                    "NOASSERTION",
                ),
            ):
                adapter.propose(state, InterventionContext(external_runtime=worker))
        self.assertEqual(worker.calls, 0)

    def test_infocqr_happy_path_replays_checksum_bound_rewrite(self) -> None:
        state = FrozenAgentState(query="current query", task_id="info-1")
        record = InfoCQRRecord(
            record_id="info-1",
            split="dev",
            mode=InfoCQRMode.RW_ZSL,
            source_query=state.query,
            conversation_hash=conversation_view_hash(state),
            rewrite="standalone rewritten query",
        )
        cache = InfoCQRCache(
            (record,),
            artifact_id="infocqr-dev-v1",
            artifact_sha256="a" * 64,
            dataset_revision="dataset-v1",
        )

        patch = InfoCQRZeroShotReplay(cache, split="dev").propose(
            state,
            InterventionContext(),
        )

        self.assertEqual(patch.apply(state).query, "standalone rewritten query")
        self.assertEqual(patch.metadata["artifact_sha256"], "a" * 64)
        self.assertEqual(patch.trace.counters["historical_model_calls"], 0)

    def test_infocqr_rejects_gold_derived_first_turn_rewrite_by_default(self) -> None:
        state = FrozenAgentState(query="current query", task_id="info-gold")
        cache = InfoCQRCache(
            (
                InfoCQRRecord(
                    record_id="info-gold",
                    split="test",
                    mode=InfoCQRMode.RW_ZSL,
                    source_query=state.query,
                    conversation_hash=conversation_view_hash(state),
                    rewrite="gold-derived rewrite",
                    first_turn_truth_rewrite=True,
                ),
            ),
            artifact_id="infocqr-test-v1",
            artifact_sha256="b" * 64,
            dataset_revision="dataset-v1",
        )

        with self.assertRaisesRegex(ArtifactError, "gold-derived"):
            InfoCQRZeroShotReplay(cache, split="test").propose(
                state,
                InterventionContext(),
            )


if __name__ == "__main__":
    unittest.main()
