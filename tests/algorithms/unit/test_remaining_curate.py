from __future__ import annotations

import hashlib
import unittest
from collections.abc import Callable, Mapping
from typing import Any

from agentinputlab.algorithms.curate.acon import (
    ACONArtifactIdentity,
    ACONHistoryControls,
    ACONHistoryRuntimeExternal,
    ACONObservationControls,
    ACONObservationRuntimeExternal,
)
from agentinputlab.algorithms.curate.recomp import (
    RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
    RECOMP_COMMON_SUPPORT_RELEASE,
    RECOMP_COMMON_SUPPORT_SPLITTER,
    RECOMPAbstractiveExternal,
    RECOMPExtractiveCommonSupport,
    RECOMPExtractiveControls,
    RECOMPExtractiveExternal,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ContextBlock,
    FrozenAgentState,
    InterventionContext,
    Message,
    PatchOp,
)
from agentinputlab.contracts.serialization import content_hash


class FakeMethodWorker:
    worker_id = "remaining-curate-fixture"

    def __init__(
        self,
        payload: Callable[[Any], Mapping[str, Any]],
        *,
        model_calls: int = 1,
        preflight_model_calls: int = 0,
    ) -> None:
        self._payload = payload
        self.model_calls = model_calls
        self.preflight_model_calls = preflight_model_calls
        self.calls = 0
        self.last_request: Any = None

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError("tests use the typed call_controlled seam")

    def call_controlled(self, request: Any) -> Mapping[str, Any]:
        self.calls += 1
        self.last_request = request
        phase = request.payload.get("phase")
        response_payload = (
            {}
            if request.method_id == "recomp" and phase == "preflight"
            else dict(self._payload(request))
        )
        if request.method_id == "acon":
            artifacts = request.payload["artifacts"]
            base_payload = {
                key: value
                for key, value in request.payload.items()
                if key not in {"phase", "decision_token"}
            }
            response_payload.update(
                decision_token=(
                    request.payload["decision_token"]
                    if phase == "execute"
                    else content_hash(base_payload)
                ),
                profile_id=f"fixture.{request.release_id}",
                compatibility_manifest_sha256="a" * 64,
                runtime_attestation={
                    "optimizer_deployment_fingerprint_hash": artifacts[
                        "optimizer_deployment_fingerprint_hash"
                    ],
                    "target_backend_fingerprint_hash": artifacts["target_backend_fingerprint_hash"],
                    "tokenizer_fingerprint_hash": artifacts["tokenizer_fingerprint_hash"],
                },
            )
        elif request.method_id == "recomp":
            runtime_fingerprint = {
                "device": "cpu",
                "python_version": "fixture",
                "python_implementation": "fixture",
                "dependency_versions": {
                    "torch": "fixture",
                    "transformers": "fixture",
                    "tokenizers": "fixture",
                },
                "container_digest": "",
                "custom_model_loader": True,
            }
            preflight_token = (
                "d" * 64 if phase == "preflight" else request.payload["preflight_token"]
            )
            response_payload.update(
                retriever_artifact_hash=request.payload["retriever_artifact_hash"],
                input_artifact_hash=request.payload["input_artifact_hash"],
                compatibility_manifest_sha256="c" * 64,
                execution_evidence="fake_checkout_plumbing",
                checkpoint_revision=request.payload["checkpoint_revision"],
                tokenizer_revision=request.payload["tokenizer_revision"],
                asset_manifest_hash=request.payload["asset_manifest_hash"],
                source_document_ids=list(request.payload["document_ids"]),
                checkpoint_license="NOASSERTION",
                checkpoint_license_acknowledged=True,
                checkpoint_tree_sha256="e" * 64,
                checkpoint_file_count=3,
                remaining_budget_hash=content_hash(request.payload["remaining_budget"]),
                input_tokens_required=20,
                output_tokens_required=(request.payload["controls"].get("max_target_tokens", 0)),
                preflight_token=preflight_token,
                runtime_fingerprint=runtime_fingerprint,
                runtime_fingerprint_hash=content_hash(runtime_fingerprint),
            )
        billed_calls = (
            0
            if request.method_id == "recomp" and phase == "preflight"
            else self.preflight_model_calls
            if phase == "preflight"
            else self.model_calls
        )
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": request.upstream_commit,
            "protocol_version": request.protocol_version,
            "payload": response_payload,
            "usage": {
                "model_calls": billed_calls,
                "retrieval_calls": 0,
                "input_tokens": 20 if billed_calls else 0,
                "output_tokens": 5 if billed_calls else 0,
            },
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.worker_id,
            "image_digest": "sha256:remaining-curate-fixture",
            "compatibility_manifest_sha256": "c" * 64,
            "execution_evidence": "fake_checkout_plumbing",
        }


ACON_OPTIMIZER_FINGERPRINT = {"transport": "fixture", "model": "optimizer-v1"}
ACON_TARGET_FINGERPRINT = {"backend": "fixture", "model_revision": "target-v1"}
ACON_TOKENIZER_FINGERPRINT = {"engine": "fixture", "revision": "tokenizer-v1"}

ACON_ARTIFACTS = ACONArtifactIdentity(
    optimizer_model_revision="optimizer-v1",
    target_model_revision="target-v1",
    tokenizer_revision="tokenizer-v1",
    prompt_artifact_hash=content_hash("prompt-v1"),
    guideline_artifact_hash=content_hash("guideline-v1"),
    optimizer_deployment_fingerprint_hash=content_hash(ACON_OPTIMIZER_FINGERPRINT),
    target_backend_fingerprint_hash=content_hash(ACON_TARGET_FINGERPRINT),
    tokenizer_fingerprint_hash=content_hash(ACON_TOKENIZER_FINGERPRINT),
)


class FingerprintedRetriever:
    def fingerprint(self) -> Mapping[str, Any]:
        return {"retriever": "frozen-fixture", "corpus_sha256": "d" * 64}


def recomp_context() -> InterventionContext:
    return InterventionContext(retriever=FingerprintedRetriever())


class ACONTargetBackend:
    def fingerprint(self) -> Mapping[str, Any]:
        return ACON_TARGET_FINGERPRINT


def acon_context(**kwargs: Any) -> InterventionContext:
    return InterventionContext(backend=ACONTargetBackend(), **kwargs)


def recomp_state() -> FrozenAgentState:
    return FrozenAgentState(
        query="Which evidence answers the question?",
        task_id="recomp-1",
        context=(
            ContextBlock("Alpha. Beta.", block_id="doc-1", source="fixture"),
            ContextBlock("Gamma.", block_id="doc-2", source="fixture"),
        ),
    )


def recomp_identity(request: Any) -> dict[str, Any]:
    payload = request.payload
    return {
        "checkpoint_revision": payload["checkpoint_revision"],
        "tokenizer_revision": payload["tokenizer_revision"],
        "asset_manifest_hash": payload["asset_manifest_hash"],
        "source_document_ids": list(payload["document_ids"]),
    }


def extractive_adapter(worker: FakeMethodWorker) -> RECOMPExtractiveExternal:
    return RECOMPExtractiveExternal(
        client=worker,
        checkpoint_revision="extractive-v1",
        tokenizer_revision="tokenizer-v1",
        sentence_splitter_revision="splitter-v1",
        asset_manifest_hash="a" * 64,
        asset_license_acknowledged=True,
        controls=RECOMPExtractiveControls(max_sentences=2),
    )


def abstractive_adapter(worker: FakeMethodWorker) -> RECOMPAbstractiveExternal:
    return RECOMPAbstractiveExternal(
        client=worker,
        checkpoint_revision="abstractive-v1",
        tokenizer_revision="tokenizer-v1",
        input_template_hash=RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
        asset_manifest_hash="b" * 64,
        asset_license_acknowledged=True,
    )


def history_state() -> FrozenAgentState:
    return FrozenAgentState(
        query="continue",
        instruction="complete the task",
        task_id="acon-history-1",
        messages=(
            Message("system", "system prompt"),
            Message("user", "first task prompt"),
            Message("assistant", "old action"),
            Message("user", "old observation"),
            Message("assistant", "latest action"),
            Message("user", "latest observation"),
        ),
    )


def history_payload(
    request: Any,
    *,
    raw_input_tokens: int,
    applied: bool,
) -> Mapping[str, Any]:
    return {
        **ACON_ARTIFACTS.as_payload(),
        "counted_material_hash": request.payload["counted_material_hash"],
        "raw_input_tokens": raw_input_tokens,
        "threshold_triggered": applied,
        "applied": applied,
        "summary": "old work summarized" if applied else "",
        "raw_response": "# History Summary\nold work summarized" if applied else "",
    }


def observation_state() -> FrozenAgentState:
    return FrozenAgentState(
        query="inspect the result",
        instruction="use the tool result",
        task_id="acon-observation-1",
        messages=(
            Message("system", "system prompt"),
            Message("user", "task prompt"),
            Message("assistant", "tool call"),
        ),
        context=(
            ContextBlock("earlier evidence", block_id="ctx-1", source="tool"),
            ContextBlock("a very long tool observation", block_id="obs-1", source="tool"),
        ),
    )


def observation_payload(
    request: Any,
    *,
    raw_input_tokens: int,
    applied: bool,
) -> Mapping[str, Any]:
    return {
        **ACON_ARTIFACTS.as_payload(),
        "source_context_id": request.payload["source_context_id"],
        "history_text_hash": request.payload["history_text_hash"],
        "opt_args_hash": request.payload["opt_args_hash"],
        "raw_input_tokens": raw_input_tokens,
        "threshold_triggered": applied,
        "applied": applied,
        "optimized_observation": (
            "compressed tool observation" if applied else request.payload["observation"]
        ),
        "raw_response": ("# Refined Observation\ncompressed tool observation" if applied else ""),
    }


class RemainingCurateTests(unittest.TestCase):
    def test_recomp_common_support_presegments_and_rebinds_selected_offsets(self) -> None:
        def response(request: Any) -> Mapping[str, Any]:
            rows = []
            scores = (0.2, 0.9, 0.8)
            for index, (document, score) in enumerate(
                zip(request.payload["documents"], scores, strict=True)
            ):
                rows.append(
                    {
                        "sentence_id": document["block_id"],
                        "document_id": document["block_id"],
                        "document_index": index,
                        "sentence_index": 0,
                        "char_start": 0,
                        "char_end": len(document["text"]),
                        "text": document["text"],
                        "score": score,
                    }
                )
            return {
                **recomp_identity(request),
                "sentence_splitter_revision": RECOMP_COMMON_SUPPORT_SPLITTER,
                "candidate_unit_kind": "presegmented_sentence",
                "scoring_api": "run_extractive_compressor.get_contriever_scores(top_k=-1)",
                "sentence_scores": rows,
            }

        worker = FakeMethodWorker(response)
        state = recomp_state()
        adapter = RECOMPExtractiveCommonSupport(
            checkpoint_revision="extractive-v1",
            tokenizer_revision="tokenizer-v1",
            asset_manifest_hash="a" * 64,
            asset_license_acknowledged=True,
            controls=RECOMPExtractiveControls(max_sentences=2),
        )

        patch = adapter.propose(
            state,
            InterventionContext(retriever=FingerprintedRetriever(), external_runtime=worker),
        )
        updated = patch.apply(state)

        self.assertEqual(patch.release_id, RECOMP_COMMON_SUPPORT_RELEASE)
        self.assertEqual(tuple(block.text for block in updated.context), ("Beta.", "Gamma."))
        self.assertEqual(patch.trace.counters["candidate_sentences"], 3)
        self.assertEqual(patch.trace.counters["selected_sentences"], 2)
        self.assertEqual(updated.context[0].metadata["source_context_block_id"], "doc-1")
        self.assertEqual(updated.context[0].metadata["source_char_start"], 7)
        self.assertEqual(updated.context[0].metadata["source_char_end"], 12)
        self.assertEqual(
            updated.context[0].metadata["splitter_revision"],
            RECOMP_COMMON_SUPPORT_SPLITTER,
        )
        self.assertFalse(patch.metadata["paper_result_parity_claimed"])

    def test_recomp_extractive_scores_then_uses_frozen_assembly(self) -> None:
        worker = FakeMethodWorker(
            lambda request: {
                **recomp_identity(request),
                "sentence_splitter_revision": "splitter-v1",
                "candidate_unit_kind": "presegmented_sentence",
                "scoring_api": "run_extractive_compressor.get_contriever_scores(top_k=-1)",
                "sentence_scores": [
                    {
                        "sentence_id": "doc-1",
                        "document_id": "doc-1",
                        "document_index": 0,
                        "sentence_index": 0,
                        "char_start": 0,
                        "char_end": 12,
                        "text": "Alpha. Beta.",
                        "score": 0.2,
                    },
                    {
                        "sentence_id": "doc-2",
                        "document_id": "doc-2",
                        "document_index": 1,
                        "sentence_index": 0,
                        "char_start": 0,
                        "char_end": 6,
                        "text": "Gamma.",
                        "score": 0.8,
                    },
                ],
            }
        )
        state = recomp_state()

        patch = extractive_adapter(worker).propose(state, recomp_context())
        updated = patch.apply(state)

        self.assertEqual(tuple(block.text for block in updated.context), ("Alpha. Beta.", "Gamma."))
        self.assertEqual(patch.operations[0].op, PatchOp.REPLACE_CONTEXT)
        self.assertEqual(patch.profile_kind, "controlled_adapter")
        self.assertEqual(patch.trace.counters["scored_sentences"], 2)

    def test_recomp_extractive_rejects_source_offset_drift(self) -> None:
        worker = FakeMethodWorker(
            lambda request: {
                **recomp_identity(request),
                "sentence_splitter_revision": "splitter-v1",
                "candidate_unit_kind": "presegmented_sentence",
                "scoring_api": "run_extractive_compressor.get_contriever_scores(top_k=-1)",
                "sentence_scores": [
                    {
                        "sentence_id": "doc-1",
                        "document_id": "doc-1",
                        "document_index": 0,
                        "sentence_index": 0,
                        "char_start": 1,
                        "char_end": 6,
                        "text": "Alpha.",
                        "score": 0.9,
                    },
                    {
                        "sentence_id": "doc-2",
                        "document_id": "doc-2",
                        "document_index": 1,
                        "sentence_index": 0,
                        "char_start": 0,
                        "char_end": 6,
                        "text": "Gamma.",
                        "score": 0.8,
                    },
                ],
            }
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "offsets"):
            extractive_adapter(worker).propose(recomp_state(), recomp_context())

    def test_recomp_abstractive_empty_output_successfully_clears_context(self) -> None:
        worker = FakeMethodWorker(
            lambda request: {
                **recomp_identity(request),
                "input_template_hash": RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
                "passages_hash": request.payload["passages_hash"],
                "prompt_hash": hashlib.sha256(
                    b"Question: Which evidence answers the question?\n Document: "
                    b"Alpha. Beta.Gamma.\n Summary: "
                ).hexdigest(),
                "generation_api": "AutoModelForSeq2SeqLM.generate+batch_decode(strip)",
                "applied": True,
                "summary": "",
                "raw_generation": "",
                "selective_augmentation": True,
                "parse_status": "generated_empty",
            }
        )
        state = recomp_state()

        patch = abstractive_adapter(worker).propose(state, recomp_context())

        self.assertEqual(patch.apply(state).context, ())
        self.assertEqual(patch.operations[0].op, PatchOp.REPLACE_CONTEXT)
        self.assertTrue(patch.metadata["selective_augmentation"])
        self.assertIn("selective_augmentation_cleared_context", patch.trace.warnings)

    def test_recomp_abstractive_rejects_false_applied_on_success(self) -> None:
        worker = FakeMethodWorker(
            lambda request: {
                **recomp_identity(request),
                "input_template_hash": RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
                "passages_hash": request.payload["passages_hash"],
                "prompt_hash": hashlib.sha256(
                    b"Question: Which evidence answers the question?\n Document: "
                    b"Alpha. Beta.Gamma.\n Summary: "
                ).hexdigest(),
                "generation_api": "AutoModelForSeq2SeqLM.generate+batch_decode(strip)",
                "applied": False,
                "summary": "",
                "raw_generation": "",
                "selective_augmentation": True,
                "parse_status": "generated_empty",
            }
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "applied=true"):
            abstractive_adapter(worker).propose(recomp_state(), recomp_context())

    def test_acon_history_threshold_equality_skips_without_usage(self) -> None:
        worker = FakeMethodWorker(
            lambda request: history_payload(
                request,
                raw_input_tokens=10,
                applied=False,
            ),
            model_calls=0,
        )
        state = history_state()
        adapter = ACONHistoryRuntimeExternal(
            client=worker,
            artifacts=ACON_ARTIFACTS,
            controls=ACONHistoryControls(
                summarization_threshold=10,
                preserve_last_k_turns=1,
            ),
        )

        patch = adapter.propose(state, acon_context())

        self.assertEqual(patch.operations[0].op, PatchOp.REPLACE_MESSAGES)
        self.assertEqual(patch.apply(state).messages, state.messages)
        self.assertFalse(patch.metadata["applied"])
        self.assertEqual(patch.trace.counters["model_calls"], 0)

    def test_acon_history_greater_threshold_replaces_messages(self) -> None:
        worker = FakeMethodWorker(
            lambda request: history_payload(
                request,
                raw_input_tokens=10,
                applied=True,
            )
        )
        state = history_state()
        adapter = ACONHistoryRuntimeExternal(
            client=worker,
            artifacts=ACON_ARTIFACTS,
            controls=ACONHistoryControls(
                summarization_threshold=9,
                preserve_last_k_turns=1,
            ),
        )

        patch = adapter.propose(state, acon_context())
        updated = patch.apply(state)

        self.assertEqual(patch.operations[0].op, PatchOp.REPLACE_MESSAGES)
        self.assertIn("<HISTORY_SUMMARY>\nold work summarized", updated.messages[1].content)
        self.assertEqual(updated.messages[-2:], state.messages[-2:])
        self.assertTrue(patch.metadata["applied"])

    def test_acon_observation_replaces_only_target_context(self) -> None:
        worker = FakeMethodWorker(
            lambda request: observation_payload(
                request,
                raw_input_tokens=12,
                applied=True,
            )
        )
        state = observation_state()
        adapter = ACONObservationRuntimeExternal(
            client=worker,
            artifacts=ACON_ARTIFACTS,
            controls=ACONObservationControls(summarization_threshold=10),
        )

        patch = adapter.propose(state, acon_context())
        updated = patch.apply(state)

        self.assertEqual(patch.operations[0].op, PatchOp.REPLACE_CONTEXT)
        self.assertEqual(updated.context[0], state.context[0])
        self.assertEqual(updated.context[1].text, "compressed tool observation")
        self.assertEqual(updated.context[1].block_id, "obs-1")
        self.assertTrue(updated.context[1].metadata["acon_applied"])

    def test_acon_observation_skip_rejects_nonzero_model_usage(self) -> None:
        worker = FakeMethodWorker(
            lambda request: observation_payload(
                request,
                raw_input_tokens=10,
                applied=False,
            ),
            model_calls=1,
            preflight_model_calls=1,
        )
        adapter = ACONObservationRuntimeExternal(
            client=worker,
            artifacts=ACON_ARTIFACTS,
            controls=ACONObservationControls(summarization_threshold=10),
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "preflight cannot bill"):
            adapter.propose(observation_state(), acon_context())


if __name__ == "__main__":
    unittest.main()
