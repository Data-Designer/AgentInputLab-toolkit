from __future__ import annotations

import hashlib
import unittest
from collections.abc import Callable, Mapping
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    CompositeResult,
    ContextBlock,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import (
    MCEOfflineRecipe,
    RQRAGLicenseClearance,
    RQRAGReleased7BExternal,
    SmartSearchArtifactIdentity,
    SmartSearchCommonSupportQueries,
    SmartSearchFrozenInference,
)
from agentinputlab.recipes.mce import (
    MCE_EMPTY_INITIAL_CONTEXT_SHA256,
    MCE_PROTOCOL,
    MCE_PROVIDER_SUBCALL_SEED_MODE,
    MCE_SEED_MODE,
    MCE_SEMANTIC_PATCH_MANIFEST,
    MCE_SEMANTIC_PATCH_MANIFEST_SHA256,
    MCE_UPSTREAM_COMMIT,
    MCE_UPSTREAM_MAIN_SHA256,
)

_MCE_INTERFACE = {
    "environment": "fixture",
    "interfaces": ({"name": "retrieve_context", "signature": "(str) -> str"},),
}
_MCE_ENVIRONMENT = {
    "name": "fixture",
    "revision": "fixture@v2",
    "source_sha256": "a" * 64,
}
_MCE_MODELS = {
    role: {
        "provider": "fixture",
        "model": f"{role}-v1",
        "revision": "1",
        "configuration_sha256": digest * 64,
    }
    for role, digest in (("target", "1"), ("meta", "2"), ("base", "3"))
}
_MCE_SOURCE_MANIFEST = {
    "schema": "fixture-source-v2",
    "commit": MCE_UPSTREAM_COMMIT,
    "tree": "4" * 40,
}
_MCE_LICENSE_MANIFEST = {
    "schema": "fixture-license-v2",
    "license_id": "MIT",
    "sha256": "5" * 64,
}
_MCE_PROVENANCE_MANIFEST = {
    "schema": "fixture-provenance-v2",
    "commit": MCE_UPSTREAM_COMMIT,
    "paper_result_parity": False,
}
_MCE_STORE = {
    "mode": "content_addressed_v1",
    "persistent": True,
    "worker_resolve_supported": True,
    "root_identity_sha256": "6" * 64,
}


class FakeMethodWorker:
    worker_id = "remaining-recipes-fixture"

    def __init__(
        self,
        payload: Callable[[Any], Mapping[str, Any]],
        *,
        fingerprint: Mapping[str, Any] | None = None,
        model_calls: int = 1,
        retrieval_calls: int = 0,
        response_extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._payload = payload
        self._fingerprint = dict(
            fingerprint
            or {
                "worker_id": self.worker_id,
                "image_digest": "sha256:remaining-recipes-fixture",
            }
        )
        self.model_calls = model_calls
        self.retrieval_calls = retrieval_calls
        self.response_extra = dict(response_extra or {})
        self.calls = 0
        self.last_request: Any = None

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError("tests use the typed call_controlled seam")

    def call_controlled(self, request: Any) -> Mapping[str, Any]:
        self.calls += 1
        self.last_request = request
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": request.upstream_commit,
            "protocol_version": request.protocol_version,
            "payload": dict(self._payload(request)),
            "usage": {
                "model_calls": self.model_calls,
                "retrieval_calls": self.retrieval_calls,
                "input_tokens": 32 if self.model_calls else 0,
                "output_tokens": 8 if self.model_calls else 0,
            },
            **self.response_extra,
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return self._fingerprint


def smartsearch_identity() -> SmartSearchArtifactIdentity:
    return SmartSearchArtifactIdentity(
        model_snapshot_manifest_sha256="1" * 64,
        model_snapshot_tree_sha256="a" * 64,
        tokenizer_revision="82bfe8a845e3e08f0d9d00d8e1e2cfb6d50b5da8",
        served_model_name="smartsearch-fixture",
        model_deployment_fingerprint_sha256="2" * 64,
        context_admission_fingerprint_sha256="b" * 64,
        retriever_revision="retriever-fixture-v1",
        retriever_deployment_fingerprint_sha256="3" * 64,
        corpus_sha256="4" * 64,
        index_sha256="5" * 64,
        retrieval_config_sha256="6" * 64,
        compatibility_manifest_sha256="7" * 64,
    )


def smartsearch_payload(*, forbidden_key: str | None = None) -> Mapping[str, Any]:
    identity = smartsearch_identity()
    payload: dict[str, Any] = {
        "profile_id": identity.profile_id,
        "artifact_identity_sha256": identity.identity_sha256,
        "artifacts": dict(identity.as_mapping()),
        "gold_answer_used": False,
        "test_time_reward_used": False,
        "external_query_refiner_used": False,
        "training_code_loaded": False,
        "trajectory": [
            {
                "index": 0,
                "query": "who wrote Dune",
                "raw_search_text": "who wrote Dune",
                "document_ids": ["doc-1"],
                "raw_search_hash": hashlib.sha256(b"who wrote Dune").hexdigest(),
                "tag_parse_status": "ok",
                "retrieval_request_sha256": "8" * 64,
                "retrieval_response_sha256": "9" * 64,
            }
        ],
        "evidence": [
            {
                "document_id": "doc-1",
                "text": "Frank Herbert wrote Dune.",
                "source": "smartsearch.wikipedia_search",
                "score": 0.9,
                "first_seen_step": 0,
                "corpus_sha256": identity.corpus_sha256,
                "index_sha256": identity.index_sha256,
                "document_sha256": content_hash(
                    {"document_id": "doc-1", "text": "Frank Herbert wrote Dune."}
                ),
                "metadata": {},
            }
        ],
        "final_answer": "Frank Herbert",
        "termination": "answer",
        "raw_transcript_sha256": "a" * 64,
        "runtime_attestation": {
            "source_commit": "5e9e7ec0d6af34f45fad145d4bff442529e7e6ff",
            "source_attestation": "verified_archive_snapshot",
            "compatibility_manifest_sha256": identity.compatibility_manifest_sha256,
            "model_transport": identity.model_transport,
            "model_snapshot_tree_sha256": identity.model_snapshot_tree_sha256,
            "context_admission_fingerprint_sha256": (identity.context_admission_fingerprint_sha256),
            "contract_hashes_verified": True,
            "immutable_model_identity_handshake_required": True,
            "context_admission_before_generation": True,
            "official_loop_called": True,
            "outer_retry_disabled": True,
            "dangerous_sandbox_used": False,
            "training_code_loaded": False,
            "seed_applied": False,
        },
    }
    if forbidden_key is not None:
        payload[forbidden_key] = "must never reach frozen inference"
    return payload


def rqrag_adapter(
    worker: FakeMethodWorker,
    *,
    license_clearance: RQRAGLicenseClearance | None = None,
) -> RQRAGReleased7BExternal:
    return RQRAGReleased7BExternal(
        client=worker,
        checkpoint_revision="rqrag-7b-v1",
        tokenizer_revision="tokenizer-v1",
        retriever_revision="retriever-v1",
        token_protocol_sha256="b" * 64,
        license_clearance=license_clearance,
    )


def mce_examples() -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    return (
        (
            {
                "example_id": "train-1",
                "input": {"query": "train input"},
                "expected_output": "train output",
            },
        ),
        (
            {
                "example_id": "validation-1",
                "input": {"query": "validation input"},
                "expected_output": "validation output",
            },
        ),
    )


def mce_recipe(worker: FakeMethodWorker) -> MCEOfflineRecipe:
    return MCEOfflineRecipe(
        client=worker,
        interface_signature_sha256=content_hash(_MCE_INTERFACE),
        initial_context_sha256=MCE_EMPTY_INITIAL_CONTEXT_SHA256,
        interface_signature=_MCE_INTERFACE,
        environment_identity=_MCE_ENVIRONMENT,
        license_manifest_allowlist=(content_hash(_MCE_LICENSE_MANIFEST),),
        provenance_manifest_allowlist=(content_hash(_MCE_PROVENANCE_MANIFEST),),
        allow_fixture_runtime=True,
    )


def attested_mce_fingerprint() -> Mapping[str, Any]:
    storage_sha256 = content_hash(_MCE_STORE)
    return {
        "worker_id": "mce-sandbox-fixture",
        "protocol_version": MCE_PROTOCOL,
        "upstream_commit": MCE_UPSTREAM_COMMIT,
        "source_tree": "4" * 40,
        "source_manifest": _MCE_SOURCE_MANIFEST,
        "source_manifest_sha256": content_hash(_MCE_SOURCE_MANIFEST),
        "license_manifest": _MCE_LICENSE_MANIFEST,
        "license_manifest_sha256": content_hash(_MCE_LICENSE_MANIFEST),
        "provenance_manifest": _MCE_PROVENANCE_MANIFEST,
        "provenance_manifest_sha256": content_hash(_MCE_PROVENANCE_MANIFEST),
        "semantic_patch_manifest": MCE_SEMANTIC_PATCH_MANIFEST,
        "semantic_patch_manifest_sha256": MCE_SEMANTIC_PATCH_MANIFEST_SHA256,
        "artifact_store": _MCE_STORE,
        "storage_fingerprint_sha256": storage_sha256,
        "interface_signature_sha256": content_hash(_MCE_INTERFACE),
        "initial_context_sha256": MCE_EMPTY_INITIAL_CONTEXT_SHA256,
        "environment_identity_sha256": content_hash(_MCE_ENVIRONMENT),
        "model_fingerprints_sha256": content_hash(_MCE_MODELS),
        "fixture_mode": True,
        "image_digest": "sha256:" + "7" * 64,
        "sandbox_attestation": {
            "attestation_id": "mce-attestation-v2",
            "runtime_mode": "fixture",
            "source_mode": "fixture",
            "isolation_kind": "container",
            "image_digest": "sha256:" + "7" * 64,
            "policy_sha256": "e" * 64,
            "test_report_sha256": "f" * 64,
            "network_mode": "none",
            "dataset_mount_mode": "read_only",
            "workspace_scope": "per_run",
            "generated_code_execution": "isolated_worker_only",
            "test_status": "passed",
            "validation_oracle_mode": "evaluator_only",
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "usage_accounting_mode": "trusted_supervisor_ipc",
            "provider_guard_mode": "provider_transport_pre_call",
            "seed_mode": MCE_SEED_MODE,
            "provider_subcall_seed_mode": MCE_PROVIDER_SUBCALL_SEED_MODE,
            "interface_validation_mode": "remote_static_no_import",
            "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
            "upstream_run_iteration_source_sha256": MCE_UPSTREAM_MAIN_SHA256,
            "host_secrets_mounted": False,
            "structured_egress_only": True,
            "generated_agents_validation_access": False,
            "generated_workspace_oracle_visible": False,
            "pre_call_budget_enforced": True,
            "wall_timeout_enforced": True,
            "source_read_only": True,
            "artifact_store_persistent": True,
            "model_identity_enforced": True,
            "environment_identity_verified": True,
            "interface_signature_verified": True,
            "semantic_patch_applied": True,
            "artifact_store_fingerprint_sha256": storage_sha256,
            "semantic_patch_manifest_sha256": MCE_SEMANTIC_PATCH_MANIFEST_SHA256,
            "resource_limits": {
                "cpu_seconds": 60,
                "memory_bytes": 1_073_741_824,
                "processes": 16,
                "wall_seconds": 120,
            },
            "required_worker_contract": {
                "oracle_mount_mode": "supervisor_only_hidden",
                "oracle_process_boundary": "trusted_supervisor_out_of_process",
                "generated_workspace_oracle_visible": False,
                "provider_guard_mode": "provider_transport_pre_call",
                "seed_mode": MCE_SEED_MODE,
                "provider_subcall_seed_mode": MCE_PROVIDER_SUBCALL_SEED_MODE,
                "validation_scoring_mode": "supervisor_only",
            },
        },
    }


def mce_context(worker: FakeMethodWorker) -> InterventionContext:
    return InterventionContext(
        external_runtime=worker,
        run_config={"mce_model_fingerprints": _MCE_MODELS},
    )


def inline_egress_payload(request: Any, forbidden_key: str) -> Mapping[str, Any]:
    model_fingerprints = request.payload["model_fingerprints"]
    return {
        "sandbox_attestation_id": request.payload["sandbox_attestation_id"],
        "host_code_executed": False,
        "artifact_egress": "manifest_and_opaque_handle_only",
        "split_audit_sha256": request.payload["split_audit_sha256"],
        "interface_signature_sha256": request.payload["interface_signature_sha256"],
        "initial_context_sha256": request.payload["initial_context_sha256"],
        "environment_identity_sha256": request.payload["environment_identity_sha256"],
        "model_fingerprints_sha256": content_hash(model_fingerprints),
        "runtime_binding_sha256": request.payload["runtime_binding_sha256"],
        "source_tree": request.payload["source_tree"],
        "source_manifest_sha256": request.payload["source_manifest_sha256"],
        "license_manifest_sha256": request.payload["license_manifest_sha256"],
        "provenance_manifest_sha256": request.payload["provenance_manifest_sha256"],
        "semantic_patch_manifest_sha256": request.payload["semantic_patch_manifest_sha256"],
        "storage_fingerprint_sha256": request.payload["storage_fingerprint_sha256"],
        "files": [
            {
                "path": "context/generated.py",
                "sha256": "1" * 64,
                "byte_size": 12,
                "media_type": "text/x-python",
                "file_type": "regular",
                "generated": True,
                "executable": False,
                forbidden_key: "print('must not leave sandbox')",
            }
        ],
    }


class RemainingRecipeTests(unittest.TestCase):
    def test_smartsearch_returns_composite_result_without_forwarding_test_labels(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: smartsearch_payload(),
            retrieval_calls=1,
        )
        state = FrozenAgentState(
            query="Who wrote Dune?",
            task_id="smartsearch-1",
            metadata={"gold_answer": "Frank Herbert", "reward": 1.0},
            budget=ExecutionBudget(
                max_model_calls=10,
                max_retrieval_calls=10,
                max_output_tokens=81_920,
            ),
        )
        adapter = SmartSearchFrozenInference(
            client=worker,
            artifacts=smartsearch_identity(),
        )

        result = adapter.run(state, InterventionContext())

        self.assertIsInstance(result, CompositeResult)
        self.assertEqual(result.final_answer, "Frank Herbert")
        self.assertEqual(tuple(block.block_id for block in result.evidence), ("doc-1",))
        self.assertEqual(result.trace.counters["search_steps"], 1)
        self.assertNotIn("gold_answer", worker.last_request.payload)
        self.assertNotIn("reward", worker.last_request.payload)

    def test_smartsearch_common_support_discards_answer_and_appends_search_queries(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: smartsearch_payload(),
            retrieval_calls=1,
        )
        state = FrozenAgentState(
            query="Who wrote Dune?",
            task_id="smartsearch-common-1",
            instruction="answer using the arena tools",
            context=(ContextBlock("arena-visible context", block_id="arena-1"),),
            metadata={"gold_answer": "must remain outside the worker"},
            tool_names=("arena_tool",),
            budget=ExecutionBudget(
                max_model_calls=10,
                max_retrieval_calls=10,
                max_output_tokens=81_920,
            ),
        )
        intervention = SmartSearchCommonSupportQueries(
            client=worker,
            artifacts=smartsearch_identity(),
        )

        patch = intervention.propose(state, InterventionContext())

        self.assertIsInstance(patch, InputPatch)
        self.assertEqual(patch.source_state_hash, state.state_hash)
        self.assertEqual(patch.operations[0].op, PatchOp.APPEND_QUERY)
        self.assertIn("who wrote Dune", patch.operations[0].value)
        self.assertTrue(patch.metadata["terminal_answer_discarded"])
        self.assertFalse(patch.metadata["retrieved_evidence_forwarded"])
        self.assertFalse(patch.metadata["paper_result_parity_claimed"])
        self.assertEqual(worker.last_request.payload["question"], state.query)
        self.assertNotIn("gold_answer", worker.last_request.payload)
        self.assertNotIn("instruction", worker.last_request.payload)
        self.assertNotIn("tool_names", worker.last_request.payload)

    def test_smartsearch_rejects_reward_and_gold_fields_from_worker(self) -> None:
        for forbidden_key in ("reward", "gold_answer"):
            with self.subTest(forbidden_key=forbidden_key):
                worker = FakeMethodWorker(
                    lambda _request, key=forbidden_key: smartsearch_payload(forbidden_key=key),
                    retrieval_calls=1,
                )
                adapter = SmartSearchFrozenInference(
                    client=worker,
                    artifacts=smartsearch_identity(),
                )

                with self.assertRaisesRegex(AlgorithmOutputError, "forbidden test-time field"):
                    adapter.run(
                        FrozenAgentState(
                            query="question",
                            budget=ExecutionBudget(
                                max_model_calls=10,
                                max_retrieval_calls=10,
                                max_output_tokens=81_920,
                            ),
                        ),
                        InterventionContext(),
                    )

    def test_smartsearch_rejects_silently_dropped_context_and_tools(self) -> None:
        for state in (
            FrozenAgentState(
                query="question",
                context=(ContextBlock("unconsumed evidence", block_id="d1"),),
            ),
            FrozenAgentState(query="question", tool_names=("calculator",)),
        ):
            with self.subTest(state=state):
                worker = FakeMethodWorker(lambda _request: smartsearch_payload())
                adapter = SmartSearchFrozenInference(
                    client=worker,
                    artifacts=smartsearch_identity(),
                )
                with self.assertRaisesRegex(ValidationError, "one question only"):
                    adapter.run(state, InterventionContext())
                self.assertEqual(worker.calls, 0)

    def test_rqrag_license_and_patch_gates_fail_before_worker_call(self) -> None:
        worker = FakeMethodWorker(lambda _request: {})
        with self.assertRaisesRegex(ExternalDependencyError, "license-gated"):
            rqrag_adapter(worker).run(
                FrozenAgentState(query="question"),
                InterventionContext(),
            )
        self.assertEqual(worker.calls, 0)

        clearance = RQRAGLicenseClearance(
            source_license_id="MIT",
            checkpoint_license_id="CUSTOM-CLEARED",
            dataset_license_id="CC-BY-4.0",
            approved_by="legal-reviewer",
            approval_reference="LEGAL-2026-001",
        )
        with self.assertRaisesRegex(ExternalDependencyError, "patch attestation"):
            rqrag_adapter(worker, license_clearance=clearance).run(
                FrozenAgentState(query="question"),
                InterventionContext(),
            )
        self.assertEqual(worker.calls, 0)

    def test_mce_missing_sandbox_attestation_fails_before_worker_call(self) -> None:
        worker = FakeMethodWorker(lambda _request: {})
        train, validation = mce_examples()

        with self.assertRaisesRegex(ExternalDependencyError, "lacks a sandbox_attestation"):
            mce_recipe(worker).fit(
                train,
                validation_examples=validation,
                context=mce_context(worker),
            )
        self.assertEqual(worker.calls, 0)

    def test_mce_rejects_inline_code_and_content_egress(self) -> None:
        train, validation = mce_examples()
        for forbidden_key in ("inline_code", "content", "body", "blob", "logs"):
            with self.subTest(forbidden_key=forbidden_key):
                worker = FakeMethodWorker(
                    lambda request, key=forbidden_key: inline_egress_payload(request, key),
                    fingerprint=attested_mce_fingerprint(),
                )

                with self.assertRaisesRegex(
                    AlgorithmOutputError,
                    "inline generated-code/content egress",
                ):
                    mce_recipe(worker).fit(
                        train,
                        validation_examples=validation,
                        context=mce_context(worker),
                    )
                self.assertEqual(worker.calls, 1)
                self.assertEqual(
                    worker.last_request.payload["semantic_patch_manifest"],
                    MCE_SEMANTIC_PATCH_MANIFEST,
                )
                self.assertEqual(
                    worker.last_request.payload["semantic_patch_manifest_sha256"],
                    MCE_SEMANTIC_PATCH_MANIFEST_SHA256,
                )

    def test_mce_rejects_undeclared_response_envelope_and_payload_fields(self) -> None:
        train, validation = mce_examples()
        envelope_worker = FakeMethodWorker(
            lambda request: inline_egress_payload(request, "content"),
            fingerprint=attested_mce_fingerprint(),
            response_extra={"debug_logs": "must not cross sandbox boundary"},
        )
        with self.assertRaisesRegex(
            AlgorithmOutputError,
            "response envelope contains undeclared field",
        ):
            mce_recipe(envelope_worker).fit(
                train,
                validation_examples=validation,
                context=mce_context(envelope_worker),
            )

        def payload_with_extra(request: Any) -> Mapping[str, Any]:
            payload = dict(inline_egress_payload(request, "content"))
            payload["debug_blob"] = "must not cross sandbox boundary"
            return payload

        payload_worker = FakeMethodWorker(
            payload_with_extra,
            fingerprint=attested_mce_fingerprint(),
        )
        with self.assertRaisesRegex(
            AlgorithmOutputError,
            "response payload contains undeclared field",
        ):
            mce_recipe(payload_worker).fit(
                train,
                validation_examples=validation,
                context=mce_context(payload_worker),
            )

    def test_mce_rejects_undeclared_error_envelope_fields(self) -> None:
        train, validation = mce_examples()

        class ErrorWorker(FakeMethodWorker):
            def call_controlled(self, request: Any) -> Mapping[str, Any]:
                self.calls += 1
                self.last_request = request
                return {
                    "status": "error",
                    "failure_code": "FIXTURE_FAILURE",
                    "message": "safe failure",
                    "consumed": False,
                    "debug_logs": "must not cross sandbox boundary",
                }

        worker = ErrorWorker(
            lambda _request: {},
            fingerprint=attested_mce_fingerprint(),
        )
        with self.assertRaisesRegex(
            AlgorithmOutputError,
            "response envelope contains undeclared field",
        ):
            mce_recipe(worker).fit(
                train,
                validation_examples=validation,
                context=mce_context(worker),
            )

    def test_mce_model_fingerprints_reject_secret_or_extension_fields(self) -> None:
        train, validation = mce_examples()
        worker = FakeMethodWorker(
            lambda _request: {},
            fingerprint=attested_mce_fingerprint(),
        )
        models = {role: dict(value) for role, value in _MCE_MODELS.items()}
        models["meta"]["api_key"] = "must-not-cross-worker-boundary"
        with self.assertRaisesRegex(
            ExternalDependencyError,
            "must contain exactly",
        ):
            mce_recipe(worker).fit(
                train,
                validation_examples=validation,
                context=InterventionContext(
                    external_runtime=worker,
                    run_config={"mce_model_fingerprints": models},
                ),
            )
        self.assertEqual(worker.calls, 0)

    def test_mce_rejects_any_undeclared_artifact_file_field(self) -> None:
        train, validation = mce_examples()
        worker = FakeMethodWorker(
            lambda request: inline_egress_payload(request, "debug_trace"),
            fingerprint=attested_mce_fingerprint(),
        )
        with self.assertRaisesRegex(
            AlgorithmOutputError,
            "artifact file contains undeclared field",
        ):
            mce_recipe(worker).fit(
                train,
                validation_examples=validation,
                context=mce_context(worker),
            )

    def test_mce_rejects_tampered_semantic_patch_manifest_before_call(self) -> None:
        train, validation = mce_examples()
        fingerprint = dict(attested_mce_fingerprint())
        tampered_manifest = dict(MCE_SEMANTIC_PATCH_MANIFEST)
        tampered_manifest["patch_id"] = "unreviewed.patch"
        fingerprint["semantic_patch_manifest"] = tampered_manifest
        fingerprint["semantic_patch_manifest_sha256"] = content_hash(tampered_manifest)
        worker = FakeMethodWorker(lambda _request: {}, fingerprint=fingerprint)

        with self.assertRaisesRegex(
            ExternalDependencyError,
            "semantic patch manifest does not match",
        ):
            mce_recipe(worker).fit(
                train,
                validation_examples=validation,
                context=mce_context(worker),
            )
        self.assertEqual(worker.calls, 0)


if __name__ == "__main__":
    unittest.main()
