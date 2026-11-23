from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

from agentinputlab.algorithms.acquire.adaptive_rag import (
    AdaptiveRAGArtifactIdentity,
    AdaptiveRAGControlledWorker,
    AdaptiveRAGControls,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers.method import MethodWorkerRequest
from agentinputlab.runner import CompositeInterventionRunner
from workers.adaptive_rag.driver import (
    PROTOCOL_VERSION,
    AdaptiveRAGWorkerDriver,
    DriverError,
    RetrievedDocument,
    RuntimeBundle,
    RuntimeIdentity,
    serve_jsonl,
    verify_component_manifests,
)
from workers.adaptive_rag.fixture_runtime import build_fixture_runtime


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _DriverClient:
    worker_id = "adaptive-rag-hermetic-fixture"

    def __init__(self, driver: AdaptiveRAGWorkerDriver) -> None:
        self.driver = driver

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        return dict(self.driver.handle(request))

    def call_controlled(self, request: MethodWorkerRequest) -> dict[str, Any]:
        return dict(self.driver.handle(request.as_dict()))

    def fingerprint(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "image_digest": "sha256:adaptive-rag-hermetic",
            "source_revision": "fixture-v1",
        }


class AdaptiveRAGWorkerDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / "source"
        self.prompts = root / "prompts"
        self.source.mkdir()
        self.prompts.mkdir()
        (self.source / "LICENSE").write_text("Apache fixture\n", encoding="utf-8")
        (self.source / "router.py").write_text("ROUTES = ('A', 'B', 'C')\n", encoding="utf-8")
        self.weights = root / "router.bin"
        self.weights.write_bytes(b"tensor-only-router-fixture")
        prompt_text = {
            "no_retrieval": "answer without retrieval",
            "single_retrieval": "answer from retrieved passages",
            "ircot": "reason, retrieve, then answer",
        }
        for name, text in prompt_text.items():
            (self.prompts / f"{name}.txt").write_text(text, encoding="utf-8")
        manifest = json.loads(
            (
                Path(__file__).parents[2] / "workers/adaptive_rag/compatibility_manifest.json"
            ).read_text(encoding="utf-8")
        )
        manifest["upstream"]["tree"] = "f" * 40
        manifest["upstream"]["license_sha256"] = _sha((self.source / "LICENSE").read_bytes())
        manifest["source_files"] = {"router.py": _sha((self.source / "router.py").read_bytes())}
        self.manifest = root / "compatibility.json"
        self.manifest.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        self.manifest_hash = content_hash(manifest)
        self.source_closure = content_hash(
            {
                "commit": manifest["upstream"]["commit"],
                "tree": manifest["upstream"]["tree"],
                "source_files": manifest["source_files"],
                "license_sha256": manifest["upstream"]["license_sha256"],
            }
        )
        self.identity = RuntimeIdentity(
            router_revision="fixture-router-v1",
            router_snapshot_sha256="a" * 64,
            router_weights_sha256=_sha(self.weights.read_bytes()),
            router_tokenizer_sha256="b" * 64,
            base_model_revision="fixture-generator-v1",
            generator_artifact_sha256="c" * 64,
            generator_config_sha256="d" * 64,
            retriever_revision="fixture-retriever-v1",
            retriever_implementation_sha256="e" * 64,
            retriever_config_sha256="f" * 64,
            corpus_revision="fixture-corpus-v1",
            corpus_manifest_sha256="5" * 64,
            index_revision="fixture-index-v1",
            index_manifest_sha256="6" * 64,
            fixture_runtime=True,
        )
        self.runtime = build_fixture_runtime(
            identity=self.identity,
            logits_by_question={
                "simple": {"A": 2.0, "B": 2.0, "C": -1.0},
                "single": {"A": -1.0, "B": 3.0, "C": 0.0},
                "multi": {"A": -1.0, "B": 0.0, "C": 4.0},
            },
            documents_by_query={
                "single": (RetrievedDocument("d1", "fixture first"),),
                "multi": (RetrievedDocument("d1", "fixture first"),),
                "retrieve fixture-second": (RetrievedDocument("d2", "fixture second"),),
            },
        )
        self.driver = AdaptiveRAGWorkerDriver(
            source_root=self.source,
            router_weights_path=self.weights,
            prompt_root=self.prompts,
            runtime=self.runtime,
            manifest_path=self.manifest,
            fixture_mode=True,
        )
        self.artifacts = AdaptiveRAGArtifactIdentity(
            router_checkpoint="fixture/router",
            worker_fingerprint_sha256=content_hash(_DriverClient(self.driver).fingerprint()),
            router_revision=self.identity.router_revision,
            router_snapshot_sha256=self.identity.router_snapshot_sha256,
            router_weights_sha256=self.identity.router_weights_sha256,
            router_tokenizer_sha256=self.identity.router_tokenizer_sha256,
            base_model_revision=self.identity.base_model_revision,
            generator_artifact_sha256=self.identity.generator_artifact_sha256,
            generator_config_sha256=self.identity.generator_config_sha256,
            retriever_revision=self.identity.retriever_revision,
            retriever_implementation_sha256=self.identity.retriever_implementation_sha256,
            retriever_config_sha256=self.identity.retriever_config_sha256,
            corpus_revision=self.identity.corpus_revision,
            corpus_manifest_sha256=self.identity.corpus_manifest_sha256,
            index_revision=self.identity.index_revision,
            index_manifest_sha256=self.identity.index_manifest_sha256,
            no_retrieval_prompt_sha256=_sha((self.prompts / "no_retrieval.txt").read_bytes()),
            single_retrieval_prompt_sha256=_sha(
                (self.prompts / "single_retrieval.txt").read_bytes()
            ),
            ircot_prompt_sha256=_sha((self.prompts / "ircot.txt").read_bytes()),
        )
        self.controls = AdaptiveRAGControls(
            single_retrieval_top_k=2,
            ircot_per_step_top_k=2,
            ircot_max_steps=1,
            max_total_documents=4,
            max_output_tokens=64,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_driver_rejects_symlinked_deployment_roots(self) -> None:
        linked_source = self.source.parent / "linked-source"
        linked_source.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(DriverError, "must not be a symlink"):
            AdaptiveRAGWorkerDriver(
                source_root=linked_source,
                router_weights_path=self.weights,
                prompt_root=self.prompts,
                runtime=self.runtime,
                manifest_path=self.manifest,
                fixture_mode=True,
            )

    def test_jsonl_boundary_executes_one_valid_request(self) -> None:
        request = self._request("single")
        source = StringIO(json.dumps(request) + "\n")
        sink = StringIO()

        self.assertEqual(serve_jsonl(self.driver, source, sink), 0)

        response = json.loads(sink.getvalue())
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["protocol_version"], PROTOCOL_VERSION)

    def test_retrieved_document_metadata_is_detached_and_immutable(self) -> None:
        source = {
            "source_uri": "fixture://original",
            "nested": {"values": ["original"]},
        }
        document = RetrievedDocument("immutable", "text", metadata=source)

        source["source_uri"] = "fixture://mutated"
        source["nested"]["values"].append("mutated")

        self.assertEqual(document.metadata["source_uri"], "fixture://original")
        self.assertEqual(document.metadata["nested"]["values"], ("original",))
        with self.assertRaises(TypeError):
            document.metadata["language"] = "en"  # type: ignore[index]
        with self.assertRaises(TypeError):
            document.metadata["nested"]["new"] = True  # type: ignore[index]

    def _method(self, *, allow_fixture: bool) -> AdaptiveRAGControlledWorker:
        return AdaptiveRAGControlledWorker(
            self.artifacts,
            controls=self.controls,
            allow_fixture_runtime=allow_fixture,
            expected_source_tree="f" * 40,
            expected_compatibility_manifest_sha256=self.manifest_hash,
            expected_source_closure_sha256=self.source_closure,
        )

    def test_full_worker_e2e_runs_all_three_selected_arms(self) -> None:
        client = _DriverClient(self.driver)
        expected = {
            "simple": ("A", "no_retrieval", ()),
            "single": ("B", "single_retrieval", ("d1",)),
            "multi": ("C", "ircot", ("d1", "d2")),
        }
        for question, (route, strategy, document_ids) in expected.items():
            with self.subTest(question=question):
                result = self._method(allow_fixture=True).run(
                    FrozenAgentState(query=question, task_id=f"fixture-{question}"),
                    InterventionContext(external_runtime=client, seed=3),
                )
                self.assertEqual(result.metadata["route_label"], route)
                self.assertEqual(result.metadata["strategy"], strategy)
                self.assertEqual(tuple(item.block_id for item in result.evidence), document_ids)
                self.assertEqual(result.metadata["execution_evidence_kind"], "hermetic_fixture")
                self.assertTrue(result.metadata["controlled_method_execution"])
                self.assertTrue(result.metadata["controlled_upstream_router_parity"])
                self.assertFalse(result.metadata["upstream_arm_execution_parity"])
                self.assertFalse(result.metadata["official_parity_claimed"])

    def test_ircot_aggregate_output_can_exceed_per_call_cap(self) -> None:
        self.controls = AdaptiveRAGControls(
            single_retrieval_top_k=2,
            ircot_per_step_top_k=2,
            ircot_max_steps=1,
            max_total_documents=4,
            max_output_tokens=1,
        )

        result = self._method(allow_fixture=True).run(
            FrozenAgentState(query="multi", task_id="fixture-multi-cap-one"),
            InterventionContext(external_runtime=_DriverClient(self.driver), seed=3),
        )

        self.assertEqual(result.metadata["strategy"], "ircot")
        self.assertEqual(result.trace.counters["model_calls"], 3)
        self.assertEqual(result.trace.counters["output_tokens"], 3)

    def test_production_host_rejects_fixture_by_default(self) -> None:
        with self.assertRaisesRegex(AlgorithmOutputError, "rejects fixture"):
            self._method(allow_fixture=False).run(
                FrozenAgentState(query="simple"),
                InterventionContext(external_runtime=_DriverClient(self.driver)),
            )

    def test_host_rejects_unlocked_worker_fingerprint_before_call(self) -> None:
        class DriftedClient(_DriverClient):
            def fingerprint(self) -> dict[str, Any]:
                return {
                    "worker_id": "different-worker",
                    "image_digest": "sha256:different",
                    "source_revision": "different-v1",
                }

        before = self.runtime.router.calls
        with self.assertRaisesRegex(
            ValidationError,
            "worker fingerprint does not match",
        ):
            self._method(allow_fixture=True).run(
                FrozenAgentState(query="simple"),
                InterventionContext(external_runtime=DriftedClient(self.driver)),
            )
        self.assertEqual(self.runtime.router.calls, before)

    def test_hermetic_components_cannot_be_relabelled_as_production(self) -> None:
        production_identity = RuntimeIdentity(
            router_revision=self.identity.router_revision,
            router_snapshot_sha256=self.identity.router_snapshot_sha256,
            router_weights_sha256=self.identity.router_weights_sha256,
            router_tokenizer_sha256=self.identity.router_tokenizer_sha256,
            base_model_revision=self.identity.base_model_revision,
            generator_artifact_sha256=self.identity.generator_artifact_sha256,
            generator_config_sha256=self.identity.generator_config_sha256,
            retriever_revision=self.identity.retriever_revision,
            retriever_implementation_sha256=self.identity.retriever_implementation_sha256,
            retriever_config_sha256=self.identity.retriever_config_sha256,
            corpus_revision=self.identity.corpus_revision,
            corpus_manifest_sha256=self.identity.corpus_manifest_sha256,
            index_revision=self.identity.index_revision,
            index_manifest_sha256=self.identity.index_manifest_sha256,
            fixture_runtime=False,
        )
        relabelled = RuntimeBundle(
            identity=production_identity,
            router=self.runtime.router,
            generator=self.runtime.generator,
            retriever=self.runtime.retriever,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "production runtime cannot use the hermetic router",
        ):
            verify_component_manifests(relabelled, fixture_mode=False)

    def test_request_cannot_inject_gold_or_attestation_fields(self) -> None:
        request = self._request("simple")
        request["payload"]["gold"] = "forbidden"
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "EVALUATION_LEAKAGE")
        self.assertEqual(self.runtime.router.calls, 0)

    def test_host_recomputes_event_hash_instead_of_trusting_worker(self) -> None:
        class TamperingClient(_DriverClient):
            def call_controlled(self, request: MethodWorkerRequest) -> dict[str, Any]:
                response = super().call_controlled(request)
                response["payload"]["execution_evidence"]["events"][0]["result"]["route_logits"][
                    "A"
                ] = 999.0
                return response

        with self.assertRaisesRegex(AlgorithmOutputError, "event-log hash|event hash"):
            self._method(allow_fixture=True).run(
                FrozenAgentState(query="simple"),
                InterventionContext(external_runtime=TamperingClient(self.driver)),
            )

    def test_document_metadata_is_strictly_allowlisted_before_generation(self) -> None:
        aliases = (
            "answer_key",
            "reference_answer",
            "expected_answer",
            "correct_option",
            "nested_oracle_alias",
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                runtime = build_fixture_runtime(
                    identity=self.identity,
                    logits_by_question={"single": {"A": 0.0, "B": 2.0, "C": 0.0}},
                    documents_by_query={
                        "single": (
                            RetrievedDocument(
                                "leak",
                                "innocent text",
                                metadata={alias: {"value": "TOP-SECRET"}},
                            ),
                        )
                    },
                )
                driver = AdaptiveRAGWorkerDriver(
                    source_root=self.source,
                    router_weights_path=self.weights,
                    prompt_root=self.prompts,
                    runtime=runtime,
                    manifest_path=self.manifest,
                    fixture_mode=True,
                )

                response = driver.handle(self._request("single"))

                self.assertEqual(response["failure_code"], "EVALUATION_LEAKAGE")
                self.assertNotIn("TOP-SECRET", json.dumps(response, sort_keys=True))
                self.assertEqual(runtime.generator.operations, [])

    def test_failed_selected_arm_returns_partial_usage_for_runner_reconciliation(self) -> None:
        class FailFinalAnswer:
            def __init__(self, inner: Any) -> None:
                self.inner = inner
                self.calls: list[str] = []

            def component_manifest(self) -> Mapping[str, Any]:
                manifest = dict(self.inner.component_manifest())
                manifest["implementation_class"] = (
                    f"{type(self).__module__}.{type(self).__qualname__}"
                )
                return manifest

            def estimate_input_tokens(self, **kwargs: Any) -> int:
                return int(self.inner.estimate_input_tokens(**kwargs))

            def generate(self, **kwargs: Any) -> Any:
                operation = str(kwargs["operation"])
                self.calls.append(operation)
                if operation == "answer.ircot":
                    raise RuntimeError("forced final-answer failure")
                return self.inner.generate(**kwargs)

        failing_generator = FailFinalAnswer(self.runtime.generator)
        runtime = RuntimeBundle(
            identity=self.identity,
            router=self.runtime.router,
            generator=failing_generator,
            retriever=self.runtime.retriever,
        )
        driver = AdaptiveRAGWorkerDriver(
            source_root=self.source,
            router_weights_path=self.weights,
            prompt_root=self.prompts,
            runtime=runtime,
            manifest_path=self.manifest,
            fixture_mode=True,
        )
        direct = driver.handle(self._request("multi"))
        self.assertEqual(direct["failure_code"], "MODEL_FAILED")
        self.assertEqual(direct["usage"]["model_calls"], 3)
        self.assertEqual(direct["usage"]["retrieval_calls"], 2)
        self.assertEqual(len(direct["partial_events"]), 4)

        runner = CompositeInterventionRunner()
        with self.assertRaises(ExternalDependencyError):
            runner.run(
                self._method(allow_fixture=True),
                FrozenAgentState(query="multi"),
                InterventionContext(external_runtime=_DriverClient(driver), seed=3),
            )
        observed = runner.monitor.list_snapshots()[0].usage
        self.assertEqual(observed.model_calls, 3)
        self.assertEqual(observed.retrieval_calls, 2)

    @unittest.skipUnless(
        os.environ.get("AGENTINPUTLAB_ADAPTIVE_RAG_CHECKOUT"),
        "real Adaptive-RAG checkout smoke is opt-in",
    )
    def test_real_pinned_checkout_source_closure_only(self) -> None:
        # This smoke proves only Git/source closure.  Deliberately inert
        # placeholders make it impossible to mistake the test for model,
        # generator, retriever, corpus, or index execution.
        source_only_component = object()
        production = AdaptiveRAGWorkerDriver(
            source_root=Path(os.environ["AGENTINPUTLAB_ADAPTIVE_RAG_CHECKOUT"]),
            router_weights_path=self.weights,
            prompt_root=self.prompts,
            runtime=RuntimeBundle(
                identity=RuntimeIdentity(
                    router_revision=self.identity.router_revision,
                    router_snapshot_sha256=self.identity.router_snapshot_sha256,
                    router_weights_sha256=self.identity.router_weights_sha256,
                    router_tokenizer_sha256=self.identity.router_tokenizer_sha256,
                    base_model_revision=self.identity.base_model_revision,
                    generator_artifact_sha256=self.identity.generator_artifact_sha256,
                    generator_config_sha256=self.identity.generator_config_sha256,
                    retriever_revision=self.identity.retriever_revision,
                    retriever_implementation_sha256=(self.identity.retriever_implementation_sha256),
                    retriever_config_sha256=self.identity.retriever_config_sha256,
                    corpus_revision=self.identity.corpus_revision,
                    corpus_manifest_sha256=self.identity.corpus_manifest_sha256,
                    index_revision=self.identity.index_revision,
                    index_manifest_sha256=self.identity.index_manifest_sha256,
                    fixture_runtime=False,
                ),
                router=source_only_component,  # type: ignore[arg-type]
                generator=source_only_component,  # type: ignore[arg-type]
                retriever=source_only_component,  # type: ignore[arg-type]
            ),
            fixture_mode=False,
        )
        fingerprint = production.source_fingerprint()
        self.assertEqual(fingerprint["commit"], self.artifacts.upstream_commit)

    def _request(self, question: str) -> dict[str, Any]:
        artifacts = self.artifacts.as_mapping()
        request = MethodWorkerRequest(
            method_id="adaptive_rag",
            release_id="acquire.adaptive_rag.controlled_reimplementation",
            upstream_commit=self.artifacts.upstream_commit,
            state_hash=content_hash(question),
            payload={
                "task_id": f"fixture-{question}",
                "question": question,
                "artifacts": artifacts,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "controls": asdict(self.controls),
                "remaining_budget": {
                    "model_calls": 8,
                    "retrieval_calls": 8,
                    "output_tokens": 4096,
                    "context_tokens": None,
                },
                "seed": 3,
                "execution_contract": {
                    "input_view": "question_only",
                    "route_labels": ["A", "B", "C"],
                    "route_rule": "first_decoder_step_logit_argmax_A_first",
                    "strategy_map": {
                        "A": "no_retrieval",
                        "B": "single_retrieval",
                        "C": "ircot",
                    },
                    "evaluation_label_access": "none",
                    "selected_arm_only": True,
                    "fallback_allowed": False,
                    "arm_execution_semantics": "controlled_reimplementation",
                    "upstream_router_parity": True,
                    "supervisor_event_schema": ("agentinputlab.adaptive-rag.supervisor-event.v1"),
                },
            },
            dependency_fingerprints={"worker": {"fixture": True}},
            minimum_reservation={"model_calls": 2},
            protocol_version=PROTOCOL_VERSION,
        )
        return request.as_dict()


if __name__ == "__main__":
    unittest.main()
