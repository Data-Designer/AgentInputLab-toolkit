from __future__ import annotations

import hashlib
import json
import os
import tempfile
import types
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agentinputlab.algorithms.acquire.r2ag import (
    R2AGArtifactIdentity,
    R2AGControlledWorker,
    R2AGControls,
    R2AGFeatureSource,
)
from agentinputlab.contracts import ContextBlock, FrozenAgentState, InterventionContext
from workers.r2ag.driver import (
    FEATURE_NAMES,
    PROTOCOL,
    RELEASE,
    UPSTREAM_COMMIT,
    ConditionedReader,
    DriverError,
    GenerationResult,
    PreparedReader,
    R2AGDriver,
    TensorStage,
    serve_jsonl,
)
from workers.r2ag.runtime import (
    OfficialR2AGRuntime,
    _decode_complete_generation,
    _preflight_reader_configuration,
    _verify_runtime_versions,
)

_RUNTIME_DEPENDENCIES = {
    "torch": "2.1.2",
    "transformers": "4.40.2",
    "sentence-transformers": "2.6.1",
}


def content_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_closure_manifest(
    root: Path,
    *,
    repository: str,
    revision: str,
    files: tuple[str, ...],
) -> str:
    manifest = {
        "schema": "agentinputlab.artifact-closure.v1",
        "repository": repository,
        "revision": revision,
        "files": {
            relative: {
                "size": (root / relative).stat().st_size,
                "sha256": file_hash(root / relative),
            }
            for relative in files
        },
    }
    path = root / "agentinputlab-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return file_hash(path)


def refresh_request_id(request: dict[str, Any]) -> None:
    request.pop("request_id", None)
    request["request_id"] = content_hash(request)


class FixtureRuntime:
    def __init__(self, artifacts: dict[str, Any]) -> None:
        self.artifacts = dict(artifacts)
        self.calls: list[str] = []
        self.generation_result = GenerationResult("fixture answer", 2)

    def verify_artifacts(self, _artifacts):
        self.calls.append("verify")
        return {
            "checkpoint_num_documents": 3,
            "checkpoint_config_sha256": self.artifacts["checkpoint_config_sha256"],
            "rformer_weights_sha256": self.artifacts["rformer_weights_sha256"],
            "projection_weights_sha256": self.artifacts["projection_weights_sha256"],
        }

    def compute_features(self, query, candidates, retriever):
        self.calls.append("features")
        assert query and len(candidates) == 3 and retriever["revision"] == "retriever-v1"
        return (
            {"rerank": 0.9, "neighbor": 0.2, "precedent": 1.0},
            {"rerank": 0.7, "neighbor": 0.3, "precedent": 0.4},
            {"rerank": 0.1, "neighbor": 0.1, "precedent": 0.2},
        )

    def prepare_reader(self, reader_input, num_k):
        self.calls.append("prepare")
        assert reader_input and num_k == 3
        return PreparedReader("tokens", (5, 10, 15), 20)

    def run_rformer(self, features):
        self.calls.append("rformer")
        assert len(features) == 3
        return TensorStage("rformer", "a" * 64)

    def project(self, value):
        self.calls.append("projection")
        assert value == "rformer"
        return TensorStage("projection", "b" * 64)

    def replace_markers(self, prepared, projected):
        self.calls.append("replace")
        assert prepared.value == "tokens" and projected == "projection"
        return ConditionedReader("conditioned", prepared.marker_positions, True)

    def generate(self, conditioned, *, max_new_tokens, seed):
        self.calls.append("generate")
        assert conditioned == "conditioned" and max_new_tokens == 8 and seed == 7
        return self.generation_result


class R2AGWorkerDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.checkpoint = self.root / "checkpoint"
        self.base = self.root / "base"
        self.tokenizer = self.root / "tokenizer"
        for path in (self.source, self.checkpoint, self.base, self.tokenizer):
            path.mkdir()
        (self.source / "model.py").write_text("class RFormer: pass\n")
        required = {"model.py": file_hash(self.source / "model.py")}
        self.manifest = self.root / "compatibility.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": "fixture",
                    "source": {
                        "repository": "fixture/RRAG",
                        "commit": UPSTREAM_COMMIT,
                        "tree": "fixture-tree",
                        "required_files": required,
                    },
                    "runtime": {"dependencies": _RUNTIME_DEPENDENCIES},
                }
            )
        )
        (self.checkpoint / "config.json").write_text('{"num_k":3}\n')
        (self.base / "model.bin").write_bytes(b"fixture-reader-weights")
        self.base_manifest_sha256 = write_closure_manifest(
            self.base,
            repository="fixture/base",
            revision="base-v1",
            files=("model.bin",),
        )
        (self.tokenizer / "tokenizer.json").write_text(
            '{"model":"fixture"}\n',
            encoding="utf-8",
        )
        self.tokenizer_manifest_sha256 = write_closure_manifest(
            self.tokenizer,
            repository="fixture/tokenizer",
            revision="tokenizer-v1",
            files=("tokenizer.json",),
        )
        self.feature_embedder = self.checkpoint / "feature-embedder"
        self.feature_embedder.mkdir()
        (self.feature_embedder / "encoder.bin").write_bytes(b"fixture-encoder")
        self.feature_embedder_manifest_sha256 = write_closure_manifest(
            self.feature_embedder,
            repository="fixture/embedder",
            revision="retriever-v1",
            files=("encoder.bin",),
        )
        self.schema = self.root / "feature-schema.json"
        self.schema.write_text('{"features":["rerank","neighbor","precedent"]}\n')
        self.artifacts = {
            "checkpoint_repository": "fixture/checkpoint",
            "checkpoint_revision": "checkpoint-v1",
            "checkpoint_config_sha256": file_hash(self.checkpoint / "config.json"),
            "rformer_weights_sha256": "2" * 64,
            "projection_weights_sha256": "3" * 64,
            "checkpoint_num_documents": 3,
            "base_model_repository": "fixture/base",
            "base_model_revision": "base-v1",
            "base_model_manifest_sha256": self.base_manifest_sha256,
            "tokenizer_repository": "fixture/tokenizer",
            "tokenizer_revision": "tokenizer-v1",
            "tokenizer_manifest_sha256": self.tokenizer_manifest_sha256,
            "feature_schema_revision": "features-v1",
            "feature_schema_sha256": file_hash(self.schema),
            "source_manifest_sha256": content_hash(required),
            "compatibility_manifest_sha256": content_hash(json.loads(self.manifest.read_text())),
            "upstream_commit": UPSTREAM_COMMIT,
        }
        self.runtime = FixtureRuntime(self.artifacts)
        self.driver = R2AGDriver(
            source_root=self.source,
            checkpoint_root=self.checkpoint,
            base_model_root=self.base,
            tokenizer_root=self.tokenizer,
            manifest_path=self.manifest,
            feature_schema_path=self.schema,
            runtime_factory=lambda *_args: self.runtime,
            fixture_mode=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_driver_rejects_symlinked_deployment_roots(self) -> None:
        linked_source = self.root / "linked-source"
        linked_source.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(DriverError, "must not be a symlink"):
            R2AGDriver(
                source_root=linked_source,
                checkpoint_root=self.checkpoint,
                base_model_root=self.base,
                tokenizer_root=self.tokenizer,
                manifest_path=self.manifest,
                runtime_factory=lambda *_: self.runtime,
                fixture_mode=True,
            )

    def request(self, *, feature_source: str = "embedder") -> dict[str, Any]:
        candidates = (
            {
                "document_id": "d1",
                "text": "first",
                "title": "one",
                "source": "fixture",
                "score": 0.9,
                "corpus_revision": "corpus-v1",
            },
            {
                "document_id": "d2",
                "text": "second",
                "title": "two",
                "source": "fixture",
                "score": 0.7,
                "corpus_revision": "corpus-v1",
            },
            {
                "document_id": "d3",
                "text": "third",
                "title": "three",
                "source": "fixture",
                "score": 0.1,
                "corpus_revision": "corpus-v1",
            },
        )
        features = (
            {"rerank": 0.9, "neighbor": 0.2, "precedent": 1.0},
            {"rerank": 0.7, "neighbor": 0.3, "precedent": 0.4},
            {"rerank": 0.1, "neighbor": 0.1, "precedent": 0.2},
        )
        feature_tuple = tuple(tuple(row[name] for name in FEATURE_NAMES) for row in features)
        pool = {
            "retriever": {
                "repository": "fixture/embedder",
                "revision": "retriever-v1",
                "manifest_sha256": self.feature_embedder_manifest_sha256,
            },
            "corpus": {"repository": "fixture/corpus", "revision": "corpus-v1"},
            "index": {"algorithm": "flat", "revision": "index-v1"},
            "top_k": 3,
            "order": "rerank_desc",
            "content_sha256": content_hash(candidates),
        }
        reader_instruction = "documents <R> <R> <R>"
        reader_input = "documents <unk> <unk> <unk>"
        feature_embedder = (
            {
                "kind": "dense_embedder",
                "fingerprint_sha256": "4" * 64,
            }
            if feature_source == "embedder"
            else None
        )
        feature_artifact = (
            {
                "repository": "fixture/features",
                "revision": "features-v1",
                "sha256": "5" * 64,
            }
            if feature_source == "candidate_metadata"
            else None
        )
        payload = {
            "task_id": "fixture",
            "query": "question",
            "candidate_documents": candidates,
            "candidate_document_ids": ("d1", "d2", "d3"),
            "candidate_documents_sha256": content_hash(candidates),
            "candidate_order_sha256": content_hash(("d1", "d2", "d3")),
            "candidate_pool_fingerprint": pool,
            "candidate_pool_fingerprint_sha256": content_hash(pool),
            "artifacts": self.artifacts,
            "artifact_identity_sha256": content_hash(self.artifacts),
            "features": None if feature_source == "worker" else features,
            "feature_tensor_sha256": (
                "" if feature_source == "worker" else content_hash(feature_tuple)
            ),
            "feature_source": feature_source,
            "feature_names": FEATURE_NAMES,
            "feature_embedder_fingerprint": feature_embedder,
            "feature_embedder_fingerprint_sha256": content_hash(feature_embedder),
            "feature_artifact_fingerprint": feature_artifact,
            "feature_artifact_fingerprint_sha256": content_hash(feature_artifact),
            "reader_instruction": reader_instruction,
            "reader_instruction_sha256": content_hash(reader_instruction),
            "reader_input": reader_input,
            "reader_input_sha256": content_hash(reader_input),
            "controls": {
                "max_documents": 30,
                "max_output_tokens": 8,
                "feature_source": feature_source,
                "checkpoint_num_documents": 3,
                "require_rerank_order": True,
                "numeric_tolerance": 1e-6,
                "max_reader_chars": 1000,
                "retrieval_token": "<R>",
                "unknown_token": "<unk>",
            },
            "remaining_budget": {
                "model_calls": 1,
                "output_tokens": 8,
                "context_tokens": 64,
            },
            "seed": 7,
            "execution_contract": {
                "allow_training": False,
                "allow_gold_metadata": False,
                "require_source_verification": True,
                "require_checkpoint_hash_verification": True,
                "require_rformer_then_projection": True,
                "replace_only_retrieval_marker_embeddings": True,
                "check_budget_before_reader_call": True,
            },
        }
        dependencies = {
            "candidate_pool": pool,
            "worker": self.driver.fingerprint(),
            "artifacts": self.artifacts,
        }
        if feature_source == "embedder":
            dependencies["feature_embedder"] = feature_embedder
        elif feature_source == "candidate_metadata":
            dependencies["feature_artifact"] = feature_artifact
        request = {
            "method_id": "r2ag",
            "release_id": RELEASE,
            "upstream_commit": UPSTREAM_COMMIT,
            "state_hash": "0" * 64,
            "payload": payload,
            "dependency_fingerprints": dependencies,
            "minimum_reservation": {"model_calls": 1},
            "protocol_version": PROTOCOL,
        }
        request["request_id"] = content_hash(request)
        return request

    def test_fixture_e2e_calls_every_stage_in_order(self) -> None:
        response = self.driver.handle(self.request())
        self.assertEqual(response["status"], "ok")
        self.assertEqual(
            self.runtime.calls,
            ["verify", "prepare", "rformer", "projection", "replace", "generate"],
        )
        self.assertEqual(response["payload"]["r_token_positions"], (5, 10, 15))
        self.assertEqual(response["usage"]["model_calls"], 1)
        self.assertEqual(response["usage"]["embedding_calls"], 0)
        self.assertTrue(response["payload"]["runtime_attestation"]["fixture_mode"])
        self.assertEqual(
            response["payload"]["runtime_attestation"]["runtime_dependencies"],
            _RUNTIME_DEPENDENCIES,
        )
        self.assertFalse(
            response["payload"]["runtime_attestation"]["runtime_dependencies_verified"]
        )
        self.assertEqual(
            self.driver.fingerprint()["runtime_dependencies"],
            _RUNTIME_DEPENDENCIES,
        )

    def test_worker_features_add_exactly_one_embedding_call(self) -> None:
        response = self.driver.handle(self.request(feature_source="worker"))
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["usage"]["embedding_calls"], 1)
        self.assertEqual(
            self.runtime.calls,
            [
                "verify",
                "features",
                "prepare",
                "rformer",
                "projection",
                "replace",
                "generate",
            ],
        )

    def test_oracle_metadata_and_num_k_drift_fail_before_runtime(self) -> None:
        request = self.request()
        request["payload"]["candidate_documents"][0]["gold_answer"] = "leak"
        refresh_request_id(request)
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "ORACLE_FIELD_REJECTED")
        self.assertEqual(self.runtime.calls, [])

        request = self.request()
        request["payload"]["controls"]["checkpoint_num_documents"] = 2
        refresh_request_id(request)
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "PROFILE_DRIFT")
        self.assertEqual(self.runtime.calls, [])

    def test_budget_and_hash_failures_happen_before_reader(self) -> None:
        request = self.request()
        request["payload"]["remaining_budget"]["model_calls"] = 0
        refresh_request_id(request)
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "MODEL_BUDGET_EXHAUSTED")
        self.assertEqual(self.runtime.calls, [])

        request = self.request()
        request["payload"]["artifacts"]["rformer_weights_sha256"] = "9" * 64
        request["payload"]["artifact_identity_sha256"] = content_hash(
            request["payload"]["artifacts"]
        )
        request["dependency_fingerprints"]["artifacts"] = request["payload"]["artifacts"]
        refresh_request_id(request)
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "CHECKPOINT_MISMATCH")
        self.assertEqual(self.runtime.calls, ["verify"])

    def test_invalid_fixture_generation_result_fails_closed(self) -> None:
        invalid_results = (
            GenerationResult("", 1),
            GenerationResult("answer", 0),
            GenerationResult("answer", 9),
        )
        for generated in invalid_results:
            with self.subTest(generated=generated):
                self.runtime.generation_result = generated
                response = self.driver.handle(self.request())
                self.assertEqual(response["failure_code"], "READER_FAILED")
                self.runtime.calls.clear()
        self.runtime.generation_result = GenerationResult("fixture answer", 2)

    def test_request_id_and_nested_fingerprint_leaks_fail_closed(self) -> None:
        request = self.request()
        request["payload"]["query"] = "mutated after hashing"
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "IDENTITY_MISMATCH")
        self.assertEqual(self.runtime.calls, [])

        request = self.request()
        pool = request["payload"]["candidate_pool_fingerprint"]
        pool["retriever"]["gold_answer"] = "leak"
        request["payload"]["candidate_pool_fingerprint_sha256"] = content_hash(pool)
        refresh_request_id(request)
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "ORACLE_FIELD_REJECTED")
        self.assertEqual(self.runtime.calls, [])

        request = self.request()
        pool = request["payload"]["candidate_pool_fingerprint"]
        pool["retriever"]["revision"] = "main"
        request["payload"]["candidate_pool_fingerprint_sha256"] = content_hash(pool)
        refresh_request_id(request)
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "IDENTITY_MISMATCH")
        self.assertEqual(self.runtime.calls, [])

        request = self.request(feature_source="candidate_metadata")
        artifact = request["payload"]["feature_artifact_fingerprint"]
        artifact["reward_label"] = 1
        request["payload"]["feature_artifact_fingerprint_sha256"] = content_hash(artifact)
        refresh_request_id(request)
        response = self.driver.handle(request)
        self.assertEqual(response["failure_code"], "ORACLE_FIELD_REJECTED")
        self.assertEqual(self.runtime.calls, [])

    def test_reader_tokenizer_and_embedder_closure_tampering_fails_before_runtime(self) -> None:
        cases = (
            (self.base / "model.bin", "embedder"),
            (self.tokenizer / "tokenizer.json", "embedder"),
            (self.feature_embedder / "encoder.bin", "worker"),
        )
        for path, feature_source in cases:
            with self.subTest(path=path.name, feature_source=feature_source):
                original = path.read_bytes()
                path.write_bytes(original + b"-tampered")
                try:
                    response = self.driver.handle(self.request(feature_source=feature_source))
                finally:
                    path.write_bytes(original)
                self.assertEqual(response["failure_code"], "ARTIFACT_MISMATCH")
                self.assertEqual(self.runtime.calls, [])

    def test_host_to_driver_roundtrip_persists_fixture_provenance(self) -> None:
        rows = (
            ("d1", "first", "one", 0.9, (0.9, 0.2, 1.0)),
            ("d2", "second", "two", 0.7, (0.7, 0.3, 0.4)),
            ("d3", "third", "three", 0.1, (0.1, 0.1, 0.2)),
        )
        state = FrozenAgentState(
            query="question",
            task_id="fixture-roundtrip",
            context=tuple(
                ContextBlock(
                    text,
                    block_id=document_id,
                    source="fixture",
                    score=score,
                    metadata={
                        "title": title,
                        "corpus_revision": "corpus-v1",
                        "rerank_score": features[0],
                        "rerank_nb_score": features[1],
                        "rerank_precedent_score": features[2],
                    },
                )
                for document_id, text, title, score, features in rows
            ),
        )
        candidates = tuple(
            {
                "document_id": block.block_id,
                "text": block.text,
                "title": block.metadata["title"],
                "source": block.source,
                "score": block.score,
                "corpus_revision": block.metadata["corpus_revision"],
            }
            for block in state.context
        )
        pool = {
            "retriever": {
                "repository": "fixture/embedder",
                "revision": "retriever-v1",
                "manifest_sha256": self.feature_embedder_manifest_sha256,
            },
            "corpus": {"repository": "fixture/corpus", "revision": "corpus-v1"},
            "index": {"algorithm": "flat", "revision": "index-v1"},
            "top_k": 3,
            "order": "rerank_desc",
            "content_sha256": content_hash(candidates),
        }
        result = R2AGControlledWorker(
            artifacts=R2AGArtifactIdentity(**self.artifacts),
            controls=R2AGControls(
                feature_source=R2AGFeatureSource.CANDIDATE_METADATA,
                checkpoint_num_documents=3,
                max_output_tokens=8,
            ),
            allow_fixture_worker=True,
        ).run(
            state,
            InterventionContext(
                external_runtime=self.driver,
                seed=7,
                run_config={
                    "candidate_pool_fingerprint": pool,
                    "r2ag_feature_artifact_fingerprint": {
                        "repository": "fixture/features",
                        "revision": "features-v1",
                        "sha256": "5" * 64,
                    },
                },
            ),
        )
        self.assertEqual(result.final_answer, "fixture answer")
        self.assertTrue(result.metadata["fixture_mode"])
        self.assertEqual(result.metadata["execution_evidence"], "fixture_source_execution")
        self.assertEqual(
            self.runtime.calls,
            ["verify", "prepare", "rformer", "projection", "replace", "generate"],
        )

    def test_jsonl_boundary_executes_one_valid_request(self) -> None:
        source = StringIO(json.dumps(self.request()) + "\n")
        sink = StringIO()
        self.assertEqual(serve_jsonl(self.driver, source, sink), 0)
        lines = sink.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["protocol_version"], PROTOCOL)

    def test_exact_runtime_distribution_versions_are_fail_closed(self) -> None:
        observed: list[str] = []

        def exact_version(distribution_name: str) -> str:
            observed.append(distribution_name)
            return _RUNTIME_DEPENDENCIES[distribution_name]

        with patch(
            "workers.r2ag.runtime.importlib_metadata.version",
            side_effect=exact_version,
        ):
            self.assertEqual(
                _verify_runtime_versions(_RUNTIME_DEPENDENCIES),
                _RUNTIME_DEPENDENCIES,
            )
        self.assertEqual(
            observed,
            ["torch", "transformers", "sentence-transformers"],
        )

        with (
            patch(
                "workers.r2ag.runtime.importlib_metadata.version",
                side_effect=lambda name: (
                    "4.41.0" if name == "transformers" else _RUNTIME_DEPENDENCIES[name]
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "exactly 4.40.2"),
        ):
            _verify_runtime_versions(_RUNTIME_DEPENDENCIES)

    def test_reader_preflight_preserves_pinned_auto_config_and_does_not_load_weights(self) -> None:
        (self.base / "config.json").write_text('{"model_type":"llama"}')
        module = types.ModuleType("transformers")
        config_type = type("LlamaConfig", (), {})
        calls = []

        def configure(path, **kwargs):
            calls.append((path, kwargs))
            return config_type()

        module.AutoConfig = SimpleNamespace(from_pretrained=configure)
        module.AutoModelForCausalLM = SimpleNamespace(_model_mapping={config_type: object})
        with (
            patch.dict("sys.modules", {"transformers": module}),
            patch(
                "workers.r2ag.runtime._verify_runtime_versions", return_value=_RUNTIME_DEPENDENCIES
            ),
            patch("workers.r2ag.runtime._import_torch") as torch,
        ):
            runtime = OfficialR2AGRuntime(
                source_root=self.source,
                checkpoint_root=self.checkpoint,
                base_model_root=self.base,
                tokenizer_root=self.tokenizer,
                expected_runtime_versions=_RUNTIME_DEPENDENCIES,
            )
        torch.assert_not_called()
        self.assertIsNone(runtime._base_model)
        self.assertIsNone(runtime._rformer_state)
        self.assertEqual(
            calls, [(str(self.base), {"local_files_only": True, "trust_remote_code": False})]
        )

    def test_reader_preflight_rejects_unknown_backbone_before_torch_or_checkpoint(self) -> None:
        module = types.ModuleType("transformers")
        module.AutoConfig = SimpleNamespace(
            from_pretrained=lambda *a, **k: (_ for _ in ()).throw(
                ValueError("unknown native model type")
            )
        )
        module.AutoModelForCausalLM = SimpleNamespace(_model_mapping={})
        for model_type in ("qwen3", "gpt_oss", "gemma4_unified", "qwen3_5"):
            with self.subTest(model_type=model_type):
                (self.base / "config.json").write_text(json.dumps({"model_type": model_type}))
                with (
                    patch.dict("sys.modules", {"transformers": module}),
                    patch(
                        "workers.r2ag.runtime._verify_runtime_versions",
                        return_value=_RUNTIME_DEPENDENCIES,
                    ),
                    patch("workers.r2ag.runtime._import_torch") as torch,
                    self.assertRaisesRegex(
                        RuntimeError, "transformers==4.40.2.*" + model_type
                    ) as caught,
                ):
                    OfficialR2AGRuntime(
                        source_root=self.source,
                        checkpoint_root=self.checkpoint,
                        base_model_root=self.base,
                        tokenizer_root=self.tokenizer,
                        expected_runtime_versions=_RUNTIME_DEPENDENCIES,
                    )
                torch.assert_not_called()
                self.assertIsInstance(caught.exception.__cause__, ValueError)
                self.assertEqual(caught.exception.code, "READER_CONFIG_INCOMPATIBLE")
                self.assertFalse(caught.exception.consumed)
                self.assertIn("no model weights were loaded", str(caught.exception))

    def test_reader_preflight_does_not_substitute_conditional_wrapper_or_follow_symlinks(
        self,
    ) -> None:
        (self.base / "config.json").write_text('{"model_type":"conditional_wrapper"}')
        module = types.ModuleType("transformers")
        module.AutoConfig = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace())
        module.AutoModelForCausalLM = SimpleNamespace(_model_mapping={})
        with (
            patch.dict("sys.modules", {"transformers": module}),
            self.assertRaisesRegex(RuntimeError, "wrappers are not silently unwrapped"),
        ):
            _preflight_reader_configuration(self.base, transformers_version="4.40.2")
        link = self.root / "reader-link"
        link.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "regular directory"):
            _preflight_reader_configuration(link, transformers_version="4.40.2")
        (self.base / "config.json").unlink()
        (self.base / "config.json").symlink_to(self.checkpoint / "config.json")
        with self.assertRaisesRegex(ValueError, "regular file"):
            _preflight_reader_configuration(self.base, transformers_version="4.40.2")

    def test_complete_generation_tensor_boundaries_and_decode(self) -> None:
        class Tokenizer:
            def __init__(self, decoded):
                self.decoded = decoded
                self.seen = None

            def batch_decode(self, outputs, *, skip_special_tokens):
                self.seen = outputs
                self.assert_skip = skip_special_tokens
                return self.decoded

        valid = SimpleNamespace(ndim=2, shape=(1, 3))
        tokenizer = Tokenizer([" complete answer "])
        result = _decode_complete_generation(
            valid,
            tokenizer=tokenizer,
            max_new_tokens=3,
        )
        self.assertIs(tokenizer.seen, valid)
        self.assertTrue(tokenizer.assert_skip)
        self.assertEqual(result, GenerationResult("complete answer", 3))

        invalid = (
            (SimpleNamespace(ndim=1, shape=(3,)), ["answer"], "rank-two"),
            (SimpleNamespace(ndim=2, shape=(2, 3)), ["a", "b"], "batch size"),
            (SimpleNamespace(ndim=2, shape=(1, 0)), ["answer"], "output length"),
            (SimpleNamespace(ndim=2, shape=(1, 4)), ["answer"], "output length"),
            (SimpleNamespace(ndim=2, shape=(1, 1)), [], "non-empty"),
            (SimpleNamespace(ndim=2, shape=(1, 1)), ["   "], "non-empty"),
            (SimpleNamespace(ndim=2, shape=(1, 1)), ["a", "b"], "non-empty"),
        )
        for outputs, decoded, message in invalid:
            with (
                self.subTest(outputs=outputs, decoded=decoded),
                self.assertRaisesRegex(ValueError, message),
            ):
                _decode_complete_generation(
                    outputs,
                    tokenizer=Tokenizer(decoded),
                    max_new_tokens=3,
                )

    @unittest.skipUnless(
        os.getenv("AGENTINPUTLAB_RUN_REAL_R2AG_SOURCE_SMOKE") == "1",
        "set AGENTINPUTLAB_RUN_REAL_R2AG_SOURCE_SMOKE=1 for pinned checkout verification",
    )
    def test_opt_in_real_checkout_source_hashes_commit_tree_and_clean_state(self) -> None:
        checkout = Path(os.environ["AGENTINPUTLAB_R2AG_CHECKOUT"])
        driver = R2AGDriver(
            source_root=checkout,
            checkpoint_root=self.checkpoint,
            base_model_root=self.base,
            tokenizer_root=self.tokenizer,
        )
        attestation = driver.verify_source_only()
        self.assertEqual(attestation["kind"], "clean_git_checkout")
        self.assertEqual(attestation["tree"], "f1757a51fc1c3d64fe9d1b8788178293c8d5ef3b")


if __name__ == "__main__":
    unittest.main()
