#!/usr/bin/env python3
"""Fail-closed R²AG R²-Former/projection/reader sidecar.

The worker never downloads or redistributes source or weights. Production uses
the bundled runtime with read-only user mounts; the injectable runtime seam is
available only when ``fixture_mode=True`` so tests cannot self-attest as
production execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

UPSTREAM_COMMIT = "61cfd1027c0101172fda701ec816bc18360b5bab"
UPSTREAM_TREE = "f1757a51fc1c3d64fe9d1b8788178293c8d5ef3b"
PROTOCOL = "agentinputlab.r2ag.v4"
RELEASE = "acquire.r2ag.controlled_same_candidates"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
DEFAULT_FEATURE_SCHEMA = Path(__file__).with_name("feature_schema.json")
FEATURE_NAMES = ("rerank", "neighbor", "precedent")
RUNTIME_DEPENDENCIES = {
    "torch": "2.1.2",
    "transformers": "4.40.2",
    "sentence-transformers": "2.6.1",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_ALIASES = frozenset({"default", "head", "latest", "main", "master"})
_ARTIFACT_MANIFEST_SCHEMA = "agentinputlab.artifact-closure.v1"
_ARTIFACT_MANIFEST_FIELDS = frozenset({"schema", "repository", "revision", "files"})
_ARTIFACT_FILE_FIELDS = frozenset({"size", "sha256"})
_FORBIDDEN_FINGERPRINT_TERMS = frozenset(
    {
        "answer",
        "api",
        "credential",
        "expected",
        "gold",
        "groundtruth",
        "label",
        "oracle",
        "password",
        "reference",
        "reward",
        "secret",
        "solution",
        "supporting",
        "target",
    }
)
_REPOSITORY_IDENTITY_REQUIRED = frozenset({"repository", "revision"})
_REPOSITORY_IDENTITY_OPTIONAL = frozenset({"manifest_sha256"})
_INDEX_IDENTITY_REQUIRED = frozenset({"algorithm", "revision"})
_INDEX_IDENTITY_OPTIONAL = frozenset({"manifest_sha256"})
_FEATURE_ARTIFACT_FIELDS = frozenset({"repository", "revision", "sha256"})
_FEATURE_EMBEDDER_FIELDS = frozenset({"kind", "fingerprint_sha256"})
_ARTIFACT_FIELDS = frozenset(
    {
        "checkpoint_repository",
        "checkpoint_revision",
        "checkpoint_config_sha256",
        "rformer_weights_sha256",
        "projection_weights_sha256",
        "checkpoint_num_documents",
        "base_model_repository",
        "base_model_revision",
        "base_model_manifest_sha256",
        "tokenizer_repository",
        "tokenizer_revision",
        "tokenizer_manifest_sha256",
        "feature_schema_revision",
        "feature_schema_sha256",
        "source_manifest_sha256",
        "compatibility_manifest_sha256",
        "upstream_commit",
    }
)
_REQUEST_ID_FIELDS = (
    "method_id",
    "release_id",
    "upstream_commit",
    "state_hash",
    "payload",
    "dependency_fingerprints",
    "minimum_reservation",
    "protocol_version",
)
_REQUEST_FIELDS = frozenset((*_REQUEST_ID_FIELDS, "request_id"))
_CANDIDATE_FIELDS = frozenset(
    {"document_id", "text", "title", "source", "score", "corpus_revision"}
)
_POOL_FIELDS = frozenset({"retriever", "corpus", "index", "top_k", "order", "content_sha256"})
_PAYLOAD_FIELDS = frozenset(
    {
        "task_id",
        "query",
        "candidate_documents",
        "candidate_document_ids",
        "candidate_documents_sha256",
        "candidate_order_sha256",
        "candidate_pool_fingerprint",
        "candidate_pool_fingerprint_sha256",
        "artifacts",
        "artifact_identity_sha256",
        "features",
        "feature_tensor_sha256",
        "feature_source",
        "feature_names",
        "feature_embedder_fingerprint",
        "feature_embedder_fingerprint_sha256",
        "feature_artifact_fingerprint",
        "feature_artifact_fingerprint_sha256",
        "reader_instruction",
        "reader_instruction_sha256",
        "reader_input",
        "reader_input_sha256",
        "controls",
        "remaining_budget",
        "seed",
        "execution_contract",
    }
)


class DriverError(RuntimeError):
    """Stable, secret-free worker failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed
        self.usage = dict(usage or {})


@dataclass(frozen=True, slots=True)
class PreparedReader:
    value: Any
    marker_positions: tuple[int, ...]
    input_tokens: int


@dataclass(frozen=True, slots=True)
class TensorStage:
    value: Any
    sha256: str


@dataclass(frozen=True, slots=True)
class ConditionedReader:
    value: Any
    marker_positions: tuple[int, ...]
    only_marker_positions_modified: bool


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str
    output_tokens: int
    accelerator_seconds: float = 0.0


class R2AGDriver:
    """Validate identity and budgets, then execute the five ordered stages."""

    worker_id = "agentinputlab.r2ag.pinned.v4"

    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint_root: Path,
        base_model_root: Path,
        tokenizer_root: Path,
        manifest_path: Path = DEFAULT_MANIFEST,
        feature_schema_path: Path = DEFAULT_FEATURE_SCHEMA,
        runtime_factory: Callable[[Path, Path, Path, Path], Any] | None = None,
        fixture_mode: bool = False,
    ) -> None:
        if runtime_factory is not None and not fixture_mode:
            raise ValueError("custom R²AG runtimes are fixture-only")
        for label, path in (
            ("source root", source_root),
            ("checkpoint root", checkpoint_root),
            ("base-model root", base_model_root),
            ("tokenizer root", tokenizer_root),
            ("compatibility manifest", manifest_path),
            ("feature schema", feature_schema_path),
        ):
            if path.is_symlink():
                raise DriverError("PATH_REJECTED", f"R²AG {label} must not be a symlink")
        self.source_root = source_root.resolve()
        self.checkpoint_root = checkpoint_root.resolve()
        self.base_model_root = base_model_root.resolve()
        self.tokenizer_root = tokenizer_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.feature_schema_path = feature_schema_path.resolve()
        self.runtime_factory = runtime_factory
        self.fixture_mode = fixture_mode
        self.manifest = _load_json(self.manifest_path, label="compatibility manifest")
        self.compatibility_manifest_sha256 = _content_hash(self.manifest)
        runtime_manifest = _mapping(self.manifest, "runtime")
        dependencies = _mapping(runtime_manifest, "dependencies")
        if dict(dependencies) != RUNTIME_DEPENDENCIES:
            raise DriverError(
                "MANIFEST_INVALID",
                "R²AG runtime dependency versions drifted",
            )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.worker_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol": PROTOCOL,
            "compatibility_manifest_sha256": self.compatibility_manifest_sha256,
            "runtime_dependencies": dict(RUNTIME_DEPENDENCIES),
            "fixture_mode": self.fixture_mode,
        }

    def call(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Implement the standard in-process ``WorkerClient`` boundary."""

        return self.handle(request)

    def call_controlled(self, request: Any) -> Mapping[str, Any]:
        """Accept a typed host request without weakening envelope validation."""

        serializer = getattr(request, "as_dict", None)
        if not callable(serializer):
            return self.handle({})
        value = serializer()
        if not isinstance(value, Mapping):
            return self.handle({})
        return self.handle(value)

    def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self._handle(request)
        except DriverError as exc:
            response: dict[str, Any] = {
                "status": "error",
                "failure_code": exc.code,
                "message": exc.public_message,
                "consumed": exc.consumed,
            }
            if exc.usage:
                response["usage"] = exc.usage
            return response

    def verify_source_only(self) -> Mapping[str, Any]:
        return self._verify_source()

    def _handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = _validate_envelope(request)
        if set(payload) != _PAYLOAD_FIELDS:
            raise DriverError("REQUEST_INVALID", "R²AG payload fields drifted")
        dependencies = _mapping(request, "dependency_fingerprints")
        contract = _mapping(payload, "execution_contract")
        _validate_contract(contract)
        controls = _mapping(payload, "controls")
        remaining = _mapping(payload, "remaining_budget")
        artifacts = _mapping(payload, "artifacts")
        _validate_artifact_identity(artifacts)
        if payload.get("artifact_identity_sha256") != _content_hash(artifacts):
            raise DriverError("IDENTITY_MISMATCH", "artifact identity hash mismatch")
        if artifacts.get("upstream_commit") != UPSTREAM_COMMIT:
            raise DriverError("IDENTITY_MISMATCH", "artifact source commit mismatch")
        if artifacts.get("compatibility_manifest_sha256") != (self.compatibility_manifest_sha256):
            raise DriverError("IDENTITY_MISMATCH", "compatibility manifest mismatch")

        candidates = _candidate_documents(payload.get("candidate_documents"))
        candidate_ids = _string_sequence(
            payload.get("candidate_document_ids"), label="candidate_document_ids"
        )
        if candidate_ids != tuple(row["document_id"] for row in candidates):
            raise DriverError("IDENTITY_MISMATCH", "candidate ID/order mismatch")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DriverError("REQUEST_INVALID", "candidate IDs must be unique")
        if payload.get("candidate_documents_sha256") != _content_hash(candidates):
            raise DriverError("IDENTITY_MISMATCH", "candidate document hash mismatch")
        if payload.get("candidate_order_sha256") != _content_hash(candidate_ids):
            raise DriverError("IDENTITY_MISMATCH", "candidate order hash mismatch")
        pool = _validate_pool(payload, candidates)
        feature_source = payload.get("feature_source")
        feature_embedder_identity, feature_artifact_identity = _validate_feature_identities(
            payload,
            feature_source=feature_source,
        )
        _validate_dependency_fingerprints(
            dependencies,
            pool=pool,
            artifacts=artifacts,
            feature_source=feature_source,
            feature_embedder_identity=feature_embedder_identity,
            feature_artifact_identity=feature_artifact_identity,
        )

        num_k = _positive_int(artifacts, "checkpoint_num_documents")
        if _positive_int(controls, "checkpoint_num_documents") != num_k:
            raise DriverError("PROFILE_DRIFT", "controls/checkpoint num_k mismatch")
        if len(candidates) != num_k:
            raise DriverError("PROFILE_DRIFT", "candidate count must equal config.num_k")
        if num_k > 30 or num_k > _positive_int(controls, "max_documents"):
            raise DriverError("INPUT_TOO_LARGE", "R²AG num_k exceeds its controlled limit")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise DriverError("PROFILE_DRIFT", "retrieval feature schema drifted")
        if controls.get("retrieval_token") != "<R>" or controls.get("unknown_token") != "<unk>":
            raise DriverError("PROFILE_DRIFT", "R²AG marker controls drifted")

        if _nonnegative_int(remaining, "model_calls") < 1:
            raise DriverError("MODEL_BUDGET_EXHAUSTED", "reader model budget exhausted")
        output_remaining = _nonnegative_int(remaining, "output_tokens")
        if output_remaining < 1:
            raise DriverError("OUTPUT_BUDGET_EXHAUSTED", "reader output budget exhausted")
        max_new_tokens = min(
            output_remaining,
            _positive_int(controls, "max_output_tokens"),
        )

        source_attestation = self._verify_source()
        self._verify_static_artifacts(artifacts)
        if feature_source == "worker":
            self._verify_feature_embedder(pool)
        try:
            runtime = self._runtime()
            verified = runtime.verify_artifacts(artifacts)
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "CHECKPOINT_INVALID",
                "R²AG runtime artifact verification failed",
            ) from exc
        if not isinstance(verified, Mapping):
            raise DriverError("CHECKPOINT_INVALID", "runtime artifact verification failed")
        expected_verified = {
            "checkpoint_num_documents": num_k,
            "checkpoint_config_sha256": artifacts["checkpoint_config_sha256"],
            "rformer_weights_sha256": artifacts["rformer_weights_sha256"],
            "projection_weights_sha256": artifacts["projection_weights_sha256"],
        }
        if any(verified.get(key) != value for key, value in expected_verified.items()):
            raise DriverError("CHECKPOINT_MISMATCH", "runtime component identity mismatch")

        embedding_calls = 0
        if feature_source == "worker":
            try:
                raw_features = runtime.compute_features(
                    _text(payload, "query"),
                    candidates,
                    _mapping(pool, "retriever"),
                )
            except Exception as exc:
                raise DriverError(
                    "FEATURE_EXTRACTION_FAILED",
                    "R²AG feature extraction failed",
                    consumed=True,
                ) from exc
            embedding_calls = 1
        elif feature_source in {"embedder", "candidate_metadata"}:
            raw_features = payload.get("features")
        else:
            raise DriverError("PROFILE_DRIFT", "unsupported R²AG feature source")
        features = _features(raw_features, expected_count=num_k)
        feature_hash = _content_hash(
            tuple(tuple(row[name] for name in FEATURE_NAMES) for row in features)
        )
        sent_feature_hash = payload.get("feature_tensor_sha256")
        if feature_source != "worker" and sent_feature_hash != feature_hash:
            raise DriverError("IDENTITY_MISMATCH", "feature tensor hash mismatch")
        _validate_descending(features)

        reader_input = _text(payload, "reader_input")
        if payload.get("reader_input_sha256") != _content_hash(reader_input):
            raise DriverError("IDENTITY_MISMATCH", "reader input hash mismatch")
        reader_instruction = _text(payload, "reader_instruction")
        if payload.get("reader_instruction_sha256") != _content_hash(reader_instruction):
            raise DriverError("IDENTITY_MISMATCH", "reader instruction hash mismatch")

        try:
            prepared: PreparedReader = runtime.prepare_reader(reader_input, num_k)
        except Exception as exc:
            raise DriverError("TOKENIZATION_FAILED", "R²AG reader tokenization failed") from exc
        _validate_prepared(prepared, num_k)
        context_limit = remaining.get("context_tokens")
        if context_limit is not None and prepared.input_tokens > _nonnegative_int(
            remaining, "context_tokens"
        ):
            raise DriverError("CONTEXT_BUDGET_EXHAUSTED", "reader context budget exhausted")

        try:
            rformer: TensorStage = runtime.run_rformer(features)
            _validate_tensor_stage(rformer, label="R²-Former")
            projection: TensorStage = runtime.project(rformer.value)
            _validate_tensor_stage(projection, label="projection")
            conditioned: ConditionedReader = runtime.replace_markers(
                prepared,
                projection.value,
            )
        except Exception as exc:
            raise DriverError(
                "R2AG_STAGE_FAILED",
                "R²AG neural conditioning stage failed",
                consumed=True,
            ) from exc
        _validate_conditioned(conditioned, prepared)

        try:
            generated: GenerationResult = runtime.generate(
                conditioned.value,
                max_new_tokens=max_new_tokens,
                seed=_nonnegative_int(payload, "seed"),
            )
        except Exception as exc:
            raise DriverError(
                "READER_FAILED",
                "R²AG reader generation failed",
                consumed=True,
            ) from exc
        _validate_generation(generated, max_new_tokens)

        attestation = {
            "fixture_mode": self.fixture_mode,
            "source_commit": UPSTREAM_COMMIT,
            "source_manifest_sha256": artifacts["source_manifest_sha256"],
            "compatibility_manifest_sha256": self.compatibility_manifest_sha256,
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
            "runtime_dependencies": dict(RUNTIME_DEPENDENCIES),
            "runtime_dependencies_verified": not self.fixture_mode,
            "execution_evidence": (
                "fixture_source_execution"
                if self.fixture_mode
                else "pinned_source_checkpoint_execution"
            ),
            "source_attestation": source_attestation,
        }
        response_payload = {
            "candidate_document_ids": candidate_ids,
            "candidate_documents_sha256": payload["candidate_documents_sha256"],
            "candidate_order_sha256": payload["candidate_order_sha256"],
            "candidate_pool_fingerprint_sha256": payload["candidate_pool_fingerprint_sha256"],
            "artifacts": artifacts,
            "artifact_identity_sha256": payload["artifact_identity_sha256"],
            "worker_config_num_k": num_k,
            "features": features,
            "feature_tensor_sha256": feature_hash,
            "feature_embedder_fingerprint_sha256": payload["feature_embedder_fingerprint_sha256"],
            "feature_artifact_fingerprint_sha256": payload["feature_artifact_fingerprint_sha256"],
            "reader_instruction_sha256": payload["reader_instruction_sha256"],
            "reader_input_sha256": payload["reader_input_sha256"],
            "r2former_output_sha256": rformer.sha256,
            "projection_output_sha256": projection.sha256,
            "r_token_count": num_k,
            "r_token_positions": prepared.marker_positions,
            "non_retrieval_unk_positions_modified": False,
            "runtime_attestation": attestation,
            "answer": generated.answer,
        }
        return {
            "status": "ok",
            "request_id": request["request_id"],
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL,
            "payload": response_payload,
            "usage": {
                "model_calls": 1,
                "retrieval_calls": 0,
                "tool_calls": 0,
                "embedding_calls": embedding_calls,
                "input_tokens": prepared.input_tokens,
                "output_tokens": generated.output_tokens,
                "accelerator_seconds": float(generated.accelerator_seconds),
            },
        }

    def _runtime(self) -> Any:
        if self.runtime_factory is not None:
            return self.runtime_factory(
                self.source_root,
                self.checkpoint_root,
                self.base_model_root,
                self.tokenizer_root,
            )
        from workers.r2ag.runtime import OfficialR2AGRuntime

        return OfficialR2AGRuntime(
            source_root=self.source_root,
            checkpoint_root=self.checkpoint_root,
            base_model_root=self.base_model_root,
            tokenizer_root=self.tokenizer_root,
            expected_runtime_versions=RUNTIME_DEPENDENCIES,
        )

    def _verify_source(self) -> Mapping[str, Any]:
        source = _mapping(self.manifest, "source")
        required = _mapping(source, "required_files")
        for relative, expected in required.items():
            if _sha256_file(_safe_file(self.source_root, relative)) != expected:
                raise DriverError("SOURCE_MISMATCH", f"source hash mismatch: {relative}")
        source_manifest = _content_hash(required)
        if self.fixture_mode:
            actual_files = {
                path.relative_to(self.source_root).as_posix()
                for path in self.source_root.rglob("*")
                if path.is_file()
            }
            if actual_files != set(required):
                raise DriverError("SOURCE_NOT_CLOSED", "fixture source tree is not closed")
            return {
                "kind": "closed_fixture_source",
                "tree": source.get("tree"),
                "content_manifest_sha256": source_manifest,
            }
        if source.get("commit") != UPSTREAM_COMMIT or source.get("tree") != UPSTREAM_TREE:
            raise DriverError("SOURCE_MISMATCH", "compatibility source identity drifted")
        head = _git(self.source_root, "rev-parse", "HEAD")
        tree = _git(self.source_root, "rev-parse", "HEAD^{tree}")
        status = _git(self.source_root, "status", "--porcelain", "--untracked-files=all")
        if head != UPSTREAM_COMMIT or tree != UPSTREAM_TREE or status:
            raise DriverError("SOURCE_MISMATCH", "source checkout is not pinned and clean")
        return {
            "kind": "clean_git_checkout",
            "tree": tree,
            "content_manifest_sha256": source_manifest,
        }

    def _verify_static_artifacts(self, artifacts: Mapping[str, Any]) -> None:
        source_required = _mapping(_mapping(self.manifest, "source"), "required_files")
        if artifacts.get("source_manifest_sha256") != _content_hash(source_required):
            raise DriverError("SOURCE_MISMATCH", "source manifest identity mismatch")
        checks = (
            (
                _safe_file(self.checkpoint_root, "config.json"),
                artifacts["checkpoint_config_sha256"],
                "checkpoint config",
            ),
            (
                self.feature_schema_path,
                artifacts["feature_schema_sha256"],
                "feature schema",
            ),
        )
        for path, expected, label in checks:
            if _sha256_file(path) != expected:
                raise DriverError("ARTIFACT_MISMATCH", f"{label} hash mismatch")
        _verify_artifact_closure(
            self.base_model_root,
            repository=artifacts.get("base_model_repository"),
            revision=artifacts.get("base_model_revision"),
            manifest_sha256=artifacts.get("base_model_manifest_sha256"),
            label="base model",
        )
        _verify_artifact_closure(
            self.tokenizer_root,
            repository=artifacts.get("tokenizer_repository"),
            revision=artifacts.get("tokenizer_revision"),
            manifest_sha256=artifacts.get("tokenizer_manifest_sha256"),
            label="tokenizer",
        )

    def _verify_feature_embedder(self, pool: Mapping[str, Any]) -> None:
        identity = _mapping(pool, "retriever")
        manifest_sha256 = identity.get("manifest_sha256")
        if not isinstance(manifest_sha256, str) or not _HEX64.fullmatch(manifest_sha256):
            raise DriverError(
                "ARTIFACT_MISMATCH",
                "worker feature source requires retriever manifest_sha256",
            )
        _verify_artifact_closure(
            self.checkpoint_root / "feature-embedder",
            repository=identity.get("repository"),
            revision=identity.get("revision"),
            manifest_sha256=manifest_sha256,
            label="feature embedder",
        )


def _validate_envelope(request: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(request, Mapping):
        raise DriverError("REQUEST_INVALID", "request must be a mapping")
    if set(request) != _REQUEST_FIELDS:
        raise DriverError("REQUEST_INVALID", "request envelope fields drifted")
    expected = {
        "method_id": "r2ag",
        "release_id": RELEASE,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol_version": PROTOCOL,
    }
    if any(request.get(key) != value for key, value in expected.items()):
        raise DriverError("IDENTITY_MISMATCH", "method/release/source/protocol mismatch")
    state_hash = request.get("state_hash")
    if not isinstance(state_hash, str) or not _HEX64.fullmatch(state_hash):
        raise DriverError("REQUEST_INVALID", "state_hash must be a SHA-256 digest")
    reservation = _mapping(request, "minimum_reservation")
    if dict(reservation) != {"model_calls": 1}:
        raise DriverError("REQUEST_INVALID", "minimum reservation must be one model call")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not _HEX64.fullmatch(request_id):
        raise DriverError("REQUEST_INVALID", "request_id must be a SHA-256 digest")
    identity = {key: request[key] for key in _REQUEST_ID_FIELDS}
    if request_id != _content_hash(identity):
        raise DriverError("IDENTITY_MISMATCH", "request_id content hash mismatch")
    return _mapping(request, "payload")


def _validate_contract(value: Mapping[str, Any]) -> None:
    expected = {
        "allow_training": False,
        "allow_gold_metadata": False,
        "require_source_verification": True,
        "require_checkpoint_hash_verification": True,
        "require_rformer_then_projection": True,
        "replace_only_retrieval_marker_embeddings": True,
        "check_budget_before_reader_call": True,
    }
    if dict(value) != expected:
        raise DriverError("EXECUTION_CONTRACT_REJECTED", "R²AG execution contract drifted")


def _candidate_documents(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("REQUEST_INVALID", "candidate_documents must be a sequence")
    rows = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _CANDIDATE_FIELDS:
            raise DriverError("ORACLE_FIELD_REJECTED", "candidate DTO fields are not allowlisted")
        for key in ("document_id", "text", "title", "source", "corpus_revision"):
            if not isinstance(raw.get(key), str):
                raise DriverError("REQUEST_INVALID", f"candidate {key} must be text")
        if not raw["document_id"] or not raw["text"]:
            raise DriverError("REQUEST_INVALID", "candidate ID/text must not be empty")
        score = raw.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise DriverError("REQUEST_INVALID", "candidate score must be finite or null")
        rows.append(dict(raw))
    return tuple(rows)


def _validate_pool(
    payload: Mapping[str, Any],
    candidates: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    pool = _mapping(payload, "candidate_pool_fingerprint")
    if set(pool) != _POOL_FIELDS:
        raise DriverError("IDENTITY_MISMATCH", "candidate pool fingerprint fields drifted")
    _reject_forbidden_fingerprint_keys(pool, label="candidate pool fingerprint")
    if pool.get("top_k") != len(candidates) or pool.get("order") != "rerank_desc":
        raise DriverError("IDENTITY_MISMATCH", "candidate pool count/order mismatch")
    retriever = _repository_identity(
        pool.get("retriever"),
        label="candidate pool retriever",
    )
    corpus = _repository_identity(
        pool.get("corpus"),
        label="candidate pool corpus",
    )
    index = _index_identity(pool.get("index"), label="candidate pool index")
    if pool.get("content_sha256") != _content_hash(candidates):
        raise DriverError("IDENTITY_MISMATCH", "candidate pool content hash mismatch")
    normalized = {
        "retriever": retriever,
        "corpus": corpus,
        "index": index,
        "top_k": pool["top_k"],
        "order": pool["order"],
        "content_sha256": pool["content_sha256"],
    }
    if dict(pool) != normalized:
        raise DriverError("IDENTITY_MISMATCH", "candidate pool fingerprint is not canonical")
    if payload.get("candidate_pool_fingerprint_sha256") != _content_hash(normalized):
        raise DriverError("IDENTITY_MISMATCH", "candidate pool identity hash mismatch")
    return normalized


def _validate_feature_identities(
    payload: Mapping[str, Any],
    *,
    feature_source: Any,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    embedder_raw = payload.get("feature_embedder_fingerprint")
    artifact_raw = payload.get("feature_artifact_fingerprint")
    if feature_source == "embedder":
        embedder = _feature_embedder_identity(
            embedder_raw,
            label="feature embedder fingerprint",
        )
        artifact = None
    elif feature_source == "candidate_metadata":
        embedder = None
        artifact = _feature_artifact_identity(
            artifact_raw,
            label="feature artifact fingerprint",
        )
    elif feature_source == "worker":
        embedder = None
        artifact = None
    else:
        raise DriverError("PROFILE_DRIFT", "unsupported R²AG feature source")
    if embedder_raw is not None and embedder is None:
        raise DriverError("PROFILE_DRIFT", "unexpected feature embedder fingerprint")
    if artifact_raw is not None and artifact is None:
        raise DriverError("PROFILE_DRIFT", "unexpected feature artifact fingerprint")
    if payload.get("feature_embedder_fingerprint_sha256") != _content_hash(embedder):
        raise DriverError("IDENTITY_MISMATCH", "feature embedder fingerprint hash mismatch")
    if payload.get("feature_artifact_fingerprint_sha256") != _content_hash(artifact):
        raise DriverError("IDENTITY_MISMATCH", "feature artifact fingerprint hash mismatch")
    return embedder, artifact


def _validate_dependency_fingerprints(
    value: Mapping[str, Any],
    *,
    pool: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    feature_source: Any,
    feature_embedder_identity: Mapping[str, Any] | None,
    feature_artifact_identity: Mapping[str, Any] | None,
) -> None:
    expected_keys = {"candidate_pool", "worker", "artifacts"}
    if feature_source == "embedder":
        expected_keys.add("feature_embedder")
    elif feature_source == "candidate_metadata":
        expected_keys.add("feature_artifact")
    if set(value) != expected_keys:
        raise DriverError("IDENTITY_MISMATCH", "dependency fingerprint fields drifted")
    _reject_forbidden_fingerprint_keys(value, label="dependency fingerprints")
    if value.get("candidate_pool") != pool or value.get("artifacts") != artifacts:
        raise DriverError("IDENTITY_MISMATCH", "dependency fingerprint identity mismatch")
    worker = value.get("worker")
    if not isinstance(worker, Mapping) or not worker:
        raise DriverError("IDENTITY_MISMATCH", "worker fingerprint must be a mapping")
    if feature_source == "embedder" and value.get("feature_embedder") != (
        feature_embedder_identity
    ):
        raise DriverError("IDENTITY_MISMATCH", "feature embedder dependency mismatch")
    if feature_source == "candidate_metadata" and value.get("feature_artifact") != (
        feature_artifact_identity
    ):
        raise DriverError("IDENTITY_MISMATCH", "feature artifact dependency mismatch")


def _features(value: Any, *, expected_count: int) -> tuple[Mapping[str, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("REQUEST_INVALID", "features must be a sequence")
    rows = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != set(FEATURE_NAMES):
            raise DriverError("REQUEST_INVALID", "feature row schema mismatch")
        row: dict[str, float] = {}
        for name in FEATURE_NAMES:
            number = raw[name]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number < -1.000001
                or number > 1.000001
            ):
                raise DriverError("REQUEST_INVALID", "feature values must be finite cosine scores")
            row[name] = float(number)
        rows.append(row)
    if len(rows) != expected_count:
        raise DriverError("REQUEST_INVALID", "feature count must equal config.num_k")
    return tuple(rows)


def _repository_identity(value: Any, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise DriverError("IDENTITY_MISMATCH", f"{label} must be a mapping")
    _reject_forbidden_fingerprint_keys(value, label=label)
    keys = set(value)
    allowed = _REPOSITORY_IDENTITY_REQUIRED | _REPOSITORY_IDENTITY_OPTIONAL
    if not _REPOSITORY_IDENTITY_REQUIRED.issubset(keys) or not keys.issubset(allowed):
        raise DriverError(
            "IDENTITY_MISMATCH",
            f"{label} must contain repository/revision and optional manifest_sha256 only",
        )
    repository = value.get("repository")
    revision = _pinned_revision(value.get("revision"), label=f"{label}.revision")
    if not isinstance(repository, str) or not repository.strip():
        raise DriverError("IDENTITY_MISMATCH", f"{label} values must be non-empty text")
    identity: dict[str, str] = {
        "repository": repository.strip(),
        "revision": revision,
    }
    if "manifest_sha256" in value:
        digest = value.get("manifest_sha256")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise DriverError("IDENTITY_MISMATCH", f"{label} manifest hash is invalid")
        identity["manifest_sha256"] = digest
    return identity


def _index_identity(value: Any, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise DriverError("IDENTITY_MISMATCH", f"{label} must be a mapping")
    _reject_forbidden_fingerprint_keys(value, label=label)
    keys = set(value)
    allowed = _INDEX_IDENTITY_REQUIRED | _INDEX_IDENTITY_OPTIONAL
    if not _INDEX_IDENTITY_REQUIRED.issubset(keys) or not keys.issubset(allowed):
        raise DriverError(
            "IDENTITY_MISMATCH",
            f"{label} must contain algorithm/revision and optional manifest_sha256 only",
        )
    algorithm = value.get("algorithm")
    revision = _pinned_revision(value.get("revision"), label=f"{label}.revision")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise DriverError("IDENTITY_MISMATCH", f"{label} values must be non-empty text")
    identity: dict[str, str] = {
        "algorithm": algorithm.strip(),
        "revision": revision,
    }
    if "manifest_sha256" in value:
        digest = value.get("manifest_sha256")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise DriverError("IDENTITY_MISMATCH", f"{label} manifest hash is invalid")
        identity["manifest_sha256"] = digest
    return identity


def _feature_embedder_identity(value: Any, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != _FEATURE_EMBEDDER_FIELDS:
        raise DriverError(
            "IDENTITY_MISMATCH",
            f"{label} must contain exactly kind and fingerprint_sha256",
        )
    _reject_forbidden_fingerprint_keys(value, label=label)
    if value.get("kind") != "dense_embedder":
        raise DriverError("IDENTITY_MISMATCH", f"{label} kind is invalid")
    digest = value.get("fingerprint_sha256")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise DriverError("IDENTITY_MISMATCH", f"{label} digest is invalid")
    return {"kind": "dense_embedder", "fingerprint_sha256": digest}


def _feature_artifact_identity(value: Any, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise DriverError(
            "IDENTITY_MISMATCH",
            f"{label} must contain exactly repository, revision, and sha256",
        )
    _reject_forbidden_fingerprint_keys(value, label=label)
    if set(value) != _FEATURE_ARTIFACT_FIELDS:
        raise DriverError(
            "IDENTITY_MISMATCH",
            f"{label} must contain exactly repository, revision, and sha256",
        )
    repository = value.get("repository")
    revision = _pinned_revision(value.get("revision"), label=f"{label}.revision")
    digest = value.get("sha256")
    if (
        not isinstance(repository, str)
        or not repository.strip()
        or not isinstance(digest, str)
        or not _HEX64.fullmatch(digest)
    ):
        raise DriverError("IDENTITY_MISMATCH", f"{label} values are invalid")
    return {
        "repository": repository.strip(),
        "revision": revision,
        "sha256": digest,
    }


def _reject_forbidden_fingerprint_keys(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise DriverError("REQUEST_INVALID", f"{label} keys must be strings")
            terms = tuple(re.findall(r"[a-z0-9]+", key.lower()))
            compact = "".join(terms)
            if any(term in _FORBIDDEN_FINGERPRINT_TERMS for term in (*terms, compact)):
                raise DriverError(
                    "ORACLE_FIELD_REJECTED",
                    f"{label} contains a forbidden metadata key",
                )
            _reject_forbidden_fingerprint_keys(nested, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_forbidden_fingerprint_keys(nested, label=label)


def _pinned_revision(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriverError("IDENTITY_MISMATCH", f"{label} must be non-empty text")
    revision = value.strip()
    if revision.lower() in _IMMUTABLE_REVISION_ALIASES:
        raise DriverError("IDENTITY_MISMATCH", f"{label} must be immutable")
    return revision


def _validate_descending(features: Sequence[Mapping[str, float]]) -> None:
    if any(
        features[index]["rerank"] + 1e-6 < features[index + 1]["rerank"]
        for index in range(len(features) - 1)
    ):
        raise DriverError("REQUEST_INVALID", "features are not in descending rerank order")


def _validate_prepared(value: Any, num_k: int) -> None:
    if not isinstance(value, PreparedReader):
        raise DriverError("TOKENIZATION_FAILED", "runtime returned invalid prepared reader")
    if (
        value.marker_positions != tuple(sorted(value.marker_positions))
        or len(value.marker_positions) != num_k
        or len(set(value.marker_positions)) != num_k
        or isinstance(value.input_tokens, bool)
        or not isinstance(value.input_tokens, int)
        or value.input_tokens < 1
    ):
        raise DriverError("MARKER_MISMATCH", "tokenized marker positions do not match num_k")


def _validate_tensor_stage(value: Any, *, label: str) -> None:
    if not isinstance(value, TensorStage) or not _HEX64.fullmatch(value.sha256):
        raise DriverError("R2AG_STAGE_FAILED", f"{label} returned invalid evidence", consumed=True)


def _validate_conditioned(value: Any, prepared: PreparedReader) -> None:
    if (
        not isinstance(value, ConditionedReader)
        or value.marker_positions != prepared.marker_positions
        or value.only_marker_positions_modified is not True
    ):
        raise DriverError(
            "MARKER_MISMATCH",
            "runtime did not replace only the retrieval markers",
            consumed=True,
        )


def _validate_generation(value: Any, max_new_tokens: int) -> None:
    if (
        not isinstance(value, GenerationResult)
        or not isinstance(value.answer, str)
        or not value.answer.strip()
        or isinstance(value.output_tokens, bool)
        or not isinstance(value.output_tokens, int)
        or value.output_tokens < 1
        or value.output_tokens > max_new_tokens
        or isinstance(value.accelerator_seconds, bool)
        or not isinstance(value.accelerator_seconds, (int, float))
        or not math.isfinite(value.accelerator_seconds)
        or value.accelerator_seconds < 0
    ):
        raise DriverError("READER_FAILED", "runtime returned invalid reader output", consumed=True)


def _validate_artifact_identity(artifacts: Mapping[str, Any]) -> None:
    if set(artifacts) != _ARTIFACT_FIELDS:
        raise DriverError("REQUEST_INVALID", "artifact identity fields drifted")
    required = {
        "checkpoint_config_sha256",
        "rformer_weights_sha256",
        "projection_weights_sha256",
        "base_model_manifest_sha256",
        "tokenizer_manifest_sha256",
        "feature_schema_sha256",
        "source_manifest_sha256",
        "compatibility_manifest_sha256",
    }
    if any(
        not isinstance(artifacts.get(key), str) or not _HEX64.fullmatch(artifacts[key])
        for key in required
    ):
        raise DriverError("REQUEST_INVALID", "artifact SHA-256 identity is invalid")
    for key in (
        "checkpoint_repository",
        "base_model_repository",
        "tokenizer_repository",
    ):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DriverError("IDENTITY_MISMATCH", f"{key} must be non-empty text")
    for key in (
        "checkpoint_revision",
        "base_model_revision",
        "tokenizer_revision",
        "feature_schema_revision",
    ):
        _pinned_revision(artifacts.get(key), label=key)


def _verify_artifact_closure(
    root: Path,
    *,
    repository: Any,
    revision: Any,
    manifest_sha256: Any,
    label: str,
) -> Mapping[str, Any]:
    """Verify one exact, symlink-free local snapshot closure."""

    if root.is_symlink():
        raise DriverError("PATH_REJECTED", f"{label} root must not be a symlink")
    try:
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DriverError("PATH_REJECTED", f"{label} root is missing") from exc
    if not resolved_root.is_dir():
        raise DriverError("PATH_REJECTED", f"{label} root must be a directory")
    manifest_path = _safe_file(resolved_root, "agentinputlab-manifest.json")
    if not isinstance(manifest_sha256, str) or not _HEX64.fullmatch(manifest_sha256):
        raise DriverError("ARTIFACT_MISMATCH", f"{label} manifest hash is invalid")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise DriverError("ARTIFACT_MISMATCH", f"{label} manifest hash mismatch")
    manifest = _load_json(manifest_path, label=f"{label} manifest")
    if set(manifest) != _ARTIFACT_MANIFEST_FIELDS:
        raise DriverError("MANIFEST_INVALID", f"{label} manifest fields drifted")
    if (
        manifest.get("schema") != _ARTIFACT_MANIFEST_SCHEMA
        or manifest.get("repository") != repository
        or manifest.get("revision") != revision
    ):
        raise DriverError("ARTIFACT_MISMATCH", f"{label} manifest identity mismatch")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise DriverError("MANIFEST_INVALID", f"{label} files closure must be non-empty")

    expected_files: dict[str, Mapping[str, Any]] = {}
    for relative, descriptor in raw_files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or relative == "agentinputlab-manifest.json"
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise DriverError("MANIFEST_INVALID", f"{label} manifest path is invalid")
        if not isinstance(descriptor, Mapping) or set(descriptor) != _ARTIFACT_FILE_FIELDS:
            raise DriverError("MANIFEST_INVALID", f"{label} file descriptor is invalid")
        size = descriptor.get("size")
        digest = descriptor.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _HEX64.fullmatch(digest)
        ):
            raise DriverError("MANIFEST_INVALID", f"{label} file descriptor is invalid")
        expected_files[relative] = {"size": size, "sha256": digest}

    entries = tuple(resolved_root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise DriverError("PATH_REJECTED", f"{label} closure contains a symlink")
    actual_files = {
        path.relative_to(resolved_root).as_posix() for path in entries if path.is_file()
    }
    if actual_files != {*expected_files, "agentinputlab-manifest.json"}:
        raise DriverError("ARTIFACT_MISMATCH", f"{label} file closure mismatch")
    for relative, descriptor in expected_files.items():
        path = _safe_file(resolved_root, relative)
        if path.stat().st_size != descriptor["size"]:
            raise DriverError("ARTIFACT_MISMATCH", f"{label} file size mismatch")
        if _sha256_file(path) != descriptor["sha256"]:
            raise DriverError("ARTIFACT_MISMATCH", f"{label} file hash mismatch")
    return manifest


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DriverError("PATH_REJECTED", "artifact path is invalid")
    candidate = root / relative
    if candidate.is_symlink():
        raise DriverError("PATH_REJECTED", "symlinks are forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise DriverError("PATH_REJECTED", "artifact path escapes its mount or is missing") from exc
    if not resolved.is_file():
        raise DriverError("PATH_REJECTED", "artifact path must be a regular file")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("MANIFEST_INVALID", f"{label} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise DriverError("MANIFEST_INVALID", f"{label} must be a JSON object")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise DriverError("REQUEST_INVALID", f"{key} must be a mapping")
    return result


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise DriverError("REQUEST_INVALID", f"{key} must be non-empty text")
    return result


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("REQUEST_INVALID", f"{label} must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise DriverError("REQUEST_INVALID", f"{label} must contain non-empty text")
    return result


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise DriverError("REQUEST_INVALID", f"{key} must be a positive integer")
    return result


def _nonnegative_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise DriverError("REQUEST_INVALID", f"{key} must be a non-negative integer")
    return result


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise DriverError("SOURCE_MISMATCH", "source Git verification failed")
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DriverError("ARTIFACT_INVALID", "artifact is unreadable") from exc
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def serve_jsonl(
    driver: R2AGDriver,
    source: TextIO = sys.stdin,
    sink: TextIO = sys.stdout,
) -> int:
    """Serve one strict JSON response for every non-empty JSON input line."""

    for line in source:
        if not line.strip():
            continue
        try:
            request = _loads_json_object(line)
            with redirect_stdout(sys.stderr):
                response = driver.handle(request)
        except (json.JSONDecodeError, DriverError, TypeError, ValueError):
            response = {
                "status": "error",
                "failure_code": "REQUEST_INVALID",
                "message": "request is not valid canonical JSON",
                "consumed": False,
            }
        sink.write(
            json.dumps(
                response,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        sink.flush()
    return 0


def _loads_json_object(value: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    decoded = json.loads(value, object_pairs_hook=reject_duplicates)
    if not isinstance(decoded, Mapping):
        raise ValueError("JSON request must be an object")
    return decoded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pinned R²AG JSONL worker")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--base-model-root", required=True, type=Path)
    parser.add_argument("--tokenizer-root", required=True, type=Path)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-schema-path", type=Path, default=DEFAULT_FEATURE_SCHEMA)
    arguments = parser.parse_args(argv)
    driver = R2AGDriver(
        source_root=arguments.source_root,
        checkpoint_root=arguments.checkpoint_root,
        base_model_root=arguments.base_model_root,
        tokenizer_root=arguments.tokenizer_root,
        manifest_path=arguments.manifest_path,
        feature_schema_path=arguments.feature_schema_path,
    )
    return serve_jsonl(driver)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FEATURE_NAMES",
    "PROTOCOL",
    "RELEASE",
    "RUNTIME_DEPENDENCIES",
    "UPSTREAM_COMMIT",
    "ConditionedReader",
    "DriverError",
    "GenerationResult",
    "PreparedReader",
    "R2AGDriver",
    "TensorStage",
    "serve_jsonl",
]
