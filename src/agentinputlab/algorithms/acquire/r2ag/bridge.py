"""Controlled, source-informed same-candidate execution path for R²AG.

The upstream implementation has two materially different stages:

1. derive three retrieval features for every *already retrieved* document; and
2. pass that tensor through a trained R²-Former/projection before replacing one
   retrieval-marker embedding per document in the frozen reader.

AgentInputLab executes the deterministic first stage locally (or accepts the
official pre-computed feature fields) and delegates only the checkpoint-bound
neural stage to a pinned worker.  The worker must echo every treatment-defining
hash and attest that it replaced exactly the intended marker embeddings.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from agentinputlab.backends.vector import DenseEmbedder
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CompositeResult,
    ContextBlock,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    dependency_fingerprint,
    finite_number,
    mapping_sequence,
    nonnegative_integer,
    optional_text,
    required_text,
    resolve_worker,
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

R2AG_UPSTREAM_COMMIT = "61cfd1027c0101172fda701ec816bc18360b5bab"
R2AG_PROTOCOL = "agentinputlab.r2ag.v4"
R2AG_FEATURE_NAMES = ("rerank", "neighbor", "precedent")
_RUNTIME_DEPENDENCIES = {
    "torch": "2.1.2",
    "transformers": "4.40.2",
    "sentence-transformers": "2.6.1",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_ALIASES = frozenset({"default", "head", "latest", "main", "master"})
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
_CANDIDATE_POOL_KEYS = frozenset(
    {"retriever", "corpus", "index", "top_k", "order", "content_sha256"}
)
_REPOSITORY_IDENTITY_REQUIRED = frozenset({"repository", "revision"})
_REPOSITORY_IDENTITY_OPTIONAL = frozenset({"manifest_sha256"})
_INDEX_IDENTITY_REQUIRED = frozenset({"algorithm", "revision"})
_INDEX_IDENTITY_OPTIONAL = frozenset({"manifest_sha256"})
_FEATURE_ARTIFACT_KEYS = frozenset({"repository", "revision", "sha256"})
_FEATURE_EMBEDDER_KEYS = frozenset({"kind", "fingerprint_sha256"})
_CANDIDATE_ORDER = "rerank_desc"

_READER_TEMPLATE = (
    "Write a high-quality answer for the given question using only the provided "
    "search results (some of which might be irrelevant). Only give me the answer "
    "and do not output any other words. The similarity information is provided "
    "in front of search results.\n\n"
    "{search_results}\n\n"
    "Only give me the answer and do not output any other words. \n"
    "Question: {question}\n"
    "Answer:"
)
_INSTRUCTION_WRAPPER = "### Instruction: \n{instruction}\n\n### Response:\n"

Vector = tuple[float, ...]
FeatureRow = tuple[float, float, float]


class R2AGFeatureSource(str, Enum):
    """Where the official three-column retrieval tensor is produced."""

    WORKER = "worker"
    EMBEDDER = "embedder"
    CANDIDATE_METADATA = "candidate_metadata"


def _pinned_revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"R²AG {label} must be a non-empty revision")
    revision = value.strip()
    if revision.lower() in _IMMUTABLE_REVISION_ALIASES:
        raise ValidationError(f"R²AG {label} must be immutable")
    return revision


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValidationError(f"R²AG {label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class R2AGArtifactIdentity:
    """Hash-bound source, checkpoint, reader, tokenizer, and feature identities.

    This identity deliberately does not imply that AgentInputLab redistributes or
    has benchmark-validated the mounted weights. When paired with the worker's
    complete file-closure manifests, it hash-binds the mounted treatment without
    asserting paper checkpoint or benchmark-result parity.
    """

    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_config_sha256: str
    rformer_weights_sha256: str
    projection_weights_sha256: str
    checkpoint_num_documents: int
    base_model_repository: str
    base_model_revision: str
    base_model_manifest_sha256: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_manifest_sha256: str
    feature_schema_revision: str
    feature_schema_sha256: str
    source_manifest_sha256: str
    compatibility_manifest_sha256: str
    upstream_commit: str = R2AG_UPSTREAM_COMMIT

    def __post_init__(self) -> None:
        if self.upstream_commit != R2AG_UPSTREAM_COMMIT:
            raise ValidationError("R²AG artifacts target an unsupported upstream commit")
        for name in (
            "checkpoint_repository",
            "checkpoint_revision",
            "base_model_repository",
            "base_model_revision",
            "tokenizer_repository",
            "tokenizer_revision",
            "feature_schema_revision",
        ):
            object.__setattr__(
                self,
                name,
                _pinned_revision(getattr(self, name), label=name),
            )
        for name in (
            "checkpoint_config_sha256",
            "rformer_weights_sha256",
            "projection_weights_sha256",
            "base_model_manifest_sha256",
            "tokenizer_manifest_sha256",
            "feature_schema_sha256",
            "source_manifest_sha256",
            "compatibility_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))
        if (
            isinstance(self.checkpoint_num_documents, bool)
            or not isinstance(self.checkpoint_num_documents, int)
            or self.checkpoint_num_documents < 2
            or self.checkpoint_num_documents > 30
        ):
            raise ValidationError(
                "R²AG checkpoint_num_documents must be an integer between 2 and 30"
            )

    def as_mapping(self) -> Mapping[str, Any]:
        return freeze_mapping(asdict(self))

    @property
    def identity_sha256(self) -> str:
        return content_hash(self.as_mapping())


@dataclass(frozen=True, slots=True)
class R2AGControls:
    max_documents: int = 30
    max_output_tokens: int = 2048
    feature_source: R2AGFeatureSource = R2AGFeatureSource.WORKER
    checkpoint_num_documents: int = 10
    require_rerank_order: bool = True
    numeric_tolerance: float = 1e-6
    max_reader_chars: int = 2_000_000
    retrieval_token: str = "<R>"
    unknown_token: str = "<unk>"

    def __post_init__(self) -> None:
        for value in (self.max_output_tokens, self.max_reader_chars):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError("R²AG document, output, and reader limits must be positive")
        if (
            isinstance(self.max_documents, bool)
            or not isinstance(self.max_documents, int)
            or self.max_documents < 2
        ):
            raise ValidationError("R²AG max_documents must be at least two")
        if not isinstance(self.feature_source, R2AGFeatureSource):
            raise ValidationError("R²AG feature_source must be an R2AGFeatureSource")
        if (
            isinstance(self.checkpoint_num_documents, bool)
            or not isinstance(self.checkpoint_num_documents, int)
            or self.checkpoint_num_documents < 2
            or self.checkpoint_num_documents > self.max_documents
        ):
            raise ValidationError(
                "R²AG checkpoint_num_documents must be between 2 and max_documents"
            )
        if self.require_rerank_order is not True:
            raise ValidationError("R²AG controlled comparison requires descending rerank order")
        if (
            isinstance(self.numeric_tolerance, bool)
            or not isinstance(self.numeric_tolerance, (int, float))
            or not math.isfinite(self.numeric_tolerance)
            or self.numeric_tolerance < 0
            or self.numeric_tolerance > 1e-3
        ):
            raise ValidationError("R²AG numeric_tolerance must be between 0 and 1e-3")
        for name in ("retrieval_token", "unknown_token"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValidationError(f"R²AG {name} must not be empty")
        if self.retrieval_token == self.unknown_token:
            raise ValidationError("R²AG retrieval_token and unknown_token must be distinct")


@dataclass(slots=True)
class R2AGControlledWorker:
    """Execute a controlled R²AG-shaped treatment on fixed candidates.

    ``feature_source=EMBEDDER`` reproduces the formulas in the pinned upstream
    ``retrieval/retrieval_utils.py`` without importing its CUDA-side-effectful
    module. ``CANDIDATE_METADATA`` consumes the three fields emitted by the
    reviewed feature-extraction scripts. ``WORKER`` retains the compatibility
    path for deployments that compute those features in the pinned worker.
    Complete closure verification hash-binds mounted artifacts; it does not
    establish paper-checkpoint or benchmark-result parity.
    """

    artifacts: R2AGArtifactIdentity
    client: WorkerClient | None = None
    embedder: DenseEmbedder | None = None
    controls: R2AGControls = R2AGControls()
    allow_fixture_worker: bool = False
    method_id: str = "r2ag"
    release_id: str = "acquire.r2ag.controlled_same_candidates"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, R2AGArtifactIdentity):
            raise ValidationError("R²AG requires R2AGArtifactIdentity")
        if not isinstance(self.controls, R2AGControls):
            raise ValidationError("R²AG controls must be R2AGControls")
        if self.controls.checkpoint_num_documents != self.artifacts.checkpoint_num_documents:
            raise ValidationError(
                "R²AG controls and artifact config must use the same checkpoint_num_documents"
            )
        if not isinstance(self.allow_fixture_worker, bool):
            raise ValidationError("R²AG allow_fixture_worker must be boolean")

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        if not state.query.strip():
            raise ValidationError("R²AG requires a non-empty query")
        if state.instruction.strip() or state.messages or state.tool_names:
            raise ValidationError(
                "R²AG controlled_same_candidates accepts query plus candidate context only"
            )
        candidate_ids = self._validate_candidates(state)
        candidate_documents = _candidate_documents(state.context)
        candidate_documents_hash = content_hash(candidate_documents)
        candidate_order_hash = content_hash(candidate_ids)
        candidate_pool_identity = _candidate_pool_fingerprint(
            context,
            candidate_documents=candidate_documents,
        )
        remaining_budget = _remaining_budget(state, context)
        if state.budget.max_cost_usd is not None:
            raise BudgetExceededError("R²AG external execution has no pre-call cost upper bound")
        if remaining_budget["model_calls"] < 1:
            raise BudgetExceededError("R²AG has no remaining reader model-call budget")
        if remaining_budget["output_tokens"] < 1:
            raise BudgetExceededError("R²AG has no remaining output-token budget")
        context_tokens = remaining_budget["context_tokens"]
        if context_tokens is not None and context_tokens <= 0:
            raise BudgetExceededError("R²AG has no remaining context-token budget")
        reader_instruction, reader_input = _build_reader_input(
            state,
            retrieval_token=self.controls.retrieval_token,
            unknown_token=self.controls.unknown_token,
            max_chars=self.controls.max_reader_chars,
        )
        expected_features, feature_dependencies = self._prepare_features(state, context)
        feature_embedder_identity = feature_dependencies.get("feature_embedder")
        feature_artifact_identity = feature_dependencies.get("feature_artifact")
        expected_feature_hash = (
            content_hash(expected_features) if expected_features is not None else ""
        )
        reader_instruction_hash = content_hash(reader_instruction)
        reader_input_hash = content_hash(reader_input)

        worker = resolve_worker(self.client, context, label="R²AG")
        worker_identity = worker_fingerprint(worker, label="R²AG")
        if worker_identity.get("runtime_dependencies") != _RUNTIME_DEPENDENCIES:
            raise ExternalDependencyError(
                "R²AG worker runtime dependency versions do not match the pinned profile"
            )
        artifact_mapping = self.artifacts.as_mapping()
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=R2AG_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "task_id": state.task_id,
                "query": state.query,
                "candidate_documents": candidate_documents,
                "candidate_document_ids": candidate_ids,
                "candidate_documents_sha256": candidate_documents_hash,
                "candidate_order_sha256": candidate_order_hash,
                "candidate_pool_fingerprint": candidate_pool_identity,
                "candidate_pool_fingerprint_sha256": content_hash(candidate_pool_identity),
                "artifacts": artifact_mapping,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "features": (
                    _feature_mappings(expected_features) if expected_features is not None else None
                ),
                "feature_tensor_sha256": expected_feature_hash,
                "feature_source": self.controls.feature_source.value,
                "feature_names": R2AG_FEATURE_NAMES,
                "feature_embedder_fingerprint": feature_embedder_identity,
                "feature_embedder_fingerprint_sha256": content_hash(feature_embedder_identity),
                "feature_artifact_fingerprint": feature_artifact_identity,
                "feature_artifact_fingerprint_sha256": content_hash(feature_artifact_identity),
                "reader_instruction": reader_instruction,
                "reader_instruction_sha256": reader_instruction_hash,
                "reader_input": reader_input,
                "reader_input_sha256": reader_input_hash,
                "controls": asdict(self.controls),
                "remaining_budget": remaining_budget,
                "seed": context.seed,
                "execution_contract": {
                    "allow_training": False,
                    "allow_gold_metadata": False,
                    "require_source_verification": True,
                    "require_checkpoint_hash_verification": True,
                    "require_rformer_then_projection": True,
                    "replace_only_retrieval_marker_embeddings": True,
                    "check_budget_before_reader_call": True,
                },
            },
            dependency_fingerprints={
                "candidate_pool": candidate_pool_identity,
                "worker": worker_identity,
                "artifacts": artifact_mapping,
                **feature_dependencies,
            },
            minimum_reservation={"model_calls": 1},
            protocol_version=R2AG_PROTOCOL,
        )
        response = call_worker(worker, request)
        payload = response.payload
        returned_ids = string_tuple(
            payload.get("candidate_document_ids"),
            label="R²AG candidate_document_ids",
            allow_empty=False,
            unique=True,
        )
        if returned_ids != candidate_ids:
            raise AlgorithmOutputError("R²AG worker changed the fixed candidate ordering")
        _require_echo_hash(
            payload,
            "candidate_documents_sha256",
            candidate_documents_hash,
        )
        _require_echo_hash(payload, "candidate_order_sha256", candidate_order_hash)
        _require_echo_hash(
            payload,
            "candidate_pool_fingerprint_sha256",
            content_hash(candidate_pool_identity),
        )
        returned_artifacts = payload.get("artifacts")
        if not isinstance(returned_artifacts, Mapping) or dict(returned_artifacts) != dict(
            artifact_mapping
        ):
            raise AlgorithmOutputError("R²AG worker artifact identity mismatch")
        _require_echo_hash(
            payload,
            "artifact_identity_sha256",
            self.artifacts.identity_sha256,
        )
        worker_num_k = nonnegative_integer(
            payload.get("worker_config_num_k"),
            label="R²AG worker_config_num_k",
        )
        if worker_num_k != self.controls.checkpoint_num_documents:
            raise AlgorithmOutputError(
                "R²AG worker config.num_k does not match checkpoint candidate count"
            )

        features = _parse_features(payload.get("features"), expected_count=len(candidate_ids))
        _validate_feature_range_and_order(
            features,
            require_rerank_order=self.controls.require_rerank_order,
            tolerance=float(self.controls.numeric_tolerance),
        )
        if expected_features is not None:
            _require_same_features(
                expected_features,
                features,
                tolerance=float(self.controls.numeric_tolerance),
            )
        actual_feature_hash = content_hash(features)
        _require_echo_hash(
            payload,
            "feature_tensor_sha256",
            expected_feature_hash if expected_features is not None else actual_feature_hash,
        )
        _require_echo_hash(
            payload,
            "feature_embedder_fingerprint_sha256",
            content_hash(feature_embedder_identity),
        )
        _require_echo_hash(
            payload,
            "feature_artifact_fingerprint_sha256",
            content_hash(feature_artifact_identity),
        )
        _require_echo_hash(
            payload,
            "reader_instruction_sha256",
            reader_instruction_hash,
        )
        _require_echo_hash(payload, "reader_input_sha256", reader_input_hash)
        r2former_output_hash = _sha256_response(
            payload,
            "r2former_output_sha256",
        )
        projection_output_hash = _sha256_response(
            payload,
            "projection_output_sha256",
        )

        r_token_count = nonnegative_integer(
            payload.get("r_token_count"),
            label="R²AG r_token_count",
        )
        if r_token_count != len(candidate_ids):
            raise AlgorithmOutputError("R²AG requires exactly one <R> token per document")
        raw_positions = payload.get("r_token_positions")
        if isinstance(raw_positions, (str, bytes)) or not isinstance(raw_positions, Sequence):
            raise AlgorithmOutputError("R²AG r_token_positions must be a sequence")
        r_token_positions = tuple(
            nonnegative_integer(item, label="R²AG r_token_position") for item in raw_positions
        )
        if (
            len(r_token_positions) != len(candidate_ids)
            or len(r_token_positions) != len(set(r_token_positions))
            or r_token_positions != tuple(sorted(r_token_positions))
        ):
            raise AlgorithmOutputError("R²AG requires one unique ordered <R> position per document")
        if payload.get("non_retrieval_unk_positions_modified") is not False:
            raise AlgorithmOutputError(
                "R²AG worker must attest that non-retrieval unknown tokens were untouched"
            )
        fixture_mode, execution_evidence = _validate_runtime_attestation(
            payload.get("runtime_attestation"),
            artifacts=self.artifacts,
            expected_num_k=self.controls.checkpoint_num_documents,
            allow_fixture=self.allow_fixture_worker,
        )
        _validate_usage(
            response.usage,
            remaining_budget=remaining_budget,
            controls=self.controls,
        )
        answer = optional_text(payload, "answer", label="R²AG response")
        if not answer.strip():
            raise AlgorithmOutputError("R²AG custom model must return a final answer")

        conditioned_blocks = tuple(
            ContextBlock(
                block_id=block.block_id,
                text=block.text,
                source=block.source,
                score=block.score,
                metadata={
                    **_safe_candidate_metadata(block),
                    "r2ag_features": feature,
                    "r2ag_feature_names": R2AG_FEATURE_NAMES,
                    "r2ag_feature_schema_revision": self.artifacts.feature_schema_revision,
                    "r2ag_candidate_index": index,
                },
            )
            for index, (block, feature) in enumerate(zip(state.context, features, strict=True))
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "fixed_candidates_and_reader_input",
                    {
                        "document_ids": candidate_ids,
                        "base_model_revision": self.artifacts.base_model_revision,
                        "checkpoint_num_documents": self.controls.checkpoint_num_documents,
                        "candidate_documents_sha256": candidate_documents_hash,
                        "candidate_pool_fingerprint_sha256": content_hash(candidate_pool_identity),
                        "reader_instruction_sha256": reader_instruction_hash,
                        "reader_input_sha256": reader_input_hash,
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "retrieval_feature_tensor",
                    {
                        "shape": (len(features), len(R2AG_FEATURE_NAMES)),
                        "features": features,
                        "feature_names": R2AG_FEATURE_NAMES,
                        "feature_source": self.controls.feature_source.value,
                        "feature_tensor_sha256": actual_feature_hash,
                        "feature_schema_revision": self.artifacts.feature_schema_revision,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "custom_model_output",
                    {
                        "r2former_output_sha256": r2former_output_hash,
                        "projection_output_sha256": projection_output_hash,
                        "r_token_count": r_token_count,
                        "r_token_positions": r_token_positions,
                        "answer": answer,
                        "answer_is_intervention_input": False,
                    },
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 1,
                "embedding_calls": response.usage.embedding_calls
                + (1 if self.controls.feature_source is R2AGFeatureSource.EMBEDDER else 0),
                "conditioned_documents": len(candidate_ids),
                "retrieval_feature_values": len(candidate_ids) * len(R2AG_FEATURE_NAMES),
            },
        )
        treatment_identity = content_hash(
            {
                "upstream_commit": R2AG_UPSTREAM_COMMIT,
                "release_id": self.release_id,
                "artifacts": artifact_mapping,
                "controls": asdict(self.controls),
                "candidate_pool": candidate_pool_identity,
                "candidate_documents_sha256": candidate_documents_hash,
                "worker": worker_identity,
                "fixture_mode": fixture_mode,
                "execution_evidence": execution_evidence,
                **feature_dependencies,
            }
        )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=answer,
            evidence=conditioned_blocks,
            trace=trace,
            implementation_identity=(
                f"controlled_source_informed:yefd/RRAG@{R2AG_UPSTREAM_COMMIT}"
            ),
            profile_kind="factorial_adapter",
            metadata={
                "implementation_origin": "controlled_source_informed_with_verified_worker",
                "paper_checkpoint_parity": False,
                "fixture_mode": fixture_mode,
                "execution_evidence": execution_evidence,
                "same_candidate_order": True,
                "requires_custom_hidden_state_model": True,
                "feature_source": self.controls.feature_source.value,
                "feature_tensor_sha256": actual_feature_hash,
                "reader_instruction_sha256": reader_instruction_hash,
                "reader_input_sha256": reader_input_hash,
                "r2former_output_sha256": r2former_output_hash,
                "projection_output_sha256": projection_output_hash,
                "artifacts": artifact_mapping,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "checkpoint_revision": self.artifacts.checkpoint_revision,
                "feature_schema_revision": self.artifacts.feature_schema_revision,
                "base_model_revision": self.artifacts.base_model_revision,
                "r_token_positions": r_token_positions,
                "candidate_pool_fingerprint": candidate_pool_identity,
                "candidate_documents_sha256": candidate_documents_hash,
                "candidate_order_sha256": candidate_order_hash,
                "worker_fingerprint": worker_identity,
                "treatment_identity_sha256": treatment_identity,
                **feature_dependencies,
            },
        )

    def _validate_candidates(self, state: FrozenAgentState) -> tuple[str, ...]:
        if not state.context:
            raise ValidationError("R²AG requires a frozen candidate-document context")
        if len(state.context) > self.controls.max_documents:
            raise ValidationError("R²AG candidate set exceeds max_documents")
        if len(state.context) != self.controls.checkpoint_num_documents:
            raise ValidationError(
                "R²AG candidate count must match checkpoint_num_documents exactly"
            )
        candidate_ids = tuple(block.block_id for block in state.context)
        if any(not value for value in candidate_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValidationError("R²AG candidates require unique non-empty document IDs")
        return candidate_ids

    def _prepare_features(
        self,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> tuple[tuple[FeatureRow, ...] | None, Mapping[str, Any]]:
        source = self.controls.feature_source
        if source is R2AGFeatureSource.WORKER:
            if self.embedder is not None:
                raise ValidationError(
                    "R²AG worker feature_source cannot also receive a constructor embedder"
                )
            return None, freeze_mapping({})
        if source is R2AGFeatureSource.CANDIDATE_METADATA:
            fingerprint = context.run_config.get("r2ag_feature_artifact_fingerprint")
            if not isinstance(fingerprint, Mapping):
                raise ExternalDependencyError(
                    "R²AG candidate_metadata features require "
                    "run_config.r2ag_feature_artifact_fingerprint"
                )
            frozen_fingerprint = _feature_artifact_identity(
                fingerprint,
                label="R²AG feature artifact fingerprint",
            )
            features = _features_from_metadata(state.context)
            _validate_feature_range_and_order(
                features,
                require_rerank_order=self.controls.require_rerank_order,
                tolerance=float(self.controls.numeric_tolerance),
            )
            return features, freeze_mapping({"feature_artifact": frozen_fingerprint})

        configured = self.embedder
        injected = context.dense_embedder
        if configured is not None and injected is not None and configured is not injected:
            raise ValidationError(
                "R²AG embedder is ambiguous; configure it either on the method or context"
            )
        embedder = injected if injected is not None else configured
        if not isinstance(embedder, DenseEmbedder):
            raise ExternalDependencyError(
                "R²AG embedder feature_source requires an injected DenseEmbedder"
            )
        documents = tuple(_document_for_embedding(block) for block in state.context)
        vectors = _validate_vectors(
            embedder.embed((state.query, *documents)),
            expected_count=len(documents) + 1,
        )
        features = _retrieval_features(vectors[0], vectors[1:])
        _validate_feature_range_and_order(
            features,
            require_rerank_order=self.controls.require_rerank_order,
            tolerance=float(self.controls.numeric_tolerance),
        )
        raw_embedder_identity = dependency_fingerprint(embedder, label="R²AG embedder")
        _reject_forbidden_fingerprint_keys(
            raw_embedder_identity,
            label="R²AG embedder fingerprint",
        )
        feature_embedder_identity = freeze_mapping(
            {
                "kind": "dense_embedder",
                "fingerprint_sha256": content_hash(raw_embedder_identity),
            }
        )
        return features, freeze_mapping({"feature_embedder": feature_embedder_identity})


def _build_reader_input(
    state: FrozenAgentState,
    *,
    retrieval_token: str,
    unknown_token: str,
    max_chars: int,
) -> tuple[str, str]:
    values = (state.query, *(block.text for block in state.context))
    if any(retrieval_token in value or unknown_token in value for value in values):
        raise ValidationError("R²AG query/documents must not contain retrieval marker tokens")
    formatted_documents: list[str] = []
    for index, block in enumerate(state.context, start=1):
        title = block.metadata.get("title", "")
        if title is None:
            title = ""
        if not isinstance(title, str):
            raise ValidationError("R²AG candidate metadata title must be text")
        if retrieval_token in title or unknown_token in title:
            raise ValidationError("R²AG document titles must not contain retrieval marker tokens")
        formatted_documents.append(
            f"[{index}]similarity: {retrieval_token}(Title: {title}) {block.text}"
        )
    instruction = _READER_TEMPLATE.format(
        search_results="\n".join(formatted_documents),
        question=state.query,
    )
    if instruction.count(retrieval_token) != len(state.context):
        raise ValidationError("R²AG reader instruction must contain one <R> marker per document")
    reader_input = _INSTRUCTION_WRAPPER.format(instruction=instruction).replace(
        retrieval_token,
        unknown_token,
    )
    if reader_input.count(unknown_token) != len(state.context):
        raise ValidationError("R²AG model input must contain one unknown marker per document")
    if len(reader_input) > max_chars:
        raise ValidationError("R²AG reader input exceeds max_reader_chars")
    return instruction, reader_input


def _candidate_documents(context: Sequence[ContextBlock]) -> tuple[Mapping[str, Any], ...]:
    """Build the only candidate DTO allowed to cross the worker boundary.

    Arbitrary ``ContextBlock.metadata`` is intentionally not copied.  In
    particular, benchmark labels such as ``is_supporting`` and ``gold_answer``
    cannot reach the neural worker through this path.
    """

    rows: list[Mapping[str, Any]] = []
    for index, block in enumerate(context):
        title = block.metadata.get("title", "")
        corpus_revision = block.metadata.get("corpus_revision", "")
        if title is None:
            title = ""
        if corpus_revision is None:
            corpus_revision = ""
        if not isinstance(title, str) or not isinstance(corpus_revision, str):
            raise ValidationError(f"R²AG candidate {index} title/corpus_revision must be text")
        score = block.score
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValidationError(f"R²AG candidate {index} score must be finite or None")
        rows.append(
            freeze_mapping(
                {
                    "document_id": block.block_id,
                    "text": block.text,
                    "title": title,
                    "source": block.source,
                    "score": None if score is None else float(score),
                    "corpus_revision": corpus_revision,
                }
            )
        )
    return tuple(rows)


def _safe_candidate_metadata(block: ContextBlock) -> Mapping[str, str]:
    """Retain only non-oracle provenance in the materialized evidence."""

    title = block.metadata.get("title", "")
    corpus_revision = block.metadata.get("corpus_revision", "")
    if title is None:
        title = ""
    if corpus_revision is None:
        corpus_revision = ""
    if not isinstance(title, str) or not isinstance(corpus_revision, str):
        raise ValidationError("R²AG title/corpus_revision metadata must be text")
    return freeze_mapping(
        {
            "title": title,
            "corpus_revision": corpus_revision,
        }
    )


def _candidate_pool_fingerprint(
    context: InterventionContext,
    *,
    candidate_documents: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    value = context.run_config.get("candidate_pool_fingerprint")
    if value is None:
        raise ExternalDependencyError(
            "R²AG requires run_config.candidate_pool_fingerprint for every feature source"
        )
    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_POOL_KEYS:
        raise ValidationError(
            "R²AG candidate_pool_fingerprint must contain exactly retriever, corpus, "
            "index, top_k, order, and content_sha256"
        )
    _reject_forbidden_fingerprint_keys(value, label="R²AG candidate pool fingerprint")
    retriever = _repository_identity(
        value.get("retriever"),
        label="R²AG candidate_pool_fingerprint.retriever",
    )
    corpus = _repository_identity(
        value.get("corpus"),
        label="R²AG candidate_pool_fingerprint.corpus",
    )
    index = _index_identity(
        value.get("index"),
        label="R²AG candidate_pool_fingerprint.index",
    )
    top_k = value.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValidationError("R²AG candidate_pool_fingerprint.top_k must be positive")
    if top_k != len(candidate_documents):
        raise ValidationError(
            "R²AG candidate_pool_fingerprint.top_k must equal the fixed candidate count"
        )
    if value.get("order") != _CANDIDATE_ORDER:
        raise ValidationError("R²AG candidate_pool_fingerprint.order must be rerank_desc")
    expected_content = content_hash(tuple(candidate_documents))
    if value.get("content_sha256") != expected_content:
        raise ValidationError("R²AG candidate_pool_fingerprint content hash mismatch")
    try:
        return freeze_mapping(
            {
                "retriever": retriever,
                "corpus": corpus,
                "index": index,
                "top_k": top_k,
                "order": _CANDIDATE_ORDER,
                "content_sha256": expected_content,
            }
        )
    except TypeError as exc:
        raise ValidationError("R²AG candidate_pool_fingerprint is not canonical") from exc


def _repository_identity(value: object, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping")
    _reject_forbidden_fingerprint_keys(value, label=label)
    keys = set(value)
    allowed = _REPOSITORY_IDENTITY_REQUIRED | _REPOSITORY_IDENTITY_OPTIONAL
    if not _REPOSITORY_IDENTITY_REQUIRED.issubset(keys) or not keys.issubset(allowed):
        raise ValidationError(
            f"{label} must contain repository/revision and optional manifest_sha256 only"
        )
    repository = value.get("repository")
    if not isinstance(repository, str) or not repository.strip():
        raise ValidationError(f"{label}.repository must be non-empty text")
    revision = _pinned_revision(value.get("revision"), label=f"{label}.revision")
    identity: dict[str, str] = {
        "repository": repository.strip(),
        "revision": revision,
    }
    if "manifest_sha256" in value:
        identity["manifest_sha256"] = _sha256(
            value.get("manifest_sha256"),
            label=f"{label}.manifest_sha256",
        )
    return freeze_mapping(identity)


def _index_identity(value: object, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping")
    _reject_forbidden_fingerprint_keys(value, label=label)
    keys = set(value)
    allowed = _INDEX_IDENTITY_REQUIRED | _INDEX_IDENTITY_OPTIONAL
    if not _INDEX_IDENTITY_REQUIRED.issubset(keys) or not keys.issubset(allowed):
        raise ValidationError(
            f"{label} must contain algorithm/revision and optional manifest_sha256 only"
        )
    algorithm = value.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValidationError(f"{label}.algorithm must be non-empty text")
    revision = _pinned_revision(value.get("revision"), label=f"{label}.revision")
    identity: dict[str, str] = {
        "algorithm": algorithm.strip(),
        "revision": revision,
    }
    if "manifest_sha256" in value:
        identity["manifest_sha256"] = _sha256(
            value.get("manifest_sha256"),
            label=f"{label}.manifest_sha256",
        )
    return freeze_mapping(identity)


def _feature_artifact_identity(value: object, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must contain exactly repository, revision, and sha256")
    _reject_forbidden_fingerprint_keys(value, label=label)
    if set(value) != _FEATURE_ARTIFACT_KEYS:
        raise ValidationError(f"{label} must contain exactly repository, revision, and sha256")
    repository = value.get("repository")
    if not isinstance(repository, str) or not repository.strip():
        raise ValidationError(f"{label}.repository must be non-empty text")
    return freeze_mapping(
        {
            "repository": repository.strip(),
            "revision": _pinned_revision(value.get("revision"), label=f"{label}.revision"),
            "sha256": _sha256(value.get("sha256"), label=f"{label}.sha256"),
        }
    )


def _reject_forbidden_fingerprint_keys(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{label} keys must be strings")
            terms = tuple(re.findall(r"[a-z0-9]+", key.lower()))
            compact = "".join(terms)
            if any(term in _FORBIDDEN_FINGERPRINT_TERMS for term in (*terms, compact)):
                raise ValidationError(f"{label} contains forbidden metadata key {key!r}")
            _reject_forbidden_fingerprint_keys(nested, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_forbidden_fingerprint_keys(nested, label=label)


def _document_for_embedding(block: ContextBlock) -> str:
    title = block.metadata.get("title", "")
    if title is None:
        title = ""
    if not isinstance(title, str):
        raise ValidationError("R²AG candidate metadata title must be text")
    return f"Title: {title}\n{block.text}"


def _validate_vectors(
    value: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> tuple[Vector, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlgorithmOutputError("R²AG embedder returned a non-sequence")
    try:
        vectors = tuple(tuple(float(item) for item in row) for row in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlgorithmOutputError("R²AG embedder returned non-numeric vectors") from exc
    if len(vectors) != expected_count:
        raise AlgorithmOutputError(
            f"R²AG embedder returned {len(vectors)} vectors; expected {expected_count}"
        )
    if not vectors or not vectors[0]:
        raise AlgorithmOutputError("R²AG embeddings require a non-zero dimension")
    dimension = len(vectors[0])
    for vector in vectors:
        if len(vector) != dimension:
            raise AlgorithmOutputError("R²AG embeddings have inconsistent dimensions")
        if not all(math.isfinite(item) for item in vector):
            raise AlgorithmOutputError("R²AG embeddings must contain only finite values")
        if _norm(vector) == 0:
            raise AlgorithmOutputError("R²AG embeddings must not contain zero vectors")
    return vectors


def compute_r2ag_retrieval_features(
    query: Sequence[float],
    documents: Sequence[Sequence[float]],
) -> tuple[Mapping[str, float], ...]:
    """Compute the three retrieval features from the pinned upstream formulas.

    The public wrapper keeps common-support runtimes on the same reviewed
    implementation used by the controlled worker while returning named rows.
    """

    try:
        query_vector = tuple(float(value) for value in query)
        document_vectors = tuple(
            tuple(float(value) for value in document) for document in documents
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("R²AG retrieval features require numeric vectors") from exc
    if not query_vector or len(document_vectors) < 2:
        raise ValidationError("R²AG retrieval features require a query and two documents")
    dimension = len(query_vector)
    if any(
        len(vector) != dimension
        or not all(math.isfinite(value) for value in vector)
        or _norm(vector) == 0
        for vector in (query_vector, *document_vectors)
    ):
        raise ValidationError("R²AG retrieval feature vectors are invalid")
    rows = _retrieval_features(query_vector, document_vectors)
    return tuple(freeze_mapping(dict(zip(R2AG_FEATURE_NAMES, row, strict=True))) for row in rows)


def _retrieval_features(query: Vector, documents: Sequence[Vector]) -> tuple[FeatureRow, ...]:
    if len(documents) < 2:
        raise ValidationError("R²AG feature extraction requires at least two documents")
    rerank = tuple(_cosine(query, document) for document in documents)
    neighbor: list[float] = []
    for index, document in enumerate(documents):
        if index == 0:
            value = _cosine(document, documents[1])
        elif index == len(documents) - 1:
            value = _cosine(documents[-2], document)
        else:
            value = (
                _cosine(document, documents[index - 1]) + _cosine(document, documents[index + 1])
            ) / 2
        neighbor.append(value)

    precedent = [1.0]
    for index in range(1, len(documents)):
        prior_scores = rerank[:index]
        maximum = max(prior_scores)
        unnormalized = tuple(math.exp(score - maximum) for score in prior_scores)
        denominator = sum(unnormalized)
        weights = tuple(value / denominator for value in unnormalized)
        weighted_precedent = tuple(
            sum(weights[row] * documents[row][column] for row in range(index))
            for column in range(len(documents[0]))
        )
        precedent.append(_cosine(documents[index], weighted_precedent))
    return tuple(
        (rerank[index], neighbor[index], precedent[index]) for index in range(len(documents))
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        raise AlgorithmOutputError("R²AG cosine similarity received a zero vector")
    value = sum(a * b for a, b in zip(left, right, strict=True)) / denominator
    if not math.isfinite(value):
        raise AlgorithmOutputError("R²AG cosine similarity is not finite")
    return max(-1.0, min(1.0, value))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _features_from_metadata(context: Sequence[ContextBlock]) -> tuple[FeatureRow, ...]:
    rows: list[FeatureRow] = []
    official_names = (
        "rerank_score",
        "rerank_nb_score",
        "rerank_precedent_score",
    )
    for index, block in enumerate(context):
        metadata = block.metadata
        if all(name in metadata for name in official_names):
            raw = tuple(metadata[name] for name in official_names)
        else:
            normalized = metadata.get("r2ag_features")
            if not isinstance(normalized, Mapping) or set(normalized) != set(R2AG_FEATURE_NAMES):
                raise ValidationError(
                    f"R²AG candidate {index} lacks the official three retrieval features"
                )
            raw = tuple(normalized[name] for name in R2AG_FEATURE_NAMES)
        try:
            row = tuple(
                _metadata_number(value, label=f"R²AG candidate {index} feature") for value in raw
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"R²AG candidate {index} has invalid retrieval features") from exc
        rows.append((row[0], row[1], row[2]))
    return tuple(rows)


def _metadata_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _parse_features(value: object, *, expected_count: int) -> tuple[FeatureRow, ...]:
    rows = mapping_sequence(value, label="R²AG features")
    if len(rows) != expected_count:
        raise AlgorithmOutputError("R²AG feature row count does not match candidates")
    parsed: list[FeatureRow] = []
    for index, row in enumerate(rows):
        if set(row) != set(R2AG_FEATURE_NAMES):
            raise AlgorithmOutputError(
                "R²AG feature rows must contain rerank, neighbor, and precedent"
            )
        values = tuple(
            finite_number(row[name], label=f"R²AG feature {index}.{name}")
            for name in R2AG_FEATURE_NAMES
        )
        parsed.append((values[0], values[1], values[2]))
    return tuple(parsed)


def _feature_mappings(features: Sequence[FeatureRow]) -> tuple[Mapping[str, float], ...]:
    return tuple(
        freeze_mapping(dict(zip(R2AG_FEATURE_NAMES, row, strict=True))) for row in features
    )


def _validate_feature_range_and_order(
    features: Sequence[FeatureRow],
    *,
    require_rerank_order: bool,
    tolerance: float,
) -> None:
    for index, row in enumerate(features):
        for name, value in zip(R2AG_FEATURE_NAMES, row, strict=True):
            if value < -1.0 - tolerance or value > 1.0 + tolerance:
                raise AlgorithmOutputError(
                    f"R²AG feature {index}.{name} is outside cosine range [-1, 1]"
                )
    if require_rerank_order and any(
        features[index][0] + tolerance < features[index + 1][0]
        for index in range(len(features) - 1)
    ):
        raise AlgorithmOutputError(
            "R²AG fixed candidates must already be ordered by descending rerank score"
        )


def _require_same_features(
    expected: Sequence[FeatureRow],
    returned: Sequence[FeatureRow],
    *,
    tolerance: float,
) -> None:
    for row_index, (left, right) in enumerate(zip(expected, returned, strict=True)):
        for column_index, (expected_value, returned_value) in enumerate(
            zip(left, right, strict=True)
        ):
            if not math.isclose(
                expected_value,
                returned_value,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                name = R2AG_FEATURE_NAMES[column_index]
                raise AlgorithmOutputError(f"R²AG worker changed feature {row_index}.{name}")


def _require_echo_hash(payload: Mapping[str, Any], key: str, expected: str) -> None:
    if required_text(payload, key, label="R²AG response") != expected:
        raise AlgorithmOutputError(f"R²AG {key} mismatch")


def _sha256_response(payload: Mapping[str, Any], key: str) -> str:
    value = required_text(payload, key, label="R²AG response")
    if not _HEX64.fullmatch(value):
        raise AlgorithmOutputError(f"R²AG response {key} must be a SHA-256 digest")
    return value


def _validate_runtime_attestation(
    value: object,
    *,
    artifacts: R2AGArtifactIdentity,
    expected_num_k: int,
    allow_fixture: bool,
) -> tuple[bool, str]:
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("R²AG response requires runtime_attestation")
    fixture_mode = value.get("fixture_mode")
    if not isinstance(fixture_mode, bool):
        raise AlgorithmOutputError("R²AG runtime attestation fixture_mode must be boolean")
    if fixture_mode and not allow_fixture:
        raise AlgorithmOutputError("R²AG production host rejects fixture workers by default")
    expected: Mapping[str, object] = {
        "source_commit": R2AG_UPSTREAM_COMMIT,
        "source_manifest_sha256": artifacts.source_manifest_sha256,
        "compatibility_manifest_sha256": artifacts.compatibility_manifest_sha256,
        "checkpoint_config_sha256": artifacts.checkpoint_config_sha256,
        "rformer_weights_sha256": artifacts.rformer_weights_sha256,
        "projection_weights_sha256": artifacts.projection_weights_sha256,
        "base_model_manifest_sha256": artifacts.base_model_manifest_sha256,
        "tokenizer_manifest_sha256": artifacts.tokenizer_manifest_sha256,
        "feature_schema_sha256": artifacts.feature_schema_sha256,
        "worker_config_num_k": expected_num_k,
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
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AlgorithmOutputError(f"R²AG runtime attestation mismatch for {key}")
    evidence_level = value.get("execution_evidence")
    expected_level = (
        "fixture_source_execution" if fixture_mode else "pinned_source_checkpoint_execution"
    )
    if evidence_level != expected_level:
        raise AlgorithmOutputError("R²AG runtime execution evidence does not match fixture mode")
    return fixture_mode, expected_level


def _validate_usage(
    usage: Any,
    *,
    remaining_budget: Mapping[str, int | None],
    controls: R2AGControls,
) -> None:
    if usage.model_calls != 1:
        raise AlgorithmOutputError("R²AG worker must account for exactly one reader model call")
    if usage.retrieval_calls != 0 or usage.tool_calls != 0:
        raise AlgorithmOutputError("R²AG worker must not perform retrieval or tool calls")
    expected_worker_embedding_calls = (
        1 if controls.feature_source is R2AGFeatureSource.WORKER else 0
    )
    if usage.embedding_calls != expected_worker_embedding_calls:
        raise AlgorithmOutputError(
            "R²AG worker embedding-call accounting disagrees with feature_source"
        )
    if usage.model_calls > remaining_budget["model_calls"]:
        raise AlgorithmOutputError("R²AG worker exceeded sent model-call budget")
    if usage.output_tokens > min(
        controls.max_output_tokens,
        remaining_budget["output_tokens"],
    ):
        raise AlgorithmOutputError("R²AG worker exceeded sent output-token budget")
    context_limit = remaining_budget["context_tokens"]
    if context_limit is not None and usage.input_tokens > context_limit:
        raise AlgorithmOutputError("R²AG worker exceeded sent context-token budget")


def _remaining_budget(
    state: FrozenAgentState,
    context: InterventionContext,
) -> Mapping[str, int | None]:
    usage = None
    ledger = context.budget_ledger
    if ledger is not None:
        ledger_budget = getattr(ledger, "budget", None)
        usage = getattr(ledger, "usage", None)
        if ledger_budget != state.budget or usage is None:
            raise ValidationError("R²AG received an incompatible budget ledger")

    def consumed(name: str) -> int:
        value = 0 if usage is None else getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(f"R²AG budget usage {name} is invalid")
        return value

    remaining_context = None
    if state.budget.max_context_tokens is not None:
        remaining_context = max(
            0,
            state.budget.max_context_tokens - consumed("input_tokens"),
        )
    return freeze_mapping(
        {
            "model_calls": max(
                0,
                state.budget.max_model_calls - consumed("model_calls"),
            ),
            "output_tokens": max(
                0,
                state.budget.max_output_tokens - consumed("output_tokens"),
            ),
            "context_tokens": remaining_context,
        }
    )


__all__ = [
    "R2AG_FEATURE_NAMES",
    "R2AG_PROTOCOL",
    "R2AG_UPSTREAM_COMMIT",
    "R2AGArtifactIdentity",
    "R2AGControlledWorker",
    "R2AGControls",
    "R2AGFeatureSource",
    "compute_r2ag_retrieval_features",
]
