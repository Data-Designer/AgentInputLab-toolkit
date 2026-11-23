"""Controlled Adaptive-RAG router-parity reimplementation.

The pinned upstream implementation trains a T5-large complexity classifier and,
at inference time, takes an argmax over the first decoder-step logits for the
tokens ``A``, ``B``, and ``C``.  Those labels select no retrieval, one retrieval,
and IRCoT respectively.  The legacy stack depends on a large, pinned
Elasticsearch/LLM environment, so this release preserves the published router
decision rule while executing explicitly controlled, replaceable arm
implementations in an isolated worker.  It is not the upstream script or an
upstream arm-parity claim.  This module validates the selected controlled arm,
evaluation-label separation, component closure, budgets, and retrieval
trajectory.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CompositeResult,
    ContextBlock,
    ExecutionTrace,
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
    finite_number,
    mapping_sequence,
    optional_text,
    required_text,
    resolve_worker,
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

from .replay import ADAPTIVE_RAG_UPSTREAM_COMMIT, AdaptiveRAGStrategy

_HEX64 = re.compile(r"[0-9a-f]{64}")
_ROUTE_ORDER = ("A", "B", "C")
_ROUTE_TO_STRATEGY = {
    "A": AdaptiveRAGStrategy.NO_RETRIEVAL,
    "B": AdaptiveRAGStrategy.SINGLE_RETRIEVAL,
    "C": AdaptiveRAGStrategy.IRCOT,
}
ADAPTIVE_RAG_UPSTREAM_TREE = "8dadc754562d61d5d6e6e3ee2fc3aa4481763320"
ADAPTIVE_RAG_COMPATIBILITY_MANIFEST_SHA256 = (
    "85299456b7f858734c34de84fe4625e0a014651242001d7242ca33b260be7382"
)
ADAPTIVE_RAG_SOURCE_CLOSURE_SHA256 = (
    "43677c9862dd16a69aafadab36a7ac9ca9cdfb037c8854321e64a2a3df14597c"
)
_SUPERVISOR_EVENT_SCHEMA = "agentinputlab.adaptive-rag.supervisor-event.v1"
_EXECUTION_EVIDENCE_SCHEMA = "agentinputlab.adaptive-rag.execution-evidence.v1"
_ALLOWED_DOCUMENT_METADATA = frozenset({"source_uri", "chunk_index", "language", "content_sha256"})
_EVENT_USAGE_FIELDS = (
    "model_calls",
    "retrieval_calls",
    "tool_calls",
    "embedding_calls",
    "unpriced_model_calls",
    "input_tokens",
    "output_tokens",
    "cpu_seconds",
    "accelerator_seconds",
    "cost_usd",
    "compute_cost_usd",
)
_EVENT_RESERVATION_FIELDS = frozenset(
    {
        "model_calls",
        "retrieval_calls",
        "estimated_input_tokens",
        "max_output_tokens",
    }
)


def _pinned_revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"Adaptive-RAG {label} must be a non-empty revision")
    revision = value.strip()
    if revision.lower() in {"default", "head", "latest", "main", "master"}:
        raise ValidationError(f"Adaptive-RAG {label} must be immutable")
    return revision


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValidationError(f"Adaptive-RAG {label} must be a lowercase SHA-256 digest")
    return value


def _validated_document_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only non-oracle corpus provenance exposed to the generator."""

    unknown = set(value) - _ALLOWED_DOCUMENT_METADATA
    if unknown:
        raise AlgorithmOutputError(
            "Adaptive-RAG document metadata contains non-allowlisted fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"source_uri", "language"}:
            if not isinstance(item, str) or not item.strip():
                raise AlgorithmOutputError(
                    f"Adaptive-RAG document metadata {key} must be non-empty text"
                )
            normalized[key] = item.strip()
        elif key == "chunk_index":
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise AlgorithmOutputError(
                    "Adaptive-RAG document metadata chunk_index must be non-negative"
                )
            normalized[key] = item
        else:
            if not isinstance(item, str) or _HEX64.fullmatch(item) is None:
                raise AlgorithmOutputError(
                    "Adaptive-RAG document metadata content_sha256 must be a SHA-256 digest"
                )
            normalized[key] = item
    return normalized


@dataclass(frozen=True, slots=True)
class AdaptiveRAGArtifactIdentity:
    """Pinned classifier, generator, retrieval, and prompt identities.

    The base model and retriever are deliberately configurable, but every
    replacement produces a different identity hash.  This prevents a router
    trained for one three-arm bundle from being silently reused with another.
    """

    router_checkpoint: str
    worker_fingerprint_sha256: str
    router_revision: str
    router_snapshot_sha256: str
    router_weights_sha256: str
    router_tokenizer_sha256: str
    base_model_revision: str
    generator_artifact_sha256: str
    generator_config_sha256: str
    retriever_revision: str
    retriever_implementation_sha256: str
    retriever_config_sha256: str
    corpus_revision: str
    corpus_manifest_sha256: str
    index_revision: str
    index_manifest_sha256: str
    no_retrieval_prompt_sha256: str
    single_retrieval_prompt_sha256: str
    ircot_prompt_sha256: str
    upstream_commit: str = ADAPTIVE_RAG_UPSTREAM_COMMIT

    def __post_init__(self) -> None:
        if self.upstream_commit != ADAPTIVE_RAG_UPSTREAM_COMMIT:
            raise ValidationError("Adaptive-RAG artifacts target an unsupported upstream commit")
        for name in (
            "router_checkpoint",
            "router_revision",
            "base_model_revision",
            "retriever_revision",
            "corpus_revision",
            "index_revision",
        ):
            object.__setattr__(
                self,
                name,
                _pinned_revision(getattr(self, name), label=name),
            )
        for name in (
            "worker_fingerprint_sha256",
            "router_snapshot_sha256",
            "router_weights_sha256",
            "router_tokenizer_sha256",
            "generator_artifact_sha256",
            "generator_config_sha256",
            "retriever_implementation_sha256",
            "retriever_config_sha256",
            "corpus_manifest_sha256",
            "index_manifest_sha256",
            "no_retrieval_prompt_sha256",
            "single_retrieval_prompt_sha256",
            "ircot_prompt_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), label=name))

    def as_mapping(self) -> Mapping[str, Any]:
        return freeze_mapping(asdict(self))

    @property
    def identity_sha256(self) -> str:
        return content_hash(self.as_mapping())


@dataclass(frozen=True, slots=True)
class AdaptiveRAGControls:
    """Explicit resource and retrieval settings for the three strategy arms."""

    single_retrieval_top_k: int = 15
    ircot_per_step_top_k: int = 6
    ircot_max_steps: int = 8
    max_total_documents: int = 64
    max_output_tokens: int = 4096
    ircot_reset_policy: str = "upstream_native"
    deduplicate_documents: bool = True

    def __post_init__(self) -> None:
        for name in (
            "single_retrieval_top_k",
            "ircot_per_step_top_k",
            "ircot_max_steps",
            "max_total_documents",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"Adaptive-RAG {name} must be a positive integer")
        if self.ircot_reset_policy not in {
            "upstream_native",
            "never",
            "replace_each_step",
        }:
            raise ValidationError("Adaptive-RAG ircot_reset_policy is unsupported")


@dataclass(frozen=True, slots=True)
class AdaptiveRAGDocument:
    document_id: str
    text: str
    title: str
    corpus_revision: str
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.text:
            raise AlgorithmOutputError("Adaptive-RAG documents require ID and text")
        if not self.corpus_revision.strip():
            raise AlgorithmOutputError("Adaptive-RAG documents require corpus_revision")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise AlgorithmOutputError("Adaptive-RAG document score must be finite or None")
        if not isinstance(self.metadata, Mapping):
            raise AlgorithmOutputError("Adaptive-RAG document metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(_validated_document_metadata(self.metadata)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AdaptiveRAGDocument:
        score_value = value.get("score")
        score = (
            None
            if score_value is None
            else finite_number(score_value, label="Adaptive-RAG document score")
        )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise AlgorithmOutputError("Adaptive-RAG document metadata must be a mapping")
        return cls(
            document_id=required_text(value, "document_id", label="Adaptive-RAG document"),
            text=required_text(value, "text", label="Adaptive-RAG document"),
            title=optional_text(value, "title", label="Adaptive-RAG document"),
            corpus_revision=required_text(
                value,
                "corpus_revision",
                label="Adaptive-RAG document",
            ),
            score=score,
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRAGInitialRetrieval:
    query: str
    document_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AdaptiveRAGInitialRetrieval:
        return cls(
            query=required_text(value, "query", label="Adaptive-RAG initial retrieval"),
            document_ids=string_tuple(
                value.get("document_ids"),
                label="Adaptive-RAG initial retrieval document_ids",
                allow_empty=False,
                unique=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRAGIRCoTStep:
    index: int
    cot_step: str
    latest_cot_sentence: str
    retrieval_query: str
    retrieved_document_ids: tuple[str, ...]
    new_document_ids: tuple[str, ...]
    active_document_ids: tuple[str, ...]
    reset_applied: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AdaptiveRAGIRCoTStep:
        index = value.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise AlgorithmOutputError("Adaptive-RAG IRCoT step index must be non-negative")
        reset = value.get("reset_applied", False)
        if not isinstance(reset, bool):
            raise AlgorithmOutputError("Adaptive-RAG IRCoT reset_applied must be boolean")
        cot_step = required_text(value, "cot_step", label="Adaptive-RAG IRCoT step")
        latest = required_text(
            value,
            "latest_cot_sentence",
            label="Adaptive-RAG IRCoT step",
        )
        query = required_text(
            value,
            "retrieval_query",
            label="Adaptive-RAG IRCoT step",
        )
        if not cot_step.rstrip().endswith(latest):
            raise AlgorithmOutputError("Adaptive-RAG IRCoT latest sentence must terminate cot_step")
        if query.strip() != latest.strip():
            raise AlgorithmOutputError(
                "Adaptive-RAG IRCoT retrieval query must equal the latest CoT sentence"
            )
        return cls(
            index=index,
            cot_step=cot_step,
            latest_cot_sentence=latest,
            retrieval_query=query,
            retrieved_document_ids=string_tuple(
                value.get("retrieved_document_ids"),
                label="Adaptive-RAG IRCoT retrieved_document_ids",
                unique=True,
            ),
            new_document_ids=string_tuple(
                value.get("new_document_ids"),
                label="Adaptive-RAG IRCoT new_document_ids",
                unique=True,
            ),
            active_document_ids=string_tuple(
                value.get("active_document_ids"),
                label="Adaptive-RAG IRCoT active_document_ids",
                unique=True,
            ),
            reset_applied=reset,
        )


@dataclass(slots=True)
class AdaptiveRAGControlledWorker:
    """Run classifier inference and its selected QA arm in one pinned worker."""

    artifacts: AdaptiveRAGArtifactIdentity
    client: WorkerClient | None = None
    controls: AdaptiveRAGControls = AdaptiveRAGControls()
    method_id: str = "adaptive_rag"
    release_id: str = "acquire.adaptive_rag.controlled_reimplementation"
    family: SemanticFamily = SemanticFamily.ACQUIRE
    expected_source_tree: str = ADAPTIVE_RAG_UPSTREAM_TREE
    expected_compatibility_manifest_sha256: str = ADAPTIVE_RAG_COMPATIBILITY_MANIFEST_SHA256
    expected_source_closure_sha256: str = ADAPTIVE_RAG_SOURCE_CLOSURE_SHA256
    allow_fixture_runtime: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, AdaptiveRAGArtifactIdentity):
            raise ValidationError("Adaptive-RAG requires AdaptiveRAGArtifactIdentity")
        if not isinstance(self.controls, AdaptiveRAGControls):
            raise ValidationError("Adaptive-RAG controls must be AdaptiveRAGControls")
        for name in (
            "expected_source_tree",
            "expected_compatibility_manifest_sha256",
            "expected_source_closure_sha256",
        ):
            value = getattr(self, name)
            pattern = re.fullmatch(
                r"[0-9a-f]{40}" if name == "expected_source_tree" else r"[0-9a-f]{64}",
                value,
            )
            if pattern is None:
                raise ValidationError(f"Adaptive-RAG {name} has an invalid digest")
        if not isinstance(self.allow_fixture_runtime, bool):
            raise ValidationError("Adaptive-RAG allow_fixture_runtime must be boolean")

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        if not state.query.strip():
            raise ValidationError("Adaptive-RAG requires a non-empty question")
        if state.instruction.strip() or state.messages or state.context:
            raise ValidationError(
                "Adaptive-RAG controlled reimplementation accepts a question-only state; "
                "instruction, messages, or preloaded context require a separate profile"
            )
        if state.budget.max_cost_usd is not None:
            raise BudgetExceededError(
                "Adaptive-RAG external execution has no pre-call cost upper bound"
            )
        remaining_budget = _remaining_budget(state, context)
        if remaining_budget["model_calls"] < 2:
            raise BudgetExceededError(
                "Adaptive-RAG requires at least classifier and answer model calls"
            )
        if remaining_budget["output_tokens"] < 1:
            raise BudgetExceededError("Adaptive-RAG has no remaining output-token budget")

        worker = resolve_worker(self.client, context, label="Adaptive-RAG")
        worker_identity = worker_fingerprint(worker, label="Adaptive-RAG")
        if content_hash(worker_identity) != self.artifacts.worker_fingerprint_sha256:
            raise ValidationError(
                "Adaptive-RAG worker fingerprint does not match the locked artifact bundle"
            )
        artifact_mapping = self.artifacts.as_mapping()
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=ADAPTIVE_RAG_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "task_id": state.task_id,
                "question": state.query,
                "artifacts": artifact_mapping,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "controls": asdict(self.controls),
                "remaining_budget": remaining_budget,
                "seed": context.seed,
                "execution_contract": {
                    "input_view": "question_only",
                    "route_labels": list(_ROUTE_ORDER),
                    "route_rule": "first_decoder_step_logit_argmax_A_first",
                    "strategy_map": {
                        "A": AdaptiveRAGStrategy.NO_RETRIEVAL.value,
                        "B": AdaptiveRAGStrategy.SINGLE_RETRIEVAL.value,
                        "C": AdaptiveRAGStrategy.IRCOT.value,
                    },
                    "evaluation_label_access": "none",
                    "selected_arm_only": True,
                    "fallback_allowed": False,
                    "arm_execution_semantics": "controlled_reimplementation",
                    "upstream_router_parity": True,
                    "supervisor_event_schema": _SUPERVISOR_EVENT_SCHEMA,
                },
            },
            dependency_fingerprints={
                "worker": worker_identity,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
            },
            minimum_reservation={"model_calls": 2},
            protocol_version="agentinputlab.adaptive-rag.v1",
        )
        response = call_worker(worker, request)
        payload = response.payload
        _validate_identity(payload, artifacts=self.artifacts)
        if required_text(payload, "question", label="Adaptive-RAG response") != state.query:
            raise AlgorithmOutputError("Adaptive-RAG worker changed the classifier question")

        route_logits = _parse_route_logits(payload.get("route_logits"))
        expected_label = max(_ROUTE_ORDER, key=lambda label: route_logits[label])
        route_label = required_text(payload, "route_label", label="Adaptive-RAG response")
        if route_label not in _ROUTE_TO_STRATEGY:
            raise AlgorithmOutputError("Adaptive-RAG route_label must be A, B, or C")
        if route_label != expected_label:
            raise AlgorithmOutputError("Adaptive-RAG route does not match first-step logit argmax")
        strategy = _ROUTE_TO_STRATEGY[route_label]
        selected_strategy = required_text(
            payload,
            "selected_strategy",
            label="Adaptive-RAG response",
        )
        if selected_strategy != strategy.value:
            raise AlgorithmOutputError(
                "Adaptive-RAG selected strategy does not match classifier route"
            )
        answer = required_text(payload, "answer", label="Adaptive-RAG response")
        documents = tuple(
            AdaptiveRAGDocument.from_mapping(item)
            for item in mapping_sequence(
                payload.get("documents", ()),
                label="Adaptive-RAG documents",
            )
        )
        document_ids = tuple(document.document_id for document in documents)
        if len(document_ids) != len(set(document_ids)):
            raise AlgorithmOutputError("Adaptive-RAG returned duplicate document records")
        if any(
            document.corpus_revision != self.artifacts.corpus_revision for document in documents
        ):
            raise AlgorithmOutputError("Adaptive-RAG returned a corpus revision mismatch")

        initial_retrieval = _parse_initial_retrieval(payload.get("initial_retrieval"))
        ircot_steps = tuple(
            AdaptiveRAGIRCoTStep.from_mapping(item)
            for item in mapping_sequence(
                payload.get("ircot_steps", ()),
                label="Adaptive-RAG IRCoT steps",
            )
        )
        active_document_ids = string_tuple(
            payload.get("active_document_ids", ()),
            label="Adaptive-RAG active_document_ids",
            unique=True,
        )
        _validate_strategy_trajectory(
            strategy=strategy,
            question=state.query,
            documents=documents,
            initial_retrieval=initial_retrieval,
            ircot_steps=ircot_steps,
            active_document_ids=active_document_ids,
            controls=self.controls,
        )
        execution_evidence = _validate_execution_evidence(
            payload.get("execution_evidence"),
            artifacts=self.artifacts,
            controls=self.controls,
            strategy=strategy,
            question=state.query,
            route_logits=route_logits,
            answer=answer,
            initial_retrieval=initial_retrieval,
            ircot_steps=ircot_steps,
            response_usage=response.usage,
            remaining_budget=remaining_budget,
            expected_source_tree=self.expected_source_tree,
            expected_manifest_sha256=self.expected_compatibility_manifest_sha256,
            expected_source_closure_sha256=self.expected_source_closure_sha256,
            allow_fixture_runtime=self.allow_fixture_runtime,
        )
        _validate_usage(
            response.usage,
            remaining_budget=remaining_budget,
            strategy=strategy,
            ircot_steps=ircot_steps,
        )

        by_id = {document.document_id: document for document in documents}
        evidence = tuple(
            _context_from_document(by_id[document_id], strategy=strategy)
            for document_id in active_document_ids
        )
        events: list[TraceEvent] = [
            TraceEvent(
                TracePhase.PREPARE,
                "controlled_three_arm_bundle",
                {
                    "artifact_identity_sha256": self.artifacts.identity_sha256,
                    "worker_fingerprint": worker_identity,
                    "controls": asdict(self.controls),
                    "remaining_budget": remaining_budget,
                    "evaluation_label_access": "none",
                },
            ),
            TraceEvent(
                TracePhase.PROPOSE,
                "complexity_route",
                {
                    "route_label": route_label,
                    "strategy": strategy.value,
                    "route_logits": route_logits,
                    "route_logits_sha256": content_hash(route_logits),
                    "prediction_type": "first_decoder_step_logit_argmax",
                    "router_checkpoint": self.artifacts.router_checkpoint,
                    "router_revision": self.artifacts.router_revision,
                },
            ),
        ]
        if initial_retrieval is not None:
            events.append(
                TraceEvent(
                    TracePhase.OBSERVE,
                    "initial_retrieval",
                    {
                        "query": initial_retrieval.query,
                        "document_ids": initial_retrieval.document_ids,
                    },
                )
            )
        events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "ircot_step",
                {
                    "index": step.index,
                    "cot_step": step.cot_step,
                    "latest_cot_sentence": step.latest_cot_sentence,
                    "retrieval_query": step.retrieval_query,
                    "retrieved_document_ids": step.retrieved_document_ids,
                    "new_document_ids": step.new_document_ids,
                    "active_document_ids": step.active_document_ids,
                    "reset_applied": step.reset_applied,
                },
            )
            for step in ircot_steps
        )
        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "selected_strategy_output",
                {
                    "strategy": strategy.value,
                    "answer": answer,
                    "active_document_ids": active_document_ids,
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(events),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 1,
                "strategy_arms_executed": 1,
                "ircot_steps": len(ircot_steps),
                "active_documents": len(active_document_ids),
            },
        )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=answer,
            evidence=evidence,
            trace=trace,
            implementation_identity=(
                "controlled_reimplementation_with_upstream_router:"
                f"starsuzi/Adaptive-RAG@{ADAPTIVE_RAG_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "controlled_reimplementation",
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "artifacts": artifact_mapping,
                "worker_fingerprint": worker_identity,
                "route_label": route_label,
                "strategy": strategy.value,
                "route_logits": route_logits,
                "route_logits_sha256": content_hash(route_logits),
                "evaluation_label_access": "none",
                "classifier_prediction_type": "first_decoder_step_logit_argmax",
                "active_document_ids": active_document_ids,
                "execution_evidence_kind": execution_evidence["evidence_kind"],
                "source_tree": execution_evidence["source_tree"],
                "source_closure_sha256": execution_evidence["source_closure_sha256"],
                "compatibility_manifest_sha256": execution_evidence[
                    "compatibility_manifest_sha256"
                ],
                "supervisor_event_log_sha256": execution_evidence["event_log_sha256"],
                "controlled_method_execution": True,
                "controlled_upstream_router_parity": True,
                "controlled_online_selected_arm": True,
                "upstream_arm_execution_parity": False,
                "official_execution_parity": False,
                "official_parity_claimed": False,
            },
        )


def _validate_identity(
    payload: Mapping[str, Any],
    *,
    artifacts: AdaptiveRAGArtifactIdentity,
) -> None:
    returned = payload.get("artifacts")
    if not isinstance(returned, Mapping) or dict(returned) != dict(artifacts.as_mapping()):
        raise AlgorithmOutputError("Adaptive-RAG worker artifact identity mismatch")
    identity = required_text(
        payload,
        "artifact_identity_sha256",
        label="Adaptive-RAG response",
    )
    if identity != artifacts.identity_sha256:
        raise AlgorithmOutputError("Adaptive-RAG worker artifact hash mismatch")


def _parse_route_logits(value: object) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_ROUTE_ORDER):
        raise AlgorithmOutputError("Adaptive-RAG route_logits must contain exactly A, B, and C")
    return freeze_mapping(
        {
            label: finite_number(
                value[label],
                label=f"Adaptive-RAG route logit {label}",
            )
            for label in _ROUTE_ORDER
        }
    )


def _parse_initial_retrieval(value: object) -> AdaptiveRAGInitialRetrieval | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("Adaptive-RAG initial_retrieval must be a mapping or null")
    return AdaptiveRAGInitialRetrieval.from_mapping(value)


def _validate_strategy_trajectory(
    *,
    strategy: AdaptiveRAGStrategy,
    question: str,
    documents: tuple[AdaptiveRAGDocument, ...],
    initial_retrieval: AdaptiveRAGInitialRetrieval | None,
    ircot_steps: tuple[AdaptiveRAGIRCoTStep, ...],
    active_document_ids: tuple[str, ...],
    controls: AdaptiveRAGControls,
) -> None:
    known = {document.document_id for document in documents}
    if strategy is AdaptiveRAGStrategy.NO_RETRIEVAL:
        if initial_retrieval is not None or ircot_steps or documents or active_document_ids:
            raise AlgorithmOutputError("Adaptive-RAG no-retrieval arm returned retrieval artifacts")
        return
    if initial_retrieval is None:
        raise AlgorithmOutputError("Adaptive-RAG retrieval arms require initial_retrieval")
    if initial_retrieval.query.strip() != question.strip():
        raise AlgorithmOutputError("Adaptive-RAG initial retrieval query must equal the question")
    initial_ids = initial_retrieval.document_ids
    top_k = (
        controls.single_retrieval_top_k
        if strategy is AdaptiveRAGStrategy.SINGLE_RETRIEVAL
        else controls.ircot_per_step_top_k
    )
    if len(initial_ids) > top_k:
        raise AlgorithmOutputError("Adaptive-RAG initial retrieval exceeds its top-k control")
    if len(initial_ids) > controls.max_total_documents:
        raise AlgorithmOutputError("Adaptive-RAG initial retrieval exceeds max_total_documents")
    if not set(initial_ids) <= known:
        raise AlgorithmOutputError("Adaptive-RAG initial retrieval lacks document records")

    if strategy is AdaptiveRAGStrategy.SINGLE_RETRIEVAL:
        if ircot_steps:
            raise AlgorithmOutputError("Adaptive-RAG single-retrieval arm returned IRCoT steps")
        if active_document_ids != initial_ids:
            raise AlgorithmOutputError(
                "Adaptive-RAG single-retrieval evidence must equal its one retrieval"
            )
        if known != set(initial_ids):
            raise AlgorithmOutputError(
                "Adaptive-RAG single-retrieval documents must match its retrieval"
            )
        return

    if len(ircot_steps) > controls.ircot_max_steps:
        raise AlgorithmOutputError("Adaptive-RAG IRCoT trajectory exceeds max steps")
    if not active_document_ids:
        raise AlgorithmOutputError("Adaptive-RAG IRCoT arm returned no active evidence")
    seen = set(initial_ids)
    active = set(initial_ids)
    for expected_index, step in enumerate(ircot_steps):
        if step.index != expected_index:
            raise AlgorithmOutputError(
                "Adaptive-RAG IRCoT step indices must be contiguous and zero-based"
            )
        retrieved = set(step.retrieved_document_ids)
        new = set(step.new_document_ids)
        next_active = set(step.active_document_ids)
        if len(step.retrieved_document_ids) > controls.ircot_per_step_top_k:
            raise AlgorithmOutputError("Adaptive-RAG IRCoT retrieval exceeds per-step top-k")
        if not retrieved <= known:
            raise AlgorithmOutputError("Adaptive-RAG IRCoT retrieved unknown documents")
        if not new <= retrieved:
            raise AlgorithmOutputError(
                "Adaptive-RAG IRCoT new documents must come from that retrieval"
            )
        if controls.deduplicate_documents and new != retrieved - seen:
            raise AlgorithmOutputError("Adaptive-RAG IRCoT new_document_ids violate deduplication")
        seen.update(new)
        if len(seen) > controls.max_total_documents:
            raise AlgorithmOutputError("Adaptive-RAG IRCoT exceeds max_total_documents")
        if not next_active <= seen:
            raise AlgorithmOutputError(
                "Adaptive-RAG IRCoT activated a document that was never retrieved"
            )
        if controls.ircot_reset_policy == "never" and step.reset_applied:
            raise AlgorithmOutputError("Adaptive-RAG IRCoT reset violates reset_policy")
        if controls.ircot_reset_policy == "replace_each_step" and not step.reset_applied:
            raise AlgorithmOutputError("Adaptive-RAG IRCoT replace_each_step requires a reset")
        if step.reset_applied:
            if next_active != retrieved:
                raise AlgorithmOutputError(
                    "Adaptive-RAG IRCoT reset evidence must equal that retrieval"
                )
        elif next_active != active | new:
            raise AlgorithmOutputError(
                "Adaptive-RAG IRCoT non-reset step lost or invented active evidence"
            )
        active = next_active
    if seen != known:
        raise AlgorithmOutputError(
            "Adaptive-RAG document records must exactly match observed retrievals"
        )
    if active_document_ids != (ircot_steps[-1].active_document_ids if ircot_steps else initial_ids):
        raise AlgorithmOutputError(
            "Adaptive-RAG final evidence disagrees with the IRCoT trajectory"
        )


def _validate_execution_evidence(
    value: object,
    *,
    artifacts: AdaptiveRAGArtifactIdentity,
    controls: AdaptiveRAGControls,
    strategy: AdaptiveRAGStrategy,
    question: str,
    route_logits: Mapping[str, float],
    answer: str,
    initial_retrieval: AdaptiveRAGInitialRetrieval | None,
    ircot_steps: tuple[AdaptiveRAGIRCoTStep, ...],
    response_usage: Any,
    remaining_budget: Mapping[str, int | None],
    expected_source_tree: str,
    expected_manifest_sha256: str,
    expected_source_closure_sha256: str,
    allow_fixture_runtime: bool,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("Adaptive-RAG response requires execution_evidence")
    expected: Mapping[str, object] = {
        "schema": _EXECUTION_EVIDENCE_SCHEMA,
        "source_commit": ADAPTIVE_RAG_UPSTREAM_COMMIT,
        "source_tree": expected_source_tree,
        "source_closure_sha256": expected_source_closure_sha256,
        "compatibility_manifest_sha256": expected_manifest_sha256,
        "router_revision": artifacts.router_revision,
        "router_snapshot_sha256": artifacts.router_snapshot_sha256,
        "router_weights_sha256": artifacts.router_weights_sha256,
        "router_tokenizer_sha256": artifacts.router_tokenizer_sha256,
        "base_model_revision": artifacts.base_model_revision,
        "generator_artifact_sha256": artifacts.generator_artifact_sha256,
        "generator_config_sha256": artifacts.generator_config_sha256,
        "retriever_revision": artifacts.retriever_revision,
        "retriever_implementation_sha256": artifacts.retriever_implementation_sha256,
        "retriever_config_sha256": artifacts.retriever_config_sha256,
        "corpus_revision": artifacts.corpus_revision,
        "corpus_manifest_sha256": artifacts.corpus_manifest_sha256,
        "index_revision": artifacts.index_revision,
        "index_manifest_sha256": artifacts.index_manifest_sha256,
        "selected_strategy": strategy.value,
        "controlled_method_execution": True,
        "controlled_upstream_router_parity": True,
        "upstream_arm_parity_claimed": False,
        "official_parity_claimed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AlgorithmOutputError(f"Adaptive-RAG execution evidence mismatch for {key}")

    fixture_runtime = value.get("fixture_runtime")
    if not isinstance(fixture_runtime, bool):
        raise AlgorithmOutputError("Adaptive-RAG fixture_runtime evidence must be boolean")
    evidence_kind = value.get("evidence_kind")
    expected_kind = "hermetic_fixture" if fixture_runtime else "controlled_component_closure"
    if evidence_kind != expected_kind:
        raise AlgorithmOutputError("Adaptive-RAG execution evidence kind is inconsistent")
    if fixture_runtime and not allow_fixture_runtime:
        raise AlgorithmOutputError("Adaptive-RAG production profile rejects fixture runtime")
    expected_source_state = "verified_fixture_snapshot" if fixture_runtime else "clean_git_checkout"
    if value.get("source_state") != expected_source_state:
        raise AlgorithmOutputError("Adaptive-RAG source-state evidence mismatch")
    prompt_hashes = value.get("prompt_sha256")
    expected_prompts = {
        "no_retrieval": artifacts.no_retrieval_prompt_sha256,
        "single_retrieval": artifacts.single_retrieval_prompt_sha256,
        "ircot": artifacts.ircot_prompt_sha256,
    }
    if not isinstance(prompt_hashes, Mapping) or dict(prompt_hashes) != expected_prompts:
        raise AlgorithmOutputError("Adaptive-RAG prompt evidence mismatch")

    events = mapping_sequence(
        value.get("events"),
        label="Adaptive-RAG supervisor events",
    )
    if len(events) < 2:
        raise AlgorithmOutputError("Adaptive-RAG supervisor event log is incomplete")
    if value.get("event_log_sha256") != content_hash(events):
        raise AlgorithmOutputError("Adaptive-RAG supervisor event-log hash mismatch")
    if value.get("final_event_sha256") != events[-1].get("event_sha256"):
        raise AlgorithmOutputError("Adaptive-RAG supervisor final-event hash mismatch")

    expected_operations = ["router.first_decoder_step"]
    if strategy is AdaptiveRAGStrategy.NO_RETRIEVAL:
        expected_operations.append("answer.no_retrieval")
    elif strategy is AdaptiveRAGStrategy.SINGLE_RETRIEVAL:
        expected_operations.extend(("retrieve.single.initial", "answer.single_retrieval"))
    else:
        expected_operations.append("retrieve.ircot.initial")
        for step in ircot_steps:
            expected_operations.extend(
                (f"ircot.reason.{step.index}", f"retrieve.ircot.step.{step.index}")
            )
        expected_operations.append("answer.ircot")
    if [event.get("operation") for event in events] != expected_operations:
        raise AlgorithmOutputError(
            "Adaptive-RAG supervisor events do not execute exactly the selected arm"
        )

    previous_hash = "0" * 64
    remaining: dict[str, int | None] = {
        "model_calls": int(remaining_budget["model_calls"]),
        "retrieval_calls": int(remaining_budget["retrieval_calls"]),
        "input_tokens": remaining_budget["context_tokens"],
        "output_tokens": int(remaining_budget["output_tokens"]),
    }
    totals: dict[str, int | float] = {
        key: 0
        if key
        not in {
            "cpu_seconds",
            "accelerator_seconds",
            "cost_usd",
            "compute_cost_usd",
        }
        else 0.0
        for key in _EVENT_USAGE_FIELDS
    }
    for index, event in enumerate(events):
        expected_event_keys = {
            "schema",
            "sequence",
            "previous_event_sha256",
            "call_kind",
            "operation",
            "pre_call_remaining",
            "reservation",
            "usage",
            "result",
            "event_sha256",
        }
        if set(event) != expected_event_keys:
            raise AlgorithmOutputError("Adaptive-RAG supervisor event fields drifted")
        if (
            event.get("schema") != _SUPERVISOR_EVENT_SCHEMA
            or event.get("sequence") != index
            or event.get("previous_event_sha256") != previous_hash
        ):
            raise AlgorithmOutputError("Adaptive-RAG supervisor event chain is invalid")
        unsigned = {key: item for key, item in event.items() if key != "event_sha256"}
        if event.get("event_sha256") != content_hash(unsigned):
            raise AlgorithmOutputError("Adaptive-RAG supervisor event hash mismatch")
        previous_hash = str(event["event_sha256"])
        before = event.get("pre_call_remaining")
        if not isinstance(before, Mapping) or dict(before) != remaining:
            raise AlgorithmOutputError("Adaptive-RAG pre-call budget snapshot is invalid")
        reservation = event.get("reservation")
        usage = event.get("usage")
        if not isinstance(reservation, Mapping) or not isinstance(usage, Mapping):
            raise AlgorithmOutputError("Adaptive-RAG supervisor budget record is invalid")
        if set(reservation) != _EVENT_RESERVATION_FIELDS:
            raise AlgorithmOutputError("Adaptive-RAG supervisor reservation fields drifted")
        if set(usage) != set(_EVENT_USAGE_FIELDS):
            raise AlgorithmOutputError("Adaptive-RAG supervisor usage fields drifted")
        call_kind = event.get("call_kind")
        expected_model = 1 if call_kind == "model" else 0
        expected_retrieval = 1 if call_kind == "retrieval" else 0
        if call_kind not in {"model", "retrieval"}:
            raise AlgorithmOutputError("Adaptive-RAG supervisor call kind is invalid")
        if (
            reservation.get("model_calls") != expected_model
            or reservation.get("retrieval_calls") != expected_retrieval
            or usage.get("model_calls") != expected_model
            or usage.get("retrieval_calls") != expected_retrieval
        ):
            raise AlgorithmOutputError("Adaptive-RAG call evidence disagrees with its kind")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens < 0
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens < 0
        ):
            raise AlgorithmOutputError("Adaptive-RAG event token usage is invalid")
        estimated_input = reservation.get("estimated_input_tokens")
        output_cap = reservation.get("max_output_tokens")
        if (
            isinstance(estimated_input, bool)
            or not isinstance(estimated_input, int)
            or estimated_input < input_tokens
            or isinstance(output_cap, bool)
            or not isinstance(output_cap, int)
        ):
            raise AlgorithmOutputError("Adaptive-RAG supervisor reservation is invalid")
        requested_output_cap = (
            1
            if event.get("operation") == "router.first_decoder_step"
            else controls.max_output_tokens
        )
        expected_output_cap = (
            min(requested_output_cap, int(remaining["output_tokens"] or 0)) if expected_model else 0
        )
        if expected_model and expected_output_cap < 1:
            raise AlgorithmOutputError(
                "Adaptive-RAG supervisor dispatched a model after output budget exhaustion"
            )
        if output_cap != expected_output_cap:
            raise AlgorithmOutputError(
                "Adaptive-RAG supervisor reservation output cap disagrees with sent controls"
            )
        if output_cap < output_tokens:
            raise AlgorithmOutputError("Adaptive-RAG post-call usage exceeds its reservation")
        remaining["model_calls"] = int(remaining["model_calls"] or 0) - expected_model
        remaining["retrieval_calls"] = int(remaining["retrieval_calls"] or 0) - expected_retrieval
        if min(int(remaining["model_calls"]), int(remaining["retrieval_calls"])) < 0:
            raise AlgorithmOutputError("Adaptive-RAG event log exceeded a call budget")
        if remaining["input_tokens"] is not None:
            remaining["input_tokens"] = int(remaining["input_tokens"]) - input_tokens
            if remaining["input_tokens"] < 0:
                raise AlgorithmOutputError("Adaptive-RAG event log exceeded context budget")
        remaining["output_tokens"] = int(remaining["output_tokens"] or 0) - output_tokens
        if remaining["output_tokens"] < 0:
            raise AlgorithmOutputError("Adaptive-RAG event log exceeded output budget")
        for key in _EVENT_USAGE_FIELDS:
            event_value = usage.get(key)
            if (
                isinstance(event_value, bool)
                or not isinstance(event_value, (int, float))
                or not math.isfinite(float(event_value))
                or event_value < 0
            ):
                raise AlgorithmOutputError("Adaptive-RAG supervisor usage contains invalid values")
            totals[key] += event_value

    first_result = events[0].get("result")
    if (
        not isinstance(first_result, Mapping)
        or first_result.get("question_sha256") != content_hash(question)
        or first_result.get("route_logits") != route_logits
    ):
        raise AlgorithmOutputError("Adaptive-RAG router event disagrees with the response")
    if strategy is not AdaptiveRAGStrategy.NO_RETRIEVAL:
        retrieval_result = events[1].get("result")
        if (
            initial_retrieval is None
            or not isinstance(retrieval_result, Mapping)
            or retrieval_result.get("query") != initial_retrieval.query
            or tuple(retrieval_result.get("document_ids", ())) != initial_retrieval.document_ids
        ):
            raise AlgorithmOutputError("Adaptive-RAG initial retrieval event mismatch")
    if strategy is AdaptiveRAGStrategy.IRCOT:
        for step in ircot_steps:
            reason_result = events[2 + step.index * 2].get("result")
            retrieval_result = events[3 + step.index * 2].get("result")
            if (
                not isinstance(reason_result, Mapping)
                or reason_result.get("text_sha256") != content_hash(step.cot_step)
                or not isinstance(retrieval_result, Mapping)
                or retrieval_result.get("query") != step.retrieval_query
                or tuple(retrieval_result.get("document_ids", ())) != step.retrieved_document_ids
            ):
                raise AlgorithmOutputError("Adaptive-RAG IRCoT event trajectory mismatch")
    final_result = events[-1].get("result")
    if not isinstance(final_result, Mapping) or final_result.get("text_sha256") != content_hash(
        answer
    ):
        raise AlgorithmOutputError("Adaptive-RAG answer event disagrees with the response")
    for key in _EVENT_USAGE_FIELDS:
        if getattr(response_usage, key) != totals[key]:
            raise AlgorithmOutputError(
                f"Adaptive-RAG response usage disagrees with supervisor events for {key}"
            )
    return freeze_mapping(value)


def _validate_usage(
    usage: Any,
    *,
    remaining_budget: Mapping[str, int | None],
    strategy: AdaptiveRAGStrategy,
    ircot_steps: tuple[AdaptiveRAGIRCoTStep, ...],
) -> None:
    if usage.model_calls < 2:
        raise AlgorithmOutputError(
            "Adaptive-RAG must account for classifier and answer model calls"
        )
    expected_retrieval_calls = {
        AdaptiveRAGStrategy.NO_RETRIEVAL: 0,
        AdaptiveRAGStrategy.SINGLE_RETRIEVAL: 1,
        AdaptiveRAGStrategy.IRCOT: 1 + len(ircot_steps),
    }[strategy]
    if usage.retrieval_calls != expected_retrieval_calls:
        raise AlgorithmOutputError(
            "Adaptive-RAG retrieval-call accounting disagrees with the selected arm"
        )
    if usage.model_calls > remaining_budget["model_calls"]:
        raise AlgorithmOutputError("Adaptive-RAG worker exceeded sent model-call budget")
    if usage.retrieval_calls > remaining_budget["retrieval_calls"]:
        raise AlgorithmOutputError("Adaptive-RAG worker exceeded sent retrieval-call budget")
    if usage.output_tokens > remaining_budget["output_tokens"]:
        raise AlgorithmOutputError("Adaptive-RAG worker exceeded sent output-token budget")
    context_limit = remaining_budget["context_tokens"]
    if context_limit is not None and usage.input_tokens > context_limit:
        raise AlgorithmOutputError("Adaptive-RAG worker exceeded sent context-token budget")


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
            raise ValidationError("Adaptive-RAG received an incompatible budget ledger")

    def consumed(name: str) -> int:
        value = 0 if usage is None else getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(f"Adaptive-RAG budget usage {name} is invalid")
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
            "retrieval_calls": max(
                0,
                state.budget.max_retrieval_calls - consumed("retrieval_calls"),
            ),
            "output_tokens": max(
                0,
                state.budget.max_output_tokens - consumed("output_tokens"),
            ),
            "context_tokens": remaining_context,
        }
    )


def _context_from_document(
    document: AdaptiveRAGDocument,
    *,
    strategy: AdaptiveRAGStrategy,
) -> ContextBlock:
    reserved = {
        "adaptive_rag_strategy",
        "corpus_revision",
        "document_id",
        "evidence_kind",
        "title",
    }
    overlap = reserved & set(document.metadata)
    if overlap:
        raise AlgorithmOutputError(
            "Adaptive-RAG document metadata uses reserved provenance keys: "
            + ", ".join(sorted(overlap))
        )
    return ContextBlock(
        block_id=document.document_id,
        text=document.text,
        source=document.corpus_revision,
        score=document.score,
        metadata={
            **dict(document.metadata),
            "document_id": document.document_id,
            "title": document.title,
            "corpus_revision": document.corpus_revision,
            "evidence_kind": "real_corpus_document",
            "adaptive_rag_strategy": strategy.value,
        },
    )


__all__ = [
    "AdaptiveRAGArtifactIdentity",
    "AdaptiveRAGControlledWorker",
    "AdaptiveRAGControls",
    "AdaptiveRAGDocument",
    "AdaptiveRAGIRCoTStep",
    "AdaptiveRAGInitialRetrieval",
]
