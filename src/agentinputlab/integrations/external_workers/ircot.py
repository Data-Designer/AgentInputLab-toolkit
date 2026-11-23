"""IRCoT controlled-worker protocol.

The upstream retrieval/reasoning loop remains inside the pinned Apache-2.0
worker.  This module validates its inputs, trajectory, lineage, and output; it
does not reimplement the loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ContextBlock,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.observability import external_usage_counters, normalize_external_usage

from .base import WorkerClient

IRCOT_UPSTREAM_COMMIT = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
_EXIT_REASONS = {"answer_detected", "controller", "max_steps", "upstream"}


@dataclass(frozen=True, slots=True)
class IRCoTControls:
    per_step_top_k: int = 5
    max_steps: int = 8
    max_total_documents: int = 64
    max_output_tokens: int = 4096
    deduplicate: bool = True
    reset_policy: str = "upstream_native"

    def __post_init__(self) -> None:
        for name in (
            "per_step_top_k",
            "max_steps",
            "max_total_documents",
            "max_output_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValidationError(f"IRCoT {name} must be positive")
        if self.reset_policy not in {"upstream_native", "never", "replace_each_step"}:
            raise ValidationError("unsupported IRCoT reset_policy")


@dataclass(frozen=True, slots=True)
class IRCoTWorkerRequest:
    request_id: str
    state_hash: str
    task_id: str
    question: str
    controls: IRCoTControls
    generator_fingerprint: Mapping[str, Any]
    retriever_fingerprint: Mapping[str, Any]
    corpus_revision: str
    index_revision: str
    prompt_hash: str
    upstream_commit: str = IRCOT_UPSTREAM_COMMIT
    protocol_version: str = "agentinputlab.ircot.v1"

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "state_hash",
            "question",
            "corpus_revision",
            "index_revision",
            "prompt_hash",
        ):
            if not str(getattr(self, name)).strip():
                raise ValidationError(f"IRCoT request {name} must not be empty")
        if self.upstream_commit != IRCOT_UPSTREAM_COMMIT:
            raise ValidationError("IRCoT request does not target the pinned upstream commit")
        object.__setattr__(
            self,
            "generator_fingerprint",
            MappingProxyType(dict(self.generator_fingerprint)),
        )
        object.__setattr__(
            self,
            "retriever_fingerprint",
            MappingProxyType(dict(self.retriever_fingerprint)),
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def external_reservation(self) -> Mapping[str, int]:
        """A successful IRCoT trajectory must account for a model call."""

        return {"model_calls": 1, "retrieval_calls": 1}

    def raise_for_external_status(self, result: object) -> None:
        """Interpret the worker status before generic usage extraction."""

        if not isinstance(result, Mapping):
            return
        status = str(result.get("status", ""))
        if status != "ok":
            code = str(result.get("failure_code", "WORKER_FAILURE"))
            raise ExternalDependencyError(f"IRCoT worker failed with {code}")


@dataclass(frozen=True, slots=True)
class IRCoTWorkerDocument:
    document_id: str
    text: str
    title: str
    corpus_revision: str
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> IRCoTWorkerDocument:
        document_id = _required_text(value, "document_id")
        text = _required_text(value, "text")
        corpus_revision = _required_text(value, "corpus_revision")
        score_raw = value.get("score")
        score = None if score_raw is None else float(score_raw)
        if score is not None and not isfinite(score):
            raise AlgorithmOutputError("IRCoT document score must be finite")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise AlgorithmOutputError("IRCoT document metadata must be a mapping")
        return cls(
            document_id=document_id,
            text=text,
            title=str(value.get("title", "")),
            corpus_revision=corpus_revision,
            score=score,
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class IRCoTStep:
    index: int
    cot_step: str
    latest_cot_sentence: str
    retrieval_query: str
    retrieved_document_ids: tuple[str, ...]
    new_document_ids: tuple[str, ...]
    active_document_ids: tuple[str, ...]
    reset_applied: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> IRCoTStep:
        try:
            index = int(value["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AlgorithmOutputError("IRCoT step index must be an integer") from exc
        cot_step = _required_text(value, "cot_step")
        latest = _required_text(value, "latest_cot_sentence")
        query = _required_text(value, "retrieval_query")
        if not cot_step.rstrip().endswith(latest):
            raise AlgorithmOutputError("IRCoT latest_cot_sentence is not the end of cot_step")
        if query.strip() != latest.strip():
            raise AlgorithmOutputError(
                "IRCoT follow-up retrieval query must equal the latest CoT sentence"
            )
        return cls(
            index=index,
            cot_step=cot_step,
            latest_cot_sentence=latest,
            retrieval_query=query,
            retrieved_document_ids=_id_tuple(value, "retrieved_document_ids"),
            new_document_ids=_id_tuple(value, "new_document_ids"),
            active_document_ids=_id_tuple(value, "active_document_ids"),
            reset_applied=bool(value.get("reset_applied", False)),
        )


@dataclass(frozen=True, slots=True)
class IRCoTWorkerResponse:
    request_id: str
    upstream_commit: str
    initial_query: str
    initial_document_ids: tuple[str, ...]
    steps: tuple[IRCoTStep, ...]
    documents: tuple[IRCoTWorkerDocument, ...]
    active_document_ids: tuple[str, ...]
    exit_reason: str
    answer: str = ""
    usage: Mapping[str, int | float] = field(default_factory=dict)
    protocol_version: str = "agentinputlab.ircot.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        request: IRCoTWorkerRequest,
    ) -> IRCoTWorkerResponse:
        status = str(value.get("status", ""))
        if status != "ok":
            code = str(value.get("failure_code", "WORKER_FAILURE"))
            raise ExternalDependencyError(f"IRCoT worker failed with {code}")
        request_id = _required_text(value, "request_id")
        if request_id != request.request_id:
            raise AlgorithmOutputError("IRCoT response request_id mismatch")
        upstream_commit = _required_text(value, "upstream_commit")
        if upstream_commit != request.upstream_commit:
            raise AlgorithmOutputError("IRCoT worker upstream revision mismatch")
        if str(value.get("protocol_version", "")) != request.protocol_version:
            raise AlgorithmOutputError("IRCoT worker protocol version mismatch")
        initial_query = _required_text(value, "initial_query")
        if initial_query.strip() != request.question.strip():
            raise AlgorithmOutputError("IRCoT initial retrieval query must equal the question")
        initial_ids = _id_tuple(value, "initial_document_ids")
        raw_steps = value.get("steps", ())
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
            raise AlgorithmOutputError("IRCoT response steps must be a sequence")
        steps = tuple(IRCoTStep.from_mapping(_mapping(item, "IRCoT step")) for item in raw_steps)
        if len(steps) > request.controls.max_steps:
            raise AlgorithmOutputError("IRCoT response exceeds max_steps")
        for expected, step in enumerate(steps):
            if step.index != expected:
                raise AlgorithmOutputError("IRCoT step indices must be contiguous and zero-based")
        raw_documents = value.get("documents", ())
        if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes)):
            raise AlgorithmOutputError("IRCoT response documents must be a sequence")
        documents = tuple(
            IRCoTWorkerDocument.from_mapping(_mapping(item, "IRCoT document"))
            for item in raw_documents
        )
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise AlgorithmOutputError("IRCoT response contains duplicate document records")
        if any(document.corpus_revision != request.corpus_revision for document in documents):
            raise AlgorithmOutputError("IRCoT response contains a corpus revision mismatch")
        active_ids = _id_tuple(value, "active_document_ids")
        known = set(document_ids)
        if not set(active_ids) <= known:
            raise AlgorithmOutputError("IRCoT active document IDs lack document records")
        _validate_trajectory(initial_ids, steps, known, request.controls)
        if steps and active_ids != steps[-1].active_document_ids:
            raise AlgorithmOutputError(
                "IRCoT final active document IDs disagree with the last step"
            )
        if not steps and active_ids != initial_ids:
            raise AlgorithmOutputError("IRCoT initial active document IDs are inconsistent")
        if not active_ids:
            raise AlgorithmOutputError("IRCoT worker returned no active evidence")
        exit_reason = _required_text(value, "exit_reason")
        if exit_reason not in _EXIT_REASONS:
            raise AlgorithmOutputError("IRCoT response has an unsupported exit_reason")
        answer = str(value.get("answer", ""))
        if exit_reason == "max_steps" and len(steps) != request.controls.max_steps:
            raise AlgorithmOutputError(
                "IRCoT max_steps exit requires exhausting the configured step budget"
            )
        if exit_reason == "answer_detected" and not answer.strip():
            raise AlgorithmOutputError("IRCoT answer_detected exit requires a non-empty answer")
        usage = _numeric_usage(value.get("usage"))
        expected_retrieval_calls = 1 + len(steps)
        if int(usage.get("retrieval_calls", -1)) != expected_retrieval_calls:
            raise AlgorithmOutputError("IRCoT worker retrieval-call accounting is inconsistent")
        if float(usage.get("output_tokens", 0)) > request.controls.max_output_tokens:
            raise AlgorithmOutputError("IRCoT worker exceeded max_output_tokens")
        return cls(
            request_id=request_id,
            upstream_commit=upstream_commit,
            initial_query=initial_query,
            initial_document_ids=initial_ids,
            steps=steps,
            documents=documents,
            active_document_ids=active_ids,
            exit_reason=exit_reason,
            answer=answer,
            usage=usage,
            protocol_version=request.protocol_version,
        )


@dataclass(slots=True)
class IRCoTControlledWorker:
    """Delegate the complete loop to a pinned worker and validate its trace."""

    corpus_revision: str
    index_revision: str
    prompt_hash: str
    client: WorkerClient | None = None
    controls: IRCoTControls = IRCoTControls()
    method_id: str = "ircot"
    release_id: str = "acquire.ircot.controlled"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def __post_init__(self) -> None:
        if self.client is not None and not isinstance(self.client, WorkerClient):
            raise ExternalDependencyError("IRCoT requires an external WorkerClient")
        for value, name in (
            (self.corpus_revision, "corpus_revision"),
            (self.index_revision, "index_revision"),
            (self.prompt_hash, "prompt_hash"),
        ):
            if not value.strip():
                raise ValidationError(f"IRCoT {name} must not be empty")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        client = self._resolve_client(context)
        if client is None:
            raise ExternalDependencyError(
                "IRCoT controlled profile requires an external WorkerClient"
            )
        if not isinstance(client, WorkerClient):
            raise ValidationError("IRCoT external runtime does not implement WorkerClient")
        if context.backend is None or context.retriever is None:
            raise ExternalDependencyError(
                "IRCoT controlled profile requires backend and retriever bindings"
            )
        try:
            generator_fingerprint = dict(context.backend.fingerprint())
            retriever_fingerprint = dict(context.retriever.fingerprint())
        except AttributeError as exc:
            raise ExternalDependencyError(
                "IRCoT bindings must expose serializable fingerprints"
            ) from exc
        request_base = {
            "state_hash": state.state_hash,
            "task_id": state.task_id,
            "question": state.query,
            "controls": asdict(self.controls),
            "generator_fingerprint": generator_fingerprint,
            "retriever_fingerprint": retriever_fingerprint,
            "corpus_revision": self.corpus_revision,
            "index_revision": self.index_revision,
            "prompt_hash": self.prompt_hash,
            "upstream_commit": IRCOT_UPSTREAM_COMMIT,
            "protocol_version": "agentinputlab.ircot.v1",
        }
        request = IRCoTWorkerRequest(
            request_id=content_hash(request_base),
            state_hash=state.state_hash,
            task_id=state.task_id,
            question=state.query,
            controls=self.controls,
            generator_fingerprint=generator_fingerprint,
            retriever_fingerprint=retriever_fingerprint,
            corpus_revision=self.corpus_revision,
            index_revision=self.index_revision,
            prompt_hash=self.prompt_hash,
        )
        controlled_call = getattr(client, "call_controlled", None)
        raw = (
            controlled_call(request)
            if callable(controlled_call)
            else client.call(request.as_dict())
        )
        if not isinstance(raw, Mapping):
            raise AlgorithmOutputError("IRCoT worker returned a non-mapping response")
        response = IRCoTWorkerResponse.from_mapping(raw, request=request)
        by_id = {document.document_id: document for document in response.documents}
        reserved_metadata = {
            "corpus_revision",
            "document_id",
            "evidence_kind",
            "title",
        }
        for document in response.documents:
            overlap = reserved_metadata & set(document.metadata)
            if overlap:
                raise AlgorithmOutputError(
                    "IRCoT document metadata uses reserved provenance keys: "
                    + ", ".join(sorted(overlap))
                )
        contexts = tuple(
            ContextBlock(
                block_id=document_id,
                text=by_id[document_id].text,
                source=self.corpus_revision,
                score=by_id[document_id].score,
                metadata={
                    **dict(by_id[document_id].metadata),
                    "document_id": document_id,
                    "title": by_id[document_id].title,
                    "corpus_revision": self.corpus_revision,
                    "evidence_kind": "real_corpus_document",
                },
            )
            for document_id in response.active_document_ids
        )
        events = [
            TraceEvent(
                TracePhase.PREPARE,
                "initial_retrieval",
                {
                    "query": response.initial_query,
                    "document_ids": response.initial_document_ids,
                },
            )
        ]
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
            for step in response.steps
        )
        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "worker_exit",
                {
                    "exit_reason": response.exit_reason,
                    "answer": response.answer,
                    "active_document_ids": response.active_document_ids,
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(events),
            counters={
                **dict(response.usage),
                "worker_calls": 1,
                "reasoning_steps": len(response.steps),
                "active_documents": len(response.active_document_ids),
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    contexts,
                    target="retrieved_evidence",
                    rationale="active evidence returned by the pinned IRCoT worker",
                ),
            ),
            trace=trace,
            implementation_identity="thin_external_worker_bridge",
            profile_kind="controlled_adapter",
            metadata={
                "upstream_commit": response.upstream_commit,
                "worker_fingerprint": dict(client.fingerprint()),
                "generator_fingerprint": generator_fingerprint,
                "retriever_fingerprint": retriever_fingerprint,
                "corpus_revision": self.corpus_revision,
                "index_revision": self.index_revision,
                "prompt_hash": self.prompt_hash,
            },
        )

    def _resolve_client(self, context: InterventionContext) -> WorkerClient | None:
        injected = context.external_runtime
        if injected is not None and self.client is not None and injected is not self.client:
            raise ValidationError(
                "IRCoT worker is ambiguous; use either the constructor or "
                "InterventionContext.external_runtime"
            )
        return injected if injected is not None else self.client


def _validate_trajectory(
    initial_ids: tuple[str, ...],
    steps: tuple[IRCoTStep, ...],
    known_documents: set[str],
    controls: IRCoTControls,
) -> None:
    if not set(initial_ids) <= known_documents:
        raise AlgorithmOutputError("IRCoT initial IDs lack document records")
    if len(initial_ids) > controls.per_step_top_k:
        raise AlgorithmOutputError("IRCoT initial retrieval exceeds per_step_top_k")
    if len(initial_ids) > controls.max_total_documents:
        raise AlgorithmOutputError("IRCoT initial evidence exceeds max_total_documents")
    seen = set(initial_ids)
    active = set(initial_ids)
    for step in steps:
        retrieved = set(step.retrieved_document_ids)
        new = set(step.new_document_ids)
        next_active = set(step.active_document_ids)
        if len(step.retrieved_document_ids) > controls.per_step_top_k:
            raise AlgorithmOutputError("IRCoT step retrieval exceeds per_step_top_k")
        if not retrieved <= known_documents:
            raise AlgorithmOutputError("IRCoT step retrieved unknown documents")
        if not new <= retrieved:
            raise AlgorithmOutputError("IRCoT new documents must come from that step's retrieval")
        if controls.deduplicate:
            if seen.intersection(new):
                raise AlgorithmOutputError("IRCoT new_document_ids violate deduplication")
            if new != retrieved - seen:
                raise AlgorithmOutputError(
                    "IRCoT new_document_ids must exactly identify unseen retrievals"
                )
        seen.update(new)
        if len(seen) > controls.max_total_documents:
            raise AlgorithmOutputError("IRCoT trajectory exceeds max_total_documents")
        if not next_active <= seen:
            raise AlgorithmOutputError("IRCoT active documents were never retrieved")
        if controls.reset_policy == "never" and step.reset_applied:
            raise AlgorithmOutputError("IRCoT reset violates reset_policy='never'")
        if controls.reset_policy == "replace_each_step" and not step.reset_applied:
            raise AlgorithmOutputError(
                "IRCoT replace_each_step requires a declared reset at every step"
            )
        if step.reset_applied:
            if next_active != retrieved:
                raise AlgorithmOutputError(
                    "IRCoT reset active documents must equal that step's retrieval"
                )
        elif next_active != active | new:
            raise AlgorithmOutputError(
                "IRCoT non-reset active documents must preserve prior evidence and add new evidence"
            )
        if len(step.active_document_ids) > controls.max_total_documents:
            raise AlgorithmOutputError("IRCoT active evidence exceeds max_total_documents")
        active = next_active
    if seen != known_documents:
        raise AlgorithmOutputError(
            "IRCoT document records must exactly match documents observed in the trajectory"
        )


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise AlgorithmOutputError(f"IRCoT field {key!r} must be non-empty text")
    return result


def _id_tuple(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = value.get(key, ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AlgorithmOutputError(f"IRCoT field {key!r} must be a sequence")
    result = tuple(str(item) for item in raw)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise AlgorithmOutputError(f"IRCoT field {key!r} contains empty or duplicate IDs")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError(f"{label} must be a mapping")
    return value


def _numeric_usage(value: Any) -> Mapping[str, int | float]:
    usage = normalize_external_usage(value)
    if usage.model_calls < 1:
        raise AlgorithmOutputError("IRCoT response must account for its model calls")
    return external_usage_counters(usage)
