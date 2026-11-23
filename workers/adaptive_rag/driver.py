#!/usr/bin/env python3
"""Source-verified, budget-supervised Adaptive-RAG worker.

The driver owns routing, arm dispatch, usage accounting, and the event log.
Injected model/retrieval components return only typed inference results; they
cannot supply attestations or caller-controlled trace records.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, TextIO

UPSTREAM_COMMIT = "0c88670af8707667eb5c1163151bb5ce61b14acb"
UPSTREAM_TREE = "8dadc754562d61d5d6e6e3ee2fc3aa4481763320"
METHOD_ID = "adaptive_rag"
RELEASE_ID = "acquire.adaptive_rag.controlled_reimplementation"
PROTOCOL_VERSION = "agentinputlab.adaptive-rag.v1"
MANIFEST_SCHEMA = "agentinputlab.adaptive-rag.compatibility.v1"
EVENT_SCHEMA = "agentinputlab.adaptive-rag.supervisor-event.v1"
EVIDENCE_SCHEMA = "agentinputlab.adaptive-rag.execution-evidence.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
_ZERO_HASH = "0" * 64
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ROUTES = ("A", "B", "C")
_STRATEGIES = {"A": "no_retrieval", "B": "single_retrieval", "C": "ircot"}
_ROOT_FIELDS = frozenset(
    {
        "method_id",
        "release_id",
        "upstream_commit",
        "state_hash",
        "payload",
        "dependency_fingerprints",
        "minimum_reservation",
        "protocol_version",
        "request_id",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "task_id",
        "question",
        "artifacts",
        "artifact_identity_sha256",
        "controls",
        "remaining_budget",
        "seed",
        "execution_contract",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "attestation",
        "gold",
        "gold_answer",
        "ground_truth",
        "label",
        "labels",
        "oracle",
        "reward",
        "silver",
        "target",
    }
)
_ALLOWED_DOCUMENT_METADATA = frozenset({"source_uri", "chunk_index", "language", "content_sha256"})


class DriverError(RuntimeError):
    """Expected fail-closed worker error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        partial_usage: Mapping[str, Any] | None = None,
        partial_events: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed
        self.partial_usage = dict(partial_usage) if partial_usage is not None else _zero_usage()
        self.partial_events = tuple(dict(event) for event in partial_events)


@dataclass(frozen=True, slots=True)
class RouterResult:
    """Raw first decoder-step label logits and observed usage."""

    logits: Mapping[str, float]
    input_tokens: int
    output_tokens: int = 1
    accelerator_seconds: float = 0.0
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    accelerator_seconds: float = 0.0
    cost_usd: float | None = None
    reset_retrieval: bool = False


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    document_id: str
    text: str
    title: str = ""
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Detach valid mappings immediately so a caller-owned retriever object
        # cannot drift between retrieval and supervisor validation.  Semantic
        # allowlisting intentionally remains in _validate_document so leakage
        # is still reported at the supervised retriever boundary.
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", _freeze_metadata_value(self.metadata))

    def as_mapping(self, *, corpus_revision: str) -> Mapping[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "title": self.title,
            "corpus_revision": corpus_revision,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
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
    fixture_runtime: bool = False

    def __post_init__(self) -> None:
        for name in (
            "router_revision",
            "base_model_revision",
            "retriever_revision",
            "corpus_revision",
            "index_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DriverError("RUNTIME_IDENTITY_INVALID", f"{name} must be non-empty")
            if value.strip().lower() in {"default", "head", "latest", "main", "master"}:
                raise DriverError("RUNTIME_IDENTITY_INVALID", f"{name} must be immutable")
        for name in (
            "router_snapshot_sha256",
            "router_weights_sha256",
            "router_tokenizer_sha256",
            "generator_artifact_sha256",
            "generator_config_sha256",
            "retriever_implementation_sha256",
            "retriever_config_sha256",
            "corpus_manifest_sha256",
            "index_manifest_sha256",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.fixture_runtime, bool):
            raise DriverError("RUNTIME_IDENTITY_INVALID", "fixture_runtime must be boolean")


class RouterBackend(Protocol):
    def component_manifest(self) -> Mapping[str, Any]: ...

    def estimate_input_tokens(self, question: str) -> int: ...

    def first_decoder_step(self, question: str, *, seed: int) -> RouterResult: ...


class GeneratorBackend(Protocol):
    def component_manifest(self) -> Mapping[str, Any]: ...

    def estimate_input_tokens(
        self,
        *,
        operation: str,
        question: str,
        documents: Sequence[RetrievedDocument],
        reasoning: Sequence[str],
        prompt: str,
    ) -> int: ...

    def generate(
        self,
        *,
        operation: str,
        question: str,
        documents: Sequence[RetrievedDocument],
        reasoning: Sequence[str],
        prompt: str,
        max_output_tokens: int,
        seed: int,
    ) -> GenerationResult: ...


class RetrieverBackend(Protocol):
    def component_manifest(self) -> Mapping[str, Any]: ...

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedDocument]: ...


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    identity: RuntimeIdentity
    router: RouterBackend
    generator: GeneratorBackend
    retriever: RetrieverBackend


@dataclass(frozen=True, slots=True)
class CompatibilityPolicy:
    commit: str
    tree: str
    license_path: str
    license_sha256: str
    require_clean_checkout: bool
    source_hashes: Mapping[str, str]
    prompt_files: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompatibilityPolicy:
        if value.get("schema_version") != MANIFEST_SCHEMA:
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG manifest schema mismatch")
        upstream = _mapping(value.get("upstream"), "manifest upstream")
        execution = _mapping(value.get("execution"), "manifest execution")
        security = _mapping(value.get("security"), "manifest security")
        if upstream.get("commit") != UPSTREAM_COMMIT:
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG manifest commit mismatch")
        tree = _git_hash(upstream.get("tree"), "manifest tree")
        if upstream.get("license_spdx") != "Apache-2.0":
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG root license must be Apache-2.0")
        if execution.get("method_id") != METHOD_ID or execution.get("release_id") != RELEASE_ID:
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG manifest release mismatch")
        if execution.get("protocol_version") != PROTOCOL_VERSION:
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG manifest protocol mismatch")
        if execution.get("route_labels") != list(_ROUTES):
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG route order must be A, B, C")
        if execution.get("strategy_map") != _STRATEGIES:
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG strategy map mismatch")
        if (
            execution.get("supervisor_event_schema") != EVENT_SCHEMA
            or execution.get("execution_evidence_schema") != EVIDENCE_SCHEMA
        ):
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG evidence schema mismatch")
        required_security = {
            "request_supplied_paths_allowed": False,
            "request_supplied_endpoints_allowed": False,
            "request_supplied_attestations_allowed": False,
            "evaluation_gold_fields_allowed": False,
            "evaluation_silver_fields_allowed": False,
            "evaluation_oracle_fields_allowed": False,
            "production_fixture_runtime_allowed": False,
            "budget_checked_before_every_external_call": True,
            "selected_arm_only": True,
            "official_parity_claimed": False,
        }
        if any(security.get(key) != expected for key, expected in required_security.items()):
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG security policy drifted")
        raw_sources = _mapping(value.get("source_files"), "manifest source_files")
        source_hashes = {
            _relative_path(path, "manifest source path"): _sha256(digest, "source hash")
            for path, digest in raw_sources.items()
        }
        prompts = {
            strategy: _relative_path(path, f"{strategy} prompt path")
            for strategy, path in _mapping(
                execution.get("prompt_files"), "manifest prompt_files"
            ).items()
        }
        if set(prompts) != set(_STRATEGIES.values()):
            raise DriverError("INVALID_MANIFEST", "Adaptive-RAG manifest must bind three prompts")
        return cls(
            commit=UPSTREAM_COMMIT,
            tree=tree,
            license_path=_relative_path(upstream.get("license_path"), "license path"),
            license_sha256=_sha256(upstream.get("license_sha256"), "license hash"),
            require_clean_checkout=upstream.get("require_clean_checkout") is True,
            source_hashes=source_hashes,
            prompt_files=prompts,
        )


class _Supervisor:
    """Only object allowed to dispatch external calls and write call evidence."""

    def __init__(
        self,
        *,
        runtime: RuntimeBundle,
        remaining: Mapping[str, Any],
    ) -> None:
        self.runtime = runtime
        self.remaining: dict[str, int | None] = {
            "model_calls": _nonnegative_int(remaining.get("model_calls"), "model budget"),
            "retrieval_calls": _nonnegative_int(
                remaining.get("retrieval_calls"), "retrieval budget"
            ),
            "input_tokens": _optional_nonnegative_int(
                remaining.get("context_tokens"), "context budget"
            ),
            "output_tokens": _nonnegative_int(remaining.get("output_tokens"), "output budget"),
        }
        self.events: list[Mapping[str, Any]] = []
        self._last_hash = _ZERO_HASH
        self._usage = _zero_usage()

    def can_model(self, count: int = 1) -> bool:
        return int(self.remaining["model_calls"] or 0) >= count

    def can_retrieve(self, count: int = 1) -> bool:
        return int(self.remaining["retrieval_calls"] or 0) >= count

    def classify(self, question: str, *, seed: int) -> Mapping[str, float]:
        estimate = self.runtime.router.estimate_input_tokens(question)
        result = self._model_call(
            operation="router.first_decoder_step",
            estimated_input_tokens=estimate,
            requested_output_tokens=1,
            invoke=lambda cap: self.runtime.router.first_decoder_step(question, seed=seed),
            result_builder=lambda value: {
                "question_sha256": _content_hash(question),
                "route_logits": dict(value.logits),
            },
        )
        if not isinstance(result, RouterResult):
            raise self.error("ROUTER_FAILED", "router returned the wrong result type")
        if set(result.logits) != set(_ROUTES):
            raise self.error("ROUTER_FAILED", "router must return exactly A/B/C logits")
        try:
            logits = {
                label: _finite_number(result.logits[label], f"router logit {label}")
                for label in _ROUTES
            }
        except DriverError as exc:
            raise self.error(exc.code, exc.public_message) from exc
        # Replace the event's raw mapping only after strict finite validation.
        event = dict(self.events[-1])
        result_payload = dict(_mapping(event["result"], "router event result"))
        result_payload["route_logits"] = logits
        event["result"] = result_payload
        self._replace_last_event(event)
        return logits

    def retrieve(
        self,
        *,
        operation: str,
        query: str,
        top_k: int,
    ) -> tuple[RetrievedDocument, ...]:
        before = self._reserve(
            operation=operation,
            call_kind="retrieval",
            model_calls=0,
            retrieval_calls=1,
            input_tokens=0,
            output_tokens=0,
        )
        try:
            raw = self.runtime.retriever.retrieve(query, top_k=top_k)
        except Exception as exc:
            raise self.error("RETRIEVER_FAILED", "retriever call failed") from exc
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise self.error("RETRIEVER_FAILED", "retriever returned a non-sequence")
        try:
            documents = tuple(_validate_document(item) for item in raw)
        except DriverError as exc:
            raise self.error(exc.code, exc.public_message) from exc
        if len(documents) > top_k:
            raise self.error("RETRIEVER_FAILED", "retriever exceeded top_k")
        ids = tuple(item.document_id for item in documents)
        if len(ids) != len(set(ids)):
            raise self.error("RETRIEVER_FAILED", "retriever returned duplicate IDs")
        self._record(
            operation=operation,
            call_kind="retrieval",
            before=before,
            reservation={
                "model_calls": 0,
                "retrieval_calls": 1,
                "estimated_input_tokens": 0,
                "max_output_tokens": 0,
            },
            usage=_zero_usage(retrieval_calls=1),
            result={
                "query": query,
                "query_sha256": _content_hash(query),
                "top_k": top_k,
                "document_ids": ids,
                "documents_sha256": _content_hash(
                    tuple(_document_event_mapping(item) for item in documents)
                ),
            },
        )
        return documents

    def generate(
        self,
        *,
        operation: str,
        question: str,
        documents: Sequence[RetrievedDocument],
        reasoning: Sequence[str],
        prompt: str,
        output_cap: int,
        seed: int,
    ) -> GenerationResult:
        estimate = self.runtime.generator.estimate_input_tokens(
            operation=operation,
            question=question,
            documents=documents,
            reasoning=reasoning,
            prompt=prompt,
        )
        result = self._model_call(
            operation=operation,
            estimated_input_tokens=estimate,
            requested_output_tokens=output_cap,
            invoke=lambda cap: self.runtime.generator.generate(
                operation=operation,
                question=question,
                documents=documents,
                reasoning=reasoning,
                prompt=prompt,
                max_output_tokens=cap,
                seed=seed,
            ),
            result_builder=lambda value: {
                "text_sha256": _content_hash(value.text),
                "document_ids": tuple(item.document_id for item in documents),
                "reset_retrieval": value.reset_retrieval,
            },
        )
        if not isinstance(result, GenerationResult) or not result.text.strip():
            raise self.error("GENERATOR_FAILED", "generator returned invalid text")
        return result

    def _model_call(
        self,
        *,
        operation: str,
        estimated_input_tokens: int,
        requested_output_tokens: int,
        invoke: Callable[[int], RouterResult | GenerationResult],
        result_builder: Callable[[RouterResult | GenerationResult], Mapping[str, Any]],
    ) -> RouterResult | GenerationResult:
        estimate = _nonnegative_int(estimated_input_tokens, "estimated input tokens")
        requested = _positive_int(requested_output_tokens, "requested output tokens")
        available_output = int(self.remaining["output_tokens"] or 0)
        cap = min(requested, available_output)
        if cap < 1:
            raise self.error("OUTPUT_BUDGET_EXHAUSTED", "model output budget exhausted")
        before = self._reserve(
            operation=operation,
            call_kind="model",
            model_calls=1,
            retrieval_calls=0,
            input_tokens=estimate,
            output_tokens=cap,
        )
        try:
            result = invoke(cap)
        except Exception as exc:
            raise self.error("MODEL_FAILED", f"{operation} failed") from exc
        if not isinstance(result, (RouterResult, GenerationResult)):
            raise self.error("MODEL_FAILED", f"{operation} returned invalid result")
        try:
            actual_input = _nonnegative_int(result.input_tokens, "observed input tokens")
            actual_output = _nonnegative_int(result.output_tokens, "observed output tokens")
            accelerator_seconds = _nonnegative_number(
                result.accelerator_seconds, "accelerator_seconds"
            )
            cost_usd = (
                None
                if result.cost_usd is None
                else _nonnegative_number(result.cost_usd, "cost_usd")
            )
        except DriverError as exc:
            raise self.error(exc.code, exc.public_message) from exc
        self._usage["input_tokens"] += actual_input
        self._usage["output_tokens"] += actual_output
        self._usage["accelerator_seconds"] += accelerator_seconds
        if cost_usd is not None:
            self._usage["cost_usd"] += cost_usd
            self._usage["unpriced_model_calls"] -= 1
        if actual_input > estimate:
            raise self.error(
                "INPUT_BUDGET_ESTIMATE_BREACH",
                f"{operation} exceeded its pre-call input reservation",
            )
        if actual_output > cap:
            raise self.error(
                "OUTPUT_BUDGET_EXCEEDED",
                f"{operation} exceeded its pre-call output cap",
            )
        self._consume_tokens(actual_input=actual_input, actual_output=actual_output)
        usage = _zero_usage(
            model_calls=1,
            unpriced_model_calls=1 if result.cost_usd is None else 0,
            input_tokens=actual_input,
            output_tokens=actual_output,
            accelerator_seconds=accelerator_seconds,
            cost_usd=0.0 if cost_usd is None else cost_usd,
        )
        self._record(
            operation=operation,
            call_kind="model",
            before=before,
            reservation={
                "model_calls": 1,
                "retrieval_calls": 0,
                "estimated_input_tokens": estimate,
                "max_output_tokens": cap,
            },
            usage=usage,
            result=result_builder(result),
        )
        return result

    def _reserve(
        self,
        *,
        operation: str,
        call_kind: str,
        model_calls: int,
        retrieval_calls: int,
        input_tokens: int,
        output_tokens: int,
    ) -> Mapping[str, int | None]:
        before = dict(self.remaining)
        if int(self.remaining["model_calls"] or 0) < model_calls:
            raise self.error(
                "MODEL_BUDGET_EXHAUSTED",
                f"{operation} model budget exhausted",
            )
        if int(self.remaining["retrieval_calls"] or 0) < retrieval_calls:
            raise self.error(
                "RETRIEVAL_BUDGET_EXHAUSTED", f"{operation} retrieval budget exhausted"
            )
        context = self.remaining["input_tokens"]
        if context is not None and context < input_tokens:
            raise self.error(
                "CONTEXT_BUDGET_EXHAUSTED",
                f"{operation} context budget exhausted",
            )
        if int(self.remaining["output_tokens"] or 0) < output_tokens:
            raise self.error(
                "OUTPUT_BUDGET_EXHAUSTED",
                f"{operation} output budget exhausted",
            )
        if call_kind not in {"model", "retrieval"}:
            raise self.error("INTERNAL_ERROR", "unknown supervisor call kind")
        self.remaining["model_calls"] = int(self.remaining["model_calls"] or 0) - model_calls
        self.remaining["retrieval_calls"] = (
            int(self.remaining["retrieval_calls"] or 0) - retrieval_calls
        )
        self._usage["model_calls"] += model_calls
        self._usage["retrieval_calls"] += retrieval_calls
        self._usage["unpriced_model_calls"] += model_calls
        # Tokens are consumed after the call, but the upper bounds are checked here.
        return before

    def _consume_tokens(self, *, actual_input: int, actual_output: int) -> None:
        context = self.remaining["input_tokens"]
        if context is not None:
            self.remaining["input_tokens"] = context - actual_input
        self.remaining["output_tokens"] = int(self.remaining["output_tokens"] or 0) - actual_output

    def _record(
        self,
        *,
        operation: str,
        call_kind: str,
        before: Mapping[str, int | None],
        reservation: Mapping[str, Any],
        usage: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        event = {
            "schema": EVENT_SCHEMA,
            "sequence": len(self.events),
            "previous_event_sha256": self._last_hash,
            "call_kind": call_kind,
            "operation": operation,
            "pre_call_remaining": dict(before),
            "reservation": dict(reservation),
            "usage": dict(usage),
            "result": dict(result),
        }
        event["event_sha256"] = _content_hash(event)
        self.events.append(event)
        self._last_hash = str(event["event_sha256"])

    def _replace_last_event(self, event: Mapping[str, Any]) -> None:
        unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
        updated = {**unsigned, "event_sha256": _content_hash(unsigned)}
        self.events[-1] = updated
        self._last_hash = str(updated["event_sha256"])

    def usage(self) -> Mapping[str, Any]:
        return dict(self._usage)

    def error(self, code: str, message: str) -> DriverError:
        consumed = bool(self._usage["model_calls"] or self._usage["retrieval_calls"])
        return DriverError(
            code,
            message,
            consumed=consumed,
            partial_usage=self._usage,
            partial_events=self.events,
        )


class AdaptiveRAGWorkerDriver:
    """Verify immutable deployment inputs and execute one selected strategy."""

    def __init__(
        self,
        *,
        source_root: Path,
        router_weights_path: Path,
        prompt_root: Path,
        runtime: RuntimeBundle,
        router_snapshot_path: Path | None = None,
        router_tokenizer_path: Path | None = None,
        generator_artifact_manifest_path: Path | None = None,
        generator_config_path: Path | None = None,
        retriever_implementation_path: Path | None = None,
        retriever_config_path: Path | None = None,
        corpus_manifest_path: Path | None = None,
        index_manifest_path: Path | None = None,
        manifest_path: Path = DEFAULT_MANIFEST,
        fixture_mode: bool = False,
    ) -> None:
        supplied_paths = {
            "source root": source_root,
            "router weights": router_weights_path,
            "prompt root": prompt_root,
            "router snapshot": router_snapshot_path,
            "router tokenizer": router_tokenizer_path,
            "generator artifact manifest": generator_artifact_manifest_path,
            "generator config": generator_config_path,
            "retriever implementation": retriever_implementation_path,
            "retriever config": retriever_config_path,
            "corpus manifest": corpus_manifest_path,
            "index manifest": index_manifest_path,
        }
        for label, path in supplied_paths.items():
            if path is not None and path.is_symlink():
                raise DriverError(
                    "PATH_INVALID",
                    f"Adaptive-RAG {label} must not be a symlink",
                )
        self.source_root = source_root.resolve()
        self.router_weights_path = router_weights_path.resolve()
        self.prompt_root = prompt_root.resolve()
        self.runtime = runtime
        self.component_paths = {
            "router_snapshot_sha256": router_snapshot_path,
            "router_tokenizer_sha256": router_tokenizer_path,
            "generator_artifact_sha256": generator_artifact_manifest_path,
            "generator_config_sha256": generator_config_path,
            "retriever_implementation_sha256": retriever_implementation_path,
            "retriever_config_sha256": retriever_config_path,
            "corpus_manifest_sha256": corpus_manifest_path,
            "index_manifest_sha256": index_manifest_path,
        }
        self.manifest_path = manifest_path.resolve()
        self.manifest = _load_json(self.manifest_path)
        self.manifest_sha256 = _content_hash(self.manifest)
        self.policy = CompatibilityPolicy.from_mapping(self.manifest)
        self.fixture_mode = fixture_mode
        if runtime.identity.fixture_runtime is not fixture_mode:
            raise DriverError("FIXTURE_MODE_MISMATCH", "runtime and driver fixture modes disagree")

    def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self._handle(request)
        except DriverError as exc:
            return {
                "status": "error",
                "failure_code": exc.code,
                "message": exc.public_message,
                "consumed": exc.consumed,
                "usage": dict(exc.partial_usage),
                "partial_events": tuple(exc.partial_events),
            }

    def source_fingerprint(self) -> Mapping[str, Any]:
        return self._verify_source()

    def _handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        _reject_forbidden_keys(request, "request")
        payload = _validate_envelope(request)
        artifacts = _mapping(payload.get("artifacts"), "artifacts")
        if payload.get("artifact_identity_sha256") != _content_hash(artifacts):
            raise DriverError("IDENTITY_MISMATCH", "artifact identity hash mismatch")
        controls = _controls(payload.get("controls"))
        _execution_contract(payload.get("execution_contract"))
        source = self._verify_source()
        prompts = self._verify_runtime_assets(artifacts)
        question = _text(payload.get("question"), "question")
        seed = _nonnegative_int(payload.get("seed"), "seed")
        supervisor = _Supervisor(
            runtime=self.runtime,
            remaining=_mapping(payload.get("remaining_budget"), "remaining_budget"),
        )
        logits = supervisor.classify(question, seed=seed)
        route = max(_ROUTES, key=lambda label: logits[label])
        strategy = _STRATEGIES[route]
        documents: list[RetrievedDocument] = []
        initial: Mapping[str, Any] | None = None
        steps: list[Mapping[str, Any]] = []
        active_ids: tuple[str, ...] = ()

        if strategy == "no_retrieval":
            if not supervisor.can_model():
                raise supervisor.error(
                    "MODEL_BUDGET_EXHAUSTED",
                    "selected arm cannot answer",
                )
            generated = supervisor.generate(
                operation="answer.no_retrieval",
                question=question,
                documents=(),
                reasoning=(),
                prompt=prompts[strategy],
                output_cap=controls["max_output_tokens"],
                seed=seed + 1,
            )
        elif strategy == "single_retrieval":
            if not supervisor.can_retrieve() or not supervisor.can_model():
                raise supervisor.error(
                    "ARM_BUDGET_EXHAUSTED",
                    "single-retrieval arm lacks budget",
                )
            retrieved = supervisor.retrieve(
                operation="retrieve.single.initial",
                query=question,
                top_k=controls["single_retrieval_top_k"],
            )
            if not retrieved:
                raise supervisor.error(
                    "EMPTY_RETRIEVAL",
                    "single-retrieval arm returned no documents",
                )
            documents.extend(retrieved)
            active_ids = tuple(item.document_id for item in retrieved)
            initial = {"query": question, "document_ids": active_ids}
            generated = supervisor.generate(
                operation="answer.single_retrieval",
                question=question,
                documents=retrieved,
                reasoning=(),
                prompt=prompts[strategy],
                output_cap=controls["max_output_tokens"],
                seed=seed + 1,
            )
        else:
            if not supervisor.can_retrieve() or not supervisor.can_model():
                raise supervisor.error(
                    "ARM_BUDGET_EXHAUSTED",
                    "IRCoT arm lacks minimum budget",
                )
            initial_docs = supervisor.retrieve(
                operation="retrieve.ircot.initial",
                query=question,
                top_k=controls["ircot_per_step_top_k"],
            )
            if not initial_docs:
                raise supervisor.error(
                    "EMPTY_RETRIEVAL",
                    "IRCoT initial retrieval returned no documents",
                )
            documents.extend(initial_docs)
            active = list(initial_docs)
            active_ids = tuple(item.document_id for item in active)
            initial = {"query": question, "document_ids": active_ids}
            reasoning: list[str] = []
            seen = set(active_ids)
            for index in range(controls["ircot_max_steps"]):
                # Preserve one final model call; a reasoning step also requires its retrieval.
                if not supervisor.can_model(2) or not supervisor.can_retrieve():
                    break
                reason = supervisor.generate(
                    operation=f"ircot.reason.{index}",
                    question=question,
                    documents=active,
                    reasoning=reasoning,
                    prompt=prompts[strategy],
                    output_cap=controls["max_output_tokens"],
                    seed=seed + 1 + index,
                )
                reasoning.append(reason.text)
                try:
                    retrieval_query = _latest_nonempty_line(reason.text)
                except DriverError as exc:
                    raise supervisor.error(exc.code, exc.public_message) from exc
                retrieved = supervisor.retrieve(
                    operation=f"retrieve.ircot.step.{index}",
                    query=retrieval_query,
                    top_k=controls["ircot_per_step_top_k"],
                )
                new_docs = tuple(item for item in retrieved if item.document_id not in seen)
                for item in new_docs:
                    seen.add(item.document_id)
                    documents.append(item)
                if len(seen) > controls["max_total_documents"]:
                    raise supervisor.error(
                        "DOCUMENT_BUDGET_EXCEEDED",
                        "IRCoT document budget exceeded",
                    )
                reset = controls["ircot_reset_policy"] == "replace_each_step" or (
                    controls["ircot_reset_policy"] == "upstream_native" and reason.reset_retrieval
                )
                if reset:
                    active = list(retrieved)
                else:
                    active.extend(new_docs)
                active_ids = tuple(item.document_id for item in active)
                steps.append(
                    {
                        "index": index,
                        "cot_step": reason.text,
                        "latest_cot_sentence": retrieval_query,
                        "retrieval_query": retrieval_query,
                        "retrieved_document_ids": tuple(item.document_id for item in retrieved),
                        "new_document_ids": tuple(item.document_id for item in new_docs),
                        "active_document_ids": active_ids,
                        "reset_applied": reset,
                    }
                )
            generated = supervisor.generate(
                operation="answer.ircot",
                question=question,
                documents=active,
                reasoning=reasoning,
                prompt=prompts[strategy],
                output_cap=controls["max_output_tokens"],
                seed=seed + 1 + len(steps),
            )

        event_log = tuple(supervisor.events)
        usage = supervisor.usage()
        evidence_kind = "hermetic_fixture" if self.fixture_mode else "controlled_component_closure"
        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "evidence_kind": evidence_kind,
            "fixture_runtime": self.fixture_mode,
            "controlled_method_execution": True,
            "controlled_upstream_router_parity": True,
            "upstream_arm_parity_claimed": False,
            "official_parity_claimed": False,
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "source_state": source["state"],
            "source_closure_sha256": source["closure_sha256"],
            "compatibility_manifest_sha256": self.manifest_sha256,
            "router_revision": artifacts["router_revision"],
            "router_snapshot_sha256": artifacts["router_snapshot_sha256"],
            "router_weights_sha256": artifacts["router_weights_sha256"],
            "router_tokenizer_sha256": artifacts["router_tokenizer_sha256"],
            "base_model_revision": artifacts["base_model_revision"],
            "generator_artifact_sha256": artifacts["generator_artifact_sha256"],
            "generator_config_sha256": artifacts["generator_config_sha256"],
            "retriever_revision": artifacts["retriever_revision"],
            "retriever_implementation_sha256": artifacts["retriever_implementation_sha256"],
            "retriever_config_sha256": artifacts["retriever_config_sha256"],
            "corpus_revision": artifacts["corpus_revision"],
            "corpus_manifest_sha256": artifacts["corpus_manifest_sha256"],
            "index_revision": artifacts["index_revision"],
            "index_manifest_sha256": artifacts["index_manifest_sha256"],
            "prompt_sha256": {
                "no_retrieval": artifacts["no_retrieval_prompt_sha256"],
                "single_retrieval": artifacts["single_retrieval_prompt_sha256"],
                "ircot": artifacts["ircot_prompt_sha256"],
            },
            "selected_strategy": strategy,
            "event_log_sha256": _content_hash(event_log),
            "final_event_sha256": event_log[-1]["event_sha256"],
            "events": event_log,
        }
        return {
            "status": "ok",
            "request_id": request["request_id"],
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "payload": {
                "question": question,
                "artifacts": dict(artifacts),
                "artifact_identity_sha256": payload["artifact_identity_sha256"],
                "route_logits": logits,
                "route_label": route,
                "selected_strategy": strategy,
                "answer": generated.text,
                "documents": tuple(
                    item.as_mapping(corpus_revision=self.runtime.identity.corpus_revision)
                    for item in documents
                ),
                "initial_retrieval": initial,
                "ircot_steps": tuple(steps),
                "active_document_ids": active_ids,
                "execution_evidence": evidence,
            },
            "usage": usage,
        }

    def _verify_source(self) -> Mapping[str, Any]:
        for relative, expected in self.policy.source_hashes.items():
            if _file_sha256(_safe_file(self.source_root, relative)) != expected:
                raise DriverError("SOURCE_MISMATCH", f"source hash mismatch: {relative}")
        if _file_sha256(_safe_file(self.source_root, self.policy.license_path)) != (
            self.policy.license_sha256
        ):
            raise DriverError("SOURCE_MISMATCH", "Adaptive-RAG license hash mismatch")
        if self.fixture_mode:
            state = "verified_fixture_snapshot"
            tree = self.policy.tree
        else:
            if not (self.source_root / ".git").exists():
                raise DriverError("SOURCE_MISMATCH", "production source must be a Git checkout")
            commit = _git(self.source_root, "rev-parse", "HEAD")
            tree = _git(self.source_root, "rev-parse", "HEAD^{tree}")
            status = _git(self.source_root, "status", "--porcelain=v1", "--untracked-files=all")
            if commit != self.policy.commit or tree != self.policy.tree or status:
                raise DriverError("SOURCE_MISMATCH", "source checkout is not exact and clean")
            state = "clean_git_checkout"
        closure = _content_hash(
            {
                "commit": self.policy.commit,
                "tree": tree,
                "source_files": dict(self.policy.source_hashes),
                "license_sha256": self.policy.license_sha256,
            }
        )
        return {
            "commit": self.policy.commit,
            "tree": tree,
            "state": state,
            "closure_sha256": closure,
        }

    def _verify_runtime_assets(self, artifacts: Mapping[str, Any]) -> Mapping[str, str]:
        identity = self.runtime.identity
        expected = {
            "upstream_commit": UPSTREAM_COMMIT,
            "router_revision": identity.router_revision,
            "router_snapshot_sha256": identity.router_snapshot_sha256,
            "router_weights_sha256": identity.router_weights_sha256,
            "router_tokenizer_sha256": identity.router_tokenizer_sha256,
            "base_model_revision": identity.base_model_revision,
            "generator_artifact_sha256": identity.generator_artifact_sha256,
            "generator_config_sha256": identity.generator_config_sha256,
            "retriever_revision": identity.retriever_revision,
            "retriever_implementation_sha256": identity.retriever_implementation_sha256,
            "retriever_config_sha256": identity.retriever_config_sha256,
            "corpus_revision": identity.corpus_revision,
            "corpus_manifest_sha256": identity.corpus_manifest_sha256,
            "index_revision": identity.index_revision,
            "index_manifest_sha256": identity.index_manifest_sha256,
        }
        for key, value in expected.items():
            if artifacts.get(key) != value:
                raise DriverError("PROFILE_DRIFT", f"runtime identity mismatch: {key}")
        weights_hash = _path_closure_sha256(self.router_weights_path)
        if identity.router_weights_sha256 != weights_hash:
            raise DriverError("ROUTER_MISMATCH", "router weights hash mismatch")
        if not self.fixture_mode:
            missing = [name for name, path in self.component_paths.items() if path is None]
            if missing:
                raise DriverError(
                    "COMPONENT_CLOSURE_MISSING",
                    "production runtime lacks component closure paths: "
                    + ", ".join(sorted(missing)),
                )
            for name, path in self.component_paths.items():
                assert path is not None
                if _path_closure_sha256(path.resolve()) != getattr(identity, name):
                    raise DriverError(
                        "COMPONENT_CLOSURE_MISMATCH",
                        f"production component closure mismatch: {name}",
                    )
        verify_component_manifests(self.runtime, fixture_mode=self.fixture_mode)
        hash_fields = {
            "no_retrieval": "no_retrieval_prompt_sha256",
            "single_retrieval": "single_retrieval_prompt_sha256",
            "ircot": "ircot_prompt_sha256",
        }
        prompts: dict[str, str] = {}
        for strategy, relative in self.policy.prompt_files.items():
            path = _safe_file(self.prompt_root, relative)
            if _file_sha256(path) != artifacts.get(hash_fields[strategy]):
                raise DriverError("PROMPT_MISMATCH", f"{strategy} prompt hash mismatch")
            prompts[strategy] = path.read_text(encoding="utf-8")
        return prompts


def verify_component_manifests(runtime: RuntimeBundle, *, fixture_mode: bool) -> None:
    identity = runtime.identity
    expected = {
        "router": {
            "component_type": "router",
            "fixture_runtime": fixture_mode,
            "revision": identity.router_revision,
            "snapshot_sha256": identity.router_snapshot_sha256,
            "weights_sha256": identity.router_weights_sha256,
            "tokenizer_sha256": identity.router_tokenizer_sha256,
        },
        "generator": {
            "component_type": "generator",
            "fixture_runtime": fixture_mode,
            "revision": identity.base_model_revision,
            "artifact_sha256": identity.generator_artifact_sha256,
            "config_sha256": identity.generator_config_sha256,
        },
        "retriever": {
            "component_type": "retriever",
            "fixture_runtime": fixture_mode,
            "revision": identity.retriever_revision,
            "implementation_sha256": identity.retriever_implementation_sha256,
            "config_sha256": identity.retriever_config_sha256,
            "corpus_revision": identity.corpus_revision,
            "corpus_manifest_sha256": identity.corpus_manifest_sha256,
            "index_revision": identity.index_revision,
            "index_manifest_sha256": identity.index_manifest_sha256,
        },
    }
    for name, component in (
        ("router", runtime.router),
        ("generator", runtime.generator),
        ("retriever", runtime.retriever),
    ):
        implementation_class = f"{type(component).__module__}.{type(component).__qualname__}"
        if (
            not fixture_mode
            and type(component).__module__ == "workers.adaptive_rag.fixture_runtime"
        ):
            raise DriverError(
                "FIXTURE_COMPONENT_REJECTED",
                f"production runtime cannot use the hermetic {name}",
            )
        provider = getattr(component, "component_manifest", None)
        if not callable(provider):
            raise DriverError(
                "COMPONENT_MANIFEST_MISSING",
                f"{name} must expose component_manifest()",
            )
        manifest = provider()
        if not isinstance(manifest, Mapping):
            raise DriverError(
                "COMPONENT_MANIFEST_INVALID",
                f"{name} component manifest must be a mapping",
            )
        required = {**expected[name], "implementation_class": implementation_class}
        if dict(manifest) != required:
            raise DriverError(
                "COMPONENT_MANIFEST_MISMATCH",
                f"{name} component manifest does not match the locked runtime identity",
            )


def _validate_envelope(request: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(request, Mapping) or set(request) != _ROOT_FIELDS:
        raise DriverError("REQUEST_INVALID", "worker request envelope fields are invalid")
    if (
        request.get("method_id") != METHOD_ID
        or request.get("release_id") != RELEASE_ID
        or request.get("upstream_commit") != UPSTREAM_COMMIT
        or request.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise DriverError("IDENTITY_MISMATCH", "Adaptive-RAG envelope identity mismatch")
    identity = {key: request[key] for key in _ROOT_FIELDS if key != "request_id"}
    if request.get("request_id") != _content_hash(identity):
        raise DriverError("IDENTITY_MISMATCH", "Adaptive-RAG request_id mismatch")
    payload = _mapping(request.get("payload"), "payload")
    if set(payload) != _PAYLOAD_FIELDS:
        raise DriverError("REQUEST_INVALID", "Adaptive-RAG payload fields are invalid")
    return payload


def _execution_contract(value: Any) -> None:
    expected = {
        "input_view": "question_only",
        "route_labels": list(_ROUTES),
        "route_rule": "first_decoder_step_logit_argmax_A_first",
        "strategy_map": _STRATEGIES,
        "evaluation_label_access": "none",
        "selected_arm_only": True,
        "fallback_allowed": False,
        "arm_execution_semantics": "controlled_reimplementation",
        "upstream_router_parity": True,
        "supervisor_event_schema": EVENT_SCHEMA,
    }
    if value != expected:
        raise DriverError("EXECUTION_CONTRACT_REJECTED", "execution contract drifted")


def _controls(value: Any) -> Mapping[str, Any]:
    raw = _mapping(value, "controls")
    expected = {
        "single_retrieval_top_k",
        "ircot_per_step_top_k",
        "ircot_max_steps",
        "max_total_documents",
        "max_output_tokens",
        "ircot_reset_policy",
        "deduplicate_documents",
    }
    if set(raw) != expected:
        raise DriverError("REQUEST_INVALID", "Adaptive-RAG controls fields are invalid")
    result = {
        key: _positive_int(raw[key], key)
        for key in (
            "single_retrieval_top_k",
            "ircot_per_step_top_k",
            "ircot_max_steps",
            "max_total_documents",
            "max_output_tokens",
        )
    }
    if raw["ircot_reset_policy"] not in {"upstream_native", "never", "replace_each_step"}:
        raise DriverError("REQUEST_INVALID", "unsupported IRCoT reset policy")
    if raw["deduplicate_documents"] is not True:
        raise DriverError("REQUEST_INVALID", "controlled runtime requires document deduplication")
    return {
        **result,
        "ircot_reset_policy": raw["ircot_reset_policy"],
        "deduplicate_documents": True,
    }


def _validate_document(value: Any) -> RetrievedDocument:
    if not isinstance(value, RetrievedDocument):
        raise DriverError("RETRIEVER_FAILED", "retriever must return RetrievedDocument")
    if not value.document_id.strip() or not value.text:
        raise DriverError("RETRIEVER_FAILED", "retrieved document requires ID and text")
    if value.score is not None:
        _finite_number(value.score, "document score")
    if not isinstance(value.metadata, Mapping):
        raise DriverError("RETRIEVER_FAILED", "document metadata must be a mapping")
    metadata = _validated_document_metadata(value.metadata)
    return RetrievedDocument(
        document_id=value.document_id,
        text=value.text,
        title=value.title,
        score=value.score,
        metadata=MappingProxyType(metadata),
    )


def _validated_document_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(value) - _ALLOWED_DOCUMENT_METADATA
    if unknown:
        raise DriverError(
            "EVALUATION_LEAKAGE",
            "document metadata contains non-allowlisted fields: "
            + ", ".join(sorted(str(item) for item in unknown)),
        )
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"source_uri", "language"}:
            if not isinstance(item, str) or not item.strip():
                raise DriverError(
                    "RETRIEVER_FAILED",
                    f"document metadata {key} must be non-empty text",
                )
            normalized[key] = item.strip()
        elif key == "chunk_index":
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise DriverError(
                    "RETRIEVER_FAILED",
                    "document metadata chunk_index must be non-negative",
                )
            normalized[key] = item
        else:
            normalized[key] = _sha256(item, "document metadata content_sha256")
    return normalized


def _freeze_metadata_value(value: Any) -> Any:
    """Detach and recursively freeze caller-owned metadata containers."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata_value(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    return value


def _document_event_mapping(value: RetrievedDocument) -> Mapping[str, Any]:
    return {
        "document_id": value.document_id,
        "text_sha256": _content_hash(value.text),
        "title_sha256": _content_hash(value.title),
        "score": value.score,
        "metadata": dict(value.metadata),
    }


def _reject_forbidden_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_KEYS or any(
                token in normalized for token in ("gold_", "silver_", "oracle_")
            ):
                raise DriverError("EVALUATION_LEAKAGE", f"forbidden evaluation field at {path}")
            _reject_forbidden_keys(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{path}[{index}]")


def _zero_usage(**updates: int | float) -> dict[str, int | float]:
    result: dict[str, int | float] = {
        "model_calls": 0,
        "retrieval_calls": 0,
        "tool_calls": 0,
        "embedding_calls": 0,
        "unpriced_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cpu_seconds": 0.0,
        "accelerator_seconds": 0.0,
        "cost_usd": 0.0,
        "compute_cost_usd": 0.0,
    }
    result.update(updates)
    return result


def _latest_nonempty_line(text: str) -> str:
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not lines:
        raise DriverError("GENERATOR_FAILED", "IRCoT reasoning step is empty", consumed=True)
    return lines[-1]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DriverError("REQUEST_INVALID", f"{label} must be a mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriverError("REQUEST_INVALID", f"{label} must be non-empty text")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result < 1:
        raise DriverError("REQUEST_INVALID", f"{label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("REQUEST_INVALID", f"{label} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    return None if value is None else _nonnegative_int(value, label)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError("RUNTIME_INVALID", f"{label} must be numeric", consumed=True)
    result = float(value)
    if not math.isfinite(result):
        raise DriverError("RUNTIME_INVALID", f"{label} must be finite", consumed=True)
    return result


def _nonnegative_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0:
        raise DriverError("RUNTIME_INVALID", f"{label} must be non-negative", consumed=True)
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DriverError("INVALID_MANIFEST", f"{label} must be SHA-256")
    return value


def _git_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise DriverError("INVALID_MANIFEST", f"{label} must be a Git object ID")
    return value


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DriverError("INVALID_MANIFEST", f"{label} must be a path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise DriverError("INVALID_MANIFEST", f"{label} is unsafe")
    return pure.as_posix()


def _safe_file(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink():
        raise DriverError("PATH_INVALID", f"symlink is forbidden: {relative}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise DriverError("PATH_INVALID", f"path escapes root: {relative}") from exc
    if not resolved.is_file():
        raise DriverError("PATH_INVALID", f"required file is absent: {relative}")
    return resolved


def _safe_absolute_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DriverError("PATH_INVALID", "router weights must be a regular file")
    return path


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverError("INVALID_MANIFEST", "manifest cannot be loaded") from exc
    if not isinstance(value, Mapping):
        raise DriverError("INVALID_MANIFEST", "manifest must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_closure_sha256(path: Path) -> str:
    """Hash one file or a symlink-free directory closure deterministically."""

    resolved = path.resolve()
    if not resolved.exists() or path.is_symlink():
        raise DriverError("COMPONENT_CLOSURE_MISSING", "component closure path is unavailable")
    if resolved.is_file():
        return _file_sha256(resolved)
    if not resolved.is_dir():
        raise DriverError("COMPONENT_CLOSURE_MISSING", "component closure path is not file/dir")
    files: dict[str, str] = {}
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise DriverError(
                "COMPONENT_CLOSURE_INVALID",
                "component closure directories may not contain symlinks",
            )
        if candidate.is_file():
            files[candidate.relative_to(resolved).as_posix()] = _file_sha256(candidate)
    if not files:
        raise DriverError("COMPONENT_CLOSURE_MISSING", "component closure directory is empty")
    return _content_hash(files)


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise DriverError("SOURCE_MISMATCH", "Git source verification failed")
    return completed.stdout.strip()


def serve_jsonl(
    driver: AdaptiveRAGWorkerDriver,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    """Serve exactly one bounded JSON request per input line."""

    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = driver.handle(request)
        except (json.JSONDecodeError, TypeError, ValueError):
            response = {
                "status": "error",
                "failure_code": "REQUEST_INVALID",
                "message": "request is not valid JSON",
                "consumed": False,
            }
        output_stream.write(
            json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        output_stream.flush()
    return 0


__all__ = [
    "EVENT_SCHEMA",
    "EVIDENCE_SCHEMA",
    "PROTOCOL_VERSION",
    "RELEASE_ID",
    "UPSTREAM_COMMIT",
    "UPSTREAM_TREE",
    "AdaptiveRAGWorkerDriver",
    "CompatibilityPolicy",
    "DriverError",
    "GenerationResult",
    "RetrievedDocument",
    "RouterResult",
    "RuntimeBundle",
    "RuntimeIdentity",
    "serve_jsonl",
]
