"""Transparent monitoring wrappers for runtime dependencies."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any

from agentinputlab.backends.llm.base import GenerationRequest, GenerationResponse, LLMBackend
from agentinputlab.backends.retrieval.base import RetrievalResult, Retriever
from agentinputlab.backends.vector import DenseEmbedder, VectorSearcher
from agentinputlab.connectors.tools.base import ToolCall, ToolConnector, ToolResult, ToolSpec
from agentinputlab.contracts.errors import (
    AlgorithmOutputError,
    BudgetExceededError,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.contracts.lifecycle import InterventionContext
from agentinputlab.contracts.state import FrozenAgentState

from .budget import BudgetLedger, PricingCatalog
from .models import EventKind, ResourceReceipt, ResourceUsage
from .monitor import RunMonitor

_EXTERNAL_USAGE_INTEGER_FIELDS = (
    "model_calls",
    "retrieval_calls",
    "tool_calls",
    "embedding_calls",
    "unpriced_model_calls",
    "input_tokens",
    "output_tokens",
)
_EXTERNAL_USAGE_FLOAT_FIELDS = (
    "cpu_seconds",
    "accelerator_seconds",
    "cost_usd",
    "compute_cost_usd",
)
_EXTERNAL_USAGE_REQUIRED_FIELDS = (
    "model_calls",
    "retrieval_calls",
    "input_tokens",
    "output_tokens",
)
_EXTERNAL_USAGE_FIELDS = frozenset((*_EXTERNAL_USAGE_INTEGER_FIELDS, *_EXTERNAL_USAGE_FLOAT_FIELDS))
_EXTERNAL_RESERVATION_FIELDS = frozenset(
    ("model_calls", "retrieval_calls", "tool_calls", "embedding_calls")
)


def normalize_external_usage(envelope: object) -> ResourceUsage:
    """Validate a common, lossless usage envelope for external executors.

    Model and retrieval call counts plus both token directions are mandatory.
    A model call without an explicit provider cost is conservatively marked
    unpriced, which makes a configured cost budget fail closed.
    """

    if not isinstance(envelope, Mapping):
        raise AlgorithmOutputError("external runtime response requires a usage mapping")
    if any(not isinstance(key, str) for key in envelope):
        raise AlgorithmOutputError("external runtime usage keys must be strings")
    unknown = set(envelope) - _EXTERNAL_USAGE_FIELDS
    if unknown:
        raise AlgorithmOutputError(
            "external runtime usage contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    missing = [field for field in _EXTERNAL_USAGE_REQUIRED_FIELDS if field not in envelope]
    if missing:
        raise AlgorithmOutputError(
            "external runtime usage lacks required fields: " + ", ".join(missing)
        )

    values = dict(envelope)
    model_calls = values["model_calls"]
    if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
        raise AlgorithmOutputError("external runtime model_calls must be a non-negative integer")
    if "cost_usd" not in values:
        if "unpriced_model_calls" in values and values["unpriced_model_calls"] != model_calls:
            raise AlgorithmOutputError(
                "external runtime usage without cost_usd must mark every model call unpriced"
            )
        values["unpriced_model_calls"] = model_calls
    else:
        values.setdefault("unpriced_model_calls", 0)
    for field in _EXTERNAL_USAGE_INTEGER_FIELDS:
        values.setdefault(field, 0)
    for field in _EXTERNAL_USAGE_FLOAT_FIELDS:
        values.setdefault(field, 0.0)
    try:
        usage = ResourceUsage(**values)
    except ValidationError as exc:
        raise AlgorithmOutputError(f"invalid external runtime usage: {exc}") from exc
    if usage.unpriced_model_calls > usage.model_calls:
        raise AlgorithmOutputError("unpriced_model_calls cannot exceed model_calls")
    return usage


def external_usage_counters(usage: ResourceUsage) -> dict[str, int | float]:
    """Return counters accepted by ``InterventionRunner`` trace reconciliation."""

    if not isinstance(usage, ResourceUsage):
        raise TypeError("usage must be a ResourceUsage")
    return {
        field: getattr(usage, field)
        for field in (*_EXTERNAL_USAGE_INTEGER_FIELDS, *_EXTERNAL_USAGE_FLOAT_FIELDS)
    }


def _request_reservation(request: object) -> ResourceUsage:
    planner = getattr(request, "external_reservation", None)
    if not callable(planner):
        return ResourceUsage()
    envelope = planner()
    if not isinstance(envelope, Mapping):
        raise ValidationError("external reservation plan must be a mapping")
    if any(not isinstance(key, str) for key in envelope):
        raise ValidationError("external reservation plan keys must be strings")
    unknown = set(envelope) - _EXTERNAL_RESERVATION_FIELDS
    if unknown:
        raise ValidationError(
            "external reservation plan contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    try:
        return ResourceUsage(**dict(envelope))
    except ValidationError as exc:
        raise ValidationError(f"invalid external reservation plan: {exc}") from exc


def _usage_delta(actual: ResourceUsage, reserved: ResourceUsage) -> ResourceUsage:
    values: dict[str, int | float] = {}
    for field in (*_EXTERNAL_USAGE_INTEGER_FIELDS, *_EXTERNAL_USAGE_FLOAT_FIELDS):
        difference = getattr(actual, field) - getattr(reserved, field)
        if difference < 0:
            raise AlgorithmOutputError(
                f"external runtime usage {field} is below its pre-execution reservation"
            )
        values[field] = difference
    return ResourceUsage(**values)


def _raise_for_declared_external_failure(request: object, result: object) -> None:
    hook = getattr(request, "raise_for_external_status", None)
    if callable(hook):
        hook(result)


def _response_usage(result: object) -> ResourceUsage:
    return normalize_external_usage(_response_usage_envelope(result))


def _response_usage_envelope(result: object) -> object:
    return result.get("usage") if isinstance(result, Mapping) else getattr(result, "usage", None)


@dataclass(slots=True)
class ObservedExternalRuntime:
    """Monitor a fingerprinted worker/runtime injected through the run context."""

    inner: Any
    monitor: RunMonitor
    ledger: BudgetLedger
    run_id: str

    @property
    def worker_id(self) -> str:
        return str(getattr(self.inner, "worker_id", "external-runtime"))

    def fingerprint(self) -> Mapping[str, Any]:
        fingerprint = getattr(self.inner, "fingerprint", None)
        if not callable(fingerprint):
            raise ValidationError("external runtime must expose fingerprint()")
        identity = fingerprint()
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError("external runtime fingerprint must be a non-empty mapping")
        return identity

    def call(self, payload: Any) -> Any:
        return self._execute("call", payload)

    def call_controlled(self, request: Any) -> Any:
        """Call a mapping-based worker while retaining request control hooks."""

        serializer = getattr(request, "as_dict", None)
        if not callable(serializer):
            raise ValidationError("controlled external request must expose as_dict()")
        return self._execute("call", request, delegated_request=serializer())

    def invoke(self, payload: Any) -> Any:
        return self._execute("invoke", payload)

    def reformulate(self, request: Any) -> Any:
        return self._execute("reformulate", request)

    def select(self, request: Any) -> Any:
        return self._execute("select", request)

    def run(self, request: Any) -> Any:
        return self._execute("run", request)

    def _execute(
        self,
        operation: str,
        request: Any,
        *,
        delegated_request: Any = None,
    ) -> Any:
        method = getattr(self.inner, operation, None)
        if not callable(method):
            raise ValidationError(f"external runtime does not implement {operation}()")
        component = f"external_runtime.{operation}"
        started = time.perf_counter_ns()
        self.monitor.emit(
            EventKind.STAGE_STARTED,
            run_id=self.run_id,
            component=component,
            payload={"runtime_fingerprint": dict(self.fingerprint())},
        )
        reservation = _request_reservation(request)
        if reservation != ResourceUsage():
            try:
                self.ledger.reserve_pending(reservation)
            except BudgetExceededError:
                self.monitor.emit(
                    EventKind.BUDGET_EXCEEDED,
                    run_id=self.run_id,
                    component=component,
                )
                raise
        try:
            result = method(request if delegated_request is None else delegated_request)
        except Exception as exc:
            self._record_resource_receipt(component=component)
            self._settle_runtime_exception(reservation, component=component)
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="external_runtime_error",
                stage=component,
            )
            raise
        self._record_resource_receipt(component=component)
        try:
            _raise_for_declared_external_failure(request, result)
        except (ExternalDependencyError, AlgorithmOutputError) as exc:
            # A declared model/algorithm failure can carry fully known actual
            # usage. It is distinct from a malformed usage envelope: settle
            # those tokens before propagating the original typed failure.
            self._settle_declared_failure(result, reservation, component=component)
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="external_runtime_error",
                stage=component,
            )
            raise
        try:
            usage = _response_usage(result)
            _usage_delta(usage, reservation)
        except AlgorithmOutputError as exc:
            # The external side effect already happened.  Preserve a known
            # minimum model call and unknown price before rejecting telemetry,
            # without double-counting a call already reserved above.
            minimum_actual = reservation + ResourceUsage(
                model_calls=0 if reservation.model_calls else 1,
                unpriced_model_calls=max(1, reservation.model_calls),
            )
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="external_usage_error",
                stage=component,
            )
            self._settle_actual(minimum_actual, reservation, component=component)
            raise
        self._settle_actual(usage, reservation, component=component)
        self.monitor.emit(
            EventKind.STAGE_FINISHED,
            run_id=self.run_id,
            component=component,
            payload={
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "usage": usage.as_dict(),
            },
        )
        return result

    def _record_resource_receipt(self, *, component: str) -> None:
        receipt = getattr(self.inner, "last_resource_receipt", None)
        if isinstance(receipt, ResourceReceipt):
            self.monitor.record_resource_receipt(
                receipt,
                run_id=self.run_id,
                component=component,
            )

    def _settle_actual(
        self,
        actual: ResourceUsage,
        reservation: ResourceUsage,
        *,
        component: str,
    ) -> None:
        delta = _usage_delta(actual, reservation)
        try:
            self.ledger.charge_actual(delta)
        except BudgetExceededError:
            self.ledger.commit_pending(reservation)
            self.monitor.record_usage(actual, run_id=self.run_id, component=component)
            self.monitor.emit(
                EventKind.BUDGET_EXCEEDED,
                run_id=self.run_id,
                component=component,
            )
            raise
        self.ledger.commit_pending(reservation)
        self.monitor.record_usage(actual, run_id=self.run_id, component=component)

    def _settle_declared_failure(
        self,
        result: object,
        reservation: ResourceUsage,
        *,
        component: str,
    ) -> None:
        envelope = _response_usage_envelope(result)
        explicitly_unconsumed = (
            isinstance(result, Mapping) and result.get("consumed") is False and envelope is None
        )
        if explicitly_unconsumed:
            self._cancel_pending(reservation)
            return
        try:
            actual = normalize_external_usage(envelope)
        except AlgorithmOutputError:
            self._commit_failed_minimum(reservation, component=component)
            return
        self._cancel_pending(reservation)
        try:
            self.ledger.charge_actual(actual)
        except BudgetExceededError:
            # The typed worker failure remains the primary cell-invalidating
            # exception, while the irreversible overage remains on the ledger.
            self.monitor.emit(
                EventKind.BUDGET_EXCEEDED,
                run_id=self.run_id,
                component=component,
            )
        self.monitor.record_usage(actual, run_id=self.run_id, component=component)

    def _commit_failed_minimum(
        self,
        reservation: ResourceUsage,
        *,
        component: str,
    ) -> None:
        """Retain the reserved lower bound unless zero consumption was explicit."""

        if reservation == ResourceUsage():
            return
        minimum_actual = reservation + ResourceUsage(unpriced_model_calls=reservation.model_calls)
        # The declared worker failure remains the primary exception.  The
        # reservation was accepted before the side effect and is therefore the
        # only trustworthy lower bound when failure telemetry is absent/broken.
        with suppress(BudgetExceededError):
            self._settle_actual(minimum_actual, reservation, component=component)

    def _settle_runtime_exception(
        self,
        reservation: ResourceUsage,
        *,
        component: str,
    ) -> None:
        if reservation == ResourceUsage():
            return
        minimum_actual = reservation + ResourceUsage(unpriced_model_calls=reservation.model_calls)
        # Preserve the runtime exception as the primary failure.  Any budget
        # overage remains sticky and the usage is already recorded.
        with suppress(BudgetExceededError):
            self._settle_actual(minimum_actual, reservation, component=component)

    def _cancel_pending(self, reservation: ResourceUsage) -> None:
        if reservation != ResourceUsage():
            self.ledger.cancel_pending(reservation)


@dataclass(slots=True)
class ObservedLLMBackend:
    inner: LLMBackend
    monitor: RunMonitor
    ledger: BudgetLedger
    run_id: str
    pricing: PricingCatalog | None = None

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def capabilities(self) -> Any:
        return self.inner.capabilities

    def fingerprint(self) -> Mapping[str, Any]:
        return self.inner.fingerprint()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        started = time.perf_counter_ns()
        self.monitor.emit(
            EventKind.STAGE_STARTED,
            run_id=self.run_id,
            component="llm",
            payload={"model_id": self.model_id, "request_hash": request.request_hash},
        )
        call_usage = ResourceUsage(model_calls=1)
        try:
            self.ledger.reserve(call_usage)
        except BudgetExceededError:
            self.monitor.emit(EventKind.BUDGET_EXCEEDED, run_id=self.run_id, component="llm")
            raise
        self.monitor.record_usage(call_usage, run_id=self.run_id, component="llm")
        try:
            response = self.inner.generate(request)
        except Exception as exc:
            # An invocation without a usable response still counts as an
            # attempt. Its token/cost consumption is unknown, even if a price
            # table exists; do not silently report the failed call as free.
            unknown_usage = ResourceUsage(unpriced_model_calls=1)
            try:
                self.ledger.charge_actual(unknown_usage)
            except BudgetExceededError:
                self.monitor.emit(EventKind.BUDGET_EXCEEDED, run_id=self.run_id, component="llm")
            self.monitor.record_usage(unknown_usage, run_id=self.run_id, component="llm")
            self.monitor.emit(
                EventKind.STAGE_FINISHED,
                run_id=self.run_id,
                component="llm",
                payload={"status": "failed", "usage_status": "unknown",
                         "token_counters": "known_lower_bound_only"},
            )
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="backend_error",
                stage="llm",
            )
            raise
        cost = response.usage.cost_usd
        unpriced_calls = 0
        if cost is None:
            catalog = self.pricing or PricingCatalog()
            if catalog.has_price(self.model_id):
                cost = catalog.estimate(
                    self.model_id,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            else:
                cost = 0.0
                unpriced_calls = 1
        usage = ResourceUsage(
            unpriced_model_calls=unpriced_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
        )
        try:
            self.ledger.charge_actual(usage)
        except BudgetExceededError:
            # The provider has already returned, so its usage is irreversible
            # and must remain visible even though the budget gate now fails.
            self.monitor.record_usage(usage, run_id=self.run_id, component="llm")
            self.monitor.emit(EventKind.BUDGET_EXCEEDED, run_id=self.run_id, component="llm")
            raise
        self.monitor.record_usage(usage, run_id=self.run_id, component="llm")
        self.monitor.emit(
            EventKind.STAGE_FINISHED,
            run_id=self.run_id,
            component="llm",
            payload={
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "finish_reason": response.finish_reason,
            },
        )
        return response


@dataclass(slots=True)
class ObservedRetriever:
    inner: Retriever
    monitor: RunMonitor
    ledger: BudgetLedger
    run_id: str

    @property
    def retriever_id(self) -> str:
        return self.inner.retriever_id

    def fingerprint(self) -> Mapping[str, Any]:
        return self.inner.fingerprint()

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        started = time.perf_counter_ns()
        self.monitor.emit(
            EventKind.STAGE_STARTED,
            run_id=self.run_id,
            component="retriever",
            payload={"top_k": top_k},
        )
        usage = ResourceUsage(retrieval_calls=1)
        try:
            self.ledger.reserve(usage)
        except BudgetExceededError:
            self.monitor.emit(
                EventKind.BUDGET_EXCEEDED,
                run_id=self.run_id,
                component="retriever",
            )
            raise
        self.monitor.record_usage(usage, run_id=self.run_id, component="retriever")
        try:
            results = self.inner.retrieve(query, top_k=top_k)
        except Exception as exc:
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="retrieval_error",
                stage="retriever",
            )
            raise
        self.monitor.emit(
            EventKind.STAGE_FINISHED,
            run_id=self.run_id,
            component="retriever",
            payload={
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "result_count": len(results),
            },
        )
        return results


@dataclass(slots=True)
class ObservedDenseEmbedder:
    inner: DenseEmbedder
    monitor: RunMonitor
    ledger: BudgetLedger
    run_id: str

    @property
    def embedder_id(self) -> str:
        return self.inner.embedder_id

    def fingerprint(self) -> Mapping[str, Any]:
        return self.inner.fingerprint()

    def embed(self, texts: Any) -> Any:
        started = time.perf_counter_ns()
        self.monitor.emit(
            EventKind.STAGE_STARTED,
            run_id=self.run_id,
            component="embedding",
            payload={"input_count": len(texts)},
        )
        usage = ResourceUsage(embedding_calls=1)
        self.ledger.reserve(usage)
        self.monitor.record_usage(usage, run_id=self.run_id, component="embedding")
        try:
            vectors = self.inner.embed(texts)
        except Exception as exc:
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="embedding_error",
                stage="embedding",
            )
            raise
        self.monitor.emit(
            EventKind.STAGE_FINISHED,
            run_id=self.run_id,
            component="embedding",
            payload={
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "vector_count": len(vectors),
            },
        )
        return vectors


@dataclass(slots=True)
class ObservedVectorSearcher:
    inner: VectorSearcher
    monitor: RunMonitor
    ledger: BudgetLedger
    run_id: str

    @property
    def searcher_id(self) -> str:
        return self.inner.searcher_id

    def fingerprint(self) -> Mapping[str, Any]:
        return self.inner.fingerprint()

    def search(self, vector: Any, *, top_k: int) -> Any:
        started = time.perf_counter_ns()
        self.monitor.emit(
            EventKind.STAGE_STARTED,
            run_id=self.run_id,
            component="vector_search",
            payload={"top_k": top_k},
        )
        usage = ResourceUsage(retrieval_calls=1)
        try:
            self.ledger.reserve(usage)
        except BudgetExceededError:
            self.monitor.emit(
                EventKind.BUDGET_EXCEEDED,
                run_id=self.run_id,
                component="vector_search",
            )
            raise
        self.monitor.record_usage(usage, run_id=self.run_id, component="vector_search")
        try:
            results = self.inner.search(vector, top_k=top_k)
        except Exception as exc:
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="retrieval_error",
                stage="vector_search",
            )
            raise
        self.monitor.emit(
            EventKind.STAGE_FINISHED,
            run_id=self.run_id,
            component="vector_search",
            payload={
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "result_count": len(results),
            },
        )
        return results


@dataclass(slots=True)
class ObservedToolConnector:
    inner: ToolConnector
    monitor: RunMonitor
    ledger: BudgetLedger
    run_id: str

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return self.inner.list_tools()

    def fingerprint(self) -> Mapping[str, Any]:
        return self.inner.fingerprint()

    def call(self, request: ToolCall) -> ToolResult:
        started = time.perf_counter_ns()
        self.monitor.emit(
            EventKind.STAGE_STARTED,
            run_id=self.run_id,
            component="tool",
            payload={"tool_name": request.name, "call_id": request.call_id},
        )
        usage = ResourceUsage(tool_calls=1)
        try:
            self.ledger.reserve(usage)
        except BudgetExceededError:
            self.monitor.emit(EventKind.BUDGET_EXCEEDED, run_id=self.run_id, component="tool")
            raise
        self.monitor.record_usage(usage, run_id=self.run_id, component="tool")
        try:
            result = self.inner.call(request)
        except Exception as exc:
            self.monitor.record_failure(
                exc,
                run_id=self.run_id,
                category="tool_error",
                stage="tool",
            )
            raise
        self.monitor.emit(
            EventKind.STAGE_FINISHED,
            run_id=self.run_id,
            component="tool",
            payload={
                "duration_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "error": bool(result.error),
            },
        )
        return result


def instrument_context(
    context: InterventionContext,
    state: FrozenAgentState,
    monitor: RunMonitor,
    *,
    run_id: str,
    pricing: PricingCatalog | None = None,
) -> InterventionContext:
    """Return a per-run context with transparent runtime accounting wrappers."""

    ledger = context.budget_ledger or BudgetLedger(state.budget)
    if not isinstance(ledger, BudgetLedger):
        raise TypeError("context.budget_ledger must be a BudgetLedger")
    if ledger.budget != state.budget:
        raise ValueError("context budget ledger does not match the frozen state budget")
    ledger.claim_for_run(run_id)
    backend = context.backend
    if isinstance(backend, ObservedLLMBackend):
        backend = backend.inner
    if backend is not None:
        backend = ObservedLLMBackend(backend, monitor, ledger, run_id, pricing)
    dense_embedder = context.dense_embedder
    if isinstance(dense_embedder, ObservedDenseEmbedder):
        dense_embedder = dense_embedder.inner
    if dense_embedder is not None:
        dense_embedder = ObservedDenseEmbedder(dense_embedder, monitor, ledger, run_id)
    retriever = context.retriever
    if isinstance(retriever, ObservedRetriever):
        retriever = retriever.inner
    if retriever is not None:
        retriever = ObservedRetriever(retriever, monitor, ledger, run_id)
    vector_searcher = context.vector_searcher
    if isinstance(vector_searcher, ObservedVectorSearcher):
        vector_searcher = vector_searcher.inner
    if vector_searcher is not None:
        vector_searcher = ObservedVectorSearcher(vector_searcher, monitor, ledger, run_id)
    tools = context.tools
    if isinstance(tools, ObservedToolConnector):
        tools = tools.inner
    if tools is not None:
        tools = ObservedToolConnector(tools, monitor, ledger, run_id)
    external_runtime = context.external_runtime
    if isinstance(external_runtime, ObservedExternalRuntime):
        external_runtime = external_runtime.inner
    if external_runtime is not None:
        external_runtime = ObservedExternalRuntime(external_runtime, monitor, ledger, run_id)
    return replace(
        context,
        backend=backend,
        dense_embedder=dense_embedder,
        retriever=retriever,
        vector_searcher=vector_searcher,
        tools=tools,
        external_runtime=external_runtime,
        budget_ledger=ledger,
        monitor=monitor,
    )
