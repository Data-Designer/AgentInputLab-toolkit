"""Shared, auditable transport for paper-specific external workers.

This module standardizes only the transport envelope, dependency fingerprints,
usage accounting, and identity checks. Method modules still validate the
original algorithm trajectory and materialize its exact intervention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExternalDependencyError,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.observability import ResourceUsage, normalize_external_usage

from .base import WorkerClient

_RESERVATION_FIELDS = frozenset({"model_calls", "retrieval_calls", "tool_calls", "embedding_calls"})
_ALGORITHM_OUTPUT_FAILURE_CODES = frozenset(
    {
        "BACKEND_OUTPUT_INVALID",
        "INVALID_JSON",
        "MODEL_RESPONSE_INVALID",
        "PATTERN_SELECTION_OUTSIDE_LIBRARY",
        "QUERY_PARSE_FAILURE",
        "TEXT_INVALID",
        "UPSTREAM_OUTPUT_INVALID",
    }
)


@dataclass(frozen=True, slots=True)
class MethodWorkerRequest:
    """Versioned request envelope used by one pinned upstream integration."""

    method_id: str
    release_id: str
    upstream_commit: str
    state_hash: str
    payload: Mapping[str, Any]
    dependency_fingerprints: Mapping[str, Any] = field(default_factory=dict)
    minimum_reservation: Mapping[str, int] = field(default_factory=dict)
    protocol_version: str = "agentinputlab.method-worker.v1"
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "method_id",
            "release_id",
            "upstream_commit",
            "state_hash",
            "protocol_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"worker request {name} must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise ValidationError("worker request payload must be a mapping")
        if not isinstance(self.dependency_fingerprints, Mapping):
            raise ValidationError("worker request dependency_fingerprints must be a mapping")
        try:
            payload = freeze_mapping(self.payload)
            dependencies = freeze_mapping(self.dependency_fingerprints)
        except TypeError as exc:
            raise ValidationError(f"worker request envelope is not canonical: {exc}") from exc
        reservation = _validate_reservation(self.minimum_reservation)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "dependency_fingerprints", dependencies)
        object.__setattr__(self, "minimum_reservation", reservation)
        identity = {
            "method_id": self.method_id,
            "release_id": self.release_id,
            "upstream_commit": self.upstream_commit,
            "state_hash": self.state_hash,
            "payload": payload,
            "dependency_fingerprints": dependencies,
            "minimum_reservation": reservation,
            "protocol_version": self.protocol_version,
        }
        object.__setattr__(self, "request_id", content_hash(identity))

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def external_reservation(self) -> Mapping[str, int]:
        return self.minimum_reservation

    def raise_for_external_status(self, result: object) -> None:
        if not isinstance(result, Mapping):
            return
        if str(result.get("status", "")) == "ok":
            return
        code = str(result.get("failure_code", "WORKER_FAILURE"))
        if code in _ALGORITHM_OUTPUT_FAILURE_CODES:
            raise AlgorithmOutputError(f"{self.release_id} worker failed with {code}")
        raise ExternalDependencyError(f"{self.release_id} worker failed with {code}")


@dataclass(frozen=True, slots=True)
class MethodWorkerResponse:
    """Identity-checked response envelope with normalized resource usage."""

    request_id: str
    upstream_commit: str
    protocol_version: str
    payload: Mapping[str, Any]
    usage: ResourceUsage

    def __post_init__(self) -> None:
        for name in ("request_id", "upstream_commit", "protocol_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"worker response {name} must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise ValidationError("worker response payload must be a mapping")
        if not isinstance(self.usage, ResourceUsage):
            raise ValidationError("worker response usage must be ResourceUsage")
        try:
            object.__setattr__(self, "payload", freeze_mapping(self.payload))
        except TypeError as exc:
            raise ValidationError(f"worker response payload is not canonical: {exc}") from exc

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        request: MethodWorkerRequest,
    ) -> MethodWorkerResponse:
        if str(value.get("status", "")) != "ok":
            code = str(value.get("failure_code", "WORKER_FAILURE"))
            raise ExternalDependencyError(f"{request.release_id} worker failed with {code}")
        request_id = required_text(value, "request_id", label="worker response")
        if request_id != request.request_id:
            raise AlgorithmOutputError("external worker response request_id mismatch")
        upstream_commit = required_text(value, "upstream_commit", label="worker response")
        if upstream_commit != request.upstream_commit:
            raise AlgorithmOutputError("external worker response upstream revision mismatch")
        protocol_version = required_text(value, "protocol_version", label="worker response")
        if protocol_version != request.protocol_version:
            raise AlgorithmOutputError("external worker response protocol version mismatch")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise AlgorithmOutputError("external worker response payload must be a mapping")
        return cls(
            request_id=request_id,
            upstream_commit=upstream_commit,
            protocol_version=protocol_version,
            payload=dict(payload),
            usage=normalize_external_usage(value.get("usage")),
        )


def resolve_worker(
    configured: WorkerClient | None,
    context: InterventionContext,
    *,
    label: str,
) -> WorkerClient:
    """Resolve one explicit worker without permitting hidden ambiguity."""

    injected = context.external_runtime
    if injected is not None and configured is not None and injected is not configured:
        raise ValidationError(
            f"{label} worker is ambiguous; use either the constructor or "
            "InterventionContext.external_runtime"
        )
    worker = injected if injected is not None else configured
    if worker is None:
        raise ExternalDependencyError(f"{label} requires a pinned external WorkerClient")
    if not isinstance(worker, WorkerClient):
        raise ValidationError(f"{label} external runtime does not implement WorkerClient")
    worker_fingerprint(worker, label=label)
    return worker


def call_worker(
    worker: WorkerClient,
    request: MethodWorkerRequest,
) -> MethodWorkerResponse:
    controlled = getattr(worker, "call_controlled", None)
    raw = controlled(request) if callable(controlled) else worker.call(request.as_dict())
    if not isinstance(raw, Mapping):
        raise AlgorithmOutputError("external worker returned a non-mapping response")
    return MethodWorkerResponse.from_mapping(raw, request=request)


def worker_fingerprint(worker: WorkerClient, *, label: str) -> Mapping[str, Any]:
    identity = worker.fingerprint()
    if not isinstance(identity, Mapping) or not identity:
        raise ValidationError(f"{label} worker fingerprint must be a non-empty mapping")
    try:
        return freeze_mapping(identity)
    except TypeError as exc:
        raise ValidationError(f"{label} worker fingerprint is not canonical: {exc}") from exc


def dependency_fingerprint(value: Any, *, label: str, required: bool = False) -> Mapping[str, Any]:
    if value is None:
        if required:
            raise ExternalDependencyError(f"{label} dependency is required")
        return {}
    fingerprint = getattr(value, "fingerprint", None)
    if not callable(fingerprint):
        raise ExternalDependencyError(f"{label} dependency must expose fingerprint()")
    identity = fingerprint()
    if not isinstance(identity, Mapping) or not identity:
        raise ValidationError(f"{label} fingerprint must be a non-empty mapping")
    try:
        return freeze_mapping(identity)
    except TypeError as exc:
        raise ValidationError(f"{label} fingerprint is not canonical: {exc}") from exc


def required_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise AlgorithmOutputError(f"{label} requires non-empty {key}")
    return result


def optional_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    result = value.get(key, "")
    if not isinstance(result, str):
        raise AlgorithmOutputError(f"{label} {key} must be text")
    return result


def string_tuple(
    value: Any,
    *,
    label: str,
    allow_empty: bool = True,
    unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlgorithmOutputError(f"{label} must be a sequence of strings")
    result = tuple(value)
    if not all(isinstance(item, str) and item.strip() for item in result):
        raise AlgorithmOutputError(f"{label} must contain non-empty strings")
    if not allow_empty and not result:
        raise AlgorithmOutputError(f"{label} must not be empty")
    if unique and len(result) != len(set(result)):
        raise AlgorithmOutputError(f"{label} must not contain duplicates")
    return result


def mapping_sequence(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlgorithmOutputError(f"{label} must be a sequence")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise AlgorithmOutputError(f"{label}[{index}] must be a mapping")
        result.append(item)
    return tuple(result)


def finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmOutputError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise AlgorithmOutputError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise AlgorithmOutputError(f"{label} must be finite")
    return result


def nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AlgorithmOutputError(f"{label} must be a non-negative integer")
    return value


def _validate_reservation(value: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise ValidationError("worker minimum_reservation must be a mapping")
    unknown = set(value) - _RESERVATION_FIELDS
    if unknown:
        raise ValidationError(
            "worker minimum_reservation contains unsupported fields: " + ", ".join(sorted(unknown))
        )
    normalized: dict[str, int] = {}
    for key, amount in value.items():
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValidationError(f"worker reservation {key} must be a non-negative integer")
        normalized[key] = amount
    return freeze_mapping(normalized)  # type: ignore[return-value]


__all__ = [
    "MethodWorkerRequest",
    "MethodWorkerResponse",
    "call_worker",
    "dependency_fingerprint",
    "finite_number",
    "mapping_sequence",
    "nonnegative_integer",
    "optional_text",
    "required_text",
    "resolve_worker",
    "string_tuple",
    "worker_fingerprint",
]
