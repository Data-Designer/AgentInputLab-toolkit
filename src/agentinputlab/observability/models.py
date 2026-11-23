"""Stable run-observability records shared by CLI, Studio, and experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.enums import StrEnum
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    USAGE_RECORDED = "usage_recorded"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTENT_CHECKED = "intent_checked"
    FAILURE_RECORDED = "failure_recorded"
    RESOURCE_RECEIPT_RECORDED = "resource_receipt_recorded"


class ObservedRunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTENT_REJECTED = "intent_rejected"


class ResourceExitReason(StrEnum):
    """Supervisor-classified reason that a worker process stopped."""

    COMPLETED = "completed"
    NONZERO_EXIT = "nonzero_exit"
    TIMED_OUT = "timed_out"
    OOM_KILLED = "oom_killed"
    STARTUP_FAILED = "startup_failed"
    SUPERVISOR_ERROR = "supervisor_error"


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Hard execution limits requested from a worker supervisor.

    ``cpus`` is a CPU quota (for example Docker ``--cpus``), not a measured
    duration.  ``wall_timeout_seconds`` remains a hard elapsed-time deadline.
    Optional limits are never treated as enforced unless a supervisor records
    the matching :attr:`limit_hash` in its receipt.
    """

    wall_timeout_seconds: float = 300.0
    memory_bytes: int | None = None
    cpus: float | None = None
    pids_limit: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.wall_timeout_seconds, bool)
            or not isinstance(self.wall_timeout_seconds, (int, float))
            or not math.isfinite(self.wall_timeout_seconds)
            or self.wall_timeout_seconds <= 0
        ):
            raise ValidationError("wall_timeout_seconds must be finite and positive")
        if self.memory_bytes is not None and (
            isinstance(self.memory_bytes, bool)
            or not isinstance(self.memory_bytes, int)
            or self.memory_bytes <= 0
        ):
            raise ValidationError("memory_bytes must be a positive integer when provided")
        if self.cpus is not None and (
            isinstance(self.cpus, bool)
            or not isinstance(self.cpus, (int, float))
            or not math.isfinite(self.cpus)
            or self.cpus <= 0
        ):
            raise ValidationError("cpus must be finite and positive when provided")
        if self.pids_limit is not None and (
            isinstance(self.pids_limit, bool)
            or not isinstance(self.pids_limit, int)
            or self.pids_limit <= 0
        ):
            raise ValidationError("pids_limit must be a positive integer when provided")
        object.__setattr__(self, "wall_timeout_seconds", float(self.wall_timeout_seconds))
        if self.cpus is not None:
            object.__setattr__(self, "cpus", float(self.cpus))

    @property
    def limit_hash(self) -> str:
        """Content identity of exactly the limits a supervisor must enforce."""

        return content_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class AcceleratorSample:
    """Optional accelerator accounting returned by a sampler.

    ``accelerator_seconds`` means device-seconds summed across devices. A
    sampler with utilization data should integrate utilization fraction over
    wall time; an allocation-only sampler may report allocated devices times
    wall time, but must identify that measurement mode in ``sampler_identity``.
    Thus two allocated devices for three seconds are six accelerator-seconds.

    ``None`` means the measurement was unavailable. It is intentionally
    distinct from a measured zero. Peak device memory is likewise optional and
    non-additive.
    """

    accelerator_seconds: float | None = None
    accelerator_peak_memory_bytes: int | None = None
    sampler_identity: Mapping[str, Any] = field(
        default_factory=lambda: {
            "status": "unavailable",
            "reason": "not_configured",
        }
    )

    def __post_init__(self) -> None:
        if self.accelerator_seconds is not None:
            if (
                isinstance(self.accelerator_seconds, bool)
                or not isinstance(self.accelerator_seconds, (int, float))
                or not math.isfinite(self.accelerator_seconds)
                or self.accelerator_seconds < 0
            ):
                raise ValidationError(
                    "accelerator_seconds must be finite and non-negative when provided"
                )
            object.__setattr__(
                self,
                "accelerator_seconds",
                float(self.accelerator_seconds),
            )
        if self.accelerator_peak_memory_bytes is not None and (
            isinstance(self.accelerator_peak_memory_bytes, bool)
            or not isinstance(self.accelerator_peak_memory_bytes, int)
            or self.accelerator_peak_memory_bytes < 0
        ):
            raise ValidationError(
                "accelerator_peak_memory_bytes must be a non-negative integer when provided"
            )
        if not self.sampler_identity:
            raise ValidationError("accelerator samples require sampler identity")
        identity = dict(self.sampler_identity)
        status = identity.get("status")
        measured = (
            self.accelerator_seconds is not None or self.accelerator_peak_memory_bytes is not None
        )
        if status is None:
            status = "available" if measured else "unavailable"
            identity["status"] = status
            if not measured:
                identity.setdefault("reason", "sampler_did_not_report")
        if status not in {"available", "unavailable"}:
            raise ValidationError("accelerator sampler status must be available or unavailable")
        if status == "unavailable" and measured:
            raise ValidationError("unavailable accelerator samples cannot contain measured values")
        if status == "available" and not measured:
            raise ValidationError(
                "available accelerator samples require seconds or peak-memory evidence"
            )
        object.__setattr__(self, "sampler_identity", freeze_mapping(identity))


@dataclass(frozen=True, slots=True)
class ResourceReceipt:
    """Immutable, non-additive execution evidence from one supervisor call.

    Peak memory belongs here rather than in :class:`ResourceUsage`: peaks do
    not compose by addition.  CPU and accelerator durations may separately be
    reconciled into additive usage by the caller, but this receipt is retained
    verbatim for audit.
    """

    limit_hash: str
    wall_seconds: float
    supervisor_identity: Mapping[str, Any]
    image_identity: Mapping[str, Any]
    cpu_seconds: float | None = None
    peak_rss_bytes: int | None = None
    accelerator_seconds: float | None = None
    accelerator_peak_memory_bytes: int | None = None
    oom_killed: bool = False
    timed_out: bool = False
    exit_reason: ResourceExitReason = ResourceExitReason.COMPLETED
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.limit_hash, str)
            or len(self.limit_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.limit_hash)
        ):
            raise ValidationError("limit_hash must be a lowercase SHA-256 hex digest")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, (int, float))
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0
        ):
            raise ValidationError("wall_seconds must be finite and non-negative")
        if self.cpu_seconds is not None and (
            isinstance(self.cpu_seconds, bool)
            or not isinstance(self.cpu_seconds, (int, float))
            or not math.isfinite(self.cpu_seconds)
            or self.cpu_seconds < 0
        ):
            raise ValidationError("cpu_seconds must be finite and non-negative when provided")
        if self.peak_rss_bytes is not None and (
            isinstance(self.peak_rss_bytes, bool)
            or not isinstance(self.peak_rss_bytes, int)
            or self.peak_rss_bytes < 0
        ):
            raise ValidationError("peak_rss_bytes must be a non-negative integer when provided")
        if self.accelerator_seconds is not None and (
            isinstance(self.accelerator_seconds, bool)
            or not isinstance(self.accelerator_seconds, (int, float))
            or not math.isfinite(self.accelerator_seconds)
            or self.accelerator_seconds < 0
        ):
            raise ValidationError(
                "accelerator_seconds must be finite and non-negative when provided"
            )
        if self.accelerator_peak_memory_bytes is not None and (
            isinstance(self.accelerator_peak_memory_bytes, bool)
            or not isinstance(self.accelerator_peak_memory_bytes, int)
            or self.accelerator_peak_memory_bytes < 0
        ):
            raise ValidationError(
                "accelerator_peak_memory_bytes must be a non-negative integer when provided"
            )
        if not isinstance(self.oom_killed, bool) or not isinstance(self.timed_out, bool):
            raise ValidationError("oom_killed and timed_out must be booleans")
        if isinstance(self.exit_code, bool) or (
            self.exit_code is not None and not isinstance(self.exit_code, int)
        ):
            raise ValidationError("exit_code must be an integer or null")
        if not isinstance(self.exit_reason, ResourceExitReason):
            try:
                object.__setattr__(self, "exit_reason", ResourceExitReason(self.exit_reason))
            except (TypeError, ValueError) as exc:
                raise ValidationError("exit_reason is not recognized") from exc
        if self.timed_out is not (self.exit_reason is ResourceExitReason.TIMED_OUT):
            raise ValidationError("timed_out must match exit_reason='timed_out'")
        if self.oom_killed is not (self.exit_reason is ResourceExitReason.OOM_KILLED):
            raise ValidationError("oom_killed must match exit_reason='oom_killed'")
        if self.timed_out and self.oom_killed:
            raise ValidationError("a receipt cannot be both timed out and OOM-killed")
        if self.exit_reason is ResourceExitReason.COMPLETED and self.exit_code not in (None, 0):
            raise ValidationError("completed receipts cannot have a non-zero exit_code")
        if self.exit_reason is ResourceExitReason.NONZERO_EXIT and self.exit_code in (None, 0):
            raise ValidationError("nonzero_exit receipts require a non-zero exit_code")
        if not self.supervisor_identity or not self.image_identity:
            raise ValidationError("resource receipts require supervisor and image identity")
        object.__setattr__(self, "wall_seconds", float(self.wall_seconds))
        if self.cpu_seconds is not None:
            object.__setattr__(self, "cpu_seconds", float(self.cpu_seconds))
        if self.accelerator_seconds is not None:
            object.__setattr__(
                self,
                "accelerator_seconds",
                float(self.accelerator_seconds),
            )
        object.__setattr__(
            self,
            "supervisor_identity",
            freeze_mapping(self.supervisor_identity),
        )
        object.__setattr__(self, "image_identity", freeze_mapping(self.image_identity))

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    model_calls: int = 0
    retrieval_calls: int = 0
    tool_calls: int = 0
    embedding_calls: int = 0
    unpriced_model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cpu_seconds: float = 0.0
    accelerator_seconds: float = 0.0
    cost_usd: float = 0.0
    compute_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "model_calls",
            "retrieval_calls",
            "tool_calls",
            "embedding_calls",
            "unpriced_model_calls",
            "input_tokens",
            "output_tokens",
        )
        if any(
            isinstance(getattr(self, field_name), bool)
            or not isinstance(getattr(self, field_name), int)
            or getattr(self, field_name) < 0
            for field_name in integer_fields
        ):
            raise ValidationError("resource usage counters must be non-negative integers")
        float_fields = ("cpu_seconds", "accelerator_seconds", "cost_usd", "compute_cost_usd")
        if any(
            isinstance(getattr(self, field_name), bool)
            or not isinstance(getattr(self, field_name), (int, float))
            or getattr(self, field_name) < 0
            or not math.isfinite(getattr(self, field_name))
            for field_name in float_fields
        ):
            raise ValidationError(
                "resource usage durations and costs must be finite and non-negative"
            )

    @property
    def total_cost_usd(self) -> float:
        return self.cost_usd + self.compute_cost_usd

    def __add__(self, other: ResourceUsage) -> ResourceUsage:
        if not isinstance(other, ResourceUsage):
            return NotImplemented
        return ResourceUsage(
            model_calls=self.model_calls + other.model_calls,
            retrieval_calls=self.retrieval_calls + other.retrieval_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            embedding_calls=self.embedding_calls + other.embedding_calls,
            unpriced_model_calls=(self.unpriced_model_calls + other.unpriced_model_calls),
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cpu_seconds=self.cpu_seconds + other.cpu_seconds,
            accelerator_seconds=self.accelerator_seconds + other.accelerator_seconds,
            cost_usd=self.cost_usd + other.cost_usd,
            compute_cost_usd=self.compute_cost_usd + other.compute_cost_usd,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["total_cost_usd"] = self.total_cost_usd
        return payload


@dataclass(frozen=True, slots=True)
class FailureRecord:
    category: str
    exception_type: str
    message: str
    stage: str = ""
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: EventKind
    timestamp_utc: str
    elapsed_ms: float
    component: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_mapping(self.payload))

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: str
    release_id: str
    state_hash: str
    status: ObservedRunStatus
    started_at_utc: str
    finished_at_utc: str | None
    duration_ms: float | None
    usage: ResourceUsage
    failure: FailureRecord | None
    events: tuple[RunEvent, ...]
    tags: Mapping[str, Any] = field(default_factory=dict)
    resource_receipts: tuple[ResourceReceipt, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "resource_receipts", tuple(self.resource_receipts))
        object.__setattr__(self, "tags", freeze_mapping(self.tags))

    def as_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["usage"] = self.usage.as_dict()
        return payload
