"""In-process event monitor with redacted JSONL persistence."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.serialization import to_jsonable

from .models import (
    EventKind,
    FailureRecord,
    ObservedRunStatus,
    ResourceReceipt,
    ResourceUsage,
    RunEvent,
    RunSnapshot,
)

_ACTIVE_RUN: ContextVar[str | None] = ContextVar("agentinputlab_active_run", default=None)
_SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "private_key",
        "x_api_key",
    }
)
_SECRET_KEY_SUFFIXES = ("_api_key", "_password", "_secret", "_access_token", "_private_key")
_SECRET_TEXT_PATTERNS = (
    re.compile(r'''(?i)\b(?:api[_-]?key|access_token|refresh_token|token|password|secret)\s*[=:]\s*(?:"[^"\n]*"|'[^'\n]*'|[^\s&,;]+)'''),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
)


@runtime_checkable
class EventSink(Protocol):
    def write(self, event: RunEvent) -> None: ...


@dataclass(slots=True)
class JsonlEventSink:
    path: Path
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = RLock()

    def write(self, event: RunEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            event.as_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


@dataclass(slots=True)
class _MutableRun:
    run_id: str
    release_id: str
    state_hash: str
    started_at_utc: str
    started_ns: int
    started_cpu_ns: int
    status: ObservedRunStatus = ObservedRunStatus.RUNNING
    finished_at_utc: str | None = None
    duration_ms: float | None = None
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    failure: FailureRecord | None = None
    events: list[RunEvent] = field(default_factory=list)
    resource_receipts: list[ResourceReceipt] = field(default_factory=list)
    tags: Mapping[str, Any] = field(default_factory=dict)


class RunMonitor:
    """Collect a full run timeline without coupling algorithms to one vendor SDK."""

    def __init__(self, *, sinks: tuple[EventSink, ...] = ()) -> None:
        self._sinks = tuple(sinks)
        self._runs: dict[str, _MutableRun] = {}
        self._lock = RLock()

    @property
    def active_run_id(self) -> str | None:
        return _ACTIVE_RUN.get()

    @contextmanager
    def run_scope(
        self,
        *,
        release_id: str,
        state_hash: str,
        tags: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        run_id = self.start_run(release_id=release_id, state_hash=state_hash, tags=tags)
        token = _ACTIVE_RUN.set(run_id)
        try:
            yield run_id
        except Exception as exc:
            snapshot = self.snapshot(run_id)
            if snapshot.status is ObservedRunStatus.RUNNING:
                if snapshot.failure is None:
                    self.record_failure(exc, run_id=run_id)
                self.finish_run(ObservedRunStatus.FAILED, run_id=run_id)
            raise
        else:
            if self.snapshot(run_id).status is ObservedRunStatus.RUNNING:
                self.finish_run(ObservedRunStatus.SUCCEEDED, run_id=run_id)
        finally:
            _ACTIVE_RUN.reset(token)

    def start_run(
        self,
        *,
        release_id: str,
        state_hash: str,
        tags: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        identity = run_id or uuid.uuid4().hex
        now = _utc_now()
        with self._lock:
            if identity in self._runs:
                raise ValueError(f"run already exists: {identity}")
            self._runs[identity] = _MutableRun(
                run_id=identity,
                release_id=release_id,
                state_hash=state_hash,
                started_at_utc=now,
                started_ns=time.perf_counter_ns(),
                started_cpu_ns=time.process_time_ns(),
                tags=_redact(tags or {}),
            )
        self.emit(EventKind.RUN_STARTED, run_id=identity, payload={"tags": tags or {}})
        return identity

    def emit(
        self,
        kind: EventKind,
        *,
        run_id: str | None = None,
        component: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> RunEvent:
        identity = self._resolve_run_id(run_id)
        with self._lock:
            run = self._runs[identity]
            if run.status is not ObservedRunStatus.RUNNING:
                is_single_terminal_event = kind is EventKind.RUN_FINISHED and not any(
                    event.kind is EventKind.RUN_FINISHED for event in run.events
                )
                if not is_single_terminal_event:
                    raise RuntimeError(f"cannot append events to finished run: {identity}")
            elif kind is EventKind.RUN_FINISHED:
                raise RuntimeError("finish runs through finish_run(), not emit()")
            event = RunEvent(
                run_id=identity,
                sequence=len(run.events) + 1,
                kind=kind,
                timestamp_utc=_utc_now(),
                elapsed_ms=(time.perf_counter_ns() - run.started_ns) / 1_000_000,
                component=component,
                payload=_redact(payload or {}),
            )
            run.events.append(event)
        for sink in self._sinks:
            sink.write(event)
        return event

    def record_usage(
        self,
        usage: ResourceUsage,
        *,
        run_id: str | None = None,
        component: str = "",
    ) -> ResourceUsage:
        identity = self._resolve_run_id(run_id)
        with self._lock:
            run = self._runs[identity]
            if run.status is not ObservedRunStatus.RUNNING:
                raise RuntimeError(f"cannot record usage on finished run: {identity}")
            run.usage = run.usage + usage
            total = run.usage
        self.emit(
            EventKind.USAGE_RECORDED,
            run_id=identity,
            component=component,
            payload={"delta": usage.as_dict(), "total": total.as_dict()},
        )
        return total

    def record_failure(
        self,
        error: BaseException,
        *,
        run_id: str | None = None,
        category: str = "runtime_error",
        stage: str = "",
        retryable: bool = False,
    ) -> FailureRecord:
        identity = self._resolve_run_id(run_id)
        failure = FailureRecord(
            category=category,
            exception_type=type(error).__name__,
            message=_safe_error_message(error),
            stage=stage,
            retryable=retryable,
        )
        with self._lock:
            run = self._runs[identity]
            if run.status is not ObservedRunStatus.RUNNING:
                raise RuntimeError(f"cannot record failure on finished run: {identity}")
            run.failure = failure
        self.emit(
            EventKind.FAILURE_RECORDED,
            run_id=identity,
            component=stage,
            payload=to_jsonable(failure),
        )
        return failure

    def record_resource_receipt(
        self,
        receipt: ResourceReceipt,
        *,
        run_id: str | None = None,
        component: str = "",
    ) -> ResourceReceipt:
        """Attach non-additive worker resource evidence to a running run.

        Identity mappings pass through the same secret redaction policy as
        normal event payloads before either the snapshot or an event sink sees
        them.  This method deliberately does not mutate additive
        :class:`ResourceUsage`; callers reconcile CPU/accelerator durations
        explicitly so a worker envelope cannot be double-counted.
        """

        if not isinstance(receipt, ResourceReceipt):
            raise TypeError("receipt must be a ResourceReceipt")
        identity = self._resolve_run_id(run_id)
        sanitized = replace(
            receipt,
            supervisor_identity=_redact(receipt.supervisor_identity),
            image_identity=_redact(receipt.image_identity),
        )
        with self._lock:
            run = self._runs[identity]
            if run.status is not ObservedRunStatus.RUNNING:
                raise RuntimeError(f"cannot record resource receipt on finished run: {identity}")
            run.resource_receipts.append(sanitized)
        self.emit(
            EventKind.RESOURCE_RECEIPT_RECORDED,
            run_id=identity,
            component=component,
            payload=sanitized.as_dict(),
        )
        return sanitized

    def finish_run(
        self,
        status: ObservedRunStatus,
        *,
        run_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> RunSnapshot:
        if not isinstance(status, ObservedRunStatus):
            raise TypeError("status must be an ObservedRunStatus")
        if status is ObservedRunStatus.RUNNING:
            raise ValueError("cannot finish a run with status='running'")
        identity = self._resolve_run_id(run_id)
        with self._lock:
            run = self._runs[identity]
            if run.status is not ObservedRunStatus.RUNNING:
                raise ValueError(f"run is already finished: {identity}")
            cpu_seconds = (time.process_time_ns() - run.started_cpu_ns) / 1_000_000_000
            if cpu_seconds:
                self.record_usage(
                    ResourceUsage(cpu_seconds=cpu_seconds),
                    run_id=identity,
                    component="process",
                )
            run.status = status
            run.finished_at_utc = _utc_now()
            run.duration_ms = (time.perf_counter_ns() - run.started_ns) / 1_000_000
            self.emit(
                EventKind.RUN_FINISHED,
                run_id=identity,
                payload={"status": status.value, **dict(payload or {})},
            )
            return self.snapshot(identity)

    def snapshot(self, run_id: str) -> RunSnapshot:
        with self._lock:
            run = self._runs[run_id]
            return RunSnapshot(
                run_id=run.run_id,
                release_id=run.release_id,
                state_hash=run.state_hash,
                status=run.status,
                started_at_utc=run.started_at_utc,
                finished_at_utc=run.finished_at_utc,
                duration_ms=run.duration_ms,
                usage=run.usage,
                failure=run.failure,
                events=tuple(run.events),
                tags=run.tags,
                resource_receipts=tuple(run.resource_receipts),
            )

    def list_snapshots(self) -> tuple[RunSnapshot, ...]:
        with self._lock:
            identities = tuple(self._runs)
        return tuple(self.snapshot(identity) for identity in identities)

    def _resolve_run_id(self, run_id: str | None) -> str:
        identity = run_id or _ACTIVE_RUN.get()
        if not identity:
            raise RuntimeError("no active monitored run")
        with self._lock:
            if identity not in self._runs:
                raise KeyError(f"unknown run: {identity}")
        return identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_error_message(error: BaseException) -> str:
    return _redact_secret_text(str(error)[:2048])


def _redact(value: Any, *, key: str = "") -> Any:
    if _is_secret_key(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (tuple, list)):
        return tuple(_redact(item) for item in value)
    if isinstance(value, str):
        return _redact_secret_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return normalized in _SECRET_KEY_NAMES or normalized.endswith(_SECRET_KEY_SUFFIXES)


def _redact_secret_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_TEXT_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted
