"""Strict external-worker clients.

The subprocess transport accepts an argv tuple and never invokes a shell.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agentinputlab.contracts.errors import ExternalDependencyError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.contracts.worker import WorkerRequest, WorkerResponse
from agentinputlab.integrations.external_workers.process import run_bounded_process


@runtime_checkable
class WorkerClient(Protocol):
    def invoke(self, request: WorkerRequest) -> WorkerResponse: ...


@dataclass(slots=True)
class FunctionWorkerClient(WorkerClient):
    fn: Callable[[WorkerRequest], WorkerResponse]

    def invoke(self, request: WorkerRequest) -> WorkerResponse:
        return self.fn(request)


@dataclass(slots=True)
class JsonLineWorkerClient(WorkerClient):
    argv: tuple[str, ...]
    timeout_seconds: float = 300.0
    max_request_bytes: int = 8 * 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024
    max_stderr_bytes: int = 1024 * 1024
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.argv = tuple(self.argv)
        if not self.argv:
            raise ValueError("worker argv must not be empty")
        if min(self.max_request_bytes, self.max_response_bytes, self.max_stderr_bytes) < 1:
            raise ValueError("worker byte limits must be positive")

    def invoke(self, request: WorkerRequest) -> WorkerResponse:
        payload = json.dumps(to_jsonable(request), sort_keys=True) + "\n"
        completed = run_bounded_process(
            self.argv,
            input_text=payload,
            timeout_seconds=self.timeout_seconds,
            max_input_bytes=self.max_request_bytes,
            max_stdout_bytes=self.max_response_bytes,
            max_stderr_bytes=self.max_stderr_bytes,
            cwd=self.cwd,
            environment=self.env,
            process_name="external worker",
        )
        if completed.returncode != 0:
            stderr = completed.stderr[-2000:]
            raise ExternalDependencyError(
                f"external worker exited {completed.returncode}: {stderr}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ExternalDependencyError("external worker must emit exactly one JSON line")
        try:
            data = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ExternalDependencyError("external worker returned invalid JSON") from exc
        try:
            response = WorkerResponse(**data)
        except (TypeError, ValueError) as exc:
            raise ExternalDependencyError("external worker returned an invalid envelope") from exc
        if response.request_id != request.request_id:
            raise ExternalDependencyError("external worker response request_id mismatch")
        return response
