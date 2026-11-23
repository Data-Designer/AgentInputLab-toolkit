"""Dependency-free JSON-lines worker transport.

The transport never invokes a shell.  It exchanges one bounded JSON object per
process invocation so legacy workers cannot silently leak arbitrary logs into
the structured result channel.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.observability.models import ResourceLimits, ResourceReceipt

from .supervisor import (
    AcceleratorSampler,
    LocalSubprocessSupervisor,
    SupervisedWorkerError,
    WorkerSupervisor,
)


@runtime_checkable
class WorkerClient(Protocol):
    worker_id: str

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return one decoded worker response or raise a typed error."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return image/source/protocol identity."""


@dataclass(slots=True)
class JsonLineSubprocessWorker:
    """Run a pinned worker command without shell interpolation."""

    command: tuple[str, ...]
    worker_id: str
    identity: Mapping[str, Any]
    timeout_seconds: float = 300.0
    max_input_bytes: int = 8 * 1024 * 1024
    max_output_bytes: int = 8 * 1024 * 1024
    max_stderr_bytes: int = 1024 * 1024
    environment: Mapping[str, str] | None = None
    pass_environment: tuple[str, ...] = ()
    working_directory: str | None = None
    resource_limits: ResourceLimits | None = None
    supervisor: WorkerSupervisor = field(default_factory=LocalSubprocessSupervisor)
    accelerator_sampler: AcceleratorSampler | None = None
    _calls: int = field(default=0, init=False, repr=False)
    _last_resource_receipt: ResourceReceipt | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.command = tuple(self.command)
        if not self.command or not all(isinstance(item, str) and item for item in self.command):
            raise ValidationError("worker command must be a non-empty argv tuple")
        if self.timeout_seconds <= 0:
            raise ValidationError("worker timeout_seconds must be positive")
        if min(self.max_input_bytes, self.max_output_bytes, self.max_stderr_bytes) < 1:
            raise ValidationError("worker byte limits must be positive")
        if not self.identity:
            raise ValidationError("worker requires explicit pinned identity")
        self.identity = freeze_mapping(self.identity)
        if self.environment is not None:
            self.environment = freeze_mapping(self.environment)
        self.pass_environment = tuple(sorted(set(self.pass_environment)))
        for name in self.pass_environment:
            if (
                not isinstance(name, str)
                or not name
                or not (name[0].isalpha() or name[0] == "_")
                or not all(character.isalnum() or character == "_" for character in name)
            ):
                raise ValidationError("passed worker environment names must be identifiers")
        if set(self.pass_environment) & set((self.environment or {}).keys()):
            raise ValidationError("passed worker environments cannot duplicate public values")
        if self.resource_limits is None:
            self.resource_limits = ResourceLimits(
                wall_timeout_seconds=self.timeout_seconds,
            )
        elif not isinstance(self.resource_limits, ResourceLimits):
            raise TypeError("resource_limits must be ResourceLimits")
        else:
            self.timeout_seconds = self.resource_limits.wall_timeout_seconds
        if not isinstance(self.supervisor, WorkerSupervisor):
            raise TypeError("supervisor must implement WorkerSupervisor")
        if self.accelerator_sampler is not None and not isinstance(
            self.accelerator_sampler,
            AcceleratorSampler,
        ):
            raise TypeError("accelerator_sampler must implement AcceleratorSampler")

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._last_resource_receipt = None
        missing = [name for name in self.pass_environment if not os.environ.get(name)]
        if missing:
            raise ExternalDependencyError(
                "external worker is missing passed environments: " + ", ".join(missing)
            )
        environment = dict(self.environment or {})
        environment.update({name: os.environ[name] for name in self.pass_environment})
        request = json.dumps(
            to_jsonable(dict(payload)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            completed = self.supervisor.execute(
                self.command,
                input_text=f"{request}\n",
                limits=self.resource_limits,
                max_input_bytes=self.max_input_bytes,
                max_stdout_bytes=self.max_output_bytes,
                max_stderr_bytes=self.max_stderr_bytes,
                cwd=self.working_directory,
                environment=environment or None,
                process_name=f"external worker {self.worker_id!r}",
                image_identity=self.identity,
                accelerator_sampler=self.accelerator_sampler,
            )
        except SupervisedWorkerError as exc:
            self._last_resource_receipt = exc.resource_receipt
            raise
        self._last_resource_receipt = completed.resource_receipt
        self._calls += 1
        if completed.returncode != 0:
            raise SupervisedWorkerError(
                f"external worker {self.worker_id!r} exited with code {completed.returncode}",
                resource_receipt=completed.resource_receipt,
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise SupervisedWorkerError(
                f"external worker {self.worker_id!r} must emit exactly one JSON line",
                resource_receipt=completed.resource_receipt,
            )
        try:
            response = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise SupervisedWorkerError(
                f"external worker {self.worker_id!r} emitted invalid JSON",
                resource_receipt=completed.resource_receipt,
            ) from exc
        if not isinstance(response, dict):
            raise SupervisedWorkerError(
                "external worker response must be a JSON object",
                resource_receipt=completed.resource_receipt,
            )
        return response

    @property
    def last_resource_receipt(self) -> ResourceReceipt | None:
        """Most recent call receipt, including failed supervised calls."""

        return self._last_resource_receipt

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            **dict(self.identity),
            "worker_id": self.worker_id,
            "protocol": "agentinputlab.jsonl.v1",
            "command_hash": content_hash(self.command),
            "working_directory_hash": content_hash(self.working_directory),
            "environment_keys": sorted((self.environment or {}).keys()),
            "environment_hash": content_hash(self.environment or {}),
            "passed_environment_names": self.pass_environment,
            "timeout_seconds": self.timeout_seconds,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "resource_limit_hash": self.resource_limits.limit_hash,
            "supervisor": dict(self.supervisor.fingerprint()),
        }
