"""Auditable local and container supervisors for external workers.

The module has no Docker or accelerator SDK dependency.  Docker is invoked as
an argv-only client through an injectable bounded runner, and accelerator
accounting is an optional protocol implemented by the experiment environment.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.observability.models import (
    AcceleratorSample,
    ResourceExitReason,
    ResourceLimits,
    ResourceReceipt,
)

from .process import (
    BoundedProcessResult,
    BoundedProcessStartupError,
    BoundedProcessTimeoutError,
    ProcessResourceObservation,
    run_bounded_process,
)

_DIGEST_PINNED_IMAGE = re.compile(r"^.+@sha256:([0-9a-f]{64})$")
_SAFE_CONTAINER_PREFIX = re.compile(r"[^a-z0-9_.-]+")
WORKER_SUPERVISOR_KIND_ENV = "AILAB_WORKER_SUPERVISOR"
WORKER_SUPERVISOR_IMAGE_ENV = "AILAB_WORKER_IMAGE"
WORKER_SUPERVISOR_ASSET_ROOT_ENV = "AILAB_WORKER_ASSET_ROOT"
_LOCAL_LOCKED_LAUNCHER_PROTOCOL = "agentinputlab.local-locked-python-launcher.v1"
_DOCKER_LOCKED_LAUNCHER_PROTOCOL = "agentinputlab.docker-locked-python-launcher.v1"
_LOCKED_PYTHON_LAUNCHER = """\
import runpy
import sys

upstream, upstream_src, adapter, entrypoint, *arguments = sys.argv[1:]
trusted_paths = [upstream]
if upstream_src:
    trusted_paths.append(upstream_src)
if adapter != upstream:
    trusted_paths.append(adapter)
sys.path[:0] = trusted_paths
sys.argv = [entrypoint, *arguments]
runpy.run_path(entrypoint, run_name="__main__")
"""


@runtime_checkable
class AcceleratorSampler(Protocol):
    """Optional accelerator evidence for exactly one supervised execution.

    :meth:`stop` may return device-seconds and sampled peak device memory.
    Implementations must state in ``AcceleratorSample.sampler_identity``
    whether seconds are utilization-integrated or allocation-time accounting.
    Unavailable measurements are ``None``, never a fabricated zero.
    """

    def start(self) -> object:
        """Start a sample window and return an opaque, per-call token."""

    def stop(self, token: object) -> AcceleratorSample:
        """Stop the matching window and return immutable accounting."""


@runtime_checkable
class ProcessRunner(Protocol):
    """Injectable argv-only process runner used by both supervisors."""

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str,
        timeout_seconds: float,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        cwd: str | None,
        environment: Mapping[str, str] | None,
        process_name: str,
    ) -> BoundedProcessResult: ...


@dataclass(frozen=True, slots=True)
class DefaultProcessRunner:
    """Adapter around the library's hard wall-time/pipe-bounded transport."""

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str,
        timeout_seconds: float,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        cwd: str | None,
        environment: Mapping[str, str] | None,
        process_name: str,
    ) -> BoundedProcessResult:
        return run_bounded_process(
            argv,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
            max_input_bytes=max_input_bytes,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            cwd=cwd,
            environment=environment,
            process_name=process_name,
        )


@dataclass(frozen=True, slots=True)
class SupervisedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    resource_receipt: ResourceReceipt


class SupervisedWorkerError(ExternalDependencyError):
    """A supervisor failure retaining the audit receipt produced before exit."""

    def __init__(self, message: str, *, resource_receipt: ResourceReceipt) -> None:
        super().__init__(message)
        self.resource_receipt = resource_receipt


@runtime_checkable
class WorkerSupervisor(Protocol):
    """Execute one bounded worker request and return non-additive evidence."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return stable supervisor identity, excluding credentials."""

    def execute(
        self,
        argv: Sequence[str],
        *,
        input_text: str,
        limits: ResourceLimits,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        cwd: str | None,
        environment: Mapping[str, str] | None,
        process_name: str,
        image_identity: Mapping[str, Any] | None = None,
        accelerator_sampler: AcceleratorSampler | None = None,
    ) -> SupervisedProcessResult: ...


@dataclass(frozen=True, slots=True)
class LocalSubprocessSupervisor:
    """Host-process supervisor with a real hard wall deadline.

    The portable transport cannot guarantee a process-tree CPU quota, memory
    cgroup, or PID controller.  Requests for any of those limits are rejected
    before process creation rather than being mislabeled as enforced.
    """

    runner: ProcessRunner = field(default_factory=DefaultProcessRunner, repr=False)
    host_identity: Mapping[str, Any] = field(
        default_factory=lambda: {"kind": "host_process", "pinned": False}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_identity", freeze_mapping(self.host_identity))

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "kind": "local_subprocess",
            "protocol": "agentinputlab.worker-supervisor.v1",
            "hard_limits": ("wall_timeout_seconds", "pipe_bytes"),
            "optional_measurements": (
                "psutil_process_tree_cpu_rss",
                "pynvml_process_tree_allocation",
            ),
            "locked_python_launcher": {
                "protocol": _LOCAL_LOCKED_LAUNCHER_PROTOCOL,
                "source_sha256": content_hash(_LOCKED_PYTHON_LAUNCHER),
                "isolated_mode": True,
                "bytecode_writes": False,
                "import_order": "upstream_then_upstream_src_then_adapter",
            },
        }

    def execute_locked_worker(
        self,
        argv: Sequence[str],
        *,
        checkout: str | Path,
        adapter_checkout: str | Path,
        entrypoint_relative_path: str,
        input_text: str,
        limits: ResourceLimits,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        environment: Mapping[str, str] | None,
        process_name: str,
        image_identity: Mapping[str, Any] | None = None,
        accelerator_sampler: AcceleratorSampler | None = None,
    ) -> SupervisedProcessResult:
        """Execute an attested Python entrypoint without implicit script paths."""

        checkout_source = _host_bind_directory(checkout, label="worker checkout")
        adapter_source = _host_bind_directory(
            adapter_checkout,
            label="worker adapter checkout",
        )
        if checkout_source != adapter_source and (
            checkout_source.is_relative_to(adapter_source)
            or adapter_source.is_relative_to(checkout_source)
        ):
            raise ValidationError("worker checkout and adapter checkout must not partially overlap")
        relative = PurePosixPath(entrypoint_relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValidationError("worker entrypoint_relative_path must stay inside the adapter")
        expected_entrypoint = adapter_source.joinpath(*relative.parts)
        if len(argv) < 2 or Path(argv[1]).resolve() != expected_entrypoint:
            raise ValidationError("worker argv does not match the attested adapter entrypoint")
        upstream_src = checkout_source / "src"
        runtime_argv = (
            argv[0],
            "-I",
            "-B",
            "-c",
            _LOCKED_PYTHON_LAUNCHER,
            str(checkout_source),
            str(upstream_src) if upstream_src.is_dir() else "",
            str(adapter_source),
            str(expected_entrypoint),
            *tuple(argv[2:]),
        )
        return self.execute(
            runtime_argv,
            input_text=input_text,
            limits=limits,
            max_input_bytes=max_input_bytes,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            cwd=str(checkout_source),
            environment=environment,
            process_name=process_name,
            image_identity=image_identity,
            accelerator_sampler=accelerator_sampler,
        )

    def execute(
        self,
        argv: Sequence[str],
        *,
        input_text: str,
        limits: ResourceLimits,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        cwd: str | None,
        environment: Mapping[str, str] | None,
        process_name: str,
        image_identity: Mapping[str, Any] | None = None,
        accelerator_sampler: AcceleratorSampler | None = None,
    ) -> SupervisedProcessResult:
        if not isinstance(limits, ResourceLimits):
            raise TypeError("limits must be ResourceLimits")
        unsupported = [
            name
            for name, value in (
                ("memory_bytes", limits.memory_bytes),
                ("cpus", limits.cpus),
                ("pids_limit", limits.pids_limit),
            )
            if value is not None
        ]
        if unsupported:
            raise ValidationError(
                "local subprocess supervisor cannot guarantee hard "
                + ", ".join(unsupported)
                + " limits; use a container supervisor"
            )
        started = time.perf_counter()
        sample_token = _start_accelerator_sample(accelerator_sampler)
        try:
            result = self.runner.run(
                argv,
                input_text=input_text,
                timeout_seconds=limits.wall_timeout_seconds,
                max_input_bytes=max_input_bytes,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
                cwd=cwd,
                environment=environment,
                process_name=process_name,
            )
        except BoundedProcessTimeoutError as exc:
            observation = exc.resource_observation
            sample = _stop_accelerator_after_process_error(
                accelerator_sampler,
                sample_token,
                observation=observation,
            )
            receipt = self._receipt(
                limits=limits,
                wall_seconds=time.perf_counter() - started,
                sample=sample,
                observation=observation,
                image_identity=image_identity,
                timed_out=True,
                exit_reason=ResourceExitReason.TIMED_OUT,
            )
            raise SupervisedWorkerError(
                str(exc),
                resource_receipt=receipt,
            ) from exc
        except BoundedProcessStartupError as exc:
            observation = exc.resource_observation
            sample = _stop_accelerator_after_process_error(
                accelerator_sampler,
                sample_token,
                observation=observation,
            )
            receipt = self._receipt(
                limits=limits,
                wall_seconds=time.perf_counter() - started,
                sample=sample,
                observation=observation,
                image_identity=image_identity,
                exit_reason=ResourceExitReason.STARTUP_FAILED,
            )
            raise SupervisedWorkerError(
                str(exc),
                resource_receipt=receipt,
            ) from exc
        except ExternalDependencyError as exc:
            observation = _error_observation(exc)
            sample = _stop_accelerator_after_process_error(
                accelerator_sampler,
                sample_token,
                observation=observation,
            )
            receipt = self._receipt(
                limits=limits,
                wall_seconds=time.perf_counter() - started,
                sample=sample,
                observation=observation,
                image_identity=image_identity,
                exit_reason=ResourceExitReason.SUPERVISOR_ERROR,
            )
            raise SupervisedWorkerError(
                str(exc),
                resource_receipt=receipt,
            ) from exc
        try:
            sample = _stop_accelerator_sample(
                accelerator_sampler,
                sample_token,
                observation=result.resource_observation,
            )
        except Exception as exc:
            receipt = self._receipt(
                limits=limits,
                wall_seconds=time.perf_counter() - started,
                sample=AcceleratorSample(),
                observation=result.resource_observation,
                image_identity=image_identity,
                exit_code=result.returncode,
                exit_reason=ResourceExitReason.SUPERVISOR_ERROR,
            )
            raise SupervisedWorkerError(
                "accelerator sampler could not finalize",
                resource_receipt=receipt,
            ) from exc
        reason = (
            ResourceExitReason.COMPLETED
            if result.returncode == 0
            else ResourceExitReason.NONZERO_EXIT
        )
        receipt = self._receipt(
            limits=limits,
            wall_seconds=time.perf_counter() - started,
            sample=sample,
            observation=result.resource_observation,
            image_identity=image_identity,
            exit_code=result.returncode,
            exit_reason=reason,
        )
        return SupervisedProcessResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            resource_receipt=receipt,
        )

    def _receipt(
        self,
        *,
        limits: ResourceLimits,
        wall_seconds: float,
        sample: AcceleratorSample,
        observation: ProcessResourceObservation,
        image_identity: Mapping[str, Any] | None,
        exit_reason: ResourceExitReason,
        exit_code: int | None = None,
        timed_out: bool = False,
    ) -> ResourceReceipt:
        return ResourceReceipt(
            limit_hash=limits.limit_hash,
            wall_seconds=wall_seconds,
            cpu_seconds=observation.cpu_seconds,
            peak_rss_bytes=observation.peak_rss_bytes,
            accelerator_seconds=sample.accelerator_seconds,
            accelerator_peak_memory_bytes=(sample.accelerator_peak_memory_bytes),
            timed_out=timed_out,
            exit_reason=exit_reason,
            exit_code=exit_code,
            supervisor_identity={
                **dict(self.fingerprint()),
                "accelerator_sampler": dict(sample.sampler_identity),
                "measurement": dict(observation.measurement_identity),
            },
            image_identity=dict(image_identity or self.host_identity),
        )


@dataclass(frozen=True, slots=True)
class DockerWorkerSupervisor:
    """Digest-pinned, network-isolated Docker worker supervisor."""

    image: str
    runner: ProcessRunner = field(default_factory=DefaultProcessRunner, repr=False)
    docker_executable: str = "docker"
    name_prefix: str = "agentinputlab-worker"
    management_timeout_seconds: float = 15.0
    asset_root: str | None = None
    container_python: str = "/usr/local/bin/python"
    container_user: str = "10001:10001"
    tmpfs_size_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or not _DIGEST_PINNED_IMAGE.fullmatch(self.image):
            raise ValidationError("Docker worker image must be pinned as name@sha256:<64 hex>")
        if not isinstance(self.docker_executable, str) or not self.docker_executable:
            raise ValidationError("docker_executable must be a non-empty string")
        if (
            not isinstance(self.container_python, str)
            or not self.container_python.startswith("/")
            or any(character in self.container_python for character in ("\x00", "\n", "\r"))
        ):
            raise ValidationError("container_python must be an absolute argument-safe path")
        if self.container_user != "10001:10001":
            raise ValidationError("Docker worker container_user must be 10001:10001")
        if (
            isinstance(self.tmpfs_size_bytes, bool)
            or not isinstance(self.tmpfs_size_bytes, int)
            or self.tmpfs_size_bytes < 1024 * 1024
        ):
            raise ValidationError("Docker worker tmpfs_size_bytes must be at least 1 MiB")
        if self.asset_root is not None:
            asset_root = _host_bind_directory(self.asset_root, label="Docker worker asset_root")
            object.__setattr__(self, "asset_root", str(asset_root))
        if (
            isinstance(self.management_timeout_seconds, bool)
            or not isinstance(self.management_timeout_seconds, (int, float))
            or not math.isfinite(self.management_timeout_seconds)
            or self.management_timeout_seconds <= 0
        ):
            raise ValidationError("management_timeout_seconds must be finite and positive")
        prefix = _SAFE_CONTAINER_PREFIX.sub("-", self.name_prefix.casefold()).strip("-.")
        if not prefix:
            raise ValidationError("name_prefix must contain a Docker-safe character")
        object.__setattr__(self, "name_prefix", prefix[:40])

    def fingerprint(self) -> Mapping[str, Any]:
        digest = self.image.rsplit("@sha256:", 1)[1]
        return {
            "kind": "docker",
            "protocol": "agentinputlab.worker-supervisor.v1",
            "docker_executable": self.docker_executable,
            "image_digest": f"sha256:{digest}",
            "network": "none",
            "read_only": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "container_python": self.container_python,
            "container_user": self.container_user,
            "checkout_target": "/opt/upstream",
            "adapter_checkout_target": "/opt/candidate",
            "asset_target": "/opt/assets" if self.asset_root is not None else None,
            "tmpfs_size_bytes": self.tmpfs_size_bytes,
            "container_measurements": "runner_declared_supervised_workload_only",
            "locked_python_launcher": {
                "protocol": _DOCKER_LOCKED_LAUNCHER_PROTOCOL,
                "source_sha256": content_hash(_LOCKED_PYTHON_LAUNCHER),
                "isolated_mode": True,
                "bytecode_writes": False,
                "import_order": "upstream_then_upstream_src_then_adapter",
            },
        }

    def execute_locked_worker(
        self,
        argv: Sequence[str],
        *,
        checkout: str | Path,
        adapter_checkout: str | Path,
        entrypoint_relative_path: str,
        input_text: str,
        limits: ResourceLimits,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        environment: Mapping[str, str] | None,
        process_name: str,
        image_identity: Mapping[str, Any] | None = None,
        accelerator_sampler: AcceleratorSampler | None = None,
    ) -> SupervisedProcessResult:
        """Run one host-attested worker through fixed container paths.

        Host executable/checkouts remain part of the caller's attestation. The
        image digest binds the container Python runtime; only the exact
        entrypoint-relative path and configured worker arguments cross the
        boundary.
        """

        checkout_source = _host_bind_directory(checkout, label="worker checkout")
        adapter_source = _host_bind_directory(
            adapter_checkout,
            label="worker adapter checkout",
        )
        if checkout_source != adapter_source and (
            checkout_source.is_relative_to(adapter_source)
            or adapter_source.is_relative_to(checkout_source)
        ):
            raise ValidationError("worker checkout and adapter checkout must not partially overlap")
        asset_source = Path(self.asset_root) if self.asset_root is not None else None
        if asset_source is not None:
            for source in (checkout_source, adapter_source):
                if (
                    asset_source == source
                    or asset_source.is_relative_to(source)
                    or source.is_relative_to(asset_source)
                ):
                    raise ValidationError("worker asset root must not overlap either checkout")
        relative = PurePosixPath(entrypoint_relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValidationError("worker entrypoint_relative_path must stay inside the adapter")
        expected_host_entrypoint = adapter_source.joinpath(*relative.parts)
        if len(argv) < 2 or Path(argv[1]).resolve() != expected_host_entrypoint:
            raise ValidationError("worker argv does not match the attested adapter entrypoint")
        mapped_environment = _container_worker_environment(
            environment or {},
            checkout=checkout_source,
            adapter_checkout=adapter_source,
            asset_root=asset_source,
        )
        mounts: list[tuple[str, str]] = [
            (str(checkout_source), "/opt/upstream"),
            (str(adapter_source), "/opt/candidate"),
        ]
        if self.asset_root is not None:
            mounts.append((self.asset_root, "/opt/assets"))
        container_adapter = (
            PurePosixPath("/opt/upstream")
            if checkout_source == adapter_source
            else PurePosixPath("/opt/candidate")
        )
        container_argv = (
            "-I",
            "-B",
            "-c",
            _LOCKED_PYTHON_LAUNCHER,
            "/opt/upstream",
            "/opt/upstream/src" if (checkout_source / "src").is_dir() else "",
            str(container_adapter),
            str(container_adapter.joinpath(relative)),
            *tuple(argv[2:]),
        )
        return self.execute(
            container_argv,
            input_text=input_text,
            limits=limits,
            max_input_bytes=max_input_bytes,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
            cwd="/opt/upstream",
            environment=mapped_environment,
            process_name=process_name,
            image_identity=image_identity,
            accelerator_sampler=accelerator_sampler,
            container_mounts=tuple(mounts),
            container_entrypoint=self.container_python,
        )

    def execute(
        self,
        argv: Sequence[str],
        *,
        input_text: str,
        limits: ResourceLimits,
        max_input_bytes: int,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        cwd: str | None,
        environment: Mapping[str, str] | None,
        process_name: str,
        image_identity: Mapping[str, Any] | None = None,
        accelerator_sampler: AcceleratorSampler | None = None,
        container_mounts: Sequence[tuple[str, str]] = (),
        container_entrypoint: str | None = None,
    ) -> SupervisedProcessResult:
        if not isinstance(limits, ResourceLimits):
            raise TypeError("limits must be ResourceLimits")
        missing = [
            name
            for name, value in (
                ("memory_bytes", limits.memory_bytes),
                ("cpus", limits.cpus),
                ("pids_limit", limits.pids_limit),
            )
            if value is None
        ]
        if missing:
            raise ValidationError(
                "Docker supervisor requires explicit hard " + ", ".join(missing) + " limits"
            )
        name = f"{self.name_prefix}-{uuid.uuid4().hex}"
        command = self._run_command(
            name=name,
            argv=argv,
            limits=limits,
            cwd=cwd,
            environment=environment,
            mounts=container_mounts,
            entrypoint=container_entrypoint,
        )
        started = time.perf_counter()
        sample_token = _start_accelerator_sample(accelerator_sampler)
        result: BoundedProcessResult | None = None
        run_error: ExternalDependencyError | None = None
        run_error_reason: ResourceExitReason | None = None
        run_observation = ProcessResourceObservation()
        timed_out = False
        state: Mapping[str, Any] = {}
        cleanup_error: ExternalDependencyError | None = None
        try:
            try:
                result = self.runner.run(
                    command,
                    input_text=input_text,
                    timeout_seconds=limits.wall_timeout_seconds,
                    max_input_bytes=max_input_bytes,
                    max_stdout_bytes=max_stdout_bytes,
                    max_stderr_bytes=max_stderr_bytes,
                    cwd=None,
                    environment=environment,
                    process_name=process_name,
                )
            except BoundedProcessTimeoutError as exc:
                timed_out = True
                run_error = exc
                run_error_reason = ResourceExitReason.TIMED_OUT
                run_observation = exc.resource_observation
            except BoundedProcessStartupError as exc:
                run_error = exc
                run_error_reason = ResourceExitReason.STARTUP_FAILED
                run_observation = exc.resource_observation
            except ExternalDependencyError as exc:
                run_error = exc
                run_error_reason = ResourceExitReason.SUPERVISOR_ERROR
                run_observation = _error_observation(exc)
            if result is not None:
                run_observation = result.resource_observation
            run_observation = _docker_workload_observation(run_observation)
            state = self._inspect_state(name)
        finally:
            try:
                self._remove_container(name)
            except ExternalDependencyError as exc:
                cleanup_error = exc

        try:
            sample = _stop_accelerator_sample(
                accelerator_sampler,
                sample_token,
                observation=run_observation,
            )
        except Exception as exc:
            sample = AcceleratorSample()
            if run_error is None:
                run_error = ExternalDependencyError("accelerator sampler could not finalize")
                run_error.__cause__ = exc
                run_error_reason = ResourceExitReason.SUPERVISOR_ERROR

        oom_killed = bool(state.get("OOMKilled", False)) and not timed_out
        exit_code = _docker_exit_code(state, result)
        if timed_out:
            reason = ResourceExitReason.TIMED_OUT
        elif oom_killed:
            reason = ResourceExitReason.OOM_KILLED
        elif run_error is not None:
            reason = run_error_reason or ResourceExitReason.SUPERVISOR_ERROR
        elif cleanup_error is not None:
            reason = ResourceExitReason.SUPERVISOR_ERROR
        elif exit_code == 0:
            reason = ResourceExitReason.COMPLETED
        else:
            reason = ResourceExitReason.NONZERO_EXIT
        receipt = ResourceReceipt(
            limit_hash=limits.limit_hash,
            wall_seconds=time.perf_counter() - started,
            cpu_seconds=run_observation.cpu_seconds,
            peak_rss_bytes=run_observation.peak_rss_bytes,
            accelerator_seconds=sample.accelerator_seconds,
            accelerator_peak_memory_bytes=(sample.accelerator_peak_memory_bytes),
            oom_killed=oom_killed,
            timed_out=timed_out,
            exit_reason=reason,
            exit_code=exit_code,
            supervisor_identity={
                **dict(self.fingerprint()),
                "accelerator_sampler": dict(sample.sampler_identity),
                "measurement": dict(run_observation.measurement_identity),
            },
            image_identity={
                "reference": self.image,
                "digest": self.fingerprint()["image_digest"],
                "worker": dict(image_identity or {}),
            },
        )
        if run_error is not None:
            raise SupervisedWorkerError(
                str(run_error),
                resource_receipt=receipt,
            ) from run_error
        if cleanup_error is not None:
            raise SupervisedWorkerError(
                "Docker supervisor could not remove its worker container",
                resource_receipt=receipt,
            ) from cleanup_error
        if result is None:  # pragma: no cover - guarded by error branches
            raise AssertionError("Docker runner produced neither a result nor an error")
        return SupervisedProcessResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            resource_receipt=receipt,
        )

    def _run_command(
        self,
        *,
        name: str,
        argv: Sequence[str],
        limits: ResourceLimits,
        cwd: str | None,
        environment: Mapping[str, str] | None,
        mounts: Sequence[tuple[str, str]],
        entrypoint: str | None,
    ) -> tuple[str, ...]:
        command = [
            self.docker_executable,
            "run",
            "--name",
            name,
            "--init",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(limits.pids_limit),
            "--memory",
            f"{limits.memory_bytes}b",
            "--memory-swap",
            f"{limits.memory_bytes}b",
            "--cpus",
            _format_number(limits.cpus),
            "--user",
            self.container_user,
            "--tmpfs",
            (f"/tmp:rw,noexec,nosuid,nodev,size={self.tmpfs_size_bytes}"),
        ]
        for source, target in mounts:
            source_path = _host_bind_directory(source, label="Docker worker mount")
            if not target.startswith("/") or any(character in target for character in ",\x00\n\r"):
                raise ValidationError("Docker worker mount target is not argument-safe")
            command.extend(
                (
                    "--mount",
                    f"type=bind,source={source_path},target={target},readonly",
                )
            )
        if cwd is not None:
            command.extend(("--workdir", cwd))
        if entrypoint is not None:
            command.extend(("--entrypoint", entrypoint))
        for key in sorted((environment or {}).keys()):
            command.extend(("--env", key))
        command.extend(("-i", self.image, *tuple(argv)))
        return tuple(command)

    def _inspect_state(self, name: str) -> Mapping[str, Any]:
        try:
            result = self.runner.run(
                (
                    self.docker_executable,
                    "inspect",
                    "--format",
                    "{{json .State}}",
                    name,
                ),
                input_text="",
                timeout_seconds=self.management_timeout_seconds,
                max_input_bytes=1,
                max_stdout_bytes=64 * 1024,
                max_stderr_bytes=64 * 1024,
                cwd=None,
                environment=None,
                process_name="Docker worker state inspection",
            )
        except ExternalDependencyError:
            return {}
        if result.returncode != 0:
            return {}
        try:
            state = json.loads(result.stdout.strip())
        except (json.JSONDecodeError, TypeError):
            return {}
        return state if isinstance(state, dict) else {}

    def _remove_container(self, name: str) -> None:
        result = self.runner.run(
            (self.docker_executable, "rm", "--force", name),
            input_text="",
            timeout_seconds=self.management_timeout_seconds,
            max_input_bytes=1,
            max_stdout_bytes=64 * 1024,
            max_stderr_bytes=64 * 1024,
            cwd=None,
            environment=None,
            process_name="Docker worker cleanup",
        )
        if result.returncode != 0:
            raise ExternalDependencyError("Docker worker cleanup exited non-zero")


def worker_supervisor_from_environment(
    environment: Mapping[str, str] | None = None,
) -> WorkerSupervisor:
    """Resolve an explicit local-or-Docker supervisor without guessing.

    Local execution remains the compatibility default. Docker execution is
    enabled only when the operator sets ``AILAB_WORKER_SUPERVISOR=docker`` and
    supplies an immutable image identity.
    """

    values = os.environ if environment is None else environment
    kind = values.get(WORKER_SUPERVISOR_KIND_ENV, "local").strip().casefold()
    image = values.get(WORKER_SUPERVISOR_IMAGE_ENV, "").strip()
    asset_root = values.get(WORKER_SUPERVISOR_ASSET_ROOT_ENV, "").strip()
    if kind == "local":
        if image or asset_root:
            raise ValidationError(
                f"Docker worker image/asset settings require {WORKER_SUPERVISOR_KIND_ENV}=docker"
            )
        return LocalSubprocessSupervisor()
    if kind != "docker":
        raise ValidationError(f"{WORKER_SUPERVISOR_KIND_ENV} must be either local or docker")
    if not image:
        raise ValidationError(f"{WORKER_SUPERVISOR_IMAGE_ENV} is required for Docker supervision")
    return DockerWorkerSupervisor(
        image=image,
        asset_root=asset_root or None,
    )


def _host_bind_directory(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValidationError(f"{label} must be absolute")
    if raw == Path(raw.anchor):
        raise ValidationError(f"{label} must not be a filesystem root")
    if raw.is_symlink():
        raise ValidationError(f"{label} must not be a symlink")
    resolved = raw.resolve()
    if resolved != raw or not resolved.is_dir():
        raise ValidationError(f"{label} must be an existing canonical directory")
    if any(character in str(resolved) for character in ",\x00\n\r"):
        raise ValidationError(f"{label} contains a Docker-unsafe character")
    return resolved


def _container_worker_environment(
    environment: Mapping[str, str],
    *,
    checkout: Path,
    adapter_checkout: Path,
    asset_root: Path | None,
) -> Mapping[str, str]:
    roots: tuple[tuple[Path, PurePosixPath], ...] = (
        (checkout, PurePosixPath("/opt/upstream")),
        (adapter_checkout, PurePosixPath("/opt/candidate")),
        *(((asset_root, PurePosixPath("/opt/assets")),) if asset_root is not None else ()),
    )
    result: dict[str, str] = {}
    for key, raw_value in environment.items():
        if not isinstance(key, str) or not isinstance(raw_value, str):
            raise ValidationError("Docker worker environment must contain strings")
        if key == "PWD":
            result[key] = "/opt/upstream"
            continue
        if key == "PYTHONPATH":
            # The attested implementation checkout owns algorithm imports.
            # Keep the adapter root available only after the complete upstream
            # search path so a same-named candidate module cannot contaminate
            # an upstream parity arm.  ``candidate/src`` is intentionally not
            # exposed: the adapter executes by absolute path and may import
            # support packages from its root, while AgentInputLab itself comes
            # from the pinned worker image.
            paths = ["/opt/upstream"]
            if (checkout / "src").is_dir():
                paths.append("/opt/upstream/src")
            if adapter_checkout != checkout:
                paths.append("/opt/candidate")
            paths.append("/opt/agentinputlab")
            result[key] = ":".join(paths)
            continue
        value_path = Path(raw_value).expanduser()
        if value_path.is_absolute():
            canonical = value_path.resolve()
            mapped: str | None = None
            for host_root, container_root in roots:
                if canonical == host_root or canonical.is_relative_to(host_root):
                    relative = canonical.relative_to(host_root)
                    mapped = str(container_root.joinpath(*relative.parts))
                    break
            if mapped is None:
                raise ValidationError(
                    f"Docker worker environment {key!r} contains an unmapped host path"
                )
            result[key] = mapped
        else:
            result[key] = raw_value
    result["HOME"] = "/tmp/worker-home"
    return freeze_mapping(result)


def _start_accelerator_sample(sampler: AcceleratorSampler | None) -> object | None:
    return None if sampler is None else sampler.start()


def _stop_accelerator_sample(
    sampler: AcceleratorSampler | None,
    token: object | None,
    *,
    observation: ProcessResourceObservation,
) -> AcceleratorSample:
    if sampler is None:
        identity = observation.measurement_identity.get("accelerator", {})
        if not isinstance(identity, Mapping) or not identity:
            identity = {
                "status": "unavailable",
                "reason": "process_runner_did_not_report",
            }
        return AcceleratorSample(
            accelerator_seconds=observation.accelerator_seconds,
            accelerator_peak_memory_bytes=(observation.accelerator_peak_memory_bytes),
            sampler_identity=identity,
        )
    sample = sampler.stop(token)
    if not isinstance(sample, AcceleratorSample):
        raise ValidationError("accelerator sampler must return AcceleratorSample")
    return sample


def _error_observation(error: ExternalDependencyError) -> ProcessResourceObservation:
    observation = getattr(error, "resource_observation", None)
    return (
        observation
        if isinstance(observation, ProcessResourceObservation)
        else ProcessResourceObservation()
    )


def _stop_accelerator_after_process_error(
    sampler: AcceleratorSampler | None,
    token: object | None,
    *,
    observation: ProcessResourceObservation,
) -> AcceleratorSample:
    try:
        return _stop_accelerator_sample(
            sampler,
            token,
            observation=observation,
        )
    except Exception as exc:
        return AcceleratorSample(
            sampler_identity={
                "status": "unavailable",
                "reason": f"finalization_failed:{type(exc).__name__}",
            }
        )


def _docker_workload_observation(
    observation: ProcessResourceObservation,
) -> ProcessResourceObservation:
    process_identity = observation.measurement_identity.get("process_tree", {})
    if (
        isinstance(process_identity, Mapping)
        and process_identity.get("scope") == "supervised_workload"
    ):
        return observation
    return ProcessResourceObservation(
        measurement_identity={
            "process_tree": {
                "status": "unavailable",
                "reason": "container_process_is_not_a_child_of_the_docker_client",
                "launcher_observation": dict(observation.measurement_identity),
            },
            "accelerator": {
                "status": "unavailable",
                "reason": "container_process_is_not_a_child_of_the_docker_client",
            },
        }
    )


def _docker_exit_code(
    state: Mapping[str, Any],
    result: BoundedProcessResult | None,
) -> int | None:
    value = state.get("ExitCode")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return result.returncode if result is not None else None


def _format_number(value: float | None) -> str:
    if value is None:  # pragma: no cover - validated by caller
        raise AssertionError("required Docker numeric limit is missing")
    return format(value, ".15g")
