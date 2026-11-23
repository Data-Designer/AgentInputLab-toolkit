"""Bounded, secret-minimizing subprocess execution for worker protocols."""

from __future__ import annotations

import importlib
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import freeze_mapping

_INHERITED_ENVIRONMENT_KEYS = (
    "LANG",
    "LC_ALL",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "WINDIR",
)
_RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.02


def _unavailable_measurement_identity(
    reason: str = "runner_did_not_report",
) -> dict[str, Any]:
    return {
        "process_tree": {
            "status": "unavailable",
            "reason": reason,
        },
        "accelerator": {
            "status": "unavailable",
            "reason": reason,
        },
    }


@dataclass(frozen=True, slots=True)
class ProcessResourceObservation:
    """Best-effort measurements from exactly one spawned process tree.

    A missing value is not a measured zero. ``measurement_identity`` records
    why each optional sampler was available or unavailable.
    """

    cpu_seconds: float | None = None
    peak_rss_bytes: int | None = None
    accelerator_seconds: float | None = None
    accelerator_peak_memory_bytes: int | None = None
    measurement_identity: Mapping[str, Any] = field(
        default_factory=_unavailable_measurement_identity
    )

    def __post_init__(self) -> None:
        for name in ("cpu_seconds", "accelerator_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValidationError(f"{name} must be finite and non-negative when provided")
            if value is not None:
                object.__setattr__(self, name, float(value))
        for name in ("peak_rss_bytes", "accelerator_peak_memory_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValidationError(f"{name} must be a non-negative integer when provided")
        identity = self.measurement_identity or _unavailable_measurement_identity()
        process_identity = identity.get("process_tree")
        accelerator_identity = identity.get("accelerator")
        if not isinstance(process_identity, Mapping) or not isinstance(
            accelerator_identity,
            Mapping,
        ):
            raise ValidationError(
                "measurement_identity requires process_tree and accelerator objects"
            )
        if process_identity.get("status") not in {"available", "unavailable"}:
            raise ValidationError("process-tree measurement status is invalid")
        if accelerator_identity.get("status") not in {"available", "unavailable"}:
            raise ValidationError("accelerator measurement status is invalid")
        if (
            self.cpu_seconds is not None or self.peak_rss_bytes is not None
        ) and process_identity.get("status") == "unavailable":
            raise ValidationError(
                "unavailable process-tree measurements cannot contain measured values"
            )
        if (
            self.accelerator_seconds is not None or self.accelerator_peak_memory_bytes is not None
        ) and accelerator_identity.get("status") == "unavailable":
            raise ValidationError(
                "unavailable accelerator measurements cannot contain measured values"
            )
        object.__setattr__(self, "measurement_identity", freeze_mapping(identity))


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    cpu_seconds: float | None = None
    peak_rss_bytes: int | None = None
    accelerator_seconds: float | None = None
    accelerator_peak_memory_bytes: int | None = None
    measurement_identity: Mapping[str, Any] = field(
        default_factory=_unavailable_measurement_identity
    )

    def __post_init__(self) -> None:
        identity = self.measurement_identity
        if not identity or identity == _unavailable_measurement_identity():
            identity = _reported_measurement_identity(
                has_process_measurement=(
                    self.cpu_seconds is not None or self.peak_rss_bytes is not None
                ),
                has_accelerator_measurement=(
                    self.accelerator_seconds is not None
                    or self.accelerator_peak_memory_bytes is not None
                ),
            )
        observation = ProcessResourceObservation(
            cpu_seconds=self.cpu_seconds,
            peak_rss_bytes=self.peak_rss_bytes,
            accelerator_seconds=self.accelerator_seconds,
            accelerator_peak_memory_bytes=self.accelerator_peak_memory_bytes,
            measurement_identity=identity,
        )
        object.__setattr__(self, "cpu_seconds", observation.cpu_seconds)
        object.__setattr__(self, "peak_rss_bytes", observation.peak_rss_bytes)
        object.__setattr__(self, "accelerator_seconds", observation.accelerator_seconds)
        object.__setattr__(
            self,
            "accelerator_peak_memory_bytes",
            observation.accelerator_peak_memory_bytes,
        )
        object.__setattr__(
            self,
            "measurement_identity",
            observation.measurement_identity,
        )

    @property
    def resource_observation(self) -> ProcessResourceObservation:
        return ProcessResourceObservation(
            cpu_seconds=self.cpu_seconds,
            peak_rss_bytes=self.peak_rss_bytes,
            accelerator_seconds=self.accelerator_seconds,
            accelerator_peak_memory_bytes=self.accelerator_peak_memory_bytes,
            measurement_identity=self.measurement_identity,
        )


class BoundedProcessError(ExternalDependencyError):
    """A bounded transport failure with any observation collected before exit."""

    def __init__(
        self,
        message: str,
        *,
        resource_observation: ProcessResourceObservation | None = None,
    ) -> None:
        super().__init__(message)
        self.resource_observation = resource_observation or ProcessResourceObservation()


class BoundedProcessTimeoutError(BoundedProcessError):
    """A bounded process exceeded its hard wall-clock deadline."""


class BoundedProcessStartupError(BoundedProcessError):
    """The requested executable could not be started."""


class BoundedProcessProtocolError(BoundedProcessError):
    """The child process violated the bounded pipe transport."""


class BoundedProcessOutputLimitError(BoundedProcessError):
    """A bounded process exceeded a configured pipe byte limit."""


class _ProcessTreeMonitor:
    """Poll a spawned process tree with optional psutil and NVML dependencies."""

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil, psutil_reason = _optional_module("psutil")
        self._known_processes: dict[int, Any] = {}
        self._latest_cpu_by_pid: dict[int, float] = {}
        self._peak_rss_bytes = 0
        self._cpu_samples = 0
        self._rss_samples = 0
        self._process_errors = 0
        self._psutil_reason = psutil_reason
        if self._psutil is not None:
            try:
                root = self._psutil.Process(pid)
                self._known_processes[int(root.pid)] = root
            except Exception as exc:  # optional telemetry must not break execution
                self._psutil_reason = f"initialization_failed:{type(exc).__name__}"
                self._psutil = None

        self._nvml, nvml_reason = _optional_module("pynvml")
        self._nvml_reason = nvml_reason
        self._nvml_handles: tuple[Any, ...] = ()
        self._nvml_initialized = False
        self._nvml_failed = False
        self._accelerator_samples = 0
        self._accelerator_seconds = 0.0
        self._accelerator_peak_memory_bytes = 0
        self._last_accelerator_sample_at: float | None = None
        self._last_active_device_count = 0
        if self._psutil is None and self._nvml is not None:
            self._nvml_reason = "psutil_required_for_process_attribution"
            self._nvml = None
        if self._nvml is not None:
            try:
                self._nvml.nvmlInit()
                self._nvml_initialized = True
                count = int(self._nvml.nvmlDeviceGetCount())
                self._nvml_handles = tuple(
                    self._nvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)
                )
            except Exception as exc:  # driver/library absence is an expected optional state
                self._nvml_reason = f"initialization_failed:{type(exc).__name__}"
                if self._nvml_initialized:
                    with suppress(Exception):
                        self._nvml.nvmlShutdown()
                    self._nvml_initialized = False
                self._nvml = None
                self._nvml_handles = ()

    def start(self) -> None:
        if self._psutil is None:
            return
        self._sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"agentinputlab-resource-monitor-{self._pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> ProcessResourceObservation:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, _RESOURCE_SAMPLE_INTERVAL_SECONDS * 5))
        if self._psutil is not None:
            self._sample()
        if self._nvml_initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:
                self._nvml_reason = f"shutdown_failed:{type(exc).__name__}"
                self._nvml = None
            finally:
                self._nvml_initialized = False

        process_available = self._psutil is not None
        cpu_available = process_available and self._cpu_samples > 0
        rss_available = process_available and self._rss_samples > 0
        accelerator_available = (
            self._nvml is not None and not self._nvml_failed and self._accelerator_samples > 0
        )
        cpu_reason = None if cpu_available else (self._psutil_reason or "no_successful_samples")
        rss_reason = None if rss_available else (self._psutil_reason or "no_successful_samples")
        accelerator_reason = (
            None if accelerator_available else (self._nvml_reason or "no_successful_samples")
        )
        process_identity = {
            "status": "available" if process_available else "unavailable",
            "sampler": "psutil.process_tree.v1",
            "scope": "spawned_process_tree",
            "dependency_version": _module_version(self._psutil),
            "sample_interval_seconds": _RESOURCE_SAMPLE_INTERVAL_SECONDS,
            "samples": max(self._cpu_samples, self._rss_samples),
            "sampling_errors": self._process_errors,
            "cpu_seconds": {
                "status": "available" if cpu_available else "unavailable",
                "reason": cpu_reason,
            },
            "peak_rss_bytes": {
                "status": "available" if rss_available else "unavailable",
                "reason": rss_reason,
                "semantics": "maximum sampled sum of resident bytes across the process tree",
            },
        }
        accelerator_identity = {
            "status": "available" if accelerator_available else "unavailable",
            "sampler": "pynvml.process_tree_allocation.v1",
            "scope": "spawned_process_tree",
            "dependency_version": _module_version(self._nvml),
            "reason": accelerator_reason,
            "samples": self._accelerator_samples,
            "seconds_semantics": "allocated device-seconds for matched process-tree PIDs",
            "peak_memory_semantics": (
                "maximum sampled sum of NVML process memory across matched devices"
            ),
        }
        return ProcessResourceObservation(
            cpu_seconds=(sum(self._latest_cpu_by_pid.values()) if cpu_available else None),
            peak_rss_bytes=self._peak_rss_bytes if rss_available else None,
            accelerator_seconds=(self._accelerator_seconds if accelerator_available else None),
            accelerator_peak_memory_bytes=(
                self._accelerator_peak_memory_bytes if accelerator_available else None
            ),
            measurement_identity={
                "process_tree": process_identity,
                "accelerator": accelerator_identity,
            },
        )

    def _sample_loop(self) -> None:
        while not self._stop.wait(_RESOURCE_SAMPLE_INTERVAL_SECONDS):
            self._sample()

    def _sample(self) -> None:
        if self._psutil is None:
            return
        processes = list(self._known_processes.values())
        for process in tuple(processes):
            try:
                children = process.children(recursive=True)
            except Exception:
                self._process_errors += 1
                continue
            for child in children:
                pid = int(child.pid)
                if pid not in self._known_processes:
                    self._known_processes[pid] = child
                    processes.append(child)

        rss_total = 0
        rss_observed = False
        cpu_observed = False
        active_pids: set[int] = set()
        for process in processes:
            pid = int(process.pid)
            try:
                cpu = process.cpu_times()
                cpu_seconds = float(cpu.user) + float(cpu.system)
                self._latest_cpu_by_pid[pid] = max(
                    cpu_seconds,
                    self._latest_cpu_by_pid.get(pid, 0.0),
                )
                cpu_observed = True
                active_pids.add(pid)
            except Exception:
                self._process_errors += 1
            try:
                rss_total += int(process.memory_info().rss)
                rss_observed = True
                active_pids.add(pid)
            except Exception:
                self._process_errors += 1
        if cpu_observed:
            self._cpu_samples += 1
        if rss_observed:
            self._rss_samples += 1
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_total)
        self._sample_accelerators(active_pids)

    def _sample_accelerators(self, active_pids: set[int]) -> None:
        if self._nvml is None or self._nvml_failed:
            return
        now = time.monotonic()
        if self._last_accelerator_sample_at is not None:
            elapsed = max(0.0, now - self._last_accelerator_sample_at)
            self._accelerator_seconds += elapsed * self._last_active_device_count
        try:
            active_devices = 0
            memory_total = 0
            for handle in self._nvml_handles:
                matched = False
                for process in _nvml_processes(self._nvml, handle):
                    try:
                        pid = int(process.pid)
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if pid not in active_pids:
                        continue
                    matched = True
                    used_memory = getattr(process, "usedGpuMemory", None)
                    if (
                        isinstance(used_memory, int)
                        and not isinstance(used_memory, bool)
                        and 0 <= used_memory < (1 << 63)
                    ):
                        memory_total += used_memory
                if matched:
                    active_devices += 1
            self._last_accelerator_sample_at = now
            self._last_active_device_count = active_devices
            self._accelerator_peak_memory_bytes = max(
                self._accelerator_peak_memory_bytes,
                memory_total,
            )
            self._accelerator_samples += 1
        except Exception as exc:
            self._nvml_reason = f"sampling_failed:{type(exc).__name__}"
            self._nvml_failed = True
            self._last_active_device_count = 0


def _optional_module(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except ModuleNotFoundError:
        return None, "dependency_missing"
    except Exception as exc:
        return None, f"import_failed:{type(exc).__name__}"


def _reported_measurement_identity(
    *,
    has_process_measurement: bool,
    has_accelerator_measurement: bool,
) -> dict[str, Any]:
    return {
        "process_tree": {
            "status": "available" if has_process_measurement else "unavailable",
            "reason": None if has_process_measurement else "runner_did_not_report",
            "sampler": "injected_process_runner.v1",
            "scope": "supervised_workload",
        },
        "accelerator": {
            "status": "available" if has_accelerator_measurement else "unavailable",
            "reason": None if has_accelerator_measurement else "runner_did_not_report",
            "sampler": "injected_process_runner.v1",
            "scope": "supervised_workload",
        },
    }


def _module_version(module: Any | None) -> str | None:
    if module is None:
        return None
    value = getattr(module, "__version__", None)
    return value if isinstance(value, str) and value else None


def _nvml_processes(module: Any, handle: Any) -> tuple[Any, ...]:
    processes: dict[int, Any] = {}
    for prefix in ("Compute", "Graphics"):
        getter = None
        for suffix in ("_v3", "_v2", ""):
            candidate = getattr(
                module,
                f"nvmlDeviceGet{prefix}RunningProcesses{suffix}",
                None,
            )
            if callable(candidate):
                getter = candidate
                break
        if getter is None:
            continue
        try:
            observed = getter(handle)
        except Exception as exc:
            not_supported = getattr(module, "NVMLError_NotSupported", ())
            if isinstance(not_supported, type) and isinstance(exc, not_supported):
                continue
            raise
        for process in observed or ():
            pid = getattr(process, "pid", None)
            if isinstance(pid, int) and not isinstance(pid, bool):
                processes[pid] = process
    return tuple(processes.values())


def minimal_worker_environment(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Keep process bootstrap variables, but never inherit credentials."""

    environment = {key: os.environ[key] for key in _INHERITED_ENVIRONMENT_KEYS if key in os.environ}
    environment["PYTHONIOENCODING"] = "utf-8"
    for key, value in (overrides or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValidationError("worker environment keys and values must be strings")
        if "\x00" in key or "\x00" in value or "=" in key:
            raise ValidationError("worker environment contains an invalid key or value")
        environment[key] = value
    return environment


def run_bounded_process(
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
    """Stream all pipes with hard in-memory bounds and a wall-clock timeout."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValidationError("worker timeout_seconds must be finite and positive")
    deadline = time.monotonic() + float(timeout_seconds)
    encoded_input = input_text.encode("utf-8")
    if len(encoded_input) > max_input_bytes:
        raise BoundedProcessOutputLimitError(f"{process_name} request exceeds its input limit")
    if min(max_input_bytes, max_stdout_bytes, max_stderr_bytes) < 1:
        raise ValidationError("worker byte limits must be positive")
    try:
        process = subprocess.Popen(
            tuple(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=cwd,
            env=minimal_worker_environment(environment),
            start_new_session=os.name == "posix",
        )
    except FileNotFoundError as exc:
        raise BoundedProcessStartupError(
            f"{process_name} executable is unavailable: {argv[0]}"
        ) from exc
    except OSError as exc:
        raise BoundedProcessStartupError(f"{process_name} could not be started") from exc

    def kill_process_tree() -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass

    if process.stdin is None or process.stdout is None or process.stderr is None:
        kill_process_tree()
        raise ExternalDependencyError(f"{process_name} did not expose worker pipes")

    monitor = _ProcessTreeMonitor(process.pid)
    monitor.start()
    stdout = bytearray()
    stderr = bytearray()
    overflows: list[str] = []
    writer_errors: list[BaseException] = []
    lock = threading.Lock()

    def read_pipe(stream, output: bytearray, limit: int, label: str) -> None:
        try:
            while True:
                remaining = limit + 1 - len(output)
                chunk = stream.read(max(1, min(64 * 1024, remaining)))
                if not chunk:
                    return
                output.extend(chunk)
                if len(output) > limit:
                    with lock:
                        overflows.append(label)
                    kill_process_tree()
                    return
        finally:
            stream.close()

    def write_input() -> None:
        try:
            process.stdin.write(encoded_input)
            process.stdin.flush()
        except BrokenPipeError:
            pass
        except BaseException as exc:  # captured and re-raised on the caller thread
            writer_errors.append(exc)
            kill_process_tree()
        finally:
            # The worker may exit between ``flush`` and ``close``.  On Python
            # 3.13 the buffered close performs a final flush and can surface
            # the same expected EPIPE even when the explicit flush above was
            # already handled.  Keep that shutdown race on the writer thread
            # from leaking as an unhandled-thread warning.
            with suppress(BrokenPipeError):
                process.stdin.close()

    threads = (
        threading.Thread(
            target=read_pipe,
            args=(process.stdout, stdout, max_stdout_bytes, "stdout"),
            daemon=True,
        ),
        threading.Thread(
            target=read_pipe,
            args=(process.stderr, stderr, max_stderr_bytes, "stderr"),
            daemon=True,
        ),
        threading.Thread(target=write_input, daemon=True),
    )
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.wait(timeout=0)
        else:
            process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_tree()
        process.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    pipes_open = any(thread.is_alive() for thread in threads)
    if pipes_open:
        kill_process_tree()
        for thread in threads:
            thread.join(timeout=0.5)
    observation = monitor.stop()

    if timed_out:
        raise BoundedProcessTimeoutError(
            f"{process_name} exceeded {timeout_seconds}s",
            resource_observation=observation,
        )
    if pipes_open:
        raise BoundedProcessProtocolError(
            f"{process_name} left worker pipes open",
            resource_observation=observation,
        )
    if writer_errors:
        raise BoundedProcessProtocolError(
            f"{process_name} request pipe failed",
            resource_observation=observation,
        ) from writer_errors[0]
    if overflows:
        streams = ",".join(sorted(set(overflows)))
        raise BoundedProcessOutputLimitError(
            f"{process_name} exceeded its {streams} limit",
            resource_observation=observation,
        )
    try:
        stdout_text = bytes(stdout).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BoundedProcessProtocolError(
            f"{process_name} emitted non-UTF-8 stdout",
            resource_observation=observation,
        ) from exc
    stderr_text = bytes(stderr).decode("utf-8", errors="replace")
    return BoundedProcessResult(
        returncode=process.returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        cpu_seconds=observation.cpu_seconds,
        peak_rss_bytes=observation.peak_rss_bytes,
        accelerator_seconds=observation.accelerator_seconds,
        accelerator_peak_memory_bytes=(observation.accelerator_peak_memory_bytes),
        measurement_identity=observation.measurement_identity,
    )
