"""Shared exact-SHA process driver for stateless method-versus-``NO_OP`` discovery.

The benchmark-specific adapter remains a small committed worker script, while
the experiment driver, process supervision, identity receipts, unit preflight,
and persistence guard are shared across stateless profile kinds. Stateful
``clarify`` and ``composite`` discovery must use the atomic external-arena
driver instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import (
    ArtifactError,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.contracts.serialization import (
    canonical_json,
    content_hash,
    freeze_mapping,
)
from agentinputlab.experiments.bindings import resolve_frozen_binding_receipt
from agentinputlab.experiments.executor import ExperimentCellResult
from agentinputlab.experiments.parity_identity import (
    HOST_EXECUTION_RECEIPT_PROTOCOL,
)
from agentinputlab.experiments.plan import ExperimentCell
from agentinputlab.experiments.schema import (
    ExperimentManifest,
    ExperimentStage,
    require_sha256,
)
from agentinputlab.experiments.secrets import (
    environment_secret_values,
    redact_secret_text,
    require_secret_free,
)
from agentinputlab.integrations.external_workers.supervisor import (
    LocalSubprocessSupervisor,
    SupervisedWorkerError,
    WorkerSupervisor,
    worker_supervisor_from_environment,
)
from agentinputlab.observability import ResourceLimits, ResourceReceipt

from .parity_process import (
    RESPONSE_PROTOCOL,
    LockedProcessWorker,
    attest_locked_process_worker,
    build_host_execution_receipt,
    execute_locked_process_worker,
    load_locked_process_worker,
    locked_process_worker_environment,
    parse_locked_process_response,
)

CONFIG_PATH_ENV = "AILAB_METHOD_PROCESS_CONFIG"
CONFIG_SHA256_ENV = "AILAB_METHOD_PROCESS_CONFIG_SHA256"
UNIT_INVENTORY_ENV = "AILAB_METHOD_DISCOVERY_UNITS"
UNIT_INVENTORY_SHA256_ENV = "AILAB_METHOD_DISCOVERY_UNITS_SHA256"
DRIVER_PROTOCOL = "agentinputlab.method-process-driver.v1"
REQUEST_PROTOCOL = "agentinputlab.method-process-request.v1"

_ARM_IDS = ("no_op", "candidate")
_CANDIDATE_CHECKOUT_ENV = "AILAB_CANDIDATE_CHECKOUT"
_BASE_PASSTHROUGH_ENVIRONMENTS = (
    "AILAB_PREPARED_DATA_ROOT",
    "AILAB_METHOD_CHECKOUT",
    "AILAB_METHOD_ASSET_MANIFEST",
    "AILAB_MODEL_CONFIG",
    "AILAB_RETRIEVER_CONFIG",
)
_WORKER_FIELDS = frozenset(
    {
        "checkout_environment",
        "expected_checkout_revision",
        "expected_checkout_tree",
        "adapter_checkout_environment",
        "expected_adapter_checkout_revision",
        "expected_adapter_checkout_tree",
        "executable_path",
        "executable_sha256",
        "entrypoint",
        "entrypoint_sha256",
        "arguments",
        "pass_environment",
        "request_parameters",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "method_id",
        "candidate_release_id",
        "workers",
        "resource_limits",
        "pipe_limits",
        "shared_request_parameters",
    }
)
_LIMIT_FIELDS = frozenset(
    {
        "wall_timeout_seconds",
        "memory_bytes",
        "cpus",
        "pids_limit",
    }
)
_PIPE_FIELDS = frozenset(
    {
        "max_input_bytes",
        "max_stdout_bytes",
        "max_stderr_bytes",
    }
)


@dataclass(frozen=True, slots=True)
class _Configuration:
    method_id: str
    candidate_release_id: str
    workers: Mapping[str, LockedProcessWorker]
    resource_limits: ResourceLimits
    pipe_limits: Mapping[str, int]
    shared_request_parameters: Mapping[str, Any]
    config_sha256: str


class MethodProcessExperimentDriver:
    """Run ``NO_OP`` and one method release through locked JSON processes."""

    def __init__(
        self,
        configuration: _Configuration,
        *,
        environment: Mapping[str, str],
        supervisor: WorkerSupervisor | None = None,
    ) -> None:
        self._configuration = configuration
        self._environment = dict(environment)
        self._supervisor = supervisor or LocalSubprocessSupervisor()
        if not isinstance(self._supervisor, WorkerSupervisor):
            raise TypeError("method-process supervisor must implement WorkerSupervisor")
        self._manifest_hash: str | None = None
        self._binding_receipt_hash: str | None = None
        self._worker_states: Mapping[str, Mapping[str, Any]] = freeze_mapping({})
        self._checkout_binding_hashes: Mapping[str, str] = freeze_mapping({})
        self._adapter_checkout_binding_hashes: Mapping[str, str] = freeze_mapping({})
        self._unit_ids: frozenset[str] = frozenset()
        self._fingerprint = freeze_mapping(self._build_fingerprint())

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        environment: Mapping[str, str] | None = None,
        supervisor: WorkerSupervisor | None = None,
    ) -> MethodProcessExperimentDriver:
        env = dict(os.environ if environment is None else environment)
        return cls(
            _load_configuration(
                Path(path),
                expected_sha256=expected_sha256,
                environment=env,
            ),
            environment=env,
            supervisor=supervisor,
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return self._fingerprint

    def preflight(self, manifest: ExperimentManifest) -> Sequence[str]:
        blockers: list[str] = []
        if manifest.stage is not ExperimentStage.DISCOVERY:
            blockers.append("method-process driver requires stage=discovery")
        if manifest.metadata.get("requires_atomic_external_replay") is True:
            blockers.append(
                "method-process driver cannot execute stateful discovery; "
                "use the external-arena atomic replay driver"
            )
        if manifest.metadata.get("method_id") != self._configuration.method_id:
            blockers.append("method-process config method_id differs from manifest")
        if (
            manifest.metadata.get("candidate_release_id")
            != self._configuration.candidate_release_id
        ):
            blockers.append("method-process candidate release differs from manifest")
        if set(arm.arm_id for arm in manifest.arms) != set(_ARM_IDS):
            blockers.append("method-process driver requires exact no_op and candidate arms")
        candidate = next(
            (arm for arm in manifest.arms if arm.arm_id == "candidate"),
            None,
        )
        if candidate is None or candidate.release_id != self._configuration.candidate_release_id:
            blockers.append("method-process candidate arm differs from config")
        if isinstance(self._supervisor, LocalSubprocessSupervisor) and any(
            value is not None
            for value in (
                self._configuration.resource_limits.memory_bytes,
                self._configuration.resource_limits.cpus,
                self._configuration.resource_limits.pids_limit,
            )
        ):
            blockers.append(
                "local method-process supervision supports hard wall/pipe limits only; "
                "use a container supervisor for memory/CPU/PID limits"
            )
        pass_names = self._manifest_pass_environment(manifest)
        for worker in self._configuration.workers.values():
            missing = sorted(
                name
                for name in (*pass_names, *worker.pass_environment)
                if not self._environment.get(name)
            )
            if missing:
                blockers.append(
                    f"method-process {worker.arm_id} missing environments: " + ", ".join(missing)
                )
        try:
            unit_ids = _load_unit_inventory(self._environment)
        except (ArtifactError, ValidationError) as exc:
            blockers.append(redact_secret_text(str(exc)))
            unit_ids = ()
        if blockers:
            return tuple(blockers)
        try:
            worker_states = {
                arm_id: attest_locked_process_worker(worker)
                for arm_id, worker in self._configuration.workers.items()
            }
            binding_receipt = resolve_frozen_binding_receipt(
                manifest,
                environment=self._environment,
            )
            checkout_hashes = {
                arm_id: binding_receipt.binding(
                    worker.checkout_environment,
                    kind="directory_env",
                ).binding_hash
                for arm_id, worker in self._configuration.workers.items()
            }
            adapter_checkout_hashes = {
                arm_id: binding_receipt.binding(
                    worker.adapter_checkout_environment,
                    kind="directory_env",
                ).binding_hash
                for arm_id, worker in self._configuration.workers.items()
            }
        except (ArtifactError, ExternalDependencyError, ValidationError) as exc:
            return (redact_secret_text(str(exc)),)
        self._manifest_hash = manifest.manifest_hash
        self._binding_receipt_hash = binding_receipt.receipt_hash
        self._worker_states = freeze_mapping(worker_states)
        self._checkout_binding_hashes = freeze_mapping(checkout_hashes)
        self._adapter_checkout_binding_hashes = freeze_mapping(adapter_checkout_hashes)
        self._unit_ids = frozenset(unit_ids)
        return ()

    def preflight_units(
        self,
        manifest: ExperimentManifest,
        resolved_units: Mapping[str, Sequence[str]],
    ) -> Sequence[str]:
        """Prove the plan is exactly the checksum-locked discovery inventory."""

        if self._manifest_hash != manifest.manifest_hash or not self._unit_ids:
            return ("unit inventory validation requires a successful driver preflight",)
        expected_keys = {f"{target.cell_id}:{target.split}" for target in manifest.benchmarks}
        if set(resolved_units) != expected_keys:
            return ("resolved unit inventories differ from method discovery targets",)
        observed: set[str] = set()
        for key in expected_keys:
            values = tuple(resolved_units[key])
            if len(values) != len(set(values)):
                return (f"{key}: resolved discovery units contain duplicates",)
            observed.update(values)
        missing = sorted(self._unit_ids - observed)
        extra = sorted(observed - self._unit_ids)
        if missing or extra:
            return (
                "resolved discovery units differ from checksum-locked inventory; "
                f"missing={missing[:5]}, extra={extra[:5]}",
            )
        return ()

    def execute_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> ExperimentCellResult:
        self._require_preflight(manifest, cell)
        outcomes: list[dict[str, Any]] = []
        receipts: list[ResourceReceipt] = []
        for arm_id in cell.arm_ids:
            worker = self._configuration.workers[arm_id]
            request = self._request(manifest, cell, arm_id)
            passed_names = (
                *self._manifest_pass_environment(manifest),
                *worker.pass_environment,
            )
            environment = locked_process_worker_environment(
                worker,
                environment=self._environment,
                pass_environment=passed_names,
                request_protocol=REQUEST_PROTOCOL,
            )
            process = execute_locked_process_worker(
                worker,
                supervisor=self._supervisor,
                input_text=canonical_json(request) + "\n",
                limits=self._configuration.resource_limits,
                pipe_limits=self._configuration.pipe_limits,
                environment=environment,
                process_name=f"method-process-{arm_id}",
                config_sha256=self._configuration.config_sha256,
                expected_worker_state=self._worker_states[arm_id],
            )
            passed_values = environment_secret_values(
                self._environment,
                passed_names,
            )
            if process.returncode != 0:
                message = redact_secret_text(
                    process.stderr.strip() or "worker exited non-zero",
                    secret_values=passed_values,
                )
                raise SupervisedWorkerError(
                    f"method-process {arm_id} failed: {message}",
                    resource_receipt=process.resource_receipt,
                )
            response = parse_locked_process_response(
                process,
                expected_request_hash=request["request_hash"],
                secret_values=passed_values,
            )
            host_receipt = build_host_execution_receipt(
                config_sha256=self._configuration.config_sha256,
                manifest=manifest,
                cell=cell,
                worker=worker,
                request=request,
                process=process,
                checkout_binding_hash=self._checkout_binding_hashes[arm_id],
                adapter_checkout_binding_hash=(self._adapter_checkout_binding_hashes[arm_id]),
            )
            outcome: dict[str, Any] = {
                "arm_id": arm_id,
                "status": response["status"],
                "request_hash": response["request_hash"],
                "usage": response["usage"],
                "output": response["output"],
                "host_execution_receipt": host_receipt,
            }
            if response["status"] == "succeeded":
                outcome["metrics"] = response["metrics"]
            else:
                outcome["failure"] = response["failure"]
            outcomes.append(outcome)
            receipts.append(process.resource_receipt)
        return ExperimentCellResult(
            cell_id=cell.cell_id,
            status="succeeded",
            payload={
                "frozen_binding_receipt_hash": self._binding_receipt_hash,
                "host_execution_receipt_protocol": HOST_EXECUTION_RECEIPT_PROTOCOL,
                "arms": outcomes,
            },
            resource_receipts=tuple(receipts),
        )

    def _build_fingerprint(self) -> dict[str, Any]:
        bindings = {
            arm_id: {
                "checkout_environment": worker.checkout_environment,
                "checkout": worker.checkout_identity.to_dict(),
                "adapter_checkout_environment": worker.adapter_checkout_environment,
                "adapter_checkout": worker.adapter_checkout_identity.to_dict(),
                "command": worker.command_identity.to_dict(),
            }
            for arm_id, worker in self._configuration.workers.items()
        }
        return {
            "schema_version": 1,
            "driver": DRIVER_PROTOCOL,
            "config_sha256": self._configuration.config_sha256,
            "method_id": self._configuration.method_id,
            "candidate_release_id": self._configuration.candidate_release_id,
            "execution_receipt_protocol": HOST_EXECUTION_RECEIPT_PROTOCOL,
            "host_process_bindings": bindings,
            "resource_limits": self._configuration.resource_limits.as_dict(),
            "pipe_limits": dict(self._configuration.pipe_limits),
            "supervisor": dict(self._supervisor.fingerprint()),
            "unit_inventory_sha256": self._environment.get(UNIT_INVENTORY_SHA256_ENV),
            "response_protocol": RESPONSE_PROTOCOL,
        }

    def _manifest_pass_environment(
        self,
        manifest: ExperimentManifest,
    ) -> tuple[str, ...]:
        declared = {item.name for item in manifest.prerequisites}
        names = [name for name in _BASE_PASSTHROUGH_ENVIRONMENTS if name in declared]
        if "AILAB_BENCHMARK_CHECKOUT" in declared:
            names.append("AILAB_BENCHMARK_CHECKOUT")
        return tuple(dict.fromkeys(names))

    def _request(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
        arm_id: str,
    ) -> Mapping[str, Any]:
        arm = next(item for item in manifest.arms if item.arm_id == arm_id)
        target = next(
            item
            for item in manifest.benchmarks
            if item.cell_id == cell.benchmark_id and item.split == cell.split
        )
        payload = {
            "schema_version": 1,
            "protocol": REQUEST_PROTOCOL,
            "manifest_hash": manifest.manifest_hash,
            "cell": {
                "cell_id": cell.cell_id,
                "study_id": cell.study_id,
                "benchmark_id": cell.benchmark_id,
                "split": cell.split,
                "unit_id": cell.unit_id,
                "model_slot": cell.model_slot,
                "seed": cell.seed,
            },
            "benchmark": {
                "external": target.external,
                "source_revision": target.source_revision,
            },
            "method": {
                "method_id": self._configuration.method_id,
                "candidate_release_id": self._configuration.candidate_release_id,
            },
            "arm": {
                "arm_id": arm_id,
                "release_id": arm.release_id,
                "family": arm.family.value,
                "profile_kind": arm.profile_kind.value,
                "parameters": arm.parameters,
            },
            "parameters": {
                "shared": self._configuration.shared_request_parameters,
                "worker": self._configuration.workers[arm_id].request_parameters,
            },
            "asset_environments": list(self._manifest_pass_environment(manifest)),
        }
        require_secret_free(payload, label="method-process worker request")
        return freeze_mapping(
            {
                **payload,
                "request_hash": content_hash(payload),
            }
        )

    def _require_preflight(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> None:
        if self._manifest_hash != manifest.manifest_hash:
            raise ArtifactError("method-process execution requires successful preflight")
        if self._binding_receipt_hash is None:
            raise ArtifactError("method-process binding receipt is unavailable")
        if cell.manifest_hash != manifest.manifest_hash:
            raise ValidationError("method-process cell belongs to another manifest")
        if set(cell.arm_ids) != set(_ARM_IDS):
            raise ValidationError("method-process cell requires no_op and candidate")
        if cell.unit_id not in self._unit_ids:
            raise ArtifactError("method-process cell is outside the preflighted unit inventory")


def factory() -> MethodProcessExperimentDriver:
    path = os.environ.get(CONFIG_PATH_ENV)
    expected = os.environ.get(CONFIG_SHA256_ENV)
    if not path:
        raise ArtifactError(f"{CONFIG_PATH_ENV} is required")
    if not expected:
        raise ArtifactError(f"{CONFIG_SHA256_ENV} is required")
    return MethodProcessExperimentDriver.from_config(
        path,
        expected_sha256=expected,
        supervisor=worker_supervisor_from_environment(),
    )


def _load_configuration(
    path: Path,
    *,
    expected_sha256: str,
    environment: Mapping[str, str],
) -> _Configuration:
    require_sha256(expected_sha256, label=CONFIG_SHA256_ENV)
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ArtifactError("method-process config must not be a symlink")
    source = unresolved.resolve()
    if not source.is_file():
        raise ArtifactError("method-process config is not an existing file")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ArtifactError("method-process config checksum differs")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("method-process config is not valid UTF-8 JSON") from exc
    data = _strict_mapping(raw, _CONFIG_FIELDS, label="method-process config")
    require_secret_free(data, label="method-process config")
    if data["schema_version"] != 1 or data["protocol"] != DRIVER_PROTOCOL:
        raise ValidationError("unsupported method-process config")
    method_id = _text(data["method_id"], "method_id")
    candidate_release_id = _text(
        data["candidate_release_id"],
        "candidate_release_id",
    )
    raw_workers = _strict_mapping(
        data["workers"],
        frozenset(_ARM_IDS),
        label="method-process workers",
    )
    workers: dict[str, LockedProcessWorker] = {}
    for arm_id in _ARM_IDS:
        worker_data = _strict_mapping(
            raw_workers[arm_id],
            _WORKER_FIELDS,
            label=f"method-process worker {arm_id}",
        )
        workers[arm_id] = load_locked_process_worker(
            arm_id,
            worker_data,
            environment=environment,
            checkout_environment=_CANDIDATE_CHECKOUT_ENV,
            adapter_checkout_environment=_CANDIDATE_CHECKOUT_ENV,
        )
    limits = ResourceLimits(
        **_strict_mapping(
            data["resource_limits"],
            _LIMIT_FIELDS,
            label="method-process resource_limits",
        )
    )
    raw_pipes = _strict_mapping(
        data["pipe_limits"],
        _PIPE_FIELDS,
        label="method-process pipe_limits",
    )
    pipes = {
        name: _positive_integer(raw_pipes[name], f"pipe_limits.{name}")
        for name in sorted(_PIPE_FIELDS)
    }
    parameters = _mapping(
        data["shared_request_parameters"],
        label="shared_request_parameters",
    )
    return _Configuration(
        method_id=method_id,
        candidate_release_id=candidate_release_id,
        workers=workers,
        resource_limits=limits,
        pipe_limits=freeze_mapping(pipes),
        shared_request_parameters=parameters,
        config_sha256=actual_sha256,
    )


def _load_unit_inventory(environment: Mapping[str, str]) -> tuple[str, ...]:
    raw_path = environment.get(UNIT_INVENTORY_ENV)
    expected = environment.get(UNIT_INVENTORY_SHA256_ENV)
    if not raw_path:
        raise ArtifactError(f"{UNIT_INVENTORY_ENV} is required")
    if not expected:
        raise ArtifactError(f"{UNIT_INVENTORY_SHA256_ENV} is required")
    require_sha256(expected, label=UNIT_INVENTORY_SHA256_ENV)
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != expected:
        raise ArtifactError("method discovery unit inventory is absent or changed")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("method discovery unit inventory is not valid UTF-8 JSON") from exc
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or not raw
        or any(not isinstance(item, str) or not item.strip() for item in raw)
        or len(raw) != len(set(raw))
    ):
        raise ValidationError(
            "method discovery unit inventory must be a non-empty unique string array"
        )
    return tuple(raw)


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValidationError(f"{label} fields are invalid")
    return dict(value)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    try:
        result = freeze_mapping(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be canonical JSON") from exc
    require_secret_free(result, label=f"method-process {label}")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CONFIG_PATH_ENV",
    "CONFIG_SHA256_ENV",
    "DRIVER_PROTOCOL",
    "REQUEST_PROTOCOL",
    "UNIT_INVENTORY_ENV",
    "UNIT_INVENTORY_SHA256_ENV",
    "MethodProcessExperimentDriver",
    "factory",
]
