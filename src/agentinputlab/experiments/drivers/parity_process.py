"""Exact-SHA, two-process driver for upstream-versus-candidate parity checks.

The host, rather than either worker, establishes implementation identity.  Each
arm is started in a fresh bounded subprocess with a checkout-specific working
directory and ``PYTHONPATH``.  The host independently verifies the checkout,
executable, entrypoint, and command before and after every invocation and emits
an execution receipt that the guarded experiment driver validates.

This driver provides execution evidence only.  A successful run is not, by
itself, a paper-result or official-parity claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from agentinputlab.contracts.errors import (
    AlgorithmOutputError,
    ArtifactError,
    ExternalDependencyError,
    ValidationError,
)
from agentinputlab.contracts.serialization import (
    canonical_json,
    content_hash,
    freeze_mapping,
    to_jsonable,
)
from agentinputlab.experiments.bindings import resolve_frozen_binding_receipt
from agentinputlab.experiments.executor import ExperimentCellResult
from agentinputlab.experiments.parity_identity import (
    HOST_EXECUTION_RECEIPT_PROTOCOL,
    build_parity_implementation_contract,
)
from agentinputlab.experiments.plan import ExperimentCell
from agentinputlab.experiments.schema import ExperimentManifest, ExperimentStage, require_sha256
from agentinputlab.experiments.secrets import (
    environment_secret_values,
    redact_secret_text,
    require_secret_free,
    require_secret_values_absent,
)
from agentinputlab.integrations.external_workers.supervisor import (
    DockerWorkerSupervisor,
    LocalSubprocessSupervisor,
    SupervisedProcessResult,
    SupervisedWorkerError,
    WorkerSupervisor,
    worker_supervisor_from_environment,
)
from agentinputlab.observability import ResourceLimits, ResourceReceipt, ResourceUsage

CONFIG_PATH_ENV = "AILAB_PARITY_PROCESS_CONFIG"
CONFIG_SHA256_ENV = "AILAB_PARITY_PROCESS_CONFIG_SHA256"
UNIT_INVENTORY_ENV = "AILAB_PARITY_UNIT_INVENTORY"
UNIT_INVENTORY_SHA256_ENV = "AILAB_PARITY_UNIT_INVENTORY_SHA256"
REQUEST_PROTOCOL = "agentinputlab.parity-process-request.v1"
RESPONSE_PROTOCOL = "agentinputlab.parity-process-response.v1"
DRIVER_PROTOCOL = "agentinputlab.parity-process-driver.v1"

_UPSTREAM_ARM = "upstream_reference"
_CANDIDATE_ARM = "agentinputlab_candidate"
_ARM_IDS = (_UPSTREAM_ARM, _CANDIDATE_ARM)
_CHECKOUT_ENVIRONMENTS = {
    _UPSTREAM_ARM: "AILAB_UPSTREAM_CHECKOUT",
    _CANDIDATE_ARM: "AILAB_CANDIDATE_CHECKOUT",
}
_BASE_PASSTHROUGH_ENVIRONMENTS = (
    "AILAB_PARITY_ASSET_ROOT",
    "AILAB_MODEL_CONFIG",
    "AILAB_RETRIEVER_CONFIG",
)
_RESERVED_WORKER_ENVIRONMENTS = frozenset(
    {
        "AILAB_PARITY_ARM_ID",
        "AILAB_PARITY_PROTOCOL",
        "AILAB_PROCESS_ARM_ID",
        "AILAB_PROCESS_PROTOCOL",
        "LANG",
        "LC_ALL",
        "PWD",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONIOENCODING",
        "PYTHONNOUSERSITE",
        "TZ",
    }
)
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_USAGE_FIELDS = frozenset(
    {
        "model_calls",
        "retrieval_calls",
        "tool_calls",
        "embedding_calls",
        "unpriced_model_calls",
        "input_tokens",
        "output_tokens",
        "cpu_seconds",
        "accelerator_seconds",
        "cost_usd",
        "compute_cost_usd",
    }
)
_PIPE_FIELDS = frozenset(
    {
        "max_input_bytes",
        "max_stdout_bytes",
        "max_stderr_bytes",
    }
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
_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "request_hash",
        "status",
        "metrics",
        "usage",
        "output",
        "failure",
    }
)
_RESPONSE_FORBIDDEN_KEYS = frozenset(
    {
        "implementation_identity",
        "host_execution_receipt",
        "resource_receipt",
    }
)


@dataclass(frozen=True, slots=True)
class _CheckoutIdentity:
    resolved_path_sha256: str
    git_head: str
    git_tree: str

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class _CommandIdentity:
    executable_resolved_path_sha256: str
    executable_content_sha256: str
    entrypoint_relative_path: str
    entrypoint_resolved_path_sha256: str
    entrypoint_content_sha256: str
    argv_sha256: str
    cwd_resolved_path_sha256: str
    pythonpath_resolved_path_sha256: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    arm_id: str
    checkout_environment: str
    checkout: Path
    expected_checkout_revision: str
    expected_checkout_tree: str
    adapter_checkout_environment: str
    adapter_checkout: Path
    expected_adapter_checkout_revision: str
    expected_adapter_checkout_tree: str
    executable: Path
    executable_sha256: str
    entrypoint_relative_path: str
    entrypoint: Path
    entrypoint_sha256: str
    arguments: tuple[str, ...]
    pass_environment: tuple[str, ...]
    request_parameters: Mapping[str, Any]
    checkout_identity: _CheckoutIdentity
    adapter_checkout_identity: _CheckoutIdentity
    command_identity: _CommandIdentity

    @property
    def argv(self) -> tuple[str, ...]:
        return (str(self.executable), str(self.entrypoint), *self.arguments)


@dataclass(frozen=True, slots=True)
class _DriverConfiguration:
    method_id: str
    candidate_release_id: str
    workers: Mapping[str, _WorkerSpec]
    resource_limits: ResourceLimits
    pipe_limits: Mapping[str, int]
    shared_request_parameters: Mapping[str, Any]
    config_sha256: str


class ParityProcessProtocolError(AlgorithmOutputError):
    """A worker violated the parity-process protocol after it was started."""

    def __init__(self, message: str, *, resource_receipt: ResourceReceipt) -> None:
        super().__init__(message)
        self.resource_receipt = resource_receipt


class ParityProcessExperimentDriver:
    """Run exact upstream and candidate adapters as independent JSON workers."""

    def __init__(
        self,
        configuration: _DriverConfiguration,
        *,
        environment: Mapping[str, str],
        supervisor: WorkerSupervisor | None = None,
    ) -> None:
        self._configuration = configuration
        self._environment = dict(environment)
        self._supervisor = supervisor or LocalSubprocessSupervisor()
        if not isinstance(self._supervisor, WorkerSupervisor):
            raise TypeError("parity-process supervisor must implement WorkerSupervisor")
        self._manifest_hash: str | None = None
        self._binding_receipt_hash: str | None = None
        self._contract_hash: str | None = None
        self._preflight_worker_state: Mapping[str, Mapping[str, Any]] = freeze_mapping({})
        self._fixture_unit_ids: frozenset[str] = frozenset()
        self._checkout_binding_hashes: Mapping[str, str] = freeze_mapping({})
        self._adapter_checkout_binding_hashes: Mapping[str, str] = freeze_mapping({})
        self._fingerprint = freeze_mapping(self._build_fingerprint())

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        environment: Mapping[str, str] | None = None,
        supervisor: WorkerSupervisor | None = None,
    ) -> ParityProcessExperimentDriver:
        env = dict(os.environ if environment is None else environment)
        configuration = _load_configuration(
            Path(path),
            expected_sha256=expected_sha256,
            environment=env,
        )
        return cls(
            configuration,
            environment=env,
            supervisor=supervisor,
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return self._fingerprint

    def preflight(self, manifest: ExperimentManifest) -> Sequence[str]:
        blockers: list[str] = []
        if manifest.stage is not ExperimentStage.PARITY:
            blockers.append("parity-process driver requires stage=parity")
        if manifest.metadata.get("method_id") != self._configuration.method_id:
            blockers.append("parity-process config method_id differs from manifest")
        if (
            manifest.metadata.get("candidate_release_id")
            != self._configuration.candidate_release_id
        ):
            blockers.append("parity-process candidate release differs from manifest")
        if set(arm.arm_id for arm in manifest.arms) != set(_ARM_IDS):
            blockers.append("parity-process driver requires exact upstream and candidate arms")
        upstream_commit = manifest.metadata.get("upstream_commit")
        if upstream_commit != self._configuration.workers[_UPSTREAM_ARM].expected_checkout_revision:
            blockers.append("parity-process upstream checkout revision differs from manifest")
        if isinstance(self._supervisor, LocalSubprocessSupervisor) and any(
            value is not None
            for value in (
                self._configuration.resource_limits.memory_bytes,
                self._configuration.resource_limits.cpus,
                self._configuration.resource_limits.pids_limit,
            )
        ):
            blockers.append(
                "local parity-process supervision supports hard wall/pipe limits only; "
                "use a container supervisor for memory/CPU/PID limits"
            )
        for worker in self._configuration.workers.values():
            missing = sorted(
                name
                for name in (
                    *_BASE_PASSTHROUGH_ENVIRONMENTS,
                    *worker.pass_environment,
                )
                if not self._environment.get(name)
            )
            if missing:
                blockers.append(
                    f"parity-process {worker.arm_id} missing worker environments: "
                    + ", ".join(missing)
                )
        try:
            fixture_units = _load_fixture_inventory(self._environment)
        except (ArtifactError, ValidationError) as exc:
            blockers.append(redact_secret_text(str(exc)))
            fixture_units = ()
        if blockers:
            return tuple(blockers)
        try:
            state = {
                arm_id: _attest_worker(worker).to_dict()
                for arm_id, worker in self._configuration.workers.items()
            }
            receipt = resolve_frozen_binding_receipt(
                manifest,
                environment=self._environment,
            )
            contract = build_parity_implementation_contract(
                manifest,
                receipt,
                self._fingerprint,
            )
        except (ArtifactError, ExternalDependencyError, ValidationError) as exc:
            return (redact_secret_text(str(exc)),)
        self._manifest_hash = manifest.manifest_hash
        self._binding_receipt_hash = receipt.receipt_hash
        self._contract_hash = contract.contract_hash
        self._preflight_worker_state = freeze_mapping(state)
        self._fixture_unit_ids = frozenset(fixture_units)
        self._checkout_binding_hashes = freeze_mapping(
            {
                arm_id: receipt.binding(
                    worker.checkout_environment,
                    kind="directory_env",
                ).binding_hash
                for arm_id, worker in self._configuration.workers.items()
            }
        )
        self._adapter_checkout_binding_hashes = freeze_mapping(
            {
                arm_id: receipt.binding(
                    worker.adapter_checkout_environment,
                    kind="directory_env",
                ).binding_hash
                for arm_id, worker in self._configuration.workers.items()
            }
        )
        return ()

    def preflight_units(
        self,
        manifest: ExperimentManifest,
        resolved_units: Mapping[str, Sequence[str]],
    ) -> Sequence[str]:
        """Prove the plan uses exactly the checksum-locked fixture inventory."""

        if self._manifest_hash != manifest.manifest_hash or not self._fixture_unit_ids:
            return ("unit inventory validation requires a successful driver preflight",)
        expected_keys = {f"{target.cell_id}:{target.split}" for target in manifest.benchmarks}
        if set(resolved_units) != expected_keys:
            return ("resolved unit inventories differ from parity manifest targets",)
        blockers: list[str] = []
        observed: set[str] = set()
        for key in sorted(expected_keys):
            values = tuple(resolved_units[key])
            if len(values) != len(set(values)):
                blockers.append(f"{key}: resolved parity units contain duplicates")
                continue
            observed.update(values)
        missing = sorted(self._fixture_unit_ids - observed)
        extra = sorted(observed - self._fixture_unit_ids)
        if missing or extra:
            blockers.append(
                "resolved parity units differ from the checksum-locked fixture inventory; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        return tuple(blockers)

    def execute_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> ExperimentCellResult:
        self._require_preflight(manifest, cell)
        arm_outcomes: list[dict[str, Any]] = []
        resource_receipts: list[ResourceReceipt] = []
        for arm_id in cell.arm_ids:
            worker = self._configuration.workers[arm_id]
            response, host_receipt, resource_receipt = self._execute_arm(
                manifest,
                cell,
                worker,
            )
            resource_receipts.append(resource_receipt)
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
            arm_outcomes.append(outcome)
        return ExperimentCellResult(
            cell_id=cell.cell_id,
            status="succeeded",
            payload={
                "frozen_binding_receipt_hash": self._binding_receipt_hash,
                "parity_contract_hash": self._contract_hash,
                "host_execution_receipt_protocol": HOST_EXECUTION_RECEIPT_PROTOCOL,
                "arms": arm_outcomes,
            },
            resource_receipts=tuple(resource_receipts),
        )

    def _build_fingerprint(self) -> dict[str, Any]:
        declarations: dict[str, Any] = {}
        bindings: dict[str, Any] = {}
        for arm_id, worker in self._configuration.workers.items():
            release_id = (
                f"upstream:{self._configuration.method_id}"
                if arm_id == _UPSTREAM_ARM
                else self._configuration.candidate_release_id
            )
            runtime_entrypoint = (
                f"{worker.adapter_checkout_environment}:{worker.entrypoint_relative_path}"
            )
            declarations[arm_id] = {
                "implementation_kind": (
                    "upstream_reference" if arm_id == _UPSTREAM_ARM else "agentinputlab_candidate"
                ),
                "implementation_id": (
                    f"host-process:{content_hash({'arm_id': arm_id, 'command': worker.command_identity})}"
                ),
                "runtime_entrypoint": runtime_entrypoint,
                "source_revision": worker.expected_checkout_revision,
                "source_tree": worker.expected_checkout_tree,
                "release_id": release_id,
            }
            bindings[arm_id] = {
                "checkout_environment": worker.checkout_environment,
                "checkout": worker.checkout_identity.to_dict(),
                "adapter_checkout_environment": worker.adapter_checkout_environment,
                "adapter_checkout": worker.adapter_checkout_identity.to_dict(),
                "command": worker.command_identity.to_dict(),
            }
        return {
            "schema_version": 1,
            "driver": DRIVER_PROTOCOL,
            "config_sha256": self._configuration.config_sha256,
            "method_id": self._configuration.method_id,
            "candidate_release_id": self._configuration.candidate_release_id,
            "execution_receipt_protocol": HOST_EXECUTION_RECEIPT_PROTOCOL,
            "parity_implementations": declarations,
            "host_process_bindings": bindings,
            "resource_limits": self._configuration.resource_limits.as_dict(),
            "pipe_limits": dict(self._configuration.pipe_limits),
            "supervisor": dict(self._supervisor.fingerprint()),
            "fixture_inventory_sha256": self._environment.get(UNIT_INVENTORY_SHA256_ENV),
        }

    def _require_preflight(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> None:
        if (
            self._manifest_hash is None
            or self._binding_receipt_hash is None
            or self._contract_hash is None
        ):
            raise ArtifactError("parity-process execute_cell requires successful preflight")
        if manifest.manifest_hash != self._manifest_hash:
            raise ArtifactError("parity-process driver received another manifest")
        if cell.manifest_hash != manifest.manifest_hash:
            raise ValidationError("parity-process cell belongs to another manifest")
        if set(cell.arm_ids) != set(_ARM_IDS):
            raise ValidationError("parity-process cell requires exact implementation arms")
        if cell.unit_id not in self._fixture_unit_ids:
            raise ArtifactError("parity-process cell is outside the preflighted fixture inventory")

    def _execute_arm(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
        worker: _WorkerSpec,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], ResourceReceipt]:
        expected_state = self._preflight_worker_state.get(worker.arm_id)
        request = _build_request(
            configuration=self._configuration,
            manifest=manifest,
            cell=cell,
            worker=worker,
        )
        input_text = canonical_json(request) + "\n"
        environment = self._worker_environment(worker)
        passed_values = environment_secret_values(
            self._environment,
            (
                *_BASE_PASSTHROUGH_ENVIRONMENTS,
                *worker.pass_environment,
            ),
        )
        process = execute_locked_process_worker(
            worker,
            supervisor=self._supervisor,
            input_text=input_text,
            limits=self._configuration.resource_limits,
            pipe_limits=self._configuration.pipe_limits,
            environment=environment,
            process_name=f"parity-process-{worker.arm_id}",
            config_sha256=self._configuration.config_sha256,
            expected_worker_state=expected_state,
        )
        if process.returncode != 0:
            message = redact_secret_text(
                process.stderr.strip() or "worker exited non-zero",
                secret_values=passed_values,
            )
            raise SupervisedWorkerError(
                f"parity-process {worker.arm_id} failed: {message}",
                resource_receipt=process.resource_receipt,
            )
        response = _parse_response(
            process,
            expected_request_hash=request["request_hash"],
            secret_values=passed_values,
        )
        host_receipt = _host_execution_receipt(
            config_sha256=self._configuration.config_sha256,
            manifest=manifest,
            cell=cell,
            worker=worker,
            request=request,
            process=process,
            checkout_binding_hash=self._checkout_binding_hashes[worker.arm_id],
            adapter_checkout_binding_hash=self._adapter_checkout_binding_hashes[worker.arm_id],
        )
        return response, host_receipt, process.resource_receipt

    def _worker_environment(self, worker: _WorkerSpec) -> Mapping[str, str]:
        names = (
            *_BASE_PASSTHROUGH_ENVIRONMENTS,
            *worker.pass_environment,
        )
        environment = dict(
            locked_process_worker_environment(
                worker,
                environment=self._environment,
                pass_environment=names,
                request_protocol=REQUEST_PROTOCOL,
            )
        )
        environment.update(
            {
                "AILAB_PARITY_ARM_ID": worker.arm_id,
                "AILAB_PARITY_PROTOCOL": REQUEST_PROTOCOL,
            }
        )
        return freeze_mapping(environment)


def factory() -> ParityProcessExperimentDriver:
    """Load the checksum-locked parity-process configuration from the environment."""

    path = os.environ.get(CONFIG_PATH_ENV)
    expected = os.environ.get(CONFIG_SHA256_ENV)
    if not path:
        raise ArtifactError(f"{CONFIG_PATH_ENV} is required")
    if not expected:
        raise ArtifactError(f"{CONFIG_SHA256_ENV} is required")
    return ParityProcessExperimentDriver.from_config(
        path,
        expected_sha256=expected,
        supervisor=worker_supervisor_from_environment(),
    )


LockedProcessWorker = _WorkerSpec


def load_locked_process_worker(
    arm_id: str,
    value: Any,
    *,
    environment: Mapping[str, str],
    checkout_environment: str,
    adapter_checkout_environment: str | None = None,
) -> LockedProcessWorker:
    """Resolve one exact-SHA worker for another host-process experiment."""

    return _load_worker(
        arm_id,
        value,
        environment=environment,
        expected_checkout_environment=checkout_environment,
        expected_adapter_checkout_environment=(
            checkout_environment
            if adapter_checkout_environment is None
            else adapter_checkout_environment
        ),
    )


def attest_locked_process_worker(
    worker: LockedProcessWorker,
) -> Mapping[str, Any]:
    """Independently re-attest a worker checkout and command."""

    return freeze_mapping(_attest_worker(worker).to_dict())


def locked_process_worker_environment(
    worker: LockedProcessWorker,
    *,
    environment: Mapping[str, str],
    pass_environment: Sequence[str],
    request_protocol: str,
) -> Mapping[str, str]:
    """Build a minimal environment with checkout-confined Python imports."""

    missing = sorted(name for name in pass_environment if not environment.get(name))
    if missing:
        raise ArtifactError(
            f"locked process {worker.arm_id} missing environments: " + ", ".join(missing)
        )
    result = {name: environment[name] for name in pass_environment}
    python_paths = [str(worker.checkout)]
    source_root = worker.checkout / "src"
    if source_root.is_dir():
        python_paths.append(str(source_root))
    result.update(
        {
            "AILAB_PROCESS_ARM_ID": worker.arm_id,
            "AILAB_PROCESS_PROTOCOL": request_protocol,
            "LANG": "C",
            "LC_ALL": "C",
            "PWD": str(worker.checkout),
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "TZ": "UTC",
        }
    )
    return freeze_mapping(result)


def execute_locked_process_worker(
    worker: LockedProcessWorker,
    *,
    supervisor: WorkerSupervisor,
    input_text: str,
    limits: ResourceLimits,
    pipe_limits: Mapping[str, int],
    environment: Mapping[str, str],
    process_name: str,
    config_sha256: str,
    expected_worker_state: Mapping[str, Any],
) -> SupervisedProcessResult:
    """Run one fresh worker and fail if its identity changes around the call."""

    before = attest_locked_process_worker(worker)
    if before != freeze_mapping(expected_worker_state):
        raise ArtifactError(
            f"locked process {worker.arm_id} command or checkout drifted after preflight"
        )
    try:
        identity = {
            "kind": "content_attested_checkout_process",
            "arm_id": worker.arm_id,
            "checkout": worker.checkout_identity.to_dict(),
            "adapter_checkout": worker.adapter_checkout_identity.to_dict(),
            "command": worker.command_identity.to_dict(),
            "config_sha256": config_sha256,
        }
        if isinstance(
            supervisor,
            (DockerWorkerSupervisor, LocalSubprocessSupervisor),
        ):
            process = supervisor.execute_locked_worker(
                worker.argv,
                checkout=worker.checkout,
                adapter_checkout=worker.adapter_checkout,
                entrypoint_relative_path=str(worker.entrypoint_relative_path),
                input_text=input_text,
                limits=limits,
                max_input_bytes=pipe_limits["max_input_bytes"],
                max_stdout_bytes=pipe_limits["max_stdout_bytes"],
                max_stderr_bytes=pipe_limits["max_stderr_bytes"],
                environment=environment,
                process_name=process_name,
                image_identity=identity,
            )
        else:
            process = supervisor.execute(
                worker.argv,
                input_text=input_text,
                limits=limits,
                max_input_bytes=pipe_limits["max_input_bytes"],
                max_stdout_bytes=pipe_limits["max_stdout_bytes"],
                max_stderr_bytes=pipe_limits["max_stderr_bytes"],
                cwd=str(worker.checkout),
                environment=environment,
                process_name=process_name,
                image_identity=identity,
            )
    except SupervisedWorkerError as exc:
        try:
            after = attest_locked_process_worker(worker)
        except (ArtifactError, ExternalDependencyError, ValidationError) as drift:
            error = ArtifactError(
                f"locked process {worker.arm_id} identity changed during failed execution"
            )
            error.resource_receipt = exc.resource_receipt  # type: ignore[attr-defined]
            raise error from drift
        if after != freeze_mapping(expected_worker_state):
            error = ArtifactError(
                f"locked process {worker.arm_id} identity drifted during failed execution"
            )
            error.resource_receipt = exc.resource_receipt  # type: ignore[attr-defined]
            raise error from exc
        raise
    try:
        after = attest_locked_process_worker(worker)
    except (ArtifactError, ExternalDependencyError, ValidationError) as exc:
        error = ArtifactError(f"locked process {worker.arm_id} identity changed during execution")
        error.resource_receipt = process.resource_receipt  # type: ignore[attr-defined]
        raise error from exc
    if after != freeze_mapping(expected_worker_state):
        error = ArtifactError(f"locked process {worker.arm_id} identity drifted during execution")
        error.resource_receipt = process.resource_receipt  # type: ignore[attr-defined]
        raise error
    return process


def parse_locked_process_response(
    process: SupervisedProcessResult,
    *,
    expected_request_hash: str,
    secret_values: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Parse the shared strict response schema and reject identity echoes."""

    return _parse_response(
        process,
        expected_request_hash=expected_request_hash,
        secret_values=secret_values,
    )


def build_host_execution_receipt(
    *,
    config_sha256: str,
    manifest: ExperimentManifest,
    cell: ExperimentCell,
    worker: LockedProcessWorker,
    request: Mapping[str, Any],
    process: SupervisedProcessResult,
    checkout_binding_hash: str,
    adapter_checkout_binding_hash: str,
) -> Mapping[str, Any]:
    """Build the host-owned receipt consumed by the guarded validator."""

    return _host_execution_receipt(
        config_sha256=config_sha256,
        manifest=manifest,
        cell=cell,
        worker=worker,
        request=request,
        process=process,
        checkout_binding_hash=checkout_binding_hash,
        adapter_checkout_binding_hash=adapter_checkout_binding_hash,
    )


def _load_configuration(
    path: Path,
    *,
    expected_sha256: str,
    environment: Mapping[str, str],
) -> _DriverConfiguration:
    require_sha256(expected_sha256, label=CONFIG_SHA256_ENV)
    unresolved_source = path.expanduser()
    if unresolved_source.is_symlink():
        raise ArtifactError("parity-process config must be a regular non-symlink file")
    source = unresolved_source.resolve()
    if not source.is_file():
        raise ArtifactError("parity-process config must be a regular non-symlink file")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ArtifactError("parity-process config checksum differs from its environment lock")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("parity-process config is not valid UTF-8 JSON") from exc
    data = _strict_mapping(raw, _CONFIG_FIELDS, label="parity-process config")
    require_secret_free(data, label="parity-process config")
    if data["schema_version"] != 1 or data["protocol"] != DRIVER_PROTOCOL:
        raise ValidationError("unsupported parity-process config schema or protocol")
    method_id = _nonempty_text(data["method_id"], "method_id")
    candidate_release_id = _nonempty_text(
        data["candidate_release_id"],
        "candidate_release_id",
    )
    raw_workers = _strict_mapping(
        data["workers"],
        frozenset(_ARM_IDS),
        label="parity-process workers",
    )
    workers = {
        arm_id: _load_worker(
            arm_id,
            raw_workers[arm_id],
            environment=environment,
        )
        for arm_id in _ARM_IDS
    }
    if workers[_UPSTREAM_ARM].checkout == workers[_CANDIDATE_ARM].checkout:
        raise ValidationError("parity-process arms cannot use the same checkout")
    if workers[_UPSTREAM_ARM].entrypoint == workers[_CANDIDATE_ARM].entrypoint:
        raise ValidationError("parity-process arms cannot use the same entrypoint")
    if workers[_UPSTREAM_ARM].argv == workers[_CANDIDATE_ARM].argv:
        raise ValidationError("parity-process arms cannot use the same command")
    limits_data = _strict_mapping(
        data["resource_limits"],
        frozenset(
            {
                "wall_timeout_seconds",
                "memory_bytes",
                "cpus",
                "pids_limit",
            }
        ),
        label="parity-process resource_limits",
    )
    resource_limits = ResourceLimits(**limits_data)
    pipe_data = _strict_mapping(
        data["pipe_limits"],
        _PIPE_FIELDS,
        label="parity-process pipe_limits",
    )
    pipe_limits = {
        name: _positive_integer(pipe_data[name], f"pipe_limits.{name}")
        for name in sorted(_PIPE_FIELDS)
    }
    shared_parameters = _canonical_mapping(
        data["shared_request_parameters"],
        label="shared_request_parameters",
    )
    require_secret_free(shared_parameters, label="parity-process shared parameters")
    return _DriverConfiguration(
        method_id=method_id,
        candidate_release_id=candidate_release_id,
        workers=workers,
        resource_limits=resource_limits,
        pipe_limits=freeze_mapping(pipe_limits),
        shared_request_parameters=shared_parameters,
        config_sha256=actual_sha256,
    )


def _load_worker(
    arm_id: str,
    raw: Any,
    *,
    environment: Mapping[str, str],
    expected_checkout_environment: str | None = None,
    expected_adapter_checkout_environment: str | None = None,
) -> _WorkerSpec:
    data = _strict_mapping(raw, _WORKER_FIELDS, label=f"worker {arm_id}")
    checkout_environment = _nonempty_text(
        data["checkout_environment"],
        f"{arm_id}.checkout_environment",
    )
    required_checkout_environment = (
        _CHECKOUT_ENVIRONMENTS[arm_id]
        if expected_checkout_environment is None
        else expected_checkout_environment
    )
    if checkout_environment != required_checkout_environment:
        raise ValidationError(f"{arm_id} must use {required_checkout_environment}")
    checkout_value = environment.get(checkout_environment)
    if not checkout_value:
        raise ArtifactError(f"missing checkout environment {checkout_environment}")
    checkout = Path(checkout_value).expanduser().resolve()
    expected_revision = _git_object_id(
        data["expected_checkout_revision"],
        f"{arm_id}.expected_checkout_revision",
    )
    expected_tree = _git_object_id(
        data["expected_checkout_tree"],
        f"{arm_id}.expected_checkout_tree",
    )
    checkout_identity = _attest_checkout(
        checkout,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
    )
    adapter_checkout_environment = _nonempty_text(
        data["adapter_checkout_environment"],
        f"{arm_id}.adapter_checkout_environment",
    )
    required_adapter_environment = (
        "AILAB_CANDIDATE_CHECKOUT"
        if expected_adapter_checkout_environment is None
        else expected_adapter_checkout_environment
    )
    if adapter_checkout_environment != required_adapter_environment:
        raise ValidationError(f"{arm_id} adapter must use {required_adapter_environment}")
    adapter_checkout_value = environment.get(adapter_checkout_environment)
    if not adapter_checkout_value:
        raise ArtifactError(f"missing adapter checkout environment {adapter_checkout_environment}")
    adapter_checkout = Path(adapter_checkout_value).expanduser().resolve()
    expected_adapter_revision = _git_object_id(
        data["expected_adapter_checkout_revision"],
        f"{arm_id}.expected_adapter_checkout_revision",
    )
    expected_adapter_tree = _git_object_id(
        data["expected_adapter_checkout_tree"],
        f"{arm_id}.expected_adapter_checkout_tree",
    )
    adapter_checkout_identity = _attest_checkout(
        adapter_checkout,
        expected_revision=expected_adapter_revision,
        expected_tree=expected_adapter_tree,
    )
    executable = (
        Path(_nonempty_text(data["executable_path"], f"{arm_id}.executable_path"))
        .expanduser()
        .resolve()
    )
    if not executable.is_file():
        raise ArtifactError(f"{arm_id} executable does not exist")
    executable_sha256 = _sha256_text_field(
        data["executable_sha256"],
        f"{arm_id}.executable_sha256",
    )
    if _sha256_file(executable) != executable_sha256:
        raise ArtifactError(f"{arm_id} executable checksum differs from config")
    entrypoint_relative_path = _relative_entrypoint(
        data["entrypoint"],
        label=f"{arm_id}.entrypoint",
    )
    unresolved_entrypoint = adapter_checkout / entrypoint_relative_path
    if unresolved_entrypoint.is_symlink():
        raise ArtifactError(f"{arm_id} entrypoint must not be a symlink")
    entrypoint = unresolved_entrypoint.resolve()
    if not entrypoint.is_relative_to(adapter_checkout):
        raise ValidationError(f"{arm_id} entrypoint escapes its adapter checkout")
    if not entrypoint.is_file() or not stat.S_ISREG(entrypoint.stat().st_mode):
        raise ArtifactError(f"{arm_id} entrypoint must be a regular non-symlink file")
    _require_tracked_file(adapter_checkout, entrypoint_relative_path)
    entrypoint_sha256 = _sha256_text_field(
        data["entrypoint_sha256"],
        f"{arm_id}.entrypoint_sha256",
    )
    if _sha256_file(entrypoint) != entrypoint_sha256:
        raise ArtifactError(f"{arm_id} entrypoint checksum differs from config")
    arguments = _string_sequence(data["arguments"], label=f"{arm_id}.arguments")
    pass_environment = _environment_names(
        data["pass_environment"],
        label=f"{arm_id}.pass_environment",
    )
    reserved = sorted(set(pass_environment) & _RESERVED_WORKER_ENVIRONMENTS)
    if reserved:
        raise ValidationError(
            f"{arm_id} pass_environment contains host-controlled names: " + ", ".join(reserved)
        )
    request_parameters = _canonical_mapping(
        data["request_parameters"],
        label=f"{arm_id}.request_parameters",
    )
    require_secret_free(
        request_parameters,
        label=f"parity-process {arm_id} request parameters",
    )
    python_paths = [checkout]
    if (checkout / "src").is_dir():
        python_paths.append(checkout / "src")
    command_identity = _CommandIdentity(
        executable_resolved_path_sha256=_sha256_text(str(executable)),
        executable_content_sha256=executable_sha256,
        entrypoint_relative_path=entrypoint_relative_path,
        entrypoint_resolved_path_sha256=_sha256_text(str(entrypoint)),
        entrypoint_content_sha256=entrypoint_sha256,
        argv_sha256=content_hash(
            {
                "argv": (
                    str(executable),
                    str(entrypoint),
                    *arguments,
                )
            }
        ),
        cwd_resolved_path_sha256=checkout_identity.resolved_path_sha256,
        pythonpath_resolved_path_sha256=tuple(
            _sha256_text(str(item.resolve())) for item in python_paths
        ),
    )
    return _WorkerSpec(
        arm_id=arm_id,
        checkout_environment=checkout_environment,
        checkout=checkout,
        expected_checkout_revision=expected_revision,
        expected_checkout_tree=expected_tree,
        adapter_checkout_environment=adapter_checkout_environment,
        adapter_checkout=adapter_checkout,
        expected_adapter_checkout_revision=expected_adapter_revision,
        expected_adapter_checkout_tree=expected_adapter_tree,
        executable=executable,
        executable_sha256=executable_sha256,
        entrypoint_relative_path=entrypoint_relative_path,
        entrypoint=entrypoint,
        entrypoint_sha256=entrypoint_sha256,
        arguments=arguments,
        pass_environment=pass_environment,
        request_parameters=request_parameters,
        checkout_identity=checkout_identity,
        adapter_checkout_identity=adapter_checkout_identity,
        command_identity=command_identity,
    )


def _attest_worker(worker: _WorkerSpec) -> _CommandIdentity:
    checkout = _attest_checkout(
        worker.checkout,
        expected_revision=worker.expected_checkout_revision,
        expected_tree=worker.expected_checkout_tree,
    )
    adapter_checkout = _attest_checkout(
        worker.adapter_checkout,
        expected_revision=worker.expected_adapter_checkout_revision,
        expected_tree=worker.expected_adapter_checkout_tree,
    )
    if _sha256_file(worker.executable) != worker.executable_sha256:
        raise ArtifactError(f"{worker.arm_id} executable changed")
    unresolved_entrypoint = worker.adapter_checkout / worker.entrypoint_relative_path
    if unresolved_entrypoint.is_symlink():
        raise ArtifactError(f"{worker.arm_id} entrypoint path changed")
    entrypoint = unresolved_entrypoint.resolve()
    if (
        entrypoint != worker.entrypoint
        or not entrypoint.is_relative_to(worker.adapter_checkout)
        or not entrypoint.is_file()
    ):
        raise ArtifactError(f"{worker.arm_id} entrypoint path changed")
    _require_tracked_file(worker.adapter_checkout, worker.entrypoint_relative_path)
    if _sha256_file(entrypoint) != worker.entrypoint_sha256:
        raise ArtifactError(f"{worker.arm_id} entrypoint changed")
    if checkout.to_dict() != worker.checkout_identity.to_dict():
        raise ArtifactError(f"{worker.arm_id} checkout identity changed")
    if adapter_checkout.to_dict() != worker.adapter_checkout_identity.to_dict():
        raise ArtifactError(f"{worker.arm_id} adapter checkout identity changed")
    return worker.command_identity


def _attest_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    expected_tree: str,
) -> _CheckoutIdentity:
    if not checkout.is_dir():
        raise ArtifactError("parity-process checkout is unavailable")
    top_level = Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve()
    if top_level != checkout:
        raise ArtifactError("parity-process checkout must identify its Git root")
    head = _git(checkout, "rev-parse", "HEAD")
    tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    if head != expected_revision:
        raise ArtifactError("parity-process checkout HEAD differs from config")
    if tree != expected_tree:
        raise ArtifactError("parity-process checkout tree differs from config")
    status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ArtifactError("parity-process checkout must have a clean working tree")
    return _CheckoutIdentity(
        resolved_path_sha256=_sha256_text(str(checkout)),
        git_head=head,
        git_tree=tree,
    )


def _require_tracked_file(checkout: Path, relative_path: str) -> None:
    try:
        _git(checkout, "ls-files", "--error-unmatch", "--", relative_path)
    except ArtifactError as exc:
        raise ArtifactError("parity-process entrypoint must be tracked by Git") from exc


def _build_request(
    *,
    configuration: _DriverConfiguration,
    manifest: ExperimentManifest,
    cell: ExperimentCell,
    worker: _WorkerSpec,
) -> Mapping[str, Any]:
    arm = next(item for item in manifest.arms if item.arm_id == worker.arm_id)
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
        "method": {
            "method_id": configuration.method_id,
            "candidate_release_id": configuration.candidate_release_id,
            "profile_kind": arm.profile_kind.value,
            "family": arm.family.value,
        },
        "arm": {
            "arm_id": worker.arm_id,
            "release_id": (
                f"upstream:{configuration.method_id}"
                if worker.arm_id == _UPSTREAM_ARM
                else configuration.candidate_release_id
            ),
        },
        "parameters": {
            "shared": configuration.shared_request_parameters,
            "worker": worker.request_parameters,
        },
        "asset_environments": list(_BASE_PASSTHROUGH_ENVIRONMENTS),
    }
    require_secret_free(payload, label="parity-process worker request")
    request_hash = content_hash(payload)
    return freeze_mapping({**payload, "request_hash": request_hash})


def _parse_response(
    process: SupervisedProcessResult,
    *,
    expected_request_hash: str,
    secret_values: Sequence[str] = (),
) -> Mapping[str, Any]:
    try:
        raw = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ParityProcessProtocolError(
            "parity-process worker stdout is not one JSON value",
            resource_receipt=process.resource_receipt,
        ) from exc
    try:
        require_secret_values_absent(
            raw,
            secret_values=secret_values,
            label="parity-process worker response",
        )
        _reject_forbidden_worker_keys(raw)
        data = _strict_mapping(
            raw,
            _RESPONSE_FIELDS,
            label="parity-process response",
        )
        if data["schema_version"] != 1 or data["protocol"] != RESPONSE_PROTOCOL:
            raise ValidationError("unsupported parity-process worker response")
        if data["request_hash"] != expected_request_hash:
            raise ValidationError("parity-process response belongs to another request")
        status = data["status"]
        if status not in {"succeeded", "failed"}:
            raise ValidationError("parity-process response status is invalid")
        metrics = _metrics(data["metrics"])
        usage_data = _strict_mapping(
            data["usage"],
            _USAGE_FIELDS,
            label="parity-process usage",
        )
        usage = ResourceUsage(**usage_data).as_dict()
        usage.pop("total_cost_usd")
        output = _canonical_mapping(data["output"], label="parity-process output")
        failure = data["failure"]
        if status == "succeeded":
            if not metrics:
                raise ValidationError("successful parity-process response requires metrics")
            if failure is not None:
                raise ValidationError("successful parity-process response cannot have failure")
        else:
            if metrics or output:
                raise ValidationError("failed parity-process response cannot have metrics/output")
            failure = _strict_mapping(
                failure,
                frozenset({"scope", "category", "message"}),
                label="parity-process failure",
            )
            if failure["scope"] != "algorithm":
                raise ValidationError("parity-process arm failure must be algorithm-scoped")
            _nonempty_text(failure["category"], "failure.category")
            _nonempty_text(failure["message"], "failure.message")
        normalized = {
            "schema_version": 1,
            "protocol": RESPONSE_PROTOCOL,
            "request_hash": expected_request_hash,
            "status": status,
            "metrics": metrics,
            "usage": usage,
            "output": output,
            "failure": failure,
        }
        require_secret_free(normalized, label="parity-process worker response")
        return freeze_mapping(normalized)
    except (ArtifactError, TypeError, ValueError, ValidationError) as exc:
        raise ParityProcessProtocolError(
            redact_secret_text(str(exc), secret_values=secret_values),
            resource_receipt=process.resource_receipt,
        ) from exc


def _host_execution_receipt(
    *,
    config_sha256: str,
    manifest: ExperimentManifest,
    cell: ExperimentCell,
    worker: _WorkerSpec,
    request: Mapping[str, Any],
    process: SupervisedProcessResult,
    checkout_binding_hash: str,
    adapter_checkout_binding_hash: str,
) -> Mapping[str, Any]:
    receipt_without_hash = {
        "schema_version": 1,
        "protocol": HOST_EXECUTION_RECEIPT_PROTOCOL,
        "arm_id": worker.arm_id,
        "manifest_hash": manifest.manifest_hash,
        "cell_id": cell.cell_id,
        "request_hash": request["request_hash"],
        "config_sha256": config_sha256,
        "checkout_environment": worker.checkout_environment,
        "checkout_binding_hash": checkout_binding_hash,
        "adapter_checkout_environment": worker.adapter_checkout_environment,
        "adapter_checkout_binding_hash": adapter_checkout_binding_hash,
        "checkout": {
            **worker.checkout_identity.to_dict(),
            "git_clean_before": True,
            "git_clean_after": True,
        },
        "adapter_checkout": {
            **worker.adapter_checkout_identity.to_dict(),
            "git_clean_before": True,
            "git_clean_after": True,
        },
        "command": worker.command_identity.to_dict(),
        "process": {
            "resource_receipt_sha256": content_hash(process.resource_receipt.as_dict()),
            "stdout_sha256": _sha256_text(process.stdout),
            "stderr_sha256": _sha256_text(process.stderr),
        },
    }
    return freeze_mapping(
        {
            **receipt_without_hash,
            "receipt_hash": content_hash(receipt_without_hash),
        }
    )


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValidationError(f"{label} fields are invalid")
    return dict(value)


def _load_fixture_inventory(
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    raw_path = environment.get(UNIT_INVENTORY_ENV)
    expected_sha256 = environment.get(UNIT_INVENTORY_SHA256_ENV)
    if not raw_path:
        raise ArtifactError(f"{UNIT_INVENTORY_ENV} is required")
    if not expected_sha256:
        raise ArtifactError(f"{UNIT_INVENTORY_SHA256_ENV} is required")
    require_sha256(expected_sha256, label=UNIT_INVENTORY_SHA256_ENV)
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ArtifactError("parity-process fixture inventory is unavailable")
    if _sha256_file(path) != expected_sha256:
        raise ArtifactError("parity-process fixture inventory checksum differs")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("parity-process fixture inventory is not valid UTF-8 JSON") from exc
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or not raw
        or any(not isinstance(item, str) or not item.strip() for item in raw)
        or len(raw) != len(set(raw))
    ):
        raise ValidationError(
            "parity-process fixture inventory must be a non-empty unique string array"
        )
    return tuple(raw)


def _canonical_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    try:
        return freeze_mapping(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be canonical JSON") from exc


def _metrics(value: Any) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValidationError("parity-process metrics must be an object")
    result: dict[str, float] = {}
    for name, raw in value.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(raw, bool)
            or not isinstance(raw, (int, float))
        ):
            raise ValidationError("parity-process metrics must be named numerics")
        numeric = float(raw)
        if not -float("inf") < numeric < float("inf"):
            raise ValidationError("parity-process metrics must be finite")
        result[name] = numeric
    return freeze_mapping(result)


def _reject_forbidden_worker_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(set(value) & _RESPONSE_FORBIDDEN_KEYS)
        if forbidden:
            raise ValidationError(
                "worker may not emit host-owned identity fields: " + ", ".join(forbidden)
            )
        for item in value.values():
            _reject_forbidden_worker_keys(item)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for item in value:
            _reject_forbidden_worker_keys(item)


def _relative_entrypoint(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValidationError(f"{label} must be a normalized relative POSIX path")
    return path.as_posix()


def _environment_names(value: Any, *, label: str) -> tuple[str, ...]:
    names = _string_sequence(value, label=label)
    if len(names) != len(set(names)):
        raise ValidationError(f"{label} cannot contain duplicates")
    if any(_ENVIRONMENT_NAME.fullmatch(name) is None for name in names):
        raise ValidationError(f"{label} contains an invalid environment name")
    return names


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) or "\x00" in item for item in value)
    ):
        raise ValidationError(f"{label} must be a string array")
    return tuple(value)


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValidationError(f"{label} must be a non-empty string")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _sha256_text_field(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a SHA-256 digest")
    require_sha256(value, label=label)
    return value


def _git_object_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_OBJECT_ID.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a full lowercase Git object ID")
    return value


def _git(path: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(path), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArtifactError("cannot independently attest parity-process Git checkout") from exc
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "CONFIG_PATH_ENV",
    "CONFIG_SHA256_ENV",
    "DRIVER_PROTOCOL",
    "REQUEST_PROTOCOL",
    "RESPONSE_PROTOCOL",
    "UNIT_INVENTORY_ENV",
    "UNIT_INVENTORY_SHA256_ENV",
    "LockedProcessWorker",
    "ParityProcessExperimentDriver",
    "ParityProcessProtocolError",
    "attest_locked_process_worker",
    "build_host_execution_receipt",
    "execute_locked_process_worker",
    "factory",
    "load_locked_process_worker",
    "locked_process_worker_environment",
    "parse_locked_process_response",
]
