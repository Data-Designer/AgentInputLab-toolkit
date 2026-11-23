"""Production builders for checksum-locked external atomic arena workers.

These builders keep τ³, STATE-Bench, ToolSandbox, and AppWorld semantics in
their pinned upstream runtimes.  AgentInputLab sends one complete two-arm
request to a reviewed worker process; that worker must restore the protected
origin twice, run independent instances, invoke the official scorer, and
return the typed atomic receipt.  The host does not emulate an official
environment or accept two independently scored single-arm callbacks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.benchmarks import (
    ATOMIC_EXTERNAL_PAIR_CONTRACT,
    AtomicReplayArmReceipt,
    AtomicReplayArmResult,
    AtomicReplayPairResult,
    AtomicReplayReceipt,
    AtomicReplayRequest,
)
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import (
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
from agentinputlab.contracts.state import (
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    Message,
)
from agentinputlab.experiments.drivers.external_arena import (
    ExternalArenaCheckpoint,
    ExternalArmPayload,
    ExternalArmPayloadBuildRequest,
    ExternalBridgeBuildRequest,
    ExternalCheckpointBuildRequest,
    ExternalUnitPreflightRequest,
)
from agentinputlab.experiments.drivers.parity_process import (
    LockedProcessWorker,
    attest_locked_process_worker,
    execute_locked_process_worker,
    load_locked_process_worker,
    locked_process_worker_environment,
)
from agentinputlab.experiments.schema import ExperimentArmKind
from agentinputlab.experiments.secrets import (
    environment_secret_values,
    redact_secret_text,
    require_secret_free,
    require_secret_values_absent,
)
from agentinputlab.integrations.external_workers.supervisor import (
    LocalSubprocessSupervisor,
    SupervisedWorkerError,
    WorkerSupervisor,
    worker_supervisor_from_environment,
)
from agentinputlab.observability import ResourceLimits, ResourceReceipt, ResourceUsage

CHECKPOINT_INVENTORY_PROTOCOL = "agentinputlab.external-checkpoint-inventory.v1"
ATOMIC_PROCESS_BRIDGE_PROTOCOL = "agentinputlab.atomic-process-bridge.v1"
ATOMIC_PROCESS_REQUEST_PROTOCOL = "agentinputlab.atomic-process-request.v1"
ATOMIC_PROCESS_RESPONSE_PROTOCOL = "agentinputlab.atomic-process-response.v1"
RELEASE_PAYLOAD_PROTOCOL = "agentinputlab.external-release-payload.v1"

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ANONYMIZED_GROUP = re.compile(r"^group_[0-9a-f]{64}$")
_MAX_INVENTORY_BYTES = 64 * 1024 * 1024
_CORE_FAMILIES = frozenset(
    {
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    }
)
_PRIVILEGED_KEY_PARTS = frozenset(
    {
        "answer",
        "criterion",
        "database",
        "evaluator",
        "gold",
        "grader",
        "label",
        "reference",
        "required_document",
        "requirement",
        "reward",
        "simulator_prompt",
        "target_action",
    }
)
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


@dataclass(frozen=True, slots=True)
class _Inventory:
    path: Path
    file_sha256: str
    checkpoints: Mapping[tuple[str, str, str], _CheckpointRecord]


@dataclass(frozen=True, slots=True)
class _CheckpointRecord:
    benchmark_id: str
    checkpoint_id: str
    group_id: str
    source_revision: str
    origin_state_hash: str
    replay_identity: str
    dependencies_by_model_slot: Mapping[str, Mapping[str, str]]
    visible_state: FrozenAgentState
    eligible_families: tuple[SemanticFamily, ...]

    def build(self, model_slot: str) -> ExternalArenaCheckpoint:
        dependencies = self.dependencies_by_model_slot.get(
            model_slot,
            self.dependencies_by_model_slot.get("*"),
        )
        if dependencies is None:
            raise ArtifactError(
                f"checkpoint inventory has no dependency binding for model slot {model_slot!r}"
            )
        return ExternalArenaCheckpoint(
            benchmark_id=self.benchmark_id,
            checkpoint_id=self.checkpoint_id,
            group_id=self.group_id,
            source_revision=self.source_revision,
            origin_state_hash=self.origin_state_hash,
            replay_identity=self.replay_identity,
            locked_dependency_hashes=dependencies,
            visible_state=self.visible_state,
            eligible_families=self.eligible_families,
        )


class FrozenCheckpointInventoryBuilder:
    """Load evaluator-blind checkpoints from an exact-byte inventory."""

    builder_kind = "arena_checkpoint"

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "builder_id": CHECKPOINT_INVENTORY_PROTOCOL,
                "schema_version": _SCHEMA_VERSION,
                "gold_policy": "recursive_key_denylist_and_visible_state_only",
                "group_policy": "privacy_safe_content_hash",
                "inventory_digest_policy": "config_lock_plus_environment_recheck.v1",
                "max_inventory_bytes": _MAX_INVENTORY_BYTES,
            }
        )

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            _load_inventory(parameters)
        except (ArtifactError, ValidationError) as exc:
            return (redact_secret_text(str(exc)),)
        return ()

    def build(self, request: ExternalCheckpointBuildRequest) -> ExternalArenaCheckpoint:
        if not isinstance(request, ExternalCheckpointBuildRequest):
            raise TypeError("checkpoint inventory builder requires ExternalCheckpointBuildRequest")
        inventory = _load_inventory(request.builder_parameters)
        key = (request.benchmark_id, request.split, request.checkpoint_id)
        try:
            record = inventory.checkpoints[key]
        except KeyError as exc:
            raise ArtifactError(
                "checkpoint inventory does not contain the requested benchmark/split/unit"
            ) from exc
        checkpoint = record.build(request.model_slot)
        if checkpoint.source_revision != request.source_revision:
            raise ArtifactError("checkpoint source revision differs from the request")
        return checkpoint

    def preflight_units(
        self,
        request: ExternalUnitPreflightRequest,
    ) -> Sequence[str]:
        if not isinstance(request, ExternalUnitPreflightRequest):
            raise TypeError("unit preflight requires ExternalUnitPreflightRequest")
        try:
            inventory = _load_inventory(request.builder_parameters)
        except (ArtifactError, ValidationError) as exc:
            return (redact_secret_text(str(exc)),)
        available = {
            checkpoint_id
            for benchmark_id, split, checkpoint_id in inventory.checkpoints
            if benchmark_id == request.benchmark_id and split == request.split
        }
        requested = set(request.unit_ids)
        missing = sorted(requested - available)
        if missing:
            return ("checkpoint inventory is missing resolved units: " + ", ".join(missing[:10]),)
        revisions = {
            record.source_revision
            for key, record in inventory.checkpoints.items()
            if key[0] == request.benchmark_id and key[1] == request.split and key[2] in requested
        }
        if revisions != {request.source_revision}:
            return ("checkpoint inventory source revision differs from unit preflight",)
        return ()

    def preflight_group_ids(
        self,
        request: ExternalUnitPreflightRequest,
    ) -> Mapping[str, str]:
        blockers = self.preflight_units(request)
        if blockers:
            raise ArtifactError("; ".join(blockers))
        inventory = _load_inventory(request.builder_parameters)
        return freeze_mapping(
            {
                checkpoint_id: inventory.checkpoints[
                    (request.benchmark_id, request.split, checkpoint_id)
                ].group_id
                for checkpoint_id in request.unit_ids
            }
        )


@dataclass(slots=True)
class _AtomicProcessBridge:
    atomic_replay_contract: str
    benchmark_id: str
    source_revision: str
    bridge_id: str
    bridge_revision: str
    scorer_id: str
    scorer_revision: str
    locked_dependency_hashes: Mapping[str, str]
    worker: LockedProcessWorker
    worker_state: Mapping[str, Any]
    supervisor: WorkerSupervisor
    resource_limits: ResourceLimits
    pipe_limits: Mapping[str, int]
    environment: Mapping[str, str] = field(repr=False, compare=False)
    config_identity: str
    last_resource_receipt: ResourceReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "locked_dependency_hashes",
            freeze_mapping(self.locked_dependency_hashes),
        )
        object.__setattr__(self, "worker_state", freeze_mapping(self.worker_state))
        object.__setattr__(self, "pipe_limits", freeze_mapping(self.pipe_limits))
        object.__setattr__(self, "environment", freeze_mapping(self.environment))

    def evaluate_pair(self, request: AtomicReplayRequest) -> AtomicReplayPairResult:
        if not isinstance(request, AtomicReplayRequest):
            raise TypeError("atomic process bridge requires AtomicReplayRequest")
        if (
            request.benchmark_id != self.benchmark_id
            or request.source_revision != self.source_revision
            or request.locked_dependency_hashes != self.locked_dependency_hashes
        ):
            raise ValidationError(
                "atomic process request differs from the constructed bridge identity"
            )
        base = {
            "schema_version": _SCHEMA_VERSION,
            "protocol": ATOMIC_PROCESS_REQUEST_PROTOCOL,
            "atomic_request": to_jsonable(request),
            "worker_parameters": self.worker.request_parameters,
        }
        require_secret_free(base, label="atomic process request")
        payload = {
            **base,
            "request_hash": content_hash(base),
        }
        environment = locked_process_worker_environment(
            self.worker,
            environment=self.environment,
            pass_environment=self.worker.pass_environment,
            request_protocol=ATOMIC_PROCESS_REQUEST_PROTOCOL,
        )
        try:
            process = execute_locked_process_worker(
                self.worker,
                supervisor=self.supervisor,
                input_text=canonical_json(payload) + "\n",
                limits=self.resource_limits,
                pipe_limits=self.pipe_limits,
                environment=environment,
                process_name=f"atomic-arena-{self.benchmark_id}",
                config_sha256=self.config_identity,
                expected_worker_state=self.worker_state,
            )
        except SupervisedWorkerError as exc:
            object.__setattr__(self, "last_resource_receipt", exc.resource_receipt)
            raise
        object.__setattr__(self, "last_resource_receipt", process.resource_receipt)
        if process.returncode != 0:
            raise SupervisedWorkerError(
                "atomic arena worker failed: "
                + redact_secret_text(
                    process.stderr.strip() or "non-zero exit",
                    secret_values=environment_secret_values(
                        self.environment,
                        self.worker.pass_environment,
                    ),
                ),
                resource_receipt=process.resource_receipt,
            )
        result = _parse_atomic_response(
            process.stdout,
            expected_request_hash=payload["request_hash"],
            secret_values=environment_secret_values(
                self.environment,
                self.worker.pass_environment,
            ),
        )
        return AtomicReplayPairResult(
            receipt=result.receipt,
            arms=result.arms,
            resource_receipts=(process.resource_receipt,),
        )


@dataclass(frozen=True, slots=True)
class _ParsedAtomicResponse:
    receipt: AtomicReplayReceipt
    arms: tuple[AtomicReplayArmResult, ...]


class AtomicProcessBridgeBuilder:
    """Construct an atomic bridge around one exact-checkout worker process."""

    builder_kind = "atomic_bridge"

    def __init__(self, supervisor: WorkerSupervisor | None = None) -> None:
        self._supervisor = supervisor or LocalSubprocessSupervisor()
        if not isinstance(self._supervisor, WorkerSupervisor):
            raise TypeError("atomic process builder supervisor must implement WorkerSupervisor")

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "builder_id": ATOMIC_PROCESS_BRIDGE_PROTOCOL,
                "schema_version": _SCHEMA_VERSION,
                "request_protocol": ATOMIC_PROCESS_REQUEST_PROTOCOL,
                "response_protocol": ATOMIC_PROCESS_RESPONSE_PROTOCOL,
                "worker_identity": "clean_git_checkout_plus_entrypoint_sha256",
                "supervisor": self._supervisor.fingerprint(),
                "atomicity": "one_process_call_per_baseline_treatment_pair",
            }
        )

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            parsed = _parse_bridge_parameters(parameters)
            worker = _load_atomic_worker(parsed, environment=os.environ)
            attest_locked_process_worker(worker)
            if isinstance(self._supervisor, LocalSubprocessSupervisor) and any(
                value is not None
                for value in (
                    parsed["resource_limits"].memory_bytes,
                    parsed["resource_limits"].cpus,
                    parsed["resource_limits"].pids_limit,
                )
            ):
                raise ValidationError(
                    "local atomic process supervision supports wall/pipe limits only"
                )
            missing = [name for name in worker.pass_environment if not os.environ.get(name)]
            if missing:
                raise ArtifactError(
                    "atomic process worker is missing environments: " + ", ".join(sorted(missing))
                )
        except (
            ArtifactError,
            ExternalDependencyError,
            ValidationError,
        ) as exc:
            return (redact_secret_text(str(exc)),)
        return ()

    def build(self, request: ExternalBridgeBuildRequest) -> _AtomicProcessBridge:
        if not isinstance(request, ExternalBridgeBuildRequest):
            raise TypeError("atomic process builder requires ExternalBridgeBuildRequest")
        parsed = _parse_bridge_parameters(request.builder_parameters)
        checkpoint = request.checkpoint
        if (
            parsed["benchmark_id"] != checkpoint.benchmark_id
            or parsed["source_revision"] != checkpoint.source_revision
        ):
            raise ValidationError("atomic process builder identity differs from its checkpoint")
        worker = _load_atomic_worker(parsed, environment=os.environ)
        state = attest_locked_process_worker(worker)
        missing = [name for name in worker.pass_environment if not os.environ.get(name)]
        if missing:
            raise ArtifactError(
                "atomic process worker is missing environments: " + ", ".join(sorted(missing))
            )
        worker_environment = {name: os.environ[name] for name in worker.pass_environment}
        return _AtomicProcessBridge(
            atomic_replay_contract=ATOMIC_EXTERNAL_PAIR_CONTRACT,
            benchmark_id=parsed["benchmark_id"],
            source_revision=parsed["source_revision"],
            bridge_id=parsed["bridge_id"],
            bridge_revision=parsed["bridge_revision"],
            scorer_id=parsed["scorer_id"],
            scorer_revision=parsed["scorer_revision"],
            locked_dependency_hashes=checkpoint.locked_dependency_hashes,
            worker=worker,
            worker_state=state,
            supervisor=self._supervisor,
            resource_limits=parsed["resource_limits"],
            pipe_limits=parsed["pipe_limits"],
            environment=worker_environment,
            config_identity=content_hash(request.builder_parameters),
        )


class ReleaseArmPayloadBuilder:
    """Serialize a release arm without interpreting or rewriting its algorithm."""

    builder_kind = "arm_payload"

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "builder_id": RELEASE_PAYLOAD_PROTOCOL,
                "schema_version": _SCHEMA_VERSION,
                "payload_semantics": "identity_plus_frozen_settings",
                "policy_arms": "unsupported_use_selector_specific_builder",
            }
        )

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        try:
            _parse_payload_parameters(parameters)
        except (ArtifactError, ValidationError) as exc:
            return (redact_secret_text(str(exc)),)
        return ()

    def build(self, request: ExternalArmPayloadBuildRequest) -> ExternalArmPayload:
        if not isinstance(request, ExternalArmPayloadBuildRequest):
            raise TypeError("release payload builder requires ExternalArmPayloadBuildRequest")
        if request.arm_kind is not ExperimentArmKind.RELEASE:
            raise ValidationError("release payload builder cannot construct selector policy arms")
        parameters = _parse_payload_parameters(request.builder_parameters)
        allowed = parameters["allowed_release_ids"]
        if request.release_id not in allowed:
            raise ValidationError("release payload is outside the builder allowlist")
        if request.release_id == "baseline.no_op" and request.family is not SemanticFamily.NO_OP:
            raise ValidationError("baseline.no_op payload requires the no_op family")
        if request.release_id != "baseline.no_op" and request.family not in _CORE_FAMILIES:
            raise ValidationError("release payload requires a core intervention family")
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "protocol": RELEASE_PAYLOAD_PROTOCOL,
            "arm_id": request.arm_id,
            "release_id": request.release_id,
            "family": request.family.value,
            "profile_kind": request.profile_kind.value,
            "model_slot": request.model_slot,
            "seed": request.seed,
            "checkpoint_identity_hash": request.checkpoint.identity_hash,
            "visible_state_hash": request.checkpoint.visible_state.state_hash,
            "pre_action": request.pre_action,
            "release_parameters": parameters["release_parameters"],
            "arm_parameters": request.arm_parameters,
        }
        require_secret_free(payload, label="external release payload")
        return ExternalArmPayload(
            arm_id=request.arm_id,
            release_id=request.release_id,
            family=request.family,
            profile_kind=request.profile_kind,
            payload=payload,
        )


def frozen_checkpoint_inventory_builder_factory() -> FrozenCheckpointInventoryBuilder:
    return FrozenCheckpointInventoryBuilder()


def atomic_process_bridge_builder_factory() -> AtomicProcessBridgeBuilder:
    return AtomicProcessBridgeBuilder(worker_supervisor_from_environment())


def release_arm_payload_builder_factory() -> ReleaseArmPayloadBuilder:
    return ReleaseArmPayloadBuilder()


def _load_inventory(parameters: Mapping[str, Any]) -> _Inventory:
    parsed = _parse_inventory_parameters(parameters)
    path_value = os.environ.get(parsed["path_environment"])
    environment_digest = os.environ.get(parsed["sha256_environment"])
    expected = parsed["expected_sha256"]
    if not path_value:
        raise ArtifactError(
            f"missing checkpoint inventory environment {parsed['path_environment']}"
        )
    if not environment_digest or _SHA256.fullmatch(environment_digest) is None:
        raise ArtifactError(
            f"missing or invalid inventory digest environment {parsed['sha256_environment']}"
        )
    if environment_digest != expected:
        raise ArtifactError("checkpoint inventory digest environment differs from the config lock")
    unresolved = Path(path_value).expanduser()
    if unresolved.is_symlink():
        raise ArtifactError("checkpoint inventory must not be a symlink")
    path = unresolved.resolve()
    if not path.is_file() or path.stat().st_size > _MAX_INVENTORY_BYTES:
        raise ArtifactError("checkpoint inventory is absent or exceeds the size limit")
    raw_bytes = path.read_bytes()
    observed = hashlib.sha256(raw_bytes).hexdigest()
    if observed != expected:
        raise ArtifactError("checkpoint inventory checksum mismatch")
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("checkpoint inventory is not valid UTF-8 JSON") from exc
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "protocol", "checkpoint_count", "checkpoints"}
        or raw["schema_version"] != _SCHEMA_VERSION
        or raw["protocol"] != CHECKPOINT_INVENTORY_PROTOCOL
        or not isinstance(raw["checkpoints"], list)
        or raw["checkpoint_count"] != len(raw["checkpoints"])
        or not raw["checkpoints"]
    ):
        raise ValidationError("checkpoint inventory schema is invalid")
    checkpoints: dict[tuple[str, str, str], _CheckpointRecord] = {}
    for index, item in enumerate(raw["checkpoints"]):
        key, record = _checkpoint_from_mapping(
            item,
            label=f"checkpoints[{index}]",
        )
        if key in checkpoints:
            raise ValidationError("checkpoint inventory contains duplicate units")
        checkpoints[key] = record
    return _Inventory(
        path=path,
        file_sha256=observed,
        checkpoints=MappingProxyType(checkpoints),
    )


def _parse_inventory_parameters(parameters: Mapping[str, Any]) -> Mapping[str, str]:
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "schema_version",
        "protocol",
        "path_environment",
        "sha256_environment",
        "expected_sha256",
    }:
        raise ValidationError("checkpoint inventory builder parameters are invalid")
    if (
        parameters["schema_version"] != _SCHEMA_VERSION
        or parameters["protocol"] != CHECKPOINT_INVENTORY_PROTOCOL
    ):
        raise ValidationError("unsupported checkpoint inventory builder protocol")
    for key in ("path_environment", "sha256_environment"):
        if (
            not isinstance(parameters[key], str)
            or _ENVIRONMENT_NAME.fullmatch(parameters[key]) is None
        ):
            raise ValidationError(f"checkpoint inventory {key} is invalid")
    if parameters["path_environment"] == parameters["sha256_environment"]:
        raise ValidationError("checkpoint inventory path and digest environments must differ")
    if (
        not isinstance(parameters["expected_sha256"], str)
        or _SHA256.fullmatch(parameters["expected_sha256"]) is None
    ):
        raise ValidationError("checkpoint inventory expected_sha256 must be a lowercase SHA-256")
    return freeze_mapping(parameters)


def _checkpoint_from_mapping(
    value: Any,
    *,
    label: str,
) -> tuple[tuple[str, str, str], _CheckpointRecord]:
    common = {
        "benchmark_id",
        "split",
        "checkpoint_id",
        "group_id",
        "source_revision",
        "origin_state_hash",
        "replay_identity",
        "visible_state",
        "visible_state_hash",
        "eligible_families",
    }
    legacy = common | {"locked_dependency_hashes"}
    model_aware = common | {"locked_dependency_hashes_by_model_slot"}
    if not isinstance(value, Mapping) or (set(value) != legacy and set(value) != model_aware):
        raise ValidationError(f"{label} fields are invalid")
    for name in ("benchmark_id", "split", "checkpoint_id", "source_revision"):
        _text(value[name], label=f"{label}.{name}")
    if (
        not isinstance(value["group_id"], str)
        or _ANONYMIZED_GROUP.fullmatch(value["group_id"]) is None
    ):
        raise ValidationError(f"{label}.group_id must be a privacy-safe group hash")
    _reject_privileged_keys(value["visible_state"], path=f"{label}.visible_state")
    state = _state_from_mapping(value["visible_state"], label=f"{label}.visible_state")
    if value["visible_state_hash"] != state.state_hash:
        raise ValidationError(f"{label}.visible_state_hash differs from the state")
    raw_families = value["eligible_families"]
    if isinstance(raw_families, (str, bytes)) or not isinstance(raw_families, Sequence):
        raise ValidationError(f"{label}.eligible_families must be an array")
    try:
        families = tuple(SemanticFamily(item) for item in raw_families)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label}.eligible_families contains an unknown family") from exc
    raw_dependencies = (
        {"*": value["locked_dependency_hashes"]}
        if "locked_dependency_hashes" in value
        else value["locked_dependency_hashes_by_model_slot"]
    )
    if (
        not isinstance(raw_dependencies, Mapping)
        or not raw_dependencies
        or any(
            not isinstance(model_slot, str)
            or not model_slot.strip()
            or not isinstance(dependencies, Mapping)
            for model_slot, dependencies in raw_dependencies.items()
        )
    ):
        raise ValidationError(f"{label}.locked_dependency_hashes_by_model_slot is invalid")
    record = _CheckpointRecord(
        benchmark_id=value["benchmark_id"],
        checkpoint_id=value["checkpoint_id"],
        group_id=value["group_id"],
        source_revision=value["source_revision"],
        origin_state_hash=value["origin_state_hash"],
        replay_identity=value["replay_identity"],
        dependencies_by_model_slot=freeze_mapping(raw_dependencies),
        visible_state=state,
        eligible_families=families,
    )
    for model_slot in raw_dependencies:
        record.build(model_slot)
    return (
        (value["benchmark_id"], value["split"], value["checkpoint_id"]),
        record,
    )


def _state_from_mapping(value: Any, *, label: str) -> FrozenAgentState:
    fields = {
        "query",
        "task_id",
        "instruction",
        "context",
        "messages",
        "tool_names",
        "metadata",
        "budget",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidationError(f"{label} fields are invalid")
    context_raw = value["context"]
    messages_raw = value["messages"]
    tools_raw = value["tool_names"]
    if (
        isinstance(context_raw, (str, bytes))
        or not isinstance(context_raw, Sequence)
        or isinstance(messages_raw, (str, bytes))
        or not isinstance(messages_raw, Sequence)
        or isinstance(tools_raw, (str, bytes))
        or not isinstance(tools_raw, Sequence)
    ):
        raise ValidationError(f"{label} sequence fields are invalid")
    context = tuple(
        _context_block(item, label=f"{label}.context[{index}]")
        for index, item in enumerate(context_raw)
    )
    messages = tuple(
        _message(item, label=f"{label}.messages[{index}]")
        for index, item in enumerate(messages_raw)
    )
    budget_raw = value["budget"]
    budget_fields = {
        "max_model_calls",
        "max_retrieval_calls",
        "max_tool_calls",
        "max_output_tokens",
        "max_context_tokens",
        "max_cost_usd",
    }
    if not isinstance(budget_raw, Mapping) or set(budget_raw) != budget_fields:
        raise ValidationError(f"{label}.budget fields are invalid")
    if not isinstance(value["metadata"], Mapping):
        raise ValidationError(f"{label}.metadata must be an object")
    return FrozenAgentState(
        query=value["query"],
        task_id=value["task_id"],
        instruction=value["instruction"],
        context=context,
        messages=messages,
        tool_names=tuple(tools_raw),
        metadata=value["metadata"],
        budget=ExecutionBudget(**budget_raw),
    )


def _context_block(value: Any, *, label: str) -> ContextBlock:
    fields = {"text", "block_id", "source", "score", "metadata"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidationError(f"{label} fields are invalid")
    if not isinstance(value["metadata"], Mapping):
        raise ValidationError(f"{label}.metadata must be an object")
    return ContextBlock(**value)


def _message(value: Any, *, label: str) -> Message:
    if not isinstance(value, Mapping) or set(value) != {"role", "content", "name"}:
        raise ValidationError(f"{label} fields are invalid")
    return Message(**value)


def _reject_privileged_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(part in normalized for part in _PRIVILEGED_KEY_PARTS):
                raise ValidationError(f"{path} contains privileged field {key!r}")
            _reject_privileged_keys(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_privileged_keys(item, path=f"{path}[{index}]")


def _parse_bridge_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "protocol",
        "benchmark_id",
        "source_revision",
        "bridge_id",
        "bridge_revision",
        "scorer_id",
        "scorer_revision",
        "worker",
        "resource_limits",
        "pipe_limits",
    }
    if not isinstance(parameters, Mapping) or set(parameters) != fields:
        raise ValidationError("atomic process bridge parameters are invalid")
    if (
        parameters["schema_version"] != _SCHEMA_VERSION
        or parameters["protocol"] != ATOMIC_PROCESS_BRIDGE_PROTOCOL
    ):
        raise ValidationError("unsupported atomic process bridge protocol")
    for name in ("benchmark_id", "source_revision", "bridge_id", "scorer_id"):
        _text(parameters[name], label=f"atomic process {name}")
    for name in ("bridge_revision", "scorer_revision"):
        _sha256(parameters[name], label=f"atomic process {name}")
    worker = parameters["worker"]
    if not isinstance(worker, Mapping) or set(worker) != _WORKER_FIELDS:
        raise ValidationError("atomic process worker fields are invalid")
    limits = parameters["resource_limits"]
    if not isinstance(limits, Mapping) or set(limits) != {
        "wall_timeout_seconds",
        "memory_bytes",
        "cpus",
        "pids_limit",
    }:
        raise ValidationError("atomic process resource_limits fields are invalid")
    pipes = parameters["pipe_limits"]
    if not isinstance(pipes, Mapping) or set(pipes) != {
        "max_input_bytes",
        "max_stdout_bytes",
        "max_stderr_bytes",
    }:
        raise ValidationError("atomic process pipe_limits fields are invalid")
    normalized_pipes: dict[str, int] = {}
    for name, value in pipes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"atomic process {name} must be a positive integer")
        normalized_pipes[name] = value
    require_secret_free(parameters, label="atomic process bridge parameters")
    return freeze_mapping(
        {
            **dict(parameters),
            "resource_limits": ResourceLimits(**limits),
            "pipe_limits": normalized_pipes,
        }
    )


def _load_atomic_worker(
    parameters: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
) -> LockedProcessWorker:
    worker = parameters["worker"]
    return load_locked_process_worker(
        "atomic_pair",
        worker,
        environment=environment,
        checkout_environment=worker["checkout_environment"],
        adapter_checkout_environment=worker["adapter_checkout_environment"],
    )


def _parse_payload_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "schema_version",
        "protocol",
        "allowed_release_ids",
        "release_parameters",
    }:
        raise ValidationError("release payload builder parameters are invalid")
    if (
        parameters["schema_version"] != _SCHEMA_VERSION
        or parameters["protocol"] != RELEASE_PAYLOAD_PROTOCOL
    ):
        raise ValidationError("unsupported release payload builder protocol")
    raw_ids = parameters["allowed_release_ids"]
    if (
        isinstance(raw_ids, (str, bytes))
        or not isinstance(raw_ids, Sequence)
        or not raw_ids
        or len(raw_ids) != len(set(raw_ids))
        or any(not isinstance(item, str) or not item.strip() for item in raw_ids)
    ):
        raise ValidationError("allowed_release_ids must be unique non-empty strings")
    if not isinstance(parameters["release_parameters"], Mapping):
        raise ValidationError("release_parameters must be an object")
    require_secret_free(parameters, label="release payload builder parameters")
    return freeze_mapping(parameters)


def _parse_atomic_response(
    text: str,
    *,
    expected_request_hash: str,
    secret_values: Sequence[str] = (),
) -> _ParsedAtomicResponse:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError("atomic process worker returned invalid JSON") from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "protocol",
        "request_hash",
        "receipt",
        "arms",
    }:
        raise ValidationError("atomic process response fields are invalid")
    if (
        raw["schema_version"] != _SCHEMA_VERSION
        or raw["protocol"] != ATOMIC_PROCESS_RESPONSE_PROTOCOL
        or raw["request_hash"] != expected_request_hash
    ):
        raise ValidationError("atomic process response identity is invalid")
    require_secret_values_absent(
        raw,
        secret_values=secret_values,
        label="atomic process response",
    )
    require_secret_free(raw, label="atomic process response")
    receipt = _receipt_from_mapping(raw["receipt"])
    arms_raw = raw["arms"]
    if isinstance(arms_raw, (str, bytes)) or not isinstance(arms_raw, Sequence):
        raise ValidationError("atomic process response arms must be an array")
    arms = tuple(_arm_result_from_mapping(item) for item in arms_raw)
    return _ParsedAtomicResponse(receipt=receipt, arms=arms)


def _receipt_from_mapping(value: Any) -> AtomicReplayReceipt:
    fields = {
        "benchmark_id",
        "checkpoint_id",
        "source_revision",
        "bridge_id",
        "bridge_revision",
        "scorer_id",
        "scorer_revision",
        "origin_state_hash",
        "replay_identity",
        "seed",
        "locked_dependency_hashes",
        "arms",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidationError("atomic process receipt fields are invalid")
    raw_arms = value["arms"]
    if isinstance(raw_arms, (str, bytes)) or not isinstance(raw_arms, Sequence):
        raise ValidationError("atomic process receipt arms must be an array")
    receipts: list[AtomicReplayArmReceipt] = []
    arm_fields = {
        "role",
        "arm_id",
        "origin_state_hash",
        "replay_identity",
        "seed",
        "locked_dependency_hashes",
        "runtime_instance_id",
        "completed",
    }
    for item in raw_arms:
        if not isinstance(item, Mapping) or set(item) != arm_fields:
            raise ValidationError("atomic process arm receipt fields are invalid")
        receipts.append(AtomicReplayArmReceipt(**item))
    return AtomicReplayReceipt(
        **{name: value[name] for name in fields - {"arms"}},
        arms=tuple(receipts),
    )


def _arm_result_from_mapping(value: Any) -> AtomicReplayArmResult:
    if not isinstance(value, Mapping) or set(value) != {
        "role",
        "arm_id",
        "metrics",
        "trace_hash",
        "usage",
    }:
        raise ValidationError("atomic process arm result fields are invalid")
    usage = value["usage"]
    if not isinstance(usage, Mapping) or set(usage) != _USAGE_FIELDS:
        raise ValidationError("atomic process arm usage fields are invalid")
    return AtomicReplayArmResult(
        role=value["role"],
        arm_id=value["arm_id"],
        metrics=value["metrics"],
        trace_hash=value["trace_hash"],
        usage=ResourceUsage(**usage),
    )


def _text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(not character.isprintable() for character in value)
    ):
        raise ValidationError(f"{label} must be trimmed non-empty printable text")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "ATOMIC_PROCESS_BRIDGE_PROTOCOL",
    "ATOMIC_PROCESS_REQUEST_PROTOCOL",
    "ATOMIC_PROCESS_RESPONSE_PROTOCOL",
    "CHECKPOINT_INVENTORY_PROTOCOL",
    "RELEASE_PAYLOAD_PROTOCOL",
    "AtomicProcessBridgeBuilder",
    "FrozenCheckpointInventoryBuilder",
    "ReleaseArmPayloadBuilder",
    "atomic_process_bridge_builder_factory",
    "frozen_checkpoint_inventory_builder_factory",
    "release_arm_payload_builder_factory",
]
