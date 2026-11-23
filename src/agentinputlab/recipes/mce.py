"""Fail-closed external-L2 adapter for Meta Context Engineering (MCE).

The upstream algorithm evolves executable context interfaces.  AgentInputLab
therefore treats the upstream process, validation oracle, usage meter, and
artifact store as one externally supervised security boundary.  The host never
loads generated source and only accepts a content-addressed handle plus a
hash-linked regular-file manifest.

This module deliberately does *not* claim paper-result parity.  A successful
run proves the protocol and method-level execution path described here; model
credentials, task environments, and the container/microVM supervisor remain
operator supplied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExternalDependencyError,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import (
    content_hash,
    deep_freeze_jsonable,
    freeze_mapping,
    to_jsonable,
)
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    MethodWorkerResponse,
    finite_number,
    mapping_sequence,
    nonnegative_integer,
    required_text,
    resolve_worker,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

MCE_UPSTREAM_COMMIT = "c4b7a7c2ce3ffc4bf4a74c52d2dd8a9a8fb14c30"
MCE_PROTOCOL = "agentinputlab.mce.v2"
MCE_UPSTREAM_MAIN_SHA256 = "769b9ea180ea468530c38893ad15178422d4ba0bbb23ff183d96bde86fdd8ffe"
MCE_SEED_MODE = "python_once_and_manager_logical_call_index_v1"
MCE_PROVIDER_SUBCALL_SEED_MODE = "sha256_logical_seed_and_stable_subcall_index_63bit_v1"
MCE_SEMANTIC_PATCH_MANIFEST = freeze_mapping(
    {
        "schema": "agentinputlab.mce.semantic-patch.v2",
        "patch_id": "agentinputlab.mce.semantic-patch.v2",
        "upstream_commit": MCE_UPSTREAM_COMMIT,
        "upstream_main_sha256": MCE_UPSTREAM_MAIN_SHA256,
        "seams": {
            "mce.main.run_meta_agent": "isolated_manager.run_agent.meta",
            "mce.main.run_base_agent": "isolated_manager.run_agent.base",
            "mce.main.load_interfaces": ("isolated_manager.validate_interfaces_static"),
            "mce.main.batch_evaluate": ("trusted_oracle_supervisor.batch_evaluate"),
        },
        "seed_derivation": {
            "python_random": "base_seed_once_per_run",
            "manager_logical_call": "base_seed_plus_logical_call_index",
            "provider_subcall": ("sha256_canonical_logical_seed_and_stable_subcall_index_63bit"),
        },
        "generated_code_imported_by_worker": False,
    }
)
MCE_SEMANTIC_PATCH_MANIFEST_SHA256 = content_hash(MCE_SEMANTIC_PATCH_MANIFEST)
_EMPTY_SHA256 = content_hash("")
_EMPTY_INITIAL_CONTEXT = {
    "schema": "agentinputlab.mce.initial-context.v1",
    "files": (),
}
MCE_EMPTY_INITIAL_CONTEXT_SHA256 = content_hash(_EMPTY_INITIAL_CONTEXT)
_INLINE_CONTENT_KEYS = frozenset(
    {
        "blob",
        "body",
        "bytes",
        "code",
        "content",
        "inline_blob",
        "inline_body",
        "inline_bytes",
        "inline_code",
        "inline_content",
        "log",
        "logs",
        "source_code",
    }
)
_ARTIFACT_FILE_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "byte_size",
        "media_type",
        "file_type",
        "generated",
        "executable",
    }
)
_RESPONSE_ENVELOPE_FIELDS = frozenset(
    {
        "status",
        "request_id",
        "upstream_commit",
        "protocol_version",
        "payload",
        "usage",
    }
)
_ERROR_RESPONSE_ENVELOPE_FIELDS = frozenset(
    {
        "status",
        "failure_code",
        "message",
        "consumed",
        "usage",
    }
)
_MODEL_FINGERPRINT_FIELDS = frozenset({"provider", "model", "revision", "configuration_sha256"})
_RESPONSE_PAYLOAD_FIELDS = frozenset(
    {
        "sandbox_attestation_id",
        "host_code_executed",
        "artifact_egress",
        "split_audit_sha256",
        "interface_signature_sha256",
        "initial_context_sha256",
        "environment_identity_sha256",
        "model_fingerprints_sha256",
        "runtime_binding_sha256",
        "source_tree",
        "source_manifest_sha256",
        "license_manifest_sha256",
        "provenance_manifest_sha256",
        "storage_fingerprint_sha256",
        "semantic_patch_manifest_sha256",
        "files",
        "iterations",
        "selected_iteration",
        "artifact_manifest_sha256",
        "skill_sha256",
        "artifact_bundle_sha256",
        "artifact_handle",
        "provenance_status",
        "runtime_attestation",
    }
)
_ALLOWED_ARTIFACT_PREFIXES = (
    ".claude/skills/learning-context/",
    "context/",
    "interfaces/",
)
_ALLOWED_ARTIFACT_FILES = frozenset({"retrieve_context.py"})
_HANDLE_PREFIX = "urn:agentinputlab:mce:sha256:"


def _require_sha256(
    value: str,
    *,
    label: str,
    error_type: type[Exception] = ValidationError,
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{label} must be text")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise error_type(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _require_sha1(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExternalDependencyError(f"{label} must be a lowercase Git tree SHA-1")
    return value


@dataclass(frozen=True, slots=True)
class MCEExample:
    example_id: str
    input: Any
    expected_output: Any
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValidationError("MCE examples require a stable example_id")
        if self.split not in {"train", "validation"}:
            raise ValidationError("MCE examples may only use train/validation splits")
        try:
            object.__setattr__(self, "input", deep_freeze_jsonable(self.input))
            object.__setattr__(
                self,
                "expected_output",
                deep_freeze_jsonable(self.expected_output),
            )
            object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        except (TypeError, ValueError) as exc:
            raise ValidationError("MCE example values must be canonically serializable") from exc

    @classmethod
    def from_value(cls, value: Any, *, split: str) -> MCEExample:
        if isinstance(value, cls):
            if value.split != split:
                raise ValidationError(
                    f"MCE example {value.example_id!r} is labeled {value.split}, expected {split}"
                )
            return value
        if not isinstance(value, Mapping):
            raise ValidationError("MCE examples must be MCEExample instances or mappings")
        try:
            example_id = value["example_id"]
            input_value = value["input"]
        except KeyError as exc:
            raise ValidationError("MCE example lacks example_id or input") from exc
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("MCE example metadata must be a mapping")
        return cls(
            example_id=str(example_id),
            input=input_value,
            expected_output=value.get("expected_output"),
            split=split,
            metadata=dict(metadata),
        )

    @property
    def input_hash(self) -> str:
        return content_hash(self.input)

    @property
    def example_hash(self) -> str:
        """Hash every semantically relevant field, including labels/metadata."""

        return content_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class MCESplitAudit:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_input_hashes: tuple[str, ...]
    validation_input_hashes: tuple[str, ...]
    train_example_hashes: tuple[str, ...]
    validation_example_hashes: tuple[str, ...]

    @property
    def audit_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class MCEControls:
    iterations: int = 3
    train_batch_size: int = 50
    max_model_calls: int = 256
    max_tool_calls: int = 256
    max_wall_seconds: int = 3600
    max_artifact_files: int = 128
    max_artifact_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            self.iterations,
            self.train_batch_size,
            self.max_model_calls,
            self.max_tool_calls,
            self.max_wall_seconds,
            self.max_artifact_files,
            self.max_artifact_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValidationError("MCE iteration, budget, and artifact limits must be integers")
        if min(values) < 1:
            raise ValidationError("MCE iteration, budget, and artifact limits must be positive")


@dataclass(frozen=True, slots=True)
class MCESandboxAttestation:
    """Typed facts supplied by the external container supervisor."""

    attestation_id: str
    runtime_mode: str
    source_mode: str
    isolation_kind: str
    image_digest: str
    policy_sha256: str
    test_report_sha256: str
    network_mode: str
    dataset_mount_mode: str
    workspace_scope: str
    generated_code_execution: str
    test_status: str
    validation_oracle_mode: str
    oracle_mount_mode: str
    oracle_process_boundary: str
    usage_accounting_mode: str
    provider_guard_mode: str
    seed_mode: str
    provider_subcall_seed_mode: str
    interface_validation_mode: str
    batch_evaluation_mode: str
    upstream_run_iteration_source_sha256: str
    host_secrets_mounted: bool
    structured_egress_only: bool
    generated_agents_validation_access: bool
    generated_workspace_oracle_visible: bool
    pre_call_budget_enforced: bool
    wall_timeout_enforced: bool
    source_read_only: bool
    artifact_store_persistent: bool
    model_identity_enforced: bool
    environment_identity_verified: bool
    interface_signature_verified: bool
    semantic_patch_applied: bool
    artifact_store_fingerprint_sha256: str
    semantic_patch_manifest_sha256: str
    resource_limits: Mapping[str, int]
    required_worker_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.attestation_id.strip():
            raise ExternalDependencyError("MCE sandbox attestation lacks identity")
        if self.runtime_mode not in {"production", "fixture"}:
            raise ExternalDependencyError("MCE sandbox runtime_mode must be production or fixture")
        expected_source_mode = "git_checkout" if self.runtime_mode == "production" else "fixture"
        if self.source_mode != expected_source_mode:
            raise ExternalDependencyError("MCE sandbox source_mode does not match runtime_mode")
        if self.isolation_kind not in {"container", "microvm"}:
            raise ExternalDependencyError("MCE requires an attested container or microVM")
        if (
            not isinstance(self.image_digest, str)
            or not self.image_digest.startswith("sha256:")
            or len(self.image_digest) != 71
        ):
            raise ExternalDependencyError("MCE sandbox image_digest must be a sha256 digest")
        for name in (
            "policy_sha256",
            "test_report_sha256",
            "artifact_store_fingerprint_sha256",
            "semantic_patch_manifest_sha256",
            "upstream_run_iteration_source_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(
                    getattr(self, name),
                    label=f"MCE {name}",
                    error_type=ExternalDependencyError,
                ),
            )
        expected_text = {
            "network_mode": "none",
            "dataset_mount_mode": "read_only",
            "workspace_scope": "per_run",
            "generated_code_execution": "isolated_worker_only",
            "test_status": "passed",
            "validation_oracle_mode": "evaluator_only",
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "usage_accounting_mode": "trusted_supervisor_ipc",
            "provider_guard_mode": "provider_transport_pre_call",
            "seed_mode": MCE_SEED_MODE,
            "provider_subcall_seed_mode": MCE_PROVIDER_SUBCALL_SEED_MODE,
            "interface_validation_mode": "remote_static_no_import",
            "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
        }
        mismatched = [
            name for name, expected in expected_text.items() if getattr(self, name) != expected
        ]
        required_true = {
            "structured_egress_only": self.structured_egress_only,
            "pre_call_budget_enforced": self.pre_call_budget_enforced,
            "wall_timeout_enforced": self.wall_timeout_enforced,
            "source_read_only": self.source_read_only,
            "artifact_store_persistent": self.artifact_store_persistent,
            "model_identity_enforced": self.model_identity_enforced,
            "environment_identity_verified": self.environment_identity_verified,
            "interface_signature_verified": self.interface_signature_verified,
            "semantic_patch_applied": self.semantic_patch_applied,
        }
        mismatched.extend(name for name, enabled in required_true.items() if not enabled)
        if self.host_secrets_mounted:
            mismatched.append("host_secrets_mounted")
        if self.generated_agents_validation_access:
            mismatched.append("generated_agents_validation_access")
        if self.generated_workspace_oracle_visible:
            mismatched.append("generated_workspace_oracle_visible")
        expected_contract = {
            "oracle_mount_mode": "supervisor_only_hidden",
            "oracle_process_boundary": "trusted_supervisor_out_of_process",
            "generated_workspace_oracle_visible": False,
            "provider_guard_mode": "provider_transport_pre_call",
            "seed_mode": MCE_SEED_MODE,
            "provider_subcall_seed_mode": MCE_PROVIDER_SUBCALL_SEED_MODE,
            "validation_scoring_mode": "supervisor_only",
        }
        if dict(self.required_worker_contract) != expected_contract:
            mismatched.append("required_worker_contract")
        if self.semantic_patch_manifest_sha256 != MCE_SEMANTIC_PATCH_MANIFEST_SHA256:
            mismatched.append("semantic_patch_manifest_sha256")
        if self.upstream_run_iteration_source_sha256 != MCE_UPSTREAM_MAIN_SHA256:
            mismatched.append("upstream_run_iteration_source_sha256")
        if mismatched:
            raise ExternalDependencyError(
                "MCE sandbox attestation violates policy: " + ", ".join(sorted(mismatched))
            )
        limits = dict(self.resource_limits)
        expected_limits = {"cpu_seconds", "memory_bytes", "processes", "wall_seconds"}
        if set(limits) != expected_limits:
            raise ExternalDependencyError(
                "MCE sandbox attestation requires exact CPU/memory/process/wall limits"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in limits.values()
        ):
            raise ExternalDependencyError("MCE sandbox resource limits must be positive integers")
        object.__setattr__(self, "resource_limits", MappingProxyType(limits))
        object.__setattr__(
            self,
            "required_worker_contract",
            freeze_mapping(self.required_worker_contract),
        )

    @classmethod
    def from_worker_fingerprint(
        cls,
        fingerprint: Mapping[str, Any],
    ) -> MCESandboxAttestation:
        raw = fingerprint.get("sandbox_attestation")
        if not isinstance(raw, Mapping):
            raise ExternalDependencyError(
                "MCE worker fingerprint lacks a sandbox_attestation; host execution is forbidden"
            )
        limits = raw.get("resource_limits")
        if not isinstance(limits, Mapping):
            raise ExternalDependencyError("MCE sandbox attestation lacks resource_limits")
        required_worker_contract = raw.get("required_worker_contract")
        if not isinstance(required_worker_contract, Mapping):
            raise ExternalDependencyError("MCE sandbox attestation lacks required_worker_contract")
        boolean_fields = (
            "host_secrets_mounted",
            "structured_egress_only",
            "generated_agents_validation_access",
            "generated_workspace_oracle_visible",
            "pre_call_budget_enforced",
            "wall_timeout_enforced",
            "source_read_only",
            "artifact_store_persistent",
            "model_identity_enforced",
            "environment_identity_verified",
            "interface_signature_verified",
            "semantic_patch_applied",
        )
        try:
            values = {
                name: _strict_bool(
                    raw.get(name),
                    label=f"MCE {name}",
                    error_type=ExternalDependencyError,
                )
                for name in boolean_fields
            }
            return cls(
                attestation_id=str(raw["attestation_id"]),
                runtime_mode=str(raw["runtime_mode"]),
                source_mode=str(raw["source_mode"]),
                isolation_kind=str(raw["isolation_kind"]),
                image_digest=str(raw["image_digest"]),
                policy_sha256=str(raw["policy_sha256"]),
                test_report_sha256=str(raw["test_report_sha256"]),
                network_mode=str(raw["network_mode"]),
                dataset_mount_mode=str(raw["dataset_mount_mode"]),
                workspace_scope=str(raw["workspace_scope"]),
                generated_code_execution=str(raw["generated_code_execution"]),
                test_status=str(raw["test_status"]),
                validation_oracle_mode=str(raw["validation_oracle_mode"]),
                oracle_mount_mode=str(raw["oracle_mount_mode"]),
                oracle_process_boundary=str(raw["oracle_process_boundary"]),
                usage_accounting_mode=str(raw["usage_accounting_mode"]),
                provider_guard_mode=str(raw["provider_guard_mode"]),
                seed_mode=str(raw["seed_mode"]),
                provider_subcall_seed_mode=str(raw["provider_subcall_seed_mode"]),
                interface_validation_mode=str(raw["interface_validation_mode"]),
                batch_evaluation_mode=str(raw["batch_evaluation_mode"]),
                upstream_run_iteration_source_sha256=str(
                    raw["upstream_run_iteration_source_sha256"]
                ),
                artifact_store_fingerprint_sha256=str(raw["artifact_store_fingerprint_sha256"]),
                semantic_patch_manifest_sha256=str(raw["semantic_patch_manifest_sha256"]),
                resource_limits=dict(limits),
                required_worker_contract=dict(required_worker_contract),
                **values,
            )
        except KeyError as exc:
            raise ExternalDependencyError(f"MCE sandbox attestation lacks {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class MCEArtifactFile:
    path: str
    sha256: str
    byte_size: int
    media_type: str
    file_type: str
    generated: bool
    executable: bool

    def __post_init__(self) -> None:
        if not _allowed_artifact_path(self.path):
            raise AlgorithmOutputError("MCE artifact path is outside the frozen allowlist")
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(
                self.sha256,
                label="MCE artifact file sha256",
                error_type=AlgorithmOutputError,
            ),
        )
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 0
        ):
            raise AlgorithmOutputError("MCE artifact file byte_size must be non-negative")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise AlgorithmOutputError("MCE artifact file media_type must not be empty")
        if self.file_type != "regular":
            raise AlgorithmOutputError("MCE artifact egress only permits regular files")
        if not isinstance(self.generated, bool) or not isinstance(self.executable, bool):
            raise AlgorithmOutputError("MCE artifact generated/executable flags must be boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MCEArtifactFile:
        normalized_keys = {str(key).lower() for key in value}
        forbidden = _INLINE_CONTENT_KEYS & normalized_keys
        if forbidden:
            raise AlgorithmOutputError(
                "MCE worker attempted inline generated-code/content egress: "
                + ", ".join(sorted(forbidden))
            )
        _reject_unknown_fields(
            value,
            allowed=_ARTIFACT_FILE_FIELDS,
            label="MCE artifact file",
        )
        return cls(
            path=required_text(value, "path", label="MCE artifact file"),
            sha256=required_text(value, "sha256", label="MCE artifact file"),
            byte_size=nonnegative_integer(
                value.get("byte_size"),
                label="MCE artifact file byte_size",
            ),
            media_type=required_text(value, "media_type", label="MCE artifact file"),
            file_type=required_text(value, "file_type", label="MCE artifact file"),
            generated=_strict_bool(value.get("generated"), label="MCE artifact generated"),
            executable=_strict_bool(
                value.get("executable"),
                label="MCE artifact executable",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True, slots=True)
class MCEIteration:
    index: int
    skill_sha256: str
    artifact_manifest_sha256: str
    validation_score: float

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise AlgorithmOutputError("MCE iteration index must be a non-negative integer")
        if (
            isinstance(self.validation_score, bool)
            or not isinstance(self.validation_score, (int, float))
            or not math.isfinite(float(self.validation_score))
        ):
            raise AlgorithmOutputError("MCE validation_score must be finite")
        object.__setattr__(self, "validation_score", float(self.validation_score))
        for name in ("skill_sha256", "artifact_manifest_sha256"):
            object.__setattr__(
                self,
                name,
                _require_sha256(
                    getattr(self, name),
                    label=f"MCE {name}",
                    error_type=AlgorithmOutputError,
                ),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MCEIteration:
        return cls(
            index=nonnegative_integer(value.get("index"), label="MCE iteration index"),
            skill_sha256=required_text(value, "skill_sha256", label="MCE iteration"),
            artifact_manifest_sha256=required_text(
                value,
                "artifact_manifest_sha256",
                label="MCE iteration",
            ),
            validation_score=finite_number(
                value.get("validation_score"),
                label="MCE validation_score",
            ),
        )


@dataclass(frozen=True, slots=True)
class MCEContextArtifact:
    artifact_id: str
    artifact_handle: str
    release_id: str
    selected_iteration: int
    artifact_manifest_sha256: str
    artifact_bundle_sha256: str
    skill_sha256: str
    interface_signature_sha256: str
    initial_context_sha256: str
    split_audit_sha256: str
    environment_identity_sha256: str
    storage_fingerprint_sha256: str
    source_manifest_sha256: str
    license_manifest_sha256: str
    provenance_manifest_sha256: str
    semantic_patch_manifest_sha256: str
    runtime_binding_sha256: str
    provenance_status: str
    files: tuple[MCEArtifactFile, ...]
    iterations: tuple[MCEIteration, ...]
    sandbox_attestation: MCESandboxAttestation
    worker_fingerprint: Mapping[str, Any]
    model_fingerprints: Mapping[str, Any]
    environment_identity: Mapping[str, Any]
    controls: Mapping[str, Any]
    usage: Mapping[str, int | float]
    runtime_attestation: Mapping[str, Any]
    upstream_commit: str = MCE_UPSTREAM_COMMIT
    host_code_executed: bool = False

    def __post_init__(self) -> None:
        if not self.artifact_id.strip() or not self.release_id.strip():
            raise ValidationError("MCE artifact identity must not be empty")
        _validate_artifact_handle(self.artifact_handle, self.artifact_bundle_sha256)
        if self.upstream_commit != MCE_UPSTREAM_COMMIT:
            raise ValidationError("MCE artifact upstream revision mismatch")
        if self.host_code_executed:
            raise ValidationError("MCE artifacts may never report host code execution")
        if self.provenance_status != "generated_unreviewed":
            raise ValidationError("MCE generated artifact provenance must remain explicit")
        files = tuple(self.files)
        iterations = tuple(self.iterations)
        if not all(isinstance(item, MCEArtifactFile) for item in files):
            raise ValidationError("MCE artifact files must be typed")
        if not iterations or not all(isinstance(item, MCEIteration) for item in iterations):
            raise ValidationError("MCE artifact requires typed iteration history")
        if self.selected_iteration > 0 and not files:
            raise ValidationError("non-baseline MCE artifact must not be empty")
        for name in (
            "artifact_manifest_sha256",
            "artifact_bundle_sha256",
            "skill_sha256",
            "interface_signature_sha256",
            "initial_context_sha256",
            "split_audit_sha256",
            "environment_identity_sha256",
            "storage_fingerprint_sha256",
            "source_manifest_sha256",
            "license_manifest_sha256",
            "provenance_manifest_sha256",
            "semantic_patch_manifest_sha256",
            "runtime_binding_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), label=f"MCE {name}"),
            )
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "worker_fingerprint", freeze_mapping(self.worker_fingerprint))
        object.__setattr__(self, "model_fingerprints", freeze_mapping(self.model_fingerprints))
        object.__setattr__(
            self,
            "environment_identity",
            freeze_mapping(self.environment_identity),
        )
        object.__setattr__(self, "controls", freeze_mapping(self.controls))
        object.__setattr__(self, "usage", freeze_mapping(self.usage))
        object.__setattr__(
            self,
            "runtime_attestation",
            freeze_mapping(self.runtime_attestation),
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MCEContextArtifact:
        """Restore canonical metadata, never artifact code or Python pickle."""
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValidationError("MCE saved artifact metadata schema drifted")
        record = dict(value)
        sandbox = MCESandboxAttestation.from_worker_fingerprint(record["worker_fingerprint"])
        if to_jsonable(sandbox) != record["sandbox_attestation"]:
            raise ValidationError("MCE saved sandbox attestation differs from its worker identity")
        record["sandbox_attestation"] = sandbox
        record["files"] = tuple(MCEArtifactFile.from_mapping(row) for row in mapping_sequence(
            record["files"], label="MCE saved files"))
        record["iterations"] = tuple(MCEIteration.from_mapping(row) for row in mapping_sequence(
            record["iterations"], label="MCE saved iterations"))
        return cls(**record)


@dataclass(slots=True)
class MCEOfflineRecipe:
    """Run MCE through a verified isolated worker and return an opaque handle."""

    client: WorkerClient | None = None
    interface_signature_sha256: str = ""
    initial_context_sha256: str = MCE_EMPTY_INITIAL_CONTEXT_SHA256
    interface_signature: Mapping[str, Any] = field(default_factory=dict)
    environment_identity: Mapping[str, Any] = field(default_factory=dict)
    license_manifest_allowlist: tuple[str, ...] = ()
    provenance_manifest_allowlist: tuple[str, ...] = ()
    allow_fixture_runtime: bool = False
    controls: MCEControls = MCEControls()
    method_id: str = "recipe.mce"
    release_id: str = "recipe.mce.sandboxed_offline"

    def __post_init__(self) -> None:
        self.interface_signature_sha256 = _require_sha256(
            self.interface_signature_sha256,
            label="MCE interface_signature_sha256",
        )
        self.initial_context_sha256 = _require_sha256(
            self.initial_context_sha256,
            label="MCE initial_context_sha256",
        )
        if not isinstance(self.allow_fixture_runtime, bool):
            raise ValidationError("MCE allow_fixture_runtime must be boolean")
        try:
            self.interface_signature = freeze_mapping(self.interface_signature)
            self.environment_identity = freeze_mapping(self.environment_identity)
        except TypeError as exc:
            raise ValidationError("MCE interface/environment identity is not canonical") from exc
        if self.interface_signature and (
            content_hash(self.interface_signature) != self.interface_signature_sha256
        ):
            raise ValidationError("MCE interface signature hash does not match its definition")
        self.license_manifest_allowlist = _digest_allowlist(
            self.license_manifest_allowlist,
            label="MCE license manifest allowlist",
        )
        self.provenance_manifest_allowlist = _digest_allowlist(
            self.provenance_manifest_allowlist,
            label="MCE provenance manifest allowlist",
        )

    def fit(
        self,
        train_examples: tuple[Any, ...],
        *,
        validation_examples: tuple[Any, ...] = (),
        context: InterventionContext,
    ) -> MCEContextArtifact:
        train = tuple(MCEExample.from_value(item, split="train") for item in train_examples)
        validation = tuple(
            MCEExample.from_value(item, split="validation") for item in validation_examples
        )
        if not train or not validation:
            raise ValidationError("MCE offline recipe requires non-empty train and validation")
        split_audit = validate_mce_split_isolation(train, validation)
        worker = resolve_worker(self.client, context, label="MCE")
        identity = worker_fingerprint(worker, label="MCE")
        sandbox = MCESandboxAttestation.from_worker_fingerprint(identity)
        if sandbox.runtime_mode == "fixture" and not self.allow_fixture_runtime:
            raise ExternalDependencyError("MCE fixture workers are disabled by default")
        if sandbox.resource_limits["wall_seconds"] > self.controls.max_wall_seconds:
            raise ExternalDependencyError(
                "MCE sandbox wall limit exceeds the configured recipe wall budget"
            )

        interface_signature = self.interface_signature
        environment_identity = self.environment_identity
        if sandbox.runtime_mode == "production":
            if self.initial_context_sha256 != MCE_EMPTY_INITIAL_CONTEXT_SHA256:
                raise ValidationError("production MCE requires the canonical empty initial context")
            _validate_interface_signature(interface_signature)
            _validate_environment_identity(environment_identity)
        model_fingerprints = _model_fingerprints(context)
        controls = {**asdict(self.controls), "seed": context.seed}
        bound = _validate_worker_identity(
            identity,
            sandbox=sandbox,
            interface_signature_sha256=self.interface_signature_sha256,
            initial_context_sha256=self.initial_context_sha256,
            environment_identity_sha256=content_hash(environment_identity),
            model_fingerprints_sha256=content_hash(model_fingerprints),
            license_allowlist=self.license_manifest_allowlist,
            provenance_allowlist=self.provenance_manifest_allowlist,
        )
        runtime_binding_sha256 = _runtime_binding_sha256(
            controls=controls,
            split_audit_sha256=split_audit.audit_hash,
            interface_signature_sha256=self.interface_signature_sha256,
            initial_context_sha256=self.initial_context_sha256,
            environment_identity_sha256=content_hash(environment_identity),
            model_fingerprints_sha256=content_hash(model_fingerprints),
            source_tree=bound["source_tree"],
            source_manifest_sha256=bound["source_manifest_sha256"],
            semantic_patch_manifest_sha256=bound["semantic_patch_manifest_sha256"],
        )
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=MCE_UPSTREAM_COMMIT,
            state_hash=split_audit.audit_hash,
            payload={
                "train_examples": tuple(item.as_dict() for item in train),
                "validation_examples": tuple(item.as_dict() for item in validation),
                "split_audit_sha256": split_audit.audit_hash,
                "interface_signature": interface_signature,
                "interface_signature_sha256": self.interface_signature_sha256,
                "initial_context_identity": _EMPTY_INITIAL_CONTEXT,
                "initial_context_sha256": self.initial_context_sha256,
                "environment_identity": environment_identity,
                "environment_identity_sha256": content_hash(environment_identity),
                "sandbox_attestation_id": sandbox.attestation_id,
                "model_fingerprints": model_fingerprints,
                "model_fingerprints_sha256": content_hash(model_fingerprints),
                "controls": controls,
                "pre_call_budget": {
                    "model_calls": self.controls.max_model_calls,
                    "tool_calls": self.controls.max_tool_calls,
                    "wall_seconds": self.controls.max_wall_seconds,
                },
                "runtime_binding_sha256": runtime_binding_sha256,
                "worker_fingerprint_sha256": content_hash(identity),
                "source_tree": bound["source_tree"],
                "source_manifest_sha256": bound["source_manifest_sha256"],
                "license_manifest_sha256": bound["license_manifest_sha256"],
                "provenance_manifest_sha256": bound["provenance_manifest_sha256"],
                "semantic_patch_manifest": MCE_SEMANTIC_PATCH_MANIFEST,
                "semantic_patch_manifest_sha256": bound["semantic_patch_manifest_sha256"],
                "storage_fingerprint_sha256": bound["storage_fingerprint_sha256"],
                "host_execution_allowed": False,
                "artifact_egress": "manifest_and_opaque_handle_only",
            },
            dependency_fingerprints={"worker": identity},
            minimum_reservation={"model_calls": 1},
            protocol_version=MCE_PROTOCOL,
        )
        response = _call_mce_worker(worker, request)
        payload = response.payload
        _reject_unknown_fields(
            payload,
            allowed=_RESPONSE_PAYLOAD_FIELDS,
            label="MCE response payload",
        )
        _validate_response_identity(
            payload,
            sandbox=sandbox,
            split_audit_sha256=split_audit.audit_hash,
            interface_signature_sha256=self.interface_signature_sha256,
            initial_context_sha256=self.initial_context_sha256,
            environment_identity_sha256=content_hash(environment_identity),
            model_fingerprints_sha256=content_hash(model_fingerprints),
            runtime_binding_sha256=runtime_binding_sha256,
            bound=bound,
        )

        files = tuple(
            MCEArtifactFile.from_mapping(row)
            for row in mapping_sequence(payload.get("files"), label="MCE artifact files")
        )
        if len(files) > self.controls.max_artifact_files:
            raise AlgorithmOutputError("MCE artifact exceeded max_artifact_files")
        if sum(item.byte_size for item in files) > self.controls.max_artifact_bytes:
            raise AlgorithmOutputError("MCE artifact exceeded max_artifact_bytes")
        paths = tuple(item.path for item in files)
        if len(paths) != len(set(paths)):
            raise AlgorithmOutputError("MCE artifact manifest contains duplicate paths")
        file_rows = tuple(item.as_dict() for item in files)
        manifest_sha256 = content_hash(file_rows)
        if _output_sha256(payload, "artifact_manifest_sha256") != manifest_sha256:
            raise AlgorithmOutputError("MCE artifact manifest hash mismatch")

        iterations = tuple(
            MCEIteration.from_mapping(row)
            for row in mapping_sequence(payload.get("iterations"), label="MCE iterations")
        )
        expected_indices = tuple(range(self.controls.iterations + 1))
        if tuple(item.index for item in iterations) != expected_indices:
            raise AlgorithmOutputError(
                "MCE iterations must contain baseline zero through the configured iteration"
            )
        if iterations[0].skill_sha256 != _EMPTY_SHA256:
            raise AlgorithmOutputError("MCE baseline iteration must have no evolved SKILL")
        if any(item.skill_sha256 == _EMPTY_SHA256 for item in iterations[1:]):
            raise AlgorithmOutputError("every non-baseline MCE iteration must contain SKILL.md")
        selected_iteration = nonnegative_integer(
            payload.get("selected_iteration"),
            label="MCE selected_iteration",
        )
        if selected_iteration > self.controls.iterations:
            raise AlgorithmOutputError("MCE selected_iteration is out of range")
        expected_selected_iteration = max(
            range(len(iterations)),
            key=lambda index: iterations[index].validation_score,
        )
        if selected_iteration != expected_selected_iteration:
            raise AlgorithmOutputError(
                "MCE selected_iteration must maximize the validation primary metric"
            )
        if selected_iteration > 0 and not files:
            raise AlgorithmOutputError("non-baseline MCE selection must contain artifact files")
        selected = iterations[selected_iteration]
        skill_sha256 = _output_sha256(payload, "skill_sha256")
        if selected.artifact_manifest_sha256 != manifest_sha256:
            raise AlgorithmOutputError("MCE selected iteration does not identify final manifest")
        if selected.skill_sha256 != skill_sha256:
            raise AlgorithmOutputError("MCE selected iteration does not identify final skill")
        if selected_iteration > 0:
            skill_files = [
                item for item in files if item.path == ".claude/skills/learning-context/SKILL.md"
            ]
            if len(skill_files) != 1 or skill_files[0].sha256 != skill_sha256:
                raise AlgorithmOutputError(
                    "non-baseline MCE artifact must bind its evolved SKILL.md"
                )

        bundle = _artifact_bundle(
            files=file_rows,
            selected_iteration=selected_iteration,
            artifact_manifest_sha256=manifest_sha256,
            skill_sha256=skill_sha256,
            split_audit_sha256=split_audit.audit_hash,
            interface_signature_sha256=self.interface_signature_sha256,
            initial_context_sha256=self.initial_context_sha256,
            environment_identity_sha256=content_hash(environment_identity),
            model_fingerprints_sha256=content_hash(model_fingerprints),
            controls_sha256=content_hash(controls),
            runtime_binding_sha256=runtime_binding_sha256,
            bound=bound,
        )
        bundle_sha256 = content_hash(bundle)
        if _output_sha256(payload, "artifact_bundle_sha256") != bundle_sha256:
            raise AlgorithmOutputError("MCE artifact bundle hash mismatch")
        artifact_handle = required_text(payload, "artifact_handle", label="MCE response")
        _validate_artifact_handle(artifact_handle, bundle_sha256)

        provenance_status = required_text(payload, "provenance_status", label="MCE response")
        if provenance_status != "generated_unreviewed":
            raise AlgorithmOutputError("MCE generated provenance status was overstated")
        runtime_attestation = _runtime_attestation(
            payload,
            sandbox=sandbox,
            runtime_binding_sha256=runtime_binding_sha256,
            bound=bound,
        )
        usage = external_usage_counters(response.usage)
        if response.usage.model_calls > self.controls.max_model_calls:
            raise AlgorithmOutputError("MCE worker exceeded max_model_calls")
        if response.usage.tool_calls > self.controls.max_tool_calls:
            raise AlgorithmOutputError("MCE worker exceeded max_tool_calls")

        artifact_base = {
            **bundle,
            "artifact_handle": artifact_handle,
            "release_id": self.release_id,
            "sandbox_attestation_id": sandbox.attestation_id,
        }
        return MCEContextArtifact(
            artifact_id=content_hash(artifact_base),
            artifact_handle=artifact_handle,
            release_id=self.release_id,
            selected_iteration=selected_iteration,
            artifact_manifest_sha256=manifest_sha256,
            artifact_bundle_sha256=bundle_sha256,
            skill_sha256=skill_sha256,
            interface_signature_sha256=self.interface_signature_sha256,
            initial_context_sha256=self.initial_context_sha256,
            split_audit_sha256=split_audit.audit_hash,
            environment_identity_sha256=content_hash(environment_identity),
            storage_fingerprint_sha256=bound["storage_fingerprint_sha256"],
            source_manifest_sha256=bound["source_manifest_sha256"],
            license_manifest_sha256=bound["license_manifest_sha256"],
            provenance_manifest_sha256=bound["provenance_manifest_sha256"],
            semantic_patch_manifest_sha256=bound["semantic_patch_manifest_sha256"],
            runtime_binding_sha256=runtime_binding_sha256,
            provenance_status=provenance_status,
            files=files,
            iterations=iterations,
            sandbox_attestation=sandbox,
            worker_fingerprint=identity,
            model_fingerprints=model_fingerprints,
            environment_identity=environment_identity,
            controls=controls,
            usage=usage,
            runtime_attestation=runtime_attestation,
            host_code_executed=False,
        )


def validate_mce_split_isolation(
    train: Sequence[MCEExample],
    validation: Sequence[MCEExample],
) -> MCESplitAudit:
    groups = {"train": tuple(train), "validation": tuple(validation)}
    for split, examples in groups.items():
        if any(example.split != split for example in examples):
            raise ValidationError(f"MCE {split} collection contains a mislabeled example")
        ids = tuple(example.example_id for example in examples)
        hashes = tuple(example.input_hash for example in examples)
        full_hashes = tuple(example.example_hash for example in examples)
        if len(ids) != len(set(ids)):
            raise ValidationError(f"MCE {split} split contains duplicate IDs")
        if len(hashes) != len(set(hashes)):
            raise ValidationError(f"MCE {split} split contains duplicate inputs")
        if len(full_hashes) != len(set(full_hashes)):
            raise ValidationError(f"MCE {split} split contains duplicate examples")
    train_ids = tuple(example.example_id for example in groups["train"])
    validation_ids = tuple(example.example_id for example in groups["validation"])
    train_hashes = tuple(example.input_hash for example in groups["train"])
    validation_hashes = tuple(example.input_hash for example in groups["validation"])
    if set(train_ids) & set(validation_ids):
        raise ValidationError("MCE train/validation example IDs overlap")
    if set(train_hashes) & set(validation_hashes):
        raise ValidationError("MCE train/validation inputs overlap")
    return MCESplitAudit(
        train_ids=train_ids,
        validation_ids=validation_ids,
        train_input_hashes=train_hashes,
        validation_input_hashes=validation_hashes,
        train_example_hashes=tuple(example.example_hash for example in groups["train"]),
        validation_example_hashes=tuple(example.example_hash for example in groups["validation"]),
    )


def _model_fingerprints(context: InterventionContext) -> Mapping[str, Any]:
    configured = context.run_config.get("mce_model_fingerprints")
    if not isinstance(configured, Mapping):
        raise ExternalDependencyError("MCE requires explicit target/meta/base model fingerprints")
    result: dict[str, Any] = {}
    for role in ("target", "meta", "base"):
        value = configured.get(role)
        if not isinstance(value, Mapping) or not value:
            raise ExternalDependencyError(f"MCE is missing the {role} model fingerprint")
        if set(value) != _MODEL_FINGERPRINT_FIELDS:
            raise ExternalDependencyError(
                f"MCE {role} model fingerprint must contain exactly "
                "provider, model, revision, and configuration_sha256"
            )
        for key in ("provider", "model", "revision", "configuration_sha256"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ExternalDependencyError(f"MCE {role} model fingerprint lacks non-empty {key}")
        _require_sha256(
            value["configuration_sha256"],
            label=f"MCE {role} configuration_sha256",
            error_type=ExternalDependencyError,
        )
        result[role] = dict(value)
    try:
        return freeze_mapping(result)
    except TypeError as exc:
        raise ValidationError("MCE model fingerprints are not canonical") from exc


def _validate_interface_signature(value: Mapping[str, Any]) -> None:
    if not value:
        raise ValidationError("production MCE requires the actual interface signature")
    if not isinstance(value.get("environment"), str) or not value["environment"].strip():
        raise ValidationError("MCE interface signature requires environment")
    interfaces = value.get("interfaces")
    if isinstance(interfaces, (str, bytes)) or not isinstance(interfaces, Sequence):
        raise ValidationError("MCE interface signature requires an interfaces sequence")


def _validate_environment_identity(value: Mapping[str, Any]) -> None:
    if not value:
        raise ValidationError("production MCE requires an environment identity")
    for key in ("name", "revision", "source_sha256"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValidationError(f"MCE environment identity lacks {key}")
    _require_sha256(value["source_sha256"], label="MCE environment source_sha256")


def _validate_worker_identity(
    identity: Mapping[str, Any],
    *,
    sandbox: MCESandboxAttestation,
    interface_signature_sha256: str,
    initial_context_sha256: str,
    environment_identity_sha256: str,
    model_fingerprints_sha256: str,
    license_allowlist: tuple[str, ...],
    provenance_allowlist: tuple[str, ...],
) -> Mapping[str, str]:
    expected = {
        "protocol_version": MCE_PROTOCOL,
        "upstream_commit": MCE_UPSTREAM_COMMIT,
        "interface_signature_sha256": interface_signature_sha256,
        "initial_context_sha256": initial_context_sha256,
        "environment_identity_sha256": environment_identity_sha256,
        "model_fingerprints_sha256": model_fingerprints_sha256,
        "fixture_mode": sandbox.runtime_mode == "fixture",
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ExternalDependencyError(f"MCE worker fingerprint mismatch: {key}")
    if identity.get("image_digest") != sandbox.image_digest:
        raise ExternalDependencyError("MCE worker/sandbox image digest mismatch")
    source_tree = _require_sha1(identity.get("source_tree"), label="MCE source_tree")
    source_manifest = _identity_manifest(identity, "source_manifest")
    license_manifest = _identity_manifest(identity, "license_manifest")
    provenance_manifest = _identity_manifest(identity, "provenance_manifest")
    semantic_patch_manifest = _identity_manifest(
        identity,
        "semantic_patch_manifest",
    )
    if semantic_patch_manifest != MCE_SEMANTIC_PATCH_MANIFEST:
        raise ExternalDependencyError(
            "MCE worker semantic patch manifest does not match the pinned host contract"
        )
    result = {
        "source_tree": source_tree,
        "source_manifest_sha256": _verified_manifest_sha256(
            identity,
            "source_manifest_sha256",
            source_manifest,
        ),
        "license_manifest_sha256": _verified_manifest_sha256(
            identity,
            "license_manifest_sha256",
            license_manifest,
        ),
        "provenance_manifest_sha256": _verified_manifest_sha256(
            identity,
            "provenance_manifest_sha256",
            provenance_manifest,
        ),
        "semantic_patch_manifest_sha256": _verified_manifest_sha256(
            identity,
            "semantic_patch_manifest_sha256",
            semantic_patch_manifest,
        ),
        "storage_fingerprint_sha256": _identity_sha256(
            identity,
            "storage_fingerprint_sha256",
        ),
    }
    store = identity.get("artifact_store")
    if not isinstance(store, Mapping):
        raise ExternalDependencyError("MCE worker fingerprint lacks artifact_store")
    if (
        store.get("mode") != "content_addressed_v1"
        or store.get("persistent") is not True
        or store.get("worker_resolve_supported") is not True
        or content_hash(store) != result["storage_fingerprint_sha256"]
    ):
        raise ExternalDependencyError("MCE artifact-store fingerprint is invalid")
    if sandbox.artifact_store_fingerprint_sha256 != result["storage_fingerprint_sha256"]:
        raise ExternalDependencyError("MCE sandbox/store fingerprint mismatch")
    if sandbox.semantic_patch_manifest_sha256 != result["semantic_patch_manifest_sha256"]:
        raise ExternalDependencyError("MCE sandbox/semantic-patch fingerprint mismatch")
    if result["license_manifest_sha256"] not in license_allowlist:
        raise ExternalDependencyError("MCE license manifest is not in the host allowlist")
    if result["provenance_manifest_sha256"] not in provenance_allowlist:
        raise ExternalDependencyError("MCE provenance manifest is not in the host allowlist")
    return freeze_mapping(result)  # type: ignore[return-value]


def _validate_response_identity(
    payload: Mapping[str, Any],
    *,
    sandbox: MCESandboxAttestation,
    split_audit_sha256: str,
    interface_signature_sha256: str,
    initial_context_sha256: str,
    environment_identity_sha256: str,
    model_fingerprints_sha256: str,
    runtime_binding_sha256: str,
    bound: Mapping[str, str],
) -> None:
    if payload.get("host_code_executed") is not False:
        raise AlgorithmOutputError("MCE worker did not attest zero host code execution")
    expected = {
        "sandbox_attestation_id": sandbox.attestation_id,
        "artifact_egress": "manifest_and_opaque_handle_only",
        "split_audit_sha256": split_audit_sha256,
        "interface_signature_sha256": interface_signature_sha256,
        "initial_context_sha256": initial_context_sha256,
        "environment_identity_sha256": environment_identity_sha256,
        "model_fingerprints_sha256": model_fingerprints_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        **bound,
    }
    for key, value in expected.items():
        if required_text(payload, key, label="MCE response") != value:
            raise AlgorithmOutputError(f"MCE {key} mismatch")


def _runtime_binding_sha256(
    *,
    controls: Mapping[str, Any],
    split_audit_sha256: str,
    interface_signature_sha256: str,
    initial_context_sha256: str,
    environment_identity_sha256: str,
    model_fingerprints_sha256: str,
    source_tree: str,
    source_manifest_sha256: str,
    semantic_patch_manifest_sha256: str,
) -> str:
    return content_hash(
        {
            "schema": "agentinputlab.mce.runtime-binding.v2",
            "source_commit": MCE_UPSTREAM_COMMIT,
            "source_tree": source_tree,
            "source_manifest_sha256": source_manifest_sha256,
            "semantic_patch_manifest_sha256": semantic_patch_manifest_sha256,
            "split_audit_sha256": split_audit_sha256,
            "interface_signature_sha256": interface_signature_sha256,
            "initial_context_sha256": initial_context_sha256,
            "environment_identity_sha256": environment_identity_sha256,
            "model_fingerprints_sha256": model_fingerprints_sha256,
            "controls_sha256": content_hash(controls),
        }
    )


def _artifact_bundle(
    *,
    files: Sequence[Mapping[str, Any]],
    selected_iteration: int,
    artifact_manifest_sha256: str,
    skill_sha256: str,
    split_audit_sha256: str,
    interface_signature_sha256: str,
    initial_context_sha256: str,
    environment_identity_sha256: str,
    model_fingerprints_sha256: str,
    controls_sha256: str,
    runtime_binding_sha256: str,
    bound: Mapping[str, str],
) -> Mapping[str, Any]:
    return {
        "schema": "agentinputlab.mce.bundle.v2",
        "files": tuple(files),
        "selected_iteration": selected_iteration,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "skill_sha256": skill_sha256,
        "source_commit": MCE_UPSTREAM_COMMIT,
        "source_tree": bound["source_tree"],
        "source_manifest_sha256": bound["source_manifest_sha256"],
        "license_manifest_sha256": bound["license_manifest_sha256"],
        "provenance_manifest_sha256": bound["provenance_manifest_sha256"],
        "semantic_patch_manifest_sha256": bound["semantic_patch_manifest_sha256"],
        "storage_fingerprint_sha256": bound["storage_fingerprint_sha256"],
        "split_audit_sha256": split_audit_sha256,
        "interface_signature_sha256": interface_signature_sha256,
        "initial_context_sha256": initial_context_sha256,
        "environment_identity_sha256": environment_identity_sha256,
        "model_fingerprints_sha256": model_fingerprints_sha256,
        "controls_sha256": controls_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
    }


def _runtime_attestation(
    payload: Mapping[str, Any],
    *,
    sandbox: MCESandboxAttestation,
    runtime_binding_sha256: str,
    bound: Mapping[str, str],
) -> Mapping[str, Any]:
    raw = payload.get("runtime_attestation")
    if not isinstance(raw, Mapping):
        raise AlgorithmOutputError("MCE response requires runtime_attestation")
    expected = {
        "source_commit": MCE_UPSTREAM_COMMIT,
        "source_tree": bound["source_tree"],
        "source_manifest_sha256": bound["source_manifest_sha256"],
        "semantic_patch_manifest_sha256": bound["semantic_patch_manifest_sha256"],
        "runtime_binding_sha256": runtime_binding_sha256,
        "semantic_patch_applied": True,
        "upstream_run_iteration_source_sha256": MCE_UPSTREAM_MAIN_SHA256,
        "official_run_iteration_called": True,
        "baseline_iteration_executed": True,
        "meta_agent_enabled": True,
        "base_agent_enabled": True,
        "validation_selection": "max_primary_metric_first_tie",
        "validation_oracle_mode": "evaluator_only",
        "oracle_mount_mode": "supervisor_only_hidden",
        "oracle_process_boundary": "trusted_supervisor_out_of_process",
        "generated_agents_validation_access": False,
        "generated_workspace_oracle_visible": False,
        "usage_accounting_mode": "trusted_supervisor_ipc",
        "provider_guard_mode": "provider_transport_pre_call",
        "pre_call_budget_enforced": True,
        "wall_timeout_enforced": True,
        "seed_mode": MCE_SEED_MODE,
        "provider_subcall_seed_mode": MCE_PROVIDER_SUBCALL_SEED_MODE,
        "generated_code_execution": "isolated_worker_only",
        "interface_validation_mode": "remote_static_no_import",
        "batch_evaluation_mode": "trusted_supervisor_no_generated_import",
        "artifact_egress": "manifest_and_opaque_handle_only",
        "artifact_store_persistent": True,
        "host_code_executed": False,
        "test_split_used": False,
        "runtime_mode": sandbox.runtime_mode,
    }
    _reject_unknown_fields(
        raw,
        allowed=frozenset(expected),
        label="MCE runtime_attestation",
    )
    for key, value in expected.items():
        if raw.get(key) != value:
            raise AlgorithmOutputError(f"MCE runtime attestation mismatch: {key}")
    try:
        return freeze_mapping(raw)
    except TypeError as exc:
        raise AlgorithmOutputError("MCE runtime attestation is not canonical") from exc


def _validate_artifact_handle(value: str, expected_bundle_sha256: str) -> None:
    expected = f"{_HANDLE_PREFIX}{expected_bundle_sha256}"
    if value != expected:
        raise AlgorithmOutputError("MCE artifact handle does not bind the bundle hash")


def _allowed_artifact_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        return False
    return value in _ALLOWED_ARTIFACT_FILES or value.startswith(_ALLOWED_ARTIFACT_PREFIXES)


def _identity_sha256(identity: Mapping[str, Any], key: str) -> str:
    value = identity.get(key)
    return _require_sha256(
        value,
        label=f"MCE worker {key}",
        error_type=ExternalDependencyError,
    )


def _identity_manifest(identity: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = identity.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ExternalDependencyError(f"MCE worker fingerprint lacks {key}")
    try:
        return freeze_mapping(value)
    except TypeError as exc:
        raise ExternalDependencyError(f"MCE worker {key} is not canonical") from exc


def _verified_manifest_sha256(
    identity: Mapping[str, Any],
    key: str,
    manifest: Mapping[str, Any],
) -> str:
    claimed = _identity_sha256(identity, key)
    if content_hash(manifest) != claimed:
        raise ExternalDependencyError(f"MCE worker {key} does not match its manifest body")
    return claimed


def _digest_allowlist(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{label} must be a sequence")
    result = tuple(_require_sha256(value, label=label) for value in values)
    if len(result) != len(set(result)):
        raise ValidationError(f"{label} contains duplicates")
    return result


def _strict_bool(
    value: Any,
    *,
    label: str,
    error_type: type[Exception] = AlgorithmOutputError,
) -> bool:
    if not isinstance(value, bool):
        raise error_type(f"{label} must be boolean")
    return value


def _output_sha256(payload: Mapping[str, Any], key: str) -> str:
    value = required_text(payload, key, label="MCE response")
    return _require_sha256(
        value,
        label=f"MCE {key}",
        error_type=AlgorithmOutputError,
    )


def _call_mce_worker(
    worker: WorkerClient,
    request: MethodWorkerRequest,
) -> MethodWorkerResponse:
    """Call MCE with a closed response-envelope schema.

    The shared method-worker transport intentionally accepts extension fields.
    MCE cannot do that because its response is the only egress from a
    generated-code sandbox.  Rejecting unknown envelope fields before parsing
    prevents an otherwise ignored field from becoming an unreviewed covert
    channel.
    """

    controlled = getattr(worker, "call_controlled", None)
    raw = controlled(request) if callable(controlled) else worker.call(request.as_dict())
    if not isinstance(raw, Mapping):
        raise AlgorithmOutputError("external worker returned a non-mapping response")
    allowed = (
        _RESPONSE_ENVELOPE_FIELDS
        if str(raw.get("status", "")) == "ok"
        else _ERROR_RESPONSE_ENVELOPE_FIELDS
    )
    _reject_unknown_fields(
        raw,
        allowed=allowed,
        label="MCE worker response envelope",
    )
    return MethodWorkerResponse.from_mapping(raw, request=request)


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = tuple(
        sorted(
            (str(key) for key in value if not isinstance(key, str) or key not in allowed),
        )
    )
    if unknown:
        raise AlgorithmOutputError(f"{label} contains undeclared field(s): " + ", ".join(unknown))


__all__ = [
    "MCE_SEMANTIC_PATCH_MANIFEST",
    "MCE_SEMANTIC_PATCH_MANIFEST_SHA256",
    "MCE_UPSTREAM_COMMIT",
    "MCE_UPSTREAM_MAIN_SHA256",
    "MCEArtifactFile",
    "MCEContextArtifact",
    "MCEControls",
    "MCEExample",
    "MCEIteration",
    "MCEOfflineRecipe",
    "MCESandboxAttestation",
    "MCESplitAudit",
    "validate_mce_split_isolation",
]
