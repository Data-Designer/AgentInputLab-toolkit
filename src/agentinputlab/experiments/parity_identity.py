"""Fail-closed implementation identities for upstream-vs-library parity runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

from .bindings import FrozenBindingReceipt
from .schema import ExperimentManifest, ExperimentStage, require_sha256

if TYPE_CHECKING:
    from .executor import ExperimentCellResult
    from .plan import ExperimentCell

_UPSTREAM_ARM = "upstream_reference"
_CANDIDATE_ARM = "agentinputlab_candidate"
_IMPLEMENTATION_KINDS = {
    _UPSTREAM_ARM: "upstream_reference",
    _CANDIDATE_ARM: "agentinputlab_candidate",
}
_GIT_OBJECT_ID_LENGTHS = frozenset({40, 64})
HOST_EXECUTION_RECEIPT_PROTOCOL = "agentinputlab.host-process-execution-receipt.v1"
_HOST_RECEIPT_FIELDS = {
    "schema_version",
    "protocol",
    "arm_id",
    "manifest_hash",
    "cell_id",
    "request_hash",
    "config_sha256",
    "checkout_environment",
    "checkout_binding_hash",
    "checkout",
    "adapter_checkout_environment",
    "adapter_checkout_binding_hash",
    "adapter_checkout",
    "command",
    "process",
    "receipt_hash",
}
_HOST_CHECKOUT_FIELDS = {
    "resolved_path_sha256",
    "git_head",
    "git_tree",
    "git_clean_before",
    "git_clean_after",
}
_HOST_PROCESS_FIELDS = {
    "resource_receipt_sha256",
    "stdout_sha256",
    "stderr_sha256",
}


@dataclass(frozen=True, slots=True)
class ParityImplementationIdentity:
    """Execution identity for exactly one parity arm."""

    arm_id: str
    implementation_kind: str
    implementation_id: str
    runtime_entrypoint: str
    source_revision: str
    source_tree: str
    release_id: str
    checkout_binding_hash: str | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported parity implementation identity schema")
        expected_kind = _IMPLEMENTATION_KINDS.get(self.arm_id)
        if expected_kind is None or self.implementation_kind != expected_kind:
            raise ValidationError("parity arm and implementation kind differ")
        for label in (
            "implementation_id",
            "runtime_entrypoint",
            "source_revision",
            "source_tree",
            "release_id",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"parity implementation {label} must be non-empty")
        if len(self.source_tree) not in _GIT_OBJECT_ID_LENGTHS or any(
            character not in "0123456789abcdef" for character in self.source_tree
        ):
            raise ValidationError(
                "parity implementation source_tree must be a full lowercase object identity"
            )
        if self.checkout_binding_hash is not None:
            require_sha256(
                self.checkout_binding_hash,
                label="parity checkout_binding_hash",
            )

    @property
    def identity_hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["identity_hash"] = self.identity_hash
        return payload


@dataclass(frozen=True, slots=True)
class ParityImplementationContract:
    """Exact two-implementation comparison contract bound to one manifest."""

    method_id: str
    manifest_hash: str
    binding_receipt_hash: str
    implementations: tuple[ParityImplementationIdentity, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported parity implementation contract schema")
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValidationError("parity contract method_id must be non-empty")
        require_sha256(self.manifest_hash, label="parity contract manifest_hash")
        require_sha256(
            self.binding_receipt_hash,
            label="parity contract binding_receipt_hash",
        )
        implementations = tuple(self.implementations)
        if not all(isinstance(item, ParityImplementationIdentity) for item in implementations):
            raise ValidationError("parity contract implementation identities must be typed")
        by_arm = {item.arm_id: item for item in implementations}
        if set(by_arm) != set(_IMPLEMENTATION_KINDS) or len(by_arm) != len(implementations):
            raise ValidationError("parity contract requires exact upstream and candidate arms")
        upstream = by_arm[_UPSTREAM_ARM]
        candidate = by_arm[_CANDIDATE_ARM]
        if upstream.implementation_id == candidate.implementation_id:
            raise ValidationError("parity implementations require distinct implementation_id")
        if upstream.runtime_entrypoint == candidate.runtime_entrypoint:
            raise ValidationError("parity implementations require distinct runtime entrypoints")
        if (
            upstream.checkout_binding_hash is None
            or candidate.checkout_binding_hash is None
            or upstream.checkout_binding_hash == candidate.checkout_binding_hash
        ):
            raise ValidationError("parity implementations require distinct host-attested checkouts")
        if upstream.identity_hash == candidate.identity_hash:
            raise ValidationError("parity implementations resolved to the same identity")
        canonical = tuple(sorted(implementations, key=lambda item: item.arm_id))
        object.__setattr__(self, "implementations", canonical)

    @property
    def contract_hash(self) -> str:
        return content_hash(
            {
                "schema_version": self.schema_version,
                "method_id": self.method_id,
                "manifest_hash": self.manifest_hash,
                "binding_receipt_hash": self.binding_receipt_hash,
                "implementations": [
                    {
                        "schema_version": item.schema_version,
                        "arm_id": item.arm_id,
                        "implementation_kind": item.implementation_kind,
                        "implementation_id": item.implementation_id,
                        "runtime_entrypoint": item.runtime_entrypoint,
                        "source_revision": item.source_revision,
                        "source_tree": item.source_tree,
                        "release_id": item.release_id,
                        "checkout_binding_hash": item.checkout_binding_hash,
                    }
                    for item in self.implementations
                ],
            }
        )

    def implementation(self, arm_id: str) -> ParityImplementationIdentity:
        for item in self.implementations:
            if item.arm_id == arm_id:
                return item
        raise ArtifactError(f"parity contract has no implementation for arm {arm_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "manifest_hash": self.manifest_hash,
            "binding_receipt_hash": self.binding_receipt_hash,
            "implementations": {item.arm_id: item.to_dict() for item in self.implementations},
            "contract_hash": self.contract_hash,
        }


def build_parity_implementation_contract(
    manifest: ExperimentManifest,
    binding_receipt: FrozenBindingReceipt,
    operator_fingerprint: Mapping[str, Any],
) -> ParityImplementationContract:
    """Validate driver declarations and bind them to host-attested assets."""

    if manifest.stage is not ExperimentStage.PARITY:
        raise ValidationError("parity implementation contract requires a parity manifest")
    if binding_receipt.manifest_hash != manifest.manifest_hash:
        raise ValidationError("parity binding receipt belongs to another manifest")
    if {arm.arm_id for arm in manifest.arms} != set(_IMPLEMENTATION_KINDS):
        raise ValidationError("parity manifest requires exact upstream and candidate arms")

    method_id = _metadata_text(manifest, "method_id")
    candidate_release = _metadata_text(manifest, "candidate_release_id")
    upstream_revision = _metadata_text(manifest, "upstream_commit")
    upstream_checkout = binding_receipt.binding(
        "AILAB_UPSTREAM_CHECKOUT",
        kind="directory_env",
    )
    candidate_checkout = binding_receipt.binding(
        "AILAB_CANDIDATE_CHECKOUT",
        kind="directory_env",
    )
    git_head = upstream_checkout.identity.get("git_head")
    git_tree = upstream_checkout.identity.get("git_tree")
    if (
        upstream_checkout.identity.get("identity_mode") != "clean_git_checkout"
        or git_head != upstream_revision
        or not isinstance(git_tree, str)
    ):
        raise ValidationError("parity upstream identity requires the pinned clean checkout receipt")
    candidate_head = candidate_checkout.identity.get("git_head")
    candidate_tree = candidate_checkout.identity.get("git_tree")
    if (
        candidate_checkout.identity.get("identity_mode") != "clean_git_checkout"
        or not isinstance(candidate_head, str)
        or not isinstance(candidate_tree, str)
    ):
        raise ValidationError("parity candidate identity requires a clean checkout receipt")

    raw_declarations = operator_fingerprint.get("parity_implementations")
    if not isinstance(raw_declarations, Mapping) or set(raw_declarations) != set(
        _IMPLEMENTATION_KINDS
    ):
        raise ValidationError(
            "driver fingerprint requires exact parity_implementations declarations"
        )
    upstream = _implementation_from_declaration(
        _UPSTREAM_ARM,
        raw_declarations[_UPSTREAM_ARM],
        checkout_binding_hash=upstream_checkout.binding_hash,
    )
    candidate = _implementation_from_declaration(
        _CANDIDATE_ARM,
        raw_declarations[_CANDIDATE_ARM],
        checkout_binding_hash=candidate_checkout.binding_hash,
    )
    if upstream.source_revision != upstream_revision or upstream.source_tree != git_tree:
        raise ValidationError("upstream implementation declaration differs from checkout HEAD/tree")
    if upstream.release_id != f"upstream:{method_id}":
        raise ValidationError("upstream implementation release_id is not method-specific")
    if candidate.release_id != candidate_release:
        raise ValidationError("candidate implementation declaration differs from manifest release")
    if candidate.source_revision != candidate_head or candidate.source_tree != candidate_tree:
        raise ValidationError(
            "candidate implementation declaration differs from checkout HEAD/tree"
        )
    return ParityImplementationContract(
        method_id=method_id,
        manifest_hash=manifest.manifest_hash,
        binding_receipt_hash=binding_receipt.receipt_hash,
        implementations=(upstream, candidate),
    )


def validate_parity_cell_result(
    cell: ExperimentCell,
    result: ExperimentCellResult,
    *,
    binding_receipt: FrozenBindingReceipt,
    contract: ParityImplementationContract,
    operator_fingerprint: Mapping[str, Any] | None = None,
) -> None:
    """Verify per-arm runtime identities before the executor can persist a result."""

    if result.status == "failed":
        return
    if set(cell.arm_ids) != set(_IMPLEMENTATION_KINDS):
        raise ValidationError("parity cell does not contain the exact implementation arms")
    payload = result.payload
    if payload.get("frozen_binding_receipt_hash") != binding_receipt.receipt_hash:
        raise ValidationError("parity result binding receipt hash is absent or different")
    if payload.get("parity_contract_hash") != contract.contract_hash:
        raise ValidationError("parity result contract hash is absent or different")
    arms = payload.get("arms")
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
        raise ValidationError("parity result requires an arms array")
    receipt_protocol = (
        operator_fingerprint.get("execution_receipt_protocol")
        if operator_fingerprint is not None
        else None
    )
    if receipt_protocol is not None:
        if receipt_protocol != HOST_EXECUTION_RECEIPT_PROTOCOL:
            raise ValidationError("parity driver declared an unknown execution receipt protocol")
        _validate_host_execution_receipts(
            cell,
            result,
            arms=arms,
            binding_receipt=binding_receipt,
            contract=contract,
            operator_fingerprint=operator_fingerprint,
        )
        return
    observed: dict[str, Mapping[str, Any]] = {}
    for item in arms:
        if not isinstance(item, Mapping):
            raise ValidationError("parity arm outcome must be an object")
        arm_id = item.get("arm_id")
        identity = item.get("implementation_identity")
        if not isinstance(arm_id, str) or not isinstance(identity, Mapping):
            raise ValidationError("parity arm outcome requires arm_id and implementation_identity")
        if arm_id in observed:
            raise ValidationError("parity result contains duplicate implementation arms")
        observed[arm_id] = identity
    if set(observed) != set(_IMPLEMENTATION_KINDS):
        raise ValidationError("parity result implementation arms differ from its contract")
    for arm_id, identity in observed.items():
        expected = freeze_mapping(contract.implementation(arm_id).to_dict())
        try:
            actual = freeze_mapping(identity)
        except (TypeError, ValueError) as exc:
            raise ValidationError("parity runtime identity must be canonical JSON") from exc
        if actual != expected:
            raise ValidationError(f"parity arm {arm_id!r} reported another implementation identity")


def validate_host_process_cell_result(
    cell: ExperimentCell,
    result: ExperimentCellResult,
    *,
    binding_receipt: FrozenBindingReceipt,
    operator_fingerprint: Mapping[str, Any],
) -> None:
    """Validate host-generated receipts for a non-parity process experiment."""

    if result.status == "failed":
        return
    if result.payload.get("frozen_binding_receipt_hash") != binding_receipt.receipt_hash:
        raise ValidationError("host-process result binding receipt hash is absent or different")
    if operator_fingerprint.get("execution_receipt_protocol") != HOST_EXECUTION_RECEIPT_PROTOCOL:
        raise ValidationError("host-process driver declared an unknown receipt protocol")
    arms = result.payload.get("arms")
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
        raise ValidationError("host-process result requires an arms array")
    _validate_host_execution_receipts(
        cell,
        result,
        arms=arms,
        binding_receipt=binding_receipt,
        contract=None,
        operator_fingerprint=operator_fingerprint,
    )


def _validate_host_execution_receipts(
    cell: ExperimentCell,
    result: ExperimentCellResult,
    *,
    arms: Sequence[Any],
    binding_receipt: FrozenBindingReceipt,
    contract: ParityImplementationContract | None,
    operator_fingerprint: Mapping[str, Any],
) -> None:
    """Validate identities produced by the host process launcher, not workers."""

    if result.payload.get("host_execution_receipt_protocol") != (HOST_EXECUTION_RECEIPT_PROTOCOL):
        raise ValidationError("host-process result execution receipt protocol is absent")
    config_sha256 = operator_fingerprint.get("config_sha256")
    require_sha256(config_sha256, label="host-process driver config_sha256")
    expected_arms = set(cell.arm_ids)
    bindings = operator_fingerprint.get("host_process_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != expected_arms:
        raise ValidationError("host-process driver bindings are invalid")
    resource_hashes = {content_hash(receipt.as_dict()) for receipt in result.resource_receipts}
    if len(result.resource_receipts) != len(expected_arms) or len(resource_hashes) != len(
        expected_arms
    ):
        raise ValidationError("host process requires one distinct resource receipt per arm")
    observed_arms: set[str] = set()
    referenced_resource_hashes: set[str] = set()
    for item in arms:
        if not isinstance(item, Mapping):
            raise ValidationError("host-process arm outcome must be an object")
        arm_id = item.get("arm_id")
        if not isinstance(arm_id, str) or arm_id not in expected_arms:
            raise ValidationError("host-process receipt has an invalid arm")
        if arm_id in observed_arms:
            raise ValidationError("host-process result contains duplicate receipt arms")
        observed_arms.add(arm_id)
        if _contains_key(item, "implementation_identity"):
            raise ValidationError(
                "worker-provided implementation_identity is forbidden "
                "when host execution receipts are enabled"
            )
        receipt = item.get("host_execution_receipt")
        if not isinstance(receipt, Mapping) or set(receipt) != _HOST_RECEIPT_FIELDS:
            raise ValidationError("host execution receipt fields are invalid")
        if receipt.get("schema_version") != 1 or receipt.get("protocol") != (
            HOST_EXECUTION_RECEIPT_PROTOCOL
        ):
            raise ValidationError("host execution receipt protocol is invalid")
        if receipt.get("arm_id") != arm_id:
            raise ValidationError("host execution receipt belongs to another arm")
        if (
            receipt.get("manifest_hash") != cell.manifest_hash
            or receipt.get("cell_id") != cell.cell_id
            or receipt.get("request_hash") != item.get("request_hash")
            or receipt.get("config_sha256") != config_sha256
        ):
            raise ValidationError("host execution receipt lineage is invalid")
        receipt_without_hash = {
            key: value for key, value in receipt.items() if key != "receipt_hash"
        }
        if receipt.get("receipt_hash") != content_hash(receipt_without_hash):
            raise ValidationError("host execution receipt hash is invalid")

        expected_binding = bindings[arm_id]
        if not isinstance(expected_binding, Mapping) or set(expected_binding) != {
            "checkout_environment",
            "checkout",
            "adapter_checkout_environment",
            "adapter_checkout",
            "command",
        }:
            raise ValidationError("host-process binding fields are invalid")
        checkout_environment = expected_binding["checkout_environment"]
        if (
            not isinstance(checkout_environment, str)
            or receipt.get("checkout_environment") != checkout_environment
        ):
            raise ValidationError("host receipt checkout environment differs")
        frozen_checkout = binding_receipt.binding(
            checkout_environment,
            kind="directory_env",
        )
        if receipt.get("checkout_binding_hash") != frozen_checkout.binding_hash:
            raise ValidationError("host receipt checkout binding differs")
        checkout = receipt.get("checkout")
        expected_checkout = expected_binding["checkout"]
        if (
            not isinstance(checkout, Mapping)
            or set(checkout) != _HOST_CHECKOUT_FIELDS
            or not isinstance(expected_checkout, Mapping)
            or set(expected_checkout) != {"resolved_path_sha256", "git_head", "git_tree"}
        ):
            raise ValidationError("host receipt checkout identity fields are invalid")
        if (
            checkout.get("resolved_path_sha256") != expected_checkout.get("resolved_path_sha256")
            or checkout.get("git_head") != expected_checkout.get("git_head")
            or checkout.get("git_tree") != expected_checkout.get("git_tree")
            or checkout.get("git_clean_before") is not True
            or checkout.get("git_clean_after") is not True
        ):
            raise ValidationError("host receipt checkout identity differs")
        frozen_identity = frozen_checkout.identity
        if (
            frozen_identity.get("resolved_path_sha256") != checkout.get("resolved_path_sha256")
            or frozen_identity.get("git_head") != checkout.get("git_head")
            or frozen_identity.get("git_tree") != checkout.get("git_tree")
        ):
            raise ValidationError("host receipt does not match frozen checkout identity")

        adapter_checkout_environment = expected_binding["adapter_checkout_environment"]
        if (
            not isinstance(adapter_checkout_environment, str)
            or receipt.get("adapter_checkout_environment") != adapter_checkout_environment
        ):
            raise ValidationError("host receipt adapter checkout environment differs")
        frozen_adapter_checkout = binding_receipt.binding(
            adapter_checkout_environment,
            kind="directory_env",
        )
        if receipt.get("adapter_checkout_binding_hash") != frozen_adapter_checkout.binding_hash:
            raise ValidationError("host receipt adapter checkout binding differs")
        adapter_checkout = receipt.get("adapter_checkout")
        expected_adapter_checkout = expected_binding["adapter_checkout"]
        if (
            not isinstance(adapter_checkout, Mapping)
            or set(adapter_checkout) != _HOST_CHECKOUT_FIELDS
            or not isinstance(expected_adapter_checkout, Mapping)
            or set(expected_adapter_checkout) != {"resolved_path_sha256", "git_head", "git_tree"}
        ):
            raise ValidationError("host receipt adapter checkout identity fields are invalid")
        if (
            adapter_checkout.get("resolved_path_sha256")
            != expected_adapter_checkout.get("resolved_path_sha256")
            or adapter_checkout.get("git_head") != expected_adapter_checkout.get("git_head")
            or adapter_checkout.get("git_tree") != expected_adapter_checkout.get("git_tree")
            or adapter_checkout.get("git_clean_before") is not True
            or adapter_checkout.get("git_clean_after") is not True
        ):
            raise ValidationError("host receipt adapter checkout identity differs")
        frozen_adapter_identity = frozen_adapter_checkout.identity
        if (
            frozen_adapter_identity.get("resolved_path_sha256")
            != adapter_checkout.get("resolved_path_sha256")
            or frozen_adapter_identity.get("git_head") != adapter_checkout.get("git_head")
            or frozen_adapter_identity.get("git_tree") != adapter_checkout.get("git_tree")
        ):
            raise ValidationError("host receipt does not match frozen adapter checkout identity")
        implementation = contract.implementation(arm_id) if contract is not None else None
        if implementation is not None and (
            implementation.source_revision != checkout.get("git_head")
            or implementation.source_tree != checkout.get("git_tree")
        ):
            raise ValidationError("parity host receipt does not match implementation identity")
        command = receipt.get("command")
        if not isinstance(command, Mapping):
            raise ValidationError("host receipt command identity is invalid")
        try:
            actual_command = freeze_mapping(command)
            declared_command = freeze_mapping(expected_binding["command"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("host receipt command must be canonical JSON") from exc
        if actual_command != declared_command:
            raise ValidationError("host receipt command differs from driver binding")
        if implementation is not None and implementation.runtime_entrypoint != (
            f"{adapter_checkout_environment}:{command.get('entrypoint_relative_path')}"
        ):
            raise ValidationError(
                "parity host receipt entrypoint differs from implementation contract"
            )
        process = receipt.get("process")
        if not isinstance(process, Mapping) or set(process) != _HOST_PROCESS_FIELDS:
            raise ValidationError("host receipt process fields are invalid")
        for field_name in _HOST_PROCESS_FIELDS:
            require_sha256(
                process.get(field_name),
                label=f"host receipt {field_name}",
            )
        resource_hash = process["resource_receipt_sha256"]
        if resource_hash not in resource_hashes or resource_hash in referenced_resource_hashes:
            raise ValidationError("host receipt references another resource receipt")
        referenced_resource_hashes.add(resource_hash)
    if observed_arms != expected_arms:
        raise ValidationError("host receipt arms differ from the frozen cell")
    if referenced_resource_hashes != resource_hashes:
        raise ValidationError("host receipts do not cover all resource receipts")


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_key(item, key) for item in value)
    return False


def _implementation_from_declaration(
    arm_id: str,
    value: Any,
    *,
    checkout_binding_hash: str | None,
) -> ParityImplementationIdentity:
    expected_fields = {
        "implementation_kind",
        "implementation_id",
        "runtime_entrypoint",
        "source_revision",
        "source_tree",
        "release_id",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValidationError(f"parity implementation declaration {arm_id!r} has invalid fields")
    expected_kind = _IMPLEMENTATION_KINDS[arm_id]
    if value["implementation_kind"] != expected_kind:
        raise ValidationError(f"parity implementation declaration {arm_id!r} has the wrong kind")
    return ParityImplementationIdentity(
        arm_id=arm_id,
        implementation_kind=expected_kind,
        implementation_id=_text(value["implementation_id"], "implementation_id"),
        runtime_entrypoint=_text(value["runtime_entrypoint"], "runtime_entrypoint"),
        source_revision=_text(value["source_revision"], "source_revision"),
        source_tree=_text(value["source_tree"], "source_tree"),
        release_id=_text(value["release_id"], "release_id"),
        checkout_binding_hash=checkout_binding_hash,
    )


def _metadata_text(manifest: ExperimentManifest, name: str) -> str:
    return _text(manifest.metadata.get(name), f"manifest metadata {name}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    return value


__all__ = [
    "HOST_EXECUTION_RECEIPT_PROTOCOL",
    "ParityImplementationContract",
    "ParityImplementationIdentity",
    "build_parity_implementation_contract",
    "validate_host_process_cell_result",
    "validate_parity_cell_result",
]
