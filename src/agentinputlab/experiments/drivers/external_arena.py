"""Checksum-locked multi-arm driver for stateful external arenas.

The public atomic replay contract is intentionally pairwise: one external
runtime operation restores the same protected origin for a baseline and one
treatment.  This driver lifts that contract to a frozen multi-arm experiment
cell by constructing every payload before execution and then running one
atomic pair per treatment.  Every pair is bound to the same checkpoint,
origin, replay identity, seed, and dependency set.  The repeated NO_OP result
must also have identical official metrics, trace, usage, and logical receipt
fields or the whole cell aborts without a terminal result.

No upstream runtime, model, selector, or intervention is selected here.
Operators provide checksum-locked zero-argument builders in one strict JSON
configuration.  Loading and preflight never download assets or execute a
benchmark cell.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.benchmarks import (
    ATOMIC_EXTERNAL_PAIR_CONTRACT,
    AtomicExternalReplayBridge,
    AtomicExternalReplayRunner,
    AtomicPairedReplayRecord,
    AtomicReplayArmRequest,
    AtomicReplayRequest,
    BenchmarkRegistry,
    interactive_benchmark_specs,
)
from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import (
    content_hash,
    deep_freeze_jsonable,
    freeze_mapping,
    to_jsonable,
)
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.experiments.executor import ExperimentCellResult
from agentinputlab.experiments.loader import resolve_driver_entrypoint
from agentinputlab.experiments.plan import ExperimentCell
from agentinputlab.experiments.schema import (
    ExperimentArmKind,
    ExperimentArmSpec,
    ExperimentBenchmarkTarget,
    ExperimentManifest,
    PrerequisiteKind,
)
from agentinputlab.selectors import (
    SelectorArtifactBundle,
    SelectorArtifactStage,
    VisibleStateFeatureExtractor,
)

CONFIG_PATH_ENV = "AILAB_EXTERNAL_ARENA_DRIVER_CONFIG"
CONFIG_SHA256_ENV = "AILAB_EXTERNAL_ARENA_DRIVER_CONFIG_SHA256"

_CONFIG_SCHEMA_VERSION = 3
_DRIVER_IMPLEMENTATION_ID = "agentinputlab.external_arena.atomic_multi_arm"
_VALID_SUCCESS_FORMULA_ID = "agentinputlab.valid_success.v1"
_CONTRACT_PRESERVED_METRIC = "contract_preserved"
_PRIVILEGED_LEAKAGE_METRIC = "privileged_leakage"
_VALID_SUCCESS_METRIC = "valid_success"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ANONYMIZED_GROUP = re.compile(r"^group_[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_POLICY_ARTIFACT_BYTES = 32 * 1024 * 1024
_BUILDER_KINDS = frozenset({"arena_checkpoint", "atomic_bridge", "arm_payload"})
_CORE_FAMILIES = frozenset(
    {
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    }
)
_EXECUTABLE_RELEASE_FAMILIES = frozenset(
    {
        *_CORE_FAMILIES,
        SemanticFamily.COMPOSITE,
    }
)
_SUPPORTED_ARENAS = frozenset(
    {
        "tau3.banking_knowledge",
        "tau3.retail",
        "state_bench.common_support",
        "toolsandbox.common_support",
        "appworld_ul.common_support",
        "webarena.external",
    }
)
_REGISTERED_OFFICIAL_SUCCESS_METRICS = {
    "tau3.banking_knowledge": "official_reward",
    # Native RewardInfo is retained separately. A completed native evaluator
    # must not turn an episode timeout or shared-budget stop into paper success.
    "tau3.retail": "terminal_success",
    "state_bench.common_support": "task_completion_pass",
    "toolsandbox.common_support": "official_similarity",
    "appworld_ul.common_support": "task_success",
}
_POLICY_KIND_BY_RELEASE = {
    "policy.locked_best_eligible_global": "locked_best_eligible_global",
    "policy.eligibility_aware_selector": "frozen_selector",
}
_REQUIRED_SHARED_DEPENDENCY_SLOTS = frozenset(
    {
        "agent_model",
        "official_runtime",
        "corpus",
        "preprocessing",
        "index",
        "embedder",
        "retriever",
        "reranker",
        "reader",
        "scorer",
    }
)
_RETAIL_ENVIRONMENT_DEPENDENCY_SLOTS = frozenset({"user_simulator", "nl_assertion_judge"})
_NOT_APPLICABLE_DEPENDENCY_HASHES = MappingProxyType(
    {
        slot: content_hash(
            {
                "agentinputlab_dependency": "not_applicable",
                "schema_version": 1,
                "slot": slot,
            }
        )
        for slot in _REQUIRED_SHARED_DEPENDENCY_SLOTS
    }
)
_GATE_BINDINGS = {
    "toolsandbox.common_support": (
        "AILAB_TOOLSANDBOX_GATE_RECEIPT",
        "AILAB_TOOLSANDBOX_GATE_RECEIPT_SHA256",
        "atomic_restore",
    ),
    "appworld_ul.common_support": (
        "AILAB_APPWORLD_UL_GATE_RECEIPT",
        "AILAB_APPWORLD_UL_GATE_RECEIPT_SHA256",
        "atomic_restore",
    ),
    "webarena.external": (
        "AILAB_WEBARENA_STACK_RECEIPT",
        "AILAB_WEBARENA_STACK_RECEIPT_SHA256",
        "independent_stack_or_verified_reset",
    ),
}
_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
)


@dataclass(frozen=True, slots=True)
class ExternalArenaCheckpoint:
    """Operator-captured visible state plus opaque protected-runtime identities."""

    benchmark_id: str
    checkpoint_id: str
    group_id: str
    source_revision: str
    origin_state_hash: str
    replay_identity: str
    locked_dependency_hashes: Mapping[str, str]
    visible_state: FrozenAgentState
    eligible_families: tuple[SemanticFamily, ...]

    def __post_init__(self) -> None:
        for label in ("benchmark_id", "checkpoint_id", "group_id", "source_revision"):
            _require_text(getattr(self, label), label=f"external checkpoint {label}")
        _require_sha256(
            self.origin_state_hash,
            label="external checkpoint origin_state_hash",
        )
        _require_sha256(
            self.replay_identity,
            label="external checkpoint replay_identity",
        )
        if not isinstance(self.visible_state, FrozenAgentState):
            raise TypeError("external checkpoint visible_state must be FrozenAgentState")
        families = tuple(self.eligible_families)
        if len(families) != len(set(families)) or any(
            family not in _EXECUTABLE_RELEASE_FAMILIES for family in families
        ):
            raise ValidationError(
                "external checkpoint eligible_families must be unique executable families"
            )
        object.__setattr__(
            self,
            "locked_dependency_hashes",
            _validated_dependency_hashes(self.locked_dependency_hashes, benchmark_id=self.benchmark_id),
        )
        object.__setattr__(self, "eligible_families", families)

    @property
    def identity_hash(self) -> str:
        return content_hash(
            {
                "benchmark_id": self.benchmark_id,
                "checkpoint_id": self.checkpoint_id,
                "group_id": self.group_id,
                "source_revision": self.source_revision,
                "origin_state_hash": self.origin_state_hash,
                "replay_identity": self.replay_identity,
                "locked_dependency_hashes": self.locked_dependency_hashes,
                "visible_state_hash": self.visible_state.state_hash,
                "eligible_families": self.eligible_families,
            }
        )


@dataclass(frozen=True, slots=True)
class ExternalArmPayload:
    """One typed, canonically serializable artifact sent to the external runtime."""

    arm_id: str
    release_id: str
    family: SemanticFamily
    profile_kind: ProfileKind
    payload: Any = field(repr=False)
    selector_decision: Mapping[str, Any] | None = None
    selector_input_state_hash: str | None = None
    selector_feature_hash: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.arm_id, label="external arm payload arm_id")
        _require_text(self.release_id, label="external arm payload release_id")
        if not isinstance(self.family, SemanticFamily):
            raise ValidationError("external arm payload family must be typed")
        if not isinstance(self.profile_kind, ProfileKind):
            raise ValidationError("external arm payload profile_kind must be typed")
        try:
            payload = deep_freeze_jsonable(self.payload)
            content_hash(payload)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"external arm payload must be canonical JSON: {exc}") from exc
        if payload is None:
            raise ValidationError("external arm payload cannot be null")
        decision = self.selector_decision
        if decision is not None:
            decision = _validated_policy_decision(decision)
        selector_hashes = (
            self.selector_input_state_hash,
            self.selector_feature_hash,
        )
        if all(value is None for value in selector_hashes):
            pass
        elif any(value is None for value in selector_hashes):
            raise ValidationError(
                "selector input state_hash and feature_hash must be supplied together"
            )
        else:
            _require_sha256(
                self.selector_input_state_hash,
                label="selector input state_hash",
            )
            _require_sha256(
                self.selector_feature_hash,
                label="selector input feature_hash",
            )
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "selector_decision", decision)

    @property
    def payload_sha256(self) -> str:
        return content_hash(
            {
                "arm_id": self.arm_id,
                "release_id": self.release_id,
                "family": self.family,
                "profile_kind": self.profile_kind,
                "payload": self.payload,
                "selector_decision": self.selector_decision,
                "selector_input_state_hash": self.selector_input_state_hash,
                "selector_feature_hash": self.selector_feature_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class ExternalCheckpointBuildRequest:
    benchmark_id: str
    split: str
    checkpoint_id: str
    source_revision: str
    model_slot: str
    seed: int
    builder_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        for label in (
            "benchmark_id",
            "split",
            "checkpoint_id",
            "source_revision",
            "model_slot",
        ):
            _require_text(getattr(self, label), label=f"checkpoint build {label}")
        _require_seed(self.seed)
        object.__setattr__(
            self,
            "builder_parameters",
            freeze_mapping(self.builder_parameters),
        )


@dataclass(frozen=True, slots=True)
class ExternalUnitPreflightRequest:
    """Exact resolved unit inventory presented to a read-only checkpoint builder."""

    benchmark_id: str
    split: str
    source_revision: str
    unit_ids: tuple[str, ...]
    builder_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        for label in ("benchmark_id", "split", "source_revision"):
            _require_text(
                getattr(self, label),
                label=f"unit preflight {label}",
            )
        unit_ids = tuple(self.unit_ids)
        if (
            not unit_ids
            or len(unit_ids) != len(set(unit_ids))
            or any(
                not isinstance(item, str) or not item.strip() or item != item.strip()
                for item in unit_ids
            )
        ):
            raise ValidationError("unit preflight requires unique trimmed non-empty unit IDs")
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(
            self,
            "builder_parameters",
            freeze_mapping(self.builder_parameters),
        )


@dataclass(frozen=True, slots=True)
class ExternalBridgeBuildRequest:
    checkpoint: ExternalArenaCheckpoint
    model_slot: str
    seed: int
    builder_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, ExternalArenaCheckpoint):
            raise TypeError("bridge build request requires ExternalArenaCheckpoint")
        _require_text(self.model_slot, label="bridge build model_slot")
        _require_seed(self.seed)
        object.__setattr__(
            self,
            "builder_parameters",
            freeze_mapping(self.builder_parameters),
        )


@dataclass(frozen=True, slots=True)
class ExternalArmPayloadBuildRequest:
    checkpoint: ExternalArenaCheckpoint
    arm_id: str
    release_id: str
    family: SemanticFamily
    profile_kind: ProfileKind
    arm_kind: ExperimentArmKind
    model_slot: str
    seed: int
    builder_parameters: Mapping[str, Any]
    arm_parameters: Mapping[str, Any]
    pre_action: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, ExternalArenaCheckpoint):
            raise TypeError("payload build request requires ExternalArenaCheckpoint")
        for label in ("arm_id", "release_id", "model_slot"):
            _require_text(getattr(self, label), label=f"payload build {label}")
        if not isinstance(self.family, SemanticFamily):
            raise ValidationError("payload build family must be typed")
        if not isinstance(self.profile_kind, ProfileKind):
            raise ValidationError("payload build profile_kind must be typed")
        if not isinstance(self.arm_kind, ExperimentArmKind):
            raise ValidationError("payload build arm_kind must be typed")
        _require_seed(self.seed)
        object.__setattr__(
            self,
            "builder_parameters",
            freeze_mapping(self.builder_parameters),
        )
        object.__setattr__(self, "arm_parameters", freeze_mapping(self.arm_parameters))
        frozen_pre_action = freeze_mapping(self.pre_action)
        for key in ("state_hash", "feature_hash"):
            _require_sha256(
                frozen_pre_action.get(key),
                label=f"payload build pre_action {key}",
            )
        object.__setattr__(self, "pre_action", frozen_pre_action)


@runtime_checkable
class LockedExternalArenaBuilder(Protocol):
    """Shared contract for operator-owned, checksum-locked component builders."""

    builder_kind: str

    def fingerprint(self) -> Mapping[str, Any]: ...

    def preflight(self, parameters: Mapping[str, Any]) -> Sequence[str]: ...

    def build(self, request: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class _LockedBuilderSpec:
    entrypoint: str
    fingerprint_sha256: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entrypoint, str) or _ENTRYPOINT.fullmatch(self.entrypoint) is None:
            raise ValidationError("external builder entrypoint must use module:factory syntax")
        _require_sha256(
            self.fingerprint_sha256,
            label="external builder fingerprint_sha256",
        )
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> _LockedBuilderSpec:
        data = _strict_mapping(
            value,
            path=path,
            keys={"entrypoint", "fingerprint_sha256", "parameters"},
            required={"entrypoint", "fingerprint_sha256"},
        )
        parameters = data.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValidationError(f"{path}.parameters must be an object")
        return cls(
            entrypoint=_text(data["entrypoint"], f"{path}.entrypoint"),
            fingerprint_sha256=_text(
                data["fingerprint_sha256"],
                f"{path}.fingerprint_sha256",
            ),
            parameters=parameters,
        )

    @property
    def binding_identity(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "entrypoint": self.entrypoint,
                "fingerprint_sha256": self.fingerprint_sha256,
                "parameters_sha256": content_hash(self.parameters),
            }
        )


@dataclass(frozen=True, slots=True)
class _BridgeIdentityLock:
    bridge_id: str
    bridge_revision: str
    scorer_id: str
    scorer_revision: str
    dependency_hashes_by_model_slot: Mapping[str, Mapping[str, str]]
    benchmark_id: str = ""

    def __post_init__(self) -> None:
        for label in ("bridge_id", "scorer_id"):
            _require_text(getattr(self, label), label=f"bridge lock {label}")
        for label in ("bridge_revision", "scorer_revision"):
            _require_sha256(getattr(self, label), label=f"bridge lock {label}")
        raw = self.dependency_hashes_by_model_slot
        if not isinstance(raw, Mapping) or not raw:
            raise ValidationError("bridge lock dependency_hashes_by_model_slot must be non-empty")
        normalized: dict[str, Mapping[str, str]] = {}
        for model_slot, dependencies in raw.items():
            _require_text(model_slot, label="bridge lock model_slot")
            normalized[model_slot] = _validated_dependency_hashes(dependencies, benchmark_id=self.benchmark_id)
        object.__setattr__(
            self,
            "dependency_hashes_by_model_slot",
            MappingProxyType(normalized),
        )

    @classmethod
    def from_mapping(cls, value: Any, *, path: str, benchmark_id: str = "") -> _BridgeIdentityLock:
        data = _strict_mapping(
            value,
            path=path,
            keys={
                "bridge_id",
                "bridge_revision",
                "scorer_id",
                "scorer_revision",
                "dependency_hashes_by_model_slot",
            },
            required={
                "bridge_id",
                "bridge_revision",
                "scorer_id",
                "scorer_revision",
                "dependency_hashes_by_model_slot",
            },
        )
        dependencies = data["dependency_hashes_by_model_slot"]
        if not isinstance(dependencies, Mapping):
            raise ValidationError(f"{path}.dependency_hashes_by_model_slot must be an object")
        return cls(
            bridge_id=_text(data["bridge_id"], f"{path}.bridge_id"),
            bridge_revision=_text(
                data["bridge_revision"],
                f"{path}.bridge_revision",
            ),
            scorer_id=_text(data["scorer_id"], f"{path}.scorer_id"),
            scorer_revision=_text(
                data["scorer_revision"],
                f"{path}.scorer_revision",
            ),
            dependency_hashes_by_model_slot=dependencies,
            benchmark_id=benchmark_id,
        )

    def dependencies(self, model_slot: str) -> Mapping[str, str]:
        try:
            return self.dependency_hashes_by_model_slot[model_slot]
        except KeyError as exc:
            raise ValidationError(
                f"bridge lock has no dependency identity for model slot {model_slot!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class _OutcomeContract:
    formula_id: str
    official_success_metric: str

    def __post_init__(self) -> None:
        if self.formula_id != _VALID_SUCCESS_FORMULA_ID:
            raise ValidationError(
                f"external outcome formula_id must be {_VALID_SUCCESS_FORMULA_ID!r}"
            )
        _require_text(
            self.official_success_metric,
            label="external outcome official_success_metric",
        )
        if self.official_success_metric in {
            _CONTRACT_PRESERVED_METRIC,
            _PRIVILEGED_LEAKAGE_METRIC,
            _VALID_SUCCESS_METRIC,
        }:
            raise ValidationError(
                "external official_success_metric conflicts with a host outcome field"
            )

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> _OutcomeContract:
        data = _strict_mapping(
            value,
            path=path,
            keys={"formula_id", "official_success_metric"},
            required={"formula_id", "official_success_metric"},
        )
        return cls(
            formula_id=_text(data["formula_id"], f"{path}.formula_id"),
            official_success_metric=_text(
                data["official_success_metric"],
                f"{path}.official_success_metric",
            ),
        )


@dataclass(frozen=True, slots=True)
class _TargetBinding:
    benchmark_id: str
    split: str
    source_revision: str
    checkpoint_builder: _LockedBuilderSpec
    bridge_builder: _LockedBuilderSpec
    bridge_lock: _BridgeIdentityLock
    outcome_contract: _OutcomeContract

    @property
    def target_key(self) -> str:
        return f"{self.benchmark_id}:{self.split}"


@dataclass(frozen=True, slots=True)
class _ArmBinding:
    release_id: str
    family: SemanticFamily
    profile_kind: ProfileKind
    arm_kind: ExperimentArmKind
    payload_builder: _LockedBuilderSpec
    policy_kind: str | None = None

    def __post_init__(self) -> None:
        if self.arm_kind is ExperimentArmKind.POLICY:
            expected = _POLICY_KIND_BY_RELEASE.get(self.release_id)
            if expected is None or self.policy_kind != expected:
                raise ValidationError(
                    "policy arm binding kind differs from its supported policy release"
                )
        elif self.policy_kind is not None:
            raise ValidationError("policy_kind is valid only for policy arm bindings")


@dataclass(frozen=True, slots=True)
class _GroupPartitionCommitment:
    members_by_split: Mapping[str, tuple[str, ...]]
    commitments_by_split: Mapping[str, str]

    def __post_init__(self) -> None:
        members = dict(self.members_by_split)
        commitments = dict(self.commitments_by_split)
        if set(members) != {"train", "calibration"} or set(commitments) != set(members):
            raise ValidationError(
                "policy group partition must contain exact train and calibration splits"
            )
        normalized_members: dict[str, tuple[str, ...]] = {}
        observed: set[str] = set()
        for split in ("train", "calibration"):
            values = tuple(members[split])
            if (
                not values
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
                or any(
                    not isinstance(item, str) or _ANONYMIZED_GROUP.fullmatch(item) is None
                    for item in values
                )
            ):
                raise ValidationError(
                    f"policy group partition {split} members must be sorted unique "
                    "privacy-safe group hashes"
                )
            if observed.intersection(values):
                raise ValidationError("policy group partition members cross splits")
            observed.update(values)
            expected = content_hash({"split": split, "group_ids": list(values)})
            if commitments[split] != expected:
                raise ValidationError(
                    f"policy group partition {split} commitment differs from its members"
                )
            normalized_members[split] = values
        object.__setattr__(
            self,
            "members_by_split",
            MappingProxyType(normalized_members),
        )
        object.__setattr__(
            self,
            "commitments_by_split",
            MappingProxyType(commitments),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        path: str,
    ) -> _GroupPartitionCommitment:
        data = _strict_mapping(
            value,
            path=path,
            keys={"schema_version", "hash_scheme", "splits"},
            required={"schema_version", "hash_scheme", "splits"},
        )
        if data["schema_version"] != 1 or data["hash_scheme"] != "benchmark_group.content_hash.v1":
            raise ValidationError(f"{path} has an unsupported group commitment schema")
        splits = _strict_mapping(
            data["splits"],
            path=f"{path}.splits",
            keys={"train", "calibration"},
            required={"train", "calibration"},
        )
        members: dict[str, tuple[str, ...]] = {}
        commitments: dict[str, str] = {}
        for split in ("train", "calibration"):
            row = _strict_mapping(
                splits[split],
                path=f"{path}.splits.{split}",
                keys={"member_hashes", "commitment_sha256"},
                required={"member_hashes", "commitment_sha256"},
            )
            raw_members = row["member_hashes"]
            if isinstance(raw_members, (str, bytes)) or not isinstance(
                raw_members,
                Sequence,
            ):
                raise ValidationError(f"{path}.splits.{split}.member_hashes must be an array")
            members[split] = tuple(raw_members)
            commitments[split] = _text(
                row["commitment_sha256"],
                f"{path}.splits.{split}.commitment_sha256",
            )
        return cls(members, commitments)

    @property
    def excluded_members(self) -> frozenset[str]:
        return frozenset(member for values in self.members_by_split.values() for member in values)

    @property
    def split_provenance_hash(self) -> str:
        return content_hash({"group_split_digests": dict(self.commitments_by_split)})

    def to_receipt(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "hash_scheme": "benchmark_group.content_hash.v1",
                "split_commitments": dict(self.commitments_by_split),
                "excluded_member_count": len(self.excluded_members),
                "excluded_members_sha256": content_hash(
                    {"member_hashes": sorted(self.excluded_members)}
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class _SelectorPolicyManifest:
    discovery_revision: str
    discovery_sha256: str
    training_table_sha256: str
    split_provenance_hash: str
    comparison_profile_hash: str
    representative_by_family: Mapping[SemanticFamily, str]
    selected_family: SemanticFamily
    selected_release_id: str
    calibration_split_digest: str
    group_partition: _GroupPartitionCommitment

    def __post_init__(self) -> None:
        _require_text(self.discovery_revision, label="policy manifest discovery revision")
        for label in (
            "discovery_sha256",
            "training_table_sha256",
            "split_provenance_hash",
            "comparison_profile_hash",
            "calibration_split_digest",
        ):
            _require_sha256(getattr(self, label), label=f"policy manifest {label}")
        representatives = _validated_policy_representatives(self.representative_by_family)
        if self.selected_family not in _CORE_FAMILIES:
            raise ValidationError("policy manifest selected_family must be a core family")
        _require_text(
            self.selected_release_id,
            label="policy manifest selected_release_id",
        )
        if representatives.get(self.selected_family) != self.selected_release_id:
            raise ValidationError(
                "policy manifest selected release differs from its frozen representative"
            )
        if not isinstance(self.group_partition, _GroupPartitionCommitment):
            raise ValidationError("policy manifest requires a typed group partition")
        if self.group_partition.split_provenance_hash != self.split_provenance_hash:
            raise ValidationError("policy group partition differs from split_provenance_hash")
        if (
            self.group_partition.commitments_by_split["calibration"]
            != self.calibration_split_digest
        ):
            raise ValidationError(
                "policy group partition calibration commitment differs from selection receipt"
            )
        object.__setattr__(
            self,
            "representative_by_family",
            MappingProxyType(representatives),
        )

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> _SelectorPolicyManifest:
        data = _strict_mapping(
            value,
            path=path,
            keys={
                "schema_version",
                "artifact",
                "stage",
                "discovery",
                "training_table_sha256",
                "split_provenance_hash",
                "comparison_profile_hash",
                "representative_by_family",
                "group_partition",
                "locked_best_eligible_global",
            },
            required={
                "schema_version",
                "artifact",
                "stage",
                "discovery",
                "training_table_sha256",
                "split_provenance_hash",
                "comparison_profile_hash",
                "representative_by_family",
                "group_partition",
                "locked_best_eligible_global",
            },
        )
        if (
            data["schema_version"] != 1
            or data["artifact"] != "agentinputlab.selector_policy_manifest"
            or data["stage"] != "frozen"
        ):
            raise ValidationError(f"{path} must be a frozen selector policy manifest schema v1")
        discovery = _strict_mapping(
            data["discovery"],
            path=f"{path}.discovery",
            keys={"revision", "sha256"},
            required={"revision", "sha256"},
        )
        locked = _strict_mapping(
            data["locked_best_eligible_global"],
            path=f"{path}.locked_best_eligible_global",
            keys={"selected_family", "selected_release_id", "selection_receipt"},
            required={"selected_family", "selected_release_id", "selection_receipt"},
        )
        receipt = _strict_mapping(
            locked["selection_receipt"],
            path=f"{path}.locked_best_eligible_global.selection_receipt",
            keys={
                "schema_version",
                "rule_id",
                "selection_split",
                "training_table_sha256",
                "calibration_split_digest",
            },
            required={
                "schema_version",
                "rule_id",
                "selection_split",
                "training_table_sha256",
                "calibration_split_digest",
            },
        )
        if (
            receipt["schema_version"] != 1
            or receipt["rule_id"] != "best_eligible_global.v1"
            or receipt["selection_split"] != "calibration"
            or receipt["training_table_sha256"] != data["training_table_sha256"]
        ):
            raise ValidationError(f"{path} locked-best selection receipt is inconsistent")
        representatives = _policy_representatives_from_mapping(
            data["representative_by_family"],
            path=f"{path}.representative_by_family",
        )
        try:
            selected_family = SemanticFamily(
                _text(
                    locked["selected_family"],
                    f"{path}.locked_best_eligible_global.selected_family",
                )
            )
        except ValueError as exc:
            raise ValidationError(f"{path} locked-best selected_family is unknown") from exc
        return cls(
            discovery_revision=_text(
                discovery["revision"],
                f"{path}.discovery.revision",
            ),
            discovery_sha256=_text(
                discovery["sha256"],
                f"{path}.discovery.sha256",
            ),
            training_table_sha256=_text(
                data["training_table_sha256"],
                f"{path}.training_table_sha256",
            ),
            split_provenance_hash=_text(
                data["split_provenance_hash"],
                f"{path}.split_provenance_hash",
            ),
            comparison_profile_hash=_text(
                data["comparison_profile_hash"],
                f"{path}.comparison_profile_hash",
            ),
            representative_by_family=representatives,
            selected_family=selected_family,
            selected_release_id=_text(
                locked["selected_release_id"],
                f"{path}.locked_best_eligible_global.selected_release_id",
            ),
            calibration_split_digest=_text(
                receipt["calibration_split_digest"],
                (f"{path}.locked_best_eligible_global.selection_receipt.calibration_split_digest"),
            ),
            group_partition=_GroupPartitionCommitment.from_mapping(
                data["group_partition"],
                path=f"{path}.group_partition",
            ),
        )

    @property
    def manifest_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class _DriverConfig:
    driver_id: str
    driver_revision_sha256: str
    baseline_arm_id: str
    benchmark_targets: Mapping[str, _TargetBinding]
    arms: Mapping[str, _ArmBinding]

    def __post_init__(self) -> None:
        _require_text(self.driver_id, label="external driver_id")
        _require_sha256(
            self.driver_revision_sha256,
            label="external driver_revision_sha256",
        )
        _require_text(self.baseline_arm_id, label="external baseline_arm_id")
        targets = dict(self.benchmark_targets)
        arms = dict(self.arms)
        if not targets or any(key != binding.target_key for key, binding in targets.items()):
            raise ValidationError(
                "external benchmark target keys must be exact cell_id:split identities"
            )
        if not arms or self.baseline_arm_id not in arms:
            raise ValidationError("external driver requires its configured baseline arm")
        baseline = arms[self.baseline_arm_id]
        if (
            baseline.release_id != "baseline.no_op"
            or baseline.family is not SemanticFamily.NO_OP
            or baseline.arm_kind is not ExperimentArmKind.RELEASE
        ):
            raise ValidationError(
                "external driver baseline must bind baseline.no_op in the no_op family"
            )
        object.__setattr__(self, "benchmark_targets", MappingProxyType(targets))
        object.__setattr__(self, "arms", MappingProxyType(arms))

    @classmethod
    def from_mapping(cls, value: Any) -> _DriverConfig:
        data = _strict_mapping(
            value,
            path="external_driver_config",
            keys={
                "schema_version",
                "driver_id",
                "driver_revision_sha256",
                "baseline_arm_id",
                "benchmark_targets",
                "arms",
            },
            required={
                "schema_version",
                "driver_id",
                "driver_revision_sha256",
                "baseline_arm_id",
                "benchmark_targets",
                "arms",
            },
        )
        if data["schema_version"] != _CONFIG_SCHEMA_VERSION:
            raise ValidationError("unsupported external arena config schema_version")
        raw_targets = _nonempty_object(
            data["benchmark_targets"],
            path="external_driver_config.benchmark_targets",
        )
        raw_arms = _nonempty_object(
            data["arms"],
            path="external_driver_config.arms",
        )
        targets = {
            key: _target_binding(
                item,
                path=f"external_driver_config.benchmark_targets[{key!r}]",
            )
            for key, item in raw_targets.items()
        }
        arms = {
            key: _arm_binding(
                item,
                path=f"external_driver_config.arms[{key!r}]",
            )
            for key, item in raw_arms.items()
        }
        return cls(
            driver_id=_text(data["driver_id"], "external_driver_config.driver_id"),
            driver_revision_sha256=_text(
                data["driver_revision_sha256"],
                "external_driver_config.driver_revision_sha256",
            ),
            baseline_arm_id=_text(
                data["baseline_arm_id"],
                "external_driver_config.baseline_arm_id",
            ),
            benchmark_targets=targets,
            arms=arms,
        )


class ExternalArenaExperimentDriver:
    """Execute exact multi-arm cells through locked atomic external bridges."""

    def __init__(self, config: _DriverConfig, *, config_sha256: str) -> None:
        if not isinstance(config, _DriverConfig):
            raise TypeError("external arena driver requires parsed configuration")
        _require_sha256(config_sha256, label="external driver config_sha256")
        self._config = config
        self._config_sha256 = config_sha256
        self._feature_extractor = VisibleStateFeatureExtractor()
        self._preflight_manifest_hash: str | None = None
        self._policy_binding_receipts: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
        self._gate_binding_receipts: Mapping[str, Mapping[str, Any]] = MappingProxyType({})

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> ExternalArenaExperimentDriver:
        """Load exact UTF-8 JSON bytes after checking their caller-supplied hash."""

        _require_sha256(expected_sha256, label="expected external config SHA-256")
        source = Path(path).expanduser()
        if not source.is_absolute():
            raise ArtifactError("external arena config path must be absolute")
        try:
            if source.stat().st_size > _MAX_CONFIG_BYTES:
                raise ArtifactError("external arena config exceeds the size limit")
            raw = source.read_bytes()
        except OSError as exc:
            raise ArtifactError("cannot read external arena driver config") from exc
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected_sha256:
            raise ArtifactError("external arena driver config checksum mismatch")
        return cls(_DriverConfig.from_mapping(_strict_json(raw)), config_sha256=observed)

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "implementation_id": _DRIVER_IMPLEMENTATION_ID,
                "schema_version": _CONFIG_SCHEMA_VERSION,
                "driver_id": self._config.driver_id,
                "driver_revision_sha256": self._config.driver_revision_sha256,
                "config_sha256": self._config_sha256,
                "baseline_arm_id": self._config.baseline_arm_id,
                "benchmark_targets": {
                    key: {
                        "benchmark_id": binding.benchmark_id,
                        "split": binding.split,
                        "source_revision": binding.source_revision,
                        "checkpoint_builder": binding.checkpoint_builder.binding_identity,
                        "bridge_builder": binding.bridge_builder.binding_identity,
                        "bridge_lock": to_jsonable(binding.bridge_lock),
                        "outcome_contract": to_jsonable(binding.outcome_contract),
                    }
                    for key, binding in self._config.benchmark_targets.items()
                },
                "arms": {
                    arm_id: {
                        "release_id": binding.release_id,
                        "family": binding.family.value,
                        "profile_kind": binding.profile_kind.value,
                        "arm_kind": binding.arm_kind.value,
                        "payload_builder": binding.payload_builder.binding_identity,
                        "policy_kind": binding.policy_kind,
                    }
                    for arm_id, binding in self._config.arms.items()
                },
                "resolved_policy_bindings": self._policy_binding_receipts,
                "resolved_gate_bindings": self._gate_binding_receipts,
                "feature_schema_id": self._feature_extractor.feature_schema_id,
                "feature_schema_hash": self._feature_extractor.feature_schema_hash,
                # Static capability attestation for Studio's second,
                # execution-confirmation gate. Successful preflight still has
                # to produce the concrete policy/gate receipts below; these
                # booleans only state which fail-closed checks this driver
                # implementation enforces.
                "studio_campaign_attestation": {
                    "schema_version": 1,
                    "shared_dependency_slots": True,
                    "selector_group_exclusion": True,
                    "typed_gate_receipts": True,
                },
            }
        )

    def preflight(self, manifest: ExperimentManifest) -> Sequence[str]:
        """Validate locks and builders without constructing checkpoints or bridges."""

        self._preflight_manifest_hash = None
        self._policy_binding_receipts = MappingProxyType({})
        self._gate_binding_receipts = MappingProxyType({})
        blockers = self._manifest_blockers(manifest)
        if blockers:
            return tuple(blockers)
        used_specs: dict[tuple[str, str, str, str], _LockedBuilderSpec] = {}
        for target in manifest.benchmarks:
            binding = self._config.benchmark_targets[_target_key(target)]
            for kind, spec in (
                ("arena_checkpoint", binding.checkpoint_builder),
                ("atomic_bridge", binding.bridge_builder),
            ):
                used_specs[
                    (
                        kind,
                        spec.entrypoint,
                        spec.fingerprint_sha256,
                        content_hash(spec.parameters),
                    )
                ] = spec
        selected_arm_ids = {
            arm_id
            for target in manifest.benchmarks
            for arm_id in (target.arm_ids or tuple(arm.arm_id for arm in manifest.arms))
        }
        for arm_id in selected_arm_ids:
            spec = self._config.arms[arm_id].payload_builder
            used_specs[
                (
                    "arm_payload",
                    spec.entrypoint,
                    spec.fingerprint_sha256,
                    content_hash(spec.parameters),
                )
            ] = spec
        for (kind, _entrypoint, _fingerprint, _parameters), spec in used_specs.items():
            try:
                builder = _resolve_locked_builder(spec, expected_kind=kind)
                messages = _builder_preflight(builder, spec.parameters)
            except Exception as exc:
                blockers.append(f"{kind} builder: {_safe_message(exc)}")
                continue
            blockers.extend(f"{kind} builder: {_safe_text(item)}" for item in messages)
        try:
            policy_receipts = self._resolve_policy_bindings(manifest)
        except Exception as exc:
            blockers.append("policy binding: " + _safe_message(exc))
            policy_receipts = MappingProxyType({})
        try:
            gate_receipts = self._resolve_gate_bindings(manifest)
        except Exception as exc:
            blockers.append("external gate binding: " + _safe_message(exc))
            gate_receipts = MappingProxyType({})
        if blockers:
            return tuple(blockers)
        self._policy_binding_receipts = policy_receipts
        self._gate_binding_receipts = gate_receipts
        self._preflight_manifest_hash = manifest.manifest_hash
        return ()

    def preflight_units(
        self,
        manifest: ExperimentManifest,
        resolved_units: Mapping[str, Sequence[str]],
    ) -> Sequence[str]:
        """Ask each locked checkpoint builder to recognize the exact planned IDs."""

        if not isinstance(manifest, ExperimentManifest):
            return ("unit preflight requires an ExperimentManifest",)
        requirement = manifest.metadata.get("require_driver_unit_preflight")
        if requirement is not None and not isinstance(requirement, bool):
            return ("require_driver_unit_preflight metadata must be boolean when supplied",)
        required = requirement is True
        if not isinstance(resolved_units, Mapping):
            return ("resolved unit inventory must be a mapping",)
        expected_keys = {_target_key(target) for target in manifest.benchmarks}
        if set(resolved_units) != expected_keys:
            return ("resolved unit inventories differ from the manifest targets",)
        blockers: list[str] = []
        forbidden_group_hashes = self._policy_excluded_group_hashes()
        for target in manifest.benchmarks:
            key = _target_key(target)
            binding = self._config.benchmark_targets.get(key)
            if binding is None:
                blockers.append(f"benchmark {key}: no locked external target binding")
                continue
            try:
                unit_ids = tuple(sorted(resolved_units[key]))
                request = ExternalUnitPreflightRequest(
                    benchmark_id=target.cell_id,
                    split=target.split,
                    source_revision=binding.source_revision,
                    unit_ids=unit_ids,
                    builder_parameters=binding.checkpoint_builder.parameters,
                )
                builder = _resolve_locked_builder(
                    binding.checkpoint_builder,
                    expected_kind="arena_checkpoint",
                )
                callback = getattr(builder, "preflight_units", None)
                if callback is None:
                    if required:
                        blockers.append(
                            f"benchmark {key}: checkpoint builder lacks required "
                            "read-only preflight_units"
                        )
                    continue
                if not callable(callback):
                    raise ValidationError("checkpoint builder preflight_units must be callable")
                messages = callback(request)
                if (
                    isinstance(messages, (str, bytes))
                    or not isinstance(messages, Sequence)
                    or any(not isinstance(item, str) or not item.strip() for item in messages)
                ):
                    raise ValidationError(
                        "checkpoint builder preflight_units must return a "
                        "sequence of non-empty blocker strings"
                    )
                blockers.extend(f"benchmark {key}: {_safe_text(item)}" for item in messages)
                if forbidden_group_hashes:
                    group_callback = getattr(builder, "preflight_group_ids", None)
                    if not callable(group_callback):
                        blockers.append(
                            f"benchmark {key}: checkpoint builder lacks required "
                            "read-only preflight_group_ids for selector leakage exclusion"
                        )
                        continue
                    raw_groups = group_callback(request)
                    if not isinstance(raw_groups, Mapping) or set(raw_groups) != set(unit_ids):
                        raise ValidationError(
                            "checkpoint builder preflight_group_ids must return an "
                            "exact unit_id-to-group_id mapping"
                        )
                    for unit_id, group_id in raw_groups.items():
                        _require_text(
                            group_id,
                            label=f"checkpoint group for unit {unit_id!r}",
                        )
                        group_hash = _anonymized_group_hash(
                            target.cell_id,
                            group_id,
                        )
                        if group_hash in forbidden_group_hashes:
                            blockers.append(
                                f"benchmark {key}: unit {unit_id!r} belongs to a "
                                "selector discovery train/calibration group"
                            )
            except Exception as exc:
                blockers.append(f"benchmark {key}: {_safe_message(exc)}")
        return tuple(blockers)

    def execute_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> ExperimentCellResult:
        """Execute a complete cell or raise before any terminal result is written."""

        if self._preflight_manifest_hash != manifest.manifest_hash:
            raise ArtifactError(
                "external arena driver requires successful preflight for this manifest"
            )
        if self._resolve_policy_bindings(manifest) != self._policy_binding_receipts:
            raise ArtifactError("external policy artifact identity changed after preflight")
        if self._resolve_gate_bindings(manifest) != self._gate_binding_receipts:
            raise ArtifactError("external gate artifact identity changed after preflight")
        target, target_binding = self._validate_cell(manifest, cell)
        expected_dependencies = target_binding.bridge_lock.dependencies(cell.model_slot)
        checkpoint = self._build_checkpoint(cell, target_binding)
        self._validate_checkpoint(
            checkpoint,
            cell=cell,
            binding=target_binding,
            expected_dependencies=expected_dependencies,
        )
        checkpoint_group_hash = _anonymized_group_hash(
            checkpoint.benchmark_id,
            checkpoint.group_id,
        )
        if checkpoint_group_hash in self._policy_excluded_group_hashes():
            raise ValidationError(
                "external checkpoint belongs to a selector discovery train/calibration group"
            )
        pre_action = self._pre_action_receipt(checkpoint.visible_state)
        bridge = self._build_bridge(
            checkpoint,
            cell=cell,
            binding=target_binding,
        )
        self._validate_bridge(
            bridge,
            binding=target_binding,
            expected_dependencies=expected_dependencies,
        )

        arm_specs = {arm.arm_id: arm for arm in manifest.arms}
        artifacts = {
            arm_id: self._build_arm_payload(
                checkpoint,
                cell=cell,
                arm_spec=arm_specs[arm_id],
                pre_action=pre_action,
            )
            for arm_id in cell.arm_ids
        }
        self._validate_artifact_set(
            cell,
            checkpoint=checkpoint,
            arm_specs=arm_specs,
            artifacts=artifacts,
        )
        baseline_id = self._config.baseline_arm_id
        baseline_artifact = artifacts[baseline_id]
        pair_records: list[tuple[str, AtomicReplayRequest, AtomicPairedReplayRecord]] = []
        baseline_consistency: str | None = None
        for treatment_id in cell.arm_ids:
            if treatment_id == baseline_id:
                continue
            request = AtomicReplayRequest(
                benchmark_id=cell.benchmark_id,
                checkpoint_id=cell.unit_id,
                source_revision=target_binding.source_revision,
                origin_state_hash=checkpoint.origin_state_hash,
                replay_identity=checkpoint.replay_identity,
                seed=cell.seed,
                locked_dependency_hashes=expected_dependencies,
                arms=(
                    AtomicReplayArmRequest(
                        role="baseline",
                        arm_id=baseline_id,
                        payload=baseline_artifact.payload,
                    ),
                    AtomicReplayArmRequest(
                        role="treatment",
                        arm_id=treatment_id,
                        payload=artifacts[treatment_id].payload,
                    ),
                ),
            )
            record = AtomicExternalReplayRunner().run(bridge, request)
            observed_consistency = _baseline_consistency_hash(record)
            if baseline_consistency is None:
                baseline_consistency = observed_consistency
            elif observed_consistency != baseline_consistency:
                raise ValidationError(
                    "atomic multi-arm NO_OP metrics, trace, usage, or logical "
                    "receipt changed across treatment pairs"
                )
            pair_records.append((treatment_id, request, record))
        if len(pair_records) != len(cell.arm_ids) - 1 or baseline_consistency is None:
            raise ValidationError("external arena driver produced a partial pair inventory")

        arms = self._arm_outcomes(
            cell,
            arm_specs=arm_specs,
            artifacts=artifacts,
            pair_records=pair_records,
            target_binding=target_binding,
        )
        if tuple(item["arm_id"] for item in arms) != cell.arm_ids:
            raise ValidationError("external arena driver produced a partial arm inventory")
        atomic_pairs = tuple(
            freeze_mapping(
                {
                    "treatment_arm_id": treatment_id,
                    "request_sha256": content_hash(request),
                    "receipt": to_jsonable(record.receipt),
                    "receipt_sha256": content_hash(record.receipt),
                    "baseline_consistency_sha256": baseline_consistency,
                    "metric_deltas": record.metric_deltas,
                    "resource_receipt_sha256s": tuple(
                        content_hash(receipt) for receipt in record.resource_receipts
                    ),
                }
            )
            for treatment_id, request, record in pair_records
        )
        resource_receipts = tuple(
            receipt
            for _treatment_id, _request, record in pair_records
            for receipt in record.resource_receipts
        )
        payload: dict[str, Any] = {
            "benchmark_id": cell.benchmark_id,
            "split": cell.split,
            "unit_id": cell.unit_id,
            "model_slot": cell.model_slot,
            "seed": cell.seed,
            "baseline_arm_id": baseline_id,
            "planned_arm_ids": cell.arm_ids,
            "checkpoint_identity": {
                "identity_sha256": checkpoint.identity_hash,
                "group_sha256": checkpoint_group_hash,
                "visible_state_sha256": checkpoint.visible_state.state_hash,
                "origin_state_sha256": checkpoint.origin_state_hash,
                "replay_identity_sha256": checkpoint.replay_identity,
                "source_revision": checkpoint.source_revision,
            },
            "pre_action": pre_action,
            "arms": arms,
            "atomic_pairs": atomic_pairs,
            "baseline_consistency_sha256": baseline_consistency,
            "observed_dependency_hashes": expected_dependencies,
            "not_applicable_dependency_slots": tuple(
                sorted(
                    slot
                    for slot, digest in expected_dependencies.items()
                    if digest == _NOT_APPLICABLE_DEPENDENCY_HASHES[slot]
                )
            ),
            "target_source_revision": target.source_revision,
            "atomic_replay_contract": ATOMIC_EXTERNAL_PAIR_CONTRACT,
        }
        gate_receipt = self._gate_binding_receipts.get(_target_key(target))
        if gate_receipt is not None:
            payload["external_gate_receipt"] = gate_receipt
        if manifest.metadata.get("selector_training_allowed") is True:
            payload["selector_record"] = {
                "schema_version": 1,
                "state": checkpoint.visible_state.canonical_dict(),
                "group_id": checkpoint.group_id,
                "eligible_families": tuple(
                    sorted(
                        {
                            arm_specs[arm_id].family.value
                            for arm_id in cell.arm_ids
                            if arm_specs[arm_id].family in _CORE_FAMILIES
                        }
                    )
                ),
            }
        return ExperimentCellResult(
            cell_id=cell.cell_id,
            status="succeeded",
            payload=payload,
            resource_receipts=resource_receipts,
        )

    def _policy_excluded_group_hashes(self) -> frozenset[str]:
        members: set[str] = set()
        for receipt in self._policy_binding_receipts.values():
            partition = receipt.get("group_partition")
            if not isinstance(partition, Mapping):
                continue
            raw = partition.get("excluded_member_hashes")
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ArtifactError("resolved policy group partition is invalid")
            for item in raw:
                if not isinstance(item, str) or _ANONYMIZED_GROUP.fullmatch(item) is None:
                    raise ArtifactError("resolved policy group member hash is invalid")
                members.add(item)
        return frozenset(members)

    def _resolve_policy_bindings(
        self,
        manifest: ExperimentManifest,
    ) -> Mapping[str, Mapping[str, Any]]:
        receipts: dict[str, Mapping[str, Any]] = {}
        configured_releases = {
            binding.release_id: binding.family
            for binding in self._config.arms.values()
            if binding.arm_kind is ExperimentArmKind.RELEASE
        }
        for arm in manifest.arms:
            if arm.arm_kind is not ExperimentArmKind.POLICY:
                continue
            binding = self._config.arms[arm.arm_id]
            parameters = dict(arm.parameters)
            common_fields = {
                "policy_kind",
                "policy_manifest_env",
                "policy_manifest_sha256_env",
            }
            selector_fields = {
                "selector_identity_env",
                "selector_bundle_env",
                "selector_bundle_sha256_env",
            }
            required = (
                common_fields | selector_fields
                if binding.policy_kind == "frozen_selector"
                else common_fields
            )
            if set(parameters) != required:
                raise ValidationError(f"policy arm {arm.arm_id!r} binding fields are invalid")
            if parameters["policy_kind"] != binding.policy_kind:
                raise ValidationError(
                    f"policy arm {arm.arm_id!r} kind differs from the locked config"
                )
            policy_bytes, policy_file_sha256 = _read_manifest_file_binding(
                manifest,
                environment_name=parameters["policy_manifest_env"],
                checksum_environment_name=parameters["policy_manifest_sha256_env"],
                label=f"policy arm {arm.arm_id!r} manifest",
            )
            policy_manifest = _SelectorPolicyManifest.from_mapping(
                _strict_json(policy_bytes),
                path=f"policy arm {arm.arm_id!r} manifest",
            )
            for family, release_id in policy_manifest.representative_by_family.items():
                if configured_releases.get(release_id) is not family:
                    raise ValidationError(
                        f"policy arm {arm.arm_id!r} representative {release_id!r} "
                        "is outside the locked release bindings"
                    )
            receipt: dict[str, Any] = {
                "policy_kind": binding.policy_kind,
                "policy_manifest_file_sha256": policy_file_sha256,
                "policy_manifest_hash": policy_manifest.manifest_hash,
                "discovery": {
                    "revision": policy_manifest.discovery_revision,
                    "sha256": policy_manifest.discovery_sha256,
                },
                "training_table_sha256": policy_manifest.training_table_sha256,
                "split_provenance_hash": policy_manifest.split_provenance_hash,
                "comparison_profile_hash": (policy_manifest.comparison_profile_hash),
                "group_partition": {
                    **dict(policy_manifest.group_partition.to_receipt()),
                    "excluded_member_hashes": tuple(
                        sorted(policy_manifest.group_partition.excluded_members)
                    ),
                },
                "representative_by_family": {
                    family.value: release_id
                    for family, release_id in sorted(
                        policy_manifest.representative_by_family.items(),
                        key=lambda item: item[0].value,
                    )
                },
            }
            if binding.policy_kind == "locked_best_eligible_global":
                receipt["selected_representative"] = {
                    "family": policy_manifest.selected_family.value,
                    "release_id": policy_manifest.selected_release_id,
                    "calibration_split_digest": (policy_manifest.calibration_split_digest),
                }
            else:
                bundle_bytes, bundle_file_sha256 = _read_manifest_file_binding(
                    manifest,
                    environment_name=parameters["selector_bundle_env"],
                    checksum_environment_name=parameters["selector_bundle_sha256_env"],
                    label=f"policy arm {arm.arm_id!r} selector bundle",
                )
                bundle = _load_frozen_selector_bundle(
                    bundle_bytes,
                    identity_environment_name=parameters["selector_identity_env"],
                    manifest=manifest,
                    label=f"policy arm {arm.arm_id!r} selector bundle",
                )
                _validate_selector_policy_provenance(
                    bundle,
                    policy_manifest=policy_manifest,
                    label=f"policy arm {arm.arm_id!r}",
                )
                receipt["selector_bundle"] = {
                    "file_sha256": bundle_file_sha256,
                    "selector_id": bundle.selector_card_id,
                    "revision": bundle.revision,
                    "bundle_hash": bundle.bundle_hash,
                    "model_hash": bundle.model_hash,
                    "stage": bundle.stage.value,
                }
            receipts[arm.arm_id] = freeze_mapping(receipt)
        return MappingProxyType(receipts)

    def _resolve_gate_bindings(
        self,
        manifest: ExperimentManifest,
    ) -> Mapping[str, Mapping[str, Any]]:
        receipts: dict[str, Mapping[str, Any]] = {}
        for target in manifest.benchmarks:
            gate_binding = _GATE_BINDINGS.get(target.cell_id)
            if gate_binding is None:
                continue
            environment_name, checksum_environment_name, reset_mode = gate_binding
            raw, file_sha256 = _read_manifest_file_binding(
                manifest,
                environment_name=environment_name,
                checksum_environment_name=checksum_environment_name,
                label=f"benchmark {target.cell_id!r} external gate receipt",
            )
            binding = self._config.benchmark_targets[_target_key(target)]
            inventory_sha256 = os.environ.get(target.unit_inventory_sha256_env)
            if not inventory_sha256:
                raise ArtifactError(
                    f"benchmark {target.cell_id!r} unit inventory SHA-256 is missing"
                )
            _require_sha256(
                inventory_sha256,
                label=f"benchmark {target.cell_id!r} unit inventory SHA-256",
            )
            receipt = _validate_external_gate_receipt(
                _strict_json(raw),
                benchmark_id=target.cell_id,
                split=target.split,
                source_revision=binding.source_revision,
                manifest_hash=manifest.manifest_hash,
                driver_config_sha256=self._config_sha256,
                unit_inventory_sha256=inventory_sha256,
                bridge_lock=binding.bridge_lock,
                reset_mode=reset_mode,
            )
            receipts[_target_key(target)] = freeze_mapping(
                {
                    "file_sha256": file_sha256,
                    "receipt_sha256": content_hash(receipt),
                    "scope_sha256": receipt["approval"]["scope_sha256"],
                    "benchmark_id": target.cell_id,
                    "split": target.split,
                    "source_revision": binding.source_revision,
                    "manifest_hash": manifest.manifest_hash,
                    "driver_config_sha256": self._config_sha256,
                    "bridge": receipt["bridge"],
                    "scorer": receipt["scorer"],
                    "approval": receipt["approval"],
                    "verification": receipt["verification"],
                    "unit_inventory_sha256": inventory_sha256,
                }
            )
        return MappingProxyType(receipts)

    def _manifest_blockers(self, manifest: ExperimentManifest) -> list[str]:
        if not isinstance(manifest, ExperimentManifest):
            return ["manifest must be an ExperimentManifest"]
        blockers: list[str] = []
        if manifest.estimand_profile_kind is ProfileKind.OFFICIAL_NATIVE:
            blockers.append(
                "external arena driver supports controlled_adapter or "
                "factorial_adapter estimands, not official_native"
            )
        manifest_arms = {arm.arm_id: arm for arm in manifest.arms}
        if self._config.baseline_arm_id not in manifest_arms:
            blockers.append("manifest omits the configured NO_OP baseline")
        elif manifest_arms[self._config.baseline_arm_id].family is not SemanticFamily.NO_OP:
            blockers.append("configured baseline arm is not no_op in the manifest")
        required_payload_builder = manifest.metadata.get("required_payload_builder")
        if required_payload_builder is not None and (
            not isinstance(required_payload_builder, str)
            or _ENTRYPOINT.fullmatch(required_payload_builder) is None
        ):
            blockers.append("manifest required_payload_builder is not an entrypoint")
        for arm_id, arm in manifest_arms.items():
            binding = self._config.arms.get(arm_id)
            if binding is None:
                blockers.append(f"arm {arm_id}: no locked payload binding")
                continue
            if (
                binding.release_id != arm.release_id
                or binding.family is not arm.family
                or binding.profile_kind is not arm.profile_kind
                or binding.arm_kind is not arm.arm_kind
            ):
                blockers.append(
                    f"arm {arm_id}: manifest identity differs from locked payload binding"
                )
            elif (
                arm.arm_kind is ExperimentArmKind.POLICY
                and arm.parameters.get("policy_kind") != binding.policy_kind
            ):
                blockers.append(
                    f"arm {arm_id}: manifest policy kind differs from locked payload binding"
                )
            elif (
                arm.release_id == "baseline.no_op_cost_shadow"
                and required_payload_builder is not None
                and binding.payload_builder.entrypoint != required_payload_builder
            ):
                blockers.append(
                    f"arm {arm_id}: payload builder differs from the manifest-required "
                    "cost-shadow builder"
                )
        for target in manifest.benchmarks:
            key = _target_key(target)
            binding = self._config.benchmark_targets.get(key)
            if binding is None:
                blockers.append(f"benchmark {key}: no locked external target binding")
                continue
            if target.cell_id not in _SUPPORTED_ARENAS:
                blockers.append(f"benchmark {target.cell_id}: unsupported external arena")
            if target.cell_id in _GATE_BINDINGS and (
                target.external is not True
                or target.unit_inventory_env is None
                or target.unit_inventory_sha256_env is None
            ):
                blockers.append(
                    f"benchmark {target.cell_id}: gated external arena requires "
                    "external=true and a checksum-bound unit inventory"
                )
            if (
                target.source_revision is not None
                and target.source_revision != binding.source_revision
            ):
                blockers.append(
                    f"benchmark {target.cell_id}: manifest source revision differs "
                    "from the external driver lock"
                )
            selected = target.arm_ids or tuple(manifest_arms)
            if self._config.baseline_arm_id not in selected:
                blockers.append(
                    f"benchmark {target.cell_id}: selected arms omit the NO_OP baseline"
                )
            unknown = sorted(set(selected) - set(self._config.arms))
            if unknown:
                blockers.append(
                    f"benchmark {target.cell_id}: selected arms lack payload bindings: "
                    + ", ".join(unknown)
                )
            for model_slot in manifest.model_slots:
                if model_slot not in binding.bridge_lock.dependency_hashes_by_model_slot:
                    blockers.append(
                        f"benchmark {target.cell_id}: bridge lock has no model slot {model_slot}"
                    )
        return blockers

    def _validate_cell(
        self,
        manifest: ExperimentManifest,
        cell: ExperimentCell,
    ) -> tuple[ExperimentBenchmarkTarget, _TargetBinding]:
        if not isinstance(cell, ExperimentCell):
            raise TypeError("external arena driver requires ExperimentCell")
        if cell.manifest_hash != manifest.manifest_hash:
            raise ValidationError("external arena cell belongs to another manifest")
        matches = tuple(
            target
            for target in manifest.benchmarks
            if target.cell_id == cell.benchmark_id and target.split == cell.split
        )
        if len(matches) != 1:
            raise ValidationError("external arena cell has no unique manifest target")
        target = matches[0]
        expected_arms = target.arm_ids or tuple(arm.arm_id for arm in manifest.arms)
        if cell.arm_ids != expected_arms:
            raise ValidationError("external arena cell arms differ from the target")
        if cell.model_slot not in manifest.model_slots or cell.seed not in manifest.seeds:
            raise ValidationError("external arena cell model slot or seed is not frozen")
        return target, self._config.benchmark_targets[_target_key(target)]

    def _build_checkpoint(
        self,
        cell: ExperimentCell,
        binding: _TargetBinding,
    ) -> ExternalArenaCheckpoint:
        spec = binding.checkpoint_builder
        builder = _resolve_locked_builder(spec, expected_kind="arena_checkpoint")
        checkpoint = builder.build(
            ExternalCheckpointBuildRequest(
                benchmark_id=cell.benchmark_id,
                split=cell.split,
                checkpoint_id=cell.unit_id,
                source_revision=binding.source_revision,
                model_slot=cell.model_slot,
                seed=cell.seed,
                builder_parameters=spec.parameters,
            )
        )
        if not isinstance(checkpoint, ExternalArenaCheckpoint):
            raise ValidationError("arena checkpoint builder must return ExternalArenaCheckpoint")
        return checkpoint

    def _validate_checkpoint(
        self,
        checkpoint: ExternalArenaCheckpoint,
        *,
        cell: ExperimentCell,
        binding: _TargetBinding,
        expected_dependencies: Mapping[str, str],
    ) -> None:
        if (
            checkpoint.benchmark_id != cell.benchmark_id
            or checkpoint.checkpoint_id != cell.unit_id
            or checkpoint.source_revision != binding.source_revision
            or checkpoint.locked_dependency_hashes != expected_dependencies
        ):
            raise ValidationError(
                "external checkpoint identity or dependencies differ from the frozen cell"
            )

    def _build_bridge(
        self,
        checkpoint: ExternalArenaCheckpoint,
        *,
        cell: ExperimentCell,
        binding: _TargetBinding,
    ) -> AtomicExternalReplayBridge:
        spec = binding.bridge_builder
        builder = _resolve_locked_builder(spec, expected_kind="atomic_bridge")
        bridge = builder.build(
            ExternalBridgeBuildRequest(
                checkpoint=checkpoint,
                model_slot=cell.model_slot,
                seed=cell.seed,
                builder_parameters=spec.parameters,
            )
        )
        if not isinstance(bridge, AtomicExternalReplayBridge):
            raise ValidationError("atomic bridge builder must return AtomicExternalReplayBridge")
        return bridge

    def _validate_bridge(
        self,
        bridge: AtomicExternalReplayBridge,
        *,
        binding: _TargetBinding,
        expected_dependencies: Mapping[str, str],
    ) -> None:
        lock = binding.bridge_lock
        if (
            bridge.atomic_replay_contract != ATOMIC_EXTERNAL_PAIR_CONTRACT
            or bridge.benchmark_id != binding.benchmark_id
            or bridge.source_revision != binding.source_revision
            or bridge.bridge_id != lock.bridge_id
            or bridge.bridge_revision != lock.bridge_revision
            or bridge.scorer_id != lock.scorer_id
            or bridge.scorer_revision != lock.scorer_revision
            or _validated_dependency_hashes(bridge.locked_dependency_hashes, benchmark_id=binding.benchmark_id)
            != expected_dependencies
        ):
            raise ValidationError(
                "constructed atomic bridge differs from its checksum-locked identity"
            )

    def _build_arm_payload(
        self,
        checkpoint: ExternalArenaCheckpoint,
        *,
        cell: ExperimentCell,
        arm_spec: ExperimentArmSpec,
        pre_action: Mapping[str, Any],
    ) -> ExternalArmPayload:
        binding = self._config.arms[arm_spec.arm_id]
        spec = binding.payload_builder
        builder = _resolve_locked_builder(spec, expected_kind="arm_payload")
        artifact = builder.build(
            ExternalArmPayloadBuildRequest(
                checkpoint=checkpoint,
                arm_id=arm_spec.arm_id,
                release_id=arm_spec.release_id,
                family=arm_spec.family,
                profile_kind=arm_spec.profile_kind,
                arm_kind=arm_spec.arm_kind,
                model_slot=cell.model_slot,
                seed=cell.seed,
                builder_parameters=spec.parameters,
                arm_parameters=arm_spec.parameters,
                pre_action=pre_action,
            )
        )
        if not isinstance(artifact, ExternalArmPayload):
            raise ValidationError("arm payload builder must return ExternalArmPayload")
        if (
            artifact.arm_id != arm_spec.arm_id
            or artifact.release_id != arm_spec.release_id
            or artifact.family is not arm_spec.family
            or artifact.profile_kind is not arm_spec.profile_kind
        ):
            raise ValidationError(
                f"arm payload builder returned another identity for {arm_spec.arm_id!r}"
            )
        if arm_spec.arm_kind is ExperimentArmKind.POLICY:
            if artifact.selector_decision is None:
                raise ValidationError(
                    f"policy arm {arm_spec.arm_id!r} requires selector decision receipt"
                )
            if (
                artifact.selector_input_state_hash != pre_action["state_hash"]
                or artifact.selector_feature_hash != pre_action["feature_hash"]
            ):
                raise ValidationError(
                    f"policy arm {arm_spec.arm_id!r} selector input identity differs "
                    "from the visible pre_action receipt"
                )
            try:
                policy_receipt = self._policy_binding_receipts[arm_spec.arm_id]
            except KeyError as exc:  # pragma: no cover - preflight invariant
                raise ArtifactError(
                    f"policy arm {arm_spec.arm_id!r} lacks a resolved binding receipt"
                ) from exc
            if binding.policy_kind == "locked_best_eligible_global":
                if (
                    set(artifact.selector_decision)
                    != {
                        "selected_family",
                        "selected_release_id",
                        "abstained",
                        "predicted_gain",
                        "confidence",
                        "policy_manifest_sha256",
                    }
                    or artifact.selector_decision["policy_manifest_sha256"]
                    != policy_receipt["policy_manifest_file_sha256"]
                ):
                    raise ValidationError(
                        f"policy arm {arm_spec.arm_id!r} decision differs from the "
                        "independently hashed policy manifest"
                    )
                selected = policy_receipt["selected_representative"]
                decision_family = artifact.selector_decision["selected_family"]
                if decision_family != SemanticFamily.NO_OP.value and (
                    decision_family != selected["family"]
                    or artifact.selector_decision["selected_release_id"] != selected["release_id"]
                ):
                    raise ValidationError(
                        f"policy arm {arm_spec.arm_id!r} selected a release other "
                        "than the frozen global representative"
                    )
                if (
                    decision_family == SemanticFamily.NO_OP.value
                    and SemanticFamily(selected["family"]) in checkpoint.eligible_families
                ):
                    raise ValidationError(
                        f"policy arm {arm_spec.arm_id!r} abstained even though the "
                        "frozen global representative is eligible"
                    )
            elif (
                set(artifact.selector_decision)
                != {
                    "selected_family",
                    "selected_release_id",
                    "abstained",
                    "predicted_gain",
                    "confidence",
                    "selector_bundle_hash",
                }
                or artifact.selector_decision["selector_bundle_hash"]
                != policy_receipt["selector_bundle"]["bundle_hash"]
            ):
                raise ValidationError(
                    f"policy arm {arm_spec.arm_id!r} selector bundle hash differs "
                    "from the independently parsed frozen bundle"
                )
            elif artifact.selector_decision["selected_family"] != SemanticFamily.NO_OP.value:
                representatives = policy_receipt["representative_by_family"]
                if (
                    representatives.get(artifact.selector_decision["selected_family"])
                    != (artifact.selector_decision["selected_release_id"])
                ):
                    raise ValidationError(
                        f"policy arm {arm_spec.arm_id!r} selected a release outside "
                        "the frozen selector representatives"
                    )
        elif (
            artifact.selector_decision is not None
            or artifact.selector_input_state_hash is not None
            or artifact.selector_feature_hash is not None
        ):
            raise ValidationError(f"release arm {arm_spec.arm_id!r} cannot carry selector evidence")
        return artifact

    def _validate_artifact_set(
        self,
        cell: ExperimentCell,
        *,
        checkpoint: ExternalArenaCheckpoint,
        arm_specs: Mapping[str, ExperimentArmSpec],
        artifacts: Mapping[str, ExternalArmPayload],
    ) -> None:
        if set(artifacts) != set(cell.arm_ids) or len(artifacts) != len(cell.arm_ids):
            raise ValidationError("external arm payload inventory is partial")
        eligible = set(checkpoint.eligible_families)
        configured_release_families = {
            binding.release_id: binding.family
            for binding in self._config.arms.values()
            if binding.arm_kind is ExperimentArmKind.RELEASE
        }
        for arm_id in cell.arm_ids:
            arm_spec = arm_specs[arm_id]
            artifact = artifacts[arm_id]
            if (
                arm_spec.arm_kind is ExperimentArmKind.RELEASE
                and arm_spec.family in _EXECUTABLE_RELEASE_FAMILIES
                and arm_spec.family not in eligible
            ):
                raise ValidationError(
                    f"checkpoint is not eligible for arm family {arm_spec.family.value}"
                )
            decision = artifact.selector_decision
            if decision is None:
                continue
            selected_family = SemanticFamily(decision["selected_family"])
            selected_release_id = decision["selected_release_id"]
            if selected_family is not SemanticFamily.NO_OP and selected_family not in eligible:
                raise ValidationError(f"policy arm {arm_id!r} selected an ineligible family")
            expected_family = configured_release_families.get(selected_release_id)
            if expected_family is None or expected_family is not selected_family:
                raise ValidationError(
                    f"policy arm {arm_id!r} selected a release outside the locked release bindings"
                )

    def _arm_outcomes(
        self,
        cell: ExperimentCell,
        *,
        arm_specs: Mapping[str, ExperimentArmSpec],
        artifacts: Mapping[str, ExternalArmPayload],
        pair_records: Sequence[tuple[str, AtomicReplayRequest, AtomicPairedReplayRecord]],
        target_binding: _TargetBinding,
    ) -> tuple[Mapping[str, Any], ...]:
        baseline_id = self._config.baseline_arm_id
        baseline = pair_records[0][2].baseline
        receipt_hashes = tuple(
            content_hash(record.receipt) for _arm_id, _request, record in pair_records
        )
        records = {arm_id: record for arm_id, _request, record in pair_records}
        outcomes: list[Mapping[str, Any]] = []
        for arm_id in cell.arm_ids:
            spec = arm_specs[arm_id]
            artifact = artifacts[arm_id]
            if arm_id == baseline_id:
                result = baseline
                receipt_data: Mapping[str, Any] = {
                    "atomic_pair_receipt_sha256s": receipt_hashes,
                }
            else:
                result = records[arm_id].treatment
                receipt_data = {
                    "atomic_pair_receipt_sha256": content_hash(records[arm_id].receipt),
                }
            metrics, composition_receipt = _compose_valid_success(
                result.metrics,
                benchmark_id=cell.benchmark_id,
                arm_id=arm_id,
                trace_hash=result.trace_hash,
                bridge_lock=target_binding.bridge_lock,
                outcome_contract=target_binding.outcome_contract,
            )
            outcome: dict[str, Any] = {
                "arm_id": arm_id,
                "release_id": spec.release_id,
                "family": spec.family.value,
                "profile_kind": spec.profile_kind.value,
                "status": "succeeded",
                "metrics": metrics,
                "trace_hash": result.trace_hash,
                "usage": result.usage.as_dict(),
                "payload_sha256": artifact.payload_sha256,
                "valid_success_composition": composition_receipt,
                **receipt_data,
            }
            if artifact.selector_decision is not None:
                outcome["decision"] = artifact.selector_decision
                outcome["selector_decision_sha256"] = content_hash(artifact.selector_decision)
                outcome["selector_input_receipt"] = {
                    "state_hash": artifact.selector_input_state_hash,
                    "feature_hash": artifact.selector_feature_hash,
                }
            outcomes.append(freeze_mapping(outcome))
        return tuple(outcomes)

    def _pre_action_receipt(self, state: FrozenAgentState) -> Mapping[str, Any]:
        observation = self._feature_extractor.extract(state)
        features = observation.features
        return freeze_mapping(
            {
                "state_hash": observation.state_hash,
                "feature_hash": observation.feature_hash,
                "feature_schema_id": self._feature_extractor.feature_schema_id,
                "feature_schema_hash": self._feature_extractor.feature_schema_hash,
                "provenance_hash": content_hash(observation.provenance),
                "defect_scores": {
                    SemanticFamily.REFORMULATE.value: (features.reformulation_need_score),
                    SemanticFamily.DECOMPOSE.value: features.multi_step_score,
                    SemanticFamily.ACQUIRE.value: features.retrieval_need_score,
                    SemanticFamily.CLARIFY.value: features.ambiguity_score,
                    SemanticFamily.CURATE.value: features.context_pressure,
                },
            }
        )


def factory() -> ExternalArenaExperimentDriver:
    """Construct the driver from explicit path and exact-byte SHA-256 bindings."""

    path = os.environ.get(CONFIG_PATH_ENV)
    expected_sha256 = os.environ.get(CONFIG_SHA256_ENV)
    if not path or not expected_sha256:
        raise ArtifactError(
            f"set both {CONFIG_PATH_ENV} and {CONFIG_SHA256_ENV} before loading "
            "the external arena driver"
        )
    return ExternalArenaExperimentDriver.from_file(
        path,
        expected_sha256=expected_sha256,
    )


def _target_binding(value: Any, *, path: str) -> _TargetBinding:
    data = _strict_mapping(
        value,
        path=path,
        keys={
            "benchmark_id",
            "split",
            "source_revision",
            "checkpoint_builder",
            "bridge_builder",
            "bridge_lock",
            "outcome_contract",
        },
        required={
            "benchmark_id",
            "split",
            "source_revision",
            "checkpoint_builder",
            "bridge_builder",
            "bridge_lock",
            "outcome_contract",
        },
    )
    benchmark_id = _text(data["benchmark_id"], f"{path}.benchmark_id")
    source_revision = _text(data["source_revision"], f"{path}.source_revision")
    if benchmark_id not in _SUPPORTED_ARENAS:
        raise ValidationError(f"{path}.benchmark_id is not a supported external arena")
    official = {spec.cell_id: spec.source_revision for spec in interactive_benchmark_specs()}.get(
        benchmark_id
    )
    if benchmark_id == "tau3.retail":
        # Retail has its own checkpoint/provider path. Registering metadata is
        # not certification by the historical five-family interactive factory.
        official = BenchmarkRegistry.default().get(benchmark_id).source_revision
    if official is not None and source_revision != official:
        raise ValidationError(
            f"{path}.source_revision differs from the packaged official source lock"
        )
    outcome_contract = _OutcomeContract.from_mapping(
        data["outcome_contract"],
        path=f"{path}.outcome_contract",
    )
    expected_success_metric = _REGISTERED_OFFICIAL_SUCCESS_METRICS.get(benchmark_id)
    if (
        expected_success_metric is not None
        and outcome_contract.official_success_metric != expected_success_metric
    ):
        raise ValidationError(
            f"{path}.outcome_contract.official_success_metric must match the "
            f"registered adapter metric {expected_success_metric!r}"
        )
    return _TargetBinding(
        benchmark_id=benchmark_id,
        split=_text(data["split"], f"{path}.split"),
        source_revision=source_revision,
        checkpoint_builder=_LockedBuilderSpec.from_mapping(
            data["checkpoint_builder"],
            path=f"{path}.checkpoint_builder",
        ),
        bridge_builder=_LockedBuilderSpec.from_mapping(
            data["bridge_builder"],
            path=f"{path}.bridge_builder",
        ),
        bridge_lock=_BridgeIdentityLock.from_mapping(
            data["bridge_lock"],
            path=f"{path}.bridge_lock",
            benchmark_id=benchmark_id,
        ),
        outcome_contract=outcome_contract,
    )


def _arm_binding(value: Any, *, path: str) -> _ArmBinding:
    data = _strict_mapping(
        value,
        path=path,
        keys={
            "release_id",
            "family",
            "profile_kind",
            "arm_kind",
            "payload_builder",
            "policy_kind",
        },
        required={"release_id", "family", "profile_kind", "payload_builder"},
    )
    try:
        family = SemanticFamily(_text(data["family"], f"{path}.family"))
        profile_kind = ProfileKind(_text(data["profile_kind"], f"{path}.profile_kind"))
        arm_kind = ExperimentArmKind(
            _text(
                data.get("arm_kind", ExperimentArmKind.RELEASE.value),
                f"{path}.arm_kind",
            )
        )
    except ValueError as exc:
        raise ValidationError(f"{path} contains an unknown enum value") from exc
    policy_kind = data.get("policy_kind")
    if policy_kind is not None:
        policy_kind = _text(policy_kind, f"{path}.policy_kind")
    return _ArmBinding(
        release_id=_text(data["release_id"], f"{path}.release_id"),
        family=family,
        profile_kind=profile_kind,
        arm_kind=arm_kind,
        payload_builder=_LockedBuilderSpec.from_mapping(
            data["payload_builder"],
            path=f"{path}.payload_builder",
        ),
        policy_kind=policy_kind,
    )


def _resolve_locked_builder(
    spec: _LockedBuilderSpec,
    *,
    expected_kind: str,
) -> LockedExternalArenaBuilder:
    if expected_kind not in _BUILDER_KINDS:  # pragma: no cover - internal invariant
        raise ValueError("unknown external builder kind")
    factory_candidate = resolve_driver_entrypoint(spec.entrypoint)
    if not callable(factory_candidate):
        raise ArtifactError("external builder entrypoint must be a zero-argument factory")
    try:
        builder = factory_candidate()
    except TypeError as exc:
        raise ArtifactError("external builder entrypoint must be a zero-argument factory") from exc
    if not isinstance(builder, LockedExternalArenaBuilder):
        raise ArtifactError("external component does not implement the locked builder contract")
    if builder.builder_kind != expected_kind:
        raise ArtifactError(
            f"external builder kind is {builder.builder_kind!r}, expected {expected_kind!r}"
        )
    identity = _builder_fingerprint(builder, label=f"{expected_kind} builder")
    if content_hash(identity) != spec.fingerprint_sha256:
        raise ArtifactError(f"{expected_kind} builder fingerprint checksum mismatch")
    return builder


def _builder_fingerprint(
    builder: LockedExternalArenaBuilder,
    *,
    label: str,
) -> Mapping[str, Any]:
    fingerprint = builder.fingerprint()
    if not isinstance(fingerprint, Mapping) or not fingerprint:
        raise ValidationError(f"{label} fingerprint must be a non-empty mapping")
    try:
        return freeze_mapping(fingerprint)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} fingerprint must be canonical JSON") from exc


def _builder_preflight(
    builder: LockedExternalArenaBuilder,
    parameters: Mapping[str, Any],
) -> tuple[str, ...]:
    blockers = builder.preflight(parameters)
    if isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
        raise ValidationError("external component preflight must return a sequence of strings")
    normalized = tuple(blockers)
    if any(not isinstance(item, str) or not item.strip() for item in normalized):
        raise ValidationError("external component preflight blockers must be non-empty strings")
    return normalized


def _compose_valid_success(
    raw_metrics: Mapping[str, float],
    *,
    benchmark_id: str,
    arm_id: str,
    trace_hash: str,
    bridge_lock: _BridgeIdentityLock,
    outcome_contract: _OutcomeContract,
) -> tuple[Mapping[str, float], Mapping[str, Any]]:
    """Create the primary valid outcome only inside the trusted host boundary."""

    if _VALID_SUCCESS_METRIC in raw_metrics:
        raise ValidationError(f"external bridge arm {arm_id!r} must not precompute valid_success")
    required = {
        outcome_contract.official_success_metric,
        _CONTRACT_PRESERVED_METRIC,
        _PRIVILEGED_LEAKAGE_METRIC,
    }
    missing = sorted(required - set(raw_metrics))
    if missing:
        raise ValidationError(
            f"external bridge arm {arm_id!r} is missing valid-success components: "
            + ", ".join(missing)
        )
    official_success = _unit_interval_metric(
        raw_metrics[outcome_contract.official_success_metric],
        label=(
            f"external bridge arm {arm_id!r} metric {outcome_contract.official_success_metric!r}"
        ),
    )
    contract_preserved = _unit_interval_metric(
        raw_metrics[_CONTRACT_PRESERVED_METRIC],
        label=f"external bridge arm {arm_id!r} metric {_CONTRACT_PRESERVED_METRIC!r}",
    )
    privileged_leakage = _unit_interval_metric(
        raw_metrics[_PRIVILEGED_LEAKAGE_METRIC],
        label=f"external bridge arm {arm_id!r} metric {_PRIVILEGED_LEAKAGE_METRIC!r}",
    )
    valid_success = official_success * contract_preserved * (1.0 - privileged_leakage)
    receipt_body = {
        "schema_version": 1,
        "formula_id": outcome_contract.formula_id,
        "benchmark_id": benchmark_id,
        "arm_id": arm_id,
        "official_success_metric": outcome_contract.official_success_metric,
        "components": {
            "official_success": official_success,
            _CONTRACT_PRESERVED_METRIC: contract_preserved,
            _PRIVILEGED_LEAKAGE_METRIC: privileged_leakage,
        },
        _VALID_SUCCESS_METRIC: valid_success,
        "outcome_contract_sha256": content_hash(outcome_contract),
        "scorer_id": bridge_lock.scorer_id,
        "scorer_revision": bridge_lock.scorer_revision,
        "trace_hash": trace_hash,
    }
    composition_receipt = freeze_mapping(
        {
            **receipt_body,
            "composition_receipt_sha256": content_hash(receipt_body),
        }
    )
    metrics = dict(raw_metrics)
    metrics[_VALID_SUCCESS_METRIC] = valid_success
    return freeze_mapping(metrics), composition_receipt


def _unit_interval_metric(value: Any, *, label: str) -> float:
    numeric = _finite_number(value, label=label)
    if not 0.0 <= numeric <= 1.0:
        raise ValidationError(f"{label} must be in [0, 1]")
    return numeric


def _baseline_consistency_hash(record: AtomicPairedReplayRecord) -> str:
    baseline_receipts = tuple(arm for arm in record.receipt.arms if arm.role == "baseline")
    if len(baseline_receipts) != 1:  # pragma: no cover - typed receipt invariant
        raise ValidationError("atomic pair lacks one baseline receipt")
    baseline_receipt = baseline_receipts[0]
    return content_hash(
        {
            "metrics": record.baseline.metrics,
            "trace_hash": record.baseline.trace_hash,
            "usage": record.baseline.usage,
            "receipt": {
                "benchmark_id": record.receipt.benchmark_id,
                "checkpoint_id": record.receipt.checkpoint_id,
                "source_revision": record.receipt.source_revision,
                "bridge_id": record.receipt.bridge_id,
                "bridge_revision": record.receipt.bridge_revision,
                "scorer_id": record.receipt.scorer_id,
                "scorer_revision": record.receipt.scorer_revision,
                "origin_state_hash": record.receipt.origin_state_hash,
                "replay_identity": record.receipt.replay_identity,
                "seed": record.receipt.seed,
                "locked_dependency_hashes": (record.receipt.locked_dependency_hashes),
                "baseline_arm": {
                    "role": baseline_receipt.role,
                    "arm_id": baseline_receipt.arm_id,
                    "origin_state_hash": baseline_receipt.origin_state_hash,
                    "replay_identity": baseline_receipt.replay_identity,
                    "seed": baseline_receipt.seed,
                    "locked_dependency_hashes": (baseline_receipt.locked_dependency_hashes),
                    "completed": baseline_receipt.completed,
                },
            },
        }
    )


def _validated_policy_decision(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("selector decision receipt must be an object")
    common = {
        "selected_family",
        "selected_release_id",
        "abstained",
        "predicted_gain",
        "confidence",
    }
    identity_fields = set(value) - common
    if identity_fields not in (
        {"selector_bundle_hash"},
        {"policy_manifest_sha256"},
    ):
        raise ValidationError("selector decision receipt fields are invalid")
    try:
        family = SemanticFamily(value["selected_family"])
    except (TypeError, ValueError) as exc:
        raise ValidationError("selector decision selected_family is invalid") from exc
    if family not in {*_CORE_FAMILIES, SemanticFamily.NO_OP}:
        raise ValidationError("selector decision selected_family is not executable")
    release_id = value["selected_release_id"]
    _require_text(release_id, label="selector decision selected_release_id")
    abstained = value["abstained"]
    if not isinstance(abstained, bool):
        raise ValidationError("selector decision abstained must be boolean")
    if abstained is not (family is SemanticFamily.NO_OP):
        raise ValidationError("selector decision abstention and selected_family are inconsistent")
    if family is SemanticFamily.NO_OP and release_id != "baseline.no_op":
        raise ValidationError("selector NO_OP decision must select baseline.no_op")
    predicted_gain = _finite_number(
        value["predicted_gain"],
        label="selector decision predicted_gain",
    )
    confidence = _finite_number(
        value["confidence"],
        label="selector decision confidence",
    )
    if not 0 <= confidence <= 1:
        raise ValidationError("selector decision confidence must be in [0, 1]")
    identity_field = next(iter(identity_fields))
    artifact_hash = value[identity_field]
    _require_sha256(
        artifact_hash,
        label=f"selector decision {identity_field}",
    )
    return freeze_mapping(
        {
            "selected_family": family.value,
            "selected_release_id": release_id,
            "abstained": abstained,
            "predicted_gain": predicted_gain,
            "confidence": confidence,
            identity_field: artifact_hash,
        }
    )


def _read_manifest_file_binding(
    manifest: ExperimentManifest,
    *,
    environment_name: Any,
    checksum_environment_name: Any,
    label: str,
) -> tuple[bytes, str]:
    for value, field_label in (
        (environment_name, "file environment"),
        (checksum_environment_name, "checksum environment"),
    ):
        if not isinstance(value, str) or _ENVIRONMENT_NAME.fullmatch(value) is None:
            raise ValidationError(f"{label} {field_label} is invalid")
    matches = tuple(
        prerequisite
        for prerequisite in manifest.prerequisites
        if prerequisite.kind is PrerequisiteKind.FILE_ENV
        and prerequisite.name == environment_name
        and prerequisite.sha256_env == checksum_environment_name
    )
    if len(matches) != 1:
        raise ValidationError(
            f"{label} is not declared as one exact checksum-bound FILE_ENV prerequisite"
        )
    raw_path = os.environ.get(environment_name)
    expected_sha256 = os.environ.get(checksum_environment_name)
    if not raw_path or not expected_sha256:
        raise ArtifactError(f"{label} environment binding is missing")
    _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
    source = Path(raw_path).expanduser()
    try:
        if not source.is_file():
            raise ArtifactError(f"{label} does not identify an existing file")
        if source.stat().st_size > _MAX_POLICY_ARTIFACT_BYTES:
            raise ArtifactError(f"{label} exceeds the size limit")
        raw = source.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"{label} cannot be read") from exc
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ArtifactError(f"{label} checksum differs from its declared checksum")
    return raw, observed_sha256


def _load_frozen_selector_bundle(
    raw: bytes,
    *,
    identity_environment_name: Any,
    manifest: ExperimentManifest,
    label: str,
) -> SelectorArtifactBundle:
    if (
        not isinstance(identity_environment_name, str)
        or _ENVIRONMENT_NAME.fullmatch(identity_environment_name) is None
    ):
        raise ValidationError(f"{label} identity environment is invalid")
    identity_prerequisites = tuple(
        prerequisite
        for prerequisite in manifest.prerequisites
        if prerequisite.kind is PrerequisiteKind.ENV
        and prerequisite.name == identity_environment_name
    )
    if len(identity_prerequisites) != 1:
        raise ValidationError(f"{label} identity is not declared as one exact ENV prerequisite")
    identity_raw = os.environ.get(identity_environment_name)
    if not identity_raw:
        raise ArtifactError(f"{label} identity environment is missing")
    identity = _strict_mapping(
        _strict_json(identity_raw.encode("utf-8")),
        path=f"{label} identity",
        keys={"selector_id", "revision", "sha256"},
        required={"selector_id", "revision", "sha256"},
    )
    try:
        bundle = SelectorArtifactBundle.from_artifact(_strict_json(raw))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} failed strict SelectorArtifactBundle parsing") from exc
    if bundle.stage is not SelectorArtifactStage.FROZEN:
        raise ArtifactError(f"{label} stage must be frozen")
    selector_id = _text(identity["selector_id"], f"{label} identity.selector_id")
    revision = _text(identity["revision"], f"{label} identity.revision")
    sha256 = _text(identity["sha256"], f"{label} identity.sha256")
    _require_sha256(sha256, label=f"{label} identity.sha256")
    if (
        selector_id != bundle.selector_card_id
        or revision != bundle.revision
        or sha256 != bundle.bundle_hash
    ):
        raise ArtifactError(f"{label} identity differs from its parsed frozen bundle")
    return bundle


def _validate_selector_policy_provenance(
    bundle: SelectorArtifactBundle,
    *,
    policy_manifest: _SelectorPolicyManifest,
    label: str,
) -> None:
    expected = {
        "training_table_sha256": policy_manifest.training_table_sha256,
        "split_provenance_hash": policy_manifest.split_provenance_hash,
        "comparison_profile_hash": policy_manifest.comparison_profile_hash,
        "representative_by_family": dict(policy_manifest.representative_by_family),
    }
    if any(getattr(bundle, field) != value for field, value in expected.items()):
        raise ValidationError(
            f"{label} selector bundle differs from policy-manifest "
            "training/profile/representative provenance"
        )
    discovery = bundle.fit_lineage.get("discovery")
    if (
        not isinstance(discovery, Mapping)
        or discovery.get("revision") != policy_manifest.discovery_revision
        or discovery.get("sha256") != policy_manifest.discovery_sha256
    ):
        raise ValidationError(
            f"{label} selector bundle fit lineage differs from policy-manifest discovery"
        )
    if bundle.calibration_receipt.get("discovery_hash") != policy_manifest.discovery_sha256:
        raise ValidationError(
            f"{label} selector calibration receipt differs from policy-manifest discovery"
        )


def _policy_representatives_from_mapping(
    value: Any,
    *,
    path: str,
) -> Mapping[SemanticFamily, str]:
    raw = _nonempty_object(value, path=path)
    representatives: dict[SemanticFamily, str] = {}
    for name, release_id in raw.items():
        try:
            family = SemanticFamily(_text(name, f"{path} family"))
        except ValueError as exc:
            raise ValidationError(f"{path} contains an unknown family") from exc
        representatives[family] = _text(
            release_id,
            f"{path}[{name!r}]",
        )
    return _validated_policy_representatives(representatives)


def _validated_policy_representatives(
    value: Mapping[SemanticFamily, str],
) -> dict[SemanticFamily, str]:
    representatives = dict(value)
    if (
        not representatives
        or any(
            family not in _CORE_FAMILIES
            or not isinstance(release_id, str)
            or not release_id.strip()
            or release_id != release_id.strip()
            for family, release_id in representatives.items()
        )
        or len(set(representatives.values())) != len(representatives)
    ):
        raise ValidationError("policy manifest representatives require unique core-family releases")
    return representatives


def _validate_external_gate_receipt(
    value: Mapping[str, Any],
    *,
    benchmark_id: str,
    split: str,
    source_revision: str,
    manifest_hash: str,
    driver_config_sha256: str,
    unit_inventory_sha256: str,
    bridge_lock: _BridgeIdentityLock,
    reset_mode: str,
) -> Mapping[str, Any]:
    path = f"external gate receipt for {benchmark_id}"
    data = _strict_mapping(
        value,
        path=path,
        keys={
            "schema_version",
            "artifact",
            "status",
            "benchmark_id",
            "split",
            "source_revision",
            "manifest_hash",
            "driver_config_sha256",
            "unit_inventory_sha256",
            "bridge",
            "scorer",
            "verification",
            "approval",
        },
        required={
            "schema_version",
            "artifact",
            "status",
            "benchmark_id",
            "split",
            "source_revision",
            "manifest_hash",
            "driver_config_sha256",
            "unit_inventory_sha256",
            "bridge",
            "scorer",
            "verification",
            "approval",
        },
    )
    if (
        data["schema_version"] != 1
        or data["artifact"] != "agentinputlab.external_arena_gate_receipt"
        or data["status"] != "approved"
    ):
        raise ValidationError(f"{path} is not an approved schema-v1 gate receipt")
    expected_identity = {
        "benchmark_id": benchmark_id,
        "split": split,
        "source_revision": source_revision,
        "manifest_hash": manifest_hash,
        "driver_config_sha256": driver_config_sha256,
        "unit_inventory_sha256": unit_inventory_sha256,
    }
    for name in ("manifest_hash", "driver_config_sha256", "unit_inventory_sha256"):
        _require_sha256(expected_identity[name], label=f"expected {name}")
    for name, expected in expected_identity.items():
        if _text(data[name], f"{path}.{name}") != expected:
            raise ValidationError(f"{path}.{name} differs from the frozen experiment")
    bridge = _strict_mapping(
        data["bridge"],
        path=f"{path}.bridge",
        keys={"bridge_id", "bridge_revision"},
        required={"bridge_id", "bridge_revision"},
    )
    scorer = _strict_mapping(
        data["scorer"],
        path=f"{path}.scorer",
        keys={"scorer_id", "scorer_revision"},
        required={"scorer_id", "scorer_revision"},
    )
    if (
        bridge.get("bridge_id") != bridge_lock.bridge_id
        or bridge.get("bridge_revision") != bridge_lock.bridge_revision
    ):
        raise ValidationError(f"{path}.bridge differs from the external driver lock")
    if (
        scorer.get("scorer_id") != bridge_lock.scorer_id
        or scorer.get("scorer_revision") != bridge_lock.scorer_revision
    ):
        raise ValidationError(f"{path}.scorer differs from the external driver lock")
    verification = _strict_mapping(
        data["verification"],
        path=f"{path}.verification",
        keys={
            "atomic_replay_contract",
            "reset_mode",
            "reset_verified",
            "reset_receipt_sha256",
            "no_op_parity_verified",
            "no_op_parity_receipt_sha256",
        },
        required={
            "atomic_replay_contract",
            "reset_mode",
            "reset_verified",
            "reset_receipt_sha256",
            "no_op_parity_verified",
            "no_op_parity_receipt_sha256",
        },
    )
    if (
        verification["atomic_replay_contract"] != ATOMIC_EXTERNAL_PAIR_CONTRACT
        or verification["reset_mode"] != reset_mode
        or verification["reset_verified"] is not True
        or verification["no_op_parity_verified"] is not True
    ):
        raise ValidationError(f"{path}.verification does not approve atomic reset and NO_OP parity")
    for name in ("reset_receipt_sha256", "no_op_parity_receipt_sha256"):
        _require_sha256(
            verification[name],
            label=f"{path}.verification.{name}",
        )
    approval = _strict_mapping(
        data["approval"],
        path=f"{path}.approval",
        keys={
            "approved",
            "approver_id",
            "approval_revision",
            "approved_at",
            "scope_sha256",
        },
        required={
            "approved",
            "approver_id",
            "approval_revision",
            "approved_at",
            "scope_sha256",
        },
    )
    if approval["approved"] is not True:
        raise ValidationError(f"{path}.approval.approved must be true")
    _require_text(approval["approver_id"], label=f"{path}.approval.approver_id")
    _require_text(approval["approved_at"], label=f"{path}.approval.approved_at")
    _require_sha256(
        approval["approval_revision"],
        label=f"{path}.approval.approval_revision",
    )
    _require_sha256(
        approval["scope_sha256"],
        label=f"{path}.approval.scope_sha256",
    )
    scope = {
        **expected_identity,
        "bridge": dict(bridge),
        "scorer": dict(scorer),
        "verification": dict(verification),
    }
    if approval["scope_sha256"] != content_hash(scope):
        raise ValidationError(f"{path}.approval.scope_sha256 differs from the approved scope")
    return freeze_mapping(data)


def _validated_dependency_hashes(
    value: Mapping[str, str],
    *, benchmark_id: str = "",
) -> Mapping[str, str]:
    # The common user simulator and NL judge are part of Retail's scientific
    # identity. Other arenas retain their existing exact ten-slot contract.
    required = _REQUIRED_SHARED_DEPENDENCY_SLOTS
    if benchmark_id == "tau3.retail":
        required = required | _RETAIL_ENVIRONMENT_DEPENDENCY_SLOTS
    if not isinstance(value, Mapping):
        raise ValidationError(
            "external locked dependency hashes must be an object containing "
            "the exact shared dependency slots"
        )
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise ValidationError(
            "external locked dependency hashes must contain the exact shared "
            f"dependency slots; missing={missing}, extra={extra}"
        )
    normalized: dict[str, str] = {}
    for name, digest in value.items():
        _require_text(name, label="external dependency name")
        _require_sha256(digest, label=f"external dependency {name!r}")
        normalized[name] = digest
    return freeze_mapping(normalized)


def _anonymized_group_hash(benchmark_id: str, raw_group_id: str) -> str:
    _require_text(benchmark_id, label="group benchmark_id")
    _require_text(raw_group_id, label="group_id")
    return "group_" + content_hash({"benchmark_id": benchmark_id, "group_id": raw_group_id})


def external_dependency_not_applicable_hash(slot: str) -> str:
    """Return the canonical sentinel digest for an inapplicable shared slot."""

    if slot not in _REQUIRED_SHARED_DEPENDENCY_SLOTS:
        raise ValidationError(f"unknown external dependency slot: {slot!r}")
    return _NOT_APPLICABLE_DEPENDENCY_HASHES[slot]


def _target_key(target: ExperimentBenchmarkTarget) -> str:
    return f"{target.cell_id}:{target.split}"


def _strict_json(raw: bytes) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _reject_json_constant(value),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("external arena config is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValidationError("external arena config root must be an object")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _strict_mapping(
    value: Any,
    *,
    path: str,
    keys: set[str],
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{path} keys must be strings")
    unknown = set(value) - keys
    missing = required - set(value)
    if unknown:
        raise ValidationError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValidationError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return value


def _nonempty_object(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError(f"{path} must be a non-empty object")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValidationError(f"{path} keys must be non-empty strings")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{path} must be trimmed non-empty text")
    return value


def _require_text(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(not character.isprintable() for character in value)
    ):
        raise ValidationError(f"{label} must be trimmed non-empty printable text")


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")


def _require_seed(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("external arena seed must be a non-negative integer")


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{label} must be finite numeric")
    return float(value)


def _safe_message(error: Exception) -> str:
    return f"{type(error).__name__}: {_safe_text(str(error))}"


def _safe_text(value: str) -> str:
    result = value[:2048]
    for pattern in _SECRET_TEXT_PATTERNS:
        result = pattern.sub("<redacted>", result)
    return result


__all__ = [
    "CONFIG_PATH_ENV",
    "CONFIG_SHA256_ENV",
    "ExternalArenaCheckpoint",
    "ExternalArenaExperimentDriver",
    "ExternalArmPayload",
    "ExternalArmPayloadBuildRequest",
    "ExternalBridgeBuildRequest",
    "ExternalCheckpointBuildRequest",
    "ExternalUnitPreflightRequest",
    "LockedExternalArenaBuilder",
    "factory",
]
