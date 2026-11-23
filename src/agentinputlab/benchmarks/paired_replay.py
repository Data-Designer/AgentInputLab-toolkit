"""Generic fail-closed contracts for atomic external paired replay.

Stateful arenas cannot establish a fair pair by scoring two callbacks
sequentially.  The external runtime must own one atomic operation that restores
the same protected origin twice, runs both arms in independent runtime
instances, invokes the locked scorer, and returns one complete receipt.

This module defines that transport-neutral boundary.  It does not certify any
of the four interactive arenas as pairing-ready; each still needs a separately
verified upstream runtime bridge.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import deep_freeze_jsonable, freeze_mapping
from agentinputlab.observability import ResourceReceipt, ResourceUsage

ATOMIC_EXTERNAL_PAIR_CONTRACT = "agentinputlab.atomic_external_pair.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLES = ("baseline", "treatment")


@dataclass(frozen=True, slots=True)
class AtomicReplayArmRequest:
    """One canonically serializable intervention artifact for external replay."""

    role: str
    arm_id: str
    payload: Any = field(repr=False)

    def __post_init__(self) -> None:
        _require_role(self.role)
        _require_text(self.arm_id, label="atomic replay arm_id")
        try:
            object.__setattr__(self, "payload", deep_freeze_jsonable(self.payload))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"atomic replay arm payload must be canonically serializable: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class AtomicReplayRequest:
    """Complete two-arm request bound to one protected checkpoint origin."""

    benchmark_id: str
    checkpoint_id: str
    source_revision: str
    origin_state_hash: str
    replay_identity: str
    seed: int
    locked_dependency_hashes: Mapping[str, str]
    arms: tuple[AtomicReplayArmRequest, ...]

    def __post_init__(self) -> None:
        for label in ("benchmark_id", "checkpoint_id", "source_revision"):
            _require_text(getattr(self, label), label=f"atomic replay {label}")
        _require_sha256(self.origin_state_hash, label="atomic replay origin_state_hash")
        _require_sha256(self.replay_identity, label="atomic replay replay_identity")
        _require_seed(self.seed)
        dependencies = _validated_dependency_hashes(self.locked_dependency_hashes)
        arms = tuple(self.arms)
        _require_arm_types(
            arms,
            AtomicReplayArmRequest,
            label="atomic replay request",
        )
        _require_complete_arms(arms, label="atomic replay request")
        object.__setattr__(self, "locked_dependency_hashes", dependencies)
        object.__setattr__(self, "arms", arms)


@dataclass(frozen=True, slots=True)
class AtomicReplayArmReceipt:
    """Per-arm proof fields emitted by the external runtime."""

    role: str
    arm_id: str
    origin_state_hash: str
    replay_identity: str
    seed: int
    locked_dependency_hashes: Mapping[str, str]
    runtime_instance_id: str
    completed: bool

    def __post_init__(self) -> None:
        _require_role(self.role)
        _require_text(self.arm_id, label="atomic replay receipt arm_id")
        _require_sha256(
            self.origin_state_hash,
            label="atomic replay receipt origin_state_hash",
        )
        _require_sha256(
            self.replay_identity,
            label="atomic replay receipt replay_identity",
        )
        _require_seed(self.seed)
        _require_sha256(
            self.runtime_instance_id,
            label="atomic replay receipt runtime_instance_id",
        )
        if self.completed is not True:
            raise ValidationError("atomic replay receipt requires a completed arm")
        object.__setattr__(
            self,
            "locked_dependency_hashes",
            _validated_dependency_hashes(self.locked_dependency_hashes),
        )


@dataclass(frozen=True, slots=True)
class AtomicReplayReceipt:
    """One indivisible receipt covering both restored runtime executions."""

    benchmark_id: str
    checkpoint_id: str
    source_revision: str
    bridge_id: str
    bridge_revision: str
    scorer_id: str
    scorer_revision: str
    origin_state_hash: str
    replay_identity: str
    seed: int
    locked_dependency_hashes: Mapping[str, str]
    arms: tuple[AtomicReplayArmReceipt, ...]

    def __post_init__(self) -> None:
        for label in (
            "benchmark_id",
            "checkpoint_id",
            "source_revision",
            "bridge_id",
            "scorer_id",
        ):
            _require_text(getattr(self, label), label=f"atomic replay receipt {label}")
        for label in ("bridge_revision", "scorer_revision", "origin_state_hash"):
            _require_sha256(
                getattr(self, label),
                label=f"atomic replay receipt {label}",
            )
        _require_sha256(
            self.replay_identity,
            label="atomic replay receipt replay_identity",
        )
        _require_seed(self.seed)
        dependencies = _validated_dependency_hashes(self.locked_dependency_hashes)
        arms = tuple(self.arms)
        _require_arm_types(
            arms,
            AtomicReplayArmReceipt,
            label="atomic replay receipt",
        )
        _require_complete_arms(arms, label="atomic replay receipt")
        if len({arm.runtime_instance_id for arm in arms}) != 2:
            raise ValidationError(
                "atomic replay arms must use different runtime instance identities"
            )
        for arm in arms:
            if (
                arm.origin_state_hash != self.origin_state_hash
                or arm.replay_identity != self.replay_identity
                or arm.seed != self.seed
                or arm.locked_dependency_hashes != dependencies
            ):
                raise ValidationError(
                    "atomic replay arm receipt differs from the shared origin, seed, "
                    "replay identity, or locked dependencies"
                )
        object.__setattr__(self, "locked_dependency_hashes", dependencies)
        object.__setattr__(self, "arms", arms)


@dataclass(frozen=True, slots=True)
class AtomicReplayArmResult:
    """Official scorer output and execution accounting for one arm."""

    role: str
    arm_id: str
    metrics: Mapping[str, float]
    trace_hash: str
    usage: ResourceUsage = field(default_factory=ResourceUsage)

    def __post_init__(self) -> None:
        _require_role(self.role)
        _require_text(self.arm_id, label="atomic replay result arm_id")
        _require_sha256(self.trace_hash, label="atomic replay result trace_hash")
        if not isinstance(self.usage, ResourceUsage):
            raise TypeError("atomic replay result usage must be ResourceUsage")
        object.__setattr__(self, "metrics", _validated_metrics(self.metrics))


@dataclass(frozen=True, slots=True)
class AtomicReplayPairResult:
    """Typed all-or-nothing result returned by an atomic bridge call."""

    receipt: AtomicReplayReceipt
    arms: tuple[AtomicReplayArmResult, ...]
    resource_receipts: tuple[ResourceReceipt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, AtomicReplayReceipt):
            raise TypeError("atomic replay pair result requires AtomicReplayReceipt")
        arms = tuple(self.arms)
        _require_arm_types(
            arms,
            AtomicReplayArmResult,
            label="atomic replay pair result",
        )
        _require_complete_arms(arms, label="atomic replay pair result")
        expected = {(arm.role, arm.arm_id) for arm in self.receipt.arms}
        actual = {(arm.role, arm.arm_id) for arm in arms}
        if actual != expected:
            raise ValidationError(
                "atomic replay result arms differ from the completed receipt arms"
            )
        metric_keys = tuple(sorted(arms[0].metrics))
        if tuple(sorted(arms[1].metrics)) != metric_keys:
            raise ValidationError("atomic replay arm metric schemas differ")
        resource_receipts = tuple(self.resource_receipts)
        if any(not isinstance(item, ResourceReceipt) for item in resource_receipts):
            raise TypeError(
                "atomic replay pair resource_receipts must contain ResourceReceipt values"
            )
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "resource_receipts", resource_receipts)


@runtime_checkable
class AtomicExternalReplayBridge(Protocol):
    """Distinct protocol that a legacy single-arm bridge cannot satisfy."""

    atomic_replay_contract: str
    benchmark_id: str
    source_revision: str
    bridge_id: str
    bridge_revision: str
    scorer_id: str
    scorer_revision: str
    locked_dependency_hashes: Mapping[str, str]

    def evaluate_pair(self, request: AtomicReplayRequest) -> AtomicReplayPairResult: ...


@dataclass(frozen=True, slots=True)
class AtomicPairedReplayRecord:
    """Validated paired result with evaluator-derived metric deltas."""

    receipt: AtomicReplayReceipt
    baseline: AtomicReplayArmResult
    treatment: AtomicReplayArmResult
    metric_deltas: Mapping[str, float]
    resource_receipts: tuple[ResourceReceipt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, AtomicReplayReceipt):
            raise TypeError("atomic paired replay record requires AtomicReplayReceipt")
        if not isinstance(self.baseline, AtomicReplayArmResult) or not isinstance(
            self.treatment,
            AtomicReplayArmResult,
        ):
            raise TypeError("atomic paired replay record requires typed arm results")
        if self.baseline.role != "baseline" or self.treatment.role != "treatment":
            raise ValidationError("atomic paired replay record has incorrect arm roles")
        receipt_arms = {(arm.role, arm.arm_id) for arm in self.receipt.arms}
        result_arms = {
            (self.baseline.role, self.baseline.arm_id),
            (self.treatment.role, self.treatment.arm_id),
        }
        if result_arms != receipt_arms:
            raise ValidationError(
                "atomic paired replay record arms differ from the completed receipt"
            )
        expected = tuple(sorted(self.baseline.metrics))
        if tuple(sorted(self.treatment.metrics)) != expected:
            raise ValidationError("atomic paired replay metric schemas differ")
        deltas = _validated_metrics(self.metric_deltas)
        if tuple(sorted(deltas)) != expected:
            raise ValidationError("atomic paired replay delta schema differs from arm metrics")
        for key in expected:
            if deltas[key] != self.treatment.metrics[key] - self.baseline.metrics[key]:
                raise ValidationError(f"atomic paired replay metric delta {key!r} is inconsistent")
        resource_receipts = tuple(self.resource_receipts)
        if any(not isinstance(item, ResourceReceipt) for item in resource_receipts):
            raise TypeError(
                "atomic paired replay resource_receipts must contain ResourceReceipt values"
            )
        object.__setattr__(self, "metric_deltas", deltas)
        object.__setattr__(self, "resource_receipts", resource_receipts)


class AtomicExternalReplayRunner:
    """Call a verified atomic bridge exactly once and validate its full receipt."""

    def run(
        self,
        bridge: AtomicExternalReplayBridge,
        request: AtomicReplayRequest,
    ) -> AtomicPairedReplayRecord:
        if not isinstance(request, AtomicReplayRequest):
            raise TypeError("atomic replay request must be AtomicReplayRequest")
        if not isinstance(bridge, AtomicExternalReplayBridge):
            raise TypeError(
                "bridge must implement AtomicExternalReplayBridge.evaluate_pair(); "
                "single-arm evaluation bridges are not accepted"
            )
        bridge_dependencies = _validate_bridge(bridge)
        if (
            bridge.benchmark_id != request.benchmark_id
            or bridge.source_revision != request.source_revision
            or bridge_dependencies != request.locked_dependency_hashes
        ):
            raise ValidationError(
                "atomic bridge identity or locked dependencies differ from the request"
            )

        result = bridge.evaluate_pair(request)
        if not isinstance(result, AtomicReplayPairResult):
            raise ValidationError(
                "atomic external bridge must return AtomicReplayPairResult; "
                "partial or bare metric results are not accepted"
            )
        _validate_receipt_against_request(result.receipt, request, bridge)
        by_role = {arm.role: arm for arm in result.arms}
        baseline = by_role["baseline"]
        treatment = by_role["treatment"]
        resource_receipts = result.resource_receipts
        bridge_resource_receipt = getattr(bridge, "last_resource_receipt", None)
        if bridge_resource_receipt is not None:
            if not isinstance(bridge_resource_receipt, ResourceReceipt):
                raise TypeError(
                    "atomic bridge last_resource_receipt must be a ResourceReceipt or null"
                )
            if bridge_resource_receipt not in resource_receipts:
                resource_receipts = (*resource_receipts, bridge_resource_receipt)
        return AtomicPairedReplayRecord(
            receipt=result.receipt,
            baseline=baseline,
            treatment=treatment,
            metric_deltas={
                key: treatment.metrics[key] - baseline.metrics[key] for key in baseline.metrics
            },
            resource_receipts=resource_receipts,
        )


def _validate_bridge(
    bridge: AtomicExternalReplayBridge,
) -> Mapping[str, str]:
    if bridge.atomic_replay_contract != ATOMIC_EXTERNAL_PAIR_CONTRACT:
        raise ValidationError("atomic external bridge contract version is unsupported")
    for label in ("benchmark_id", "source_revision", "bridge_id", "scorer_id"):
        _require_text(getattr(bridge, label), label=f"atomic bridge {label}")
    _require_sha256(bridge.bridge_revision, label="atomic bridge bridge_revision")
    _require_sha256(bridge.scorer_revision, label="atomic bridge scorer_revision")
    return _validated_dependency_hashes(bridge.locked_dependency_hashes)


def _validate_receipt_against_request(
    receipt: AtomicReplayReceipt,
    request: AtomicReplayRequest,
    bridge: AtomicExternalReplayBridge,
) -> None:
    expected = {
        "benchmark_id": request.benchmark_id,
        "checkpoint_id": request.checkpoint_id,
        "source_revision": request.source_revision,
        "bridge_id": bridge.bridge_id,
        "bridge_revision": bridge.bridge_revision,
        "scorer_id": bridge.scorer_id,
        "scorer_revision": bridge.scorer_revision,
        "origin_state_hash": request.origin_state_hash,
        "replay_identity": request.replay_identity,
        "seed": request.seed,
    }
    for field_name, value in expected.items():
        if getattr(receipt, field_name) != value:
            raise ValidationError(
                f"atomic replay receipt {field_name} differs from its request or bridge"
            )
    if receipt.locked_dependency_hashes != request.locked_dependency_hashes:
        raise ValidationError("atomic replay receipt locked dependencies differ from request")
    requested_arms = {(arm.role, arm.arm_id) for arm in request.arms}
    received_arms = {(arm.role, arm.arm_id) for arm in receipt.arms}
    if requested_arms != received_arms:
        raise ValidationError("atomic replay receipt arms differ from request arms")


def _require_complete_arms(arms: tuple[Any, ...], *, label: str) -> None:
    if len(arms) != 2:
        raise ValidationError(f"{label} requires exactly two arms")
    roles = tuple(sorted(getattr(arm, "role", "") for arm in arms))
    if roles != _ROLES:
        raise ValidationError(f"{label} requires baseline and treatment roles")
    identifiers = [getattr(arm, "arm_id", "") for arm in arms]
    if len(set(identifiers)) != 2:
        raise ValidationError(f"{label} requires unique arm IDs")


def _require_arm_types(
    arms: tuple[Any, ...],
    expected_type: type[Any],
    *,
    label: str,
) -> None:
    if not all(isinstance(arm, expected_type) for arm in arms):
        raise TypeError(f"{label} requires {expected_type.__name__} instances")


def _require_role(value: Any) -> None:
    if value not in _ROLES:
        raise ValidationError("atomic replay arm role must be baseline or treatment")


def _validated_dependency_hashes(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError("atomic replay locked_dependency_hashes must be non-empty")
    dependencies: dict[str, str] = {}
    for name, digest in value.items():
        _require_text(name, label="atomic replay dependency name")
        _require_sha256(digest, label=f"atomic replay dependency {name!r}")
        dependencies[name] = digest
    return freeze_mapping(dependencies)


def _validated_metrics(value: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError("atomic replay metrics must be a non-empty mapping")
    metrics: dict[str, float] = {}
    for name, raw in value.items():
        _require_text(name, label="atomic replay metric name")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValidationError(f"atomic replay metric {name!r} must be numeric")
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise ValidationError(f"atomic replay metric {name!r} must be finite")
        metrics[name] = numeric
    return freeze_mapping(metrics)


def _require_text(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(not character.isprintable() for character in value)
    ):
        raise ValidationError(f"{label} must be trimmed, non-empty printable text")


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")


def _require_seed(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("atomic replay seed must be a non-negative integer")


__all__ = [
    "ATOMIC_EXTERNAL_PAIR_CONTRACT",
    "AtomicExternalReplayBridge",
    "AtomicExternalReplayRunner",
    "AtomicPairedReplayRecord",
    "AtomicReplayArmReceipt",
    "AtomicReplayArmRequest",
    "AtomicReplayArmResult",
    "AtomicReplayPairResult",
    "AtomicReplayReceipt",
    "AtomicReplayRequest",
]
