"""Pre- and post-run dependency balance auditing."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentinputlab.contracts.dependency import DependencyBinding
from agentinputlab.contracts.enums import ProfileKind
from agentinputlab.contracts.errors import BalanceError, ValidationError
from agentinputlab.contracts.lifecycle import InterventionContext
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

_RUNTIME_COMPONENT_SLOTS = (
    "backend",
    "dense_embedder",
    "retriever",
    "vector_searcher",
    "tools",
    "external_runtime",
    "artifact_manager",
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_RESERVED_FINGERPRINT_SLOTS = frozenset(
    (
        *_RUNTIME_COMPONENT_SLOTS,
        "seed",
        "pricing_catalog",
        "intent_contract",
        "strict_intent",
        "run_config",
    )
)


@dataclass(frozen=True, slots=True)
class ComparisonProfile:
    profile_id: str
    kind: ProfileKind
    required_slots: tuple[str, ...]
    varied_slots: tuple[str, ...] = ()
    seed_policy: str = "fixed"
    cache_policy: str = "frozen"

    def __post_init__(self) -> None:
        if set(self.required_slots) & set(self.varied_slots):
            raise ValidationError("a slot cannot be both required-common and varied")


@dataclass(frozen=True, slots=True)
class ComparisonArm:
    arm_id: str
    release_id: str
    bindings: tuple[DependencyBinding, ...]

    def binding_map(self) -> dict[str, DependencyBinding]:
        mapping = {binding.slot: binding for binding in self.bindings}
        if len(mapping) != len(self.bindings):
            raise ValidationError(f"arm {self.arm_id} contains duplicate dependency slots")
        return mapping


@dataclass(frozen=True, slots=True)
class BalanceReport:
    eligible: bool
    differences: tuple[str, ...]
    missing: tuple[str, ...]

    def require_eligible(self) -> None:
        if not self.eligible:
            details = "; ".join((*self.missing, *self.differences))
            raise BalanceError(f"comparison is not balanced: {details}")


class BalanceAuditor:
    """Compare resolved dependency hashes, never human-readable aliases."""

    def audit(
        self,
        profile: ComparisonProfile,
        arms: tuple[ComparisonArm, ...],
        *,
        runtime_fingerprints: Mapping[str, Mapping[str, str]] | None = None,
    ) -> BalanceReport:
        if len(arms) < 2:
            raise ValidationError("balance audit requires at least two arms")
        if len({arm.arm_id for arm in arms}) != len(arms):
            raise ValidationError("balance audit requires unique arm IDs")
        maps = {arm.arm_id: arm.binding_map() for arm in arms}
        missing: list[str] = []
        differences: list[str] = []

        for slot in profile.required_slots:
            absent = [arm.arm_id for arm in arms if slot not in maps[arm.arm_id]]
            if absent:
                missing.append(f"{slot} missing in {','.join(absent)}")
                continue
            hashes = {maps[arm.arm_id][slot].resolved_hash for arm in arms}
            if len(hashes) != 1:
                differences.append(f"{slot} differs across arms")

        for slot in profile.varied_slots:
            absent = [arm.arm_id for arm in arms if slot not in maps[arm.arm_id]]
            if absent:
                missing.append(f"{slot} missing in {','.join(absent)}")

        all_slots = set().union(*(set(mapping) for mapping in maps.values()))
        undeclared = all_slots - set(profile.required_slots) - set(profile.varied_slots)
        for slot in sorted(undeclared):
            present = [maps[arm.arm_id].get(slot) for arm in arms]
            if None in present:
                differences.append(f"{slot} is an undeclared asymmetric dependency")
                continue
            hashes = {binding.resolved_hash for binding in present if binding is not None}
            if len(hashes) > 1:
                differences.append(f"{slot} varies but is not declared crossed")

        if runtime_fingerprints is not None:
            arm_ids = {arm.arm_id for arm in arms}
            if set(runtime_fingerprints) != arm_ids:
                raise ValidationError("runtime fingerprints must be present for every arm")
            for arm_id, fingerprints in runtime_fingerprints.items():
                if not isinstance(fingerprints, Mapping):
                    raise ValidationError(f"runtime fingerprints for {arm_id} must be a mapping")
                for slot, fingerprint in fingerprints.items():
                    if not isinstance(slot, str) or not slot:
                        raise ValidationError("runtime fingerprint slots must be non-empty strings")
                    if not isinstance(fingerprint, str) or not _HASH_PATTERN.fullmatch(fingerprint):
                        raise ValidationError(
                            f"runtime fingerprint {arm_id}.{slot} must be a SHA-256 hex string"
                        )
            observed_slots = set().union(*(set(runtime_fingerprints[arm_id]) for arm_id in arm_ids))
            for slot in profile.required_slots:
                if slot not in _RUNTIME_COMPONENT_SLOTS and slot not in observed_slots:
                    continue
                absent = [
                    arm.arm_id for arm in arms if slot not in runtime_fingerprints[arm.arm_id]
                ]
                if absent:
                    _append_unique(missing, f"{slot} runtime missing in {','.join(absent)}")
                    continue
                hashes = {runtime_fingerprints[arm.arm_id][slot] for arm in arms}
                if len(hashes) != 1:
                    _append_unique(differences, f"{slot} runtime differs across arms")

            for slot in profile.varied_slots:
                if slot not in _RUNTIME_COMPONENT_SLOTS and slot not in observed_slots:
                    continue
                absent = [
                    arm.arm_id for arm in arms if slot not in runtime_fingerprints[arm.arm_id]
                ]
                if absent:
                    _append_unique(missing, f"{slot} runtime missing in {','.join(absent)}")

            for slot in sorted(observed_slots):
                present = [runtime_fingerprints[arm.arm_id].get(slot) for arm in arms]
                if slot == "seed":
                    if profile.seed_policy == "fixed" and len(set(present)) != 1:
                        _append_unique(differences, "seed runtime differs across arms")
                    continue
                if slot in profile.required_slots or slot in profile.varied_slots:
                    continue
                if None in present:
                    _append_unique(
                        differences,
                        f"{slot} is an undeclared asymmetric runtime dependency",
                    )
                elif len(set(present)) > 1:
                    _append_unique(
                        differences,
                        f"{slot} runtime varies but is not declared crossed",
                    )

        return BalanceReport(
            eligible=not missing and not differences,
            differences=tuple(differences),
            missing=tuple(missing),
        )


def context_dependency_fingerprints(context: InterventionContext) -> dict[str, str]:
    """Derive hashes from the objects that will actually execute an arm."""

    observed: dict[str, str] = {"seed": content_hash(context.seed)}
    for slot in _RUNTIME_COMPONENT_SLOTS:
        dependency = getattr(context, slot)
        if dependency is None:
            continue
        fingerprint = getattr(dependency, "fingerprint", None)
        if not callable(fingerprint):
            raise ValidationError(f"runtime {slot} does not expose fingerprint()")
        identity = fingerprint()
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError(f"runtime {slot} fingerprint must be a non-empty mapping")
        observed[slot] = _canonical_fingerprint_hash(identity, label=f"runtime {slot} fingerprint")

    if context.pricing_catalog is not None:
        observed["pricing_catalog"] = content_hash(context.pricing_catalog)
    if context.intent_contract is not None:
        observed["intent_contract"] = content_hash(context.intent_contract)
    observed["strict_intent"] = content_hash(context.strict_intent)

    declared = context.run_config.get("dependency_fingerprints", {})
    if not isinstance(declared, Mapping):
        raise ValidationError("dependency_fingerprints must be a mapping")
    for slot, identity in declared.items():
        if slot in _RESERVED_FINGERPRINT_SLOTS:
            raise ValidationError(
                f"reserved dependency slot must come from its actual context field: {slot}"
            )
        if slot in observed:
            raise ValidationError(f"runtime dependency slot is reserved or duplicated: {slot}")
        if not isinstance(slot, str) or not slot:
            raise ValidationError("declared dependency slots must be non-empty strings")
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError(
                f"declared dependency identity for {slot} must be a non-empty mapping"
            )
        observed[slot] = _canonical_fingerprint_hash(
            identity,
            label=f"declared dependency identity for {slot}",
        )

    remaining_config: dict[str, Any] = {
        key: value for key, value in context.run_config.items() if key != "dependency_fingerprints"
    }
    if remaining_config:
        observed["run_config"] = content_hash(remaining_config)
    return observed


def _append_unique(values: list[str], message: str) -> None:
    if message not in values:
        values.append(message)


def _canonical_fingerprint_hash(identity: Mapping[str, Any], *, label: str) -> str:
    try:
        normalized = freeze_mapping(identity)
        if not normalized:
            raise ValidationError(f"{label} must be a non-empty mapping")
        return content_hash(normalized)
    except ValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be canonically serializable: {exc}") from exc
