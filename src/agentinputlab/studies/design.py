"""Deterministic native, controlled, and factorial experiment planning."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.dependency import DependencyBinding, DependencyRole
from agentinputlab.contracts.enums import ProfileKind
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.runner.balance import BalanceAuditor, ComparisonArm, ComparisonProfile


@dataclass(frozen=True, slots=True)
class StudyArmSpec:
    arm_id: str
    release_id: str
    bindings: tuple[DependencyBinding, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))
        ComparisonArm(self.arm_id, self.release_id, self.bindings).binding_map()


@dataclass(frozen=True, slots=True)
class StudyCell:
    benchmark_id: str
    example_id: str
    seed: int
    profile_id: str
    arms: tuple[StudyArmSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", tuple(self.arms))
        if not self.benchmark_id or not self.example_id:
            raise ValueError("study cell requires benchmark and example IDs")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("study cell seed must be an integer")
        if len(self.arms) < 2:
            raise ValueError("study cell requires at least two comparison arms")
        if len({arm.arm_id for arm in self.arms}) != len(self.arms):
            raise ValueError("study cell arm IDs must be unique")

    @property
    def cell_id(self) -> str:
        return content_hash(self)[:20]


@dataclass(frozen=True, slots=True)
class StudyPlan:
    study_id: str
    profile: ComparisonProfile
    cells: tuple[StudyCell, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.study_id or not self.cells:
            raise ValueError("study plan requires an ID and at least one cell")
        if any(cell.profile_id != self.profile.profile_id for cell in self.cells):
            raise ValueError("study cells must reference the plan comparison profile")
        if len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("study plan contains duplicate cells")

    @property
    def plan_hash(self) -> str:
        return content_hash(self)


def build_study_plan(
    *,
    study_id: str,
    profile: ComparisonProfile,
    benchmark_id: str,
    example_ids: Sequence[str],
    seeds: Sequence[int],
    arms: Sequence[StudyArmSpec],
    factorial_levels: Mapping[str, Sequence[DependencyBinding]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StudyPlan:
    """Expand a frozen manifest without executing or silently dropping cells."""

    examples = tuple(example_ids)
    seed_values = tuple(seeds)
    base_arms = tuple(arms)
    if not examples or not seed_values or not base_arms:
        raise ValueError("study expansion requires examples, seeds, and arms")
    if len(set(examples)) != len(examples):
        raise ValueError("example IDs must be unique")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    levels = factorial_levels or {}
    if levels and profile.kind is not ProfileKind.FACTORIAL_ADAPTER:
        raise ValueError("factorial levels require a factorial comparison profile")
    if profile.kind is ProfileKind.FACTORIAL_ADAPTER and set(levels) != set(profile.varied_slots):
        raise ValueError("factorial level slots must exactly match profile.varied_slots")
    expanded_arms = _expand_arms(base_arms, levels)
    if profile.kind is not ProfileKind.OFFICIAL_NATIVE:
        BalanceAuditor().audit(
            profile,
            tuple(ComparisonArm(arm.arm_id, arm.release_id, arm.bindings) for arm in expanded_arms),
        ).require_eligible()
    cells = tuple(
        StudyCell(
            benchmark_id=benchmark_id,
            example_id=example_id,
            seed=seed,
            profile_id=profile.profile_id,
            arms=expanded_arms,
        )
        for example_id in examples
        for seed in seed_values
    )
    return StudyPlan(
        study_id=study_id,
        profile=profile,
        cells=cells,
        metadata=metadata or {},
    )


def _expand_arms(
    arms: tuple[StudyArmSpec, ...],
    levels: Mapping[str, Sequence[DependencyBinding]],
) -> tuple[StudyArmSpec, ...]:
    if not levels:
        return arms
    slot_names = tuple(sorted(levels))
    choices = tuple(tuple(levels[slot]) for slot in slot_names)
    if any(not values for values in choices):
        raise ValueError("each factorial slot requires at least one dependency level")
    for slot, values in zip(slot_names, choices, strict=True):
        if any(binding.slot != slot for binding in values):
            raise ValueError(f"factorial bindings for {slot} must use that slot name")
        if any(binding.role is not DependencyRole.CROSSED for binding in values):
            raise ValueError(f"factorial bindings for {slot} must have role='crossed'")
    expanded: list[StudyArmSpec] = []
    for arm in arms:
        base = {binding.slot: binding for binding in arm.bindings}
        overlap = set(base) & set(slot_names)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(
                f"factorial slots cannot silently replace base-arm dependencies: {joined}"
            )
        for combination in itertools.product(*choices):
            bindings = dict(base)
            bindings.update({binding.slot: binding for binding in combination})
            suffix = ".".join(
                f"{binding.slot}={binding.resolved_hash[:8]}" for binding in combination
            )
            expanded.append(
                StudyArmSpec(
                    arm_id=f"{arm.arm_id}::{suffix}",
                    release_id=arm.release_id,
                    bindings=tuple(bindings[key] for key in sorted(bindings)),
                    parameters=arm.parameters,
                )
            )
    return tuple(expanded)
