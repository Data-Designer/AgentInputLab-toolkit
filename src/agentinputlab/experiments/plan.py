"""Stable cell expansion and deterministic sharding for experiment manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import (
    canonical_json,
    content_hash,
    freeze_mapping,
    to_jsonable,
)

from .schema import ExperimentManifest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ExperimentCell:
    study_id: str
    benchmark_id: str
    split: str
    unit_id: str
    model_slot: str
    seed: int
    arm_ids: tuple[str, ...]
    manifest_hash: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.study_id,
                self.benchmark_id,
                self.split,
                self.unit_id,
                self.model_slot,
                self.manifest_hash,
            )
        ):
            raise ValidationError("experiment cell identities must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError("experiment cell seed must be non-negative")
        arms = tuple(self.arm_ids)
        if len(arms) < 2 or len(arms) != len(set(arms)):
            raise ValidationError("experiment cell requires unique comparison arms")
        object.__setattr__(self, "arm_ids", arms)

    @property
    def cell_id(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    study_id: str
    manifest_hash: str
    cells: tuple[ExperimentCell, ...]
    unit_inventories: Mapping[str, tuple[str, ...]]
    unit_inventory_hashes: Mapping[str, str]
    parent_plan_hash: str | None = None
    parent_cell_ids: tuple[str, ...] = ()
    shard_index: int | None = None
    shard_count: int | None = None

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if not cells:
            raise ValidationError("experiment plan requires at least one cell")
        if len({item.cell_id for item in cells}) != len(cells):
            raise ValidationError("experiment plan contains duplicate cells")
        if any(
            item.study_id != self.study_id or item.manifest_hash != self.manifest_hash
            for item in cells
        ):
            raise ValidationError("experiment cells differ from their parent plan")
        parent_cell_ids = tuple(self.parent_cell_ids)
        shard_fields = (
            self.parent_plan_hash,
            self.shard_index,
            self.shard_count,
        )
        if all(value is None for value in shard_fields):
            if parent_cell_ids:
                raise ValidationError("root experiment plan cannot declare parent_cell_ids")
        elif any(value is None for value in shard_fields):
            raise ValidationError("experiment shard requires all parent and shard fields")
        else:
            if (
                not isinstance(self.parent_plan_hash, str)
                or _SHA256.fullmatch(self.parent_plan_hash) is None
            ):
                raise ValidationError("experiment shard parent_plan_hash must be SHA-256")
            if (
                isinstance(self.shard_index, bool)
                or not isinstance(self.shard_index, int)
                or isinstance(self.shard_count, bool)
                or not isinstance(self.shard_count, int)
                or self.shard_count < 1
                or self.shard_index < 0
                or self.shard_index >= self.shard_count
            ):
                raise ValidationError("experiment shard requires 0 <= index < count")
            if (
                not parent_cell_ids
                or len(parent_cell_ids) != len(set(parent_cell_ids))
                or any(
                    not isinstance(item, str) or _SHA256.fullmatch(item) is None
                    for item in parent_cell_ids
                )
            ):
                raise ValidationError(
                    "experiment shard requires unique parent cell SHA-256 identities"
                )
            if not {cell.cell_id for cell in cells}.issubset(parent_cell_ids):
                raise ValidationError("experiment shard cells are outside the parent plan")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "parent_cell_ids", parent_cell_ids)
        inventories: dict[str, tuple[str, ...]] = {}
        for key, raw_units in self.unit_inventories.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError("experiment unit inventory keys must be non-empty strings")
            units = tuple(raw_units)
            if (
                not units
                or len(units) != len(set(units))
                or any(not isinstance(unit_id, str) or not unit_id.strip() for unit_id in units)
            ):
                raise ValidationError(
                    f"experiment unit inventory {key} requires unique non-empty unit IDs"
                )
            inventories[key] = tuple(sorted(units))
        hashes = dict(self.unit_inventory_hashes)
        if set(inventories) != set(hashes):
            raise ValidationError(
                "experiment unit inventories and inventory hashes require identical keys"
            )
        for key, units in inventories.items():
            expected_hash = content_hash({"key": key, "unit_ids": list(units)})
            if hashes[key] != expected_hash:
                raise ValidationError(f"experiment unit inventory hash mismatch for {key}")
        object.__setattr__(
            self,
            "unit_inventories",
            freeze_mapping(inventories),
        )
        object.__setattr__(
            self,
            "unit_inventory_hashes",
            freeze_mapping(hashes),
        )

    @property
    def plan_hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["plan_hash"] = self.plan_hash
        payload["cell_count"] = len(self.cells)
        payload["cells"] = [{**to_jsonable(cell), "cell_id": cell.cell_id} for cell in self.cells]
        return payload

    def shard(self, *, index: int, count: int) -> ExperimentPlan:
        if self.parent_plan_hash is not None:
            raise ValidationError("an experiment shard cannot be sharded again")
        if (
            isinstance(index, bool)
            or isinstance(count, bool)
            or not isinstance(index, int)
            or not isinstance(count, int)
            or count < 1
            or index < 0
            or index >= count
        ):
            raise ValidationError("shard requires 0 <= index < count")
        selected = tuple(
            cell for position, cell in enumerate(self.cells) if position % count == index
        )
        if not selected:
            raise ValidationError("requested shard contains no experiment cells")
        return ExperimentPlan(
            study_id=self.study_id,
            manifest_hash=self.manifest_hash,
            cells=selected,
            unit_inventories=self.unit_inventories,
            unit_inventory_hashes=self.unit_inventory_hashes,
            parent_plan_hash=self.plan_hash,
            parent_cell_ids=tuple(cell.cell_id for cell in self.cells),
            shard_index=index,
            shard_count=count,
        )


def build_experiment_plan(
    manifest: ExperimentManifest,
    resolved_units: Mapping[str, tuple[str, ...]],
) -> ExperimentPlan:
    expected = {f"{target.cell_id}:{target.split}" for target in manifest.benchmarks}
    if set(resolved_units) != expected:
        raise ValidationError("resolved unit inventories do not match benchmark targets")
    all_arm_ids = tuple(arm.arm_id for arm in manifest.arms)
    cells: list[ExperimentCell] = []
    inventory_hashes: dict[str, str] = {}
    for target in manifest.benchmarks:
        arm_ids = target.arm_ids or all_arm_ids
        key = f"{target.cell_id}:{target.split}"
        units = tuple(resolved_units[key])
        if not units:
            raise ValidationError(f"benchmark target {key} has no resolved units")
        if len(units) != len(set(units)):
            raise ValidationError(f"benchmark target {key} has duplicate units")
        if target.unit_ids and tuple(sorted(target.unit_ids)) != tuple(sorted(units)):
            raise ValidationError(
                f"resolved unit inventory differs from inline manifest units for {key}"
            )
        inventory_hashes[key] = content_hash({"key": key, "unit_ids": sorted(units)})
        for unit_id in sorted(units):
            for model_slot in manifest.model_slots:
                for seed in manifest.seeds:
                    cells.append(
                        ExperimentCell(
                            study_id=manifest.study_id,
                            benchmark_id=target.cell_id,
                            split=target.split,
                            unit_id=unit_id,
                            model_slot=model_slot,
                            seed=seed,
                            arm_ids=arm_ids,
                            manifest_hash=manifest.manifest_hash,
                        )
                    )
    return ExperimentPlan(
        study_id=manifest.study_id,
        manifest_hash=manifest.manifest_hash,
        cells=tuple(cells),
        unit_inventories={
            key: tuple(sorted(resolved_units[key])) for key in sorted(resolved_units)
        },
        unit_inventory_hashes=inventory_hashes,
    )


def validate_experiment_plan(
    manifest: ExperimentManifest,
    plan: ExperimentPlan,
) -> None:
    """Require a root/shard plan to be the full deterministic manifest expansion."""

    if not isinstance(manifest, ExperimentManifest) or not isinstance(plan, ExperimentPlan):
        raise TypeError("experiment plan validation requires typed manifest and plan")
    root_plan = build_experiment_plan(
        manifest,
        {key: tuple(unit_ids) for key, unit_ids in plan.unit_inventories.items()},
    )
    expected = (
        root_plan
        if plan.parent_plan_hash is None
        else root_plan.shard(index=plan.shard_index, count=plan.shard_count)
    )
    if canonical_json(plan.to_dict()) != canonical_json(expected.to_dict()):
        raise ValidationError(
            "experiment plan is not the complete deterministic expansion of its manifest"
        )


__all__ = [
    "ExperimentCell",
    "ExperimentPlan",
    "build_experiment_plan",
    "validate_experiment_plan",
]
