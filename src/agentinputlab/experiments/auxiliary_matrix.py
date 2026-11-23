"""Capability-locked 27-method by seven-static-dataset execution matrix.

The matrix distinguishes structural eligibility from execution evidence.  A
static benchmark is eligible only when it declares the method's semantic
family.  Ineligible pairs remain explicit rows; they are never converted into
zero scores or silently omitted from denominators.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any

from agentinputlab.benchmarks import BenchmarkRegistry, production_benchmark_specs
from agentinputlab.contracts.enums import ProfileKind
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.registry import AlgorithmRegistry

AUXILIARY_MATRIX_PROTOCOL = "agentinputlab.auxiliary-method-cell-matrix.v1"
AUXILIARY_PROFILE_PROTOCOL = "agentinputlab.auxiliary-method-profiles.v1"


@dataclass(frozen=True, slots=True)
class AuxiliaryMethodProfile:
    method_id: str
    release_id: str
    profile_kind: ProfileKind

    def __post_init__(self) -> None:
        for label, value in (
            ("method_id", self.method_id),
            ("release_id", self.release_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"auxiliary profile {label} must be non-empty")
        if not isinstance(self.profile_kind, ProfileKind):
            raise TypeError("auxiliary profile_kind must be ProfileKind")


def auxiliary_static_cells() -> Mapping[str, str]:
    """Return the exact seven production static cells and frozen splits."""

    cells = {
        spec.cell_id: spec.split
        for spec in production_benchmark_specs()
        if spec.adapter_kind in {"retrieval", "qa"}
    }
    if len(cells) != 7:
        raise ValidationError(f"auxiliary matrix requires seven static cells, got {len(cells)}")
    return MappingProxyType(dict(sorted(cells.items())))


def load_auxiliary_method_profiles() -> Mapping[str, AuxiliaryMethodProfile]:
    """Load and validate the packaged method-to-executable-profile selection."""

    try:
        source = resources.files("agentinputlab.registry").joinpath(
            "data/auxiliary_method_profiles.json"
        )
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("auxiliary method profile catalog is unavailable") from exc
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "protocol", "method_count", "methods"}
        or raw.get("schema_version") != 1
        or raw.get("protocol") != AUXILIARY_PROFILE_PROTOCOL
        or not isinstance(raw.get("methods"), list)
        or raw.get("method_count") != len(raw["methods"])
    ):
        raise ValidationError("auxiliary method profile catalog schema is invalid")

    registry = AlgorithmRegistry.default()
    profiles: dict[str, AuxiliaryMethodProfile] = {}
    for index, item in enumerate(raw["methods"]):
        if not isinstance(item, dict) or set(item) != {
            "method_id",
            "release_id",
            "profile_kind",
        }:
            raise ValidationError(f"auxiliary method profile row {index} is invalid")
        try:
            profile = AuxiliaryMethodProfile(
                method_id=item["method_id"],
                release_id=item["release_id"],
                profile_kind=ProfileKind(item["profile_kind"]),
            )
            method, release = registry.get_release(profile.release_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                f"auxiliary method profile row {index} cannot be resolved"
            ) from exc
        if profile.method_id in profiles:
            raise ValidationError("auxiliary method profile catalog contains duplicates")
        if method.method_id != profile.method_id:
            raise ValidationError("auxiliary profile release belongs to another method")
        if profile.profile_kind not in release.profile_modes:
            raise ValidationError("auxiliary profile kind is unsupported by its release")
        if release.entrypoint is None:
            raise ValidationError("auxiliary profile release has no executable entrypoint")
        profiles[profile.method_id] = profile

    registered = {method.method_id for method in registry.list_methods()}
    if set(profiles) != registered or len(profiles) != 27:
        raise ValidationError("auxiliary profiles must cover the exact 27-method registry")
    return MappingProxyType(dict(sorted(profiles.items())))


def build_auxiliary_method_cell_matrix() -> Mapping[str, Any]:
    """Build the deterministic 27x7 eligibility and launch matrix."""

    registry = AlgorithmRegistry.default()
    benchmark_registry = BenchmarkRegistry.default()
    profiles = load_auxiliary_method_profiles()
    cells = auxiliary_static_cells()
    rows: list[dict[str, Any]] = []
    family_method_counts: dict[str, int] = {}
    for method in sorted(registry.list_methods(), key=lambda value: value.method_id):
        family = method.semantic_family.value
        family_method_counts[family] = family_method_counts.get(family, 0) + 1
        profile = profiles[method.method_id]
        for cell_id, split in cells.items():
            cell = benchmark_registry.get(cell_id)
            eligible = method.semantic_family in cell.families
            reason = (
                None
                if eligible
                else (
                    f"{cell_id} does not declare {method.semantic_family.value} support; "
                    "the pair is structurally ineligible"
                )
            )
            rows.append(
                {
                    "method_id": method.method_id,
                    "display_name": method.display_name,
                    "release_id": profile.release_id,
                    "profile_kind": profile.profile_kind.value,
                    "family": family,
                    "benchmark_cell_id": cell_id,
                    "split": split,
                    "benchmark_family_support": tuple(
                        item.value for item in sorted(cell.families, key=lambda value: value.value)
                    ),
                    "eligible": eligible,
                    "state": "launchable" if eligible else "structurally_ineligible",
                    "reason": reason,
                    "one_click_command": (
                        "python scripts/experiments/run_auxiliary_method_cell.py "
                        "<frozen-campaign.json> "
                        f"--method {method.method_id} --cell {cell_id}"
                        if eligible
                        else None
                    ),
                }
            )
    eligible_count = sum(bool(row["eligible"]) for row in rows)
    if len(rows) != 189 or eligible_count != 117:
        raise ValidationError(
            f"auxiliary matrix invariant failed: rows={len(rows)}, eligible={eligible_count}"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": AUXILIARY_MATRIX_PROTOCOL,
        "record_type": "EXECUTION_DESIGN",
        "observed_results": False,
        "method_count": 27,
        "dataset_count": 7,
        "row_count": len(rows),
        "eligible_pair_count": eligible_count,
        "structurally_ineligible_pair_count": len(rows) - eligible_count,
        "family_method_counts": dict(sorted(family_method_counts.items())),
        "cells": dict(cells),
        "rows": rows,
    }
    payload["matrix_hash"] = content_hash(payload)
    return freeze_mapping(payload)


def auxiliary_pair(method_id: str, cell_id: str) -> Mapping[str, Any]:
    """Resolve one row and reject unknown identities without guessing."""

    for row in build_auxiliary_method_cell_matrix()["rows"]:
        if row["method_id"] == method_id and row["benchmark_cell_id"] == cell_id:
            return row
    raise KeyError(f"unknown auxiliary method-cell pair: {method_id} x {cell_id}")


__all__ = [
    "AUXILIARY_MATRIX_PROTOCOL",
    "AUXILIARY_PROFILE_PROTOCOL",
    "AuxiliaryMethodProfile",
    "auxiliary_pair",
    "auxiliary_static_cells",
    "build_auxiliary_method_cell_matrix",
    "load_auxiliary_method_profiles",
]
