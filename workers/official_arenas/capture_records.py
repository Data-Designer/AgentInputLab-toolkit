"""Visible/protected checkpoint projection, shared by installed capture tools."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentinputlab.contracts.serialization import content_hash
from .protocol import REPLAY_INVENTORY_PROTOCOL, OfficialArenaProvider, validate_dependency_hashes


def _records(
    provider: OfficialArenaProvider,
    config: Mapping[str, Any],
    *,
    split: str,
    output_root: Path,
    dependencies_by_model_slot: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    blockers = provider.doctor(config)
    if blockers:
        raise RuntimeError("official provider preflight failed: " + "; ".join(blockers))
    captured = provider.capture(config, output_root=output_root)
    if isinstance(captured, (str, bytes)) or not isinstance(captured, Sequence) or not captured:
        raise RuntimeError("official provider capture returned no checkpoints")

    visible_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    unit_ids: list[str] = []
    for index, raw in enumerate(captured):
        if not isinstance(raw, Mapping) or set(raw) != {
            "checkpoint_id",
            "group_key",
            "visible_state",
            "eligible_families",
            "protected",
        }:
            raise RuntimeError(f"captured checkpoint {index} has invalid fields")
        checkpoint_id = raw["checkpoint_id"]
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id.strip()
            or checkpoint_id in unit_ids
        ):
            raise RuntimeError("captured checkpoint IDs must be unique non-empty strings")
        if not isinstance(raw["visible_state"], Mapping) or not isinstance(
            raw["protected"], Mapping
        ):
            raise RuntimeError("captured visible/protected state must be objects")
        families = raw["eligible_families"]
        if isinstance(families, (str, bytes)) or not isinstance(families, Sequence) or not families:
            raise RuntimeError("captured eligible_families must be a non-empty array")
        origin_state_hash = content_hash(
            {
                "benchmark_id": provider.benchmark_id,
                "source_revision": provider.source_revision,
                "checkpoint_id": checkpoint_id,
                "protected": raw["protected"],
            }
        )
        replay_identity = content_hash(
            {
                "protocol": REPLAY_INVENTORY_PROTOCOL,
                "benchmark_id": provider.benchmark_id,
                "source_revision": provider.source_revision,
                "checkpoint_id": checkpoint_id,
                "origin_state_hash": origin_state_hash,
            }
        )
        group_id = "group_" + content_hash(
            {
                "benchmark_id": provider.benchmark_id,
                "group_key": raw["group_key"],
            }
        )
        visible_state_hash = content_hash(raw["visible_state"])
        dependency_hashes = validate_dependency_hashes(
            raw["protected"].get("locked_dependency_hashes"),
            label="protected capture",
        )
        visible_row = {
            "benchmark_id": provider.benchmark_id,
            "split": split,
            "checkpoint_id": checkpoint_id,
            "group_id": group_id,
            "source_revision": provider.source_revision,
            "origin_state_hash": origin_state_hash,
            "replay_identity": replay_identity,
            "visible_state": raw["visible_state"],
            "visible_state_hash": visible_state_hash,
            "eligible_families": list(families),
        }
        if dependencies_by_model_slot is None:
            visible_row["locked_dependency_hashes"] = dependency_hashes
        else:
            visible_row["locked_dependency_hashes_by_model_slot"] = {
                slot: dict(dependencies)
                for slot, dependencies in dependencies_by_model_slot.items()
            }
        visible_rows.append(visible_row)
        protected_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "origin_state_hash": origin_state_hash,
                "replay_identity": replay_identity,
                "protected": raw["protected"],
            }
        )
        unit_ids.append(checkpoint_id)
    return visible_rows, protected_rows, unit_ids
