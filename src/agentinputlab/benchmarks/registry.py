"""Packaged benchmark inventory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import resources
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import RegistryError

EXPECTED_BENCHMARK_CELL_COUNT = 31


@dataclass(frozen=True, slots=True)
class BenchmarkCell:
    cell_id: str
    suite: str
    task: str
    size: str
    families: tuple[SemanticFamily, ...]
    source_revision: str
    metrics: tuple[str, ...]
    eligibility_gate: str = ""
    family_support: str = "cell_declared"
    adapter_metrics: tuple[str, ...] = ()
    paired_metrics: tuple[str, ...] = ()
    study_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family_support not in {"cell_declared", "checkpoint_conditional"}:
            raise ValueError("benchmark family_support has an unsupported value")
        if not self.metrics or len(self.metrics) != len(set(self.metrics)):
            raise ValueError("benchmark metrics must be non-empty and unique")
        adapter = tuple(self.adapter_metrics)
        paired = tuple(self.paired_metrics)
        study = tuple(self.study_metrics)
        if not adapter and not paired and not study:
            adapter = tuple(self.metrics)
        layered = (*adapter, *paired, *study)
        if len(layered) != len(set(layered)) or set(layered) != set(self.metrics):
            raise ValueError("benchmark metric layers must partition metrics exactly")
        object.__setattr__(self, "adapter_metrics", adapter)
        object.__setattr__(self, "paired_metrics", paired)
        object.__setattr__(self, "study_metrics", study)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkCell:
        try:
            return cls(
                cell_id=str(value["cell_id"]),
                suite=str(value["suite"]),
                task=str(value["task"]),
                size=str(value["size"]),
                families=tuple(SemanticFamily(item) for item in value["families"]),
                source_revision=str(value["source_revision"]),
                metrics=tuple(str(item) for item in value["metrics"]),
                eligibility_gate=str(value.get("eligibility_gate", "")),
                family_support=str(value.get("family_support", "cell_declared")),
                adapter_metrics=tuple(str(item) for item in value.get("adapter_metrics", ())),
                paired_metrics=tuple(str(item) for item in value.get("paired_metrics", ())),
                study_metrics=tuple(str(item) for item in value.get("study_metrics", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"invalid benchmark cell: {value!r}") from exc


class BenchmarkRegistry:
    def __init__(self, cells: tuple[BenchmarkCell, ...]) -> None:
        if len(cells) != EXPECTED_BENCHMARK_CELL_COUNT:
            raise RegistryError(
                "benchmark registry requires "
                f"{EXPECTED_BENCHMARK_CELL_COUNT} cells, got {len(cells)}"
            )
        if len({cell.cell_id for cell in cells}) != len(cells):
            raise RegistryError("benchmark cell IDs must be unique")
        self._cells = cells
        self._by_id = {cell.cell_id: cell for cell in cells}

    @classmethod
    def default(cls) -> BenchmarkRegistry:
        resource = resources.files("agentinputlab.benchmarks").joinpath("data", "registry.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("cell_count") != EXPECTED_BENCHMARK_CELL_COUNT
        ):
            raise RegistryError("unsupported benchmark registry snapshot")
        return cls(tuple(BenchmarkCell.from_mapping(item) for item in payload["cells"]))

    def get(self, cell_id: str) -> BenchmarkCell:
        try:
            return self._by_id[cell_id]
        except KeyError as exc:
            matches = get_close_matches(cell_id, self._by_id, n=3)
            hint = f"; did you mean: {', '.join(matches)}?" if matches else "; use agentinputlab datasets list"
            raise RegistryError(f"unknown benchmark cell: {cell_id}{hint}") from exc

    def list(
        self,
        *,
        family: SemanticFamily | str | None = None,
    ) -> tuple[BenchmarkCell, ...]:
        selected = SemanticFamily(family) if isinstance(family, str) else family
        if selected is None:
            return self._cells
        return tuple(cell for cell in self._cells if selected in cell.families)

    def __len__(self) -> int:
        return len(self._cells)
