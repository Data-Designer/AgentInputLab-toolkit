"""Selector contracts with explicit abstention/NO_OP."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.state import FrozenAgentState


@dataclass(frozen=True, slots=True)
class SelectionFeatures:
    ambiguity_score: float = 0.0
    multi_step_score: float = 0.0
    retrieval_need_score: float = 0.0
    context_pressure: float = 0.0
    reformulation_need_score: float = 0.0

    def __post_init__(self) -> None:
        values = self.as_vector()
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            or value > 1.0
            for value in values
        ):
            raise ValueError("selector features must be finite numbers in [0, 1]")

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.ambiguity_score,
            self.multi_step_score,
            self.retrieval_need_score,
            self.context_pressure,
            self.reformulation_need_score,
        )


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    family: SemanticFamily
    release_id: str | None
    selection_score: float
    rationale: str
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.family, SemanticFamily):
            raise ValueError("selection family must be a SemanticFamily")
        if self.release_id is not None and (
            not isinstance(self.release_id, str) or not self.release_id.strip()
        ):
            raise ValueError("selection release_id must be non-empty or None")
        if (
            isinstance(self.selection_score, bool)
            or not isinstance(self.selection_score, (int, float))
            or not math.isfinite(self.selection_score)
        ):
            raise ValueError("selection score must be finite")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("selection rationale must not be empty")
        frozen_scores = dict(self.scores)
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for name, value in frozen_scores.items()
        ):
            raise ValueError("selection scores require non-empty names and finite values")
        object.__setattr__(self, "scores", MappingProxyType(frozen_scores))

    @property
    def confidence(self) -> float:
        """Backward-compatible alias; this is a score, not a calibrated probability."""

        return self.selection_score


@runtime_checkable
class Selector(Protocol):
    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision: ...


@runtime_checkable
class UtilityGainProvider(Selector, Protocol):
    """Selector whose per-family scores are explicitly utility gains over NO_OP.

    Generic selector scores (for example ambiguity or retrieval-need
    heuristics) are intentionally not interchangeable with utility gains.
    Guardrail policies that subtract cost or risk penalties must require this
    stronger protocol instead of consuming arbitrary ``SelectionDecision``
    scores.
    """

    def predict_gains(
        self,
        features: SelectionFeatures,
    ) -> Mapping[SemanticFamily, float]: ...
