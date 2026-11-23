"""Transparent selector baseline used before learned selector confirmation."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.contracts.state import FrozenAgentState

from .base import SelectionDecision, SelectionFeatures


@dataclass(frozen=True, slots=True)
class RuleSelector:
    threshold: float = 0.6
    margin: float = 0.1

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("rule selector threshold must be finite and in [0, 1]")
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or not math.isfinite(self.margin)
            or self.margin < 0
        ):
            raise ValueError("rule selector margin must be finite and non-negative")

    @property
    def model_hash(self) -> str:
        return content_hash(
            {
                "selector": "rule",
                "threshold": self.threshold,
                "margin": self.margin,
            }
        )

    def select(
        self,
        state: FrozenAgentState,
        features: SelectionFeatures,
        *,
        eligible_families: Collection[SemanticFamily] | None = None,
    ) -> SelectionDecision:
        del state
        scores = {
            SemanticFamily.CLARIFY.value: features.ambiguity_score,
            SemanticFamily.DECOMPOSE.value: features.multi_step_score,
            SemanticFamily.ACQUIRE.value: features.retrieval_need_score,
            SemanticFamily.CURATE.value: features.context_pressure,
            SemanticFamily.REFORMULATE.value: features.reformulation_need_score,
        }
        eligible = _eligible_actions(eligible_families)
        ordered = sorted(
            ((family.value, scores[family.value]) for family in eligible),
            key=lambda item: (-item[1], item[0]),
        )
        if not ordered:
            return SelectionDecision(
                family=SemanticFamily.NO_OP,
                release_id="baseline.no_op",
                selection_score=0.0,
                rationale="abstained: no intervention family is eligible",
                scores=scores,
            )
        best_name, best_score = ordered[0]
        runner_up = max(0.0, ordered[1][1] if len(ordered) > 1 else 0.0)
        if best_score < self.threshold or best_score - runner_up <= self.margin:
            return SelectionDecision(
                family=SemanticFamily.NO_OP,
                release_id="baseline.no_op",
                selection_score=0.0,
                rationale="abstained: no family passed threshold and margin",
                scores=scores,
            )
        return SelectionDecision(
            family=SemanticFamily(best_name),
            release_id=None,
            selection_score=best_score,
            rationale=f"selected highest eligible family score: {best_name}",
            scores=scores,
        )


def _eligible_actions(
    eligible_families: Collection[SemanticFamily] | None,
) -> tuple[SemanticFamily, ...]:
    actions = (
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    )
    if eligible_families is None:
        return actions
    eligible = tuple(eligible_families)
    if any(not isinstance(family, SemanticFamily) for family in eligible):
        raise ValueError("eligible_families must contain SemanticFamily values")
    return tuple(family for family in actions if family in eligible)
