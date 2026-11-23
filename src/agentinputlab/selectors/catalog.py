"""Stable public inventory for selector discovery surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentinputlab.contracts.enums import StrEnum

_SELECTOR_ID = re.compile(r"^[a-z][a-z0-9_]*\.v[1-9][0-9]*$")


class SelectorKind(StrEnum):
    PREDICTOR = "predictor"
    GUARDRAIL = "guardrail"


@dataclass(frozen=True, slots=True)
class SelectorCard:
    selector_id: str
    display_name: str
    kind: SelectorKind
    requires_fit: bool
    optional_dependency: str = ""

    def __post_init__(self) -> None:
        if not _SELECTOR_ID.fullmatch(self.selector_id):
            raise ValueError("selector_id must be a stable name.vN identifier")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("selector display_name must be non-empty")
        if not isinstance(self.kind, SelectorKind):
            raise ValueError("selector kind must be typed")
        if not isinstance(self.requires_fit, bool):
            raise ValueError("selector requires_fit must be boolean")
        if not isinstance(self.optional_dependency, str):
            raise ValueError("selector optional_dependency must be a string")


_CARDS = (
    SelectorCard("rule.v1", "Transparent defect rule", SelectorKind.PREDICTOR, False),
    SelectorCard("ridge_utility.v1", "Direct ridge utility", SelectorKind.PREDICTOR, True),
    SelectorCard("ridge_t_learner.v1", "Ridge T-learner", SelectorKind.PREDICTOR, True),
    SelectorCard(
        "logistic_t_learner.v1",
        "Logistic T-learner",
        SelectorKind.PREDICTOR,
        True,
    ),
    SelectorCard(
        "gbdt_t_learner.v1",
        "GBDT T-learner",
        SelectorKind.PREDICTOR,
        True,
        "scikit-learn",
    ),
    SelectorCard("knn_utility.v1", "KNN utility", SelectorKind.PREDICTOR, True),
    SelectorCard(
        "cost_risk_guardrail.v1",
        "Cost/risk-aware guardrail",
        SelectorKind.GUARDRAIL,
        True,
    ),
)


def list_selector_cards(*, kind: SelectorKind | None = None) -> tuple[SelectorCard, ...]:
    """Return immutable cards in stable publication order."""

    if kind is not None and not isinstance(kind, SelectorKind):
        raise ValueError("selector catalog kind must be SelectorKind or None")
    return tuple(card for card in _CARDS if kind is None or card.kind is kind)


def get_selector_card(selector_id: str) -> SelectorCard:
    for card in _CARDS:
        if card.selector_id == selector_id:
            return card
    raise KeyError(selector_id)


__all__ = [
    "SelectorCard",
    "SelectorKind",
    "get_selector_card",
    "list_selector_cards",
]
