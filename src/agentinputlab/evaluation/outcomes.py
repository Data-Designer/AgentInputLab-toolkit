"""Outcome records used by paired analyses and the leaderboard."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agentinputlab.contracts.enums import StrEnum


class FailureCategory(StrEnum):
    NONE = "none"
    ASSET_MISSING = "asset_missing"
    LICENSE_NOT_ACCEPTED = "license_not_accepted"
    ENVIRONMENT_BOOT_FAILURE = "environment_boot_failure"
    CAPABILITY_MISMATCH = "capability_mismatch"
    PARSER_FAILURE = "parser_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FALLBACK = "fallback"
    INTENT_DRIFT = "intent_drift"
    BALANCE_FAILURE = "balance_failure"
    UPSTREAM_ERROR = "upstream_error"


@dataclass(frozen=True, slots=True)
class Outcome:
    example_id: str
    release_id: str
    success: bool
    score: float | None
    failure: FailureCategory = FailureCategory.NONE
    metrics: Mapping[str, float] = field(default_factory=dict)
