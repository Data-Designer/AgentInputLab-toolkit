"""Paired success, harm, and effect summaries without third-party dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean


@dataclass(frozen=True, slots=True)
class PairedSummary:
    count: int
    mean_effect: float
    win_rate: float
    tie_rate: float
    harm_rate: float


def summarize_pairs(
    baseline: tuple[float, ...],
    treatment: tuple[float, ...],
    *,
    harm_threshold: float = 0.0,
) -> PairedSummary:
    if len(baseline) != len(treatment):
        raise ValueError("paired arrays must have equal length")
    if not baseline:
        raise ValueError("paired summary requires at least one observation")
    if not isfinite(harm_threshold) or harm_threshold < 0:
        raise ValueError("harm_threshold must be finite and non-negative")
    if not all(isfinite(value) for value in baseline + treatment):
        raise ValueError("paired scores must be finite")
    effects = tuple(after - before for before, after in zip(baseline, treatment, strict=True))
    return PairedSummary(
        count=len(effects),
        mean_effect=fmean(effects),
        win_rate=sum(effect > 0 for effect in effects) / len(effects),
        tie_rate=sum(effect == 0 for effect in effects) / len(effects),
        harm_rate=sum(effect < -harm_threshold for effect in effects) / len(effects),
    )
