"""Dataset split-overlap records used by contamination-aware evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OverlapRecord:
    left_split: str
    right_split: str
    shared_ids: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.shared_ids)


def detect_id_overlap(
    left_ids: Iterable[str],
    right_ids: Iterable[str],
    *,
    left_split: str,
    right_split: str,
) -> OverlapRecord:
    shared = tuple(sorted(set(left_ids) & set(right_ids)))
    return OverlapRecord(left_split, right_split, shared)
