"""Deterministic JSONL preprocessing primitives."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from agentinputlab.contracts.serialization import canonical_json


def normalize_records(
    records: Iterable[dict[str, Any]],
    transform: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    id_field: str = "id",
) -> tuple[dict[str, Any], ...]:
    normalized = [transform(dict(record)) for record in records]
    if any(id_field not in record for record in normalized):
        raise ValueError(f"all normalized records require {id_field!r}")
    normalized.sort(key=lambda record: str(record[id_field]))
    if len({str(record[id_field]) for record in normalized}) != len(normalized):
        raise ValueError("normalized record IDs must be unique")
    return tuple(normalized)


def to_jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "\n".join(canonical_json(record) for record in records) + "\n"


def from_jsonl(content: str) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(line) for line in content.splitlines() if line.strip())
