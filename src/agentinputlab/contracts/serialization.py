"""Canonical serialization and hashing helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any


def deep_freeze_jsonable(value: Any) -> Any:
    """Detach and recursively freeze a canonically serializable value.

    Mutable caller-owned containers must never be able to change a state,
    dependency, trace, or patch hash after construction.
    """

    if dataclasses.is_dataclass(value):
        return deep_freeze_jsonable(to_jsonable(value))
    if isinstance(value, Enum):
        return deep_freeze_jsonable(value.value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mapping keys must be strings")
            frozen[key] = deep_freeze_jsonable(item)
        return MappingProxyType(frozen)
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not permitted in canonical serialized values")
    if isinstance(value, (tuple, list)):
        return tuple(deep_freeze_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported canonical value: {type(value).__name__}")


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a detached, recursively immutable string-keyed mapping."""

    frozen = deep_freeze_jsonable(dict(value or {}))
    if not isinstance(frozen, Mapping):  # pragma: no cover - construction invariant
        raise TypeError("freeze_mapping did not produce a mapping")
    return frozen


def to_jsonable(value: Any) -> Any:
    """Convert public dataclasses and common immutable containers to JSON values."""

    if dataclasses.is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not permitted in canonical serialized values")
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize without platform- or insertion-order-dependent variation."""

    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Return a SHA-256 content identity for a canonical value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
