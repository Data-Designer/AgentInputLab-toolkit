"""Versioned external-worker request and response envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationError
from .serialization import freeze_mapping


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    schema_version: int
    release_id: str
    request_id: str
    payload: Mapping[str, Any]
    dependency_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError(f"unsupported worker request schema: {self.schema_version}")
        object.__setattr__(self, "payload", freeze_mapping(self.payload))
        object.__setattr__(self, "dependency_hashes", freeze_mapping(self.dependency_hashes))


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    schema_version: int
    request_id: str
    status: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    error_code: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError(f"unsupported worker response schema: {self.schema_version}")
        if self.status not in {"ok", "error"}:
            raise ValidationError(f"invalid worker response status: {self.status}")
        object.__setattr__(self, "payload", freeze_mapping(self.payload))
