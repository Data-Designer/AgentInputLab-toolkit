"""Dependency-free Query2doc bridge models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.enums import StrEnum
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import freeze_mapping


class QueryGymMode(StrEnum):
    ZERO_SHOT = "zs"
    COT = "cot"
    FEW_SHOT = "few_shot"

    @classmethod
    def parse(cls, value: str | QueryGymMode) -> QueryGymMode:
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "zero_shot": cls.ZERO_SHOT,
            "zero": cls.ZERO_SHOT,
            "zs": cls.ZERO_SHOT,
            "cot": cls.COT,
            "fs": cls.FEW_SHOT,
            "fewshot": cls.FEW_SHOT,
            "few_shot": cls.FEW_SHOT,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValidationError(f"unsupported QueryGym Query2doc mode: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class OfficialQuery2DocRow:
    query_id: str
    query: str
    pseudo_doc: str
    split: str = ""

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValidationError("Query2doc cache row requires query_id")
        if not self.query:
            raise ValidationError("Query2doc cache row requires query")
        if not self.pseudo_doc:
            raise ValidationError("Query2doc cache row requires pseudo_doc")


@dataclass(frozen=True, slots=True)
class QueryGymRequest:
    query_id: str
    query: str
    mode: QueryGymMode
    demonstration_ids: tuple[str, ...] = ()
    temperature: float = 0.7
    max_tokens: int = 256
    seed: int = 0
    backend_fingerprint: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "demonstration_ids", tuple(self.demonstration_ids))
        object.__setattr__(self, "backend_fingerprint", freeze_mapping(self.backend_fingerprint))
        if not self.query:
            raise ValidationError("QueryGym request requires a non-empty query")
        if self.max_tokens < 1:
            raise ValidationError("QueryGym max_tokens must be positive")
        if self.temperature < 0:
            raise ValidationError("QueryGym temperature must be non-negative")
        if self.mode is QueryGymMode.FEW_SHOT and not self.demonstration_ids:
            raise ValidationError("QueryGym few-shot mode requires fixed demonstration IDs")

    def external_reservation(self) -> Mapping[str, int]:
        """One Query2doc reformulation always attempts exactly one model call."""

        return {"model_calls": 1}


@dataclass(frozen=True, slots=True)
class QueryGymRawResult:
    reformulated: str
    generated_text: str = ""
    prompt_hash: str = ""
    fallback: bool = False
    usage: Mapping[str, int | float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", freeze_mapping(self.usage))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
