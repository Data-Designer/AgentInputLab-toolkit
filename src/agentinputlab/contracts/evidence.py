"""Evidence types that keep synthetic retrieval surrogates distinct from sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import StrEnum
from .errors import ValidationError
from .serialization import freeze_mapping


class EvidenceKind(StrEnum):
    RETRIEVED_DOCUMENT = "retrieved_document"
    SYNTHETIC_QUERY_SURROGATE = "synthetic_query_surrogate"
    TOOL_OBSERVATION = "tool_observation"


@dataclass(frozen=True, slots=True)
class Evidence:
    kind: EvidenceKind
    text: str
    evidence_id: str
    source_revision: str = ""
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.text:
            raise ValidationError("evidence text must not be empty")
        if self.kind is EvidenceKind.RETRIEVED_DOCUMENT and (
            not self.evidence_id or not self.source_revision
        ):
            raise ValidationError("retrieved evidence requires document id and source revision")

    @property
    def citable(self) -> bool:
        return self.kind in {
            EvidenceKind.RETRIEVED_DOCUMENT,
            EvidenceKind.TOOL_OBSERVATION,
        }
