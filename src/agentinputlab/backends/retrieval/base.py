"""Standard knowledge retrieval interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import freeze_mapping


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    corpus_revision: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValidationError("document requires a non-empty doc_id")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    document: Document
    score: float
    rank: int
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.document, Document):
            raise ValidationError("retrieval result requires a Document")
        if self.rank < 1:
            raise ValidationError("retrieval result rank must be positive")
        object.__setattr__(self, "features", freeze_mapping(self.features))


@runtime_checkable
class Retriever(Protocol):
    retriever_id: str

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        """Return deterministic ranked candidates."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return corpus/index/retriever identity for balance auditing."""
