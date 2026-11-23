"""Versioned index identity used by retrieval balance checks."""

from __future__ import annotations

from dataclasses import dataclass

from agentinputlab.contracts.serialization import content_hash


@dataclass(frozen=True, slots=True)
class IndexManifest:
    corpus_revision: str
    document_filter_hash: str
    chunker_hash: str
    embedder_revision: str
    index_revision: str
    metric: str

    @property
    def lineage_hash(self) -> str:
        return content_hash(self)
