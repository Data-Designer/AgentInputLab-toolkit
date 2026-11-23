"""Dense embedding and vector-search runtime contracts.

These intentionally match the lightweight seams used by paper-faithful HyDE
adapters.  They remain distinct from :class:`EmbeddingBackend`, whose request
and response objects carry provider usage metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from .retrieval import RetrievalResult


@runtime_checkable
class DenseEmbedder(Protocol):
    embedder_id: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode texts in the provided order."""

    def fingerprint(self) -> Mapping[str, object]:
        """Return the model/tokenizer/revision identity."""


@runtime_checkable
class VectorSearcher(Protocol):
    searcher_id: str

    def search(self, vector: Sequence[float], *, top_k: int) -> Sequence[RetrievalResult]:
        """Search a fixed dense index with an already-computed query vector."""

    def fingerprint(self) -> Mapping[str, object]:
        """Return the corpus/index/metric identity."""


__all__ = ["DenseEmbedder", "VectorSearcher"]
