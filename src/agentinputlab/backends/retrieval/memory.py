"""Small deterministic retrievers for fixtures and controlled experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .base import Document, RetrievalResult, Retriever

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def default_tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN.findall(text))


@dataclass(slots=True)
class InMemoryBM25Retriever(Retriever):
    documents: Iterable[Document]
    retriever_id: str = "memory-bm25"
    k1: float = 1.2
    b: float = 0.75
    tokenizer: Callable[[str], tuple[str, ...]] = default_tokenize
    tokenizer_id: str = "agentinputlab.default_tokenize.v1"
    _documents: tuple[Document, ...] = field(init=False, repr=False)
    _term_counts: tuple[Counter[str], ...] = field(init=False, repr=False)
    _doc_freq: Counter[str] = field(init=False, repr=False)
    _avgdl: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (("k1", self.k1), ("b", self.b)):
            try:
                finite = math.isfinite(value)
            except (TypeError, OverflowError):
                finite = False
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not finite):
                raise ValidationError(f"BM25 {name} must be finite and numeric")
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValidationError("BM25 requires k1 > 0 and 0 <= b <= 1")
        if self.tokenizer is not default_tokenize and self.tokenizer_id == (
            "agentinputlab.default_tokenize.v1"
        ):
            raise ValidationError("custom BM25 tokenizer requires an explicit tokenizer_id")
        if not self.tokenizer_id:
            raise ValidationError("BM25 tokenizer_id must be non-empty")
        self._documents = tuple(self.documents)
        self._term_counts = tuple(
            Counter(self.tokenizer(f"{doc.title} {doc.text}")) for doc in self._documents
        )
        self._doc_freq = Counter()
        for counts in self._term_counts:
            self._doc_freq.update(counts.keys())
        self._avgdl = (
            sum(sum(counts.values()) for counts in self._term_counts) / len(self._term_counts)
            if self._term_counts
            else 0.0
        )

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        if top_k < 1:
            return ()
        query_terms = self.tokenizer(query)
        scored: list[tuple[float, str, int]] = []
        n_docs = len(self._documents)
        for index, counts in enumerate(self._term_counts):
            length = sum(counts.values())
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (1 - self.b + self.b * length / max(self._avgdl, 1e-9))
                score += idf * (tf * (self.k1 + 1)) / norm
            scored.append((score, self._documents[index].doc_id, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for rank, (score, _, index) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievalResult(
                    document=self._documents[index],
                    score=score,
                    rank=rank,
                    features={"bm25_score": score},
                )
            )
        return tuple(results)

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "retriever_id": self.retriever_id,
            "implementation": "agentinputlab.memory_bm25.v1",
            "corpus_hash": content_hash(self._documents),
            "document_count": len(self._documents),
            "k1": self.k1,
            "b": self.b,
            "tokenizer_id": self.tokenizer_id,
        }


@dataclass(slots=True)
class FunctionRetriever(Retriever):
    fn: Callable[[str, int], Iterable[RetrievalResult]]
    retriever_id: str = "function"
    identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValidationError("function retriever requires explicit stable identity")
        if not self.identity.get("implementation") or not self.identity.get("revision"):
            raise ValidationError(
                "function retriever identity requires implementation and revision"
            )
        self.identity = freeze_mapping(self.identity)

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        return tuple(self.fn(query, top_k))

    def fingerprint(self) -> Mapping[str, Any]:
        return {"retriever_id": self.retriever_id, **dict(self.identity)}
