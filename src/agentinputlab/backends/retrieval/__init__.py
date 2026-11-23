"""Retrieval backend contracts and small deterministic implementations."""

from .base import Document, RetrievalResult, Retriever
from .dense import (
    DENSE_RETRIEVER_PROFILE_IDS,
    RETRIEVER_PROFILES,
    BatchTextEncoder,
    DenseIndexSummary,
    EmbeddingBackendEncoder,
    RetrieverProfile,
    SentenceTransformerDenseRetriever,
    build_retriever,
    retriever_profile,
)
from .memory import FunctionRetriever, InMemoryBM25Retriever, default_tokenize
from .sqlite_fts import SQLiteFTS5Retriever

__all__ = [
    "DENSE_RETRIEVER_PROFILE_IDS",
    "RETRIEVER_PROFILES",
    "BatchTextEncoder",
    "DenseIndexSummary",
    "Document",
    "EmbeddingBackendEncoder",
    "FunctionRetriever",
    "InMemoryBM25Retriever",
    "RetrievalResult",
    "Retriever",
    "RetrieverProfile",
    "SQLiteFTS5Retriever",
    "SentenceTransformerDenseRetriever",
    "build_retriever",
    "default_tokenize",
    "retriever_profile",
]
