"""Standard embedding seams and dependency-free fixture implementations."""

from .base import (
    EmbeddingBackend,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    normalize_vector,
    validate_embedding_response,
)
from .callable import CallableEmbeddingBackend, EmbeddingClient
from .replay import DeterministicHashEmbeddingBackend, ReplayEmbeddingBackend

__all__ = [
    "CallableEmbeddingBackend",
    "DeterministicHashEmbeddingBackend",
    "EmbeddingBackend",
    "EmbeddingClient",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "ReplayEmbeddingBackend",
    "normalize_vector",
    "validate_embedding_response",
]
