"""Backend-neutral embedding request, response, and validation contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping


def _is_finite_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    texts: tuple[str, ...]
    normalize: bool = False
    dimensions: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "texts", tuple(self.texts))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not all(isinstance(text, str) for text in self.texts):
            raise ValidationError("embedding request texts must all be strings")
        if self.dimensions is not None and self.dimensions < 1:
            raise ValidationError("embedding request dimensions must be positive")

    @property
    def request_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    input_count: int = 0
    input_tokens: int | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.input_count, bool) or not isinstance(self.input_count, int):
            raise ValidationError("embedding usage input_count must be a non-negative integer")
        if self.input_count < 0:
            raise ValidationError("embedding usage input_count must be a non-negative integer")
        if self.input_tokens is not None and (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or self.input_tokens < 0
        ):
            raise ValidationError(
                "embedding usage input_tokens must be a non-negative integer or None"
            )
        if self.cost_usd is not None and not _is_finite_nonnegative_number(self.cost_usd):
            raise ValidationError(
                "embedding usage cost_usd must be finite and non-negative or None"
            )


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    vectors: tuple[tuple[float, ...], ...]
    model: str = ""
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vectors = tuple(tuple(float(value) for value in vector) for vector in self.vectors)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise ValidationError("embedding response vectors have inconsistent dimensions")
        if vectors and dimensions == {0}:
            raise ValidationError("embedding response vectors must not be empty")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValidationError("embedding response contains a non-finite value")

    @property
    def dimensions(self) -> int | None:
        return len(self.vectors[0]) if self.vectors else None


@runtime_checkable
class EmbeddingBackend(Protocol):
    model_id: str

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed a batch without changing input order."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return stable, secret-free identity fields."""


def normalize_vector(vector: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return tuple(value / norm for value in vector)


def validate_embedding_response(
    request: EmbeddingRequest,
    response: EmbeddingResponse,
    *,
    backend_dimensions: int | None = None,
) -> EmbeddingResponse:
    if len(response.vectors) != len(request.texts):
        raise ValidationError(
            "embedding response count does not match request: "
            f"{len(response.vectors)} != {len(request.texts)}"
        )
    required_dimensions = request.dimensions or backend_dimensions
    if required_dimensions is not None and any(
        len(vector) != required_dimensions for vector in response.vectors
    ):
        raise ValidationError(
            f"embedding response dimension does not match required {required_dimensions}"
        )
    if request.normalize:
        return EmbeddingResponse(
            vectors=tuple(normalize_vector(vector) for vector in response.vectors),
            model=response.model,
            usage=response.usage,
            metadata=response.metadata,
        )
    return response


__all__ = [
    "EmbeddingBackend",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "normalize_vector",
    "validate_embedding_response",
]
