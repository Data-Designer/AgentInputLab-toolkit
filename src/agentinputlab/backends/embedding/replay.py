"""Offline replay and deterministic fixture embedding backends."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash

from .base import (
    EmbeddingBackend,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    validate_embedding_response,
)


@dataclass(slots=True)
class ReplayEmbeddingBackend(EmbeddingBackend):
    """Replay exact text-to-vector fixtures; missing texts fail closed."""

    vectors: Mapping[str, Sequence[float]]
    model_id: str = "replay-embedding"
    revision: str = "fixture-v1"
    _vectors: Mapping[str, tuple[float, ...]] = field(init=False, repr=False)
    _dimensions: int | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        converted = {
            text: tuple(float(value) for value in vector) for text, vector in self.vectors.items()
        }
        EmbeddingResponse(vectors=tuple(converted.values()), model=self.model_id)
        dimensions = {len(vector) for vector in converted.values()}
        if len(dimensions) > 1:
            raise ValidationError("replay embedding fixtures have inconsistent dimensions")
        self._vectors = converted
        self._dimensions = next(iter(dimensions), None)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        missing = [text for text in request.texts if text not in self._vectors]
        if missing:
            raise ValidationError(
                f"replay embedding fixture is missing {len(missing)} requested text(s)"
            )
        response = EmbeddingResponse(
            vectors=tuple(self._vectors[text] for text in request.texts),
            model=self.model_id,
            usage=EmbeddingUsage(input_count=len(request.texts)),
            metadata={"replay": True},
        )
        return validate_embedding_response(
            request,
            response,
            backend_dimensions=self._dimensions,
        )

    def fingerprint(self) -> Mapping[str, Any]:
        fixtures = [(text, self._vectors[text]) for text in sorted(self._vectors)]
        return {
            "backend": "agentinputlab.replay_embedding.v1",
            "model_id": self.model_id,
            "revision": self.revision,
            "dimensions": self._dimensions,
            "fixture_count": len(fixtures),
            "fixture_hash": content_hash(fixtures),
            "network": False,
        }


@dataclass(frozen=True, slots=True)
class DeterministicHashEmbeddingBackend(EmbeddingBackend):
    """Small dependency-free embedding backend for tests only."""

    model_id: str = "fixture-hash-embedding"
    dimensions: int = 32
    salt: str = "agentinputlab-v1"

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValidationError("hash embedding dimensions must be positive")

    def _vector(self, text: str) -> tuple[float, ...]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            payload = f"{self.salt}\0{counter}\0{text}".encode()
            digest = hashlib.sha256(payload).digest()
            for offset in range(0, len(digest), 4):
                integer = int.from_bytes(digest[offset : offset + 4], "big")
                values.append((integer / 0xFFFFFFFF) * 2.0 - 1.0)
                if len(values) == self.dimensions:
                    break
            counter += 1
        return tuple(values)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.dimensions is not None and request.dimensions != self.dimensions:
            raise ValidationError(
                f"hash embedding dimension is fixed at {self.dimensions}, "
                f"requested {request.dimensions}"
            )
        response = EmbeddingResponse(
            vectors=tuple(self._vector(text) for text in request.texts),
            model=self.model_id,
            usage=EmbeddingUsage(input_count=len(request.texts)),
            metadata={"fixture_only": True},
        )
        return validate_embedding_response(
            request,
            response,
            backend_dimensions=self.dimensions,
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "agentinputlab.hash_embedding.v1",
            "model_id": self.model_id,
            "dimensions": self.dimensions,
            "salt_hash": content_hash(self.salt),
            "fixture_only": True,
            "network": False,
        }


__all__ = ["DeterministicHashEmbeddingBackend", "ReplayEmbeddingBackend"]
