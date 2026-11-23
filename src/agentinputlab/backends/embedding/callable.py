"""Embedding backend using an explicitly injected callable client."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .base import (
    EmbeddingBackend,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    validate_embedding_response,
)

EmbeddingClient = Callable[
    [EmbeddingRequest],
    EmbeddingResponse | Iterable[Iterable[float]],
]


@dataclass(slots=True)
class CallableEmbeddingBackend(EmbeddingBackend):
    """Adapt a local function, SDK client, RPC bridge, or test double.

    The callable itself is never fingerprinted through ``repr``.  Callers must
    supply stable identity fields for any behavior that affects an experiment.
    """

    client: EmbeddingClient
    model_id: str
    revision: str = "unversioned"
    dimensions: int | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)
    network: bool = False

    def __post_init__(self) -> None:
        if not self.revision or self.revision == "unversioned":
            raise ValidationError("callable embedding requires an explicit stable revision")
        if not self.identity:
            raise ValidationError("callable embedding requires explicit identity fields")
        self.identity = freeze_mapping(self.identity)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raw = self.client(request)
        if isinstance(raw, EmbeddingResponse):
            response = raw
        else:
            response = EmbeddingResponse(
                vectors=tuple(tuple(float(value) for value in vector) for vector in raw),
                model=self.model_id,
                usage=EmbeddingUsage(input_count=len(request.texts)),
            )
        return validate_embedding_response(
            request,
            response,
            backend_dimensions=self.dimensions,
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "agentinputlab.callable_embedding.v1",
            "model_id": self.model_id,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "identity_hash": content_hash(self.identity),
            "network": self.network,
        }


__all__ = ["CallableEmbeddingBackend", "EmbeddingClient"]
