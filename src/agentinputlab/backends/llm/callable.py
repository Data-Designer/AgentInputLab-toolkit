"""Dependency-free callable and deterministic replay backends."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash

from .base import (
    GenerationRequest,
    GenerationResponse,
    LLMBackend,
    ModelCapability,
    Usage,
)


@dataclass(slots=True)
class CallableBackend(LLMBackend):
    fn: Callable[[GenerationRequest], GenerationResponse | str | Mapping[str, Any]]
    model_id: str = "callable"
    revision: str = "unversioned"
    callable_id: str = ""
    capabilities: frozenset[ModelCapability] = frozenset(
        {ModelCapability.CHAT, ModelCapability.COMPLETION, ModelCapability.SEED}
    )
    calls: list[GenerationRequest] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.revision or self.revision == "unversioned":
            raise ValidationError("callable backend requires an explicit stable revision")
        if not self.callable_id:
            raise ValidationError("callable backend requires an explicit stable callable_id")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls.append(request)
        value = self.fn(request)
        if isinstance(value, GenerationResponse):
            return value
        if isinstance(value, str):
            return GenerationResponse(text=value, model=self.model_id)
        return GenerationResponse(
            text=str(value.get("text", "")),
            model=str(value.get("model", self.model_id)),
            finish_reason=str(value.get("finish_reason", "")),
            usage=Usage(**dict(value.get("usage", {}))),
            raw=dict(value),
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "callable",
            "model_id": self.model_id,
            "revision": self.revision,
            "callable_id": self.callable_id,
            "capabilities": sorted(item.value for item in self.capabilities),
        }


@dataclass(slots=True)
class ReplayBackend(LLMBackend):
    """Return recorded outputs in order and fail loudly if the cache is exhausted."""

    responses: Iterable[str | GenerationResponse]
    model_id: str = "replay"
    capabilities: frozenset[ModelCapability] = frozenset(
        {ModelCapability.CHAT, ModelCapability.COMPLETION, ModelCapability.SEED}
    )
    strict_request_hashes: tuple[str, ...] = ()
    _queue: deque[str | GenerationResponse] = field(init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)
    _fixture_hash: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        values = tuple(self.responses)
        self._queue = deque(values)
        self._fixture_hash = content_hash(values)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.strict_request_hashes:
            if self._calls >= len(self.strict_request_hashes):
                raise RuntimeError("replay request-hash fixture exhausted")
            expected = self.strict_request_hashes[self._calls]
            if request.request_hash != expected:
                raise ValueError(
                    f"replay request mismatch at {self._calls}: "
                    f"expected {expected}, got {request.request_hash}"
                )
        if not self._queue:
            raise RuntimeError("replay backend exhausted")
        self._calls += 1
        value = self._queue.popleft()
        if isinstance(value, GenerationResponse):
            return value
        return GenerationResponse(text=value, model=self.model_id, finish_reason="replay")

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "replay",
            "model_id": self.model_id,
            "strict": bool(self.strict_request_hashes),
            "fixture_hash": self._fixture_hash,
            "request_fixture_hash": content_hash(self.strict_request_hashes),
        }
