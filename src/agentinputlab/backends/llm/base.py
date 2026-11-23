"""Backend-neutral language-model request and response contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.enums import StrEnum
from agentinputlab.contracts.errors import CapabilityError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.contracts.state import Message


def _is_finite_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


class ModelCapability(StrEnum):
    CHAT = "chat"
    COMPLETION = "completion"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    TOOL_CALLING = "tool_calling"
    LOGPROBS = "logprobs"
    SEED = "seed"
    STREAMING = "streaming"


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str = ""
    messages: tuple[Message, ...] = ()
    system: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    stop: tuple[str, ...] = ()
    logprobs: bool = False
    top_logprobs: int | None = None
    response_format: Mapping[str, Any] | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        stop = tuple(self.stop)
        if not all(isinstance(message, Message) for message in messages):
            raise ValidationError("generation messages must be Message instances")
        if not all(isinstance(item, str) for item in stop):
            raise ValidationError("generation stop entries must be strings")
        if not isinstance(self.logprobs, bool):
            raise ValidationError("logprobs must be boolean")
        if self.top_logprobs is not None and (
            isinstance(self.top_logprobs, bool)
            or not isinstance(self.top_logprobs, int)
            or not 1 <= self.top_logprobs <= 20
        ):
            raise ValidationError("top_logprobs must be an integer from 1 to 20 or None")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValidationError("top_logprobs requires logprobs=True")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(
            self,
            "tools",
            tuple(freeze_mapping(tool) for tool in self.tools),
        )
        if self.response_format is not None:
            object.__setattr__(
                self,
                "response_format",
                freeze_mapping(self.response_format),
            )
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.prompt and not self.messages:
            raise ValidationError("generation request requires prompt or messages")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValidationError("max_tokens must be positive")
        if not _is_finite_nonnegative_number(self.temperature):
            raise ValidationError("temperature must be finite and non-negative")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValidationError("seed must be a non-negative integer or None")

    @property
    def request_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.input_tokens, self.output_tokens)
        ):
            raise ValidationError("usage token counts must be non-negative integers")
        if self.cost_usd is not None and not _is_finite_nonnegative_number(self.cost_usd):
            raise ValidationError("usage cost_usd must be finite and non-negative or None")


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    model: str = ""
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", freeze_mapping(self.raw))


@runtime_checkable
class LLMBackend(Protocol):
    model_id: str
    capabilities: frozenset[ModelCapability]

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate one response or raise a typed backend error."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return serializable identity fields used by balance audits."""


def require_capabilities(
    backend: LLMBackend,
    required: set[ModelCapability] | frozenset[ModelCapability],
) -> None:
    missing = set(required) - set(backend.capabilities)
    if missing:
        names = ", ".join(sorted(item.value for item in missing))
        raise CapabilityError(f"backend {backend.model_id!r} lacks capabilities: {names}")
