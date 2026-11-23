"""User-response backend contract for clarification and interactive benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash


@dataclass(frozen=True, slots=True)
class UserRequest:
    request_id: str
    case_id: str
    turn_index: int
    prompt: str
    state_hash: str = ""
    choices: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", tuple(self.choices))
        if not self.request_id:
            raise ValidationError("user request_id must be non-empty")
        if not self.case_id:
            raise ValidationError("user case_id must be non-empty")
        if self.turn_index < 0:
            raise ValidationError("user turn_index must be non-negative")

    @property
    def prompt_hash(self) -> str:
        return content_hash(
            {
                "prompt": self.prompt,
                "choices": self.choices,
                "state_hash": self.state_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class UserResponse:
    request_id: str
    text: str
    responder_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValidationError("user response request_id must be non-empty")


@runtime_checkable
class UserResponder(Protocol):
    responder_id: str

    def respond(self, request: UserRequest) -> UserResponse:
        """Return one response without consulting evaluator-only data."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return stable, secret-free identity fields."""


__all__ = ["UserRequest", "UserResponder", "UserResponse"]
