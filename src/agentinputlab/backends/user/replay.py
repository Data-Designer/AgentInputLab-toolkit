"""Strict, prompt-bound replay user responder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .base import UserRequest, UserResponder, UserResponse


@dataclass(frozen=True, slots=True)
class ReplayTurn:
    case_id: str
    turn_index: int
    response: str
    request_id: str | None = None
    prompt_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.request_id or f"{self.case_id}:{self.turn_index}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.case_id:
            raise ValidationError("replay user case_id must be non-empty")
        if self.turn_index < 0:
            raise ValidationError("replay user turn_index must be non-negative")
        if self.request_id == "":
            raise ValidationError("replay user request_id must be null or non-empty")


@dataclass(slots=True)
class ReplayUserResponder(UserResponder):
    turns: Iterable[ReplayTurn]
    responder_id: str = "replay-user"
    revision: str = "fixture-v1"
    _turns: tuple[ReplayTurn, ...] = field(init=False, repr=False)
    _by_key: Mapping[str, ReplayTurn] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        turns = tuple(self.turns)
        by_key = {turn.key: turn for turn in turns}
        if len(by_key) != len(turns):
            raise ValidationError("replay user script contains duplicate turn keys")
        self._turns = turns
        self._by_key = by_key

    def respond(self, request: UserRequest) -> UserResponse:
        turn = self._by_key.get(request.request_id)
        if turn is None:
            turn = self._by_key.get(f"{request.case_id}:{request.turn_index}")
        if turn is None:
            raise ValidationError(
                f"replay user script has no response for {request.request_id} "
                f"({request.case_id}:{request.turn_index})"
            )
        if turn.case_id != request.case_id or turn.turn_index != request.turn_index:
            raise ValidationError("replay user turn identity does not match request")
        if turn.prompt_hash is not None and turn.prompt_hash != request.prompt_hash:
            raise ValidationError("replay user prompt hash does not match request")
        return UserResponse(
            request_id=request.request_id,
            text=turn.response,
            responder_id=self.responder_id,
            metadata={"replay": True, **dict(turn.metadata)},
        )

    def fingerprint(self) -> Mapping[str, Any]:
        ordered = sorted(
            self._turns,
            key=lambda item: (item.case_id, item.turn_index, item.request_id or ""),
        )
        return {
            "backend": "agentinputlab.replay_user.v1",
            "responder_id": self.responder_id,
            "revision": self.revision,
            "turn_count": len(ordered),
            "script_hash": content_hash(ordered),
            "network": False,
        }


__all__ = ["ReplayTurn", "ReplayUserResponder"]
