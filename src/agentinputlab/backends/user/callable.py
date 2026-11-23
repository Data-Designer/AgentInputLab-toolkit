"""User responder using an explicitly injected callable."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .base import UserRequest, UserResponder, UserResponse

UserClient = Callable[[UserRequest], UserResponse | str]


@dataclass(slots=True)
class CallableUserResponder(UserResponder):
    client: UserClient
    responder_id: str = "callable-user"
    revision: str = "unversioned"
    identity: Mapping[str, Any] = field(default_factory=dict)
    network: bool = False

    def __post_init__(self) -> None:
        if not self.revision or self.revision == "unversioned":
            raise ValidationError("callable user responder requires an explicit stable revision")
        if not self.identity:
            raise ValidationError("callable user responder requires explicit identity fields")
        self.identity = freeze_mapping(self.identity)

    def respond(self, request: UserRequest) -> UserResponse:
        value = self.client(request)
        if isinstance(value, UserResponse):
            if value.request_id != request.request_id:
                raise ValidationError("callable user response request_id does not match request")
            return value
        return UserResponse(
            request_id=request.request_id,
            text=str(value),
            responder_id=self.responder_id,
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "agentinputlab.callable_user.v1",
            "responder_id": self.responder_id,
            "revision": self.revision,
            "identity_hash": content_hash(self.identity),
            "network": self.network,
        }


__all__ = ["CallableUserResponder", "UserClient"]
