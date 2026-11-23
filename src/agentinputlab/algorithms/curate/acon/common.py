"""Shared controls and source-faithful helpers for ACON runtime bridges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    Message,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers.method import (
    dependency_fingerprint,
    nonnegative_integer,
    required_text,
)

ACON_UPSTREAM_COMMIT = "d63f9ae18959dc7215ff62899c94c5e8c56847ae"
ACON_HISTORY_PROTOCOL = "agentinputlab.acon.history.v2"
ACON_OBSERVATION_PROTOCOL = "agentinputlab.acon.observation.v2"
ACON_RESERVED_OPT_ARGS = frozenset({"task", "observation", "history"})


@dataclass(frozen=True, slots=True)
class ACONArtifactIdentity:
    """Frozen identities required for a fair ACON runtime comparison."""

    optimizer_model_revision: str
    target_model_revision: str
    tokenizer_revision: str
    prompt_artifact_hash: str
    guideline_artifact_hash: str
    optimizer_deployment_fingerprint_hash: str
    target_backend_fingerprint_hash: str
    tokenizer_fingerprint_hash: str

    def __post_init__(self) -> None:
        for name in (
            "optimizer_model_revision",
            "target_model_revision",
            "tokenizer_revision",
            "prompt_artifact_hash",
            "guideline_artifact_hash",
            "optimizer_deployment_fingerprint_hash",
            "target_backend_fingerprint_hash",
            "tokenizer_fingerprint_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"ACON {name} must not be empty")
            if name.endswith("_hash") and (
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValidationError(f"ACON {name} must be a lowercase SHA-256 digest")

    def as_payload(self) -> dict[str, str]:
        return {
            "optimizer_model_revision": self.optimizer_model_revision,
            "target_model_revision": self.target_model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "prompt_artifact_hash": self.prompt_artifact_hash,
            "guideline_artifact_hash": self.guideline_artifact_hash,
            "optimizer_deployment_fingerprint_hash": (self.optimizer_deployment_fingerprint_hash),
            "target_backend_fingerprint_hash": self.target_backend_fingerprint_hash,
            "tokenizer_fingerprint_hash": self.tokenizer_fingerprint_hash,
        }


def target_backend_fingerprint(
    backend: Any,
    identity: ACONArtifactIdentity,
) -> Mapping[str, Any]:
    """Bind the actual target backend deployment to the declared ACON artifact."""

    fingerprint = dependency_fingerprint(backend, label="ACON target backend", required=True)
    if content_hash(fingerprint) != identity.target_backend_fingerprint_hash:
        raise ValidationError(
            "ACON target backend fingerprint disagrees with the frozen artifact identity"
        )
    if identity.target_model_revision not in _fingerprint_text_values(fingerprint):
        raise ValidationError(
            "ACON target_model_revision is not attested by the target backend fingerprint"
        )
    return fingerprint


def validate_runtime_identity(
    payload: Mapping[str, Any],
    identity: ACONArtifactIdentity,
) -> tuple[str, str, Mapping[str, Any]]:
    """Validate worker profile, manifest and runtime-attestation echoes."""

    validate_artifact_echo(payload, identity)
    profile_id = required_text(payload, "profile_id", label="ACON response")
    manifest_hash = required_text(
        payload,
        "compatibility_manifest_sha256",
        label="ACON response",
    )
    if len(manifest_hash) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_hash
    ):
        raise AlgorithmOutputError("ACON compatibility manifest identity is invalid")
    attestation = payload.get("runtime_attestation")
    if not isinstance(attestation, Mapping):
        raise AlgorithmOutputError("ACON response requires runtime_attestation")
    expected = {
        "optimizer_deployment_fingerprint_hash": (identity.optimizer_deployment_fingerprint_hash),
        "target_backend_fingerprint_hash": identity.target_backend_fingerprint_hash,
        "tokenizer_fingerprint_hash": identity.tokenizer_fingerprint_hash,
    }
    if dict(attestation) != expected:
        raise AlgorithmOutputError("ACON runtime attestation disagrees with frozen artifacts")
    return profile_id, manifest_hash, attestation


def _fingerprint_text_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, str):
        values.add(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            values.update(_fingerprint_text_values(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            values.update(_fingerprint_text_values(item))
    return values


@dataclass(frozen=True, slots=True)
class ACONHistoryControls:
    summarization_threshold: int = -1
    preserve_last_k_turns: int = 1
    summary_rule: str = "accumulate"
    summary_interval: int = -1
    history_version: int = 1

    def __post_init__(self) -> None:
        _validate_threshold(self.summarization_threshold)
        if (
            isinstance(self.preserve_last_k_turns, bool)
            or not isinstance(self.preserve_last_k_turns, int)
            or self.preserve_last_k_turns < 0
        ):
            raise ValidationError("ACON preserve_last_k_turns must be non-negative")
        if self.summary_rule not in {"accumulate", "reset"}:
            raise ValidationError("ACON summary_rule must be accumulate or reset")
        if (
            isinstance(self.summary_interval, bool)
            or not isinstance(self.summary_interval, int)
            or (self.summary_interval != -1 and self.summary_interval < 1)
        ):
            raise ValidationError("ACON summary_interval must be -1 or positive")
        if (
            isinstance(self.history_version, bool)
            or not isinstance(self.history_version, int)
            or self.history_version != 1
        ):
            raise ValidationError(
                "pinned ACON MemoryManager does not wire history_version to V2; use version 1"
            )


@dataclass(frozen=True, slots=True)
class ACONObservationControls:
    summarization_threshold: int = -1
    context_index: int = -1
    add_previous_history_summary: bool = False
    observation_version: int = 1

    def __post_init__(self) -> None:
        _validate_threshold(self.summarization_threshold)
        if isinstance(self.context_index, bool) or not isinstance(self.context_index, int):
            raise ValidationError("ACON context_index must be an integer")
        if not isinstance(self.add_previous_history_summary, bool):
            raise ValidationError("ACON add_previous_history_summary must be boolean")
        if (
            isinstance(self.observation_version, bool)
            or not isinstance(self.observation_version, int)
            or self.observation_version != 1
        ):
            raise ValidationError(
                "pinned ACON MemoryManager does not wire obs_version to V2; use version 1"
            )


def validate_artifact_echo(payload: Mapping[str, Any], identity: ACONArtifactIdentity) -> None:
    for key, expected in identity.as_payload().items():
        if required_text(payload, key, label="ACON response") != expected:
            raise AlgorithmOutputError(f"ACON {key} mismatch")


def validate_threshold_response(
    payload: Mapping[str, Any],
    *,
    threshold: int,
) -> tuple[int, bool]:
    token_count = nonnegative_integer(
        payload.get("raw_input_tokens"), label="ACON raw_input_tokens"
    )
    expected = threshold == -1 or token_count > threshold
    triggered = payload.get("threshold_triggered")
    if not isinstance(triggered, bool):
        raise AlgorithmOutputError("ACON threshold_triggered must be boolean")
    if triggered is not expected:
        comparator = "always" if threshold == -1 else "strictly_greater_than"
        raise AlgorithmOutputError(
            f"ACON threshold decision must use upstream {comparator} semantics"
        )
    return token_count, triggered


def history_text(messages: tuple[Message, ...]) -> str:
    """Mirror ``MemoryManager.convert_llm_history_to_text`` for v1."""

    parts: list[str] = []
    for index, message in enumerate(messages):
        if index == 1 and message.role == "user":
            continue
        if message.role == "user":
            parts.append(f"USER:\n{message.content}\n\n")
        elif message.role == "assistant":
            parts.append(f"ASSISTANT:\n{message.content}\n\n")
    return "".join(parts).strip()


def preserved_turn_indices(
    messages: tuple[Message, ...], preserve_last_k_turns: int
) -> tuple[int, ...]:
    """Mirror ACON's assistant/user-pair scan without mutating session state."""

    if not messages or preserve_last_k_turns <= 0:
        return ()
    turns: list[int] = []
    index = len(messages) - 1
    count = 0
    while index >= 0 and count < preserve_last_k_turns:
        if messages[index].role != "user":
            index -= 1
            continue
        user_index = index
        assistant_index = index - 1
        while assistant_index >= 0 and messages[assistant_index].role != "assistant":
            assistant_index -= 1
        if assistant_index >= 0:
            turns[0:0] = (assistant_index, user_index)
            count += 1
            index = assistant_index - 1
        else:
            turns.insert(0, user_index)
            count += 1
            index -= 1
    first_user_content = messages[1].content if len(messages) > 1 else ""
    return tuple(
        item
        for item in turns
        if messages[item].role != "system"
        and not (messages[item].role == "user" and messages[item].content == first_user_content)
    )


def _validate_threshold(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValidationError("ACON summarization threshold must be -1 or non-negative")


__all__ = [
    "ACON_HISTORY_PROTOCOL",
    "ACON_OBSERVATION_PROTOCOL",
    "ACON_RESERVED_OPT_ARGS",
    "ACON_UPSTREAM_COMMIT",
    "ACONArtifactIdentity",
    "ACONHistoryControls",
    "ACONObservationControls",
    "history_text",
    "preserved_turn_indices",
    "target_backend_fingerprint",
    "validate_artifact_echo",
    "validate_runtime_identity",
    "validate_threshold_response",
]
