"""Opt-in GPT-OSS stop compatibility for separately reported rerun variants.

Legacy completion-style stops can end a Harmony analysis message before the
model reaches its final channel. This adapter applies the same stops to final
content only. It does not promote reasoning to an answer or increase any token
cap. The server may generate past the logical final stop; *all* server-reported
tokens are retained and charged. Historical results must not be relabeled with
this adapter. It is intentionally not enabled by the production builder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from agentinputlab.contracts.errors import ValidationError

from .base import GenerationRequest, GenerationResponse, ModelCapability
from .openai_compatible import OpenAICompatibleBackend

POLICY_ID = "gpt-oss.final-channel-stop.v1"


@dataclass(slots=True)
class GPTOSSFinalChannelStopBackend:
    """Keep stop strings out of Harmony analysis, with no automatic new call."""

    inner: OpenAICompatibleBackend

    def __post_init__(self) -> None:
        if (
            self.inner.model_id.casefold() != "gpt-oss-20b"
            or self.inner.deployment_identity.get("server") != "vllm"
        ):
            raise ValidationError("final-channel stop compatibility requires GPT-OSS/vLLM")

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        return self.inner.capabilities

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "gpt_oss_final_channel_stop",
            "policy_id": POLICY_ID,
            "inner": dict(self.inner.fingerprint()),
            "reasoning_is_final": False,
            "charge_all_generated_tokens": True,
        }

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not request.stop:
            return self.inner.generate(request)
        if any(not value for value in request.stop):
            raise ValidationError("final-channel stop entries must be non-empty")
        # Only logical stop handling changes. In particular max_tokens, seed,
        # messages, tools, temperature, and metadata are not regenerated.
        response = self.inner.generate(replace(request, stop=()))
        matches = [
            (response.text.find(value), index, value)
            for index, value in enumerate(request.stop)
            if value in response.text
        ]
        match = min(matches) if matches else None
        return replace(
            response,
            text=response.text[:match[0]] if match else response.text,
            finish_reason="stop" if match else response.finish_reason,
            raw={
                **dict(response.raw),
                "agentinputlab_final_channel_stop": {
                    "policy_id": POLICY_ID,
                    "requested_stop": list(request.stop),
                    "matched_stop": match[2] if match else None,
                    "provider_finish_reason": response.finish_reason,
                    "provider_usage_unchanged": True,
                    "new_call_added_by_adapter": False,
                },
            },
        )
