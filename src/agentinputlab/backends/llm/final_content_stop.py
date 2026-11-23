"""Explicit final-content stops for OpenAI-compatible reasoning responses.

The provider does not receive logical stop strings, so it cannot terminate
inside its separate reasoning channel. Apply stops only to the parsed final
text; keep ALL provider usage, even tokens generated past the logical stop.
This is opt-in and does not rewrite historical results or backend defaults.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from agentinputlab.contracts import ValidationError

from .base import GenerationRequest, GenerationResponse, ModelCapability
from .openai_compatible import OpenAICompatibleBackend

FINAL_CONTENT_STOP_POLICY_ID = "openai-compatible.final-content-stop.v1"


@dataclass(slots=True)
class FinalContentStopBackend:
    """Wrap a backend whose parser separates final content from raw reasoning.

    No second generation is performed to recover truncated/empty content.
    This wrapper exposes buffered ``generate`` only, not streaming. A generic
    text decoder that merges reasoning into text cannot establish this contract.
    """

    inner: OpenAICompatibleBackend

    def __post_init__(self) -> None:
        if not isinstance(self.inner, OpenAICompatibleBackend):
            raise ValidationError("final-content stops require an OpenAICompatibleBackend parser")
        if "stop" in self.inner.extra_body:
            raise ValidationError("configure logical stops on GenerationRequest, not extra_body")

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        return self.inner.capabilities - {ModelCapability.STREAMING}

    def fingerprint(self) -> Mapping[str, Any]:
        return {"backend": "final_content_stop", "policy_id": FINAL_CONTENT_STOP_POLICY_ID,
            "inner": dict(self.inner.fingerprint()), "reasoning_is_final": False,
            "charge_all_generated_tokens": True, "streaming": False}

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if "stop" in self.inner.extra_body:
            raise ValidationError("configure logical stops on GenerationRequest, not extra_body")
        if not request.stop:
            return self.inner.generate(request)
        if any(not value for value in request.stop):
            raise ValidationError("final-content stop entries must be non-empty")
        response = self.inner.generate(replace(request, stop=()))
        matches = [(response.text.find(stop), index, stop) for index, stop in enumerate(request.stop)
                   if stop in response.text]
        match = min(matches) if matches else None
        return replace(response, text=response.text[:match[0]] if match else response.text,
            finish_reason="stop" if match else response.finish_reason,
            raw={**dict(response.raw), "agentinputlab_final_content_stop": {
                "policy_id": FINAL_CONTENT_STOP_POLICY_ID, "requested_stop": list(request.stop),
                "matched_stop": match[2] if match else None,
                "provider_finish_reason": response.finish_reason,
                "provider_usage_unchanged": True, "new_call_added_by_adapter": False}})


__all__ = ["FINAL_CONTENT_STOP_POLICY_ID", "FinalContentStopBackend"]
