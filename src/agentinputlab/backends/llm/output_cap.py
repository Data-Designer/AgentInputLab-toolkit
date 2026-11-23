"""Explicit, fingerprinted per-call and named-stage output ceilings."""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import freeze_mapping

from .base import GenerationRequest, GenerationResponse, LLMBackend, ModelCapability


@dataclass(slots=True)
class PerCallOutputCapBackend:
    """Bound requests before transport; cumulative accounting stays with the run.

    Overrides use exact ``method_id:phase`` request metadata. Nothing is inferred
    from a total budget, and provider responses are never truncated or retried.
    """
    inner: LLMBackend
    max_output_tokens_per_call: int
    stage_output_caps: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        value = self.max_output_tokens_per_call
        if type(value) is not int or value < 1:
            raise ValidationError("max_output_tokens_per_call must be a positive integer")
        if not isinstance(self.inner, LLMBackend):
            raise ValidationError("per-call output cap requires an LLMBackend")
        if not isinstance(self.stage_output_caps, Mapping) or any(
            not isinstance(k, str) or ':' not in k or any(not p.strip() or p != p.strip() for p in k.split(':',1))
            or type(v) is not int or v < 1 for k, v in self.stage_output_caps.items()
        ):
            raise ValidationError("stage_output_caps requires method:phase keys and positive integer caps")
        self.stage_output_caps = freeze_mapping(self.stage_output_caps)

    @property
    def model_id(self):return self.inner.model_id

    @property
    def capabilities(self) -> frozenset[ModelCapability]:return self.inner.capabilities

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        stage=f"{request.metadata.get('method_id','')}:{request.metadata.get('phase','')}"
        cap=self.stage_output_caps.get(stage,self.max_output_tokens_per_call)
        if request.max_tokens > cap:
            request=replace(request,max_tokens=cap)
        return self.inner.generate(request)

    def fingerprint(self) -> Mapping[str, Any]:
        value=dict(backend='per_call_output_cap',model=self.model_id,
            policy_id='explicit-per-call-output-cap.v1',max_output_tokens_per_call=self.max_output_tokens_per_call,
            inner=dict(self.inner.fingerprint()))
        if self.stage_output_caps:
            value['stage_output_caps']=dict(self.stage_output_caps)
        return value
