"""Opt-in GPT-OSS final-channel transport for the six-field UDCS protocol.

The server renders its own pinned chat template, including the opening XML
delimiter. One raw completion follows; there is no chat-parser suppression,
analysis-to-answer fallback, retry, change to model weights, or extra budget.
This is an explicit decoding variant, not recovery of a failed native trial.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

from agentinputlab.contracts import ValidationError

from .base import GenerationRequest, GenerationResponse
from .openai_compatible import (
    OpenAICompatibleBackend,
    OpenAICompatibleTransportError,
    _contains_exact_text,
    _parse_generation_response,
)

POLICY_ID = "gpt_oss.udcs.final_channel_xml_completion.v2"
XML_REMAINDER = (
    r"[^<]*</think>\s*<u_request>\s*(?:0(?:\.[0-9]+)?|1(?:\.0+)?)\s*</u_request>"
    r"\s*<u_request_explanation>[^<]*</u_request_explanation>"
    r"\s*<action>[^<]+</action>\s*<confidence>\s*(?:0(?:\.[0-9]+)?|1(?:\.0+)?)\s*</confidence>"
    r"\s*<explanation>[^<]*</explanation>\s*"
)
RELEASES = frozenset({"clarify.udcs.repo_native_prompt_route",
                      "clarify.udcs.paper_spec_deterministic_route"})


@dataclass(slots=True)
class UDCSFinalCompletionBackend(OpenAICompatibleBackend):
    def __post_init__(self):
        OpenAICompatibleBackend.__post_init__(self)
        if self.model_id != "gpt-oss-20b" or not self.base_url.endswith("/v1"):
            raise ValidationError("UDCS final completion requires gpt-oss-20b and a /v1 API root")
        if (set(self.extra_body) - {"reasoning_effort"}
                or self.extra_body.get("reasoning_effort", "low") not in {"low", "medium", "high"}):
            raise ValidationError("UDCS final completion accepts only downstream reasoning_effort in extra_body")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        release = request.metadata.get("release_id")
        if release is None:
            # Ordinary downstream prediction retains the normal chat backend;
            # the XML grammar is specific to the UDCS intervention only.
            return OpenAICompatibleBackend.generate(self, request)
        if (release not in RELEASES or request.tools
                or request.response_format or request.stop or request.logprobs):
            raise ValidationError("UDCS final completion accepts only native six-field UDCS requests without tools/stops/JSON")
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        if request.messages:
            messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        elif request.prompt:
            messages.append({"role": "user", "content": request.prompt})
        messages.append({"role": "assistant", "content": "<think>"})
        rendered = {"model": self.model_id, "messages": messages,
                    "add_generation_prompt": False, "continue_final_message": True,
                    "return_token_strs": True}
        key = self.api_key or os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json", **dict(self.extra_headers)}
        if key:
            headers["Authorization"] = "Bearer " + key
        failed = False
        tokenized = data = None
        try:
            tokenized = self.transport(self.base_url[:-3] + "/tokenize", headers, rendered, self.timeout_seconds)
            tokens = tokenized.get("tokens") if isinstance(tokenized, Mapping) else None
            strings = tokenized.get("token_strs") if isinstance(tokenized, Mapping) else None
            if (not isinstance(tokens, list) or not tokens
                    or any(type(x) is not int or x < 0 for x in tokens)
                    or not isinstance(strings, list) or len(strings) != len(tokens)
                    or not all(isinstance(x, str) for x in strings)
                    or not "".join(strings).endswith("assistant<|channel|>final<|message|><think>")):
                raise ValueError("server did not render the required final-channel prefix")
            payload = {"model": self.model_id, "prompt": tokens,
                       "temperature": request.temperature, "max_tokens": request.max_tokens,
                       "skip_special_tokens": False,
                       "structured_outputs": {"regex": XML_REMAINDER}}
            if request.seed is not None:
                payload["seed"] = request.seed
            data = self.transport(self.base_url + "/completions", headers, payload, self.timeout_seconds)
        except Exception:
            failed = True
        if failed:
            raise OpenAICompatibleTransportError()
        if key and (_contains_exact_text(tokenized, key) or _contains_exact_text(data, key)):
            raise RuntimeError("UDCS completion response contains the configured credential")
        response = None
        invalid = False
        try:
            response = _parse_generation_response(data, fallback_model=self.model_id)
            text = response.text
            for suffix in ("<|return|>", "<|end|>", "<|endoftext|>"):
                if text.endswith(suffix):
                    text = text[:-len(suffix)]
            if any(marker in text for marker in ("<|channel|>", "<|start|>", "<|message|>", "<|return|>", "<|end|>")):
                raise ValueError("unexpected channel transition")
            response = replace(response, text="<think>" + text, raw={
                "tokenization": tokenized, "completion": data,
                "agentinputlab_decoding": {"policy_id": POLICY_ID,
                    "opening_delimiter_is_input_prefill": True,
                    "new_generative_requests": 1, "automatic_retry": False}})
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            invalid = True
        if invalid or response is None:
            raise RuntimeError("UDCS final completion returned an invalid envelope/channel; no retry")
        return response

    def fingerprint(self):
        return {**dict(OpenAICompatibleBackend.fingerprint(self)), "backend": POLICY_ID,
                "explicit_decoding_variant": True, "server_chat_template": True,
                "assistant_prefill": "<think>", "analysis_promoted_to_answer": False,
                "format_constraint": "native_six_XML_fields_no_answer_or_action_selected_by_adapter",
                "ordinary_downstream_requests": "unchanged_OpenAI_compatible_chat",
                "helper_reasoning_effort": "explicit_final_prefill_not_private_analysis",
                "automatic_retry": False}
