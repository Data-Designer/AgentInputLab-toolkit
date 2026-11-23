"""Explicit raw-completion scoring for the controlled RQ-RAG tree.

One echoed completion supplies prompt and generated-token likelihoods. No extra
teacher-forcing request, hidden retry, chat-template substitution or local model
copy is made. This is a backend-neutral controlled profile, not released-7B
tokenizer or checkpoint parity.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm.base import GenerationRequest, GenerationResponse, ModelCapability
from agentinputlab.backends.llm.openai_compatible import (
    OpenAICompatibleBackend,
    _contains_exact_text,
    _parse_generation_response,
)
from agentinputlab.contracts import CapabilityError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable

_SAMPLING_EXTRAS = frozenset({
    "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty",
})


def _log_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("missing token log probability")
    result = float(value)
    if not math.isfinite(result) or result > 0:
        raise ValueError("token log probability must be finite and non-positive")
    return result


def echoed_path_score(data: Mapping[str, Any], prompt: str) -> tuple[str, dict[str, Any]]:
    """Reject incomplete prompt likelihoods rather than call completion-only PPL full-path."""
    choice = data["choices"][0]
    text = choice["text"]
    if not isinstance(text, str) or not text.startswith(prompt):
        raise ValueError("echoed completion does not preserve the exact input prompt")
    usage, logs = data["usage"], choice["logprobs"]
    prompt_count, completion_count = usage["prompt_tokens"], usage["completion_tokens"]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
           for n in (prompt_count, completion_count)) or prompt_count < 1:
        raise ValueError("echoed completion has invalid native usage")
    values, top = logs["token_logprobs"], logs["top_logprobs"]
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
            or not isinstance(top, Sequence) or isinstance(top, (str, bytes))
            or len(values) != len(top)
            or not prompt_count < len(values) <= prompt_count + completion_count):
        raise ValueError("echoed completion does not contain full prompt and answer likelihoods")
    path_values = []
    for index, value in enumerate(values):
        if index == 0 and value is None:
            continue  # The unconditional first prompt token has no predecessor.
        path_values.append(_log_probability(value))
    if not path_values:
        raise ValueError("echoed path has no scored tokens")
    confidence = []
    for candidates in top[prompt_count:]:
        if not isinstance(candidates, Mapping) or not candidates:
            raise ValueError("answer likelihoods require non-empty top-token probabilities")
        confidence.append(math.exp(max(_log_probability(value) for value in candidates.values())))
    perplexity = math.exp(-math.fsum(path_values) / len(path_values))
    return text[len(prompt):], {
        "confidence": math.fsum(confidence) / len(confidence),
        "perplexity": perplexity,
        "perplexity_scope": "full_path",
        "scoring_profile": "raw_completion_echo_all_returned_path_tokens.v1",
        "source": "native_echoed_prompt_and_completion_logprobs",
        "prompt_tokens": prompt_count,
        "scored_path_tokens": len(path_values),
        "scored_answer_tokens": len(confidence),
        "unreturned_stop_tokens_excluded": prompt_count + completion_count - len(values),
        "first_unconditional_token_excluded": values[0] is None,
        "chat_template_applied": False,
        "released_7b_checkpoint_parity": False,
    }


@dataclass(slots=True)
class RQRAGCompletionBackend(OpenAICompatibleBackend):
    """Opt-in /completions client with exact full-path likelihood accounting.

    Callers supply the RQ-RAG serialized prompt. Chat messages, tools, JSON mode,
    unknown body overrides and absent native prompt logprobs fail closed. Every
    successful ``generate`` is exactly one physical API/model generation.
    """

    def __post_init__(self) -> None:
        OpenAICompatibleBackend.__post_init__(self)
        if set(self.extra_body) - _SAMPLING_EXTRAS:
            raise ValidationError("RQ-RAG raw completion accepts sampling extras only")
        self.capabilities = frozenset({
            ModelCapability.COMPLETION, ModelCapability.LOGPROBS, ModelCapability.SEED,
        })

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if (not request.prompt or request.messages or request.system or request.tools
                or request.response_format is not None):
            raise CapabilityError("RQ-RAG raw completion requires a prompt without chat/tools/JSON mode")
        payload = dict(to_jsonable({
            **dict(self.extra_body), "model": self.model_id, "prompt": request.prompt,
            "max_tokens": request.max_tokens, "temperature": request.temperature,
            "echo": True, "logprobs": max(5, request.top_logprobs or 0),
            "add_special_tokens": False,
        }))
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)
        key = self.api_key or os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json", **dict(self.extra_headers)}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data, failed = None, False
        try:
            data = self.transport(f"{self.base_url}/completions", headers, payload, self.timeout_seconds)
        except Exception:
            failed = True
        if failed:
            raise RuntimeError("RQ-RAG completion transport failed; no automatic retry")
        if key and _contains_exact_text(data, key):
            raise RuntimeError("RQ-RAG completion response contains the configured credential")
        response = None
        try:
            text, score = echoed_path_score(data, request.prompt)
            parsed = _parse_generation_response(data, fallback_model=self.model_id)
            response = GenerationResponse(text=text, model=parsed.model,
                finish_reason=parsed.finish_reason, usage=parsed.usage,
                raw={**dict(data), "rqrag_score": score})
        except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError):
            failed = True
        if failed or response is None:
            raise RuntimeError("RQ-RAG completion lacks valid full-path probabilities")
        return response

    def fingerprint(self) -> Mapping[str, Any]:
        return {**dict(OpenAICompatibleBackend.fingerprint(self)),
            "backend": "rqrag_raw_completion_echo",
            "scoring_profile": "raw_completion_echo_all_returned_path_tokens.v1",
            "chat_template_applied": False, "automatic_retry": False}
