"""Raw-completion transport for the pinned DecomP GPT3Generator seam."""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import to_jsonable

from .openai_compatible import OpenAICompatibleBackend, _contains_exact_text

_FIELDS = frozenset({"prompt", "engine", "temperature", "max_tokens", "top_p",
    "frequency_penalty", "presence_penalty", "stop", "n", "best_of", "logprobs"})


def validate_native_parameters(value: Mapping) -> dict:
    """Validate the explicit single-choice native profile without rewriting it."""
    if (not isinstance(value, Mapping) or set(value) - _FIELDS
            or _FIELDS - {"best_of"} - set(value)
            or not isinstance(value.get("prompt"), str) or not value["prompt"]
            or not isinstance(value.get("engine"), str) or not value["engine"]):
        raise ValidationError("invalid DecomP native completion fields")
    if (type(value["n"]) is not int or value["n"] != 1
            or (value.get("best_of") is not None and
                (type(value["best_of"]) is not int or value["best_of"] != 1))):
        raise ValidationError("DecomP live profile requires n=1 and best_of=1 or None")
    if type(value["max_tokens"]) is not int or value["max_tokens"] < 1:
        raise ValidationError("DecomP native max_tokens must be positive")
    for name, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1),
                               ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
        number = value[name]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or not lower <= number <= upper):
            raise ValidationError("invalid DecomP native sampling parameter")
    if value["top_p"] == 0:
        raise ValidationError("DecomP native top_p must be positive")
    stop = value["stop"]
    if not isinstance(stop, (list, tuple)) or not all(isinstance(s, str) and s for s in stop):
        raise ValidationError("DecomP native stop must be a string sequence")
    logs = value["logprobs"]
    if logs is not None and (type(logs) is not int or not 0 <= logs <= 20):
        raise ValidationError("invalid DecomP native logprobs")
    return to_jsonable(value)


@dataclass(slots=True)
class DecompCompletionBackend(OpenAICompatibleBackend):
    """Preserve original completion parameters and native branch-score inputs.

    The caller-bound model replaces the legacy engine identity. The native
    GPT3Generator still parses/sorts choices; no chat template, scoring rewrite,
    response cache, fallback, or automatic retry is introduced here.
    """

    def __post_init__(self):
        OpenAICompatibleBackend.__post_init__(self)
        if self.extra_body:
            raise ValidationError("DecomP native parameters cannot be overridden by extra_body")

    def complete_native(self, parameters: Mapping, *, seed: int) -> Mapping:
        value = validate_native_parameters(parameters)
        if type(seed) is not int or seed < 0:
            raise ValidationError("DecomP seed must be non-negative")
        payload = {k: v for k, v in value.items() if k != "engine"}
        payload.update(model=self.model_id, seed=seed)
        key = self.api_key or os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json", **dict(self.extra_headers)}
        if key:
            headers["Authorization"] = "Bearer " + key
        failed, response = False, None
        try:
            response = self.transport(self.base_url + "/completions", headers, payload, self.timeout_seconds)
        except Exception:
            failed = True
        if failed:
            raise RuntimeError("DecomP native completion transport failed; no retry")
        if key and _contains_exact_text(response, key):
            raise RuntimeError("DecomP completion response contains the configured credential")
        if not isinstance(response, Mapping):
            raise RuntimeError("DecomP native completion returned an invalid envelope")
        return response

    def fingerprint(self):
        return {**dict(OpenAICompatibleBackend.fingerprint(self)),
            "backend": "decomp_native_raw_completion.v1", "chat_template_applied": False,
            "legacy_engine_binding": "explicit_caller_model_override", "automatic_retry": False}
