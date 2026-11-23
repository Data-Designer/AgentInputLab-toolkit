"""OpenAI-compatible backend for hosted APIs and self-hosted vLLM servers."""

from __future__ import annotations

import http.client
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import CapabilityError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

from . import responses_strict
from .base import (
    GenerationRequest,
    GenerationResponse,
    LLMBackend,
    ModelCapability,
    Usage,
)

Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_SENSITIVE_EXTRA_HEADERS = frozenset(
    {
        "api-key",
        "authorization",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-amz-security-token",
        "x-api-key",
        "x-auth-token",
        "x-goog-api-key",
    }
)
_TRANSPORT_ATTEMPT_POLICY = "single_transport_no_implicit_retry.v1"


class OpenAICompatibleTransportError(RuntimeError):
    """Sanitized single-attempt failure, optionally carrying an HTTP status.

    Usage can be unknown after a failed request. Callers must not interpret
    this exception as permission to retry or as evidence of zero token cost.
    Provider bodies, URLs and credentials are deliberately not retained.
    """

    def __init__(self, http_status: int | None = None) -> None:
        # Only a plain status integer may cross the untrusted transport seam.
        self.http_status = (
            http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        )
        message = "OpenAI-compatible transport failed"
        if self.http_status is not None:
            hints = {
                401: "check api_key or api_key_env",
                403: "check endpoint access and model permissions",
                404: "check base_url (usually ending in /v1) and model_id",
                429: "check provider quota or concurrency limits",
            }
            hint = hints.get(self.http_status, "check the provider or local server status")
            message += f" (HTTP {self.http_status}); {hint}; no automatic retry"
        super().__init__(message)


class _OpenAICompatibleHTTPStatusError(RuntimeError):
    """Sanitized status-only failure without remote response content."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"OpenAI-compatible endpoint returned HTTP {status}")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    opener = urllib.request.build_opener(_NoRedirectHandler())
    http_status: int | None = None
    endpoint_unavailable = False
    protocol_failure = False
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise RuntimeError("OpenAI-compatible response exceeds the size limit")
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # A remote endpoint can reflect Authorization values or other opaque
        # request material in its error body.  Never copy that untrusted body
        # into exceptions because experiment monitors may persist them.
        http_status = exc.code
        # Cleanup diagnostics are remote/transport-controlled too; the fixed
        # status-only error below remains the sole public message.
        with suppress(Exception):
            exc.close()
    except urllib.error.URLError:
        # ``reason`` is likewise remote/transport-controlled and may contain a
        # credential-bearing URL.  Keep the diagnostic intentionally generic.
        endpoint_unavailable = True
    except http.client.HTTPException:
        # Invalid status lines and incomplete HTTP frames can embed raw,
        # remote-controlled bytes in their exception text.
        protocol_failure = True
    # Raise after leaving the handler so neither ``__cause__`` nor
    # ``__context__`` retains the remote-controlled exception for traceback
    # loggers to render.
    if http_status is not None:
        raise _OpenAICompatibleHTTPStatusError(http_status)
    if endpoint_unavailable:
        raise RuntimeError("OpenAI-compatible endpoint is unavailable")
    if protocol_failure:
        raise RuntimeError("OpenAI-compatible endpoint returned an invalid HTTP response")
    raise RuntimeError("OpenAI-compatible transport failed without a diagnostic")


@dataclass(slots=True)
class OpenAICompatibleBackend(LLMBackend):
    model_id: str
    base_url: str
    deployment_identity: Mapping[str, Any] = field(default_factory=dict)
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 120.0
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    extra_body: Mapping[str, Any] = field(default_factory=dict, repr=False)
    capabilities: frozenset[ModelCapability] = frozenset(
        {
            ModelCapability.CHAT,
            ModelCapability.COMPLETION,
            ModelCapability.JSON_OBJECT,
            ModelCapability.TOOL_CALLING,
            ModelCapability.SEED,
        }
    )
    transport: Transport = _default_transport
    response_model_aliases: tuple[str, ...] = ()
    structured_output_transport: str = "chat_completions"

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValidationError("model_id must be a non-empty string matching the served model")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValidationError("timeout_seconds must be a finite positive number")
        if not isinstance(self.base_url, str) or any(c.isspace() for c in self.base_url):
            raise ValidationError("base_url must be an absolute http(s) URL")
        try:
            parsed = urllib.parse.urlparse(self.base_url)
            valid_address = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
            _ = parsed.port  # Validate malformed and out-of-range ports before any request.
        except ValueError:
            raise ValidationError("base_url must be a valid absolute http(s) URL") from None
        if not valid_address:
            raise ValidationError("base_url must be an absolute http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValidationError("base_url must not contain credentials, query, or fragment")
        if parsed.path.rstrip("/").endswith("/chat/completions"):
            raise ValidationError(
                "base_url is the API root (usually ending in /v1), not /chat/completions"
            )
        if not self.deployment_identity or not self.deployment_identity.get("revision"):
            raise ValidationError("OpenAI-compatible backend requires deployment_identity.revision")
        if not isinstance(self.extra_body, Mapping):
            raise ValidationError("extra_body must be an object")
        reserved = {
            "model", "messages", "temperature", "max_tokens", "max_length",
            "max_completion_tokens", "seed", "n", "stream",
        }
        conflicts = reserved.intersection(self.extra_body)
        # An explicit single choice is identical to this backend's contract.
        # Retain it on the wire for native configurations; branching still
        # requires a different request/accounting API and is rejected here.
        if type(self.extra_body.get("n")) is int and self.extra_body["n"] == 1:
            conflicts.discard("n")
        if conflicts:
            raise ValidationError(
                "extra_body must not override request-owned generation fields: "
                + ", ".join(sorted(conflicts))
                + "; set method parameters / GenerationRequest instead. "
                "max_tokens is the output cap, not total sequence max_length."
            )
        for name, value in self.extra_headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValidationError("OpenAI-compatible extra headers must contain strings")
            if name.casefold() in _SENSITIVE_EXTRA_HEADERS:
                raise ValidationError(
                    f"sensitive extra header {name!r} is forbidden; use api_key or api_key_env"
                )
        self.base_url = self.base_url.rstrip("/")
        self.deployment_identity = freeze_mapping(self.deployment_identity)
        self.extra_headers = freeze_mapping(self.extra_headers)
        self.extra_body = freeze_mapping(self.extra_body)
        aliases = self.response_model_aliases
        if (not isinstance(aliases, (tuple, list)) or
            any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in aliases) or
            len(set(aliases)) != len(aliases)):
            raise ValidationError("response_model_aliases must be a unique string array")
        self.response_model_aliases = tuple(aliases)
        self.capabilities = frozenset(self.capabilities)
        if self.structured_output_transport not in ("chat_completions", "responses_strict"):
            raise ValidationError("structured_output_transport must be chat_completions or responses_strict")
        if self.structured_output_transport == "responses_strict" and (
            ModelCapability.JSON_SCHEMA not in self.capabilities or "response_format" in self.extra_body
        ):
            raise ValidationError("responses_strict requires json_schema capability and a request-owned schema")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.tools and ModelCapability.TOOL_CALLING not in self.capabilities:
            raise CapabilityError(f"backend {self.model_id!r} does not support tools")
        if request.seed is not None and ModelCapability.SEED not in self.capabilities:
            raise CapabilityError(f"backend {self.model_id!r} does not support seed")
        if request.logprobs and ModelCapability.LOGPROBS not in self.capabilities:
            raise CapabilityError(f"backend {self.model_id!r} does not support logprobs")
        if (
            request.response_format
            and ModelCapability.JSON_OBJECT not in self.capabilities
            and ModelCapability.JSON_SCHEMA not in self.capabilities
        ):
            raise CapabilityError(f"backend {self.model_id!r} does not support response_format")
        if (request.response_format and request.response_format.get("type") == "json_schema"
                and ModelCapability.JSON_SCHEMA not in self.capabilities):
            raise CapabilityError(
                "This request requires json_schema; select an endpoint with that capability. "
                "JSON-object support alone cannot enforce the requested schema. No model call was made."
            )
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        if request.messages:
            for message in request.messages:
                item = {"role": message.role, "content": message.content}
                if message.name:
                    item["name"] = message.name
                messages.append(item)
        elif request.prompt:
            messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            **dict(self.extra_body),
            "model": self.model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.logprobs:
            payload["logprobs"] = True
            if request.top_logprobs is not None:
                payload["top_logprobs"] = request.top_logprobs
        if request.response_format:
            payload["response_format"] = dict(request.response_format)
        if request.tools:
            payload["tools"] = list(request.tools)

        # ``freeze_mapping`` recursively protects backend/request contracts,
        # so nested values may be MappingProxyType instances.  Materialize the
        # complete payload—not just its outer mapping—before handing it to a
        # custom transport or ``json.dumps``.
        payload = dict(to_jsonable(payload))
        strict_response = (self.structured_output_transport == "responses_strict"
                           and (request.response_format or {}).get("type") == "json_schema")
        if strict_response:
            try:
                payload = responses_strict.request_payload(payload)
            except (KeyError, TypeError, ValueError, RecursionError):
                raise CapabilityError("Request is outside the supported strict Responses schema or options; no model call was made") from None
        endpoint = "responses" if strict_response else "chat/completions"

        key = self.api_key or os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json", **dict(self.extra_headers)}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data: Mapping[str, Any] | None = None
        transport_failed = False
        http_status: int | None = None
        try:
            data = self.transport(
                f"{self.base_url}/{endpoint}",
                headers,
                payload,
                self.timeout_seconds,
            )
        except _OpenAICompatibleHTTPStatusError as exc:
            transport_failed = True
            if type(exc.status) is int and 100 <= exc.status <= 599:
                http_status = exc.status
        except Exception:
            # Custom transports are an extension boundary and may raise
            # exceptions containing request headers or provider bodies.  Do
            # not let their text or traceback context enter run artifacts.
            transport_failed = True
        if transport_failed:
            # A failed request may already have generated/billed tokens. Do
            # not hide another model attempt inside one budget reservation,
            # or replace its unknown usage with a successful retry's usage.
            raise OpenAICompatibleTransportError(http_status)
        if key and _contains_exact_text(data, key):
            raise RuntimeError("OpenAI-compatible response contains the configured credential")
        response: GenerationResponse | None = None
        schema_error = False
        try:
            response = (responses_strict.parse_response(data, payload, self.response_model_aliases) if strict_response
                        else _parse_generation_response(data, fallback_model=self.model_id))
        except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError, RecursionError):
            schema_error = True
        # Leave the handler before raising so remote-controlled values cannot
        # survive in an exception context rendered by a traceback logger.
        if schema_error or response is None:
            if strict_response:
                raise RuntimeError("Strict Responses result violates its schema, usage or model binding; no automatic retry")
            raise RuntimeError(
                "OpenAI-compatible response schema is invalid; require choices and explicit "
                "usage.prompt_tokens / usage.completion_tokens; missing usage is unknown, "
                "not zero; no automatic retry"
            )
        if self.response_model_aliases and response.model not in {self.model_id, *self.response_model_aliases}:
            raise RuntimeError("Provider returned a model outside the explicitly configured response identity aliases; no automatic retry")
        return response

    def fingerprint(self) -> Mapping[str, Any]:
        parsed = urllib.parse.urlparse(self.base_url)
        identity = {
            "backend": "openai_compatible",
            "model_id": self.model_id,
            "endpoint_origin": f"{parsed.scheme}://{parsed.netloc}",
            "base_url": self.base_url,
            "deployment_identity_hash": content_hash(self.deployment_identity),
            "capabilities": sorted(item.value for item in self.capabilities),
            "extra_body_hash": content_hash(self.extra_body),
            "extra_header_names": sorted(self.extra_headers),
            "extra_headers_hash": content_hash(self.extra_headers),
            "transport_attempt_policy": _TRANSPORT_ATTEMPT_POLICY,
        }
        revision = self.deployment_identity.get("revision")
        if isinstance(revision, str) and revision.strip():
            # Source-bound workers need the declared public revision as well as
            # its deployment hash. Never expose the entire operator mapping.
            identity["model_revision"] = revision
        if self.response_model_aliases:
            identity["response_model_aliases"] = list(self.response_model_aliases)
        if self.structured_output_transport != "chat_completions":
            identity["structured_output_transport"] = self.structured_output_transport
        return identity


def _parse_generation_response(
    data: Any,
    *,
    fallback_model: str,
) -> GenerationResponse:
    if not isinstance(data, Mapping):
        raise TypeError("response must be a mapping")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response choices must be a non-empty list")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise TypeError("response choice must be a mapping")
    message = choice.get("message")
    text: Any = None
    if message is not None:
        if not isinstance(message, Mapping):
            raise TypeError("response message must be a mapping")
        text = message.get("content")
    if text is None:
        text = choice.get("text", "")
    if not isinstance(text, str):
        raise TypeError("response text must be a string")
    model = data.get("model", fallback_model)
    if not isinstance(model, str):
        raise TypeError("response model must be a string")
    finish_reason = choice.get("finish_reason", "")
    if finish_reason is None:
        finish_reason = ""
    if not isinstance(finish_reason, str):
        raise TypeError("response finish_reason must be a string")
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        raise TypeError("response usage must be a mapping")
    return GenerationResponse(
        text=text,
        model=model,
        finish_reason=finish_reason,
        usage=Usage(
            input_tokens=_response_token_count(usage["prompt_tokens"]),
            output_tokens=_response_token_count(usage["completion_tokens"]),
        ),
        raw=data,
    )


def _response_token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("response token count must be a non-negative integer")
    return value


def _contains_exact_text(value: Any, needle: str, *, _seen: set[int] | None = None) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        seen = set() if _seen is None else _seen
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        return any(
            (isinstance(key, str) and needle in key)
            or _contains_exact_text(item, needle, _seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        seen = set() if _seen is None else _seen
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        return any(_contains_exact_text(item, needle, _seen=seen) for item in value)
    return False
