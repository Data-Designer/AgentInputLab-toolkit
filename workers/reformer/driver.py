#!/usr/bin/env python3
"""Pinned, user-mounted ReFormeR inference worker.

No upstream source is vendored.  The worker verifies a caller-mounted checkout,
loads ``Reformulator`` from that checkout, and directly calls the official
``select_best_pattern`` and ``apply_pattern`` methods.  AgentInputLab only
supplies a bounded OpenAI-compatible model seam and materializes the documented
paper/repository output-semantics divergence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Protocol, TextIO


def fingerprint_aliases(identity):
    """Explicit aliases only; keep the standalone worker standard-library-only."""
    for _ in range(16):
        if not isinstance(identity, Mapping):return ()
        if "response_model_aliases" in identity:
            value = identity["response_model_aliases"]
            if (not isinstance(value, (tuple, list)) or
                any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in value) or
                len(set(value)) != len(value)):
                raise ValueError("response_model_aliases must be a unique string array")
            return tuple(value)
        identity = identity.get("inner", identity.get("backend"))
    raise ValueError("backend fingerprint nesting exceeds the supported depth")

UPSTREAM_COMMIT = "72e52450a922bc0051b5b39c3a6307186a555137"
METHOD_ID = "reformer"
PROTOCOL_VERSION = "agentinputlab.reformer.v2"
ADAPTER_REVISION = "unique-library-name-reference-and-failure-usage.v2"
MANIFEST_SCHEMA = "agentinputlab.reformer.compatibility.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
DEFAULT_PATTERN_PATH = "reformer/patterns/extracted_patterns.json"
PROMPT_PATH = "reformer/prompts.json"
CORE_PATH = "reformer/core/reformer.py"
PROMPT_MANAGER_PATH = "reformer/prompt_manager.py"
INDUCTION_PATH = "reformer/patterns/iterative_pattern_extraction.py"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CONTEXT_BLOCKS = 256
MAX_TEXT_CHARS = 1_000_000
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_REF_RE = re.compile(r"refs/[A-Za-z0-9._/-]+")
_ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
_ROOT_FIELDS = frozenset(
    {
        "method_id",
        "release_id",
        "upstream_commit",
        "state_hash",
        "payload",
        "dependency_fingerprints",
        "minimum_reservation",
        "protocol_version",
        "request_id",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "task_id",
        "query",
        "context",
        "top_document_count",
        "pattern_artifact_hash",
        "prompt_artifact_hash",
        "semantics",
        "pattern_mode",
        "manual_pattern_id",
        "source_license_acknowledged",
        "controls",
        "seed",
    }
)
_CONTROL_FIELDS = frozenset({"max_model_calls", "max_tokens", "max_cost_usd"})


class DriverError(RuntimeError):
    """Expected, secret-free worker failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        usage: Mapping[str, int | float] | None = None,
        provider_usage: Mapping[str, int] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.consumed = consumed
        self.usage = dict(usage) if usage is not None else None
        self.provider_usage = dict(provider_usage) if provider_usage is not None else None
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True, slots=True)
class CompatibilityPolicy:
    commit: str
    license_status: str
    license_file_present: bool
    license_evidence_path: str
    license_evidence_sha256: str
    require_clean_checkout: bool
    source_hashes: Mapping[str, str]
    profiles: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompatibilityPolicy:
        if value.get("schema_version") != MANIFEST_SCHEMA:
            raise DriverError("INVALID_MANIFEST", "ReFormeR manifest schema mismatch")
        upstream = _mapping(value.get("upstream"), "manifest upstream")
        source_files = _mapping(value.get("source_files"), "manifest source_files")
        profiles = _mapping(value.get("profiles"), "manifest profiles")
        if upstream.get("commit") != UPSTREAM_COMMIT:
            raise DriverError("INVALID_MANIFEST", "manifest commit is not approved")
        if upstream.get("license_spdx") != "NOASSERTION":
            raise DriverError("INVALID_MANIFEST", "ReFormeR license must remain NOASSERTION")
        if upstream.get("license_file_present") is not False:
            raise DriverError("INVALID_MANIFEST", "manifest must record the absent license file")
        clean = upstream.get("require_clean_checkout")
        if not isinstance(clean, bool):
            raise DriverError("INVALID_MANIFEST", "clean-checkout flag must be boolean")
        hashes: dict[str, str] = {}
        for raw_path, raw_hash in source_files.items():
            path = _safe_relative_path(raw_path, "manifest source path")
            hashes[path] = _hash(raw_hash, f"manifest hash for {path}")
        required = {
            "README.md",
            CORE_PATH,
            PROMPT_MANAGER_PATH,
            PROMPT_PATH,
            DEFAULT_PATTERN_PATH,
            INDUCTION_PATH,
        }
        if not required <= set(hashes):
            raise DriverError("INVALID_MANIFEST", "manifest omits required upstream sources")
        normalized_profiles: dict[str, Mapping[str, str]] = {}
        expected = {
            "rewrite.reformer.repo_native_r_only": ("r_only", "automatic"),
            "rewrite.reformer.paper_spec_q_concat_r": ("q_concat_r", "automatic"),
            "rewrite.reformer.controlled_r_only": ("r_only", "manual"),
            "rewrite.reformer.controlled_q_concat_r": ("q_concat_r", "manual"),
        }
        if set(profiles) != set(expected):
            raise DriverError("INVALID_MANIFEST", "manifest must define exactly four profiles")
        for release_id, identity in expected.items():
            item = _mapping(profiles[release_id], f"manifest profile {release_id}")
            _exact_keys(item, {"semantics", "pattern_mode"}, "manifest profile")
            if (item["semantics"], item["pattern_mode"]) != identity:
                raise DriverError("INVALID_MANIFEST", "manifest profile identity mismatch")
            normalized_profiles[release_id] = dict(item)
        evidence = _safe_relative_path(
            upstream.get("license_evidence_path"), "license evidence path"
        )
        return cls(
            commit=UPSTREAM_COMMIT,
            license_status="NOASSERTION",
            license_file_present=False,
            license_evidence_path=evidence,
            license_evidence_sha256=_hash(
                upstream.get("license_evidence_sha256"), "license evidence hash"
            ),
            require_clean_checkout=clean,
            source_hashes=hashes,
            profiles=normalized_profiles,
        )


def _fingerprint_model(value: Mapping[str, Any]) -> str | None:
    """Resolve model identity from native or AgentInputLab backend fingerprints."""

    for key in ("model", "model_id"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    nested = value.get("backend")
    if isinstance(nested, Mapping):
        return _fingerprint_model(nested)
    return None


@dataclass(frozen=True, slots=True)
class BackendConfig:
    base_url: str
    api_key_env: str
    model: str
    fingerprint: Mapping[str, Any]
    extra_body: Mapping[str, Any]
    timeout_seconds: float
    max_input_tokens_per_call: int
    max_output_tokens: int
    temperature: float
    input_cost_per_million: float
    output_cost_per_million: float

    @classmethod
    def from_json(cls, raw: str) -> BackendConfig:
        value = _mapping(_loads_strict_json(raw), "backend configuration")
        _exact_keys(
            value,
            {
                "base_url",
                "api_key_env",
                "model",
                "fingerprint",
                "timeout_seconds",
                "extra_body",
                "max_input_tokens_per_call",
                "max_output_tokens",
                "temperature",
                "input_cost_per_million",
                "output_cost_per_million",
            },
            "backend configuration",
        )
        model = _text(value["model"], "backend model")
        fingerprint = dict(_mapping(value["fingerprint"], "backend fingerprint"))
        if not fingerprint or _fingerprint_model(fingerprint) != model:
            raise DriverError(
                "INVALID_BACKEND_CONFIG", "backend fingerprint must bind the exact model"
            )
        try:fingerprint_aliases(fingerprint)
        except ValueError:
            raise DriverError("INVALID_BACKEND_CONFIG", "response_model_aliases must be an explicit unique string array") from None
        extra_body = dict(_mapping(value["extra_body"], "backend extra_body"))
        reserved = {"model", "messages", "temperature", "seed", "max_tokens"}
        if reserved & set(extra_body):
            raise DriverError(
                "INVALID_BACKEND_CONFIG", "backend extra_body overrides reserved fields"
            )
        _canonical_json(extra_body)
        api_key_env = _text(value["api_key_env"], "backend api_key_env")
        if _ENV_NAME_RE.fullmatch(api_key_env) is None:
            raise DriverError("INVALID_BACKEND_CONFIG", "api_key_env is invalid")
        temperature = _nonnegative_float(value["temperature"], "backend temperature")
        if temperature > 2:
            raise DriverError("INVALID_BACKEND_CONFIG", "backend temperature exceeds 2")
        return cls(
            base_url=_validate_base_url(_text(value["base_url"], "backend base_url")),
            api_key_env=api_key_env,
            model=model,
            fingerprint=fingerprint,
            extra_body=extra_body,
            timeout_seconds=_positive_float(value["timeout_seconds"], "backend timeout"),
            max_input_tokens_per_call=_positive_int(
                value["max_input_tokens_per_call"],
                "backend max_input_tokens_per_call",
            ),
            max_output_tokens=_positive_int(
                value["max_output_tokens"], "backend max_output_tokens"
            ),
            temperature=temperature,
            input_cost_per_million=_nonnegative_float(
                value["input_cost_per_million"], "backend input price"
            ),
            output_cost_per_million=_nonnegative_float(
                value["output_cost_per_million"], "backend output price"
            ),
        )


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None


class CompletionBackend(Protocol):
    def fingerprint(self) -> Mapping[str, Any]: ...

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        seed: int,
        temperature: float,
        max_output_tokens: int,
    ) -> CompletionResult: ...


class OpenAICompatibleBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def fingerprint(self) -> Mapping[str, Any]:
        return self.config.fingerprint

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        seed: int,
        temperature: float,
        max_output_tokens: int,
    ) -> CompletionResult:
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise DriverError("BACKEND_CREDENTIAL_UNAVAILABLE", "backend credential unavailable")
        body = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_output_tokens,
        }
        body.update(self.config.extra_body)
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=_canonical_json(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise DriverError(
                "BACKEND_CALL_FAILED",
                "backend completion request failed",
                consumed=True,
            ) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise DriverError(
                "BACKEND_RESPONSE_TOO_LARGE",
                "backend response exceeded limit",
                consumed=True,
            )
        provider_usage = None
        diagnostics: dict[str, Any] = {"response_sha256": _sha256_bytes(raw)}
        try:
            root = _mapping(_loads_strict_json(raw.decode("utf-8")), "backend response")
            usage = _mapping(root.get("usage"), "backend usage")
            provider_usage = {
                "input_tokens": _nonnegative_int(usage.get("prompt_tokens"), "prompt_tokens"),
                "output_tokens": _nonnegative_int(usage.get("completion_tokens"), "completion_tokens"),
            }
            if root.get("model") not in {None, self.config.model, *fingerprint_aliases(self.config.fingerprint)}:
                raise DriverError("BACKEND_IDENTITY_MISMATCH", "provider returned another model")
            choices = root.get("choices")
            if (
                not isinstance(choices, Sequence)
                or isinstance(choices, (str, bytes))
                or len(choices) != 1
            ):
                raise DriverError("BACKEND_OUTPUT_INVALID", "backend must return one choice")
            choice = _mapping(choices[0], "backend choice")
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason in {
                "stop", "length", "content_filter", "tool_calls"
            }:
                diagnostics["finish_reason"] = finish_reason
            message = _mapping(choice.get("message"), "message")
            text = message.get("content")
            diagnostics["content_type"] = type(text).__name__
            if text is None or (isinstance(text, str) and not text.strip()):
                diagnostics["failure_kind"] = (
                    "empty_final_at_output_limit" if finish_reason == "length" else "empty_final"
                )
                raise DriverError("BACKEND_OUTPUT_INVALID", "backend returned no final text")
            if not isinstance(text, str):
                raise DriverError("BACKEND_OUTPUT_INVALID", "backend content must be text")
            return CompletionResult(
                text=text,
                input_tokens=provider_usage["input_tokens"],
                output_tokens=provider_usage["output_tokens"],
                finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            )
        except (DriverError, UnicodeDecodeError) as exc:
            if isinstance(exc, DriverError):
                exc.consumed = True
                exc.provider_usage = provider_usage
                exc.diagnostics.update(diagnostics)
                raise
            raise DriverError(
                "BACKEND_OUTPUT_INVALID",
                "backend response must be UTF-8 JSON",
                consumed=True,
            ) from exc


@dataclass(frozen=True, slots=True)
class Controls:
    max_model_calls: int
    max_tokens: int
    max_cost_usd: float


@dataclass(frozen=True, slots=True)
class ValidatedRequest:
    request_id: str
    release_id: str
    query: str
    documents: tuple[Mapping[str, str], ...]
    pattern_hash: str
    semantics: str
    pattern_mode: str
    manual_pattern_id: str
    controls: Controls
    seed: int


class UsageTracker:
    def __init__(self, controls: Controls, config: BackendConfig) -> None:
        self.controls = controls
        self.config = config
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.stages: list[dict[str, Any]] = []

    def complete(
        self,
        backend: CompletionBackend,
        *,
        stage: str,
        messages: Sequence[Mapping[str, str]],
        seed: int,
    ) -> str:
        if self.model_calls >= self.controls.max_model_calls:
            raise DriverError("MODEL_CALL_BUDGET_EXCEEDED", "ReFormeR max_model_calls exhausted")
        token_ceiling = self.config.max_input_tokens_per_call + self.config.max_output_tokens
        remaining_tokens = self.controls.max_tokens - self.input_tokens - self.output_tokens
        if token_ceiling > remaining_tokens:
            raise DriverError(
                "TOKEN_BUDGET_EXCEEDED",
                "ReFormeR attested input/output ceiling exceeds the remaining token budget",
            )
        call_cost_ceiling = (
            self.config.max_input_tokens_per_call * self.config.input_cost_per_million
            + self.config.max_output_tokens * self.config.output_cost_per_million
        ) / 1_000_000
        if self.cost_usd + call_cost_ceiling > self.controls.max_cost_usd:
            raise DriverError(
                "COST_BUDGET_EXCEEDED",
                "ReFormeR attested provider-price ceiling exceeds the remaining cost budget",
            )
        self.model_calls += 1
        try:
            result = backend.complete(
                messages=messages,
                seed=seed,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
        except DriverError as exc:
            if exc.code == "BACKEND_CREDENTIAL_UNAVAILABLE" and not exc.consumed:
                self.model_calls -= 1
            else:
                exc.consumed = True
                if exc.provider_usage is not None:
                    self._add_usage(
                        exc.provider_usage["input_tokens"], exc.provider_usage["output_tokens"]
                    )
                self.stages.append(
                    {
                        "index": len(self.stages),
                        "stage": stage,
                        "seed": seed,
                        "failure_code": exc.code,
                        **exc.diagnostics,
                        "provider_usage": exc.provider_usage,
                    }
                )
                exc.usage = self.as_dict()
            raise
        if not isinstance(result, CompletionResult):
            raise DriverError(
                "BACKEND_OUTPUT_INVALID",
                "backend returned invalid completion",
                consumed=True,
                usage=self.as_dict(),
            )
        input_tokens = _nonnegative_int(result.input_tokens, "completion input_tokens")
        output_tokens = _nonnegative_int(result.output_tokens, "completion output_tokens")
        self._add_usage(input_tokens, output_tokens)
        if (
            input_tokens > self.config.max_input_tokens_per_call
            or output_tokens > self.config.max_output_tokens
        ):
            raise DriverError(
                "BACKEND_USAGE_ATTESTATION_EXCEEDED",
                "ReFormeR provider usage exceeded the startup token attestation",
                consumed=True,
                usage=self.as_dict(),
            )
        if self.input_tokens + self.output_tokens > self.controls.max_tokens:
            raise DriverError(
                "TOKEN_BUDGET_EXCEEDED",
                "ReFormeR max_tokens exceeded",
                consumed=True,
                usage=self.as_dict(),
            )
        if self.cost_usd > self.controls.max_cost_usd:
            raise DriverError(
                "COST_BUDGET_EXCEEDED",
                "ReFormeR max_cost_usd exceeded",
                consumed=True,
                usage=self.as_dict(),
            )
        if not isinstance(result.text, str):
            raise DriverError(
                "BACKEND_OUTPUT_INVALID",
                "backend content must be text",
                consumed=True,
                usage=self.as_dict(),
            )
        self.stages.append(
            {
                "index": len(self.stages),
                "stage": stage,
                "seed": seed,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "response_sha256": _sha256_bytes(result.text.encode("utf-8")),
                "finish_reason": result.finish_reason,
            }
        )
        if not result.text.strip():
            raise DriverError(
                "BACKEND_OUTPUT_INVALID",
                "backend returned no final text",
                consumed=True,
                usage=self.as_dict(),
            )
        return result.text

    def _add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += (
            input_tokens * self.config.input_cost_per_million
            + output_tokens * self.config.output_cost_per_million
        ) / 1_000_000

    def as_dict(self) -> dict[str, int | float]:
        return {
            "model_calls": self.model_calls,
            "retrieval_calls": 0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class _MessageTokenizer:
    """Preserve official messages while replacing only the vLLM serialization seam."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize is not False or add_generation_prompt is not True:
            raise DriverError("UPSTREAM_API_MISMATCH", "official chat-template call changed")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise DriverError("UPSTREAM_API_MISMATCH", "official messages are invalid")
        normalized: list[dict[str, str]] = []
        for message in messages:
            item = _mapping(message, "official prompt message")
            if set(item) != {"role", "content"}:
                raise DriverError("UPSTREAM_API_MISMATCH", "official message shape changed")
            normalized.append(
                {
                    "role": _text(item["role"], "official message role"),
                    "content": _text(item["content"], "official message content"),
                }
            )
        return normalized


class _OfficialLLMShim:
    def __init__(self, backend: CompletionBackend, tracker: UsageTracker, seed: int) -> None:
        self.backend = backend
        self.tracker = tracker
        self.seed = seed
        self.stage = ""
        self.fatal_error: DriverError | None = None
        self.raw_text: dict[str, str] = {}

    def generate(self, *, prompts, sampling_params):
        expected_sampling = {
            "max_tokens": self.tracker.config.max_output_tokens,
            "temperature": self.tracker.config.temperature,
        }
        try:
            observed_sampling = vars(sampling_params)
        except TypeError as exc:
            raise DriverError(
                "UPSTREAM_API_MISMATCH", "official sampling parameters changed"
            ) from exc
        if observed_sampling != expected_sampling:
            raise DriverError("UPSTREAM_API_MISMATCH", "official sampling parameters changed")
        if self.stage not in {"pattern_selection", "pattern_application"}:
            raise DriverError("UPSTREAM_API_MISMATCH", "official call occurred outside a stage")
        if not isinstance(prompts, Sequence) or len(prompts) != 1:
            raise DriverError("UPSTREAM_API_MISMATCH", "official generate call shape changed")
        messages = prompts[0]
        offset = 0 if self.stage == "pattern_selection" else 1
        try:
            text = self.tracker.complete(
                self.backend,
                stage=self.stage,
                messages=messages,
                seed=self.seed + offset,
            )
        except DriverError as exc:
            self.fatal_error = exc
            raise
        self.raw_text[self.stage] = text
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]

    def raise_fatal(self) -> None:
        if self.fatal_error is not None:
            raise self.fatal_error


class _BlockedNativeVLLMSymbol:
    """Fail closed if pinned source escapes the audited remote-model seam.

    The pinned ReFormeR module imports ``LLM`` and ``SamplingParams`` at module
    load time, even though AgentInputLab bypasses its constructor and injects
    ``_OfficialLLMShim`` plus an explicit sampling-parameter object.  Requiring
    the full vLLM package in every short-lived worker would therefore add an
    unused accelerator runtime and make the worker environment differ from the
    declared OpenAI-compatible dependency.  These symbols satisfy only the
    source-locked import.  Any future upstream path that tries to instantiate
    either native symbol fails instead of silently loading another model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise DriverError(
            "UNDECLARED_NATIVE_MODEL_LOAD",
            "pinned ReFormeR attempted to bypass the audited remote-model seam",
        )


class ReFormeRRuntime:
    def __init__(
        self,
        checkout: Path,
        backend_config: BackendConfig,
        backend: CompletionBackend,
        *,
        pattern_path: Path | None = None,
        source_license_acknowledged: bool = False,
        policy: CompatibilityPolicy | None = None,
    ) -> None:
        if source_license_acknowledged is not True:
            raise DriverError(
                "SOURCE_LICENSE_NOT_ACKNOWLEDGED",
                "ReFormeR source is NOASSERTION and requires operator acknowledgement",
            )
        self.policy = policy or load_compatibility_policy()
        self.checkout = verify_checkout(checkout, self.policy)
        self.backend_config = backend_config
        self.backend = backend
        if _canonical_json(backend.fingerprint()) != _canonical_json(backend_config.fingerprint):
            raise DriverError("BACKEND_IDENTITY_MISMATCH", "backend differs from startup config")
        selected_path = pattern_path or (self.checkout / DEFAULT_PATTERN_PATH)
        self.pattern_path = selected_path.expanduser().resolve(strict=True)
        if not self.pattern_path.is_file() or self.pattern_path.is_symlink():
            raise DriverError("PATTERN_ARTIFACT_UNAVAILABLE", "pattern artifact is unavailable")
        self.pattern_hash = _sha256_file(self.pattern_path)
        self.patterns = _load_patterns(self.pattern_path)
        self.prompt_hash = _sha256_file(self.checkout / PROMPT_PATH)
        self.reformulator_class, self.prompt_manager_class = _load_official_runtime(self.checkout)

    def run(self, value: Mapping[str, Any]) -> dict[str, Any]:
        request = validate_request(
            value,
            self.policy,
            self.backend_config,
            pattern_hash=self.pattern_hash,
            prompt_hash=self.prompt_hash,
        )
        tracker = UsageTracker(request.controls, self.backend_config)
        try:
            return self._run_validated(request, tracker)
        except DriverError as exc:
            # Parsing or library validation can fail after a paid model call.
            # Preserve that cost just as for transport-level failures.
            if tracker.model_calls:
                exc.consumed = True
                exc.usage = tracker.as_dict()
            exc.diagnostics.update(
                {"adapter_revision": ADAPTER_REVISION, "model_trace": tracker.stages}
            )
            raise

    def _run_validated(
        self, request: ValidatedRequest, tracker: UsageTracker
    ) -> dict[str, Any]:
        llm = _OfficialLLMShim(self.backend, tracker, request.seed)
        reformulator = self.reformulator_class.__new__(self.reformulator_class)
        reformulator.patterns = self.patterns
        reformulator.prompt_manager = self.prompt_manager_class(str(self.checkout / PROMPT_PATH))
        reformulator.tokenizer = _MessageTokenizer()
        reformulator.llm = llm
        reformulator.sampling_params = SimpleNamespace(
            max_tokens=self.backend_config.max_output_tokens,
            temperature=self.backend_config.temperature,
        )

        if request.pattern_mode == "manual":
            selected = _pattern_by_id(self.patterns, request.manual_pattern_id)
            parse_status = "manual"
            fallback_id = ""
        elif not request.documents:
            selected = reformulator.select_best_pattern(request.query, [])
            llm.raise_fatal()
            parse_status = "no_documents_first_pattern"
            fallback_id = _pattern_id(selected)
        else:
            llm.stage = "pattern_selection"
            selected = reformulator.select_best_pattern(
                request.query,
                [document["text"] for document in request.documents],
            )
            llm.raise_fatal()
            raw_selection = llm.raw_text.get("pattern_selection", "")
            parsed = _try_json(raw_selection)
            if isinstance(parsed, Mapping):
                parse_status = "valid_json"
                fallback_id = ""
            else:
                parse_status = "first_pattern_fallback"
                fallback_id = _pattern_id(self.patterns[0])
        raw_selected_hash = _canonical_hash(selected)
        name_reference = isinstance(selected, Mapping) and set(selected) == {"pattern_name"}
        selected = _require_library_pattern(selected, self.patterns)
        selection_match = (
            "unique_name_reference" if name_reference else
            "exact" if raw_selected_hash == _canonical_hash(selected) else "application_fields"
        )
        llm.stage = "pattern_application"
        reformulation = reformulator.apply_pattern(request.query, selected)
        llm.raise_fatal()
        if not isinstance(reformulation, str) or not reformulation.strip():
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "official application returned empty text")
        pattern_id = _pattern_id(selected)
        pattern_json = _canonical_json(selected)
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "payload": {
                "adapter_revision": ADAPTER_REVISION,
                "selection_library_match": selection_match,
                "raw_selected_pattern_hash": raw_selected_hash,
                "pattern_library_hash": self.pattern_hash,
                "prompt_artifact_hash": self.prompt_hash,
                "selected_pattern_id": pattern_id,
                "selected_pattern": _text(
                    selected.get("transformation_rule"), "selected transformation_rule"
                ),
                "selected_pattern_hash": _sha256_bytes(pattern_json.encode("utf-8")),
                "selected_examples_hash": _canonical_hash(selected.get("examples", [])),
                "reformulation": reformulation.strip(),
                "top_document_ids": [document["document_id"] for document in request.documents],
                "retrieval_state_hash": _canonical_hash(request.documents),
                "selection_parse_status": parse_status,
                "fallback_first_pattern_id": fallback_id,
                "semantics": request.semantics,
                "pattern_mode": request.pattern_mode,
                "manual_pattern_id": request.manual_pattern_id,
                "official_calls": (
                    ["Reformulator.apply_pattern"]
                    if request.pattern_mode == "manual"
                    else [
                        "Reformulator.select_best_pattern",
                        "Reformulator.apply_pattern",
                    ]
                    if request.documents
                    else [
                        "Reformulator.select_best_pattern:no_documents",
                        "Reformulator.apply_pattern",
                    ]
                ),
                "model_fingerprint": dict(self.backend_config.fingerprint),
                "model_trace": tracker.stages,
                "source_license": "NOASSERTION",
                "source_license_acknowledged": True,
                "vendored_upstream_source": False,
                "paper_result_parity_claimed": False,
            },
            "usage": tracker.as_dict(),
        }


def load_compatibility_policy(path: Path = DEFAULT_MANIFEST) -> CompatibilityPolicy:
    try:
        value = _loads_strict_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DriverError("MANIFEST_UNAVAILABLE", "compatibility manifest unavailable") from exc
    return CompatibilityPolicy.from_mapping(_mapping(value, "compatibility manifest"))


def verify_checkout(checkout: Path, policy: CompatibilityPolicy) -> Path:
    try:
        root = checkout.expanduser().resolve(strict=True)
    except OSError as exc:
        raise DriverError("CHECKOUT_UNAVAILABLE", "ReFormeR checkout unavailable") from exc
    if not root.is_dir():
        raise DriverError("CHECKOUT_UNAVAILABLE", "checkout must be a directory")
    if read_checkout_head(root) != policy.commit:
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "checkout is not at pinned commit")
    license_candidates = tuple(
        path
        for pattern in ("LICENSE*", "COPYING*", "NOTICE*")
        for path in root.glob(pattern)
        if path.is_file()
    )
    if license_candidates:
        raise DriverError(
            "LICENSE_STATE_MISMATCH",
            "pinned ReFormeR revision is expected to have no license file",
        )
    evidence = _checkout_file(root, policy.license_evidence_path)
    if _sha256_file(evidence) != policy.license_evidence_sha256:
        raise DriverError("LICENSE_EVIDENCE_MISMATCH", "README license evidence hash mismatch")
    if "License-MIT" not in evidence.read_text(encoding="utf-8", errors="strict"):
        raise DriverError("LICENSE_EVIDENCE_MISMATCH", "README MIT badge evidence is absent")
    for relative, expected in policy.source_hashes.items():
        source = _checkout_file(root, relative)
        if _sha256_file(source) != expected:
            raise DriverError("UPSTREAM_SOURCE_MISMATCH", f"source hash mismatch: {relative}")
    audit_source_symbols(root, policy)
    if policy.require_clean_checkout:
        _require_clean_checkout(root)
    return root


def audit_source_symbols(root: Path, policy: CompatibilityPolicy) -> dict[str, tuple[str, ...]]:
    del policy
    expected = {
        CORE_PATH: {
            "Reformulator": (
                "select_best_pattern",
                "apply_pattern",
                "reformulate_query",
            )
        },
        INDUCTION_PATH: {
            "IterativePatternExtractor": (
                "extract_patterns_from_batch",
                "run_iterative_extraction",
            )
        },
    }
    found: dict[str, tuple[str, ...]] = {}
    for relative, classes in expected.items():
        try:
            tree = ast.parse(_checkout_file(root, relative).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise DriverError("UPSTREAM_SOURCE_INVALID", f"cannot parse {relative}") from exc
        definitions = {
            node.name: {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        for class_name, methods in classes.items():
            if class_name not in definitions or not set(methods) <= definitions[class_name]:
                raise DriverError(
                    "UPSTREAM_SYMBOL_MISMATCH", f"official symbols changed: {relative}"
                )
            found[class_name] = methods
    return found


def validate_request(
    value: Mapping[str, Any],
    policy: CompatibilityPolicy,
    config: BackendConfig,
    *,
    pattern_hash: str,
    prompt_hash: str,
) -> ValidatedRequest:
    _exact_keys(value, set(_ROOT_FIELDS), "worker request")
    if value["method_id"] != METHOD_ID:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "request targets another method")
    release_id = _text(value["release_id"], "release_id")
    profile = policy.profiles.get(release_id)
    if profile is None:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "unknown ReFormeR profile")
    if value["upstream_commit"] != UPSTREAM_COMMIT:
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "request revision mismatch")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise DriverError("PROTOCOL_MISMATCH", "request protocol mismatch")
    request_id = _hash(value["request_id"], "request_id")
    _hash(value["state_hash"], "state_hash")
    identity = {key: value[key] for key in _ROOT_FIELDS if key != "request_id"}
    if request_id != _canonical_hash(identity):
        raise DriverError("REQUEST_ID_MISMATCH", "request_id is invalid")
    payload = _mapping(value["payload"], "worker payload")
    _exact_keys(payload, set(_PAYLOAD_FIELDS), "worker payload")
    if payload["source_license_acknowledged"] is not True:
        raise DriverError("SOURCE_LICENSE_NOT_ACKNOWLEDGED", "per-request license gate failed")
    semantics = _text(payload["semantics"], "semantics")
    pattern_mode = _text(payload["pattern_mode"], "pattern_mode")
    if semantics != profile["semantics"] or pattern_mode != profile["pattern_mode"]:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "request profile axes mismatch")
    requested_pattern_hash = _hash(payload["pattern_artifact_hash"], "pattern hash")
    if requested_pattern_hash != pattern_hash:
        raise DriverError("PATTERN_IDENTITY_MISMATCH", "pattern artifact hash mismatch")
    if payload["prompt_artifact_hash"] != prompt_hash:
        raise DriverError("PROMPT_IDENTITY_MISMATCH", "prompt artifact hash mismatch")
    dependencies = _mapping(value["dependency_fingerprints"], "dependencies")
    if set(dependencies) != {"backend", "retriever"}:
        raise DriverError("DEPENDENCY_IDENTITY_MISMATCH", "dependency fields mismatch")
    backend_fp = _mapping(dependencies["backend"], "backend fingerprint")
    if _canonical_json(backend_fp) != _canonical_json(config.fingerprint):
        raise DriverError("BACKEND_IDENTITY_MISMATCH", "request model differs from startup")
    retriever_fp = _mapping(dependencies["retriever"], "retriever fingerprint")
    if pattern_mode == "automatic" and not retriever_fp:
        raise DriverError("DEPENDENCY_IDENTITY_MISMATCH", "automatic profile requires retriever")
    manual_pattern_id = payload["manual_pattern_id"]
    if not isinstance(manual_pattern_id, str):
        raise DriverError("INVALID_SCHEMA", "manual_pattern_id must be text")
    if pattern_mode == "manual" and not manual_pattern_id.strip():
        raise DriverError("INVALID_SCHEMA", "manual profile requires pattern ID")
    if pattern_mode == "automatic" and manual_pattern_id:
        raise DriverError("INVALID_SCHEMA", "automatic profile cannot set manual pattern")
    top_count = _positive_int(payload["top_document_count"], "top_document_count")
    documents = _validate_documents(payload["context"], top_count)
    expected_calls = 2 if pattern_mode == "automatic" and documents else 1
    reservation = _mapping(value["minimum_reservation"], "minimum_reservation")
    if reservation != {"model_calls": expected_calls}:
        raise DriverError("INVALID_RESERVATION", "model-call reservation mismatches trajectory")
    controls = _validate_controls(payload["controls"])
    if controls.max_model_calls < expected_calls:
        raise DriverError("MODEL_CALL_BUDGET_EXCEEDED", "profile cannot fit max_model_calls")
    query = _text(payload["query"], "query")
    if len(query) > MAX_TEXT_CHARS:
        raise DriverError("REQUEST_TOO_LARGE", "query exceeds limit")
    task_id = payload["task_id"]
    if not isinstance(task_id, str):
        raise DriverError("INVALID_SCHEMA", "task_id must be text")
    return ValidatedRequest(
        request_id=request_id,
        release_id=release_id,
        query=query,
        documents=documents,
        pattern_hash=requested_pattern_hash,
        semantics=semantics,
        pattern_mode=pattern_mode,
        manual_pattern_id=manual_pattern_id,
        controls=controls,
        seed=_nonnegative_int(payload["seed"], "seed"),
    )


def _validate_documents(value: Any, count: int) -> tuple[Mapping[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DriverError("INVALID_SCHEMA", "context must be a sequence")
    if len(value) > MAX_CONTEXT_BLOCKS:
        raise DriverError("REQUEST_TOO_LARGE", "too many context blocks")
    documents: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value[:count]):
        item = _mapping(raw, f"context[{index}]")
        _exact_keys(item, {"text", "block_id", "source", "score", "metadata"}, "context block")
        text = _text(item["text"], f"context[{index}] text")
        document_id = _text(item["block_id"], f"context[{index}] block_id")
        if document_id in seen:
            raise DriverError("INVALID_SCHEMA", "context document IDs must be unique")
        seen.add(document_id)
        _canonical_json(item["metadata"])
        score = item["score"]
        if score is not None:
            _finite_float(score, f"context[{index}] score")
        documents.append({"document_id": document_id, "text": text})
    if len(_canonical_json(documents)) > MAX_TEXT_CHARS:
        raise DriverError("REQUEST_TOO_LARGE", "context text exceeds limit")
    return tuple(documents)


def _validate_controls(value: Any) -> Controls:
    item = _mapping(value, "controls")
    _exact_keys(item, set(_CONTROL_FIELDS), "controls")
    return Controls(
        max_model_calls=_positive_int(item["max_model_calls"], "max_model_calls"),
        max_tokens=_positive_int(item["max_tokens"], "max_tokens"),
        max_cost_usd=_nonnegative_float(item["max_cost_usd"], "max_cost_usd"),
    )


def _load_patterns(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        value = _loads_strict_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DriverError("PATTERN_ARTIFACT_UNAVAILABLE", "cannot read pattern artifact") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise DriverError("PATTERN_ARTIFACT_INVALID", "pattern artifact must be non-empty list")
    patterns: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"pattern[{index}]")
        _exact_keys(
            item,
            {"pattern_name", "description", "transformation_rule", "examples"},
            "pattern",
        )
        pattern_id = _text(item["pattern_name"], "pattern_name")
        if pattern_id in ids:
            raise DriverError("PATTERN_ARTIFACT_INVALID", "pattern names must be unique")
        ids.add(pattern_id)
        _text(item["description"], "pattern description")
        _text(item["transformation_rule"], "pattern transformation_rule")
        examples = item["examples"]
        if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
            raise DriverError("PATTERN_ARTIFACT_INVALID", "pattern examples must be a sequence")
        for pair in examples:
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) != 2
                or not all(isinstance(part, str) for part in pair)
            ):
                raise DriverError("PATTERN_ARTIFACT_INVALID", "pattern example must be text pair")
        patterns.append(dict(item))
    return tuple(patterns)


def _load_official_runtime(root: Path) -> tuple[type[Any], type[Any]]:
    namespace = f"_agentinputlab_reformer_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    reformer_package = f"{namespace}.reformer"
    core_package = f"{reformer_package}.core"
    for name, path in (
        (namespace, root),
        (reformer_package, root / "reformer"),
        (core_package, root / "reformer" / "core"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    missing = object()
    previous_vllm = sys.modules.get("vllm", missing)
    vllm_shim = types.ModuleType("vllm")
    vllm_shim.LLM = _BlockedNativeVLLMSymbol
    vllm_shim.SamplingParams = _BlockedNativeVLLMSymbol
    sys.modules["vllm"] = vllm_shim
    sys.dont_write_bytecode = True
    try:
        prompt_module = _load_module(
            f"{reformer_package}.prompt_manager", root / PROMPT_MANAGER_PATH
        )
        core_module = _load_module(f"{core_package}.reformer", root / CORE_PATH)
    except ModuleNotFoundError as exc:
        raise DriverError(
            "UPSTREAM_DEPENDENCY_UNAVAILABLE",
            f"pinned ReFormeR dependency unavailable: {exc.name or 'unknown'}",
        ) from exc
    except (ImportError, OSError, SyntaxError) as exc:
        raise DriverError("UPSTREAM_IMPORT_FAILED", "pinned ReFormeR import failed") from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if previous_vllm is missing:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = previous_vllm
    reformulator = getattr(core_module, "Reformulator", None)
    prompt_manager = getattr(prompt_module, "PromptManager", None)
    if not isinstance(reformulator, type) or not isinstance(prompt_manager, type):
        raise DriverError("UPSTREAM_SYMBOL_MISMATCH", "official runtime classes are missing")
    return reformulator, prompt_manager


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DriverError("UPSTREAM_IMPORT_FAILED", "cannot create upstream module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pattern_by_id(patterns: Sequence[Mapping[str, Any]], pattern_id: str):
    for pattern in patterns:
        if pattern.get("pattern_name") == pattern_id:
            return pattern
    raise DriverError("MANUAL_PATTERN_NOT_FOUND", "manual pattern ID is not in pattern set")


def _require_library_pattern(selected: Any, patterns: Sequence[Mapping[str, Any]]):
    if not isinstance(selected, Mapping):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "official selector returned no pattern")
    same_name = [p for p in patterns if p.get("pattern_name") == selected.get("pattern_name")]
    # A complete name-only JSON object references the pinned library, not a new
    # generated rule. Never fuzzy-match, resolve ambiguity or ignore conflicting
    # operative fields. This adapter revision records the resolution explicitly.
    if set(selected) == {"pattern_name"} and len(same_name) == 1:
        return same_name[0]
    # The pinned apply_pattern reads only transformation_rule and examples.
    # Description/annotation differences must not reject an otherwise identical
    # application. Objects supplying rules/examples must match them exactly.
    matches = [
        pattern
        for pattern in patterns
        if selected.get("pattern_name") == pattern.get("pattern_name")
        and selected.get("transformation_rule") == pattern.get("transformation_rule")
        and _canonical_json(selected.get("examples", []))
        == _canonical_json(pattern.get("examples", []))
    ]
    if len(matches) == 1:
        return matches[0]
    raise DriverError(
        "PATTERN_SELECTION_OUTSIDE_LIBRARY",
        "selector returned unknown pattern",
        diagnostics={
            "pattern_selection": {
                "same_name_candidates": len(same_name),
                "rule_matches_named_pattern": any(
                    p.get("transformation_rule") == selected.get("transformation_rule")
                    for p in same_name
                ),
                "examples_match_named_pattern": any(
                    _canonical_json(p.get("examples", []))
                    == _canonical_json(selected.get("examples", []))
                    for p in same_name
                ),
            }
        },
    )


def _pattern_id(pattern: Mapping[str, Any]) -> str:
    return _text(pattern.get("pattern_name"), "selected pattern_name")


def _try_json(value: str) -> Any:
    try:
        return _loads_strict_json(value)
    except DriverError:
        return None


def read_checkout_head(root: Path) -> str:
    git_entry = root / ".git"
    if git_entry.is_dir():
        git_dir = git_entry.resolve(strict=True)
    elif git_entry.is_file():
        raw = git_entry.read_text(encoding="utf-8").strip()
        if not raw.startswith("gitdir: "):
            raise DriverError("GIT_METADATA_INVALID", "checkout .git file malformed")
        candidate = Path(raw.removeprefix("gitdir: "))
        git_dir = (
            (root / candidate).resolve(strict=True)
            if not candidate.is_absolute()
            else candidate.resolve(strict=True)
        )
    else:
        raise DriverError("GIT_METADATA_INVALID", "checkout lacks Git metadata")
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if _COMMIT_RE.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        raise DriverError("GIT_METADATA_INVALID", "checkout HEAD malformed")
    ref = head.removeprefix("ref: ")
    if _REF_RE.fullmatch(ref) is None or ".." in PurePosixPath(ref).parts:
        raise DriverError("GIT_METADATA_INVALID", "checkout HEAD reference unsafe")
    for base in (git_dir, _common_git_dir(git_dir)):
        loose = _safe_child(base, ref)
        if loose.is_file():
            value = loose.read_text(encoding="utf-8").strip()
            if _COMMIT_RE.fullmatch(value):
                return value
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                commit, separator, name = line.partition(" ")
                if separator and name == ref and _COMMIT_RE.fullmatch(commit):
                    return commit
    raise DriverError("GIT_METADATA_INVALID", "checkout HEAD commit unavailable")


def _common_git_dir(git_dir: Path) -> Path:
    marker = git_dir / "commondir"
    if not marker.is_file():
        return git_dir
    candidate = Path(marker.read_text(encoding="utf-8").strip())
    return (
        (git_dir / candidate).resolve(strict=True)
        if not candidate.is_absolute()
        else candidate.resolve(strict=True)
    )


def _require_clean_checkout(root: Path) -> None:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        result = subprocess.run(
            (
                "git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DriverError("GIT_STATUS_FAILED", "clean-checkout check failed") from exc
    if result.returncode != 0:
        raise DriverError("GIT_STATUS_FAILED", "clean-checkout check failed")
    if result.stdout:
        raise DriverError("DIRTY_CHECKOUT", "checkout contains changes")


def serve(runtime: ReFormeRRuntime, input_stream: TextIO, output_stream: TextIO) -> None:
    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        request_id = ""
        try:
            if len(raw_line.encode("utf-8")) > MAX_REQUEST_BYTES:
                raise DriverError("REQUEST_TOO_LARGE", "request exceeds byte limit")
            value = _mapping(_loads_strict_json(raw_line), "JSONL request")
            if isinstance(value.get("request_id"), str):
                request_id = value["request_id"][:128]
            with redirect_stdout(sys.stderr):
                response = runtime.run(value)
        except DriverError as exc:
            response = _failure_response(request_id, exc)
        except Exception as exc:  # pragma: no cover
            response = _failure_response(
                request_id,
                DriverError("INTERNAL_WORKER_ERROR", f"worker failed with {type(exc).__name__}"),
            )
        output_stream.write(_canonical_json(response) + "\n")
        output_stream.flush()


def _failure_response(request_id: str, error: DriverError) -> dict[str, Any]:
    response: dict[str, Any] = {
        "status": "error",
        "request_id": request_id,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol_version": PROTOCOL_VERSION,
        "failure_code": error.code,
        "message": str(error),
        "consumed": error.consumed,
    }
    if error.usage is not None:
        response["usage"] = dict(error.usage)
    if error.diagnostics:
        response["diagnostics"] = dict(error.diagnostics)
    return response


def _checkout_file(root: Path, relative: str) -> Path:
    path = _safe_child(root, relative)
    if not path.is_file() or path.is_symlink():
        raise DriverError("UPSTREAM_FILE_UNAVAILABLE", f"pinned file unavailable: {relative}")
    return path


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise DriverError("UNSAFE_PATH", "path escapes fixed root")
    return candidate


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DriverError("INVALID_MANIFEST", f"{label} invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DriverError("INVALID_MANIFEST", f"{label} unsafe")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DriverError("NON_CANONICAL_JSON", "value is not canonical JSON") from exc


def _loads_strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError("INVALID_JSON", "input is not strict JSON") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DriverError("INVALID_SCHEMA", f"{label} must be string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DriverError("INVALID_SCHEMA", f"{label} fields do not match protocol")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriverError("INVALID_SCHEMA", f"{label} must be non-empty text")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise DriverError("INVALID_SCHEMA", f"{label} must be SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriverError("INVALID_SCHEMA", f"{label} must be positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be non-negative integer")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError("INVALID_SCHEMA", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DriverError("INVALID_SCHEMA", f"{label} must be finite")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be non-negative")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _nonnegative_float(value, label)
    if result <= 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be positive")
    return result


def _validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DriverError("INVALID_BACKEND_CONFIG", "base_url must be HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DriverError("INVALID_BACKEND_CONFIG", "base_url contains forbidden components")
    return value.rstrip("/")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--patterns", type=Path)
    parser.add_argument("--compatibility-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--backend-config-json")
    parser.add_argument("--acknowledge-noassertion-license", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    policy = load_compatibility_policy(args.compatibility_manifest)
    if args.verify_only:
        root = verify_checkout(args.checkout, policy)
        print(
            _canonical_json(
                {
                    "status": "ok",
                    "upstream_commit": UPSTREAM_COMMIT,
                    "source_license": "NOASSERTION",
                    "symbols": audit_source_symbols(root, policy),
                }
            )
        )
        return 0
    if not args.backend_config_json:
        parser.error("--backend-config-json is required unless --verify-only is used")
    config = BackendConfig.from_json(args.backend_config_json)
    sys.dont_write_bytecode = True
    with redirect_stdout(sys.stderr):
        runtime = ReFormeRRuntime(
            args.checkout,
            config,
            OpenAICompatibleBackend(config),
            pattern_path=args.patterns,
            source_license_acknowledged=args.acknowledge_noassertion_license,
            policy=policy,
        )
    serve(runtime, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
