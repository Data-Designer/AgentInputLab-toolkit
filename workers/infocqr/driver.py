#!/usr/bin/env python3
"""Pinned, user-mounted InfoCQR teacher/cache worker.

The driver never vendors InfoCQR. It verifies a caller-mounted checkout and
then either (a) calls the exact official ``generate_rewrite`` function through
a bounded OpenAI-compatible seam, or (b) replays one checksum-locked official
author JSON/JSONL artifact. The unreleased T5 student is represented by a fail-closed
checkpoint gate, not by substitute weights.
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
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
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

sys.dont_write_bytecode = True

UPSTREAM_COMMIT = "2953c461aa0c6817b1b7656dbd5168050227b31d"
METHOD_ID = "infocqr"
PROTOCOL_VERSION = "agentinputlab.infocqr.v1"
MANIFEST_SCHEMA = "agentinputlab.infocqr.compatibility.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CACHE_BYTES = 1024 * 1024 * 1024
MAX_CACHE_RECORDS = 1_000_000
MAX_TEXT_CHARS = 1_000_000
OFFICIAL_MODEL_NAME = "gpt-3.5-turbo"
OFFICIAL_MAX_TOKENS = 2560
POSTPROCESS_PATH = "post_processing.py"
POSTPROCESS_SOURCE_SHA256 = "7201ceaeb16beaaf3af874faf083ce4f64dd630636a2e302edf398cc4da8d3dd"
FUSE_PATH = "fuse_rewrite.py"
T5_TEST_PATH = "T5QR/T5test.py"
T5_TRAIN_PATH = "T5QR/T5train.py"
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
_PROFILES = frozenset({"rw_zsl", "rw_fsl", "t5_edit", "self_edit"})
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
        "operation",
        "task_id",
        "dataset_id",
        "dataset_revision",
        "split",
        "conversation_source",
        "conversation_no",
        "turn_no",
        "query",
        "new_context",
        "prompt_profile",
        "prompt_source_sha256",
        "cache_artifact_id",
        "cache_artifact_sha256",
        "allow_first_turn_truth_rewrite",
        "source_license_acknowledged",
        "retrieval_corpus_id",
        "profile_kind",
        "initial_rewrite",
        "initial_rewrite_source_sha256",
        "controls",
        "seed",
    }
)
_CONTROL_FIELDS = frozenset({"max_model_calls", "max_tokens", "max_cost_usd"})
_RELEASES = {
    "rewrite.infocqr.cached_rw_zsl": ("cached_replay", "rw_zsl"),
    "rewrite.infocqr.cached_rw_fsl": ("cached_replay", "rw_fsl"),
    "rewrite.infocqr.cached_edit": ("cached_replay", "t5_edit"),
    "rewrite.infocqr.live_teacher_external": ("live_teacher", "*"),
    "rewrite.infocqr.common_support_rw_zsl": ("live_teacher", "rw_zsl"),
    "rewrite.infocqr.t5_student_external": ("t5_student", "t5_student"),
}


class DriverError(RuntimeError):
    """Expected, secret-free worker failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        usage: Mapping[str, int | float] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.consumed = consumed
        self.usage = dict(usage) if usage is not None else None


@dataclass(frozen=True, slots=True)
class CompatibilityPolicy:
    commit: str
    license_status: str
    license_file_present: bool
    license_evidence_path: str
    license_evidence_sha256: str
    require_clean_checkout: bool
    source_hashes: Mapping[str, str]
    prompt_profiles: Mapping[str, Mapping[str, str]]
    known_caches: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompatibilityPolicy:
        if value.get("schema_version") != MANIFEST_SCHEMA:
            raise DriverError("INVALID_MANIFEST", "InfoCQR manifest schema mismatch")
        upstream = _mapping(value.get("upstream"), "manifest upstream")
        if upstream.get("commit") != UPSTREAM_COMMIT:
            raise DriverError("INVALID_MANIFEST", "manifest commit is not approved")
        if upstream.get("license_spdx") != "NOASSERTION":
            raise DriverError("INVALID_MANIFEST", "InfoCQR license must remain NOASSERTION")
        if upstream.get("license_file_present") is not False:
            raise DriverError("INVALID_MANIFEST", "manifest must record absent license file")
        clean = upstream.get("require_clean_checkout")
        if not isinstance(clean, bool):
            raise DriverError("INVALID_MANIFEST", "clean-checkout flag must be boolean")

        source_files = _mapping(value.get("source_files"), "manifest source_files")
        hashes = {
            _safe_relative_path(path, "manifest source path"): _hash(
                digest, f"manifest hash for {path}"
            )
            for path, digest in source_files.items()
        }
        required = {
            "README.md",
            "requirements.txt",
            "rewrite_chatgpt_zsl.py",
            "rewrite_chatgpt_inctx.py",
            "rewrite_chatgpt_editor.py",
            "rewrite_chatgpt_self_editor.py",
            POSTPROCESS_PATH,
            FUSE_PATH,
            "dataset_cleaning.py",
            T5_TEST_PATH,
            T5_TRAIN_PATH,
        }
        if not required <= set(hashes):
            raise DriverError("INVALID_MANIFEST", "manifest omits required InfoCQR sources")

        raw_profiles = _mapping(value.get("prompt_profiles"), "manifest prompt_profiles")
        if set(raw_profiles) != _PROFILES:
            raise DriverError("INVALID_MANIFEST", "manifest prompt profile set changed")
        profiles: dict[str, Mapping[str, str]] = {}
        for profile, raw_item in raw_profiles.items():
            item = _mapping(raw_item, f"prompt profile {profile}")
            _exact_keys(
                item,
                {"source_path", "source_sha256", "official_function"},
                "prompt profile",
            )
            path = _safe_relative_path(item["source_path"], "prompt source path")
            digest = _hash(item["source_sha256"], "prompt source hash")
            if hashes.get(path) != digest or item["official_function"] != "generate_rewrite":
                raise DriverError("INVALID_MANIFEST", "prompt profile identity mismatch")
            profiles[profile] = {
                "source_path": path,
                "source_sha256": digest,
                "official_function": "generate_rewrite",
            }

        raw_caches = _mapping(value.get("known_author_caches"), "known author caches")
        caches: dict[str, Mapping[str, str]] = {}
        for cache_id, raw_item in raw_caches.items():
            item = _mapping(raw_item, f"known cache {cache_id}")
            _exact_keys(item, {"path", "sha256", "profile", "split"}, "known cache")
            profile = _text(item["profile"], "known cache profile")
            if profile not in _PROFILES:
                raise DriverError("INVALID_MANIFEST", "known cache profile is unsupported")
            caches[_text(cache_id, "known cache id")] = {
                "path": _safe_relative_path(item["path"], "known cache path"),
                "sha256": _hash(item["sha256"], "known cache hash"),
                "profile": profile,
                "split": _text(item["split"], "known cache split"),
            }

        execution = _mapping(value.get("execution"), "manifest execution")
        if (
            execution.get("vendored_upstream_source") is not False
            or execution.get("t5_student_runtime_enabled") is not False
            or execution.get("released_t5_checkpoint_present") is not False
            or execution.get("paper_result_parity_claimed") is not False
            or execution.get("common_support_profile") != "controlled_adapter_visible_dialogue_only"
            or execution.get("post_processing_compatibility_hash") != hashes[POSTPROCESS_PATH]
        ):
            raise DriverError("INVALID_MANIFEST", "execution safety declarations changed")
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
            prompt_profiles=profiles,
            known_caches=caches,
        )


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
                "extra_body",
                "timeout_seconds",
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
        fingerprint_model = fingerprint.get("model_id", fingerprint.get("model"))
        if fingerprint_model is None:
            nested = fingerprint.get("backend")
            if isinstance(nested, Mapping):
                fingerprint_model = nested.get("model_id", nested.get("model"))
        if not fingerprint or fingerprint_model != model:
            raise DriverError(
                "INVALID_BACKEND_CONFIG",
                "backend fingerprint must bind the exact model",
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
        try:
            root = _mapping(_loads_strict_json(raw.decode("utf-8")), "backend response")
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
            message = _mapping(choice.get("message"), "backend message")
            text = message.get("content")
            if not isinstance(text, str):
                raise DriverError("BACKEND_OUTPUT_INVALID", "backend content must be text")
            usage = _mapping(root.get("usage"), "backend usage")
            return CompletionResult(
                text=text,
                input_tokens=_nonnegative_int(usage.get("prompt_tokens"), "prompt_tokens"),
                output_tokens=_nonnegative_int(usage.get("completion_tokens"), "completion_tokens"),
            )
        except (DriverError, UnicodeDecodeError) as exc:
            if isinstance(exc, DriverError):
                exc.consumed = True
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
    operation: str
    task_id: str
    dataset_id: str
    dataset_revision: str
    split: str
    conversation_source: str
    conversation_no: int
    turn_no: int
    query: str
    new_context: tuple[str, ...]
    prompt_profile: str
    prompt_source_sha256: str
    cache_artifact_id: str
    cache_artifact_sha256: str
    allow_first_turn_truth_rewrite: bool
    retrieval_corpus_id: str
    profile_kind: str
    initial_rewrite: str
    initial_rewrite_source_sha256: str
    controls: Controls
    seed: int
    backend_fingerprint: Mapping[str, Any]
    retriever_fingerprint: Mapping[str, Any]


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
        official_max_output_tokens: int,
    ) -> CompletionResult:
        if self.model_calls >= self.controls.max_model_calls:
            raise DriverError("MODEL_CALL_BUDGET_EXCEEDED", "InfoCQR max_model_calls exhausted")
        output_ceiling = min(
            official_max_output_tokens,
            self.config.max_output_tokens,
        )
        token_ceiling = self.config.max_input_tokens_per_call + output_ceiling
        remaining_tokens = self.controls.max_tokens - self.input_tokens - self.output_tokens
        if token_ceiling > remaining_tokens:
            raise DriverError(
                "TOKEN_BUDGET_EXCEEDED",
                "InfoCQR attested input/output ceiling exceeds the remaining token budget",
            )
        call_cost_ceiling = (
            self.config.max_input_tokens_per_call * self.config.input_cost_per_million
            + output_ceiling * self.config.output_cost_per_million
        ) / 1_000_000
        if self.cost_usd + call_cost_ceiling > self.controls.max_cost_usd:
            raise DriverError(
                "COST_BUDGET_EXCEEDED",
                "InfoCQR attested provider-price ceiling exceeds the remaining cost budget",
            )
        self.model_calls += 1
        try:
            result = backend.complete(
                messages=messages,
                seed=seed,
                temperature=self.config.temperature,
                max_output_tokens=output_ceiling,
            )
        except DriverError as exc:
            if exc.code == "BACKEND_CREDENTIAL_UNAVAILABLE" and not exc.consumed:
                self.model_calls -= 1
            else:
                exc.consumed = True
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
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += (
            input_tokens * self.config.input_cost_per_million
            + output_tokens * self.config.output_cost_per_million
        ) / 1_000_000
        if input_tokens > self.config.max_input_tokens_per_call or output_tokens > output_ceiling:
            raise DriverError(
                "BACKEND_USAGE_ATTESTATION_EXCEEDED",
                "InfoCQR provider usage exceeded the startup token attestation",
                consumed=True,
                usage=self.as_dict(),
            )
        if self.input_tokens + self.output_tokens > self.controls.max_tokens:
            raise DriverError(
                "TOKEN_BUDGET_EXCEEDED",
                "InfoCQR max_tokens exceeded",
                consumed=True,
                usage=self.as_dict(),
            )
        if self.cost_usd > self.controls.max_cost_usd:
            raise DriverError(
                "COST_BUDGET_EXCEEDED",
                "InfoCQR max_cost_usd exceeded",
                consumed=True,
                usage=self.as_dict(),
            )
        self.stages.append(
            {
                "index": len(self.stages),
                "stage": stage,
                "seed": seed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response_sha256": _sha256_bytes(result.text.encode("utf-8")),
            }
        )
        return result

    def as_dict(self) -> dict[str, int | float]:
        return {
            "model_calls": self.model_calls,
            "retrieval_calls": 0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class _LegacyChatCompletion:
    """Map the official pre-v1 OpenAI call to one pinned backend."""

    def __init__(
        self,
        backend: CompletionBackend,
        tracker: UsageTracker,
        *,
        stage: str,
        seed: int,
    ) -> None:
        self.backend = backend
        self.tracker = tracker
        self.stage = stage
        self.seed = seed
        self.fatal_error: DriverError | None = None
        self.call_count = 0

    def create(self, **kwargs: Any) -> Mapping[str, Any]:
        if self.fatal_error is not None:
            raise self.fatal_error
        if set(kwargs) != {"model", "messages", "temperature", "max_tokens", "n"}:
            raise DriverError("UPSTREAM_API_MISMATCH", "official ChatCompletion shape changed")
        if (
            kwargs["model"] != OFFICIAL_MODEL_NAME
            or kwargs["temperature"] != 0
            or kwargs["max_tokens"] != OFFICIAL_MAX_TOKENS
            or kwargs["n"] != 1
        ):
            raise DriverError("UPSTREAM_API_MISMATCH", "official generation controls changed")
        messages = _messages(kwargs["messages"])
        try:
            result = self.tracker.complete(
                self.backend,
                stage=self.stage,
                messages=messages,
                seed=self.seed,
                official_max_output_tokens=OFFICIAL_MAX_TOKENS,
            )
        except DriverError as exc:
            self.fatal_error = exc
            raise
        self.call_count += 1
        return {
            "model": self.tracker.config.model,
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
            "choices": [
                {
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        }

    def raise_fatal(self) -> None:
        if self.fatal_error is not None:
            raise self.fatal_error


@dataclass(frozen=True, slots=True)
class MountedCache:
    path: Path
    artifact_id: str
    sha256: str
    profile: str
    split: str
    artifact_format: str
    records: Mapping[tuple[str, int, int, str, str], Mapping[str, Any]]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        artifact_id: str,
        expected_sha256: str,
        profile: str,
        split: str,
    ) -> MountedCache:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise DriverError("CACHE_ARTIFACT_UNAVAILABLE", "cache artifact is unavailable")
        if resolved.stat().st_size > MAX_CACHE_BYTES:
            raise DriverError("CACHE_ARTIFACT_TOO_LARGE", "cache artifact exceeds size limit")
        digest = _sha256_file(resolved)
        if digest != _hash(expected_sha256, "cache artifact hash"):
            raise DriverError("CACHE_CHECKSUM_MISMATCH", "cache artifact checksum mismatch")
        if profile not in _PROFILES:
            raise DriverError("CACHE_SCHEMA_INVALID", "cache profile is unsupported")
        records: dict[tuple[str, int, int, str, str], Mapping[str, Any]] = {}

        def add_record(value: Any, record_no: int) -> None:
            if len(records) >= MAX_CACHE_RECORDS:
                raise DriverError("CACHE_ARTIFACT_TOO_LARGE", "cache record count exceeds limit")
            record = _validate_author_cache_record(
                _mapping(value, f"cache record {record_no}"), profile, record_no
            )
            key = (
                record["conversation_source"],
                record["conversation_no"],
                record["turn_no"],
                record["query"],
                _canonical_hash(record["new_context"]),
            )
            if key not in records:
                # Upstream appends some published artifacts and can repeat turns.
                # Byte order is checksum-bound; the first occurrence is a stable,
                # disclosed keyed view rather than a silent candidate merge.
                records[key] = record

        try:
            with resolved.open("r", encoding="utf-8") as handle:
                prefix = handle.read(4096)
            first = next((character for character in prefix if not character.isspace()), "")
            if first == "[":
                artifact_format = "json_array"
                value = _loads_strict_json(resolved.read_text(encoding="utf-8"))
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                    raise DriverError("CACHE_SCHEMA_INVALID", "cache JSON must contain an array")
                for record_no, item in enumerate(value, start=1):
                    add_record(item, record_no)
            else:
                artifact_format = "jsonl"
                with resolved.open("r", encoding="utf-8") as handle:
                    for line_no, raw in enumerate(handle, start=1):
                        if raw.strip():
                            add_record(_loads_strict_json(raw), line_no)
        except UnicodeDecodeError as exc:
            raise DriverError("CACHE_SCHEMA_INVALID", "cache must be UTF-8 JSON or JSONL") from exc
        if not records:
            raise DriverError("CACHE_SCHEMA_INVALID", "cache must contain records")
        return cls(
            path=resolved,
            artifact_id=_text(artifact_id, "cache artifact id"),
            sha256=digest,
            profile=profile,
            split=_text(split, "cache split"),
            artifact_format=artifact_format,
            records=records,
        )


class InfoCQRRuntime:
    def __init__(
        self,
        checkout: Path,
        backend_config: BackendConfig | None = None,
        backend: CompletionBackend | None = None,
        *,
        cache: MountedCache | None = None,
        source_license_acknowledged: bool = False,
        policy: CompatibilityPolicy | None = None,
    ) -> None:
        if source_license_acknowledged is not True:
            raise DriverError(
                "SOURCE_LICENSE_NOT_ACKNOWLEDGED",
                "InfoCQR source is NOASSERTION and requires operator acknowledgement",
            )
        self.policy = policy or load_compatibility_policy()
        self.checkout, self.checkout_identity_mode = verify_checkout(checkout, self.policy)
        self.cache = cache
        self.backend_config = backend_config
        self.backend = backend
        if (backend_config is None) != (backend is None):
            raise DriverError(
                "INVALID_BACKEND_CONFIG", "backend config and backend must be supplied together"
            )
        if (
            backend_config is not None
            and backend is not None
            and _canonical_json(backend.fingerprint())
            != _canonical_json(backend_config.fingerprint)
        ):
            raise DriverError("BACKEND_IDENTITY_MISMATCH", "backend differs from startup config")
        self.generators = _load_official_generators(self.checkout, self.policy)
        self.source_symbols = audit_source_symbols(self.checkout, self.policy)

    def run(self, value: Mapping[str, Any]) -> dict[str, Any]:
        request = validate_request(
            value,
            self.policy,
            backend_config=self.backend_config,
            mounted_cache=self.cache,
        )
        if request.operation == "t5_student":
            raise DriverError(
                "T5_CHECKPOINT_UNAVAILABLE",
                "InfoCQR publishes a T5 training recipe but no released student checkpoint",
            )
        if request.operation == "cached_replay":
            return self._run_cache(request)
        return self._run_live(request)

    def _run_cache(self, request: ValidatedRequest) -> dict[str, Any]:
        cache = self.cache
        if cache is None:  # pragma: no cover - validated before dispatch
            raise DriverError("CACHE_ARTIFACT_UNAVAILABLE", "cache is not mounted")
        key = (
            request.conversation_source,
            request.conversation_no,
            request.turn_no,
            request.query,
            _canonical_hash(request.new_context),
        )
        try:
            record = cache.records[key]
        except KeyError as exc:
            raise DriverError("CACHE_RECORD_UNAVAILABLE", "cache has no requested turn") from exc
        if record["query"] != request.query or tuple(record["new_context"]) != request.new_context:
            raise DriverError("CACHE_RECORD_IDENTITY_MISMATCH", "cache conversation view mismatch")
        first_turn_truth = request.turn_no == 1 and not request.new_context
        if first_turn_truth and not request.allow_first_turn_truth_rewrite:
            raise DriverError(
                "GOLD_DERIVED_FIRST_TURN_BLOCKED",
                "official first-turn cache copies Truth_rewrite and is disabled by policy",
            )
        candidate, historical_usage = _legacy_candidate(record["candidate"])
        initial = ""
        if request.prompt_profile in {"t5_edit", "self_edit"}:
            initial, _ = _legacy_candidate(record["initial_rewrite"])
        rewrite, postprocess_status = _official_postprocess_compat(
            candidate,
            question=request.query,
            turn_no=request.turn_no,
            editor=request.prompt_profile in {"t5_edit", "self_edit"},
            initial_rewrite=initial,
        )
        return self._response(
            request,
            rewrite=rewrite,
            raw_candidates=(candidate,),
            initial_rewrite=initial,
            postprocess_status=postprocess_status,
            calls=(
                "fuse_rewrite.record_extraction",
                "post_processing.pinned_branch_reconstruction",
            ),
            execution_label="compatibility_replay",
            usage={
                "model_calls": 0,
                "retrieval_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            usage_stages=(),
            historical_usage=historical_usage,
        )

    def _run_live(self, request: ValidatedRequest) -> dict[str, Any]:
        if self.backend_config is None or self.backend is None:
            raise DriverError("BACKEND_UNAVAILABLE", "live teacher backend is not configured")
        tracker = UsageTracker(request.controls, self.backend_config)
        line: dict[str, Any] = {
            "NewContext": list(request.new_context),
            "Question": request.query,
        }
        raw_candidates: list[str] = []
        official_calls: list[str] = []
        initial = request.initial_rewrite

        if request.prompt_profile in {"rw_zsl", "rw_fsl"}:
            candidate = self._call_official(
                request.prompt_profile,
                line,
                tracker,
                stage=request.prompt_profile,
                seed=request.seed,
            )
            raw_candidates.append(candidate)
            official_calls.append(
                f"{self.policy.prompt_profiles[request.prompt_profile]['source_path']}::generate_rewrite"
            )
            rewrite, status = _official_postprocess_compat(
                candidate,
                question=request.query,
                turn_no=request.turn_no,
                editor=False,
                initial_rewrite="",
            )
        elif request.prompt_profile == "self_edit":
            generated = self._call_official(
                "rw_fsl",
                line,
                tracker,
                stage="rw_fsl",
                seed=request.seed,
            )
            raw_candidates.append(generated)
            official_calls.append("rewrite_chatgpt_inctx.py::generate_rewrite")
            initial, initial_status = _official_postprocess_compat(
                generated,
                question=request.query,
                turn_no=request.turn_no,
                editor=False,
                initial_rewrite="",
            )
            line["GPT_rewrite"] = initial
            edited = self._call_official(
                "self_edit",
                line,
                tracker,
                stage="self_edit",
                seed=request.seed + 1,
            )
            raw_candidates.append(edited)
            official_calls.append("rewrite_chatgpt_self_editor.py::generate_rewrite")
            rewrite, edit_status = _official_postprocess_compat(
                edited,
                question=request.query,
                turn_no=request.turn_no,
                editor=True,
                initial_rewrite=initial,
            )
            status = f"initial:{initial_status};edit:{edit_status}"
        else:
            line["T5_rewrite"] = initial
            edited = self._call_official(
                "t5_edit",
                line,
                tracker,
                stage="t5_edit",
                seed=request.seed,
            )
            raw_candidates.append(edited)
            official_calls.append("rewrite_chatgpt_editor.py::generate_rewrite")
            rewrite, status = _official_postprocess_compat(
                edited,
                question=request.query,
                turn_no=request.turn_no,
                editor=True,
                initial_rewrite=initial,
            )
        return self._response(
            request,
            rewrite=rewrite,
            raw_candidates=tuple(raw_candidates),
            initial_rewrite=initial,
            postprocess_status=status,
            calls=tuple(official_calls),
            execution_label="official_function_calls",
            usage=tracker.as_dict(),
            usage_stages=tuple(tracker.stages),
            historical_usage={},
        )

    def _call_official(
        self,
        profile: str,
        line: Mapping[str, Any],
        tracker: UsageTracker,
        *,
        stage: str,
        seed: int,
    ) -> str:
        module = self.generators[profile]
        shim = _LegacyChatCompletion(self.backend, tracker, stage=stage, seed=seed)
        module.openai = SimpleNamespace(ChatCompletion=shim)
        module.time = SimpleNamespace(sleep=lambda _seconds: None)
        result = module.generate_rewrite(dict(line))
        shim.raise_fatal()
        if shim.call_count != 1:
            raise DriverError("UPSTREAM_API_MISMATCH", "official teacher call count changed")
        text, _usage = _legacy_candidate(result)
        if not text.strip():
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "official teacher returned empty text")
        return text

    def _response(
        self,
        request: ValidatedRequest,
        *,
        rewrite: str,
        raw_candidates: Sequence[str],
        initial_rewrite: str,
        postprocess_status: str,
        calls: Sequence[str],
        execution_label: str,
        usage: Mapping[str, int | float],
        usage_stages: Sequence[Mapping[str, Any]],
        historical_usage: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not rewrite.strip():
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "InfoCQR produced empty rewrite")
        cache_hash = self.cache.sha256 if self.cache is not None else ""
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "payload": {
                "rewrite": rewrite.strip(),
                "raw_candidates": list(raw_candidates),
                "initial_rewrite": initial_rewrite,
                "postprocess_status": postprocess_status,
                "postprocess_source_sha256": POSTPROCESS_SOURCE_SHA256,
                "prompt_profile": request.prompt_profile,
                "prompt_source_sha256": request.prompt_source_sha256,
                "operation": request.operation,
                "dataset_id": request.dataset_id,
                "dataset_revision": request.dataset_revision,
                "split": request.split,
                "conversation_source": request.conversation_source,
                "conversation_no": request.conversation_no,
                "turn_no": request.turn_no,
                "conversation_view_hash": _canonical_hash(
                    {"query": request.query, "new_context": request.new_context}
                ),
                "retrieval_corpus_id": request.retrieval_corpus_id,
                "cache_artifact_id": self.cache.artifact_id if self.cache else "",
                "cache_artifact_sha256": cache_hash,
                "cache_artifact_format": (
                    self.cache.artifact_format if self.cache is not None else ""
                ),
                "cache_duplicate_policy": (
                    f"first_occurrence_in_checksum_bound_{self.cache.artifact_format}"
                    if self.cache is not None
                    else ""
                ),
                "historical_usage": dict(historical_usage),
                "usage_stages": list(usage_stages),
                "execution_label": execution_label,
                "official_calls": (
                    list(calls) if execution_label == "official_function_calls" else []
                ),
                "reconstructed_calls": (
                    list(calls) if execution_label == "compatibility_replay" else []
                ),
                "official_requested_model": OFFICIAL_MODEL_NAME,
                "configured_teacher_model": (
                    self.backend_config.model if self.backend_config is not None else ""
                ),
                "profile_kind": request.profile_kind,
                "source_license_id": "NOASSERTION",
                "source_license_acknowledged": True,
                "checkout_identity_mode": self.checkout_identity_mode,
                "vendored_upstream_source": False,
                "t5_student_checkpoint_used": False,
                "paper_result_parity_claimed": False,
            },
            "usage": dict(usage),
        }


def load_compatibility_policy(path: Path = DEFAULT_MANIFEST) -> CompatibilityPolicy:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DriverError("INVALID_MANIFEST", "InfoCQR manifest is unavailable") from exc
    return CompatibilityPolicy.from_mapping(
        _mapping(_loads_strict_json(raw), "compatibility manifest")
    )


def verify_checkout(checkout: Path, policy: CompatibilityPolicy) -> tuple[Path, str]:
    root = checkout.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise DriverError("CHECKOUT_UNAVAILABLE", "InfoCQR checkout must be a real directory")
    for name in ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE"):
        if (root / name).exists():
            raise DriverError(
                "LICENSE_STATE_MISMATCH", "pinned checkout unexpectedly contains a license file"
            )
    for relative, expected in policy.source_hashes.items():
        path = _checkout_file(root, relative)
        if _sha256_file(path) != expected:
            raise DriverError("SOURCE_HASH_MISMATCH", f"source hash mismatch for {relative}")
    if _sha256_file(_checkout_file(root, policy.license_evidence_path)) != (
        policy.license_evidence_sha256
    ):
        raise DriverError("LICENSE_STATE_MISMATCH", "license evidence hash mismatch")

    git_marker = root / ".git"
    if not git_marker.exists():
        return root, "source_hash_set"
    head = read_checkout_head(root)
    if head != policy.commit:
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "InfoCQR checkout revision mismatch")
    if policy.require_clean_checkout:
        _require_clean_checkout(root)
    return root, "git_commit_and_source_hash_set"


def audit_source_symbols(root: Path, policy: CompatibilityPolicy) -> dict[str, tuple[str, ...]]:
    del policy
    expected_functions = {
        "rewrite_chatgpt_zsl.py": ("generate_rewrite",),
        "rewrite_chatgpt_inctx.py": ("generate_rewrite",),
        "rewrite_chatgpt_editor.py": ("generate_rewrite",),
        "rewrite_chatgpt_self_editor.py": ("generate_rewrite",),
        POSTPROCESS_PATH: ("main",),
        FUSE_PATH: ("main",),
        T5_TEST_PATH: ("main",),
        T5_TRAIN_PATH: ("main",),
    }
    result: dict[str, tuple[str, ...]] = {}
    for relative, names in expected_functions.items():
        tree = ast.parse(_checkout_file(root, relative).read_text(encoding="utf-8"))
        found = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        if not set(names) <= found:
            raise DriverError("UPSTREAM_API_MISMATCH", f"missing official symbol in {relative}")
        result[relative] = names
    return result


def _load_official_generators(root: Path, policy: CompatibilityPolicy) -> dict[str, ModuleType]:
    loaded: dict[str, ModuleType] = {}
    sentinel = object()
    original = {name: sys.modules.get(name, sentinel) for name in ("openai", "jsonlines", "tqdm")}
    openai_stub = ModuleType("openai")
    openai_stub.ChatCompletion = SimpleNamespace(
        create=lambda **_kwargs: (_ for _ in ()).throw(
            DriverError("UPSTREAM_API_MISMATCH", "teacher called before backend binding")
        )
    )
    jsonlines_stub = ModuleType("jsonlines")
    jsonlines_stub.open = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        DriverError("UPSTREAM_API_MISMATCH", "JSONL main path is not enabled")
    )
    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda value, *_args, **_kwargs: value
    sys.modules.update({"openai": openai_stub, "jsonlines": jsonlines_stub, "tqdm": tqdm_stub})
    try:
        for index, profile in enumerate(sorted(_PROFILES)):
            source = policy.prompt_profiles[profile]["source_path"]
            module = _load_module(f"_agentinputlab_infocqr_{profile}_{index}", root / source)
            function = getattr(module, "generate_rewrite", None)
            if not callable(function):
                raise DriverError("UPSTREAM_API_MISMATCH", "generate_rewrite is unavailable")
            loaded[profile] = module
    finally:
        for name, value in original.items():
            if value is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return loaded


def validate_request(
    value: Mapping[str, Any],
    policy: CompatibilityPolicy,
    *,
    backend_config: BackendConfig | None,
    mounted_cache: MountedCache | None,
) -> ValidatedRequest:
    _exact_keys(value, set(_ROOT_FIELDS), "worker request")
    if value.get("method_id") != METHOD_ID:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "request targets another method")
    release_id = _text(value.get("release_id"), "release_id")
    if release_id not in _RELEASES:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "unsupported InfoCQR release")
    if value.get("upstream_commit") != policy.commit:
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "request revision mismatch")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise DriverError("PROTOCOL_MISMATCH", "request protocol mismatch")
    request_id = _hash(value.get("request_id"), "request_id")
    identity = {key: value[key] for key in _ROOT_FIELDS if key != "request_id"}
    if request_id != _canonical_hash(identity):
        raise DriverError("REQUEST_ID_MISMATCH", "request_id is invalid")

    payload = _mapping(value.get("payload"), "request payload")
    _exact_keys(payload, set(_PAYLOAD_FIELDS), "request payload")
    if payload.get("source_license_acknowledged") is not True:
        raise DriverError("SOURCE_LICENSE_NOT_ACKNOWLEDGED", "per-request license gate failed")
    operation = _text(payload.get("operation"), "operation")
    prompt_profile = _text(payload.get("prompt_profile"), "prompt profile")
    expected_operation, expected_profile = _RELEASES[release_id]
    if operation != expected_operation or (
        expected_profile != "*" and prompt_profile != expected_profile
    ):
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "release operation/profile mismatch")
    if operation == "live_teacher" and prompt_profile not in _PROFILES:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "live prompt profile is unsupported")
    if operation == "t5_student" and prompt_profile != "t5_student":
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "T5 student profile mismatch")

    prompt_hash = _text(payload.get("prompt_source_sha256"), "prompt source hash")
    if operation != "t5_student":
        prompt_hash = _hash(prompt_hash, "prompt source hash")
        if prompt_hash != policy.prompt_profiles[prompt_profile]["source_sha256"]:
            raise DriverError("PROMPT_IDENTITY_MISMATCH", "prompt source hash mismatch")
    elif prompt_hash:
        if prompt_hash != policy.source_hashes[T5_TEST_PATH]:
            raise DriverError("PROMPT_IDENTITY_MISMATCH", "T5 source hash mismatch")

    dataset_id = _text(payload.get("dataset_id"), "dataset id")
    common_support = release_id == "rewrite.infocqr.common_support_rw_zsl"
    if not common_support and dataset_id.lower() != "qrecc":
        raise DriverError("DATASET_IDENTITY_MISMATCH", "InfoCQR worker requires QReCC")
    if common_support and payload.get("profile_kind") != "controlled_adapter":
        raise DriverError(
            "PROFILE_IDENTITY_MISMATCH",
            "InfoCQR common-support release requires controlled_adapter",
        )
    dataset_revision = _text(payload.get("dataset_revision"), "dataset revision")
    split = _text(payload.get("split"), "split")
    conversation_source = _text(payload.get("conversation_source"), "conversation source")
    conversation_no = _positive_int(payload.get("conversation_no"), "conversation_no")
    turn_no = _positive_int(payload.get("turn_no"), "turn_no")
    query = _bounded_text(payload.get("query"), "query")
    new_context = _string_sequence(payload.get("new_context"), "new_context")
    if len(new_context) % 2 != 0:
        raise DriverError(
            "CONVERSATION_SCHEMA_INVALID", "QReCC NewContext must alternate complete Q/A pairs"
        )
    if turn_no == 1 and new_context:
        raise DriverError("CONVERSATION_SCHEMA_INVALID", "first turn cannot have NewContext")

    dependencies = _mapping(value.get("dependency_fingerprints"), "dependency fingerprints")
    _exact_keys(dependencies, {"backend", "retriever"}, "dependency fingerprints")
    backend_fp = dict(_mapping(dependencies.get("backend"), "backend fingerprint"))
    retriever_fp = dict(_mapping(dependencies.get("retriever"), "retriever fingerprint"))
    if not retriever_fp:
        raise DriverError(
            "RETRIEVER_IDENTITY_MISSING", "retrieval corpus/retriever fingerprint is required"
        )
    retrieval_corpus_id = _text(payload.get("retrieval_corpus_id"), "retrieval corpus id")
    if operation == "live_teacher":
        if backend_config is None or _canonical_json(backend_fp) != _canonical_json(
            backend_config.fingerprint
        ):
            raise DriverError("BACKEND_IDENTITY_MISMATCH", "request model differs from startup")
    elif backend_fp:
        raise DriverError("BACKEND_IDENTITY_MISMATCH", "non-live request cannot bind a backend")

    cache_id = _optional_text(payload.get("cache_artifact_id"), "cache artifact id")
    cache_hash = _optional_hash(payload.get("cache_artifact_sha256"), "cache artifact hash")
    if operation == "cached_replay":
        if mounted_cache is None:
            raise DriverError("CACHE_ARTIFACT_UNAVAILABLE", "cache is not mounted")
        if (
            cache_id != mounted_cache.artifact_id
            or cache_hash != mounted_cache.sha256
            or prompt_profile != mounted_cache.profile
            or split != mounted_cache.split
        ):
            raise DriverError("CACHE_IDENTITY_MISMATCH", "mounted cache identity mismatch")
    elif cache_id or cache_hash:
        raise DriverError("CACHE_IDENTITY_MISMATCH", "live request cannot inject a cache")

    allow_truth = payload.get("allow_first_turn_truth_rewrite")
    if not isinstance(allow_truth, bool):
        raise DriverError("INVALID_SCHEMA", "allow_first_turn_truth_rewrite must be boolean")
    profile_kind = _text(payload.get("profile_kind"), "profile kind")
    if profile_kind not in {"official_native", "controlled_adapter", "artifact_replay"}:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "unsupported profile kind")
    if operation == "cached_replay" and profile_kind != "artifact_replay":
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "cache must be artifact_replay")
    if operation == "live_teacher" and profile_kind == "artifact_replay":
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "live teacher cannot be artifact_replay")
    if operation == "live_teacher" and profile_kind == "official_native":
        if (
            backend_config is None
            or not backend_config.model.startswith(OFFICIAL_MODEL_NAME)
            or backend_config.temperature != 0
            or backend_config.max_output_tokens < OFFICIAL_MAX_TOKENS
        ):
            raise DriverError(
                "PROFILE_IDENTITY_MISMATCH",
                "official_native requires the official teacher family and generation controls",
            )
        if prompt_profile == "t5_edit":
            raise DriverError(
                "PROFILE_IDENTITY_MISMATCH",
                "user-supplied T5 candidates are controlled_adapter inputs",
            )

    initial_rewrite = _optional_bounded_text(payload.get("initial_rewrite"), "initial rewrite")
    initial_hash = _optional_hash(
        payload.get("initial_rewrite_source_sha256"), "initial rewrite source hash"
    )
    if operation == "live_teacher" and prompt_profile == "t5_edit":
        if not initial_rewrite or not initial_hash:
            raise DriverError(
                "INITIAL_REWRITE_IDENTITY_MISSING",
                "official T5 editor requires a candidate and artifact hash",
            )
    elif initial_rewrite or initial_hash:
        raise DriverError(
            "INITIAL_REWRITE_IDENTITY_MISMATCH", "profile cannot inject an initial rewrite"
        )

    controls = _validate_controls(payload.get("controls"))
    reservation = _mapping(value.get("minimum_reservation"), "minimum_reservation")
    expected_calls = (
        2
        if operation == "live_teacher" and prompt_profile == "self_edit"
        else (1 if operation == "live_teacher" else 0)
    )
    _exact_keys(reservation, {"model_calls"}, "minimum_reservation")
    if _nonnegative_int(reservation.get("model_calls"), "reserved model calls") != expected_calls:
        raise DriverError("RESERVATION_MISMATCH", "model-call reservation mismatch")
    if controls.max_model_calls < expected_calls:
        raise DriverError("MODEL_CALL_BUDGET_EXCEEDED", "reservation exceeds model-call budget")

    return ValidatedRequest(
        request_id=request_id,
        release_id=release_id,
        operation=operation,
        task_id=_text(payload.get("task_id"), "task id"),
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        split=split,
        conversation_source=conversation_source,
        conversation_no=conversation_no,
        turn_no=turn_no,
        query=query,
        new_context=new_context,
        prompt_profile=prompt_profile,
        prompt_source_sha256=prompt_hash,
        cache_artifact_id=cache_id,
        cache_artifact_sha256=cache_hash,
        allow_first_turn_truth_rewrite=allow_truth,
        retrieval_corpus_id=retrieval_corpus_id,
        profile_kind=profile_kind,
        initial_rewrite=initial_rewrite,
        initial_rewrite_source_sha256=initial_hash,
        controls=controls,
        seed=_nonnegative_int(payload.get("seed"), "seed"),
        backend_fingerprint=backend_fp,
        retriever_fingerprint=retriever_fp,
    )


def _validate_author_cache_record(
    value: Mapping[str, Any], profile: str, line_no: int
) -> Mapping[str, Any]:
    try:
        source = _text(value["Conversation_source"], "cache Conversation_source")
        conversation_no = _positive_int(value["Conversation_no"], "cache Conversation_no")
        turn_no = _positive_int(value["Turn_no"], "cache Turn_no")
        query = _bounded_text(value["Question"], "cache Question")
        new_context = _string_sequence(value["NewContext"], "cache NewContext")
    except KeyError as exc:
        raise DriverError(
            "CACHE_SCHEMA_INVALID", f"cache line {line_no} lacks a QReCC identity field"
        ) from exc
    if len(new_context) % 2 != 0:
        raise DriverError("CACHE_SCHEMA_INVALID", "cache NewContext must contain Q/A pairs")
    candidate_key = "GPT_rewrite" if profile in {"rw_zsl", "rw_fsl"} else "Editor_rewrite"
    if candidate_key not in value:
        raise DriverError("CACHE_SCHEMA_INVALID", f"cache line lacks {candidate_key}")
    initial_key = "T5_rewrite" if profile == "t5_edit" else "GPT_rewrite"
    initial: Any = ""
    if profile in {"t5_edit", "self_edit"}:
        if initial_key not in value:
            raise DriverError("CACHE_SCHEMA_INVALID", f"cache line lacks {initial_key}")
        initial = value[initial_key]
        _legacy_candidate(initial)
    _legacy_candidate(value[candidate_key])
    return {
        "conversation_source": source,
        "conversation_no": conversation_no,
        "turn_no": turn_no,
        "query": query,
        "new_context": new_context,
        "candidate": value[candidate_key],
        "initial_rewrite": initial,
    }


def _legacy_candidate(value: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(value, str):
        if not value.strip():
            raise DriverError("CACHE_SCHEMA_INVALID", "legacy candidate is empty")
        return value, {}
    root = _mapping(value, "legacy ChatCompletion response")
    choices = root.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or len(choices) < 1:
        raise DriverError("CACHE_SCHEMA_INVALID", "legacy response choices are invalid")
    message = _mapping(_mapping(choices[0], "legacy choice").get("message"), "legacy message")
    text = _bounded_text(message.get("content"), "legacy candidate")
    raw_usage = root.get("usage", {})
    usage = dict(_mapping(raw_usage, "legacy usage")) if raw_usage is not None else {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in usage:
            _nonnegative_int(usage[key], f"legacy {key}")
    return text, usage


def _official_postprocess_compat(
    candidate: str,
    *,
    question: str,
    turn_no: int,
    editor: bool,
    initial_rewrite: str,
) -> tuple[str, str]:
    """Exact branch order from pinned ``post_processing.py`` lines 33-51."""

    if turn_no <= 1:
        return candidate, "first_turn_unchanged"
    fallback = initial_rewrite if editor else question
    if editor and not fallback.strip():
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "editor fallback rewrite is empty")
    if "N/A" in candidate:
        return fallback, "n_a_fallback"
    if not candidate or candidate[-1] != "?":
        return fallback, "non_question_fallback"
    lowered = candidate.lower()
    if "no edit" in lowered or "no need to edit" in lowered:
        return fallback, "no_edit_fallback"
    return candidate, "unchanged"


def read_checkout_head(root: Path) -> str:
    marker = root / ".git"
    git_dir = marker
    if marker.is_file():
        raw = marker.read_text(encoding="utf-8").strip()
        if not raw.startswith("gitdir: "):
            raise DriverError("UPSTREAM_REVISION_MISMATCH", "invalid .git pointer")
        target = Path(raw[8:])
        git_dir = (root / target).resolve() if not target.is_absolute() else target.resolve()
    if not git_dir.is_dir():
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "checkout Git metadata missing")
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if _COMMIT_RE.fullmatch(head):
        return head.lower()
    if not head.startswith("ref: "):
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "checkout HEAD is invalid")
    ref = _safe_git_ref(head[5:])
    ref_file = git_dir / ref
    if ref_file.is_file():
        commit = ref_file.read_text(encoding="utf-8").strip()
        if _COMMIT_RE.fullmatch(commit):
            return commit.lower()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("^"):
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1] == ref and _COMMIT_RE.fullmatch(parts[0]):
                return parts[0].lower()
    raise DriverError("UPSTREAM_REVISION_MISMATCH", "checkout ref cannot be resolved")


def _require_clean_checkout(root: Path) -> None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DriverError("CHECKOUT_DIRTY", "cannot verify clean checkout") from exc
    if completed.returncode != 0 or completed.stdout.strip():
        raise DriverError("CHECKOUT_DIRTY", "InfoCQR checkout must be clean")


def _safe_git_ref(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value.startswith("refs/"):
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "unsafe Git ref")
    return value


def serve(runtime: InfoCQRRuntime, input_stream: TextIO, output_stream: TextIO) -> None:
    for raw_line in input_stream:
        request_id = ""
        try:
            if len(raw_line.encode("utf-8")) > MAX_REQUEST_BYTES:
                raise DriverError("REQUEST_TOO_LARGE", "request exceeds byte limit")
            value = _mapping(_loads_strict_json(raw_line), "JSONL request")
            if isinstance(value.get("request_id"), str):
                request_id = value["request_id"][:128]
            response = runtime.run(value)
        except DriverError as exc:
            response = _failure_response(request_id, exc)
        except Exception as exc:  # pragma: no cover - defensive redaction boundary
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
        "failure_code": error.code,
        "message": str(error),
        "consumed": error.consumed,
    }
    if error.usage is not None:
        response["usage"] = dict(error.usage)
    return response


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DriverError("UPSTREAM_IMPORT_FAILED", "cannot construct source module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise DriverError("UPSTREAM_IMPORT_FAILED", "cannot import official source") from exc
    return module


def _checkout_file(root: Path, relative: str) -> Path:
    path = _safe_child(root, relative)
    if not path.is_file() or path.is_symlink():
        raise DriverError("SOURCE_UNAVAILABLE", f"required source unavailable: {relative}")
    return path


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / _safe_relative_path(relative, "relative path")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DriverError("INVALID_PATH", "path escapes checkout") from exc
    return candidate


def _safe_relative_path(value: Any, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DriverError("INVALID_PATH", f"{label} is unsafe")
    return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DriverError("SOURCE_UNAVAILABLE", "cannot read required file") from exc
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _loads_strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise DriverError("INVALID_JSON", f"non-finite JSON value {value} is forbidden")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DriverError("INVALID_JSON", f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise DriverError("INVALID_JSON", "invalid JSON input") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DriverError("INVALID_SCHEMA", f"{label} must be a string-keyed object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DriverError("INVALID_SCHEMA", f"{label} fields do not match the protocol")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriverError("INVALID_SCHEMA", f"{label} must be non-empty text")
    return value


def _optional_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise DriverError("INVALID_SCHEMA", f"{label} must be text")
    return value


def _bounded_text(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) > MAX_TEXT_CHARS:
        raise DriverError("INVALID_SCHEMA", f"{label} exceeds text limit")
    return text


def _optional_bounded_text(value: Any, label: str) -> str:
    text = _optional_text(value, label)
    if len(text) > MAX_TEXT_CHARS:
        raise DriverError("INVALID_SCHEMA", f"{label} exceeds text limit")
    return text


def _hash(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    if _HASH_RE.fullmatch(text) is None:
        raise DriverError("INVALID_SCHEMA", f"{label} must be a SHA-256 digest")
    return text


def _optional_hash(value: Any, label: str) -> str:
    text = _optional_text(value, label)
    return "" if not text else _hash(text, label)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriverError("INVALID_SCHEMA", f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be a non-negative integer")
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
    result = _finite_float(value, label)
    if result <= 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be positive")
    return result


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("INVALID_SCHEMA", f"{label} must be a string sequence")
    if len(value) > 100_000:
        raise DriverError("INVALID_SCHEMA", f"{label} exceeds sequence limit")
    return tuple(_optional_bounded_text(item, f"{label} item") for item in value)


def _messages(value: Any) -> tuple[Mapping[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 1:
        raise DriverError("UPSTREAM_API_MISMATCH", "official messages must contain one item")
    item = _mapping(value[0], "official message")
    _exact_keys(item, {"role", "content"}, "official message")
    if item.get("role") != "user":
        raise DriverError("UPSTREAM_API_MISMATCH", "official message role changed")
    return ({"role": "user", "content": _bounded_text(item.get("content"), "prompt")},)


def _validate_controls(value: Any) -> Controls:
    item = _mapping(value, "controls")
    _exact_keys(item, set(_CONTROL_FIELDS), "controls")
    return Controls(
        max_model_calls=_nonnegative_int(item.get("max_model_calls"), "max_model_calls"),
        max_tokens=_nonnegative_int(item.get("max_tokens"), "max_tokens"),
        max_cost_usd=_nonnegative_float(item.get("max_cost_usd"), "max_cost_usd"),
    )


def _validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise DriverError("INVALID_BACKEND_CONFIG", "backend base_url is invalid")
    if parsed.username or parsed.password:
        raise DriverError("INVALID_BACKEND_CONFIG", "backend URL cannot contain credentials")
    return value.rstrip("/")


def _mounted_cache_from_args(
    args: argparse.Namespace, checkout: Path, policy: CompatibilityPolicy
) -> MountedCache | None:
    if args.known_cache:
        if any((args.cache_path, args.cache_sha256, args.cache_profile, args.cache_split)):
            raise DriverError(
                "INVALID_CACHE_CONFIG", "known cache cannot be combined with custom cache fields"
            )
        try:
            item = policy.known_caches[args.known_cache]
        except KeyError as exc:
            raise DriverError("INVALID_CACHE_CONFIG", "unknown manifest cache id") from exc
        return MountedCache.load(
            _safe_child(checkout, item["path"]),
            artifact_id=args.known_cache,
            expected_sha256=item["sha256"],
            profile=item["profile"],
            split=item["split"],
        )
    custom = (args.cache_path, args.cache_sha256, args.cache_profile, args.cache_split)
    if not any(custom):
        return None
    if not all(custom) or not args.cache_artifact_id:
        raise DriverError(
            "INVALID_CACHE_CONFIG", "custom cache requires path, id, SHA, profile, and split"
        )
    return MountedCache.load(
        Path(args.cache_path),
        artifact_id=args.cache_artifact_id,
        expected_sha256=args.cache_sha256,
        profile=args.cache_profile,
        split=args.cache_split,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--backend-config-json")
    parser.add_argument("--known-cache")
    parser.add_argument("--cache-path")
    parser.add_argument("--cache-artifact-id")
    parser.add_argument("--cache-sha256")
    parser.add_argument("--cache-profile", choices=sorted(_PROFILES))
    parser.add_argument("--cache-split")
    parser.add_argument("--acknowledge-noassertion-license", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = load_compatibility_policy(Path(args.manifest))
        checkout, identity_mode = verify_checkout(Path(args.checkout), policy)
        audit_source_symbols(checkout, policy)
        cache = _mounted_cache_from_args(args, checkout, policy)
        if args.verify_only:
            print(
                _canonical_json(
                    {
                        "status": "ok",
                        "upstream_commit": UPSTREAM_COMMIT,
                        "license_spdx": "NOASSERTION",
                        "checkout_identity_mode": identity_mode,
                        "cache_artifact_sha256": cache.sha256 if cache else "",
                        "t5_student_checkpoint_available": False,
                    }
                )
            )
            return 0
        backend_config = (
            BackendConfig.from_json(args.backend_config_json) if args.backend_config_json else None
        )
        backend = OpenAICompatibleBackend(backend_config) if backend_config else None
        runtime = InfoCQRRuntime(
            checkout,
            backend_config,
            backend,
            cache=cache,
            source_license_acknowledged=args.acknowledge_noassertion_license,
            policy=policy,
        )
        serve(runtime, sys.stdin, sys.stdout)
        return 0
    except DriverError as exc:
        print(_canonical_json(_failure_response("", exc)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
