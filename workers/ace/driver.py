#!/usr/bin/env python3
"""Checksum-pinned JSONL driver for upstream ACE generic text tasks.

The worker deliberately keeps deployment configuration outside request payloads:
the operator supplies one verified checkout and one explicit LLM configuration at
process startup. Requests can contain examples and bounded scientific controls,
but never a module, callable, command, endpoint, or filesystem path.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

UPSTREAM_COMMIT = "bcb7cea0504afad6f55fec4845dd4864c9f9eee7"
PROTOCOL_VERSION = "agentinputlab.ace.offline.v1"
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_REF_RE = re.compile(r"refs/[A-Za-z0-9._/-]+")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLAYBOOK_LINE_RE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")
_ROLE_NAMES = ("target", "reflector", "curator")
_PROVIDERS = frozenset(("openai", "sambanova", "together", "commonstack"))
_STANDARD_SECTIONS = (
    "STRATEGIES & INSIGHTS",
    "FORMULAS & CALCULATIONS",
    "CODE SNIPPETS & TEMPLATES",
    "COMMON MISTAKES TO AVOID",
    "PROBLEM-SOLVING HEURISTICS",
    "CONTEXT CLUES & INDICATORS",
    "OTHERS",
)
_MAX_JSONL_BYTES = 8 * 1024 * 1024
_MAX_UPSTREAM_STDOUT_BYTES = 1024 * 1024
_UPSTREAM_MODULES = ("ace", "llm", "logger", "playbook_utils", "utils")
_OPENAI_CONSTRUCTOR_ENV_LOCK = threading.RLock()


class WorkerFailure(RuntimeError):
    """Expected fail-closed worker error with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BoundedHashSink(io.TextIOBase):
    """Discard upstream text while retaining only bounded audit metadata."""

    def __init__(self, max_bytes: int = _MAX_UPSTREAM_STDOUT_BYTES) -> None:
        super().__init__()
        self.max_bytes = max_bytes
        self.byte_count = 0
        self._digest = hashlib.sha256()

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("upstream stdout must be text")
        encoded = value.encode("utf-8", errors="replace")
        if self.byte_count + len(encoded) > self.max_bytes:
            raise WorkerFailure(
                "UPSTREAM_OUTPUT_LIMIT_EXCEEDED",
                "ACE upstream stdout exceeded the bounded diagnostic limit",
            )
        self._digest.update(encoded)
        self.byte_count += len(encoded)
        return len(value)

    def flush(self) -> None:
        return None

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CompatibilityPolicy:
    commit: str
    license_path: str
    license_spdx: str
    license_sha256: str
    source_hashes: Mapping[str, str]
    require_clean_checkout: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompatibilityPolicy:
        upstream = _require_mapping(value.get("upstream"), "compatibility upstream")
        sources = _require_mapping(value.get("source_files"), "compatibility source_files")
        commit = _require_text(upstream.get("commit"), "compatibility commit")
        license_path = _require_text(upstream.get("license_path"), "compatibility license_path")
        license_spdx = _require_text(upstream.get("license_spdx"), "compatibility license_spdx")
        license_sha256 = _require_hash(
            upstream.get("license_sha256"), "compatibility license_sha256"
        )
        require_clean_checkout = upstream.get("require_clean_checkout")
        if not isinstance(require_clean_checkout, bool):
            raise WorkerFailure(
                "INVALID_COMPATIBILITY_MANIFEST",
                "ACE compatibility policy must declare require_clean_checkout",
            )
        if commit != UPSTREAM_COMMIT or license_spdx != "Apache-2.0":
            raise WorkerFailure(
                "INVALID_COMPATIBILITY_MANIFEST",
                "ACE compatibility policy must pin the approved commit and Apache-2.0",
            )
        source_hashes: dict[str, str] = {}
        for raw_path, raw_hash in sources.items():
            path = _safe_relative_path(raw_path, "compatibility source path")
            source_hashes[path] = _require_hash(raw_hash, f"compatibility hash for {path}")
        if not source_hashes:
            raise WorkerFailure(
                "INVALID_COMPATIBILITY_MANIFEST",
                "ACE compatibility policy requires source hashes",
            )
        return cls(
            commit=commit,
            license_path=_safe_relative_path(license_path, "compatibility license path"),
            license_spdx=license_spdx,
            license_sha256=license_sha256,
            source_hashes=source_hashes,
            require_clean_checkout=require_clean_checkout,
        )


@dataclass(frozen=True, slots=True)
class RoleLLMConfig:
    model: str
    fingerprint: Mapping[str, Any]
    input_cost_per_million: float
    output_cost_per_million: float


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_provider: str
    max_tokens_per_call: int
    json_mode: bool
    roles: Mapping[str, RoleLLMConfig]
    base_url: str | None = None
    api_key_env: str | None = None
    extra_body: Mapping[str, Any] | None = None

    @classmethod
    def from_json(cls, raw: str) -> LLMConfig:
        value = _loads_strict_json(raw)
        root = _require_mapping(value, "LLM configuration")
        required = {"api_provider", "max_tokens_per_call", "json_mode", "roles"}
        allowed = required | {"base_url", "api_key_env", "extra_body"}
        if not required.issubset(root) or not set(root).issubset(allowed):
            raise WorkerFailure(
                "INVALID_SCHEMA", "LLM configuration fields do not match the protocol"
            )
        provider = _require_text(root["api_provider"], "LLM api_provider")
        if provider not in _PROVIDERS:
            raise WorkerFailure(
                "INVALID_LLM_CONFIG",
                f"LLM api_provider must be one of {sorted(_PROVIDERS)}",
            )
        max_tokens = _require_positive_int(root["max_tokens_per_call"], "LLM max_tokens_per_call")
        json_mode = root["json_mode"]
        if not isinstance(json_mode, bool):
            raise WorkerFailure("INVALID_LLM_CONFIG", "LLM json_mode must be boolean")
        raw_roles = _require_mapping(root["roles"], "LLM roles")
        if set(raw_roles) != set(_ROLE_NAMES):
            raise WorkerFailure(
                "INVALID_LLM_CONFIG",
                "LLM roles must bind target, reflector, and curator exactly",
            )
        roles: dict[str, RoleLLMConfig] = {}
        for role in _ROLE_NAMES:
            role_value = _require_mapping(raw_roles[role], f"LLM role {role}")
            _require_exact_keys(
                role_value,
                {
                    "model",
                    "fingerprint",
                    "input_cost_per_million",
                    "output_cost_per_million",
                },
                f"LLM role {role}",
            )
            fingerprint = _require_mapping(role_value["fingerprint"], f"LLM {role} fingerprint")
            if not fingerprint:
                raise WorkerFailure(
                    "INVALID_LLM_CONFIG", f"LLM {role} fingerprint must be non-empty"
                )
            _canonical_json(fingerprint)
            roles[role] = RoleLLMConfig(
                model=_require_text(role_value["model"], f"LLM {role} model"),
                fingerprint=dict(fingerprint),
                input_cost_per_million=_require_non_negative_float(
                    role_value["input_cost_per_million"],
                    f"LLM {role} input_cost_per_million",
                ),
                output_cost_per_million=_require_non_negative_float(
                    role_value["output_cost_per_million"],
                    f"LLM {role} output_cost_per_million",
                ),
            )
        base_url_value = root.get("base_url")
        api_key_env_value = root.get("api_key_env")
        if (base_url_value is None) != (api_key_env_value is None):
            raise WorkerFailure(
                "INVALID_LLM_CONFIG",
                "LLM base_url and api_key_env must be supplied together",
            )
        base_url: str | None = None
        api_key_env: str | None = None
        if base_url_value is not None:
            if provider != "openai":
                raise WorkerFailure(
                    "INVALID_LLM_CONFIG",
                    "custom OpenAI-compatible endpoints require api_provider=openai",
                )
            base_url = _require_http_url(base_url_value, "LLM base_url")
            api_key_env = _require_env_name(api_key_env_value, "LLM api_key_env")
        extra_body_value = root.get("extra_body")
        extra_body: Mapping[str, Any] | None = None
        if extra_body_value is not None:
            if base_url is None:
                raise WorkerFailure(
                    "INVALID_LLM_CONFIG",
                    "LLM extra_body is permitted only for an operator-bound endpoint",
                )
            extra_body = dict(_require_mapping(extra_body_value, "LLM extra_body"))
            _canonical_json(extra_body)
        return cls(
            api_provider=provider,
            max_tokens_per_call=max_tokens,
            json_mode=json_mode,
            roles=roles,
            base_url=base_url,
            api_key_env=api_key_env,
            extra_body=extra_body,
        )


@dataclass(frozen=True, slots=True)
class Controls:
    max_rounds: int
    max_model_calls: int
    max_tokens: int
    max_cost_usd: float
    seed: int


@dataclass(frozen=True, slots=True)
class ValidatedRequest:
    request_id: str
    train_examples: tuple[Mapping[str, Any], ...]
    validation_examples: tuple[Mapping[str, Any], ...]
    initial_bullets: tuple[dict[str, Any], ...]
    controls: Controls
    split_audit_hash: str


class GenericTextProcessor:
    """Minimal task-agnostic exact-match processor required by upstream ACE."""

    @staticmethod
    def answer_is_correct(predicted: str, ground_truth: str) -> bool:
        return predicted.strip().casefold() == ground_truth.strip().casefold()

    def evaluate_accuracy(self, predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground truths must have equal length")
        if not predictions:
            return 0.0
        correct = sum(
            self.answer_is_correct(prediction, truth)
            for prediction, truth in zip(predictions, ground_truths, strict=True)
        )
        return correct / len(predictions)


class ProviderAttemptCompletions:
    """Transparent completion proxy that enforces the shared attempt budget."""

    def __init__(self, delegate: Any, tracker: UsageTracker, role: str) -> None:
        self._delegate = delegate
        self._tracker = tracker
        self._role = role

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._tracker.before_provider_attempt(self._role)
        try:
            result = self._delegate.create(*args, **kwargs)
        except Exception:
            self._tracker.failed_provider_attempt()
            raise
        self._tracker.after_provider_response(self._role, result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class ProviderAttemptChat:
    """Transparent chat proxy retaining all non-completion client behavior."""

    def __init__(self, delegate: Any, tracker: UsageTracker, role: str) -> None:
        self._delegate = delegate
        self.completions = ProviderAttemptCompletions(delegate.completions, tracker, role)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class ProviderAttemptClient:
    """Transparent upstream client proxy with a counted completion seam."""

    def __init__(self, delegate: Any, tracker: UsageTracker, role: str) -> None:
        self._delegate = delegate
        self.chat = ProviderAttemptChat(delegate.chat, tracker, role)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class CompletionDefaults:
    """Inject immutable deployment-only completion fields before observation."""

    def __init__(self, delegate: Any, extra_body: Mapping[str, Any]) -> None:
        self._delegate = delegate
        self._extra_body = dict(extra_body)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if "extra_body" in kwargs:
            raise WorkerFailure(
                "UPSTREAM_INTERFACE_MISMATCH",
                "upstream ACE unexpectedly supplied request-level extra_body",
            )
        if (
            "seed" in self._extra_body
            and "seed" in kwargs
            and (type(kwargs["seed"]) is not int or kwargs["seed"] != self._extra_body["seed"])
        ):
            raise WorkerFailure(
                "UPSTREAM_INTERFACE_MISMATCH", "upstream ACE seed differs from the fit controls"
            )
        return self._delegate.create(*args, extra_body=dict(self._extra_body), **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class CompletionDefaultChat:
    def __init__(self, delegate: Any, extra_body: Mapping[str, Any]) -> None:
        self._delegate = delegate
        self.completions = CompletionDefaults(delegate.completions, extra_body)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class CompletionDefaultClient:
    def __init__(self, delegate: Any, extra_body: Mapping[str, Any]) -> None:
        self._delegate = delegate
        self.chat = CompletionDefaultChat(delegate.chat, extra_body)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class UsageTracker:
    """Observe upstream logical calls, provider attempts, and returned usage."""

    def __init__(self, controls: Controls, llm_config: LLMConfig) -> None:
        self.controls = controls
        self.llm_config = llm_config
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.call_sequence: list[str] = []
        self.provider_attempt_sequence: list[str] = []
        self.provider_input_tokens = 0
        self.provider_output_tokens = 0
        self.provider_cost_usd = 0.0
        self.provider_usage_responses = 0
        self.failed_provider_attempts = 0
        self.first_component_failure: WorkerFailure | None = None
        self._lock = threading.Lock()

    def install(self, ace_system: Any) -> None:
        self._install_component(ace_system.generator, "generate", "target")
        self._install_component(ace_system.reflector, "reflect", "reflector")
        self._install_component(ace_system.curator, "curate", "curator")

    def _install_component(self, component: Any, method_name: str, role: str) -> None:
        client = getattr(component, "api_client", None)
        chat = getattr(client, "chat", None)
        completions = getattr(chat, "completions", None)
        if not callable(getattr(completions, "create", None)):
            raise WorkerFailure(
                "UPSTREAM_INTERFACE_MISMATCH",
                f"upstream ACE {role} lacks api_client.chat.completions.create()",
            )
        component.api_client = ProviderAttemptClient(client, self, role)
        self._wrap(component, method_name, role)

    def _wrap(self, component: Any, method_name: str, role: str) -> None:
        original = getattr(component, method_name, None)
        if not callable(original):
            raise WorkerFailure(
                "UPSTREAM_INTERFACE_MISMATCH",
                f"upstream ACE {role} does not expose {method_name}()",
            )

        @functools.wraps(original)
        def observed(*args: Any, **kwargs: Any) -> Any:
            self._before_component_call(role)
            try:
                result = original(*args, **kwargs)
                self._after_call(role, result)
            except WorkerFailure as exc:
                # Native validation may catch this and then fail in its own
                # empty-results branch. Keep the original guard failure/costs.
                with self._lock:
                    if self.first_component_failure is None:
                        self.first_component_failure = exc
                raise
            return result

        setattr(component, method_name, observed)

    def _before_component_call(self, role: str) -> None:
        with self._lock:
            self.call_sequence.append(role)

    def before_provider_attempt(self, role: str) -> None:
        with self._lock:
            if self.model_calls >= self.controls.max_model_calls:
                raise WorkerFailure(
                    "MODEL_CALL_BUDGET_EXCEEDED",
                    "ACE reached max_model_calls before the next provider attempt",
                )
            self.model_calls += 1
            self.provider_attempt_sequence.append(role)

    def _after_call(self, role: str, result: Any) -> None:
        if not isinstance(result, tuple) or not result:
            raise WorkerFailure(
                "UPSTREAM_USAGE_UNAVAILABLE",
                f"upstream ACE {role} returned an unexpected result",
            )
        info = result[-1]
        if not isinstance(info, Mapping):
            raise WorkerFailure(
                "UPSTREAM_USAGE_UNAVAILABLE",
                f"upstream ACE {role} omitted call metadata",
            )
        # Pinned ACE returns an INCORRECT placeholder and error metadata (without
        # token fields) for an empty API response. Preserve that failure instead
        # of misreporting it as an invalid integer in the usage schema. Physical
        # usage has already been observed before upstream parsing; do not invent
        # zero token counts, retry, or let the placeholder reach a scorer.
        if any(
            info.get(flag) is True
            for flag in (
                "training_marked_incorrect_due_to_empty_response",
                "test_marked_incorrect_due_to_empty_response",
            )
        ):
            raise WorkerFailure(
                "UPSTREAM_EMPTY_RESPONSE",
                f"upstream ACE {role} received an empty model response; "
                "the attempt and available provider usage are retained, without retry",
            )
        input_tokens = _require_non_negative_int(
            info.get("prompt_num_tokens"), f"upstream {role} prompt_num_tokens"
        )
        output_tokens = _require_non_negative_int(
            info.get("response_num_tokens"), f"upstream {role} response_num_tokens"
        )
        expected_model = self.llm_config.roles[role].model
        if info.get("model") != expected_model:
            raise WorkerFailure(
                "UPSTREAM_MODEL_MISMATCH",
                f"upstream {role} reported a model other than the configured model",
            )
        pricing = self.llm_config.roles[role]
        cost = (
            input_tokens * pricing.input_cost_per_million
            + output_tokens * pricing.output_cost_per_million
        ) / 1_000_000
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += cost
            if self.input_tokens + self.output_tokens > self.controls.max_tokens:
                raise WorkerFailure(
                    "TOKEN_BUDGET_EXCEEDED", "ACE exceeded the request token budget"
                )
            if self.cost_usd > self.controls.max_cost_usd:
                raise WorkerFailure("COST_BUDGET_EXCEEDED", "ACE exceeded the request cost budget")

    def failed_provider_attempt(self) -> None:
        with self._lock:
            self.failed_provider_attempts += 1

    def after_provider_response(self, role: str, response: Any) -> None:
        # Observe SDK usage before upstream JSON parsing can fail or retry.
        # Older native test doubles may expose usage only in component metadata;
        # that existing path remains available, without double-counting tokens.
        usage = (
            response.get("usage")
            if isinstance(response, Mapping)
            else getattr(response, "usage", None)
        )
        if usage is None:
            return
        read = usage.get if isinstance(usage, Mapping) else lambda key: getattr(usage, key, None)
        incoming = _require_non_negative_int(read("prompt_tokens"), "provider prompt_tokens")
        outgoing = _require_non_negative_int(
            read("completion_tokens"), "provider completion_tokens"
        )
        pricing = self.llm_config.roles[role]
        with self._lock:
            self.provider_input_tokens += incoming
            self.provider_output_tokens += outgoing
            self.provider_usage_responses += 1
            self.provider_cost_usd += (
                incoming * pricing.input_cost_per_million
                + outgoing * pricing.output_cost_per_million
            ) / 1_000_000
            if self.provider_input_tokens + self.provider_output_tokens > self.controls.max_tokens:
                raise WorkerFailure(
                    "TOKEN_BUDGET_EXCEEDED", "ACE exceeded the request token budget"
                )
            if self.provider_cost_usd > self.controls.max_cost_usd:
                raise WorkerFailure("COST_BUDGET_EXCEEDED", "ACE exceeded the request cost budget")

    def require_upstream_chain(self) -> None:
        self._require_sequence(self.call_sequence, "agent")
        self._require_sequence(self.provider_attempt_sequence, "provider-attempt")

    @staticmethod
    def _require_sequence(sequence: Sequence[str], label: str) -> None:
        try:
            first_target = sequence.index("target")
            first_reflector = sequence.index("reflector")
            first_curator = sequence.index("curator")
        except ValueError as exc:
            raise WorkerFailure(
                "UPSTREAM_CHAIN_INCOMPLETE",
                f"ACE did not execute the complete {label} chain",
            ) from exc
        if not first_target < first_reflector < first_curator:
            raise WorkerFailure(
                "UPSTREAM_CHAIN_REORDERED",
                f"ACE did not preserve Generator -> Reflector -> Curator {label} order",
            )

    def as_dict(self) -> dict[str, int | float]:
        # Provider and component totals describe the same successful requests.
        # Taking their maximum retains pre-parse costs without summing twice.
        incoming = max(self.input_tokens, self.provider_input_tokens)
        outgoing = max(self.output_tokens, self.provider_output_tokens)
        return {
            "model_calls": self.model_calls,
            "input_tokens": incoming,
            "output_tokens": outgoing,
            "tokens": incoming + outgoing,
            "cost_usd": max(self.cost_usd, self.provider_cost_usd),
            "provider_usage_responses": self.provider_usage_responses,
            "failed_provider_attempts_with_unknown_tokens": self.failed_provider_attempts,
        }


class ACEWorkerRuntime:
    """One fixed checkout and LLM configuration serving many JSONL requests."""

    def __init__(
        self,
        checkout: Path,
        llm_config: LLMConfig,
        *,
        policy: CompatibilityPolicy | None = None,
        data_processor: Any | None = None,
    ) -> None:
        if data_processor is not None:
            for name in ("answer_is_correct", "evaluate_accuracy", "fingerprint"):
                if not callable(getattr(data_processor, name, None)):
                    raise WorkerFailure("INVALID_TASK_PROCESSOR", f"task processor lacks {name}()")
            fingerprint = data_processor.fingerprint()
            if not isinstance(fingerprint, Mapping):
                raise WorkerFailure(
                    "INVALID_TASK_PROCESSOR", "task processor fingerprint must be an object"
                )
            self.task_processor_fingerprint = json.loads(_canonical_json(fingerprint))
        else:
            self.task_processor_fingerprint = None
        self.data_processor = (
            data_processor if data_processor is not None else GenericTextProcessor()
        )
        self.policy = policy or load_compatibility_policy()
        self.checkout = verify_checkout(checkout, self.policy)
        self.llm_config = llm_config
        self.ace_class, self.playbook_utils = load_upstream(self.checkout)
        self.last_tracker: UsageTracker | None = None

    @contextlib.contextmanager
    def _bound_constructor_environment(self):
        """Let native initialization use the explicitly configured provider.

        Upstream initializes clients before the adapter can replace them. Its
        constructor reads the default OpenAI environment names even when the
        operator has chosen another key name. Bind those names for construction
        only; restore the worker's environment on success and on failure.
        """
        with _OPENAI_CONSTRUCTOR_ENV_LOCK:
            if self.llm_config.base_url is None:
                yield
                return
            key_name = self.llm_config.api_key_env
            api_key = os.environ.get(key_name, "") if key_name else ""
            if not api_key:
                raise WorkerFailure(
                    "MISSING_API_KEY",
                    f"ACE OpenAI-compatible runtime requires environment {key_name}",
                )
            values = {"OPENAI_API_KEY": api_key, "OPENAI_BASE_URL": self.llm_config.base_url}
            previous = {name: os.environ.get(name) for name in values}
            try:
                os.environ.update(values)
                yield
            finally:
                for name, old in previous.items():
                    if old is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = old

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.last_tracker = None
        request = validate_request(payload, self.llm_config)
        validator = getattr(self.data_processor, "validate_examples", None)
        if callable(validator):
            try:
                validator(request.train_examples, request.validation_examples)
            except (TypeError, ValueError) as exc:
                raise WorkerFailure(
                    "INVALID_EXAMPLES", "task processor rejected the fit groups"
                ) from exc
        if (
            self.task_processor_fingerprint is not None
            and json.loads(_canonical_json(self.data_processor.fingerprint()))
            != self.task_processor_fingerprint
        ):
            raise WorkerFailure("INVALID_TASK_PROCESSOR", "task processor fingerprint changed")
        initial_text = render_playbook(request.initial_bullets)
        roles = self.llm_config.roles
        diagnostics = BoundedHashSink()
        with contextlib.redirect_stdout(diagnostics):
            with self._bound_constructor_environment():
                ace_system = self.ace_class(
                    api_provider=self.llm_config.api_provider,
                    generator_model=roles["target"].model,
                    reflector_model=roles["reflector"].model,
                    curator_model=roles["curator"].model,
                    max_tokens=self.llm_config.max_tokens_per_call,
                    initial_playbook=initial_text,
                    use_bulletpoint_analyzer=False,
                )
            if self.llm_config.base_url is not None:
                self._bind_openai_compatible_clients(ace_system, seed=request.controls.seed)
        get_next_id = getattr(self.playbook_utils, "get_next_global_id", None)
        if not callable(get_next_id):
            raise WorkerFailure(
                "UPSTREAM_INTERFACE_MISMATCH",
                "pinned playbook_utils lacks get_next_global_id()",
            )
        ace_system.next_global_id = get_next_id(initial_text)
        tracker = UsageTracker(request.controls, self.llm_config)
        self.last_tracker = tracker
        tracker.install(ace_system)
        train_samples = [adapt_example(item) for item in request.train_examples]
        validation_samples = [adapt_example(item) for item in request.validation_examples]
        controls = request.controls
        if self.llm_config.max_tokens_per_call > controls.max_tokens:
            raise WorkerFailure(
                "INVALID_BUDGET",
                "configured max_tokens_per_call exceeds the request token budget",
            )
        random_state = random.getstate()
        try:
            random.seed(controls.seed)
            with tempfile.TemporaryDirectory(prefix="agentinputlab-ace-") as save_dir:
                upstream_config = {
                    "num_epochs": 1,
                    "max_num_rounds": controls.max_rounds,
                    "curator_frequency": 1,
                    "eval_steps": len(train_samples),
                    "save_steps": len(train_samples) + 1,
                    "playbook_token_budget": controls.max_tokens,
                    "task_name": "agentinputlab_generic_text",
                    "json_mode": self.llm_config.json_mode,
                    "no_ground_truth": False,
                    "save_dir": save_dir,
                    "test_workers": 1,
                    "use_bulletpoint_analyzer": False,
                }
                with contextlib.redirect_stdout(diagnostics):
                    raw_result = ace_system.run(
                        mode="offline",
                        train_samples=train_samples,
                        val_samples=validation_samples,
                        test_samples=None,
                        data_processor=self.data_processor,
                        config=upstream_config,
                    )
        except UnboundLocalError as exc:
            final = exc.__traceback__
            while final is not None and final.tb_next is not None:
                final = final.tb_next
            if (
                final is not None
                and final.tb_frame.f_code.co_name == "evaluate_test_set"
                and Path(final.tb_frame.f_code.co_filename).resolve() == self.checkout / "utils.py"
                and "final_results" in str(exc)
            ):
                if tracker.first_component_failure is not None:
                    raise tracker.first_component_failure from exc
                raise WorkerFailure(
                    "UPSTREAM_VALIDATION_NO_RESULTS",
                    "pinned ACE validation returned no valid results; fit failed, usage retained",
                ) from exc
            raise
        finally:
            random.setstate(random_state)
        tracker.require_upstream_chain()
        score = extract_validation_score(raw_result)
        best_playbook = getattr(ace_system, "best_playbook", None)
        if not isinstance(best_playbook, str):
            raise WorkerFailure("UPSTREAM_RESULT_INVALID", "upstream ACE omitted its best playbook")
        candidate = parse_playbook(best_playbook, initial=request.initial_bullets)
        operations = diff_playbooks(request.initial_bullets, candidate)
        return {
            "status": "ok",
            "request_id": request.request_id,
            "protocol_version": PROTOCOL_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "split_audit_hash": request.split_audit_hash,
            "train_ids": [str(item["example_id"]) for item in request.train_examples],
            "validation_ids": [str(item["example_id"]) for item in request.validation_examples],
            "iterations": [
                {
                    "index": 0,
                    "operations": operations,
                    "candidate_playbook_hash": playbook_hash(candidate),
                    "validation_score": score,
                }
            ],
            "selected_iteration": 0,
            "final_playbook": list(candidate),
            "usage": tracker.as_dict(),
            "diagnostics": {
                **(
                    {"operator_task_processor": self.task_processor_fingerprint}
                    if self.task_processor_fingerprint is not None
                    else {}
                ),
                "upstream_stdout_bytes": diagnostics.byte_count,
                "upstream_stdout_sha256": diagnostics.sha256,
                "content_retained": False,
                "provider_binding": (
                    "operator_openai_compatible"
                    if self.llm_config.base_url is not None
                    else "upstream_native"
                ),
            },
        }

    def _bind_openai_compatible_clients(self, ace_system: Any, *, seed: int) -> None:
        """Replace only upstream client objects using operator-startup configuration."""

        base_url = self.llm_config.base_url
        api_key_env = self.llm_config.api_key_env
        if base_url is None or api_key_env is None:  # pragma: no cover - caller invariant
            raise WorkerFailure("INVALID_LLM_CONFIG", "ACE endpoint binding is incomplete")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise WorkerFailure(
                "MISSING_API_KEY",
                f"ACE OpenAI-compatible runtime requires environment {api_key_env}",
            )
        try:
            openai_module = importlib.import_module("openai")
            client_class = openai_module.OpenAI
            clients = tuple(
                client_class(api_key=api_key, base_url=base_url, max_retries=0) for _ in _ROLE_NAMES
            )
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            raise WorkerFailure(
                "OPENAI_COMPATIBLE_CLIENT_UNAVAILABLE",
                "ACE could not initialize its operator-bound OpenAI-compatible clients",
            ) from exc
        extra_body = dict(self.llm_config.extra_body or {})
        if "seed" in extra_body and (
            type(extra_body["seed"]) is not int or extra_body["seed"] != seed
        ):
            raise WorkerFailure(
                "INVALID_LLM_CONFIG", "deployment seed differs from the fit controls"
            )
        extra_body["seed"] = seed
        clients = tuple(CompletionDefaultClient(client, extra_body) for client in clients)
        components = (
            ace_system.generator,
            ace_system.reflector,
            ace_system.curator,
        )
        for component, client in zip(components, clients, strict=True):
            component.api_client = client
        ace_system.generator_client = clients[0]
        ace_system.reflector_client = clients[1]
        ace_system.curator_client = clients[2]


def load_compatibility_policy() -> CompatibilityPolicy:
    path = Path(__file__).with_name("compatibility_manifest.json")
    try:
        raw = _loads_strict_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkerFailure(
            "COMPATIBILITY_MANIFEST_UNAVAILABLE", "ACE compatibility manifest is unavailable"
        ) from exc
    return CompatibilityPolicy.from_mapping(_require_mapping(raw, "compatibility manifest"))


def verify_checkout(checkout: Path, policy: CompatibilityPolicy) -> Path:
    try:
        root = checkout.expanduser().resolve(strict=True)
    except OSError as exc:
        raise WorkerFailure("CHECKOUT_UNAVAILABLE", "ACE checkout does not exist") from exc
    if not root.is_dir():
        raise WorkerFailure("CHECKOUT_UNAVAILABLE", "ACE checkout must be a directory")
    if read_checkout_head(root) != policy.commit:
        raise WorkerFailure(
            "UPSTREAM_REVISION_MISMATCH", "ACE checkout is not at the pinned commit"
        )
    license_file = safe_checkout_file(root, policy.license_path)
    if file_sha256(license_file) != policy.license_sha256:
        raise WorkerFailure("UPSTREAM_LICENSE_MISMATCH", "ACE license hash does not match")
    license_text = license_file.read_text(encoding="utf-8", errors="strict")[:500]
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise WorkerFailure(
            "UPSTREAM_LICENSE_MISMATCH", "ACE checkout does not contain Apache-2.0 text"
        )
    source_paths: list[Path] = []
    for relative, expected in policy.source_hashes.items():
        source = safe_checkout_file(root, relative)
        if file_sha256(source) != expected:
            raise WorkerFailure("UPSTREAM_SOURCE_MISMATCH", f"ACE source hash mismatch: {relative}")
        source_paths.append(source)
    reject_appworld_imports(source_paths)
    if policy.require_clean_checkout:
        require_clean_git_checkout(root)
    return root


def require_clean_git_checkout(root: Path) -> None:
    """Reject tracked or untracked checkout changes with a fixed Git query."""

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
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerFailure(
            "GIT_STATUS_FAILED", "ACE clean-checkout verification could not run"
        ) from exc
    if result.returncode != 0:
        raise WorkerFailure("GIT_STATUS_FAILED", "ACE clean-checkout verification failed")
    if result.stdout:
        raise WorkerFailure("DIRTY_CHECKOUT", "ACE checkout contains tracked or untracked changes")


def read_checkout_head(root: Path) -> str:
    git_entry = root / ".git"
    if git_entry.is_dir():
        git_dir = git_entry.resolve(strict=True)
    elif git_entry.is_file():
        raw = git_entry.read_text(encoding="utf-8").strip()
        if not raw.startswith("gitdir: "):
            raise WorkerFailure("GIT_METADATA_INVALID", "ACE .git file is malformed")
        candidate = Path(raw.removeprefix("gitdir: "))
        git_dir = (
            (root / candidate).resolve(strict=True)
            if not candidate.is_absolute()
            else candidate.resolve(strict=True)
        )
    else:
        raise WorkerFailure("GIT_METADATA_INVALID", "ACE checkout lacks Git metadata")
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if _COMMIT_RE.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        raise WorkerFailure("GIT_METADATA_INVALID", "ACE HEAD is malformed")
    ref = head.removeprefix("ref: ")
    if not _REF_RE.fullmatch(ref) or ".." in PurePosixPath(ref).parts:
        raise WorkerFailure("GIT_METADATA_INVALID", "ACE HEAD reference is unsafe")
    common_dir = git_dir
    common_file = git_dir / "commondir"
    if common_file.is_file():
        common_candidate = Path(common_file.read_text(encoding="utf-8").strip())
        common_dir = (
            (git_dir / common_candidate).resolve(strict=True)
            if not common_candidate.is_absolute()
            else common_candidate.resolve(strict=True)
        )
    for base in (git_dir, common_dir):
        loose = safe_child(base, ref)
        if loose.is_file():
            value = loose.read_text(encoding="utf-8").strip()
            if _COMMIT_RE.fullmatch(value):
                return value
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                value, separator, name = line.partition(" ")
                if separator and name == ref and _COMMIT_RE.fullmatch(value):
                    return value
    raise WorkerFailure("GIT_METADATA_INVALID", "ACE HEAD commit cannot be resolved")


def safe_checkout_file(root: Path, relative: str) -> Path:
    path = safe_child(root, relative)
    if not path.is_file() or path.is_symlink():
        raise WorkerFailure(
            "UPSTREAM_FILE_UNAVAILABLE", f"ACE pinned file is missing or unsafe: {relative}"
        )
    return path


def safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise WorkerFailure("UNSAFE_PATH", "path escapes its fixed root")
    return candidate


def reject_appworld_imports(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise WorkerFailure(
                "UPSTREAM_SOURCE_INVALID", "ACE pinned Python source cannot be parsed"
            ) from exc
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            if any(_is_appworld_module(name) for name in names):
                raise WorkerFailure("APPWORLD_FORBIDDEN", "ACE worker refuses to import AppWorld")


def load_upstream(checkout: Path) -> tuple[type[Any], Any]:
    checkout = checkout.resolve(strict=True)
    for name in tuple(sys.modules):
        if any(name == root or name.startswith(f"{root}.") for root in _UPSTREAM_MODULES):
            del sys.modules[name]
    old_dont_write = sys.dont_write_bytecode
    had_dotenv_setting = "PYTHON_DOTENV_DISABLED" in os.environ
    old_dotenv_setting = os.environ.get("PYTHON_DOTENV_DISABLED")
    sys.dont_write_bytecode = True
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, str(checkout))
    before = set(sys.modules)
    try:
        with contextlib.redirect_stdout(BoundedHashSink()):
            module = importlib.import_module("ace.ace")
            playbook_utils = importlib.import_module("playbook_utils")
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "openai", "dotenv", "tiktoken"}:
            raise WorkerFailure(
                "UPSTREAM_DEPENDENCY_MISSING",
                "install agentinputlab[ace] in the isolated worker environment",
            ) from None
        raise WorkerFailure(
            "UPSTREAM_IMPORT_FAILED", "pinned ACE import requires its verified source dependencies"
        ) from None
    except Exception as exc:
        raise WorkerFailure(
            "UPSTREAM_IMPORT_FAILED",
            f"pinned ACE import failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        sys.path.remove(str(checkout))
        sys.dont_write_bytecode = old_dont_write
        if had_dotenv_setting:
            assert old_dotenv_setting is not None
            os.environ["PYTHON_DOTENV_DISABLED"] = old_dotenv_setting
        else:
            os.environ.pop("PYTHON_DOTENV_DISABLED", None)
    imported = set(sys.modules) - before
    if any(_is_appworld_module(name) for name in imported):
        raise WorkerFailure("APPWORLD_FORBIDDEN", "ACE worker imported AppWorld")
    for imported_module in (module, playbook_utils):
        origin = getattr(imported_module, "__file__", None)
        if not isinstance(origin, str):
            raise WorkerFailure(
                "UPSTREAM_IMPORT_MISMATCH", "ACE upstream module has no source origin"
            )
        resolved = Path(origin).resolve(strict=True)
        if checkout not in resolved.parents:
            raise WorkerFailure(
                "UPSTREAM_IMPORT_MISMATCH", "ACE upstream module came from another checkout"
            )
    ace_class = getattr(module, "ACE", None)
    if not isinstance(ace_class, type) or not callable(getattr(ace_class, "run", None)):
        raise WorkerFailure("UPSTREAM_INTERFACE_MISMATCH", "pinned ace.ace.ACE.run is unavailable")
    return ace_class, playbook_utils


def validate_request(payload: Mapping[str, Any], llm_config: LLMConfig) -> ValidatedRequest:
    _require_exact_keys(
        payload,
        {
            "request_id",
            "train_examples",
            "validation_examples",
            "initial_playbook",
            "controls",
            "model_fingerprints",
            "split_audit_hash",
            "upstream_commit",
            "protocol_version",
        },
        "ACE request",
    )
    if payload["upstream_commit"] != UPSTREAM_COMMIT:
        raise WorkerFailure(
            "UPSTREAM_REVISION_MISMATCH", "ACE request targets another upstream commit"
        )
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise WorkerFailure("PROTOCOL_MISMATCH", "ACE request protocol version mismatch")
    request_id = _require_hash(payload["request_id"], "ACE request_id")
    split_audit_hash = _require_hash(payload["split_audit_hash"], "ACE split_audit_hash")
    train = validate_examples(payload["train_examples"], "train")
    validation = validate_examples(payload["validation_examples"], "validation")
    train_ids = tuple(str(item["example_id"]) for item in train)
    validation_ids = tuple(str(item["example_id"]) for item in validation)
    if set(train_ids) & set(validation_ids):
        raise WorkerFailure("SPLIT_OVERLAP", "ACE train and validation IDs overlap")
    initial = validate_initial_playbook(payload["initial_playbook"])
    controls = validate_controls(payload["controls"])
    if llm_config.base_url is not None and "seed" in (llm_config.extra_body or {}):
        declared_seed = llm_config.extra_body["seed"]
        if type(declared_seed) is not int or declared_seed != controls.seed:
            raise WorkerFailure(
                "INVALID_LLM_CONFIG", "deployment seed differs from the fit controls"
            )
    validate_model_fingerprints(payload["model_fingerprints"], llm_config)
    expected_split_hash = split_hash(train, validation)
    if split_audit_hash != expected_split_hash:
        raise WorkerFailure("SPLIT_AUDIT_MISMATCH", "ACE split audit hash is invalid")
    request_base = {
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "train_examples": train,
        "validation_examples": validation,
        "initial_playbook_hash": playbook_hash(initial),
        # Preserve the request's numeric JSON representation (for example,
        # ``1`` versus ``1.0``) because the recipe hashes the pre-normalized
        # control mapping when it constructs request_id.
        "controls": dict(_require_mapping(payload["controls"], "ACE controls")),
        "model_fingerprints": payload["model_fingerprints"],
        "split_audit_hash": split_audit_hash,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    if request_id != canonical_hash(request_base):
        raise WorkerFailure("REQUEST_ID_MISMATCH", "ACE request_id is invalid")
    return ValidatedRequest(
        request_id=request_id,
        train_examples=train,
        validation_examples=validation,
        initial_bullets=initial,
        controls=controls,
        split_audit_hash=split_audit_hash,
    )


def validate_examples(value: Any, split: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise WorkerFailure("INVALID_EXAMPLES", f"ACE {split} examples must be non-empty")
    examples: list[Mapping[str, Any]] = []
    ids: list[str] = []
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"ACE {split} example {index}")
        _require_exact_keys(
            item,
            {"example_id", "input", "expected_output", "split", "metadata"},
            f"ACE {split} example {index}",
        )
        example_id = _require_text(item["example_id"], f"ACE {split} example_id")
        if item["split"] != split:
            raise WorkerFailure("INVALID_EXAMPLES", f"ACE {split} example has wrong split")
        if item["expected_output"] is None:
            raise WorkerFailure("INVALID_EXAMPLES", f"ACE {split} example requires expected_output")
        _canonical_json(item["input"])
        _canonical_json(item["expected_output"])
        _canonical_json(_require_mapping(item["metadata"], "ACE example metadata"))
        examples.append(dict(item))
        ids.append(example_id)
    if len(ids) != len(set(ids)):
        raise WorkerFailure("INVALID_EXAMPLES", f"ACE {split} example IDs must be unique")
    return tuple(examples)


def validate_initial_playbook(value: Any) -> tuple[dict[str, Any], ...]:
    playbook = _require_mapping(value, "ACE initial_playbook")
    _require_exact_keys(playbook, {"bullets"}, "ACE initial_playbook")
    raw_bullets = playbook["bullets"]
    if not isinstance(raw_bullets, Sequence) or isinstance(raw_bullets, (str, bytes)):
        raise WorkerFailure("INVALID_PLAYBOOK", "ACE initial bullets must be a sequence")
    bullets: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, raw in enumerate(raw_bullets):
        item = _require_mapping(raw, f"ACE initial bullet {index}")
        _require_exact_keys(
            item,
            {"bullet_id", "content", "helpful", "harmful", "metadata"},
            f"ACE initial bullet {index}",
        )
        bullet_id = _require_text(item["bullet_id"], "ACE bullet_id")
        content = _require_text(item["content"], "ACE bullet content")
        if any(character in bullet_id for character in ("\n", "\r", "]")):
            raise WorkerFailure("INVALID_PLAYBOOK", "ACE bullet_id is not line-safe")
        if "\n" in content or "\r" in content:
            raise WorkerFailure("INVALID_PLAYBOOK", "ACE bullet content must be one line")
        helpful = _require_non_negative_int(item["helpful"], "ACE helpful counter")
        harmful = _require_non_negative_int(item["harmful"], "ACE harmful counter")
        metadata = _require_mapping(item["metadata"], "ACE bullet metadata")
        _canonical_json(metadata)
        bullets.append(
            {
                "bullet_id": bullet_id,
                "content": content,
                "helpful": helpful,
                "harmful": harmful,
                "metadata": dict(metadata),
            }
        )
        ids.append(bullet_id)
    if len(ids) != len(set(ids)):
        raise WorkerFailure("INVALID_PLAYBOOK", "ACE bullet IDs must be unique")
    return tuple(bullets)


def validate_controls(value: Any) -> Controls:
    controls = _require_mapping(value, "ACE controls")
    _require_exact_keys(
        controls,
        {"max_rounds", "max_model_calls", "max_tokens", "max_cost_usd", "seed"},
        "ACE controls",
    )
    return Controls(
        max_rounds=_require_positive_int(controls["max_rounds"], "ACE max_rounds"),
        max_model_calls=_require_positive_int(controls["max_model_calls"], "ACE max_model_calls"),
        max_tokens=_require_positive_int(controls["max_tokens"], "ACE max_tokens"),
        max_cost_usd=_require_non_negative_float(controls["max_cost_usd"], "ACE max_cost_usd"),
        seed=_require_non_negative_int(controls["seed"], "ACE seed"),
    )


def validate_model_fingerprints(value: Any, config: LLMConfig) -> None:
    fingerprints = _require_mapping(value, "ACE model_fingerprints")
    if set(fingerprints) != set(_ROLE_NAMES):
        raise WorkerFailure(
            "MODEL_FINGERPRINT_MISMATCH",
            "ACE request must bind target, reflector, and curator fingerprints",
        )
    for role in _ROLE_NAMES:
        supplied = _require_mapping(fingerprints[role], f"ACE {role} fingerprint")
        if _canonical_json(supplied) != _canonical_json(config.roles[role].fingerprint):
            raise WorkerFailure(
                "MODEL_FINGERPRINT_MISMATCH",
                f"ACE {role} fingerprint differs from worker startup configuration",
            )


def split_hash(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> str:
    value = {
        "train_ids": [str(item["example_id"]) for item in train],
        "validation_ids": [str(item["example_id"]) for item in validation],
        "test_ids": [],
        "train_input_hashes": [canonical_hash(item["input"]) for item in train],
        "validation_input_hashes": [canonical_hash(item["input"]) for item in validation],
        "test_input_hashes": [],
    }
    return canonical_hash(value)


def adapt_example(item: Mapping[str, Any]) -> dict[str, str]:
    raw_input = item["input"]
    context = ""
    if isinstance(raw_input, str):
        question = raw_input
    elif isinstance(raw_input, Mapping):
        aliases = [
            name
            for name in ("question", "query", "prompt", "instruction", "q")
            if name in raw_input
        ]
        if len(aliases) > 1:
            raise WorkerFailure(
                "AMBIGUOUS_TEXT_INPUT", "ACE text input contains multiple question fields"
            )
        question = render_text(raw_input[aliases[0]]) if aliases else render_text(raw_input)
        if "context" in raw_input:
            context = render_text(raw_input["context"])
    else:
        question = render_text(raw_input)
    if not question.strip():
        raise WorkerFailure("INVALID_TEXT_INPUT", "ACE text question must be non-empty")
    target = render_text(item["expected_output"])
    if not target.strip():
        raise WorkerFailure("INVALID_TEXT_INPUT", "ACE text target must be non-empty")
    return {
        "example_id": str(item["example_id"]),
        "context": context,
        "question": question,
        "target": target,
    }


def render_text(value: Any) -> str:
    return value if isinstance(value, str) else _canonical_json(value)


def render_playbook(bullets: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[str]] = {section: [] for section in _STANDARD_SECTIONS}
    for bullet in bullets:
        metadata = _require_mapping(bullet.get("metadata", {}), "ACE bullet metadata")
        raw_section = metadata.get("section", "OTHERS")
        section = normalize_section(raw_section)
        grouped[section].append(
            f"[{bullet['bullet_id']}] helpful={bullet['helpful']} "
            f"harmful={bullet['harmful']} :: {bullet['content']}"
        )
    lines: list[str] = []
    for section in _STANDARD_SECTIONS:
        if lines:
            lines.append("")
        lines.append(f"## {section}")
        lines.extend(grouped[section])
    return "\n".join(lines)


def normalize_section(value: Any) -> str:
    if not isinstance(value, str):
        raise WorkerFailure("INVALID_PLAYBOOK", "ACE bullet section must be text")
    normalized = value.strip().upper().replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    aliases = {
        "STRATEGIES AND INSIGHTS": "STRATEGIES & INSIGHTS",
        "FORMULAS AND CALCULATIONS": "FORMULAS & CALCULATIONS",
        "CODE SNIPPETS AND TEMPLATES": "CODE SNIPPETS & TEMPLATES",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in _STANDARD_SECTIONS else "OTHERS"


def parse_playbook(
    text: str, *, initial: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    initial_by_id = {str(item["bullet_id"]): item for item in initial}
    parsed_by_id: dict[str, dict[str, Any]] = {}
    parsed_order: list[str] = []
    section = "OTHERS"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("##"):
            section = line.removeprefix("##").strip() or "OTHERS"
            continue
        match = _PLAYBOOK_LINE_RE.fullmatch(line)
        if match is None:
            raise WorkerFailure(
                "UPSTREAM_PLAYBOOK_INVALID", "upstream ACE emitted an unparseable playbook line"
            )
        bullet_id, helpful, harmful, content = match.groups()
        if not content.strip():
            raise WorkerFailure("UPSTREAM_PLAYBOOK_INVALID", "upstream ACE emitted an empty bullet")
        if bullet_id in parsed_by_id:
            raise WorkerFailure(
                "UPSTREAM_PLAYBOOK_INVALID", "upstream ACE emitted duplicate bullet IDs"
            )
        old = initial_by_id.get(bullet_id)
        metadata = dict(old.get("metadata", {})) if old is not None else {"section": section}
        parsed_by_id[bullet_id] = {
            "bullet_id": bullet_id,
            "content": content,
            "helpful": int(helpful),
            "harmful": int(harmful),
            "metadata": metadata,
        }
        parsed_order.append(bullet_id)
    stable_order = [
        str(item["bullet_id"]) for item in initial if str(item["bullet_id"]) in parsed_by_id
    ]
    stable_order.extend(item for item in parsed_order if item not in initial_by_id)
    return tuple(parsed_by_id[bullet_id] for bullet_id in stable_order)


def diff_playbooks(
    initial: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    initial_by_id = {str(item["bullet_id"]): item for item in initial}
    candidate_by_id = {str(item["bullet_id"]): item for item in candidate}
    operations: list[dict[str, Any]] = []
    for item in initial:
        bullet_id = str(item["bullet_id"])
        replacement = candidate_by_id.get(bullet_id)
        if replacement is None:
            operations.append({"operation": "delete", "bullet_id": bullet_id})
        elif any(
            item[key] != replacement[key] for key in ("content", "helpful", "harmful", "metadata")
        ):
            operations.append(
                {
                    "operation": "edit",
                    "bullet_id": bullet_id,
                    "content": replacement["content"],
                    "helpful": replacement["helpful"],
                    "harmful": replacement["harmful"],
                    "metadata": replacement["metadata"],
                }
            )
    for item in candidate:
        bullet_id = str(item["bullet_id"])
        if bullet_id not in initial_by_id:
            operations.append(
                {
                    "operation": "add",
                    "bullet_id": bullet_id,
                    "content": item["content"],
                    "helpful": item["helpful"],
                    "harmful": item["harmful"],
                    "metadata": item["metadata"],
                }
            )
    return operations


def playbook_hash(bullets: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash({"bullets": list(bullets)})


def extract_validation_score(value: Any) -> float:
    root = _require_mapping(value, "upstream ACE result")
    training = _require_mapping(root.get("training_results"), "upstream training_results")
    score = _require_non_negative_float(
        training.get("best_validation_accuracy"), "upstream validation accuracy"
    )
    if score > 1:
        raise WorkerFailure("UPSTREAM_RESULT_INVALID", "upstream validation accuracy exceeds one")
    return score


def serve(runtime: ACEWorkerRuntime, input_stream: TextIO, output_stream: TextIO) -> None:
    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        request_id = ""
        runtime.last_tracker = None
        try:
            if len(raw_line.encode("utf-8")) > _MAX_JSONL_BYTES:
                raise WorkerFailure("REQUEST_TOO_LARGE", "ACE JSONL request is too large")
            value = _loads_strict_json(raw_line)
            payload = _require_mapping(value, "ACE JSONL request")
            if isinstance(payload.get("request_id"), str):
                request_id = payload["request_id"][:128]
            response = runtime.run(payload)
        except WorkerFailure as exc:
            response = failure_response(request_id, exc.code, str(exc))
        except Exception as exc:  # pragma: no cover - final process boundary
            response = failure_response(
                request_id,
                "INTERNAL_WORKER_ERROR",
                f"ACE worker failed with {type(exc).__name__}",
            )
        if response.get("status") == "error" and runtime.last_tracker is not None:
            response["usage"] = runtime.last_tracker.as_dict()
            response["consumed"] = runtime.last_tracker.model_calls > 0
            response["failed_attempt_tokens_unknown"] = (
                runtime.last_tracker.failed_provider_attempts > 0
            )
        output_stream.write(_canonical_json(response) + "\n")
        output_stream.flush()


def failure_response(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "request_id": request_id,
        "protocol_version": PROTOCOL_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "failure_code": code,
        "message": message,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
        raise WorkerFailure("NON_CANONICAL_JSON", "value is not canonical JSON") from exc


def _loads_strict_json(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerFailure("INVALID_JSON", "input is not strict JSON") from exc


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise WorkerFailure("INVALID_SCHEMA", f"{label} keys must be strings")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise WorkerFailure("INVALID_SCHEMA", f"{label} fields do not match the protocol")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be non-empty text")
    return value


def _require_env_name(value: Any, label: str) -> str:
    text = _require_text(value, label)
    if _ENV_RE.fullmatch(text) is None:
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be an environment identifier")
    return text


def _require_http_url(value: Any, label: str) -> str:
    text = _require_text(value, label).rstrip("/")
    parsed = urllib.parse.urlsplit(text)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise WorkerFailure(
            "INVALID_SCHEMA",
            f"{label} must be an HTTP(S) URL without credentials, query, or fragment",
        )
    return text


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be a SHA-256 hex digest")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    result = _require_non_negative_int(value, label)
    if result == 0:
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be positive")
    return result


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be a non-negative integer")
    return value


def _require_non_negative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be finite and non-negative")
    return result


def _safe_relative_path(value: Any, label: str) -> str:
    text = _require_text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise WorkerFailure("INVALID_SCHEMA", f"{label} must be a safe relative path")
    return path.as_posix()


def _is_appworld_module(name: str) -> bool:
    lowered = name.casefold().replace("-", "_")
    return lowered == "appworld" or lowered.startswith("appworld.") or "ace_appworld" in lowered


def load_operator_data_processor(factory: str, raw_config: str = "{}") -> Any:
    """Load a caller-selected feedback factory at process startup, never from JSONL."""
    if not isinstance(factory, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", factory
    ):
        raise WorkerFailure("INVALID_TASK_PROCESSOR", "factory must be module:function")
    config = _require_mapping(_loads_strict_json(raw_config), "task processor configuration")
    module_name, symbol = factory.split(":")
    diagnostics = BoundedHashSink()
    with contextlib.redirect_stdout(diagnostics):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise WorkerFailure(
                "INVALID_TASK_PROCESSOR", "cannot import configured feedback factory"
            ) from exc
    source = getattr(module, "__file__", None)
    if source is None or not Path(source).is_file():
        raise WorkerFailure("INVALID_TASK_PROCESSOR", "factory module requires a source identity")
    create = getattr(module, symbol, None)
    if not callable(create):
        raise WorkerFailure("INVALID_TASK_PROCESSOR", "factory symbol must be callable")
    with contextlib.redirect_stdout(diagnostics):
        delegate = create(**dict(config))
    for name in ("answer_is_correct", "evaluate_accuracy", "fingerprint"):
        if not callable(getattr(delegate, name, None)):
            raise WorkerFailure("INVALID_TASK_PROCESSOR", f"factory result lacks {name}()")
    binding = dict(
        factory=factory,
        factory_module_sha256=file_sha256(Path(source)),
        operator_configuration_sha256=canonical_hash(config),
        factory_stdout_sha256=diagnostics.sha256,
        factory_stdout_bytes=diagnostics.byte_count,
    )

    class BoundProcessor:
        def fingerprint(self):
            return dict(operator_binding=binding, task_processor=delegate.fingerprint())

        def __getattr__(self, name):
            return getattr(delegate, name)

    return BoundProcessor()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pinned AgentInputLab ACE JSONL worker")
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument(
        "--llm-config-json",
        help="Explicit provider/model/fingerprint/pricing JSON; required unless --verify-only",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--task-processor-factory",
        help="Operator-bound module:function feedback factory; never accepted in JSONL requests",
    )
    parser.add_argument(
        "--task-processor-config-json",
        help="Startup-only keyword JSON for the feedback factory (no credentials)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = load_compatibility_policy()
    checkout = verify_checkout(args.checkout, policy)
    if args.verify_only:
        load_upstream(checkout)
        sys.stdout.write(
            _canonical_json(
                {
                    "status": "ok",
                    "upstream_commit": UPSTREAM_COMMIT,
                    "license_spdx": policy.license_spdx,
                    "import": "ace.ace.ACE.run",
                }
            )
            + "\n"
        )
        return 0
    if args.llm_config_json is None:
        raise WorkerFailure("INVALID_LLM_CONFIG", "--llm-config-json is required in serve mode")
    if args.task_processor_config_json is not None and args.task_processor_factory is None:
        raise WorkerFailure("INVALID_TASK_PROCESSOR", "processor configuration requires a factory")
    processor = (
        load_operator_data_processor(
            args.task_processor_factory, args.task_processor_config_json or "{}"
        )
        if args.task_processor_factory is not None
        else None
    )
    runtime = ACEWorkerRuntime(
        checkout,
        LLMConfig.from_json(args.llm_config_json),
        policy=policy,
        data_processor=processor,
    )
    serve(runtime, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through JSONL smoke tests
    try:
        raise SystemExit(main())
    except WorkerFailure as error:
        sys.stderr.write(f"{error.code}: {error}\n")
        raise SystemExit(2) from error
