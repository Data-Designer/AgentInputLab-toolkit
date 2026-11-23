#!/usr/bin/env python3
"""Pinned standalone component worker for Ask or Assume?.

The official ``IntentAgent`` is tightly coupled to OpenHands State,
ConversationMemory, Condenser, LLMRegistry, and controller transitions. This
worker therefore does not claim to run UA-Multi. It loads the exact pinned
IntentAgent prompt and ``ClarifyDecisionTool``/``ClarifyTool`` schemas without
executing checkout code, then runs the two separable model paths through one
operator-configured OpenAI-compatible backend.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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
from typing import Any, Protocol, TextIO

UPSTREAM_COMMIT = "728464a755f15fd448aff61a5cb32dce97fd9c1d"
METHOD_ID = "ask_or_assume"
RELEASE_ID = "clarify.ask_or_assume.intent_monitor_component"
PROTOCOL_VERSION = "agentinputlab.ask-or-assume.monitor.v1"
MANIFEST_SCHEMA = "agentinputlab.ask-or-assume.compatibility.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
INTENT_PROMPT_PATH = "openhands/agenthub/intent_agent/prompts/system_prompt.j2"
INTENT_TOOL_PATH = "openhands/agenthub/intent_agent/tools/clarify_decision.py"
CLARIFY_TOOL_PATH = "openhands/agenthub/clarify_agent/tools/clarify.py"
TOOL_NAMES_PATH = "openhands/llm/tool_names.py"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_HISTORY_ITEMS = 2048
MAX_TEXT_CHARS = 500_000
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
        "task_view",
        "history",
        "turn_index",
        "clarification_turns",
        "controls",
        "intent_prompt_hash",
        "seed",
    }
)
_CONTROL_FIELDS = frozenset(
    {
        "skip_first_turn",
        "max_coding_turns",
        "max_user_turns",
        "max_model_calls",
        "max_tokens",
        "max_cost_usd",
    }
)
_ROLE_NAMES = ("intent", "coding")


def _requires_qwen38_initial_system_message(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    alias = model.strip().casefold().replace("_", "-")
    return alias == "qwen3.8" or alias.startswith("qwen3.8-")


def _coalesce_chat_system_messages(messages: Sequence[Any]) -> list[Any]:
    copied = [dict(item) if isinstance(item, Mapping) else item for item in messages]
    system_items: list[dict[str, Any]] = []
    non_system_items: list[Any] = []
    for item in copied:
        if isinstance(item, Mapping) and item.get("role") == "system":
            content = item.get("content")
            if not isinstance(content, str):
                raise DriverError("INVALID_SCHEMA", "Qwen3.8 system message content must be text")
            system_items.append(dict(item))
        else:
            non_system_items.append(item)
    if not system_items:
        return copied
    merged = dict(system_items[0])
    merged["content"] = "\n\n".join(item["content"] for item in system_items)
    return [merged, *non_system_items]


class DriverError(RuntimeError):
    """Expected fail-closed worker error with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CompatibilityPolicy:
    commit: str
    license_path: str
    license_spdx: str
    license_sha256: str
    require_clean_checkout: bool
    excluded_prefixes: tuple[str, ...]
    source_hashes: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompatibilityPolicy:
        if value.get("schema_version") != MANIFEST_SCHEMA:
            raise DriverError("INVALID_MANIFEST", "Ask-or-Assume manifest schema mismatch")
        upstream = _mapping(value.get("upstream"), "manifest upstream")
        sources = _mapping(value.get("source_files"), "manifest source_files")
        commit = _text(upstream.get("commit"), "manifest commit")
        license_spdx = _text(upstream.get("license_spdx"), "manifest license_spdx")
        if commit != UPSTREAM_COMMIT or license_spdx != "MIT":
            raise DriverError(
                "INVALID_MANIFEST",
                "Ask-or-Assume manifest must pin the approved commit and MIT root license",
            )
        clean = upstream.get("require_clean_checkout")
        if not isinstance(clean, bool):
            raise DriverError("INVALID_MANIFEST", "manifest clean-checkout flag must be boolean")
        raw_excluded = upstream.get("excluded_prefixes")
        if not isinstance(raw_excluded, Sequence) or isinstance(raw_excluded, (str, bytes)):
            raise DriverError("INVALID_MANIFEST", "manifest excluded_prefixes must be a sequence")
        excluded = tuple(_safe_prefix(item) for item in raw_excluded)
        if "enterprise/" not in excluded:
            raise DriverError("INVALID_MANIFEST", "manifest must exclude enterprise/")
        source_hashes: dict[str, str] = {}
        for raw_path, raw_hash in sources.items():
            path = _safe_relative_path(raw_path, "manifest source path")
            _reject_excluded_path(path, excluded)
            source_hashes[path] = _hash(raw_hash, f"manifest hash for {path}")
        required = {INTENT_PROMPT_PATH, INTENT_TOOL_PATH, CLARIFY_TOOL_PATH, TOOL_NAMES_PATH}
        if not required <= set(source_hashes):
            raise DriverError("INVALID_MANIFEST", "manifest omits an official prompt/tool source")
        return cls(
            commit=commit,
            license_path=_safe_relative_path(
                _text(upstream.get("license_path"), "manifest license_path"),
                "manifest license_path",
            ),
            license_spdx=license_spdx,
            license_sha256=_hash(upstream.get("license_sha256"), "manifest license hash"),
            require_clean_checkout=clean,
            excluded_prefixes=excluded,
            source_hashes=source_hashes,
        )


@dataclass(frozen=True, slots=True)
class RoleConfig:
    model: str
    fingerprint: Mapping[str, Any]
    max_output_tokens: int
    input_cost_per_million: float
    output_cost_per_million: float


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
    timeout_seconds: float
    roles: Mapping[str, RoleConfig]

    @classmethod
    def from_json(cls, raw: str) -> BackendConfig:
        root = _mapping(_loads_strict_json(raw), "backend configuration")
        _exact_keys(root, {"base_url", "api_key_env", "timeout_seconds", "roles"}, "backend")
        base_url = _validate_base_url(_text(root["base_url"], "backend base_url"))
        api_key_env = _text(root["api_key_env"], "backend api_key_env")
        if _ENV_NAME_RE.fullmatch(api_key_env) is None:
            raise DriverError("INVALID_BACKEND_CONFIG", "backend api_key_env is invalid")
        timeout = _positive_float(root["timeout_seconds"], "backend timeout_seconds")
        raw_roles = _mapping(root["roles"], "backend roles")
        if set(raw_roles) != set(_ROLE_NAMES):
            raise DriverError("INVALID_BACKEND_CONFIG", "backend must bind intent and coding roles")
        roles: dict[str, RoleConfig] = {}
        for role in _ROLE_NAMES:
            item = _mapping(raw_roles[role], f"backend {role} role")
            _exact_keys(
                item,
                {
                    "model",
                    "fingerprint",
                    "max_output_tokens",
                    "input_cost_per_million",
                    "output_cost_per_million",
                },
                f"backend {role} role",
            )
            model = _text(item["model"], f"backend {role} model")
            fingerprint = dict(_mapping(item["fingerprint"], f"backend {role} fingerprint"))
            if not fingerprint or _fingerprint_model(fingerprint) != model:
                raise DriverError(
                    "INVALID_BACKEND_CONFIG",
                    f"backend {role} fingerprint must bind its exact model",
                )
            _canonical_json(fingerprint)
            roles[role] = RoleConfig(
                model=model,
                fingerprint=fingerprint,
                max_output_tokens=_positive_int(
                    item["max_output_tokens"], f"backend {role} max_output_tokens"
                ),
                input_cost_per_million=_nonnegative_float(
                    item["input_cost_per_million"], f"backend {role} input price"
                ),
                output_cost_per_million=_nonnegative_float(
                    item["output_cost_per_million"], f"backend {role} output price"
                ),
            )
        return cls(base_url=base_url, api_key_env=api_key_env, timeout_seconds=timeout, roles=roles)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    arguments: Mapping[str, Any]
    input_tokens: int
    output_tokens: int


class CompletionBackend(Protocol):
    def fingerprint(self, role: str) -> Mapping[str, Any]: ...

    def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        tool: Mapping[str, Any],
        seed: int,
    ) -> CompletionResult: ...


class OpenAICompatibleBackend:
    """Small startup-configured OpenAI-compatible chat-completions client."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def fingerprint(self, role: str) -> Mapping[str, Any]:
        return self.config.roles[role].fingerprint

    def complete(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        tool: Mapping[str, Any],
        seed: int,
    ) -> CompletionResult:
        role_config = self.config.roles[role]
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise DriverError("BACKEND_CREDENTIAL_UNAVAILABLE", "backend credential is unavailable")
        tool_name = _tool_name(tool)
        request_messages = list(messages)
        if _requires_qwen38_initial_system_message(role_config.model):
            request_messages = _coalesce_chat_system_messages(request_messages)
        body = {
            "model": role_config.model,
            "messages": request_messages,
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "temperature": 0,
            "seed": seed,
            "max_tokens": role_config.max_output_tokens,
        }
        encoded = _canonical_json(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(MAX_REQUEST_BYTES + 1)
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise DriverError("BACKEND_CALL_FAILED", "backend completion request failed") from exc
        if len(raw) > MAX_REQUEST_BYTES:
            raise DriverError("BACKEND_RESPONSE_TOO_LARGE", "backend response exceeded its limit")
        root = _mapping(_loads_strict_json(raw.decode("utf-8")), "backend response")
        choices = root.get("choices")
        if (
            not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or len(choices) != 1
        ):
            raise DriverError("BACKEND_OUTPUT_INVALID", "backend must return exactly one choice")
        message = _mapping(_mapping(choices[0], "backend choice").get("message"), "backend message")
        calls = message.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or len(calls) != 1:
            raise DriverError("BACKEND_OUTPUT_INVALID", "backend must return exactly one tool call")
        function = _mapping(
            _mapping(calls[0], "backend tool call").get("function"), "tool function"
        )
        if function.get("name") != tool_name:
            raise DriverError("BACKEND_OUTPUT_INVALID", "backend called a non-forced tool")
        arguments_raw = function.get("arguments")
        if not isinstance(arguments_raw, str):
            raise DriverError("BACKEND_OUTPUT_INVALID", "backend tool arguments must be JSON text")
        arguments = _mapping(_loads_strict_json(arguments_raw), "backend tool arguments")
        usage = _mapping(root.get("usage"), "backend usage")
        return CompletionResult(
            arguments=dict(arguments),
            input_tokens=_nonnegative_int(usage.get("prompt_tokens"), "backend prompt_tokens"),
            output_tokens=_nonnegative_int(
                usage.get("completion_tokens"), "backend completion_tokens"
            ),
        )


@dataclass(frozen=True, slots=True)
class Controls:
    skip_first_turn: bool
    max_coding_turns: int
    max_user_turns: int
    max_model_calls: int
    max_tokens: int
    max_cost_usd: float


@dataclass(frozen=True, slots=True)
class ValidatedRequest:
    request_id: str
    state_hash: str
    task_id: str
    task_view: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    turn_index: int
    clarification_turns: int
    controls: Controls
    seed: int
    dependencies: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class OfficialAssets:
    intent_prompt: str
    intent_prompt_sha256: str
    intent_tool: Mapping[str, Any]
    clarify_tool: Mapping[str, Any]
    intent_tool_schema_hash: str
    clarify_tool_schema_hash: str


class UsageTracker:
    def __init__(self, controls: Controls, config: BackendConfig) -> None:
        self.controls = controls
        self.config = config
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def complete(
        self,
        backend: CompletionBackend,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        tool: Mapping[str, Any],
        seed: int,
    ) -> Mapping[str, Any]:
        if self.model_calls >= self.controls.max_model_calls:
            raise DriverError(
                "MODEL_CALL_BUDGET_EXCEEDED",
                "Ask-or-Assume reached max_model_calls before the next provider attempt",
            )
        self.model_calls += 1
        result = backend.complete(role=role, messages=messages, tool=tool, seed=seed)
        if not isinstance(result, CompletionResult):
            raise DriverError("BACKEND_OUTPUT_INVALID", "backend returned an invalid completion")
        self.input_tokens += _nonnegative_int(result.input_tokens, "completion input_tokens")
        self.output_tokens += _nonnegative_int(result.output_tokens, "completion output_tokens")
        pricing = self.config.roles[role]
        self.cost_usd += (
            result.input_tokens * pricing.input_cost_per_million
            + result.output_tokens * pricing.output_cost_per_million
        ) / 1_000_000
        if self.input_tokens + self.output_tokens > self.controls.max_tokens:
            raise DriverError("TOKEN_BUDGET_EXCEEDED", "Ask-or-Assume exceeded max_tokens")
        if self.cost_usd > self.controls.max_cost_usd:
            raise DriverError("COST_BUDGET_EXCEEDED", "Ask-or-Assume exceeded max_cost_usd")
        return dict(result.arguments)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "model_calls": self.model_calls,
            "retrieval_calls": 0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class AskOrAssumeRuntime:
    """Execute the controlled component harness against one verified checkout."""

    def __init__(
        self,
        checkout: Path,
        backend_config: BackendConfig,
        backend: CompletionBackend,
        *,
        policy: CompatibilityPolicy | None = None,
    ) -> None:
        self.policy = policy or load_compatibility_policy()
        self.checkout = verify_checkout(checkout, self.policy)
        self.assets = load_official_assets(self.checkout)
        self.backend_config = backend_config
        self.backend = backend
        for role in _ROLE_NAMES:
            if _canonical_json(backend.fingerprint(role)) != _canonical_json(
                backend_config.roles[role].fingerprint
            ):
                raise DriverError(
                    "BACKEND_IDENTITY_MISMATCH",
                    f"runtime {role} backend differs from startup configuration",
                )

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = validate_request(payload, self.backend_config, self.assets)
        controls = request.controls
        if request.turn_index >= controls.max_coding_turns:
            raise DriverError(
                "CODING_TURN_BUDGET_EXCEEDED",
                "Ask-or-Assume effective coding-turn budget is exhausted",
            )
        if controls.skip_first_turn and request.turn_index == 0:
            raise DriverError(
                "FIRST_TURN_NOT_DELEGATED",
                "Ask-or-Assume IntentAgent must not run before the first coding turn",
            )
        tracker = UsageTracker(controls, self.backend_config)
        history_document = _canonical_json(
            {
                "task_id": request.task_id,
                "task_view": request.task_view,
                "conversation_history": request.history,
                "effective_coding_turns": request.turn_index,
                "effective_user_turns": request.clarification_turns,
            }
        )
        intent_arguments = tracker.complete(
            self.backend,
            role="intent",
            messages=(
                {"role": "system", "content": self.assets.intent_prompt},
                {"role": "user", "content": history_document},
            ),
            tool=self.assets.intent_tool,
            seed=request.seed,
        )
        decision = _validate_intent_decision(intent_arguments)
        needs_clarification = decision["needs_clarification"]
        reasons = decision["reasons"]
        message = decision["message"]
        questions: tuple[Mapping[str, Any], ...] = ()
        clarification_message = ""
        trace: list[dict[str, Any]] = [
            {
                "index": 0,
                "path": "intent_decision",
                "role": "intent",
                "tool_name": _tool_name(self.assets.intent_tool),
                "decision": needs_clarification,
                "reasons": reasons,
                "message": message,
            }
        ]
        if needs_clarification:
            if request.clarification_turns >= controls.max_user_turns:
                raise DriverError(
                    "USER_TURN_BUDGET_EXCEEDED",
                    "Ask-or-Assume effective user-turn budget is exhausted",
                )
            reminder = "Before proceeding further, ask a clarifying question."
            if reasons:
                reminder += f"\nReason: {reasons.strip()}"
            clarify_arguments = tracker.complete(
                self.backend,
                role="coding",
                messages=(
                    {"role": "user", "content": history_document},
                    {"role": "system", "content": reminder},
                ),
                tool=self.assets.clarify_tool,
                seed=request.seed + 1,
            )
            clarify = _validate_clarify_call(clarify_arguments)
            questions = clarify["questions"]
            clarification_message = _render_official_clarify_message(clarify["message"], questions)
            trace.append(
                {
                    "index": 1,
                    "path": "forced_clarify",
                    "role": "coding",
                    "tool_name": _tool_name(self.assets.clarify_tool),
                    "message": clarify["message"],
                    "questions": questions,
                    "wait_for_response": clarify["wait"],
                    "rendered_message": clarification_message,
                }
            )
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "payload": {
                "needs_clarification": needs_clarification,
                "reasons": [reasons] if reasons else [],
                "message": message,
                "clarification_question": clarification_message,
                "clarification_questions": list(questions),
                "intent_summary": message,
                "transition": (
                    "await"
                    if needs_clarification and clarify["wait"]
                    else "force"
                    if needs_clarification
                    else "resume"
                ),
                "delegated": True,
                "forced": needs_clarification,
                "first_turn_skipped": False,
                "decision_trace": trace,
                "execution_mode": "controlled_component_harness",
                "component_only": True,
                "openhands_full_system": False,
                "ua_multi_parity_claimed": False,
                "intent_prompt_sha256": self.assets.intent_prompt_sha256,
                "intent_tool_schema_hash": self.assets.intent_tool_schema_hash,
                "clarify_tool_schema_hash": self.assets.clarify_tool_schema_hash,
                "effective_coding_turns": request.turn_index,
                "effective_user_turns": request.clarification_turns,
                "model_fingerprints": {
                    role: dict(self.backend_config.roles[role].fingerprint) for role in _ROLE_NAMES
                },
            },
            "usage": tracker.as_dict(),
        }


def load_compatibility_policy(path: Path = DEFAULT_MANIFEST) -> CompatibilityPolicy:
    try:
        value = _loads_strict_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DriverError("MANIFEST_UNAVAILABLE", "compatibility manifest is unavailable") from exc
    return CompatibilityPolicy.from_mapping(_mapping(value, "compatibility manifest"))


def verify_checkout(checkout: Path, policy: CompatibilityPolicy) -> Path:
    try:
        root = checkout.expanduser().resolve(strict=True)
    except OSError as exc:
        raise DriverError("CHECKOUT_UNAVAILABLE", "Ask-or-Assume checkout is unavailable") from exc
    if not root.is_dir():
        raise DriverError("CHECKOUT_UNAVAILABLE", "Ask-or-Assume checkout must be a directory")
    if read_checkout_head(root) != policy.commit:
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "checkout is not at the pinned commit")
    license_path = _checkout_file(root, policy.license_path)
    if _file_sha256(license_path) != policy.license_sha256:
        raise DriverError("UPSTREAM_LICENSE_MISMATCH", "root license hash does not match")
    license_text = license_path.read_text(encoding="utf-8", errors="strict")
    if "MIT License" not in license_text or "enterprise/" not in license_text:
        raise DriverError(
            "UPSTREAM_LICENSE_MISMATCH",
            "root license must identify MIT content and the separate enterprise subtree",
        )
    verified_sources: list[Path] = []
    for relative, expected_hash in policy.source_hashes.items():
        _reject_excluded_path(relative, policy.excluded_prefixes)
        source = _checkout_file(root, relative)
        if _file_sha256(source) != expected_hash:
            raise DriverError("UPSTREAM_SOURCE_MISMATCH", f"source hash mismatch: {relative}")
        verified_sources.append(source)
    _reject_enterprise_imports(verified_sources)
    if policy.require_clean_checkout:
        _require_clean_checkout(root)
    return root


def load_official_assets(checkout: Path) -> OfficialAssets:
    root = checkout.resolve(strict=True)
    prompt_path = _checkout_file(root, INTENT_PROMPT_PATH)
    prompt = prompt_path.read_text(encoding="utf-8", errors="strict")
    names = _literal_assignments(_checkout_file(root, TOOL_NAMES_PATH), {})
    intent_values = _literal_assignments(_checkout_file(root, INTENT_TOOL_PATH), names)
    clarify_values = _literal_assignments(_checkout_file(root, CLARIFY_TOOL_PATH), names)
    intent_tool = _mapping(intent_values.get("ClarifyDecisionTool"), "ClarifyDecisionTool")
    clarify_tool = _mapping(clarify_values.get("ClarifyTool"), "ClarifyTool")
    if _tool_name(intent_tool) != "clarify_decision" or _tool_name(clarify_tool) != "clarify":
        raise DriverError("UPSTREAM_SCHEMA_INVALID", "official tool names are invalid")
    _validate_tool_schema(intent_tool)
    _validate_tool_schema(clarify_tool)
    return OfficialAssets(
        intent_prompt=prompt,
        intent_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        intent_tool=dict(intent_tool),
        clarify_tool=dict(clarify_tool),
        intent_tool_schema_hash=_canonical_hash(intent_tool),
        clarify_tool_schema_hash=_canonical_hash(clarify_tool),
    )


def validate_request(
    value: Mapping[str, Any], config: BackendConfig, assets: OfficialAssets
) -> ValidatedRequest:
    _exact_keys(value, set(_ROOT_FIELDS), "worker request")
    if value["method_id"] != METHOD_ID or value["release_id"] != RELEASE_ID:
        raise DriverError("PROFILE_IDENTITY_MISMATCH", "worker request targets another profile")
    if value["upstream_commit"] != UPSTREAM_COMMIT:
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "worker request targets another revision")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise DriverError("PROTOCOL_MISMATCH", "worker protocol version mismatch")
    request_id = _hash(value["request_id"], "worker request_id")
    state_hash = _hash(value["state_hash"], "worker state_hash")
    identity = {key: value[key] for key in _ROOT_FIELDS if key != "request_id"}
    if request_id != _canonical_hash(identity):
        raise DriverError("REQUEST_ID_MISMATCH", "worker request_id is invalid")
    reservation = _mapping(value["minimum_reservation"], "minimum_reservation")
    if reservation != {"model_calls": 1}:
        raise DriverError("INVALID_RESERVATION", "worker must reserve one IntentAgent model call")
    dependencies = _mapping(value["dependency_fingerprints"], "dependency_fingerprints")
    if set(dependencies) != {"intent_model", "coding_model"}:
        raise DriverError("BACKEND_IDENTITY_MISMATCH", "request must bind both model roles")
    normalized_dependencies: dict[str, Mapping[str, Any]] = {}
    for role, key in (("intent", "intent_model"), ("coding", "coding_model")):
        fingerprint = _mapping(dependencies[key], f"{key} fingerprint")
        if _canonical_json(fingerprint) != _canonical_json(config.roles[role].fingerprint):
            raise DriverError(
                "BACKEND_IDENTITY_MISMATCH", f"request {key} differs from worker startup"
            )
        normalized_dependencies[key] = dict(fingerprint)
    payload = _mapping(value["payload"], "worker payload")
    _exact_keys(payload, set(_PAYLOAD_FIELDS), "worker payload")
    if payload["intent_prompt_hash"] != assets.intent_prompt_sha256:
        raise DriverError("PROMPT_IDENTITY_MISMATCH", "request intent prompt hash is not pinned")
    task_id = payload["task_id"]
    if not isinstance(task_id, str):
        raise DriverError("INVALID_SCHEMA", "task_id must be text")
    task_view = _validate_task_view(payload["task_view"])
    history = _validate_history(payload["history"])
    turn_index = _nonnegative_int(payload["turn_index"], "effective coding turns")
    clarification_turns = _nonnegative_int(payload["clarification_turns"], "effective user turns")
    controls = _validate_controls(payload["controls"])
    if clarification_turns > controls.max_user_turns:
        raise DriverError("USER_TURN_BUDGET_EXCEEDED", "effective user turns exceed the budget")
    return ValidatedRequest(
        request_id=request_id,
        state_hash=state_hash,
        task_id=task_id,
        task_view=task_view,
        history=history,
        turn_index=turn_index,
        clarification_turns=clarification_turns,
        controls=controls,
        seed=_nonnegative_int(payload["seed"], "worker seed"),
        dependencies=normalized_dependencies,
    )


def _validate_task_view(value: Any) -> Mapping[str, Any]:
    task = _mapping(value, "task_view")
    _exact_keys(task, {"query", "instruction", "context", "tool_names"}, "task_view")
    query = _plain_text(task["query"], "task query")
    instruction = _plain_text(task["instruction"], "task instruction")
    if not query.strip() and not instruction.strip():
        raise DriverError("INVALID_SCHEMA", "task view requires query or instruction")
    context = task["context"]
    tool_names = task["tool_names"]
    _canonical_json(context)
    if not isinstance(tool_names, Sequence) or isinstance(tool_names, (str, bytes)):
        raise DriverError("INVALID_SCHEMA", "task tool_names must be a sequence")
    if not all(isinstance(item, str) and item for item in tool_names):
        raise DriverError("INVALID_SCHEMA", "task tool_names must contain non-empty text")
    normalized = {
        "query": query,
        "instruction": instruction,
        "context": context,
        "tool_names": list(tool_names),
    }
    if len(_canonical_json(normalized)) > MAX_TEXT_CHARS:
        raise DriverError("REQUEST_TOO_LARGE", "task view exceeds its text limit")
    return normalized


def _validate_history(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DriverError("INVALID_SCHEMA", "history must be a sequence")
    if len(value) > MAX_HISTORY_ITEMS:
        raise DriverError("REQUEST_TOO_LARGE", "history contains too many messages")
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"history[{index}]")
        _exact_keys(item, {"role", "content", "name"}, f"history[{index}]")
        role = _text(item["role"], f"history[{index}] role")
        content = _plain_text(item["content"], f"history[{index}] content")
        name = item["name"]
        if name is not None and not isinstance(name, str):
            raise DriverError("INVALID_SCHEMA", f"history[{index}] name must be text or null")
        result.append({"role": role, "content": content, "name": name})
    return tuple(result)


def _validate_controls(value: Any) -> Controls:
    item = _mapping(value, "controls")
    _exact_keys(item, set(_CONTROL_FIELDS), "controls")
    skip = item["skip_first_turn"]
    if not isinstance(skip, bool):
        raise DriverError("INVALID_SCHEMA", "skip_first_turn must be boolean")
    return Controls(
        skip_first_turn=skip,
        max_coding_turns=_positive_int(item["max_coding_turns"], "max_coding_turns"),
        max_user_turns=_positive_int(item["max_user_turns"], "max_user_turns"),
        max_model_calls=_positive_int(item["max_model_calls"], "max_model_calls"),
        max_tokens=_positive_int(item["max_tokens"], "max_tokens"),
        max_cost_usd=_nonnegative_float(item["max_cost_usd"], "max_cost_usd"),
    )


def _validate_intent_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) - {"needs_clarification", "message", "reasons"}:
        raise DriverError("BACKEND_OUTPUT_INVALID", "clarify_decision returned unknown fields")
    decision = value.get("needs_clarification")
    if not isinstance(decision, bool):
        raise DriverError("BACKEND_OUTPUT_INVALID", "needs_clarification must be boolean")
    message = value.get("message", "")
    reasons = value.get("reasons", "")
    if not isinstance(message, str) or not isinstance(reasons, str):
        raise DriverError("BACKEND_OUTPUT_INVALID", "decision message and reasons must be text")
    if decision and not reasons.strip():
        raise DriverError("BACKEND_OUTPUT_INVALID", "clarification decision requires reasons")
    return {
        "needs_clarification": decision,
        "message": message.strip(),
        "reasons": reasons.strip(),
    }


def _validate_clarify_call(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) - {"message", "questions", "wait_for_response"}:
        raise DriverError("BACKEND_OUTPUT_INVALID", "clarify call returned unknown fields")
    message = value.get("message", "")
    if not isinstance(message, str):
        raise DriverError("BACKEND_OUTPUT_INVALID", "clarify message must be text")
    wait = value.get("wait_for_response", True)
    if not isinstance(wait, bool):
        raise DriverError("BACKEND_OUTPUT_INVALID", "wait_for_response must be boolean")
    raw_questions = value.get("questions")
    if not isinstance(raw_questions, Sequence) or isinstance(raw_questions, (str, bytes)):
        raise DriverError("BACKEND_OUTPUT_INVALID", "clarify questions must be a sequence")
    if not raw_questions:
        raise DriverError("BACKEND_OUTPUT_INVALID", "clarify requires at least one question")
    questions: list[Mapping[str, Any]] = []
    for index, raw in enumerate(raw_questions):
        item = _mapping(raw, f"clarify question {index}")
        if set(item) - {"id", "text", "options", "default"}:
            raise DriverError("BACKEND_OUTPUT_INVALID", "clarify question has unknown fields")
        question_id = _text(item.get("id"), f"clarify question {index} id")
        text = _text(item.get("text"), f"clarify question {index} text")
        options = item.get("options", [])
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise DriverError("BACKEND_OUTPUT_INVALID", "clarify options must be a sequence")
        if not all(isinstance(option, str) and option.strip() for option in options):
            raise DriverError("BACKEND_OUTPUT_INVALID", "clarify options must be non-empty text")
        default = item.get("default", "")
        if not isinstance(default, str):
            raise DriverError("BACKEND_OUTPUT_INVALID", "clarify default must be text")
        questions.append(
            {"id": question_id, "text": text, "options": list(options), "default": default}
        )
    return {"message": message.strip(), "questions": tuple(questions), "wait": wait}


def _render_official_clarify_message(preamble: str, questions: Sequence[Mapping[str, Any]]) -> str:
    """Mirror the pinned MIT ``clarify_agent.function_calling`` renderer."""

    lines = [f"{preamble}\n" if preamble else "I need some clarification before proceeding:\n"]
    if questions:
        lines.append("**Please answer the following:**\n")
        for index, question in enumerate(questions, 1):
            lines.append(f"Q{index}. {question['text']}")
            lines.append("")
    return "\n".join(lines)


def _literal_assignments(path: Path, initial: Mapping[str, Any]) -> dict[str, Any]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise DriverError("UPSTREAM_SCHEMA_INVALID", "official schema source is invalid") from exc
    environment = dict(initial)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                environment[target.id] = _safe_ast_value(node.value, environment)
            except DriverError:
                if target.id in {"ClarifyDecisionTool", "ClarifyTool"}:
                    raise
    return environment


def _safe_ast_value(node: ast.AST, environment: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if isinstance(node, ast.Dict):
        return {
            _safe_ast_value(key, environment): _safe_ast_value(value, environment)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_ast_value(item, environment) for item in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in {"ChatCompletionToolParam", "ChatCompletionToolParamFunctionChunk"}:
            raise DriverError("UPSTREAM_SCHEMA_INVALID", "official schema uses an unsafe call")
        if node.args or any(keyword.arg is None for keyword in node.keywords):
            raise DriverError(
                "UPSTREAM_SCHEMA_INVALID", "official schema call shape is unsupported"
            )
        return {
            str(keyword.arg): _safe_ast_value(keyword.value, environment)
            for keyword in node.keywords
        }
    raise DriverError("UPSTREAM_SCHEMA_INVALID", "official schema contains unsupported syntax")


def _validate_tool_schema(value: Mapping[str, Any]) -> None:
    _exact_keys(value, {"type", "function"}, "official tool")
    if value["type"] != "function":
        raise DriverError("UPSTREAM_SCHEMA_INVALID", "official tool type is invalid")
    function = _mapping(value["function"], "official tool function")
    _exact_keys(function, {"name", "description", "parameters"}, "official tool function")
    parameters = _mapping(function["parameters"], "official tool parameters")
    if parameters.get("type") != "object" or parameters.get("additionalProperties") is not False:
        raise DriverError("UPSTREAM_SCHEMA_INVALID", "official tool schema is not closed")


def _tool_name(value: Mapping[str, Any]) -> str:
    function = _mapping(value.get("function"), "tool function")
    return _text(function.get("name"), "tool name")


def read_checkout_head(root: Path) -> str:
    git_entry = root / ".git"
    if git_entry.is_dir():
        git_dir = git_entry.resolve(strict=True)
    elif git_entry.is_file():
        raw = git_entry.read_text(encoding="utf-8").strip()
        if not raw.startswith("gitdir: "):
            raise DriverError("GIT_METADATA_INVALID", "checkout .git file is malformed")
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
        raise DriverError("GIT_METADATA_INVALID", "checkout HEAD is malformed")
    ref = head.removeprefix("ref: ")
    if _REF_RE.fullmatch(ref) is None or ".." in PurePosixPath(ref).parts:
        raise DriverError("GIT_METADATA_INVALID", "checkout HEAD reference is unsafe")
    common_dir = git_dir
    common_file = git_dir / "commondir"
    if common_file.is_file():
        candidate = Path(common_file.read_text(encoding="utf-8").strip())
        common_dir = (
            (git_dir / candidate).resolve(strict=True)
            if not candidate.is_absolute()
            else candidate.resolve(strict=True)
        )
    for base in (git_dir, common_dir):
        loose = _safe_child(base, ref)
        if loose.is_file():
            value = loose.read_text(encoding="utf-8").strip()
            if _COMMIT_RE.fullmatch(value):
                return value
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                commit, separator, name = line.partition(" ")
                if separator and name == ref and _COMMIT_RE.fullmatch(commit):
                    return commit
    raise DriverError("GIT_METADATA_INVALID", "checkout HEAD commit cannot be resolved")


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
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DriverError("GIT_STATUS_FAILED", "clean-checkout verification could not run") from exc
    if result.returncode != 0:
        raise DriverError("GIT_STATUS_FAILED", "clean-checkout verification failed")
    if result.stdout:
        raise DriverError("DIRTY_CHECKOUT", "checkout contains tracked or untracked changes")


def _reject_enterprise_imports(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise DriverError("UPSTREAM_SOURCE_INVALID", "pinned source cannot be parsed") from exc
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            if any(name == "enterprise" or name.startswith("enterprise.") for name in names):
                raise DriverError("ENTERPRISE_EXCLUDED", "worker refuses enterprise imports")


def serve(runtime: AskOrAssumeRuntime, input_stream: TextIO, output_stream: TextIO) -> None:
    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        request_id = ""
        try:
            if len(raw_line.encode("utf-8")) > MAX_REQUEST_BYTES:
                raise DriverError("REQUEST_TOO_LARGE", "worker request exceeds its byte limit")
            value = _mapping(_loads_strict_json(raw_line), "JSONL request")
            if isinstance(value.get("request_id"), str):
                request_id = value["request_id"][:128]
            response = runtime.run(value)
        except DriverError as exc:
            response = _failure_response(request_id, exc.code, str(exc))
        except Exception as exc:  # pragma: no cover - final process boundary
            response = _failure_response(
                request_id,
                "INTERNAL_WORKER_ERROR",
                f"Ask-or-Assume worker failed with {type(exc).__name__}",
            )
        output_stream.write(_canonical_json(response) + "\n")
        output_stream.flush()


def _failure_response(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "request_id": request_id,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol_version": PROTOCOL_VERSION,
        "failure_code": code,
        "message": message,
    }


def _checkout_file(root: Path, relative: str) -> Path:
    path = _safe_child(root, relative)
    if not path.is_file() or path.is_symlink():
        raise DriverError("UPSTREAM_FILE_UNAVAILABLE", f"pinned file is unavailable: {relative}")
    return path


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise DriverError("UNSAFE_PATH", "path escapes its fixed root")
    return candidate


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DriverError("INVALID_MANIFEST", f"{label} is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DriverError("INVALID_MANIFEST", f"{label} is unsafe")
    return path.as_posix()


def _safe_prefix(value: Any) -> str:
    result = _safe_relative_path(value, "excluded prefix")
    return f"{result.rstrip('/')}/"


def _reject_excluded_path(relative: str, excluded: Sequence[str]) -> None:
    normalized = f"{relative.rstrip('/')}/" if relative.endswith("/") else relative
    if any(normalized.startswith(prefix) for prefix in excluded):
        raise DriverError(
            "ENTERPRISE_EXCLUDED", "enterprise/ is outside the worker license boundary"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DriverError("INVALID_BACKEND_CONFIG", "backend base_url must be HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DriverError(
            "INVALID_BACKEND_CONFIG", "backend base_url must not contain secrets/query"
        )
    return value.rstrip("/")


def _canonical_hash(value: Any) -> str:
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
        raise DriverError("INVALID_SCHEMA", f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DriverError("INVALID_SCHEMA", f"{label} fields do not match the protocol")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriverError("INVALID_SCHEMA", f"{label} must be non-empty text")
    return value


def _plain_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise DriverError("INVALID_SCHEMA", f"{label} must be text")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise DriverError("INVALID_SCHEMA", f"{label} must be a SHA-256 hash")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriverError("INVALID_SCHEMA", f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be a non-negative integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    result = _nonnegative_float(value, label)
    if result <= 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be positive")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError("INVALID_SCHEMA", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise DriverError("INVALID_SCHEMA", f"{label} must be finite and non-negative")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--backend-config-json")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    policy = load_compatibility_policy()
    checkout = verify_checkout(args.checkout, policy)
    assets = load_official_assets(checkout)
    if args.verify_only:
        print(
            _canonical_json(
                {
                    "status": "ok",
                    "upstream_commit": UPSTREAM_COMMIT,
                    "release_id": RELEASE_ID,
                    "execution_mode": "controlled_component_harness",
                    "component_only": True,
                    "openhands_full_system": False,
                    "intent_prompt_sha256": assets.intent_prompt_sha256,
                    "intent_tool_schema_hash": assets.intent_tool_schema_hash,
                    "clarify_tool_schema_hash": assets.clarify_tool_schema_hash,
                }
            )
        )
        return 0
    if args.backend_config_json is None:
        raise DriverError("INVALID_BACKEND_CONFIG", "serve mode requires backend configuration")
    config = BackendConfig.from_json(args.backend_config_json)
    runtime = AskOrAssumeRuntime(
        checkout,
        config,
        OpenAICompatibleBackend(config),
        policy=policy,
    )
    serve(runtime, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess deployment
    try:
        raise SystemExit(main())
    except DriverError as exc:
        print(_canonical_json(_failure_response("", exc.code, str(exc))))
        raise SystemExit(2) from None
