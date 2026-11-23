#!/usr/bin/env python3
"""Pinned, fail-closed JSONL worker for the official LLMCompiler runtime.

The worker deliberately supports frozen model and tool replays only.  It uses
the official Planner, TaskFetchingUnit, LLMCompiler join, and replan loop; the
local code supplies transport verification, bounded tracing, and side-effect
policy rather than a clean-room implementation of the algorithm.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

UPSTREAM_COMMIT = "a00c9d35507507da70e8c637eee64efc8c1857ae"
PROTOCOL_VERSION = "agentinputlab.llm-compiler.v1"
COMPATIBILITY_MANIFEST_SHA256 = "4b22356ddd587b0fac952511ab53dcf1ae0007b9645e02cd6375122a41ddedd6"
_MANIFEST_PATH = Path(__file__).with_name("compatibility_manifest.json")
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DISTRIBUTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RELEASE_MODES = {
    "plan.llm_compiler.planner_component": "planner_component",
    "plan.llm_compiler.full": "full",
}


class WorkerFailure(RuntimeError):
    """One sanitized, typed failure returned over the JSONL boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        usage: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed
        self.usage = dict(usage) if usage is not None else None


class _DiscardTextIO(io.TextIOBase):
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class Controls:
    max_tasks: int
    max_replans: int
    concurrency: int
    timeout_seconds: float
    streaming: bool


@dataclass(frozen=True, slots=True)
class ValidatedRequest:
    request_id: str
    release_id: str
    execution_mode: str
    query: str
    seed: int
    prompt_hash: str
    tool_specs: tuple[Mapping[str, Any], ...]
    controls: Controls
    backend_fingerprint: Mapping[str, Any]
    tools_fingerprint: Mapping[str, Any]
    tool_argument_order: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedUpstream:
    planner: ModuleType
    task_fetching: ModuleType
    compiler: ModuleType
    tools: ModuleType
    logger: ModuleType


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_hash(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _read_bytes(path: Path, *, label: str, limit: int = _MAX_MANIFEST_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise WorkerFailure("MANIFEST_INVALID", f"{label} must be a regular non-symlink file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise WorkerFailure("MANIFEST_INVALID", f"{label} is unreadable") from exc
    if size > limit:
        raise WorkerFailure("MANIFEST_INVALID", f"{label} exceeds the byte limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WorkerFailure("MANIFEST_INVALID", f"{label} is unreadable") from exc


def _load_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    raw = _read_bytes(path, label=label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerFailure("MANIFEST_INVALID", f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise WorkerFailure("MANIFEST_INVALID", f"{label} must contain a JSON object")
    return value, _sha256_bytes(raw)


def _expect_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise WorkerFailure("SCHEMA_INVALID", f"{label} contains unsupported or missing fields")


def _required_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise WorkerFailure("SCHEMA_INVALID", f"{label}.{key} must be non-empty text")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerFailure("SCHEMA_INVALID", f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result < 1:
        raise WorkerFailure("SCHEMA_INVALID", f"{label} must be positive")
    return result


def _verify_static_manifests() -> Mapping[str, Any]:
    manifest, digest = _load_json(_MANIFEST_PATH, label="LLMCompiler compatibility manifest")
    if digest != COMPATIBILITY_MANIFEST_SHA256:
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "compatibility manifest hash mismatch")
    _expect_keys(
        manifest,
        {
            "schema_version",
            "upstream",
            "source_files",
            "legacy_dependencies",
            "patch_manifest",
            "execution_policy",
        },
        label="compatibility manifest",
    )
    if manifest["schema_version"] != "agentinputlab.llmcompiler.compatibility.v1":
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "compatibility schema mismatch")
    upstream = manifest["upstream"]
    if not isinstance(upstream, Mapping) or upstream.get("commit") != UPSTREAM_COMMIT:
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "pinned upstream identity mismatch")
    patch = manifest["patch_manifest"]
    if not isinstance(patch, Mapping) or set(patch) != {"path", "sha256"}:
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "patch manifest declaration is invalid")
    if patch["path"] != "patch_manifest.json" or not _HEX64.fullmatch(str(patch["sha256"])):
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "patch manifest identity is invalid")
    patch_path = _MANIFEST_PATH.parent / str(patch["path"])
    if (
        _sha256_bytes(_read_bytes(patch_path, label="LLMCompiler patch manifest"))
        != patch["sha256"]
    ):
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "patch manifest hash mismatch")
    source_files = manifest["source_files"]
    legacy = manifest["legacy_dependencies"]
    if not isinstance(source_files, Mapping):
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "source hash manifest is invalid")
    if not isinstance(legacy, Mapping) or set(legacy) != {
        "requirements_sha256",
        "required_versions",
    }:
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "legacy dependency declaration is invalid")
    requirements_sha = legacy["requirements_sha256"]
    if (
        not isinstance(requirements_sha, str)
        or not _HEX64.fullmatch(requirements_sha)
        or requirements_sha != source_files.get("requirements.txt")
    ):
        raise WorkerFailure(
            "WORKER_INTEGRITY_FAILURE",
            "legacy dependency requirements hash is invalid",
        )
    versions = legacy["required_versions"]
    if not isinstance(versions, Mapping) or not versions:
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "required dependency versions are empty")
    for distribution_name, version in versions.items():
        if (
            not isinstance(distribution_name, str)
            or not _DISTRIBUTION_NAME.fullmatch(distribution_name)
            or not isinstance(version, str)
            or not version.strip()
        ):
            raise WorkerFailure(
                "WORKER_INTEGRITY_FAILURE",
                "required dependency version entry is invalid",
            )
    return manifest


def _verify_required_dependencies(manifest: Mapping[str, Any]) -> None:
    """Verify every manifest-pinned distribution and its imported package origin."""

    legacy = manifest["legacy_dependencies"]
    assert isinstance(legacy, Mapping)  # established by _verify_static_manifests
    required_versions = legacy["required_versions"]
    assert isinstance(required_versions, Mapping)
    for distribution_name, expected_version in required_versions.items():
        assert isinstance(distribution_name, str)
        assert isinstance(expected_version, str)
        import_name = distribution_name.replace("-", "_")
        requirement = f"{distribution_name}=={expected_version}"
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise WorkerFailure(
                "LEGACY_DEPENDENCY_MISSING",
                f"pinned LLMCompiler dependency is unavailable: {requirement}",
            ) from exc
        if distribution.version != expected_version:
            raise WorkerFailure(
                "LEGACY_DEPENDENCY_VERSION_MISMATCH",
                f"pinned LLMCompiler requires {requirement}",
            )
        try:
            module = importlib.import_module(import_name)
        except ModuleNotFoundError as exc:
            raise WorkerFailure(
                "LEGACY_DEPENDENCY_MISSING",
                f"pinned LLMCompiler dependency cannot be imported: {requirement}",
            ) from exc
        except Exception as exc:
            raise WorkerFailure(
                "LEGACY_DEPENDENCY_IMPORT_FAILURE",
                f"pinned LLMCompiler dependency import failed: {requirement}",
            ) from exc
        module_file = getattr(module, "__file__", None)
        try:
            distribution_package = Path(distribution.locate_file(import_name)).resolve()
            module_path = Path(module_file).resolve() if isinstance(module_file, str) else None
            if module_path is None:
                raise ValueError
            module_path.relative_to(distribution_package)
        except (OSError, TypeError, ValueError) as exc:
            raise WorkerFailure(
                "LEGACY_DEPENDENCY_ORIGIN_MISMATCH",
                f"pinned LLMCompiler dependency escaped its distribution: {distribution_name}",
            ) from exc


def _git(checkout: Path, *arguments: str) -> str:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        result = subprocess.run(
            ("git", "-C", str(checkout), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise WorkerFailure("GIT_UNAVAILABLE", "git is required to verify the checkout") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkerFailure("CHECKOUT_VERIFICATION_FAILED", "git verification failed") from exc
    if result.returncode != 0:
        raise WorkerFailure("CHECKOUT_VERIFICATION_FAILED", "git verification failed")
    return result.stdout.strip()


def verify_checkout(checkout: Path) -> Mapping[str, Any]:
    """Verify commit, cleanliness, license, source hashes, and patch attestation."""

    manifest = _verify_static_manifests()
    checkout = checkout.resolve()
    if not checkout.is_dir():
        raise WorkerFailure("CHECKOUT_MISSING", "the pinned LLMCompiler checkout is missing")
    if _git(checkout, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise WorkerFailure("CHECKOUT_REVISION_MISMATCH", "LLMCompiler checkout revision mismatch")
    if _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
        raise WorkerFailure("CHECKOUT_DIRTY", "LLMCompiler checkout must be clean")
    upstream = manifest["upstream"]
    license_path = checkout / "LICENSE"
    expected_license = str(upstream["license_sha256"])
    if _sha256_bytes(_read_bytes(license_path, label="upstream LICENSE")) != expected_license:
        raise WorkerFailure("LICENSE_MISMATCH", "LLMCompiler MIT license hash mismatch")
    source_files = manifest["source_files"]
    if not isinstance(source_files, Mapping) or not source_files:
        raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "source hash manifest is empty")
    for relative, expected in source_files.items():
        if not isinstance(relative, str) or not _HEX64.fullmatch(str(expected)):
            raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "source hash entry is invalid")
        candidate = (checkout / relative).resolve()
        try:
            candidate.relative_to(checkout)
        except ValueError as exc:
            raise WorkerFailure("WORKER_INTEGRITY_FAILURE", "source path escapes checkout") from exc
        actual = _sha256_bytes(_read_bytes(candidate, label=f"upstream source {relative}"))
        if actual != expected:
            raise WorkerFailure("SOURCE_MISMATCH", f"pinned upstream source mismatch: {relative}")
    return manifest


def _validate_controls(value: Any) -> Controls:
    if not isinstance(value, Mapping):
        raise WorkerFailure("SCHEMA_INVALID", "payload.controls must be an object")
    _expect_keys(
        value,
        {"max_tasks", "max_replans", "concurrency", "timeout_seconds", "streaming"},
        label="payload.controls",
    )
    timeout = value["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise WorkerFailure("SCHEMA_INVALID", "controls.timeout_seconds must be positive")
    streaming = value["streaming"]
    if not isinstance(streaming, bool):
        raise WorkerFailure("SCHEMA_INVALID", "controls.streaming must be boolean")
    return Controls(
        max_tasks=_positive_int(value["max_tasks"], label="controls.max_tasks"),
        max_replans=_nonnegative_int(value["max_replans"], label="controls.max_replans"),
        concurrency=_positive_int(value["concurrency"], label="controls.concurrency"),
        timeout_seconds=float(timeout),
        streaming=streaming,
    )


def _validate_tool_specs(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise WorkerFailure("SCHEMA_INVALID", "payload.tool_specs must be a non-empty list")
    result: list[Mapping[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise WorkerFailure("SCHEMA_INVALID", f"tool_specs[{index}] must be an object")
        _expect_keys(
            item,
            {"name", "description", "input_schema", "read_only", "network"},
            label=f"tool_specs[{index}]",
        )
        name = _required_text(item, "name", label=f"tool_specs[{index}]")
        if not _TOOL_NAME.fullmatch(name) or name == "join" or name in names:
            raise WorkerFailure("TOOL_POLICY_DENIED", "tool names must be unique safe identifiers")
        if not isinstance(item["description"], str) or not item["description"].strip():
            raise WorkerFailure("SCHEMA_INVALID", "tool description must be non-empty")
        if not isinstance(item["input_schema"], Mapping):
            raise WorkerFailure("SCHEMA_INVALID", "tool input_schema must be an object")
        if item["read_only"] is not True or item["network"] is not False:
            raise WorkerFailure(
                "TOOL_POLICY_DENIED",
                "LLMCompiler worker accepts only read-only, no-network tools",
            )
        try:
            _canonical_json(item["input_schema"])
        except (TypeError, ValueError) as exc:
            raise WorkerFailure("SCHEMA_INVALID", "tool schema is not canonical JSON") from exc
        names.add(name)
        result.append(dict(item))
    return tuple(result)


def validate_request(value: Any) -> ValidatedRequest:
    if not isinstance(value, Mapping):
        raise WorkerFailure("SCHEMA_INVALID", "worker request must be a JSON object")
    _expect_keys(
        value,
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
        },
        label="request",
    )
    request_id = _required_text(value, "request_id", label="request")
    if not _HEX64.fullmatch(request_id):
        raise WorkerFailure("REQUEST_ID_MISMATCH", "request_id must be a SHA-256 digest")
    identity = {key: value[key] for key in value if key != "request_id"}
    try:
        expected_request_id = _canonical_hash(identity)
    except (TypeError, ValueError) as exc:
        raise WorkerFailure("SCHEMA_INVALID", "request is not canonical JSON") from exc
    if request_id != expected_request_id:
        raise WorkerFailure("REQUEST_ID_MISMATCH", "request_id content hash mismatch")
    if value["method_id"] != "llm_compiler":
        raise WorkerFailure("IDENTITY_MISMATCH", "worker accepts only llm_compiler")
    release_id = _required_text(value, "release_id", label="request")
    if release_id not in _RELEASE_MODES:
        raise WorkerFailure("IDENTITY_MISMATCH", "unsupported LLMCompiler release")
    if value["upstream_commit"] != UPSTREAM_COMMIT:
        raise WorkerFailure("IDENTITY_MISMATCH", "upstream commit mismatch")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise WorkerFailure("IDENTITY_MISMATCH", "protocol version mismatch")
    _required_text(value, "state_hash", label="request")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise WorkerFailure("SCHEMA_INVALID", "request.payload must be an object")
    expected_payload = {
            "task_id",
            "query",
            "instruction",
            "messages",
            "tool_specs",
            "controls",
            "prompt_hash",
            "execution_mode",
            "seed",
    }
    if (not expected_payload <= set(payload)
            or set(payload) - expected_payload - {"execution_budget", "tool_argument_order"}):
        raise WorkerFailure("SCHEMA_INVALID", "payload contains unsupported or missing fields")
    query = _required_text(payload, "query", label="payload")
    if payload["instruction"] != "" or payload["messages"] != []:
        raise WorkerFailure(
            "UNSUPPORTED_CONVERSATION_STATE",
            "official LLMCompiler accepts a single query without instruction/history",
        )
    mode = _required_text(payload, "execution_mode", label="payload")
    if mode != _RELEASE_MODES[release_id]:
        raise WorkerFailure("IDENTITY_MISMATCH", "release and execution_mode disagree")
    seed = _nonnegative_int(payload["seed"], label="payload.seed")
    dependencies = value["dependency_fingerprints"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != {"backend", "tools"}:
        raise WorkerFailure("SCHEMA_INVALID", "dependency fingerprints must bind backend/tools")
    backend = dependencies["backend"]
    tools = dependencies["tools"]
    if not isinstance(backend, Mapping) or not isinstance(tools, Mapping):
        raise WorkerFailure("SCHEMA_INVALID", "dependency fingerprints must be objects")
    reservation = value["minimum_reservation"]
    if (
        not isinstance(reservation, Mapping)
        or _nonnegative_int(reservation.get("model_calls"), label="minimum_reservation.model_calls")
        < 1
    ):
        raise WorkerFailure("SCHEMA_INVALID", "worker requires a model-call reservation")
    specs = _validate_tool_specs(payload["tool_specs"])
    order = payload.get("tool_argument_order", {})
    by_name = {spec["name"]: spec for spec in specs}
    if (not isinstance(order, Mapping) or not set(order) <= set(by_name)
            or any(not isinstance(keys, (list, tuple)) or not all(isinstance(k, str) for k in keys)
                or len(set(keys)) != len(keys)
                or set(keys) != set(by_name[name]["input_schema"].get("properties", {}))
                for name, keys in order.items())):
        raise WorkerFailure("SCHEMA_INVALID", "tool_argument_order must bind exact tool parameter names")
    return ValidatedRequest(
        request_id=request_id,
        release_id=release_id,
        execution_mode=mode,
        query=query,
        seed=seed,
        prompt_hash=_required_text(payload, "prompt_hash", label="payload"),
        tool_specs=specs,
        controls=_validate_controls(payload["controls"]),
        backend_fingerprint=dict(backend),
        tools_fingerprint=dict(tools),
        tool_argument_order={name: tuple(keys) for name, keys in order.items()},
    )


def _fingerprint_sha(value: Mapping[str, Any], key: str, *, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not _HEX64.fullmatch(result):
        raise WorkerFailure("DEPENDENCY_MISMATCH", f"{label} must bind {key}")
    return result


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    channel: str
    prompt_sha256: str
    response: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    prompts: Mapping[str, Any]
    entries: tuple[ReplayEntry, ...]


def load_model_replay(path: Path, request: ValidatedRequest) -> ReplayBundle:
    value, digest = _load_json(path, label="LLMCompiler model replay")
    if digest != _fingerprint_sha(
        request.backend_fingerprint,
        "replay_manifest_sha256",
        label="backend fingerprint",
    ):
        raise WorkerFailure("DEPENDENCY_MISMATCH", "model replay manifest hash mismatch")
    _expect_keys(
        value,
        {"schema_version", "upstream_commit", "seed", "prompts", "prompt_hash", "responses"},
        label="model replay",
    )
    if (
        value["schema_version"] != "agentinputlab.llmcompiler.model-replay.v1"
        or value["upstream_commit"] != UPSTREAM_COMMIT
        or value["seed"] != request.seed
    ):
        raise WorkerFailure("DEPENDENCY_MISMATCH", "model replay identity mismatch")
    prompts = value["prompts"]
    if not isinstance(prompts, Mapping):
        raise WorkerFailure("MANIFEST_INVALID", "model replay prompts must be an object")
    _expect_keys(
        prompts,
        {
            "planner_example_prompt",
            "planner_example_prompt_replan",
            "planner_stop",
            "joinner_prompt",
            "joinner_prompt_final",
        },
        label="model replay prompts",
    )
    if not all(
        isinstance(prompts[key], str)
        for key in (
            "planner_example_prompt",
            "planner_example_prompt_replan",
            "joinner_prompt",
            "joinner_prompt_final",
        )
    ):
        raise WorkerFailure("MANIFEST_INVALID", "model replay prompt fields must be text")
    stop = prompts["planner_stop"]
    if (
        isinstance(stop, (str, bytes))
        or not isinstance(stop, Sequence)
        or not all(isinstance(item, str) and item for item in stop)
    ):
        raise WorkerFailure("MANIFEST_INVALID", "planner_stop must be a list of strings")
    computed_prompt_hash = _canonical_hash(prompts)
    if value["prompt_hash"] != computed_prompt_hash or request.prompt_hash != computed_prompt_hash:
        raise WorkerFailure("DEPENDENCY_MISMATCH", "prompt bundle hash mismatch")
    responses = value["responses"]
    if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence) or not responses:
        raise WorkerFailure("MANIFEST_INVALID", "model replay responses must be non-empty")
    entries: list[ReplayEntry] = []
    for index, item in enumerate(responses):
        if not isinstance(item, Mapping):
            raise WorkerFailure("MANIFEST_INVALID", f"model replay response {index} is invalid")
        _expect_keys(
            item,
            {
                "channel",
                "prompt_sha256",
                "response",
                "response_sha256",
                "input_tokens",
                "output_tokens",
            },
            label=f"model replay response {index}",
        )
        channel = item["channel"]
        response = item["response"]
        if channel not in {"planner", "join"} or not isinstance(response, str) or not response:
            raise WorkerFailure("MANIFEST_INVALID", "model replay channel/response is invalid")
        if not _HEX64.fullmatch(str(item["prompt_sha256"])):
            raise WorkerFailure("MANIFEST_INVALID", "model replay prompt hash is invalid")
        if item["response_sha256"] != _sha256_text(response):
            raise WorkerFailure("MANIFEST_INVALID", "model replay response hash mismatch")
        entries.append(
            ReplayEntry(
                channel=channel,
                prompt_sha256=str(item["prompt_sha256"]),
                response=response,
                input_tokens=_nonnegative_int(
                    item["input_tokens"], label="model replay input_tokens"
                ),
                output_tokens=_nonnegative_int(
                    item["output_tokens"], label="model replay output_tokens"
                ),
            )
        )
    return ReplayBundle(prompts=dict(prompts), entries=tuple(entries))


@dataclass(slots=True)
class ToolReplay:
    cases: Mapping[str, Any]
    calls: int = 0

    async def invoke(self, name: str, arguments: Sequence[Any]) -> Any:
        self.calls += 1
        try:
            key = _canonical_hash(list(arguments))
        except (TypeError, ValueError) as exc:
            raise WorkerFailure(
                "TOOL_ARGUMENT_DENIED",
                "tool arguments are not canonical JSON",
                consumed=True,
            ) from exc
        compound = f"{name}:{key}"
        if compound not in self.cases:
            raise WorkerFailure(
                "TOOL_REPLAY_MISS",
                "tool arguments are absent from the frozen allowlist",
                consumed=True,
            )
        return self.cases[compound]


def load_tool_replay(path: Path, request: ValidatedRequest) -> ToolReplay:
    value, digest = _load_json(path, label="LLMCompiler tool replay")
    if digest != _fingerprint_sha(
        request.tools_fingerprint,
        "tool_manifest_sha256",
        label="tools fingerprint",
    ):
        raise WorkerFailure("DEPENDENCY_MISMATCH", "tool replay manifest hash mismatch")
    _expect_keys(value, {"schema_version", "upstream_commit", "tools"}, label="tool replay")
    if (
        value["schema_version"] != "agentinputlab.llmcompiler.tool-replay.v1"
        or value["upstream_commit"] != UPSTREAM_COMMIT
    ):
        raise WorkerFailure("DEPENDENCY_MISMATCH", "tool replay identity mismatch")
    tools = value["tools"]
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise WorkerFailure("MANIFEST_INVALID", "tool replay tools must be a list")
    by_name: dict[str, Mapping[str, Any]] = {}
    cases: dict[str, Any] = {}
    for index, item in enumerate(tools):
        if not isinstance(item, Mapping):
            raise WorkerFailure("MANIFEST_INVALID", f"tool replay item {index} is invalid")
        _expect_keys(
            item,
            {
                "name",
                "description",
                "input_schema",
                "read_only",
                "network",
                "implementation",
                "cases",
            },
            label=f"tool replay item {index}",
        )
        name = _required_text(item, "name", label=f"tool replay item {index}")
        if name in by_name or not _TOOL_NAME.fullmatch(name) or name == "join":
            raise WorkerFailure("MANIFEST_INVALID", "tool replay names are invalid or duplicated")
        if (
            item["read_only"] is not True
            or item["network"] is not False
            or item["implementation"] != "replay_table"
            or not isinstance(item["input_schema"], Mapping)
        ):
            raise WorkerFailure("TOOL_POLICY_DENIED", "tool replay violates no-side-effect policy")
        rows = item["cases"]
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise WorkerFailure("MANIFEST_INVALID", "tool replay cases must be a list")
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"arguments", "output"}:
                raise WorkerFailure("MANIFEST_INVALID", "tool replay case is invalid")
            arguments = row["arguments"]
            if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
                raise WorkerFailure("MANIFEST_INVALID", "tool replay arguments must be a list")
            try:
                key = f"{name}:{_canonical_hash(list(arguments))}"
                _canonical_json(row["output"])
            except (TypeError, ValueError) as exc:
                raise WorkerFailure(
                    "MANIFEST_INVALID", "tool replay case is not canonical"
                ) from exc
            if key in cases:
                raise WorkerFailure("MANIFEST_INVALID", "tool replay contains duplicate cases")
            cases[key] = row["output"]
        by_name[name] = item
    requested_names = {str(item["name"]) for item in request.tool_specs}
    if not requested_names <= set(by_name):
        raise WorkerFailure(
            "TOOL_POLICY_DENIED", "request contains a tool absent from the manifest"
        )
    for spec in request.tool_specs:
        manifest_spec = by_name[str(spec["name"])]
        comparable = {
            key: manifest_spec[key]
            for key in ("name", "description", "input_schema", "read_only", "network")
        }
        if _canonical_json(comparable) != _canonical_json(spec):
            raise WorkerFailure("DEPENDENCY_MISMATCH", "tool schema differs from frozen manifest")
    return ToolReplay(cases=cases)


class _ReplayModel:
    def __init__(self, entries: Sequence[ReplayEntry]) -> None:
        self._entries = list(entries)
        self._cursor = 0
        self.consumed: list[ReplayEntry] = []
        self.planner_responses: list[str] = []

    def usage(self, *, tool_calls: int = 0) -> dict[str, int]:
        return {
            "model_calls": len(self.consumed),
            "retrieval_calls": 0,
            "tool_calls": tool_calls,
            "input_tokens": sum(item.input_tokens for item in self.consumed),
            "output_tokens": sum(item.output_tokens for item in self.consumed),
        }

    def _consume(self, channel: str, prompt: str) -> ReplayEntry:
        if self._cursor >= len(self._entries):
            raise WorkerFailure(
                "MODEL_REPLAY_EXHAUSTED",
                "model replay has no response for this call",
                consumed=bool(self.consumed),
                usage=self.usage(),
            )
        entry = self._entries[self._cursor]
        if entry.channel != channel or entry.prompt_sha256 != _sha256_text(prompt):
            raise WorkerFailure(
                "MODEL_PROMPT_MISMATCH",
                "model replay prompt/channel mismatch",
                consumed=bool(self.consumed),
                usage=self.usage(),
            )
        self._cursor += 1
        self.consumed.append(entry)
        if channel == "planner":
            self.planner_responses.append(entry.response)
        return entry

    @staticmethod
    async def _notify(callbacks: Any, response: str) -> None:
        if callbacks is None:
            return
        items = callbacks if isinstance(callbacks, Sequence) else (callbacks,)
        run_id = uuid4()
        for token in response:
            for callback in items:
                method = getattr(callback, "on_llm_new_token", None)
                if callable(method):
                    pending = method(token, run_id=run_id)
                    if inspect.isawaitable(pending):
                        await pending
        for callback in items:
            method = getattr(callback, "on_llm_end", None)
            if callable(method):
                pending = method(None, run_id=run_id)
                if inspect.isawaitable(pending):
                    await pending

    async def apredict(
        self,
        prompt: str,
        *,
        callbacks: Any = None,
        stop: Any = None,
        **_: Any,
    ) -> str:
        del stop
        entry = self._consume("planner", prompt)
        await self._notify(callbacks, entry.response)
        return entry.response

    async def agenerate_prompt(
        self,
        prompts: Sequence[Any],
        stop: Any = None,
        callbacks: Any = None,
        **_: Any,
    ) -> Any:
        del stop, callbacks
        if len(prompts) != 1:
            raise WorkerFailure("MODEL_REPLAY_INVALID_CALL", "join requires exactly one prompt")
        value = prompts[0]
        if hasattr(value, "text"):
            prompt = value.text
        elif callable(getattr(value, "to_string", None)):
            prompt = value.to_string()
        else:
            prompt = str(value)
        entry = self._consume("join", prompt)
        generation = SimpleNamespace(text=entry.response)
        return SimpleNamespace(generations=[[generation]])

    def assert_exhausted(self) -> None:
        if self._cursor != len(self._entries):
            raise WorkerFailure(
                "MODEL_REPLAY_UNUSED",
                "model replay contains unused responses",
                consumed=bool(self.consumed),
                usage=self.usage(),
            )


def _module_in_checkout(module: ModuleType, checkout: Path) -> bool:
    location = getattr(module, "__file__", None)
    if not isinstance(location, str):
        return False
    try:
        Path(location).resolve().relative_to(checkout.resolve())
    except ValueError:
        return False
    return True


def import_upstream(checkout: Path) -> LoadedUpstream:
    """Import only fixed module names after checkout verification."""

    checkout = checkout.resolve()
    manifest = _verify_static_manifests()
    _verify_required_dependencies(manifest)
    sys.path.insert(0, str(checkout))
    discarded = _DiscardTextIO()
    try:
        with contextlib.redirect_stdout(discarded):
            task_module = importlib.import_module("src.llm_compiler.task_fetching_unit")
            tools_module = importlib.import_module("src.tools.base")
            planner_module = importlib.import_module("src.llm_compiler.planner")
            compiler_module = importlib.import_module("src.llm_compiler.llm_compiler")
            logger_module = importlib.import_module("src.utils.logger_utils")
    except ModuleNotFoundError as exc:
        missing = str(exc.name or "legacy dependency").split(".")[0]
        raise WorkerFailure(
            "LEGACY_DEPENDENCY_MISSING",
            f"pinned LLMCompiler dependency is unavailable: {missing}",
        ) from exc
    except Exception as exc:
        raise WorkerFailure(
            "UPSTREAM_IMPORT_FAILURE",
            f"pinned LLMCompiler import failed: {type(exc).__name__}",
        ) from exc
    modules = (task_module, tools_module, planner_module, compiler_module, logger_module)
    if not all(_module_in_checkout(module, checkout) for module in modules):
        raise WorkerFailure("UPSTREAM_IMPORT_FAILURE", "upstream module escaped pinned checkout")
    enable_logging = getattr(logger_module, "enable_logging", None)
    if callable(enable_logging):
        enable_logging(False)
    return LoadedUpstream(
        planner=planner_module,
        task_fetching=task_module,
        compiler=compiler_module,
        tools=tools_module,
        logger=logger_module,
    )


def _patch_model_type_gates(upstream: LoadedUpstream) -> None:
    class _NeverChatModel:
        pass

    upstream.planner.BaseChatModel = _NeverChatModel
    upstream.planner.BaseLLM = _ReplayModel
    upstream.compiler.BaseChatModel = _NeverChatModel
    upstream.compiler.BaseLLM = _ReplayModel


def _build_tools(
    upstream: LoadedUpstream,
    specs: Sequence[Mapping[str, Any]],
    replay: ToolReplay,
) -> list[Any]:
    tool_type = getattr(upstream.tools, "Tool", None)
    if tool_type is None:
        raise WorkerFailure("UPSTREAM_API_MISMATCH", "upstream Tool is unavailable")
    result: list[Any] = []
    for spec in specs:
        name = str(spec["name"])

        async def invoke(*arguments: Any, _name: str = name) -> Any:
            return await replay.invoke(_name, arguments)

        try:
            result.append(tool_type(name=name, func=invoke, description=str(spec["description"])))
        except Exception as exc:
            raise WorkerFailure(
                "UPSTREAM_API_MISMATCH",
                f"upstream Tool construction failed: {type(exc).__name__}",
            ) from exc
    return result


@dataclass(slots=True)
class TraceRecorder:
    allowed_tools: frozenset[str]
    max_tasks: int
    concurrency: int
    argument_order: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    tasks: dict[int, Any] = field(default_factory=dict)
    scheduled_ids: set[int] = field(default_factory=set)
    dependency_graph: dict[int, set[int]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    failure: WorkerFailure | None = None
    active_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    planner_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    def register_batch(self, tasks: Mapping[Any, Any]) -> None:
        """Validate one planner batch atomically before the scheduler can see it."""

        if not isinstance(tasks, Mapping):
            raise WorkerFailure("UPSTREAM_PLAN_INVALID", "upstream task batch is malformed")
        normalized: list[tuple[int, Any, bool, set[int]]] = []
        batch_ids: set[int] = set()
        executable_count = 0
        for key, task in tasks.items():
            task_id = getattr(task, "idx", None)
            name = getattr(task, "name", None)
            dependencies = getattr(task, "dependencies", None)
            is_join = bool(getattr(task, "is_join", False))
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id < 1
                or isinstance(key, bool)
                or not isinstance(key, int)
                or key != task_id
                or not isinstance(name, str)
                or (is_join and name != "join")
                or (not is_join and name not in self.allowed_tools)
                or isinstance(dependencies, (str, bytes, Mapping))
                or not isinstance(dependencies, (Sequence, set, frozenset))
            ):
                raise WorkerFailure("UPSTREAM_PLAN_INVALID", "upstream task is malformed")
            dependency_tuple = tuple(dependencies)
            if any(
                isinstance(dependency, bool) or not isinstance(dependency, int) or dependency < 1
                for dependency in dependency_tuple
            ):
                raise WorkerFailure("UPSTREAM_PLAN_INVALID", "upstream dependency is malformed")
            dependency_ids = set(dependency_tuple)
            if len(dependency_ids) != len(dependency_tuple) or task_id in dependency_ids:
                raise WorkerFailure(
                    "UPSTREAM_PLAN_INVALID",
                    "upstream task has duplicate or self dependencies",
                )
            if task_id in self.scheduled_ids or task_id in batch_ids:
                raise WorkerFailure("UPSTREAM_PLAN_INVALID", "upstream task IDs are duplicated")
            batch_ids.add(task_id)
            executable_count += int(not is_join)
            normalized.append((task_id, task, is_join, dependency_ids))

        if len(self.tasks) + executable_count > self.max_tasks:
            raise WorkerFailure("TASK_LIMIT_EXCEEDED", "upstream plan exceeded max_tasks")
        known_ids = self.scheduled_ids | batch_ids
        if any(not dependencies <= known_ids for _, _, _, dependencies in normalized):
            raise WorkerFailure(
                "UPSTREAM_PLAN_INVALID",
                "upstream task references an unknown dependency",
            )
        candidate_graph = {
            node: set(dependencies) for node, dependencies in self.dependency_graph.items()
        }
        candidate_graph.update(
            {task_id: set(dependencies) for task_id, _, _, dependencies in normalized}
        )
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(node: int) -> None:
            if node in visiting:
                raise WorkerFailure("UPSTREAM_PLAN_INVALID", "upstream task graph has a cycle")
            if node in visited:
                return
            visiting.add(node)
            for dependency in candidate_graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for task_id in batch_ids:
            visit(task_id)

        self.scheduled_ids.update(batch_ids)
        self.dependency_graph = candidate_graph
        for task_id, task, is_join, _ in normalized:
            if not is_join:
                self.tasks[task_id] = task
                self.emit(task_id, "planned")

    def emit(self, task_id: int, state: str) -> None:
        self.events.append({"index": len(self.events), "task_id": task_id, "state": state})

    def fail(self, task_id: int, exc: BaseException) -> None:
        if self.failure is None:
            self.failure = (
                exc
                if isinstance(exc, WorkerFailure)
                else WorkerFailure(
                    "TOOL_EXECUTION_FAILURE",
                    f"allowlisted tool failed: {type(exc).__name__}",
                    consumed=True,
                )
            )
        self.emit(task_id, "failed")

    def fail_planner(self, exc: BaseException) -> WorkerFailure:
        if self.failure is None:
            self.failure = (
                exc
                if isinstance(exc, WorkerFailure)
                else WorkerFailure(
                    "PLANNER_STREAM_FAILURE",
                    f"upstream streaming planner failed: {type(exc).__name__}",
                )
            )
        return self.failure

    def task_rows(self, specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        properties = {
            str(spec["name"]): self.argument_order.get(str(spec["name"]), tuple(spec["input_schema"].get("properties", {}).keys()))
            if isinstance(spec["input_schema"].get("properties", {}), Mapping)
            else ()
            for spec in specs
        }
        rows: list[dict[str, Any]] = []
        for task_id in sorted(self.tasks):
            task = self.tasks[task_id]
            arguments = list(getattr(task, "args", ()))
            names = properties.get(str(task.name), ())
            if len(arguments) == 1 and isinstance(arguments[0], Mapping):
                mapped_arguments = dict(arguments[0])
            elif len(names) == len(arguments) or (str(task.name) in self.argument_order and len(arguments) <= len(names)):
                mapped_arguments = dict(zip(names[:len(arguments)], arguments, strict=True))
            else:
                mapped_arguments = {"__args__": arguments}
            try:
                _canonical_json(mapped_arguments)
            except (TypeError, ValueError) as exc:
                raise WorkerFailure("UPSTREAM_PLAN_INVALID", "task arguments are not JSON") from exc
            rows.append(
                {
                    "task_id": task_id,
                    "tool_name": str(task.name),
                    "arguments": mapped_arguments,
                    "dependencies": sorted(getattr(task, "dependencies", ())),
                }
            )
        if not rows:
            raise WorkerFailure(
                "EMPTY_EXECUTABLE_PLAN", "plan contains only join/no executable task"
            )
        return rows


def _traced_task_fetching_unit(base: type[Any], recorder: TraceRecorder) -> type[Any]:
    # Upstream creates a new TFU for each replan, but its parser uses global task
    # IDs (and join depends on every preceding ID). Make completed observations
    # available for dependency lookup without scheduling or iterating them again.
    # These maps belong to one _run_full invocation, never to another request.
    completed_tasks: dict[int, Any] = {}
    completed_events: dict[int, asyncio.Event] = {}

    class RoundTasks(dict[int, Any]):
        def __missing__(self, task_id: int) -> Any:
            return completed_tasks[task_id]

    class RoundEvents(dict[int, asyncio.Event]):
        def __missing__(self, task_id: int) -> asyncio.Event:
            return completed_events[task_id]

    class TracedTaskFetchingUnit(base):
        def __init__(self) -> None:
            super().__init__()
            self.tasks = RoundTasks(self.tasks)
            self.tasks_done = RoundEvents(self.tasks_done)
            self._agentinputlab_semaphore = asyncio.Semaphore(recorder.concurrency)

        def set_tasks(self, tasks: Mapping[Any, Any]) -> None:
            recorder.register_batch(tasks)
            super().set_tasks(tasks)

        async def _run_task(self, task: Any) -> None:
            current = asyncio.current_task()
            if current is not None:
                recorder.active_tasks.add(current)
            if bool(getattr(task, "is_join", False)):
                await super()._run_task(task)
                completed_tasks[task.idx] = task
                completed_events[task.idx] = self.tasks_done[task.idx]
                return
            task_id = int(task.idx)
            recorder.emit(task_id, "ready")
            async with self._agentinputlab_semaphore:
                if recorder.failure is not None:
                    return
                recorder.emit(task_id, "running")
                try:
                    await super()._run_task(task)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    recorder.fail(task_id, exc)
                    for event in tuple(self.tasks_done.values()):
                        event.set()
                    self.remaining_tasks.clear()
                    return
                recorder.emit(task_id, "completed")
                completed_tasks[task_id] = task
                completed_events[task_id] = self.tasks_done[task_id]

    TracedTaskFetchingUnit.__name__ = "AgentInputLabTracedTaskFetchingUnit"
    return TracedTaskFetchingUnit


async def _cancel_and_await(tasks: Sequence[asyncio.Task[Any]] | set[asyncio.Task[Any]]) -> None:
    current = asyncio.current_task()
    selected = tuple(dict.fromkeys(task for task in tasks if task is not current))
    for task in selected:
        if not task.done():
            task.cancel()
    if selected:
        await asyncio.gather(*selected, return_exceptions=True)


async def _settle_planner_tasks(recorder: TraceRecorder) -> None:
    tasks = tuple(task for task in recorder.planner_tasks if task is not asyncio.current_task())
    if not tasks:
        return
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            recorder.fail_planner(
                WorkerFailure("PLANNER_STREAM_FAILURE", "upstream streaming planner was cancelled")
            )
        elif isinstance(result, BaseException):
            recorder.fail_planner(result)


def _planner(
    upstream: LoadedUpstream,
    model: _ReplayModel,
    prompts: Mapping[str, Any],
    tools: Sequence[Any],
) -> Any:
    planner_type = getattr(upstream.planner, "Planner", None)
    if planner_type is None:
        raise WorkerFailure("UPSTREAM_API_MISMATCH", "upstream Planner is unavailable")
    return planner_type(
        llm=model,
        example_prompt=prompts["planner_example_prompt"],
        example_prompt_replan=prompts["planner_example_prompt_replan"],
        tools=tools,
        stop=list(prompts["planner_stop"]),
    )


async def _run_planner_component(
    request: ValidatedRequest,
    upstream: LoadedUpstream,
    model: _ReplayModel,
    prompts: Mapping[str, Any],
    tools: Sequence[Any],
) -> tuple[list[Any], list[dict[str, Any]], int, str]:
    planner = _planner(upstream, model, prompts, tools)
    inputs = {"input": request.query}
    recorder = TraceRecorder(
        allowed_tools=frozenset(str(spec["name"]) for spec in request.tool_specs),
        max_tasks=request.controls.max_tasks,
        concurrency=request.controls.concurrency,
        argument_order=request.tool_argument_order,
    )
    if request.controls.streaming:
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def produce() -> None:
            try:
                await planner.aplan(inputs=inputs, task_queue=queue, is_replan=False)
            except asyncio.CancelledError:
                raise
            except WorkerFailure:
                raise
            except Exception as exc:
                raise WorkerFailure(
                    "PLANNER_STREAM_FAILURE",
                    f"upstream streaming planner failed: {type(exc).__name__}",
                ) from exc
            finally:
                queue.put_nowait(None)

        planner_task = asyncio.create_task(produce())
        recorder.planner_tasks.add(planner_task)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                task_id = getattr(item, "idx", None)
                recorder.register_batch({task_id: item})
            await _settle_planner_tasks(recorder)
            if recorder.failure is not None:
                raise recorder.failure
        finally:
            await _cancel_and_await(recorder.planner_tasks)
    else:
        planned = await planner.plan(inputs=inputs, is_replan=False)
        if not isinstance(planned, Mapping):
            raise WorkerFailure("UPSTREAM_PLAN_INVALID", "Planner.plan did not return a task map")
        recorder.register_batch(planned)
    rows = recorder.task_rows(request.tool_specs)
    return rows, [], 0, ""


async def _run_full(
    request: ValidatedRequest,
    upstream: LoadedUpstream,
    model: _ReplayModel,
    prompts: Mapping[str, Any],
    tools: Sequence[Any],
) -> tuple[list[Any], list[dict[str, Any]], int, str]:
    compiler_type = getattr(upstream.compiler, "LLMCompiler", None)
    base_tfu = getattr(upstream.task_fetching, "TaskFetchingUnit", None)
    if compiler_type is None or base_tfu is None:
        raise WorkerFailure("UPSTREAM_API_MISMATCH", "upstream full runtime is unavailable")
    recorder = TraceRecorder(
        allowed_tools=frozenset(str(spec["name"]) for spec in request.tool_specs),
        max_tasks=request.controls.max_tasks,
        concurrency=request.controls.concurrency,
        argument_order=request.tool_argument_order,
    )
    traced_tfu = _traced_task_fetching_unit(base_tfu, recorder)
    original_tfu = getattr(upstream.compiler, "TaskFetchingUnit", None)
    upstream.compiler.TaskFetchingUnit = traced_tfu
    try:
        compiler = compiler_type(
            tools=tools,
            planner_llm=model,
            planner_example_prompt=prompts["planner_example_prompt"],
            planner_example_prompt_replan=prompts["planner_example_prompt_replan"],
            planner_stop=list(prompts["planner_stop"]),
            planner_stream=request.controls.streaming,
            agent_llm=model,
            joinner_prompt=prompts["joinner_prompt"],
            joinner_prompt_final=prompts["joinner_prompt_final"],
            max_replans=request.controls.max_replans + 1,
            benchmark=False,
        )
        original_aplan = compiler.planner.aplan

        async def guarded_aplan(*args: Any, **kwargs: Any) -> Any:
            current = asyncio.current_task()
            if current is not None:
                recorder.planner_tasks.add(current)
            task_queue = kwargs.get("task_queue")
            if task_queue is None and len(args) > 1:
                task_queue = args[1]
            try:
                return await original_aplan(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise recorder.fail_planner(exc) from exc
            finally:
                put_nowait = getattr(task_queue, "put_nowait", None)
                if callable(put_nowait):
                    put_nowait(None)

        compiler.planner.aplan = guarded_aplan
        original_join = compiler.join

        async def guarded_join(*args: Any, **kwargs: Any) -> Any:
            await _settle_planner_tasks(recorder)
            if recorder.failure is not None:
                raise recorder.failure
            return await original_join(*args, **kwargs)

        object.__setattr__(compiler, "join", guarded_join)
        result = await compiler._acall({"input": request.query})
    finally:
        upstream.compiler.TaskFetchingUnit = original_tfu
        await _cancel_and_await(recorder.planner_tasks | recorder.active_tasks)
    if recorder.failure is not None:
        raise recorder.failure
    if not isinstance(result, Mapping) or not isinstance(result.get("output"), str):
        raise WorkerFailure("UPSTREAM_OUTPUT_INVALID", "LLMCompiler returned an invalid output")
    answer = result["output"].strip()
    if not answer:
        raise WorkerFailure("UPSTREAM_OUTPUT_INVALID", "LLMCompiler returned an empty answer")
    replans = max(0, len(model.planner_responses) - 1)
    if replans > request.controls.max_replans:
        raise WorkerFailure("REPLAN_LIMIT_EXCEEDED", "LLMCompiler exceeded max_replans")
    return recorder.task_rows(request.tool_specs), recorder.events, replans, answer


async def execute_request(
    request: ValidatedRequest,
    upstream: LoadedUpstream,
    model_bundle: ReplayBundle,
    tool_replay: ToolReplay,
    *,
    runtime_model: _ReplayModel | None = None,
) -> Mapping[str, Any]:
    runtime: dict[str, _ReplayModel] = {}

    async def run() -> Mapping[str, Any]:
        _patch_model_type_gates(upstream)
        model = runtime_model if runtime_model is not None else _ReplayModel(model_bundle.entries)
        runtime["model"] = model
        tools = _build_tools(upstream, request.tool_specs, tool_replay)
        discarded = _DiscardTextIO()
        try:
            with contextlib.redirect_stdout(discarded):
                if request.execution_mode == "planner_component":
                    tasks, events, replans, final_answer = await _run_planner_component(
                        request, upstream, model, model_bundle.prompts, tools
                    )
                    termination = "planner_complete"
                else:
                    tasks, events, replans, final_answer = await _run_full(
                        request, upstream, model, model_bundle.prompts, tools
                    )
                    termination = "finish"
            model.assert_exhausted()
        except WorkerFailure as exc:
            if model.consumed or tool_replay.calls:
                exc.usage = model.usage(tool_calls=tool_replay.calls)
                exc.consumed = True
            raise
        except Exception as exc:
            raise WorkerFailure(
                "UPSTREAM_RUNTIME_FAILURE",
                f"pinned LLMCompiler runtime failed: {type(exc).__name__}",
                consumed=bool(model.consumed or tool_replay.calls),
                usage=model.usage(tool_calls=tool_replay.calls),
            ) from exc
        raw_plan_hash = _canonical_hash(model.planner_responses)
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL_VERSION,
            "payload": {
                "tasks": tasks,
                "scheduler_events": events,
                "replans": replans,
                "streaming": request.controls.streaming,
                "raw_plan_hash": raw_plan_hash,
                "termination": termination,
                "final_answer": final_answer,
            },
            "usage": model.usage(tool_calls=tool_replay.calls),
        }

    try:
        return await asyncio.wait_for(run(), timeout=request.controls.timeout_seconds)
    # Python 3.10 uses a distinct asyncio timeout exception; 3.11+ aliases it.
    except asyncio.TimeoutError as exc:
        model = runtime.get("model")
        usage = (
            model.usage(tool_calls=tool_replay.calls)
            if model is not None
            else {
                "model_calls": 0,
                "retrieval_calls": 0,
                "tool_calls": tool_replay.calls,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        raise WorkerFailure(
            "RUN_TIMEOUT",
            "LLMCompiler exceeded controls.timeout_seconds",
            consumed=bool(usage["model_calls"] or usage["tool_calls"]),
            usage=usage,
        ) from exc


def _failure_response(exc: WorkerFailure, *, request_id: str = "") -> Mapping[str, Any]:
    response: dict[str, Any] = {
        "status": "error",
        "failure_code": exc.code,
        "message": exc.public_message,
        "consumed": exc.consumed,
    }
    if request_id:
        response["request_id"] = request_id
    if exc.usage is not None:
        response["usage"] = exc.usage
    return response


def process_request(
    raw_request: Any,
    *,
    checkout: Path,
    model_replay_path: Path,
    tool_replay_path: Path,
    upstream: LoadedUpstream | None = None,
) -> Mapping[str, Any]:
    verify_checkout(checkout)
    request = validate_request(raw_request)
    model_bundle = load_model_replay(model_replay_path, request)
    tool_replay = load_tool_replay(tool_replay_path, request)
    loaded = upstream if upstream is not None else import_upstream(checkout)
    return asyncio.run(execute_request(request, loaded, model_bundle, tool_replay))


def _read_request_line() -> Any:
    raw = sys.stdin.buffer.readline(_MAX_MANIFEST_BYTES + 1)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise WorkerFailure("REQUEST_TOO_LARGE", "worker request exceeds the byte limit")
    if not raw:
        raise WorkerFailure("SCHEMA_INVALID", "worker requires one JSON line")
    if sys.stdin.buffer.read(1):
        raise WorkerFailure("SCHEMA_INVALID", "worker accepts exactly one JSON line")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerFailure("SCHEMA_INVALID", "worker request is not valid UTF-8 JSON") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--model-replay", type=Path)
    parser.add_argument("--tool-replay", type=Path)
    parser.add_argument("--smoke-import", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_id = ""
    try:
        verify_checkout(args.checkout)
        if args.smoke_import:
            import_upstream(args.checkout)
            response: Mapping[str, Any] = {
                "status": "ok",
                "upstream_commit": UPSTREAM_COMMIT,
                "protocol_version": PROTOCOL_VERSION,
                "smoke_import": True,
            }
        else:
            if args.model_replay is None or args.tool_replay is None:
                raise WorkerFailure(
                    "CONFIGURATION_MISSING",
                    "normal execution requires --model-replay and --tool-replay",
                )
            raw_request = _read_request_line()
            if isinstance(raw_request, Mapping) and isinstance(raw_request.get("request_id"), str):
                request_id = str(raw_request["request_id"])
            request = validate_request(raw_request)
            model_bundle = load_model_replay(args.model_replay, request)
            tool_replay = load_tool_replay(args.tool_replay, request)
            upstream = import_upstream(args.checkout)
            response = asyncio.run(execute_request(request, upstream, model_bundle, tool_replay))
    except WorkerFailure as exc:
        response = _failure_response(exc, request_id=request_id)
    except Exception as exc:  # pragma: no cover - last-resort JSONL containment
        response = _failure_response(
            WorkerFailure(
                "WORKER_INTERNAL_FAILURE",
                f"worker failed closed: {type(exc).__name__}",
            ),
            request_id=request_id,
        )
    sys.stdout.write(_canonical_json(response) + "\n")
    sys.stdout.flush()
    return 0 if response.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
