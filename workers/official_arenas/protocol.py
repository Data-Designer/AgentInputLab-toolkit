"""Shared, dependency-light contracts for official arena providers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

REPLAY_INVENTORY_PROTOCOL = "agentinputlab.official-arena-replay-inventory.v1"
EXECUTABLE_PAYLOAD_PROTOCOL = "agentinputlab.executable-release-payload.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHARED_BUDGET_PATH_ENV = "AILAB_SHARED_BUDGET_OVERRIDE_PATH"
SHARED_BUDGET_SHA256_ENV = "AILAB_SHARED_BUDGET_OVERRIDE_SHA256"
_BUDGET_FIELDS = frozenset(
    {
        "profile_id",
        "max_model_calls",
        "max_retrieval_calls",
        "max_tool_calls",
        "max_output_tokens",
        "max_context_tokens",
        "max_output_tokens_per_call",
    }
)
_STANDARD_SHARED_BUDGET = {
    "max_model_calls": 8,
    "max_retrieval_calls": 8,
    "max_tool_calls": 8,
    "max_output_tokens": 4096,
    # ExecutionBudget.max_context_tokens is a cumulative input-token budget,
    # not the per-request model context window.  Eight calls may each carry a
    # growing interactive history, so the frozen run-level cap is eight times
    # the 32,768-token per-call context control.
    "max_context_tokens": 262144,
    "max_output_tokens_per_call": 512,
}


class OfficialArenaError(RuntimeError):
    """Stable fail-closed error raised by an official runtime provider."""


@dataclass(frozen=True, slots=True)
class ProviderArmResult:
    """One completed official episode plus scorer and usage evidence."""

    metrics: Mapping[str, float]
    trace: Mapping[str, Any]
    usage: Mapping[str, int | float] = field(default_factory=dict)
    runtime_instance_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise OfficialArenaError("official provider metrics must be non-empty")
        for name, value in self.metrics.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise OfficialArenaError("official provider metrics are invalid")
        if not isinstance(self.trace, Mapping) or not self.trace:
            raise OfficialArenaError("official provider trace must be non-empty")
        if not isinstance(self.usage, Mapping):
            raise OfficialArenaError("official provider usage must be an object")
        if not isinstance(self.runtime_instance_id, str) or len(self.runtime_instance_id) != 64:
            raise OfficialArenaError(
                "official provider runtime_instance_id must be a SHA-256 digest"
            )


@runtime_checkable
class OfficialArenaProvider(Protocol):
    """Runtime-specific capture and continuation boundary."""

    benchmark_id: str
    source_revision: str
    scorer_id: str
    scorer_revision: str

    def doctor(self, config: Mapping[str, Any]) -> tuple[str, ...]: ...

    def capture(
        self,
        config: Mapping[str, Any],
        *,
        output_root: Path,
    ) -> tuple[Mapping[str, Any], ...]: ...

    def run_arm(
        self,
        config: Mapping[str, Any],
        *,
        replay_record: Mapping[str, Any],
        arm_payload: Mapping[str, Any],
        seed: int,
        role: str,
        work_dir: Path,
    ) -> ProviderArmResult: ...


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_shared_budget(value: Any) -> Mapping[str, int | str]:
    """Validate one paper-wide resource budget shared by every arena arm."""

    if not isinstance(value, Mapping) or set(value) != _BUDGET_FIELDS:
        raise OfficialArenaError("shared budget override fields are invalid")
    if not isinstance(value["profile_id"], str) or not value["profile_id"].strip():
        raise OfficialArenaError("shared budget profile_id must be non-empty")
    for name in _BUDGET_FIELDS - {"profile_id"}:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise OfficialArenaError(f"shared budget {name} must be a positive integer")
    if value["max_output_tokens_per_call"] > value["max_output_tokens"]:
        raise OfficialArenaError(
            "shared budget per-call output cap exceeds the cumulative output cap"
        )
    return value


def standard_shared_budget(multiplier: float = 1.0) -> dict[str, int | str]:
    """Materialize the frozen paper control profile at one resource multiplier."""

    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not math.isfinite(float(multiplier))
        or multiplier <= 0
    ):
        raise OfficialArenaError("shared budget multiplier must be finite and positive")
    return {
        "profile_id": f"paper-shared-budget-x{multiplier:g}",
        **{
            name: max(1, round(value * multiplier))
            for name, value in _STANDARD_SHARED_BUDGET.items()
        },
    }


def resolve_shared_budget_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Attach an authenticated process-level budget factor to a provider config.

    The factor is read by every official provider before capture.  Consequently
    the exact same budget becomes part of the captured visible state, the
    intervention input hash, and the fair-comparison dependency locks.
    """

    raw_path = os.environ.get(SHARED_BUDGET_PATH_ENV)
    raw_sha256 = os.environ.get(SHARED_BUDGET_SHA256_ENV)
    if not raw_path and not raw_sha256:
        return config
    if not raw_path or not raw_sha256 or _SHA256.fullmatch(raw_sha256) is None:
        raise OfficialArenaError("shared budget environment binding is incomplete")
    unresolved = Path(raw_path).expanduser()
    path = unresolved.resolve()
    if unresolved.is_symlink() or not path.is_file() or sha256_file(path) != raw_sha256:
        raise OfficialArenaError("shared budget environment artifact differs")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialArenaError("shared budget artifact is invalid JSON") from exc
    override = validate_shared_budget(loaded)
    existing = config.get("budget_override")
    if existing is not None and existing != override:
        raise OfficialArenaError("explicit and environment budget overrides differ")
    return {**config, "budget_override": dict(override)}


def visible_shared_budget(config: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact budget exposed to intervention code at capture time."""

    raw = config.get("budget_override")
    if raw is None:
        return dict(fallback)
    override = validate_shared_budget(raw)
    return {
        "max_model_calls": override["max_model_calls"],
        "max_retrieval_calls": override["max_retrieval_calls"],
        "max_tool_calls": override["max_tool_calls"],
        "max_output_tokens": override["max_output_tokens"],
        "max_context_tokens": override["max_context_tokens"],
        "max_cost_usd": None,
    }


def require_materialized_budget(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Fail closed if a materialized arm changed the shared resource budget."""

    raw = config.get("budget_override")
    budget = state.get("budget")
    if not isinstance(budget, Mapping):
        raise OfficialArenaError("materialized state has no execution budget")
    if raw is None:
        return budget
    expected = visible_shared_budget(config, {})
    if dict(budget) != expected:
        raise OfficialArenaError("materialized arm changed the shared budget factor")
    return budget


def native_extra_body(config: Mapping[str, Any], field: str, *,
                      defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Copy explicit transport extensions without overriding native task/budget fields."""
    from agentinputlab.contracts.serialization import to_jsonable

    mode = config.get(field + "_mode", "merge")
    if not isinstance(mode, str) or mode not in {"merge", "replace"}:
        raise OfficialArenaError(f"{field}_mode must be merge or replace")
    value = config.get(field, {})
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OfficialArenaError(f"{field} must be a JSON object")
    reserved = {"model", "messages", "input", "instructions", "tools", "tool_choice",
                "parallel_tool_calls", "temperature", "seed", "n", "stream", "stop",
                "max_tokens", "max_completion_tokens", "max_output_tokens", "max_length",
                "response_format", "text", "reasoning", "previous_response_id", "truncation"}
    conflicts = reserved.intersection(value)
    if conflicts:
        raise OfficialArenaError(f"{field} must not override native request fields: "
                                 + ", ".join(sorted(conflicts)))
    try:
        # No references to caller-owned nested dictionaries survive this boundary.
        inherited = (defaults or {}) if mode == "merge" else {}
        return json.loads(json.dumps(to_jsonable({**inherited, **value}), allow_nan=False))
    except (TypeError, ValueError, RecursionError):
        raise OfficialArenaError(f"{field} must contain finite JSON values") from None


def native_timeout_options(config: Mapping[str, Any], field: str) -> dict[str, float]:
    """Bind an explicit per-request timeout without changing legacy defaults."""
    if field not in config:
        return {}
    value = config[field]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise OfficialArenaError(f"{field} must be a finite positive number")
    return {"timeout": float(value)}


def validate_dependency_hashes(
    value: Any,
    *,
    label: str,
) -> Mapping[str, str]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            for name, digest in value.items()
        )
    ):
        raise OfficialArenaError(f"{label} dependency locks are invalid")
    return value


def materialized_state(arm_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        not isinstance(arm_payload, Mapping)
        or arm_payload.get("schema_version") != 1
        or arm_payload.get("protocol") != EXECUTABLE_PAYLOAD_PROTOCOL
    ):
        raise OfficialArenaError(
            "official arena requires an executable materialized release payload"
        )
    execution = arm_payload.get("intervention_execution")
    if not isinstance(execution, Mapping):
        raise OfficialArenaError("intervention_execution is missing")
    state = execution.get("materialized_state")
    if not isinstance(state, Mapping):
        raise OfficialArenaError("materialized intervention state is missing")
    if content_hash(state) != execution.get("output_state_hash"):
        raise OfficialArenaError("materialized intervention state hash differs")
    if execution.get("input_state_hash") != arm_payload.get("visible_state_hash"):
        raise OfficialArenaError("intervention input state differs from the checkpoint")
    return state


def intervention_usage(arm_payload: Mapping[str, Any]) -> Mapping[str, int | float]:
    execution = arm_payload.get("intervention_execution")
    if not isinstance(execution, Mapping) or not isinstance(execution.get("usage"), Mapping):
        raise OfficialArenaError("intervention usage is missing")
    return execution["usage"]


def merge_usage(*values: Mapping[str, Any]) -> dict[str, int | float]:
    integer_fields = (
        "model_calls",
        "retrieval_calls",
        "tool_calls",
        "embedding_calls",
        "unpriced_model_calls",
        "input_tokens",
        "output_tokens",
    )
    float_fields = (
        "cpu_seconds",
        "accelerator_seconds",
        "cost_usd",
        "compute_cost_usd",
    )
    result: dict[str, int | float] = {name: 0 for name in integer_fields}
    result.update({name: 0.0 for name in float_fields})
    for value in values:
        if not isinstance(value, Mapping):
            raise OfficialArenaError("usage contribution must be an object")
        for name in integer_fields:
            raw = value.get(name, 0)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise OfficialArenaError(f"usage field {name} must be non-negative integer")
            result[name] += raw
        for name in float_fields:
            raw = value.get(name, 0.0)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or raw < 0
            ):
                raise OfficialArenaError(f"usage field {name} must be non-negative finite")
            result[name] += float(raw)
    return result


__all__ = [
    "EXECUTABLE_PAYLOAD_PROTOCOL",
    "REPLAY_INVENTORY_PROTOCOL",
    "OfficialArenaError",
    "OfficialArenaProvider",
    "ProviderArmResult",
    "canonical_json",
    "content_hash",
    "intervention_usage",
    "materialized_state",
    "merge_usage",
    "sha256_file",
    "validate_dependency_hashes",
]
