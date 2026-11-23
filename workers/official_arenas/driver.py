#!/usr/bin/env python3
"""Strict atomic worker for official stateful benchmark runtimes.

The host sends one baseline/treatment request.  This worker loads a
checksum-locked protected replay inventory, starts two distinct official
runtime instances, injects the already materialized intervention state, calls
the upstream scorer, and returns one indivisible receipt.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .protocol import (
        REPLAY_INVENTORY_PROTOCOL,
        OfficialArenaError,
        OfficialArenaProvider,
        content_hash,
        intervention_usage,
        materialized_state,
        merge_usage,
        sha256_file,
    )
except ImportError:  # direct script execution
    from protocol import (  # type: ignore[no-redef]
        REPLAY_INVENTORY_PROTOCOL,
        OfficialArenaError,
        OfficialArenaProvider,
        content_hash,
        intervention_usage,
        materialized_state,
        merge_usage,
        sha256_file,
    )

REQUEST_PROTOCOL = "agentinputlab.atomic-process-request.v1"
RESPONSE_PROTOCOL = "agentinputlab.atomic-process-response.v1"
WORKER_PROTOCOL = "agentinputlab.official-arena-worker.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "atomic_request",
        "worker_parameters",
        "request_hash",
    }
)
_PARAMETER_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "provider_entrypoint",
        "provider_config",
        "replay_inventory_environment",
        "replay_inventory_sha256_environment",
        "expected_replay_inventory_sha256",
        "bridge_id",
        "bridge_revision",
        "scorer_id",
        "scorer_revision",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "benchmark_id",
        "checkpoint_id",
        "source_revision",
        "origin_state_hash",
        "replay_identity",
        "seed",
        "locked_dependency_hashes",
        "arms",
    }
)
_USAGE_FIELDS = (
    "model_calls",
    "retrieval_calls",
    "tool_calls",
    "embedding_calls",
    "unpriced_model_calls",
    "input_tokens",
    "output_tokens",
    "cpu_seconds",
    "accelerator_seconds",
    "cost_usd",
    "compute_cost_usd",
)


def _provider(entrypoint: str) -> OfficialArenaProvider:
    if not isinstance(entrypoint, str) or ":" not in entrypoint or entrypoint.startswith("_"):
        raise OfficialArenaError("provider_entrypoint must use module:factory syntax")
    module_name, attribute = entrypoint.split(":", 1)
    if not module_name.startswith("workers.official_arenas.providers."):
        raise OfficialArenaError("provider_entrypoint is outside the official provider allowlist")
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        provider = factory()
    except Exception as exc:
        raise OfficialArenaError("cannot load official arena provider") from exc
    if not isinstance(provider, OfficialArenaProvider):
        raise OfficialArenaError("official arena provider does not implement the protocol")
    return provider


def _parameters(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PARAMETER_FIELDS:
        raise OfficialArenaError("official arena worker parameters are invalid")
    if value["schema_version"] != 1 or value["protocol"] != WORKER_PROTOCOL:
        raise OfficialArenaError("unsupported official arena worker protocol")
    for name in (
        "expected_replay_inventory_sha256",
        "bridge_revision",
        "scorer_revision",
    ):
        if not isinstance(value[name], str) or _SHA256.fullmatch(value[name]) is None:
            raise OfficialArenaError(f"worker parameter {name} must be a SHA-256 digest")
    for name in (
        "replay_inventory_environment",
        "replay_inventory_sha256_environment",
    ):
        if not isinstance(value[name], str) or _ENV.fullmatch(value[name]) is None:
            raise OfficialArenaError(f"worker parameter {name} is invalid")
    if not isinstance(value["provider_config"], Mapping):
        raise OfficialArenaError("provider_config must be an object")
    for name in ("bridge_id", "scorer_id"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise OfficialArenaError(f"worker parameter {name} must be non-empty")
    return value


def _inventory(parameters: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    path_text = os.environ.get(parameters["replay_inventory_environment"])
    digest_text = os.environ.get(parameters["replay_inventory_sha256_environment"])
    expected = parameters["expected_replay_inventory_sha256"]
    if not path_text or digest_text != expected:
        raise OfficialArenaError("protected replay inventory environment is absent or changed")
    path = Path(path_text).expanduser()
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise OfficialArenaError("protected replay inventory is absent or changed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialArenaError("protected replay inventory is not valid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema_version",
            "protocol",
            "benchmark_id",
            "source_revision",
            "record_count",
            "records",
        }
        or value["schema_version"] != 1
        or value["protocol"] != REPLAY_INVENTORY_PROTOCOL
        or not isinstance(value["records"], list)
        or value["record_count"] != len(value["records"])
        or not value["records"]
    ):
        raise OfficialArenaError("protected replay inventory schema is invalid")
    records: dict[str, Mapping[str, Any]] = {}
    for item in value["records"]:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "checkpoint_id",
                "origin_state_hash",
                "replay_identity",
                "protected",
            }
            or not isinstance(item["checkpoint_id"], str)
            or not isinstance(item["protected"], Mapping)
        ):
            raise OfficialArenaError("protected replay record schema is invalid")
        if item["checkpoint_id"] in records:
            raise OfficialArenaError("protected replay inventory contains duplicate checkpoints")
        for name in ("origin_state_hash", "replay_identity"):
            if not isinstance(item[name], str) or _SHA256.fullmatch(item[name]) is None:
                raise OfficialArenaError("protected replay record identity is invalid")
        records[item["checkpoint_id"]] = item
    return {
        "__inventory__": {
            "benchmark_id": value["benchmark_id"],
            "source_revision": value["source_revision"],
        },
        **records,
    }


def _atomic_request(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
        raise OfficialArenaError("atomic request schema is invalid")
    if (
        not isinstance(value["seed"], int)
        or isinstance(value["seed"], bool)
        or value["seed"] < 0
        or not isinstance(value["locked_dependency_hashes"], Mapping)
    ):
        raise OfficialArenaError("atomic request seed or dependency locks are invalid")
    for name in ("origin_state_hash", "replay_identity"):
        if not isinstance(value[name], str) or _SHA256.fullmatch(value[name]) is None:
            raise OfficialArenaError(f"atomic request {name} is invalid")
    arms = value["arms"]
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence) or len(arms) != 2:
        raise OfficialArenaError("atomic request requires two arms")
    roles: set[str] = set()
    arm_ids: set[str] = set()
    for arm in arms:
        if (
            not isinstance(arm, Mapping)
            or set(arm) != {"role", "arm_id", "payload"}
            or arm["role"] not in {"baseline", "treatment"}
            or not isinstance(arm["arm_id"], str)
            or not isinstance(arm["payload"], Mapping)
        ):
            raise OfficialArenaError("atomic request arm schema is invalid")
        roles.add(arm["role"])
        arm_ids.add(arm["arm_id"])
    if roles != {"baseline", "treatment"} or len(arm_ids) != 2:
        raise OfficialArenaError("atomic request arms must be distinct baseline and treatment")
    return value


def _validate_dependency_binding(
    *,
    request: Mapping[str, Any],
    parameters: Mapping[str, Any],
    replay_record: Mapping[str, Any],
) -> None:
    provider_dependencies = parameters["provider_config"].get("locked_dependency_hashes")
    protected = replay_record.get("protected")
    protected_dependencies = (
        protected.get("locked_dependency_hashes") if isinstance(protected, Mapping) else None
    )
    if (
        not isinstance(provider_dependencies, Mapping)
        or not provider_dependencies
        or not isinstance(protected_dependencies, Mapping)
        or not protected_dependencies
    ):
        raise OfficialArenaError(
            "provider and protected replay must bind non-empty dependency hashes"
        )
    if dict(provider_dependencies) != dict(protected_dependencies):
        raise OfficialArenaError("provider and protected replay dependency locks differ")
    request_dependencies = request["locked_dependency_hashes"]
    for name, digest in provider_dependencies.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or request_dependencies.get(name) != digest
        ):
            raise OfficialArenaError(
                "provider dependency locks are not a matching subset of the arm locks"
            )


def _validate_shared_budget_usage(
    arm_payload: Mapping[str, Any],
    usage: Mapping[str, int | float],
) -> None:
    """Reject any completed arm that escaped its materialized cumulative caps."""

    state = materialized_state(arm_payload)
    budget = state.get("budget")
    if not isinstance(budget, Mapping):
        raise OfficialArenaError("materialized arm has no budget for usage validation")
    caps = {
        "model_calls": "max_model_calls",
        "retrieval_calls": "max_retrieval_calls",
        "tool_calls": "max_tool_calls",
        "output_tokens": "max_output_tokens",
        "input_tokens": "max_context_tokens",
    }
    for usage_name, budget_name in caps.items():
        cap = budget.get(budget_name)
        if cap is None:
            continue
        if (
            isinstance(cap, bool)
            or not isinstance(cap, int)
            or cap < 1
            or float(usage[usage_name]) > cap
        ):
            raise OfficialArenaError(
                "official arm exceeded materialized "
                f"{budget_name}: actual={usage[usage_name]} cap={cap}"
            )


def _intervention_failure(arm_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    execution = arm_payload.get("intervention_execution")
    if not isinstance(execution, Mapping):
        raise OfficialArenaError("intervention_execution is missing")
    status = execution.get("status")
    failure = execution.get("failure")
    if status == "failed":
        required = {"scope", "category", "exception_type", "message", "monitor_run_id"}
        if (
            not isinstance(failure, Mapping)
            or not required <= set(failure)
            or failure.get("scope") != "algorithm"
            or any(not isinstance(failure.get(name), str) for name in required)
        ):
            raise OfficialArenaError("failed intervention has no typed algorithm failure")
        return dict(failure)
    if failure is not None:
        raise OfficialArenaError("successful intervention cannot carry a failure receipt")
    return None


def _tool_call_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("tool_calls")
    names: list[str] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for call in raw:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else call.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    sender = str(row.get("sender", "")).lower()
    recipient = str(row.get("recipient", "")).lower()
    content = row.get("content")
    if (
        sender.endswith("agent")
        and recipient.endswith("execution_environment")
        and isinstance(content, str)
    ):
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\(", content)
        if match is not None:
            names.append(match.group(1))
    return tuple(names)


def _potentially_irreversible(name: str) -> bool:
    """Conservatively separate read-only retrieval from environment mutation."""

    normalized = name.casefold().replace("-", "_").replace(".", "_")
    read_only_prefixes = (
        "get_",
        "list_",
        "search_",
        "retrieve_",
        "find_",
        "lookup_",
        "read_",
        "check_",
        "inspect_",
        "view_",
        "calculate_",
        "compute_",
    )
    return not normalized.startswith(read_only_prefixes)


def _is_environment_result(row: Mapping[str, Any]) -> bool:
    role = str(row.get("role", "")).lower()
    sender = str(row.get("sender", "")).lower()
    return role.endswith("tool") or sender.endswith("execution_environment")


def _result_succeeded(row: Mapping[str, Any]) -> bool:
    content = row.get("content", row)
    text = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).casefold()
    failure_markers = (
        '"success": false',
        '"ok": false',
        "traceback",
        "exception",
        "tool error",
        "execution failed",
    )
    return not any(marker in text for marker in failure_markers)


def _safe_trace_measurement(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return auditable trajectory signals without exporting protected traces."""

    provider = str(trace.get("provider", "unknown"))
    raw_messages = trace.get("messages", trace.get("conversation", ()))
    if provider == "toolsandbox":
        ending = trace.get("ending_context")
        databases = ending.get("_dbs") if isinstance(ending, Mapping) else None
        raw_messages = databases.get("SANDBOX", ()) if isinstance(databases, Mapping) else ()
    messages = (
        tuple(row for row in raw_messages if isinstance(row, Mapping))
        if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes))
        else ()
    )

    def is_assistant(row: Mapping[str, Any]) -> bool:
        role = str(row.get("role", row.get("sender", ""))).lower()
        return role.endswith("assistant") or role.endswith("agent")

    assistant = tuple(row for row in messages if is_assistant(row))
    first = assistant[0] if assistant else None
    content = first.get("content") if isinstance(first, Mapping) else None
    tool_calls = first.get("tool_calls") if isinstance(first, Mapping) else None
    first_has_content = isinstance(content, str) and bool(content.strip())
    first_tool_count = (
        len(tool_calls)
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes))
        else 0
    )
    controls = trace.get("generation_controls")
    hard_stops = controls.get("hard_stops", 0) if isinstance(controls, Mapping) else 0
    termination = trace.get("termination_reason")
    irreversible_index: int | None = None
    irreversible_name: str | None = None
    for index, row in enumerate(messages):
        names = _tool_call_names(row)
        selected = next((name for name in names if _potentially_irreversible(name)), None)
        if selected is not None:
            irreversible_index = index
            irreversible_name = selected
            break
    irreversible_succeeded = False
    if irreversible_index is not None:
        result_row = next(
            (row for row in messages[irreversible_index + 1 :] if _is_environment_result(row)),
            None,
        )
        irreversible_succeeded = result_row is not None and _result_succeeded(result_row)
    return {
        "measurement_protocol": "agentinputlab.safe-trajectory-structure.v2",
        "irreversible_action_rule": "conservative-read-prefix-v1",
        "provider": provider,
        "message_count": len(messages),
        "assistant_turn_count": len(assistant),
        "first_action_present": bool(first_has_content or first_tool_count),
        "first_action_tool_calls": first_tool_count,
        "first_irreversible_action_present": irreversible_index is not None,
        "first_irreversible_action_succeeded": irreversible_succeeded,
        "first_irreversible_action_message_index": irreversible_index,
        "first_irreversible_action_name_hash": (
            content_hash({"tool_name": irreversible_name})
            if irreversible_name is not None
            else None
        ),
        "structurally_completed": bool(assistant) and int(hard_stops or 0) == 0,
        "hard_stops": int(hard_stops or 0),
        "termination_reason": str(termination) if termination is not None else "not_exposed",
        "protected_trace_exported": False,
    }


def run(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROOT_FIELDS:
        raise OfficialArenaError("official arena request fields are invalid")
    if value["schema_version"] != 1 or value["protocol"] != REQUEST_PROTOCOL:
        raise OfficialArenaError("unsupported atomic process request protocol")
    base = {
        "schema_version": value["schema_version"],
        "protocol": value["protocol"],
        "atomic_request": value["atomic_request"],
        "worker_parameters": value["worker_parameters"],
    }
    if value["request_hash"] != content_hash(base):
        raise OfficialArenaError("atomic process request hash differs")
    parameters = _parameters(value["worker_parameters"])
    request = _atomic_request(value["atomic_request"])
    records = _inventory(parameters)
    inventory_identity = records["__inventory__"]
    if (
        inventory_identity["benchmark_id"] != request["benchmark_id"]
        or inventory_identity["source_revision"] != request["source_revision"]
    ):
        raise OfficialArenaError("protected inventory identity differs from atomic request")
    try:
        replay_record = records[request["checkpoint_id"]]
    except KeyError as exc:
        raise OfficialArenaError("checkpoint is absent from protected replay inventory") from exc
    if (
        replay_record["origin_state_hash"] != request["origin_state_hash"]
        or replay_record["replay_identity"] != request["replay_identity"]
    ):
        raise OfficialArenaError("protected replay identity differs from atomic request")
    _validate_dependency_binding(
        request=request,
        parameters=parameters,
        replay_record=replay_record,
    )

    provider = _provider(parameters["provider_entrypoint"])
    if (
        provider.benchmark_id != request["benchmark_id"]
        or provider.source_revision != request["source_revision"]
        or provider.scorer_id != parameters["scorer_id"]
        or provider.scorer_revision != parameters["scorer_revision"]
    ):
        raise OfficialArenaError("official provider identity differs from worker lock")
    blockers = provider.doctor(parameters["provider_config"])
    if blockers:
        raise OfficialArenaError("official provider preflight failed: " + "; ".join(blockers))

    arm_results: list[dict[str, Any]] = []
    arm_receipts: list[dict[str, Any]] = []
    runtime_ids: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="agentinputlab-official-arena-") as temporary:
        root = Path(temporary)
        for arm in request["arms"]:
            role = arm["role"]
            work_dir = root / role
            work_dir.mkdir()
            result = provider.run_arm(
                parameters["provider_config"],
                replay_record=replay_record,
                arm_payload=arm["payload"],
                seed=request["seed"],
                role=role,
                work_dir=work_dir,
            )
            if result.runtime_instance_id in runtime_ids:
                raise OfficialArenaError(
                    "official provider reused a runtime instance across paired arms"
                )
            runtime_ids.add(result.runtime_instance_id)
            usage = merge_usage(intervention_usage(arm["payload"]), result.usage)
            _validate_shared_budget_usage(arm["payload"], usage)
            intervention_failure = _intervention_failure(arm["payload"])
            arm_results.append(
                {
                    "role": role,
                    "arm_id": arm["arm_id"],
                    "metrics": dict(result.metrics),
                    "trace_hash": content_hash(result.trace),
                    "trace_measurement": _safe_trace_measurement(result.trace),
                    "usage": {name: usage[name] for name in _USAGE_FIELDS},
                    "intervention_failure": intervention_failure,
                }
            )
            arm_receipts.append(
                {
                    "role": role,
                    "arm_id": arm["arm_id"],
                    "origin_state_hash": request["origin_state_hash"],
                    "replay_identity": request["replay_identity"],
                    "seed": request["seed"],
                    "locked_dependency_hashes": request["locked_dependency_hashes"],
                    "runtime_instance_id": result.runtime_instance_id,
                    "completed": True,
                }
            )

    receipt = {
        "benchmark_id": request["benchmark_id"],
        "checkpoint_id": request["checkpoint_id"],
        "source_revision": request["source_revision"],
        "bridge_id": parameters["bridge_id"],
        "bridge_revision": parameters["bridge_revision"],
        "scorer_id": parameters["scorer_id"],
        "scorer_revision": parameters["scorer_revision"],
        "origin_state_hash": request["origin_state_hash"],
        "replay_identity": request["replay_identity"],
        "seed": request["seed"],
        "locked_dependency_hashes": request["locked_dependency_hashes"],
        "arms": arm_receipts,
    }
    return {
        "schema_version": 1,
        "protocol": RESPONSE_PROTOCOL,
        "request_hash": value["request_hash"],
        "receipt": receipt,
        "arms": arm_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one official atomic arena pair")
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()
    if args.doctor:
        print(json.dumps({"protocol": WORKER_PROTOCOL, "ready": True}, sort_keys=True))
        return 0
    try:
        request = json.load(sys.stdin)
        response = run(request)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
