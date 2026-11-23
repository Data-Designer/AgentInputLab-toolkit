"""Core bridge adapter for the pinned DecomP worker driver."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentinputlab.algorithms.decompose.decomp.worker_bridge import (
    DecompFrontierEvent,
    DecompGraphNode,
    DecompParticipantCall,
    DecompWorkerRequest,
    DecompWorkerResponse,
    validate_decomp_full_response,
)
from agentinputlab.contracts import AlgorithmOutputError, ExternalDependencyError
from agentinputlab.contracts.serialization import to_jsonable

from .driver import DEFAULT_MANIFEST, UPSTREAM_COMMIT, DecompDriver, DriverError

_UPSTREAM_PROCESS_STATE_LOCK = threading.RLock()


class PinnedDecompRuntime:
    """Adapt :class:`DecompDriver` to ``DecompWorkerRuntime`` without a fallback."""

    def __init__(
        self,
        *,
        manifest_path: Path = DEFAULT_MANIFEST,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.driver = DecompDriver(
            manifest_path=manifest_path,
            environment=environment,
        )

    def fingerprint(self) -> Mapping[str, Any]:
        try:
            with _UPSTREAM_PROCESS_STATE_LOCK:
                upstream = self.driver.smoke_upstream_core()
            driver_sha = _sha256_file(Path(__file__).with_name("driver.py"))
            adapter_sha = _sha256_file(Path(__file__))
        except DriverError as exc:
            raise ExternalDependencyError(
                f"pinned DecomP runtime identity failed with {exc.code}"
            ) from exc
        return {
            "worker_id": "agentinputlab.decomp.pinned.in_process.v1",
            "execution_mode": "in_process_external_environment",
            "concurrency_mode": "process_global_serial_rlock",
            "upstream_revision": UPSTREAM_COMMIT,
            "upstream_symbol": upstream["symbol"],
            "compatibility_manifest_sha256": upstream["manifest_sha256"],
            "driver_sha256": driver_sha,
            "runtime_adapter_sha256": adapter_sha,
        }

    def run(self, request: DecompWorkerRequest) -> DecompWorkerResponse:
        if not isinstance(request, DecompWorkerRequest):
            raise TypeError("PinnedDecompRuntime requires DecompWorkerRequest")
        request_mapping = {
            "case_id": request.case_id,
            "question": request.question,
            "paras": list(request.paras),
            "paragraph_ids": list(request.paragraph_ids),
            "context_hash": request.context_hash,
            "config_artifact_id": request.config_artifact_id,
            "backend_binding": to_jsonable(request.backend_binding),
            "max_steps": request.max_steps,
            "upstream_revision": request.upstream_revision,
            "schema_version": request.schema_version,
        }
        try:
            # Upstream mutates process-global CWD/sys.path while importing its
            # pinned package, so all in-process identities share one serial lock.
            with _UPSTREAM_PROCESS_STATE_LOCK:
                raw = self.driver.run(request_mapping)
        except DriverError as exc:
            error_type = (
                AlgorithmOutputError
                if exc.code
                in {
                    "UPSTREAM_OUTPUT_INVALID",
                    "USAGE_ACCOUNTING_UNAVAILABLE",
                }
                else ExternalDependencyError
            )
            raise error_type(f"pinned DecomP driver failed with {exc.code}") from exc
        response = _parse_response(raw)
        validate_decomp_full_response(response)
        return response


def _parse_response(value: Mapping[str, Any]) -> DecompWorkerResponse:
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("DecomP driver response must be a mapping")
    try:
        nodes = tuple(
            DecompGraphNode(
                node_id=_text(item, "node_id", "node"),
                question=_text(item, "question", "node", allow_empty=True),
                answer=_text(item, "answer", "node", allow_empty=True),
                handler=_text(item, "handler", "node", allow_empty=True),
                depth=_nonnegative_integer(item.get("depth"), "node.depth"),
            )
            for item in _mapping_sequence(value.get("nodes"), "nodes")
        )
        edges = tuple(_edge(item) for item in _sequence(value.get("dependency_edges"), "edges"))
        participant_calls = tuple(
            DecompParticipantCall(
                index=_nonnegative_integer(item.get("index"), "participant_call.index"),
                participant=_text(item, "participant", "participant_call"),
                input_state_id=_text(item, "input_state_id", "participant_call"),
                output_state_id=_text(item, "output_state_id", "participant_call"),
                command=_text(item, "command", "participant_call", allow_empty=True),
            )
            for item in _mapping_sequence(value.get("participant_calls"), "participant_calls")
        )
        frontier = tuple(
            DecompFrontierEvent(
                index=_nonnegative_integer(item.get("index"), "frontier.index"),
                state_id=_text(item, "state_id", "frontier"),
                score=_finite_number(item.get("score"), "frontier.score"),
                next_command=_text(
                    item,
                    "next_command",
                    "frontier",
                    allow_empty=True,
                ),
                action=_text(item, "action", "frontier"),
            )
            for item in _mapping_sequence(value.get("frontier"), "frontier")
        )
        usage = value.get("usage")
        artifacts = value.get("native_artifacts", {})
        if not isinstance(usage, Mapping) or not isinstance(artifacts, Mapping):
            raise AlgorithmOutputError("DecomP response usage/artifacts must be mappings")
        response = DecompWorkerResponse(
            case_id=_text(value, "case_id", "response"),
            profile_id=_text(value, "profile_id", "response"),
            upstream_revision=_text(value, "upstream_revision", "response"),
            paragraph_ids=tuple(
                _text_value(item, "response.paragraph_ids")
                for item in _sequence(value.get("paragraph_ids"), "paragraph_ids")
            ),
            context_hash=_text(value, "context_hash", "response"),
            nodes=nodes,
            dependency_edges=edges,
            participant_calls=participant_calls,
            frontier=frontier,
            reasoning_chain=_text(value, "reasoning_chain", "response"),
            final_answer=_text(value, "final_answer", "response", allow_empty=True),
            termination=_text(value, "termination", "response"),
            component_only=value.get("component_only", False),
            usage=dict(usage),
            schema_version=_text(value, "schema_version", "response"),
            native_artifacts={str(key): str(item) for key, item in artifacts.items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AlgorithmOutputError):
            raise
        raise AlgorithmOutputError("DecomP driver response is malformed") from exc
    return response


def _mapping_sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    result = _sequence(value, label)
    if not all(isinstance(item, Mapping) for item in result):
        raise AlgorithmOutputError(f"DecomP {label} must contain mappings")
    return result  # type: ignore[return-value]


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlgorithmOutputError(f"DecomP {label} must be a sequence")
    return tuple(value)


def _edge(value: Any) -> tuple[str, str]:
    edge = _sequence(value, "edge")
    if len(edge) != 2 or not all(isinstance(item, str) and item for item in edge):
        raise AlgorithmOutputError("DecomP edge must contain two state IDs")
    return edge[0], edge[1]


def _text(
    value: Mapping[str, Any],
    key: str,
    label: str,
    *,
    allow_empty: bool = False,
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        raise AlgorithmOutputError(f"DecomP {label}.{key} must be text")
    return item


def _text_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlgorithmOutputError(f"DecomP {label} must contain text")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AlgorithmOutputError(f"DecomP {label} must be a non-negative integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmOutputError(f"DecomP {label} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise AlgorithmOutputError(f"DecomP {label} must be finite")
    return result


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DriverError("RUNTIME_SOURCE_MISSING", "runtime source file is unavailable") from exc


__all__ = ["PinnedDecompRuntime"]
