"""Unix-socket transport for a process-persistent R²AG common-support runtime.

The transport is deliberately absent from the scientific fingerprint: it only
changes process placement and model lifetime.  The server returns the exact
fingerprint and response produced by ``LocalR2AGCommonSupportRuntime``.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import freeze_mapping, to_jsonable

from .common_support import R2AGCommonSupportRuntime, R2AGRuntimeRequest

R2AG_RUNTIME_SOCKET_ENV = "AGENTINPUTLAB_R2AG_RUNTIME_SOCKET"
R2AG_RUNTIME_TIMEOUT_ENV = "AGENTINPUTLAB_R2AG_RUNTIME_TIMEOUT_SECONDS"
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_MAX_UNIX_SOCKET_PATH_BYTES = 100


def _validated_socket_path(socket_path: Path) -> Path:
    path = socket_path.expanduser().resolve()
    if len(os.fsencode(path)) > _MAX_UNIX_SOCKET_PATH_BYTES:
        raise ValidationError("R²AG runtime socket path is too long for AF_UNIX")
    return path


def _exchange(socket_path: Path, payload: Mapping[str, Any], *, timeout: float) -> Any:
    encoded = (
        json.dumps(to_jsonable(payload), ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_MESSAGE_BYTES:
        raise ValidationError("R²AG socket request exceeds the transport limit")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(socket_path))
            client.sendall(encoded)
            with client.makefile("rb") as handle:
                response = handle.readline(_MAX_MESSAGE_BYTES + 1)
    except (OSError, TimeoutError) as exc:
        raise ExternalDependencyError(f"R²AG socket runtime is unavailable: {exc}") from exc
    if not response or len(response) > _MAX_MESSAGE_BYTES:
        raise ExternalDependencyError("R²AG socket runtime returned an invalid frame")
    try:
        decoded = json.loads(response)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalDependencyError("R²AG socket runtime returned invalid JSON") from exc
    if not isinstance(decoded, Mapping) or not isinstance(decoded.get("ok"), bool):
        raise ExternalDependencyError("R²AG socket runtime returned an invalid envelope")
    if not decoded["ok"]:
        error_type = decoded.get("error_type", "RuntimeError")
        message = decoded.get("message", "unknown shared-runtime failure")
        raise ExternalDependencyError(f"R²AG shared runtime {error_type}: {message}")
    return decoded.get("result")


def _request_payload(request: R2AGRuntimeRequest) -> dict[str, Any]:
    return {
        "query": request.query,
        "documents": [
            {
                "document": {
                    "doc_id": item.document.doc_id,
                    "text": item.document.text,
                    "title": item.document.title,
                    "corpus_revision": item.document.corpus_revision,
                    "metadata": dict(item.document.metadata),
                },
                "score": item.score,
                "rank": item.rank,
                "features": dict(item.features),
            }
            for item in request.documents
        ],
        "max_new_tokens": request.max_new_tokens,
        "seed": request.seed,
    }


def _request_from_payload(value: Any) -> R2AGRuntimeRequest:
    if not isinstance(value, Mapping):
        raise ValidationError("R²AG socket request payload must be an object")
    raw_documents = value.get("documents")
    if not isinstance(raw_documents, list):
        raise ValidationError("R²AG socket request documents must be a list")
    documents: list[RetrievalResult] = []
    for raw in raw_documents:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("document"), Mapping):
            raise ValidationError("R²AG socket request contains an invalid document")
        source = raw["document"]
        metadata = source.get("metadata", {})
        features = raw.get("features", {})
        if not isinstance(metadata, Mapping) or not isinstance(features, Mapping):
            raise ValidationError("R²AG socket document mappings are invalid")
        documents.append(
            RetrievalResult(
                document=Document(
                    doc_id=str(source.get("doc_id", "")),
                    text=str(source.get("text", "")),
                    title=str(source.get("title", "")),
                    corpus_revision=str(source.get("corpus_revision", "")),
                    metadata=dict(metadata),
                ),
                score=float(raw.get("score", 0.0)),
                rank=int(raw.get("rank", 0)),
                features={str(name): float(number) for name, number in features.items()},
            )
        )
    return R2AGRuntimeRequest(
        query=str(value.get("query", "")),
        documents=tuple(documents),
        max_new_tokens=int(value.get("max_new_tokens", 0)),
        seed=int(value.get("seed", -1)),
    )


@dataclass(slots=True)
class UnixSocketR2AGCommonSupportRuntime:
    """Client implementing the regular runtime protocol over a local socket."""

    socket_path: Path
    timeout_seconds: float = 3600.0
    worker_id: str = "r2ag-common-support-local"
    _fingerprint: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = _validated_socket_path(self.socket_path)
        if not path.is_absolute() or not path.exists() or not stat.S_ISSOCK(path.stat().st_mode):
            raise ValidationError("R²AG runtime socket must be an existing Unix socket")
        if self.timeout_seconds <= 0:
            raise ValidationError("R²AG runtime socket timeout must be positive")
        observed = _exchange(path, {"operation": "fingerprint"}, timeout=self.timeout_seconds)
        if not isinstance(observed, Mapping):
            raise ExternalDependencyError("R²AG socket fingerprint is invalid")
        object.__setattr__(self, "socket_path", path)
        object.__setattr__(self, "_fingerprint", freeze_mapping(dict(observed)))

    def run(self, request: R2AGRuntimeRequest) -> Mapping[str, Any]:
        if not isinstance(request, R2AGRuntimeRequest):
            raise TypeError("R²AG runtime requires R2AGRuntimeRequest")
        result = _exchange(
            self.socket_path,
            {"operation": "run", "request": _request_payload(request)},
            timeout=self.timeout_seconds,
        )
        if not isinstance(result, Mapping):
            raise ExternalDependencyError("R²AG socket result is invalid")
        return freeze_mapping(dict(result))

    def fingerprint(self) -> Mapping[str, Any]:
        return self._fingerprint


def serve_unix_socket_r2ag_runtime(
    socket_path: Path,
    runtime: R2AGCommonSupportRuntime,
    *,
    stop_event: threading.Event | None = None,
    max_requests: int | None = None,
    ready_callback: Callable[[], None] | None = None,
    event_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> int:
    """Serve one already-loaded runtime; requests are serialized by design."""

    if not isinstance(runtime, R2AGCommonSupportRuntime):
        raise TypeError("R²AG socket server requires R2AGCommonSupportRuntime")
    path = _validated_socket_path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if stat.S_ISSOCK(path.stat().st_mode):
            path.unlink()
        else:
            raise ValidationError("R²AG socket path exists and is not a socket")
    stop = stop_event or threading.Event()
    served = 0
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(16)
        server.settimeout(1.0)
        if ready_callback is not None:
            ready_callback()
        while not stop.is_set() and (max_requests is None or served < max_requests):
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                started = time.monotonic()
                operation: Any = None
                try:
                    with connection.makefile("rb") as handle:
                        frame = handle.readline(_MAX_MESSAGE_BYTES + 1)
                    if not frame or len(frame) > _MAX_MESSAGE_BYTES:
                        raise ValidationError("R²AG socket request frame is invalid")
                    request = json.loads(frame)
                    if not isinstance(request, Mapping):
                        raise ValidationError("R²AG socket request must be an object")
                    operation = request.get("operation")
                    if operation == "fingerprint":
                        result = runtime.fingerprint()
                    elif operation == "run":
                        result = runtime.run(_request_from_payload(request.get("request")))
                    else:
                        raise ValidationError("R²AG socket operation is unsupported")
                    response = {"ok": True, "result": to_jsonable(result)}
                except Exception as exc:  # Preserve a typed, auditable failure envelope.
                    response = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                encoded = (
                    json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                with suppress(OSError):
                    connection.sendall(encoded)
            served += 1
            if event_callback is not None:
                event_callback(
                    {
                        "operation": operation,
                        "ok": response["ok"],
                        "request_index": served,
                        "seconds": round(time.monotonic() - started, 6),
                    }
                )
    finally:
        server.close()
        if path.exists() and stat.S_ISSOCK(path.stat().st_mode):
            path.unlink()
    return served


__all__ = [
    "R2AG_RUNTIME_SOCKET_ENV",
    "R2AG_RUNTIME_TIMEOUT_ENV",
    "UnixSocketR2AGCommonSupportRuntime",
    "serve_unix_socket_r2ag_runtime",
]
