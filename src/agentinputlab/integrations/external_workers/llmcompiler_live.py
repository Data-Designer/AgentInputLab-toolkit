"""Live, context-bound callbacks for the isolated pinned LLMCompiler worker.

The worker process has no provider credential or request-selected network tool.
A per-call authenticated loopback endpoint exposes only the already bound model
and read-only ToolConnector. The ordinary supervisor still bounds its process
and pipes. Caller backends/tools remain responsible for their own I/O timeouts.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import secrets
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agentinputlab.backends import GenerationRequest, LLMBackend
from agentinputlab.connectors.tools import ToolCall, ToolConnector
from agentinputlab.connectors.tools.openapi import validate_json_value
from agentinputlab.contracts import ExecutionBudget, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .base import JsonLineSubprocessWorker
from .supervisor import SupervisedWorkerError

_KEY_ENV = "AGENTINPUTLAB_LLMC_CALLBACK_KEY"


def _strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


class _Callbacks:
    def __init__(self, owner, request):
        self.owner, self.request = owner, request
        budget = request["payload"].get("execution_budget")
        if not isinstance(budget, Mapping):
            raise ValidationError("live LLMCompiler requires the state execution_budget")
        self.budget = ExecutionBudget(**budget)
        if self.budget.max_context_tokens is not None or self.budget.max_cost_usd is not None:
            raise ValidationError("live callbacks do not yet attest pre-call input-token or priced-cost ceilings")
        self.specs = {s.name: s for s in owner.tools.list_tools()}
        self.argument_order = {name: tuple(keys) for name, keys in owner.argument_order.items()}
        if to_jsonable(owner.argument_order) != request["payload"].get("tool_argument_order", {}):
            raise ValidationError("tool positional argument order differs from the request")
        if not self.specs or any(not s.read_only or s.network for s in self.specs.values()):
            raise ValidationError("live LLMCompiler tools must be read-only and no-network")
        if to_jsonable(tuple(self.specs.values())) != request["payload"]["tool_specs"]:
            raise ValidationError("live tool schemas differ from the request")
        deps = request["dependency_fingerprints"]
        if (content_hash(deps["backend"]) != content_hash(owner.backend.fingerprint())
                or content_hash(deps["tools"]) != content_hash(owner.tools.fingerprint())):
            raise ValidationError("live dependencies differ from the request")
        self.lock = threading.RLock()
        self.model_calls = self.tool_calls = self.input_tokens = self.output_tokens = 0
        self.reserved_output = self.inflight = self.unknown_usage_calls = 0
        self.cost = 0.0
        self.unpriced = 0
        self.closed = self.fatal = False
        self.attested = False
        self.expected_attestation = owner.fingerprint()["worker_attestation"]
        self.seen = set()
        self.events = []

    def usage(self):
        with self.lock:
            result = {"model_calls": self.model_calls, "tool_calls": self.tool_calls,
                "retrieval_calls": 0, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "unpriced_model_calls": self.model_calls if self.unpriced else 0}
            if self.unpriced == 0:
                result["cost_usd"] = self.cost
            return result

    def dispatch(self, value):
        if (not isinstance(value, dict) or set(value) != {"request_id", "call_id", "kind", "payload"}
                or value["request_id"] != self.request["request_id"]
                or not isinstance(value["call_id"], str) or not value["call_id"]):
            raise ValueError("invalid callback envelope")
        with self.lock:
            if self.closed or self.fatal or value["call_id"] in self.seen:
                raise ValueError("closed, failed, or duplicate callback")
            self.seen.add(value["call_id"])
        if value["kind"] == "attest":
            if self.attested or value["payload"] != self.expected_attestation:
                raise ValueError("worker source/version attestation differs")
            self.attested = True
            self.events.append({"kind": "attestation", "matched": True})
            return {"attested": True}
        if not self.attested:
            raise ValueError("native worker must attest before model/tool dispatch")
        if value["kind"] == "model":
            return self.model(value["payload"])
        if value["kind"] == "tool":
            return self.tool(value["payload"])
        raise ValueError("unknown callback kind")

    def model(self, value):
        if (not isinstance(value, dict) or set(value) != {"prompt", "channel", "stop", "seed"}
                or value["channel"] not in {"planner", "join"}
                or value["seed"] != self.request["payload"]["seed"]
                or not isinstance(value["prompt"], str) or not value["prompt"]
                or not isinstance(value["stop"], list)
                or not all(isinstance(s, str) and s for s in value["stop"])):
            raise ValueError("invalid live model request")
        cap = self.owner.per_call_output_cap
        request = GenerationRequest(prompt=value["prompt"], seed=value["seed"],
            stop=tuple(value["stop"]), max_tokens=cap, temperature=0.0,
            metadata={"method_id": "llm_compiler", "stage": value["channel"],
                      "transport_profile": "controlled_host_callback_v1"})
        with self.lock:
            if (self.fatal or self.closed or self.model_calls >= self.budget.max_model_calls
                    or self.output_tokens + self.reserved_output + cap > self.budget.max_output_tokens):
                raise ValueError("live model budget exhausted before dispatch")
            self.model_calls += 1
            self.unpriced += 1
            self.reserved_output += cap
            self.inflight += 1
        event = {"kind": "model", "request": to_jsonable(request)}
        self.events.append(event)
        try:
            response = self.owner.backend.generate(request)
            usage = response.usage
            with self.lock:
                self.input_tokens += usage.input_tokens
                self.output_tokens += usage.output_tokens
                self.reserved_output -= cap
                if usage.cost_usd is not None:
                    self.unpriced -= 1
                    self.cost += usage.cost_usd
                if usage.output_tokens > cap:
                    self.fatal = True
            event["response"] = to_jsonable(response)
            if usage.output_tokens > cap:
                raise RuntimeError("provider exceeded the requested output cap")
            return {"text": response.text, "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens, "finish_reason": response.finish_reason}
        except Exception as exc:
            with self.lock:
                self.fatal = True
                if "response" not in event:
                    self.unknown_usage_calls += 1
                    # Keep the full output reservation when actual usage is unknown.
            event["error_type"] = type(exc).__name__
            raise RuntimeError("bound model callback failed; no retry") from None
        finally:
            with self.lock:
                self.inflight -= 1

    def tool(self, value):
        if (not isinstance(value, dict) or set(value) != {"name", "arguments"}
                or value["name"] not in self.specs):
            raise ValueError("invalid bound tool request")
        spec = self.specs[value["name"]]
        properties = spec.input_schema.get("properties", {})
        arguments = value["arguments"]
        if isinstance(arguments, list):
            if len(arguments) == 1 and isinstance(arguments[0], dict):
                arguments = arguments[0]
            else:
                order = self.argument_order.get(spec.name)
                if order is None:
                    if len(properties) > 1:
                        raise ValueError("multi-argument tools require explicit positional argument_order")
                    order = tuple(properties)
                if len(arguments) > len(order):
                    raise ValueError("positional argument count differs from its explicit binding")
                arguments = dict(zip(order[:len(arguments)], arguments, strict=True))
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object or positional list")
        validate_json_value(arguments, spec.input_schema)
        with self.lock:
            if self.fatal or self.closed or self.tool_calls >= self.budget.max_tool_calls:
                raise ValueError("live tool budget exhausted before dispatch")
            self.tool_calls += 1
            self.inflight += 1
        call = ToolCall(value["name"], arguments, call_id=secrets.token_hex(16))
        event = {"kind": "tool", "request": to_jsonable(call)}
        self.events.append(event)
        try:
            result = self.owner.tools.call(call)
            event["response"] = to_jsonable(result)
            if result.error:
                raise RuntimeError("bound tool returned an error")
            return {"output": to_jsonable(result.output)}
        except Exception as exc:
            self.fatal = True
            event["error_type"] = type(exc).__name__
            raise RuntimeError("bound read-only tool failed; no synthetic observation") from None
        finally:
            with self.lock:
                self.inflight -= 1


def _server(callbacks, key, limit):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                self.connection.settimeout(10)
                if (self.path != "/callback" or not hmac.compare_digest(
                        self.headers.get("Authorization", ""), "Bearer " + key)):
                    self.send_error(403)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > limit:
                    raise ValueError("callback byte limit exceeded")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("truncated callback")
                result = {"ok": True, "payload": callbacks.dispatch(_strict_json(raw))}
            except Exception as exc:
                result = {"ok": False, "failure_code": "HOST_CALLBACK_FAILED",
                    "error_type": type(exc).__name__}
            body = json.dumps(result, allow_nan=False).encode()
            if len(body) > limit:
                body = b'{"ok":false,"failure_code":"CALLBACK_RESPONSE_TOO_LARGE"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.block_on_close = False
    return server


@dataclass
class LLMCompilerLiveWorker:
    """Execute pinned Planner/TFU/join with caller-owned live dependencies.

    Keep the legacy worker environment separate from the host package. Tools
    must be read-only/no-network, and backends/tools must set their own bounded
    I/O timeouts. Streaming controls use buffered callback delivery, not provider
    streaming latency/parity. Priced/input-token ceilings fail before dispatch.
    """

    python_executable: str
    checkout: Path
    prompt_bundle: Path
    backend: LLMBackend
    tools: ToolConnector
    per_call_output_cap: int = 512
    timeout_seconds: float = 400
    argument_order: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    worker_id: str = "agentinputlab.llmcompiler.controlled-live.v1"
    _last_resource_receipt: Any = field(default=None, init=False, repr=False)
    callback_events: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if (isinstance(self.per_call_output_cap, bool) or not isinstance(self.per_call_output_cap, int)
                or self.per_call_output_cap < 1):
            raise ValidationError("per_call_output_cap must be positive")
        if (not isinstance(self.python_executable, str) or not self.python_executable
                or isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValidationError("worker interpreter and finite positive timeout required")
        if any(not isinstance(name, str) or not isinstance(order, (list, tuple))
               or not all(isinstance(key, str) and key for key in order)
               or len(set(order)) != len(order) for name, order in self.argument_order.items()):
            raise ValidationError("argument_order must map tool names to unique parameter names")
        self.checkout, self.prompt_bundle = Path(self.checkout), Path(self.prompt_bundle)

    @property
    def last_resource_receipt(self):
        return self._last_resource_receipt

    def fingerprint(self):
        import hashlib
        from importlib import resources

        package = resources.files("workers").joinpath("llmcompiler")
        driver_sha = hashlib.sha256(package.joinpath("driver.py").read_bytes()).hexdigest()
        live_sha = hashlib.sha256(package.joinpath("live.py").read_bytes()).hexdigest()
        manifest = package.joinpath("compatibility_manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        attestation = {"schema": "agentinputlab.llmcompiler.live-attestation.v1",
            "driver_sha256": driver_sha, "live_driver_sha256": live_sha,
            "compatibility_manifest_sha256": manifest_sha,
            "required_versions": json.loads(manifest)["legacy_dependencies"]["required_versions"],
            "source_files_verified": True, "required_versions_verified": True, "import_origins_verified": True}
        return {"worker_id": self.worker_id,
            "upstream_commit": "a00c9d35507507da70e8c637eee64efc8c1857ae",
            "driver_sha256": driver_sha, "live_driver_sha256": live_sha,
            "compatibility_manifest_sha256": manifest_sha, "worker_attestation": attestation,
            "prompt_bundle_sha256": hashlib.sha256(self.prompt_bundle.read_bytes()).hexdigest(),
            "model": dict(self.backend.fingerprint()), "tools": dict(self.tools.fingerprint()),
            "model_transport": "context_bound_host_callback", "tool_transport": "context_bound_readonly_callback",
            "streaming_delivery": "buffered_completion_callbacks_not_native_streaming",
            "positional_argument_order": to_jsonable(self.argument_order),
            "per_call_output_cap": self.per_call_output_cap, "timeout_seconds": self.timeout_seconds}

    def call(self, payload):
        if not isinstance(payload, Mapping):
            raise ValidationError("live LLMCompiler requires a typed request mapping")
        callbacks = _Callbacks(self, payload)
        identity = self.fingerprint()
        self.callback_events = callbacks.events
        self._last_resource_receipt = None
        key = secrets.token_urlsafe(32)
        server = _server(callbacks, key, 4 * 1024 * 1024)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        worker = JsonLineSubprocessWorker(
            command=(self.python_executable, "-m", "workers.llmcompiler.live",
                "--checkout", str(self.checkout), "--prompt-bundle", str(self.prompt_bundle),
                "--callback-url", f"http://127.0.0.1:{server.server_port}/callback",
                "--callback-key-env", _KEY_ENV), worker_id=self.worker_id,
            identity=identity, timeout_seconds=self.timeout_seconds,
            environment={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
                "CUDA_VISIBLE_DEVICES": "", _KEY_ENV: key})
        try:
            try:
                response = dict(worker.call(payload))
            except SupervisedWorkerError:
                response = {"status": "error", "request_id": payload["request_id"],
                    "failure_code": "LIVE_WORKER_EXECUTION_FAILED", "message": "supervised native worker failed"}
            with callbacks.lock:
                callbacks.closed = True
                usage = callbacks.usage()
                incomplete = callbacks.inflight > 0
            if incomplete:
                response = {"status": "error", "request_id": payload["request_id"],
                    "failure_code": "CALLBACK_USAGE_INCOMPLETE", "message": "bound callback still in flight at worker exit"}
            elif response.get("status") == "ok" and any(
                response.get("usage", {}).get(k) != usage[k]
                for k in ("model_calls", "tool_calls", "input_tokens", "output_tokens")):
                response = {"status": "error", "request_id": payload["request_id"],
                    "failure_code": "CALLBACK_USAGE_MISMATCH", "message": "worker and host usage differ"}
            response["usage"] = usage
            response["consumed"] = bool(usage["model_calls"] or usage["tool_calls"])
            response["callback_diagnostics"] = {"unknown_usage_calls": callbacks.unknown_usage_calls,
                "usage_incomplete": incomplete, "automatic_retry": False}
            return response
        finally:
            self._last_resource_receipt = worker.last_resource_receipt
            callbacks.closed = True
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
