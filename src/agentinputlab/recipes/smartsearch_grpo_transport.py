"""Loopback-only coordinator RPC for one native GRPO run, with no retries.

This carries measured costs and native helper completions, never Python code,
shell commands, pickle or import paths. Retrieval has its separate fixed-tool
service. The coordinator owns ONE ledger and ONE helper callback for all ranks.
"""
from __future__ import annotations

import hmac
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable

from .smartsearch_grpo_meter import SmartSearchGRPOCostLedger, meter_smartsearch_grpo_helper
from .smartsearch_grpo_tools import _save
from .smartsearch_teacher import SmartSearchTeacherCompletion

_LIMIT = 16 * 1024 * 1024
_METHODS = {
    "reserve": {"kind", "worker", "input_token_counts", "samples", "max_new_tokens"},
    "settle": {"ticket", "output_token_counts"},
    "fail": {"ticket", "error_type"},
    "optimizer_begin": {"rank"},
    "optimizer_end": {"rank"},
    "snapshot": set(),
}


class SmartSearchGRPOCoordinator:
    """Own aggregate accounting and a serialized, measured native helper."""

    def __init__(self, *, ledger, generate, count_input_tokens, model_fingerprint_sha256):
        if not isinstance(ledger, SmartSearchGRPOCostLedger):
            raise ValidationError("SmartSearch GRPO coordinator requires one shared cost ledger")
        self.ledger = ledger
        self.helper = meter_smartsearch_grpo_helper(generate, ledger=ledger, worker="shared-native-teacher",
            fingerprint=model_fingerprint_sha256, count_input_tokens=count_input_tokens)
        self.fingerprint = model_fingerprint_sha256
        self._lock = threading.Lock()
        self._helper_number = 0
        self.workers = {}

    def dispatch(self, operation, arguments):
        if operation in _METHODS and set(arguments) == _METHODS[operation]:
            return getattr(self.ledger, operation)(**arguments)
        if operation == "helper" and set(arguments) == {"prompt", "temperature", "max_new_tokens", "stop", "seed"}:
            with self._lock:
                self._helper_number += 1
                number = self._helper_number
            # A missing transport response is not free work: the ledger and
            # these coordinator-side records survive a disconnected worker.
            _save(self.ledger.root, f"native-helper-{number:06d}-request.json", arguments)
            result = self.helper(**arguments)
            _save(self.ledger.root, f"native-helper-{number:06d}-response.json", result)
            return to_jsonable(result)
        if operation == "worker_completed" and set(arguments) == {"rank", "role", "usage"}:
            rank, role, usage = arguments["rank"], arguments["role"], arguments["usage"]
            fields = {"forward_attempts", "forward_responses", "input_token_positions", "nonfinite_native_metrics"}
            if type(rank) is not int or not 0 <= rank < self.ledger.limits.actor_ranks or (
                role not in {"actor_rollout", "actor_rollout_ref", "actor", "rollout", "ref"}
                or not isinstance(usage, dict) or set(usage) != fields
                or any(type(n) is not int or n < 0 for n in usage.values())
                or usage["forward_attempts"] != usage["forward_responses"]
                or usage["nonfinite_native_metrics"]
            ):
                raise ExternalDependencyError("SmartSearch GRPO worker has incomplete/nonfinite native observations")
            key = f"{role}-rank-{rank}"
            with self._lock:
                if key in self.workers:
                    raise ExternalDependencyError("SmartSearch GRPO worker completion cannot be repeated")
                _save(self.ledger.root, key + "-terminal.json", arguments)
                self.workers[key] = to_jsonable(arguments)
            return None
        raise ValidationError("SmartSearch GRPO coordinator operation/argument schema differs")

    def assert_healthy(self, *, expected_optimizer_steps, required_roles):
        result = self.ledger.assert_healthy(expected_optimizer_steps=expected_optimizer_steps)
        expected = {f"{role}-rank-{rank}" for role in required_roles
                    for rank in range(self.ledger.limits.actor_ranks)}
        if set(self.workers) != expected or any(
            not row["usage"]["forward_responses"] for row in self.workers.values()
        ):
            raise ExternalDependencyError("SmartSearch GRPO actor/reference rank receipts are incomplete")
        return {**result, "workers": to_jsonable(self.workers)}


def make_smartsearch_grpo_coordinator_server(coordinator, *, capability, port=0):
    """Start separately with serve_forever; bound to 127.0.0.1 only."""
    if not isinstance(coordinator, SmartSearchGRPOCoordinator) or not isinstance(capability, str) or (
        not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", capability)
        or type(port) is not int or not 0 <= port <= 65535
    ):
        raise ValidationError("SmartSearch GRPO coordinator needs an explicit random capability")
    slots = threading.BoundedSemaphore(16)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(600)

        def log_message(self, *_args):
            pass

        def send_json(self, status, value):
            raw = json.dumps(to_jsonable(value), ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            parts = self.path.split("/")
            if len(parts) != 3 or not hmac.compare_digest(parts[1].encode(), capability.encode()):
                self.send_json(401, {"error_type": "Unauthorized"})
                return
            if parts[2] != "rpc":
                self.send_json(404, {"error_type": "NotFound"})
                return
            entered = False
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit() or (
                    not 0 < int(lengths[0]) <= _LIMIT or self.headers.get_content_type() != "application/json"
                ):
                    raise ValidationError("SmartSearch GRPO RPC body contract differs")
                entered = slots.acquire(blocking=False)
                if not entered:
                    raise ExternalDependencyError("SmartSearch GRPO coordinator is at its explicit concurrency limit")

                def unique(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValidationError("SmartSearch GRPO RPC duplicate key")
                        result[key] = value
                    return result

                raw = self.rfile.read(int(lengths[0]))
                if len(raw) != int(lengths[0]):
                    raise ValidationError("SmartSearch GRPO RPC body is incomplete")
                body = json.loads(raw, object_pairs_hook=unique,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
                if not isinstance(body, dict) or set(body) != {"operation", "arguments"} or (
                    not isinstance(body["operation"], str) or not isinstance(body["arguments"], dict)
                ):
                    raise ValidationError("SmartSearch GRPO RPC envelope differs")
                result = coordinator.dispatch(body["operation"], body["arguments"])
            except BaseException as exc:
                coordinator.ledger.fail("authenticated-rpc", error_type=type(exc).__name__)
                self.send_json(409, {"error_type": type(exc).__name__, "error": str(exc)})
                return
            finally:
                if entered:
                    slots.release()
            self.send_json(200, {"result": result})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    # Do not discard a paid in-flight helper when the driver fails. Closing
    # this server joins its own authenticated requests before the run exits.
    server.daemon_threads = False
    return server


class SmartSearchGRPOCoordinatorClient:
    """Synchronous, JSON-only worker client; failed neural requests never retry."""

    def __init__(self, endpoint, *, timeout=600):
        try:
            url = urlsplit(endpoint)
            port = url.port
        except (TypeError, ValueError) as exc:
            raise ValidationError("SmartSearch GRPO coordinator URL is invalid") from exc
        if url.scheme != "http" or url.hostname != "127.0.0.1" or not port or (
            url.username or url.password or url.query or url.fragment
            or not re.fullmatch(r"/[A-Za-z0-9_-]{32,128}/rpc", url.path)
            or type(timeout) is not int or not 1 <= timeout <= 3600
        ):
            raise ValidationError("SmartSearch GRPO coordinator must be an operator-owned loopback endpoint")
        self.endpoint, self.timeout, self.failed = endpoint, timeout, False
        # Neither proxy settings nor HTTP redirects may move native prompts
        # or the run capability to a remote service.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_args, **_kwargs):
                return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def _rpc(self, operation, arguments):
        if self.failed and operation not in {"fail", "settle", "optimizer_end", "worker_completed", "snapshot"}:
            raise ExternalDependencyError("SmartSearch GRPO coordinator request failed; no automatic neural retry")
        raw = json.dumps(to_jsonable(dict(operation=operation, arguments=arguments)), allow_nan=False).encode()
        if len(raw) > _LIMIT:
            raise ValidationError("SmartSearch GRPO RPC input exceeds its byte bound")
        request = urllib.request.Request(self.endpoint, data=raw, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                value = json.loads(response.read(_LIMIT + 1))
            if not isinstance(value, dict) or set(value) != {"result"}:
                raise ExternalDependencyError("SmartSearch GRPO coordinator response schema differs")
            return value["result"]
        except BaseException as exc:
            self.failed = True
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    error = json.loads(exc.read(16384))
                except (ValueError, UnicodeError):
                    error = {}
                if error.get("error_type") == "BudgetExceededError":
                    raise BudgetExceededError("SmartSearch GRPO aggregate coordinator allowance exhausted") from exc
            raise ExternalDependencyError("SmartSearch GRPO coordinator transport failed; no automatic retry") from exc

    def reserve(self, **arguments):
        return self._rpc("reserve", arguments)

    def settle(self, ticket, **arguments):
        return self._rpc("settle", {"ticket": ticket, **arguments})

    def fail(self, ticket, **arguments):
        return self._rpc("fail", {"ticket": ticket, **arguments})

    def optimizer_begin(self, rank):
        return self._rpc("optimizer_begin", {"rank": rank})

    def optimizer_end(self, rank):
        return self._rpc("optimizer_end", {"rank": rank})

    def snapshot(self):
        return self._rpc("snapshot", {})

    def worker_completed(self, *, rank, role, usage):
        return self._rpc("worker_completed", dict(rank=rank, role=role, usage=usage))

    def generate(self, prompt, *, temperature, max_new_tokens, stop, seed):
        value = self._rpc("helper", dict(prompt=prompt, temperature=temperature,
            max_new_tokens=max_new_tokens, stop=list(stop), seed=seed))
        try:
            return SmartSearchTeacherCompletion(**value)
        except (TypeError, ValidationError) as exc:
            self.failed = True
            raise ExternalDependencyError("SmartSearch GRPO coordinator omitted typed measured helper usage") from exc
