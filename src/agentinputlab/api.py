"""Small Python API and authenticated loopback HTTP transport for the SDK.

No experiment archives, remote imports, arbitrary commands or client-supplied
backend URLs are accepted. Operators configure dependencies in Python.
"""

from __future__ import annotations

import argparse
import hmac
import inspect
import json
import math
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, get_type_hints

from agentinputlab import ContextBlock, ExecutionBudget, FrozenAgentState, Message
from agentinputlab.benchmarks import BenchmarkRegistry
from agentinputlab.contracts import InterventionContext
from agentinputlab.contracts.errors import (
    AgentInputLabError,
    AlgorithmOutputError,
    AlgorithmUnavailableError,
    BudgetExceededError,
    CapabilityError,
    RegistryError,
    ValidationError,
)
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.registry import AlgorithmRegistry
from agentinputlab.runner import CompositeInterventionRunner, InterventionRunner
from agentinputlab.selectors import NoOpIntervention


def _config_fields(annotation: Any, depth: int = 0) -> list[dict[str, Any]] | None:
    if depth > 3 or not inspect.isclass(annotation) or not is_dataclass(annotation):
        return None
    try:
        hints = get_type_hints(annotation)
    except (NameError, TypeError):
        hints = {}
    result = []
    for f in fields(annotation):
        if not f.init:
            continue
        item = {"name": f.name, "required": f.default is MISSING and f.default_factory is MISSING}
        if f.default is not MISSING:
            try:
                item["default"] = to_jsonable(f.default)
            except TypeError:
                item["default"] = repr(f.default)
        nested = _config_fields(hints.get(f.name), depth + 1)
        if nested is not None:
            item["fields"] = nested
        result.append(item)
    return result


def describe_intervention(
    release_id: str, *, registry: AlgorithmRegistry | None = None
) -> dict[str, Any]:
    """Inspect a constructor without creating a runtime or making model calls."""
    registry = registry or AlgorithmRegistry.default()
    if release_id == "baseline.no_op":
        factory = NoOpIntervention
        metadata = {"id": release_id, "execution_kind": "input_patch"}
    else:
        _, release = registry.get_release(release_id)
        metadata = release.to_dict()
        if not release.entrypoint_available:
            return {**metadata, "parameters": [], "blocker": release.blocker}
        factory = registry.resolve_entrypoint(release_id)
    try:
        hints = get_type_hints(factory if is_dataclass(factory) else factory.__init__)
    except (NameError, TypeError, AttributeError):
        hints = {}
    parameters = []
    for name, param in inspect.signature(factory).parameters.items():
        if name == "release_id":
            continue  # Bound by the registry, not chosen by the caller.
        item = {
            "name": name,
            "kind": param.kind.name.lower(),
            "required": param.default is inspect.Parameter.empty
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD),
            "annotation": (
                None if param.annotation is inspect.Parameter.empty
                else inspect.formatannotation(param.annotation)
            ),
        }
        if param.default is not inspect.Parameter.empty:
            default = param.default
            item["default"] = (
                default if default is None or isinstance(default, (str, int, float, bool))
                else repr(default)
            )
        nested = _config_fields(hints.get(name))
        if nested is not None:
            item["fields"] = nested
        parameters.append(item)
    return {**metadata, "parameters": parameters, "help": inspect.getdoc(factory)}


def create_intervention(release_id: str, **parameters: Any) -> Any:
    """Instantiate a registered profile; optional runtime assets remain explicit.

    Parameters use the adapter's original Python signature. This does not turn
    composite agents into input-patch methods or assert benchmark eligibility.
    """
    factory = (
        NoOpIntervention if release_id == "baseline.no_op"
        else AlgorithmRegistry.default().resolve_entrypoint(release_id)
    )
    signature = inspect.signature(factory)
    if "release_id" in signature.parameters:
        parameters["release_id"] = release_id
    try:
        signature.bind(**parameters)
    except TypeError as exc:
        raise ValidationError(
            f"invalid parameters for {release_id}: {exc}; "
            f"inspect with describe_intervention({release_id!r}) or "
            f"agentinputlab show {release_id} --parameters"
        ) from None
    result = factory(**parameters)
    actual = getattr(result, "release_id", release_id)
    if actual != release_id:
        raise ValidationError(
            f"registered profile {release_id} constructed a different release: {actual}"
        )
    return result


def run_intervention(
    intervention: Any, state: FrozenAgentState, context: InterventionContext | None = None
) -> dict[str, Any]:
    """Run one patch intervention, preserving the native trace, budget and errors.

    A successful patch is NOT a benchmark score. Continue the patched state in
    the benchmark's native runtime and official scorer to obtain an endpoint.
    """
    record = InterventionRunner().run(intervention, state, context or InterventionContext())
    return {
        "record": to_jsonable(record),
        "state": to_jsonable(record.patch.apply(state)),
        "scored": False,
        "benchmark_score": None,
    }


def run_composite_intervention(
    intervention: Any, state: FrozenAgentState, context: InterventionContext | None = None
) -> dict[str, Any]:
    """Run an answer-producing architecture with observed calls and cost.

    The result is not a state patch or a benchmark score. The independent
    benchmark provider/scorer still evaluates the returned final answer.
    """
    record = CompositeInterventionRunner().run(
        intervention, state, context or InterventionContext()
    )
    return {
        "record": to_jsonable(record),
        "final_answer": record.result.final_answer,
        "evidence": to_jsonable(record.result.evidence),
        "scored": False,
        "benchmark_score": None,
    }


def catalog() -> dict[str, Any]:
    registry = AlgorithmRegistry.default()
    return {
        "methods": [m.to_dict() for m in registry.list_methods()],
        "benchmarks": [to_jsonable(b) for b in BenchmarkRegistry.default().list()],
        "note": "Inventory is not a claim that every method/dataset/model combination is supported or tested.",
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _state(value: Any) -> FrozenAgentState:
    if not isinstance(value, dict):
        raise ValidationError("state must be an object")
    allowed = {
        "query",
        "task_id",
        "instruction",
        "context",
        "messages",
        "tool_names",
        "metadata",
        "budget",
    }
    if value.keys() - allowed:
        raise ValidationError("unknown state fields")
    for key in ("query", "task_id", "instruction"):
        if key in value and not isinstance(value[key], str):
            raise ValidationError(f"{key} must be a string")
    for key in ("context", "messages", "tool_names"):
        if key in value and not isinstance(value[key], list):
            raise ValidationError(f"{key} must be an array")
    for key in ("metadata", "budget"):
        if key in value and not isinstance(value[key], dict):
            raise ValidationError(f"{key} must be an object")
    try:
        return FrozenAgentState(
            **{
                **value,
                "context": tuple(ContextBlock(**x) for x in value.get("context", [])),
                "messages": tuple(Message(**x) for x in value.get("messages", [])),
                "budget": ExecutionBudget(**value.get("budget", {})),
            }
        )
    except (TypeError, AttributeError) as exc:
        raise ValidationError("invalid nested state fields") from exc


class ToolkitService:
    """Operator-owned factories and contexts; clients cannot inject Python code.

    All Python profiles remain available through create_intervention. HTTP only
    exposes explicitly configured patch profiles. Construct a fresh context per
    request; do not share a mutable budget ledger between calls.
    """

    def __init__(
        self,
        factories: Mapping[str, Callable[[], Any]] | None = None,
        context_factory: Callable[[int], InterventionContext] | None = None,
    ):
        self.factories = dict(
            factories
            if factories is not None
            else {
                "baseline.no_op": NoOpIntervention,
                "plan.plan_and_solve.ps_prompt": lambda: create_intervention(
                    "plan.plan_and_solve.ps_prompt"
                ),
            }
        )
        self.context_factory = context_factory or (lambda seed: InterventionContext(seed=seed))

    def execute(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict) or body.keys() - {"release_id", "state", "seed"}:
            raise ValidationError("expected release_id, state and optional seed")
        rid = body.get("release_id")
        if not isinstance(rid, str):
            raise ValidationError("release_id must be a string")
        if rid not in self.factories:
            raise AlgorithmUnavailableError("profile is not configured on this server")
        seed = body.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValidationError("seed must be a non-negative integer")
        state = _state(body.get("state"))
        return run_intervention(self.factories[rid](), state, self.context_factory(seed))


def make_server(service: ToolkitService, *, token: str, port: int = 8765) -> ThreadingHTTPServer:
    """Authenticated loopback-only development service, not a public web host.

    Bounded request bodies and one active execution. Put a production ASGI
    deployment with TLS, quotas and job persistence in front for shared use.
    Client timeouts do not cancel running model calls; there is no auto retry.
    """
    if (
        not isinstance(token, str) or len(token) < 24
        or any(not 33 <= ord(char) <= 126 for char in token)
    ):
        raise ValueError("API token must be at least 24 printable ASCII characters without whitespace")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_args):
            pass  # Never log bearer tokens, user prompts or upstream errors.

        def send_json(self, status: int, value: Any):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            given = self.headers.get("Authorization", "").encode()
            if not hmac.compare_digest(given, ("Bearer " + token).encode()):
                self.send_json(
                    401,
                    {"error": {"code": "unauthorized", "message": "A valid API token is required"}},
                )
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/v1/catalog":
                self.send_json(
                    200, {"data": {**catalog(), "configured_releases": sorted(service.factories)}}
                )
            elif self.path == "/v1/health":
                self.send_json(200, {"data": {"status": "ok", "protocol": "agentinputlab.v1"}})
            else:
                self.send_json(404, {"error": {"code": "not_found", "message": "Unknown resource"}})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/interventions":
                self.send_json(404, {"error": {"code": "not_found", "message": "Unknown resource"}})
                return
            if "Transfer-Encoding" in self.headers or len(self.headers.get_all("Content-Length", [])) > 1:
                self.send_json(
                    400,
                    {
                        "error": {
                            "code": "invalid_body",
                            "message": "Transfer-Encoding and repeated Content-Length are unsupported",
                        }
                    },
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0 or length > 1024 * 1024:
                self.send_json(
                    413 if length > 0 else 411,
                    {
                        "error": {
                            "code": "body_size",
                            "message": "A body of at most 1 MiB is required",
                        }
                    },
                )
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(
                    415, {"error": {"code": "content_type", "message": "Use application/json"}}
                )
                return
            if not lock.acquire(blocking=False):
                self.send_json(
                    429,
                    {
                        "error": {
                            "code": "busy",
                            "message": "An execution is already active; no attempt started",
                        }
                    },
                )
                return
            try:
                raw_body = self.rfile.read(length)
                if len(raw_body) != length:
                    raise ValueError("incomplete request body")
                body = json.loads(
                    raw_body,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
                    parse_float=_finite_json_float,
                    object_pairs_hook=_unique_json_object,
                )
                result = service.execute(body)
                self.send_json(200, {"data": result})
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                # Output errors inherit ValueError through ValidationError.
                code = (
                    "algorithm_output"
                    if isinstance(exc, AlgorithmOutputError)
                    else "invalid_request"
                )
                self.send_json(
                    422 if code == "algorithm_output" else 400,
                    {
                        "error": {
                            "code": code,
                            "message": "Invalid output"
                            if code == "algorithm_output"
                            else "Invalid request schema",
                        }
                    },
                )
            except (AlgorithmUnavailableError, CapabilityError, RegistryError):
                self.send_json(
                    422,
                    {
                        "error": {
                            "code": "unavailable",
                            "message": "Required profile dependencies are not configured",
                        }
                    },
                )
            except BudgetExceededError:
                self.send_json(
                    422,
                    {
                        "error": {
                            "code": "budget_exhausted",
                            "message": "Execution budget exhausted",
                        }
                    },
                )
            except AgentInputLabError:
                self.send_json(
                    422,
                    {
                        "error": {
                            "code": "execution_failed",
                            "message": "Execution rejected; no score was fabricated",
                        }
                    },
                )
            except Exception:
                self.send_json(
                    500,
                    {
                        "error": {
                            "code": "internal_error",
                            "message": "Execution failed; inspect the operator environment",
                        }
                    },
                )
            finally:
                lock.release()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Authenticated loopback AgentInputLab SDK service")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = make_server(
            ToolkitService(), token=os.environ.get("AGENTINPUTLAB_API_TOKEN", ""), port=args.port
        )
    except ValueError as exc:
        parser.error(str(exc) + "; set AGENTINPUTLAB_API_TOKEN before starting the server")
    except OSError:
        parser.error("cannot bind the loopback port; choose an available --port")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
