import asyncio
import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_llmcompiler_worker_driver import fake_upstream, prompts

from agentinputlab.backends import GenerationResponse, Usage
from agentinputlab.connectors.tools import ToolResult, ToolSpec
from agentinputlab.contracts import ExecutionBudget, ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.integrations.external_workers.llmcompiler_live import (
    _Callbacks,
    _server,
    _strict_json,
)
from agentinputlab.integrations.external_workers.method import MethodWorkerRequest
from workers.llmcompiler import driver, live


class Backend:
    def __init__(self, *, failure=False):
        self.calls, self.failure = [], failure

    def fingerprint(self):
        return {"model_id": "fixture", "revision": "fixed"}

    def generate(self, request):
        self.calls.append(request)
        if self.failure:
            raise RuntimeError("SECRET provider body")
        text = "PLAN:first" if len(self.calls) == 1 else "Thought: done\nAction: Finish(done)"
        return GenerationResponse(text, usage=Usage(5, 3))


class Tools:
    def __init__(self, *, failure=False):
        self.calls, self.failure = [], failure

    def list_tools(self):
        return (ToolSpec("echo", "echo(text: string) -> string", {"type": "object",
            "properties": {"text": {"type": "string"}}, "required": ["text"],
            "additionalProperties": False}),)

    def fingerprint(self):
        return {"kind": "fixture_readonly"}

    def call(self, call):
        self.calls.append(call)
        return ToolResult(call.call_id, str(call.arguments["text"]).upper(),
                          error="failed" if self.failure else "")


def request(backend, tools, *, budget=None, streaming=False):
    return MethodWorkerRequest(method_id="llm_compiler", release_id="plan.llm_compiler.full",
        upstream_commit=driver.UPSTREAM_COMMIT, state_hash="a" * 64,
        payload={"task_id": "fixture", "query": "question", "instruction": "", "messages": [],
            "tool_specs": to_jsonable(tools.list_tools()), "seed": 7,
            "controls": {"max_tasks": 4, "max_replans": 1, "concurrency": 2,
                         "timeout_seconds": 5, "streaming": streaming},
            "execution_mode": "full", "prompt_hash": driver._canonical_hash(prompts()),
            "execution_budget": asdict(budget or ExecutionBudget())},
        dependency_fingerprints={"backend": backend.fingerprint(), "tools": tools.fingerprint()},
        minimum_reservation={"model_calls": 1}, protocol_version=driver.PROTOCOL_VERSION).as_dict()


def callbacks(backend=None, tools=None, budget=None):
    backend, tools = backend or Backend(), tools or Tools()
    owner = SimpleNamespace(backend=backend, tools=tools, per_call_output_cap=512, argument_order={},
        fingerprint=lambda: {"worker_attestation": {"schema": "fixture-attestation"}})
    host = _Callbacks(owner, request(backend, tools, budget=budget))
    host.attested = True  # Individual callback tests; handshake tested separately below.
    return host, owner


@pytest.mark.parametrize("streaming", [False, True])
def test_live_source_loop_uses_real_host_callbacks_instead_of_response_or_tool_replay(monkeypatch, streaming):
    host, owner = callbacks()
    raw = request(owner.backend, owner.tools, streaming=streaming)
    host.request = raw
    server = _server(host, "fixture-secret", 4 * 1024 * 1024)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("FIXTURE_CALLBACK_KEY", "fixture-secret")
    try:
        validated = driver.validate_request(raw)
        callback = live.Callback(f"http://127.0.0.1:{server.server_port}/callback", "FIXTURE_CALLBACK_KEY", validated)
        upstream, upstream_calls = fake_upstream()
        result = asyncio.run(driver.execute_request(validated, upstream,
            driver.ReplayBundle(prompts(), ()), live.LiveTools(callback), runtime_model=live.LiveModel(callback)))
        assert result["status"] == "ok" and result["payload"]["final_answer"] == "done"
        assert len(owner.backend.calls) == 2 and len(owner.tools.calls) == 1
        assert host.usage()["model_calls"] == 2 and host.usage()["output_tokens"] == 6
        assert host.usage()["unpriced_model_calls"] == 2
        assert result["usage"]["model_calls"] == 2 and result["usage"]["tool_calls"] == 1
        assert all(r.max_tokens == 512 and r.seed == 7 for r in owner.backend.calls)
        assert result["payload"]["scheduler_events"]
        assert upstream_calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("budget", [ExecutionBudget(max_model_calls=0), ExecutionBudget(max_output_tokens=511)])
def test_model_budget_denies_before_any_provider_call(budget):
    host, owner = callbacks(budget=budget)
    with pytest.raises(ValueError, match="budget exhausted"):
        host.model({"prompt": "q", "channel": "planner", "stop": [], "seed": 7})
    assert owner.backend.calls == [] and host.usage()["model_calls"] == 0


def test_unknown_consumed_model_failure_is_not_refunded_or_retired_as_zero_cost():
    host, owner = callbacks(backend=Backend(failure=True))
    with pytest.raises(RuntimeError) as error:
        host.model({"prompt": "q", "channel": "planner", "stop": [], "seed": 7})
    assert "SECRET" not in str(error.value) and error.value.__cause__ is None
    assert host.unknown_usage_calls == 1 and host.reserved_output == 512
    assert host.usage()["model_calls"] == host.usage()["unpriced_model_calls"] == 1
    with pytest.raises(ValueError):
        host.model({"prompt": "q", "channel": "planner", "stop": [], "seed": 7})
    assert len(owner.backend.calls) == 1


def test_schema_unknown_tools_duplicates_and_tool_budget_are_pre_dispatch_failures():
    host, owner = callbacks(budget=ExecutionBudget(max_tool_calls=1))
    with pytest.raises(ValueError):
        host.tool({"name": "shell", "arguments": ["secret"]})
    with pytest.raises(ValidationError):
        host.tool({"name": "echo", "arguments": {"text": 3}})
    assert host.tool({"name": "echo", "arguments": ["first"]}) == {"output": "FIRST"}
    with pytest.raises(ValueError, match="budget exhausted"):
        host.tool({"name": "echo", "arguments": ["second"]})
    assert len(owner.tools.calls) == 1
    row = {"request_id": host.request["request_id"], "call_id": "duplicate", "kind": "bad", "payload": {}}
    with pytest.raises(ValueError):
        host.dispatch(row)
    with pytest.raises(ValueError, match="duplicate"):
        host.dispatch({**row, "kind": "model"})


@pytest.mark.parametrize("change", ["model", "tools", "schema", "cost", "context"])
def test_changed_bindings_and_unsupported_strict_budget_fail_closed(change):
    host, owner = callbacks()
    raw = request(owner.backend, owner.tools)
    if change in {"model", "tools"}:
        raw["dependency_fingerprints"]["backend" if change == "model" else "tools"] = {"other": True}
    elif change == "schema":
        raw["payload"]["tool_specs"][0]["network"] = True
    else:
        raw["payload"]["execution_budget"]["max_cost_usd" if change == "cost" else "max_context_tokens"] = 1
    with pytest.raises(ValidationError):
        _Callbacks(owner, raw)
    assert owner.backend.calls == [] and owner.tools.calls == []


@pytest.mark.parametrize("url", ["http://localhost:1234/callback", "http://example.com/callback",
    "http://127.0.0.1:1234/other", "http://user@127.0.0.1:1234/callback"])
def test_worker_cannot_select_a_non_loopback_callback_target(monkeypatch, url):
    monkeypatch.setenv("FIXTURE_CALLBACK_KEY", "fixture-secret")
    with pytest.raises(driver.WorkerFailure):
        live.Callback(url, "FIXTURE_CALLBACK_KEY", SimpleNamespace())


def test_strict_callback_json_and_failed_tools_preserve_failures():
    with pytest.raises(ValueError):
        _strict_json('{"x": 1, "x": 2}')
    with pytest.raises(ValueError):
        _strict_json('{"x": NaN}')
    host, owner = callbacks(tools=Tools(failure=True))
    with pytest.raises(RuntimeError, match="no synthetic"):
        host.tool({"name": "echo", "arguments": ["first"]})
    assert host.usage()["tool_calls"] == 1 and host.fatal
    assert len(owner.tools.calls) == 1


def test_worker_api_response_empty_final_retains_native_usage():
    class Callback:
        request = SimpleNamespace(seed=17)

        async def call(self, *args):
            return {"text": "", "input_tokens": 4, "output_tokens": 512, "finish_reason": "length"}

    model = live.LiveModel(Callback())
    with pytest.raises(driver.WorkerFailure, match="empty final"):
        asyncio.run(model.apredict("question"))
    assert model.usage()["model_calls"] == 1 and model.usage()["output_tokens"] == 512


def test_source_version_handshake_is_required_before_any_model_or_tool_dispatch():
    host, owner = callbacks()
    host.attested = False
    def envelope(kind, payload, identifier):
        return {"request_id": host.request["request_id"], "call_id": identifier, "kind": kind, "payload": payload}
    with pytest.raises(ValueError, match="attest"):
        host.dispatch(envelope("model", {}, "before-attest"))
    with pytest.raises(ValueError, match="differs"):
        host.dispatch(envelope("attest", {"schema": "wrong"}, "wrong-attest"))
    assert host.dispatch(envelope("attest", {"schema": "fixture-attestation"}, "right-attest")) == {"attested": True}
    assert owner.backend.calls == [] and owner.tools.calls == []
    assert host.events == [{"kind": "attestation", "matched": True}]


def test_explicit_positional_order_is_not_replaced_by_json_sorted_property_order():
    host, owner = callbacks()
    spec = ToolSpec("mix", "mix(z: string, a: string)", {"type": "object", "properties": {
        "a": {"type": "string"}, "z": {"type": "string"}}, "required": ["z"], "additionalProperties": False})
    owner.argument_order = {"mix": ("z", "a")}
    host.argument_order = dict(owner.argument_order)
    host.specs = {"mix": spec}
    calls = []
    owner.tools.call = lambda call: (calls.append(call) or ToolResult(call.call_id, output="done"))
    assert host.tool({"name": "mix", "arguments": ["Z", "A"]}) == {"output": "done"}
    assert dict(calls[0].arguments) == {"z": "Z", "a": "A"}
    recorder = driver.TraceRecorder(frozenset({"mix"}), 4, 1, argument_order=owner.argument_order)
    recorder.tasks[1] = SimpleNamespace(name="mix", args=("Z", "A"), dependencies=())
    assert recorder.task_rows(to_jsonable((spec,)))[0]["arguments"] == {"z": "Z", "a": "A"}
    assert host.tool({"name": "mix", "arguments": ["Z"]}) == {"output": "done"}
    assert dict(calls[1].arguments) == {"z": "Z"}
