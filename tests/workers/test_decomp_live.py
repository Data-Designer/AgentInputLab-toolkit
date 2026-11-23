"""Native completion boundary tests; never substitute these for live acceptance."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab.algorithms.decompose.decomp import (
    DecompFullBridge,
    DecompWorkerRequest,
    decomp_context_hash,
)
from agentinputlab.api import create_intervention
from agentinputlab.backends.llm.decomp_completion import DecompCompletionBackend
from agentinputlab.contracts import (
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.integrations.external_workers.decomp_live import _Callbacks
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import CompositeInterventionRunner
from workers.decomp import driver
from workers.decomp.live import failure_response


def parameters():
    return {"prompt": "native prompt", "engine": "text-davinci-002", "temperature": 0,
        "max_tokens": 100, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0,
        "stop": ["\n"], "n": 1, "best_of": 1, "logprobs": 0}


def response():
    return {"model": "bound-model", "choices": [{"text": "native answer", "logprobs": {
        "tokens": ["native", " answer"], "token_logprobs": [-0.1, -0.3]}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2}}


def backend(transport=None):
    return DecompCompletionBackend(model_id="bound-model", base_url="http://127.0.0.1:8000/v1",
        deployment_identity={"revision": "fixed"}, transport=transport or (lambda *_: response()))


def test_native_backend_preserves_parameters_and_raw_choice_logprobs_without_chat_template():
    seen = []
    model = backend(lambda *args: seen.append(args) or response())
    result = model.complete_native(parameters(), seed=17)
    assert result == response()
    url, _headers, payload, _timeout = seen[0]
    assert url.endswith("/v1/completions")
    assert payload == {**{k:v for k,v in parameters().items() if k != "engine"}, "model":"bound-model", "seed":17}
    assert "messages" not in payload
    assert model.fingerprint()["automatic_retry"] is False


@pytest.mark.parametrize("changed", [{"n": 2}, {"best_of": 2}, {"max_tokens": True},
    {"max_tokens": 0}, {"logprobs": True}, {"top_p": 0}, {"temperature": float("nan")},
    {"stop": "\n"}, {"api_key": "secret"}, {"host": "http://arbitrary"}])
def test_native_unsupported_parameters_fail_before_transport(changed):
    calls = []
    model = backend(lambda *args: calls.append(args) or response())
    with pytest.raises(ValidationError):
        model.complete_native({**parameters(), **changed}, seed=17)
    assert calls == []


def test_native_transport_error_is_sanitized_no_retry_or_secret_exception_chain():
    calls = []
    def failed(*args):
        calls.append(args)
        raise ValueError("opaque provider body contains SECRET")
    with pytest.raises(RuntimeError) as error:
        backend(failed).complete_native(parameters(), seed=17)
    assert len(calls) == 1
    assert "SECRET" not in str(error.value)
    assert error.value.__cause__ is None and error.value.__context__ is None


def callbacks(*, budget=None, transport=None, input_bound=None):
    model = backend(transport)
    owner = SimpleNamespace(backend=model, per_call_output_cap=512,
                            per_call_input_token_bound=input_bound)
    request = DecompWorkerRequest("case", "question", (), (), decomp_context_hash((),()),
        "decomp.commaqa.coarse.p1", model.fingerprint(), 20,
        execution_budget=budget or ExecutionBudget(), seed=17)
    identity = {"worker_attestation": {"fixed":"source"}}
    host = _Callbacks(owner, request, identity)
    return host, owner


def envelope(host, kind, payload, call_id):
    return {"request_id":host.request_id,"call_id":call_id,"kind":kind,"payload":payload}


def test_native_source_attestation_is_required_and_unique_before_completion():
    host, _owner = callbacks()
    with pytest.raises(ValueError, match="attest"):
        host.dispatch(envelope(host, "model", parameters(), "early"))
    with pytest.raises(ValueError, match="differs"):
        host.dispatch(envelope(host, "attest", {"fixed":"wrong"}, "bad"))
    assert host.dispatch(envelope(host, "attest", {"fixed":"source"}, "right")) == {"attested":True}
    assert host.dispatch(envelope(host, "model", parameters(), "model")) == response()
    assert host.usage() == {"model_calls":1,"input_tokens":7,"output_tokens":2,
                            "retrieval_calls":0,"unpriced_model_calls":1}
    with pytest.raises(ValueError, match="duplicate"):
        host.dispatch(envelope(host, "model", parameters(), "model"))
    assert host.calls == 1


@pytest.mark.parametrize("budget", [ExecutionBudget(max_model_calls=0), ExecutionBudget(max_output_tokens=99)])
def test_native_budget_rejection_neither_shrinks_config_cap_nor_dispatches(budget):
    seen = []
    host, _ = callbacks(budget=budget, transport=lambda *a: seen.append(a) or response())
    host.attested = True
    with pytest.raises(ValueError, match="budget"):
        host.dispatch(envelope(host,"model",parameters(),"first"))
    assert seen == [] and host.calls == 0 and parameters()["max_tokens"] == 100


def test_native_unknown_consumption_keeps_reservation_and_cannot_retry():
    def failed(*_):
        raise TimeoutError("secret wire data")
    host, _ = callbacks(transport=failed)
    host.attested = True
    with pytest.raises(RuntimeError):
        host.dispatch(envelope(host,"model",parameters(),"first"))
    assert host.reserved == 100 and host.unknown_usage_calls == 1
    assert host.usage()["model_calls"] == 1 and host.usage()["unpriced_model_calls"] == 1
    with pytest.raises(ValueError):
        host.dispatch(envelope(host,"model",parameters(),"another"))


@pytest.mark.parametrize("mutation", ["missing_usage", "overspend", "model", "logs"])
def test_bad_native_response_fails_closed_and_retains_real_cost(mutation):
    raw = response()
    if mutation == "missing_usage":
        del raw["usage"]
    elif mutation == "overspend":
        raw["usage"]["completion_tokens"] = 101
    elif mutation == "model":
        raw["model"] = "other-model"
    elif mutation == "logs":
        raw["choices"][0]["logprobs"]["token_logprobs"] = [None]
    host, _ = callbacks(transport=lambda *_: raw)
    host.attested = True
    with pytest.raises(RuntimeError):
        host.dispatch(envelope(host,"model",parameters(),"first"))
    assert host.fatal and host.calls == 1
    assert host.unknown_usage_calls == int(mutation == "missing_usage")
    if mutation == "overspend":
        assert host.output_tokens == 101


def test_live_context_binding_and_unattested_input_cost_budgets_are_not_ignored():
    host, owner = callbacks()
    with pytest.raises(ValidationError, match="differs"):
        _Callbacks(owner, replace(host.request, backend_binding={"other":"model"}), host.identity)
    for budget in (ExecutionBudget(max_context_tokens=100), ExecutionBudget(max_cost_usd=1)):
        with pytest.raises(ValidationError, match="ceilings"):
            _Callbacks(owner, replace(host.request,execution_budget=budget),host.identity)


def test_native_input_upper_bound_reserves_then_settles_actual_cumulative_usage():
    seen = []
    host, _ = callbacks(budget=ExecutionBudget(max_context_tokens=14), input_bound=7,
                        transport=lambda *a: seen.append(a) or response())
    host.attested = True
    for index in range(2):
        host.dispatch(envelope(host, "model", parameters(), str(index)))
    assert host.input_tokens == 14 and host.input_reserved == 0
    assert host.calls == len(seen) == 2
    with pytest.raises(ValueError, match="budget"):
        host.dispatch(envelope(host, "model", parameters(), "third"))
    assert host.calls == len(seen) == 2
    assert all(call[2]["max_tokens"] == 100 for call in seen)


def test_input_reservations_include_calls_currently_in_flight():
    seen = []
    host = None
    def transport(*args):
        seen.append(args)
        assert host.input_reserved == 7
        with pytest.raises(ValueError, match="budget"):
            host.dispatch(envelope(host, "model", parameters(), "overlapping"))
        return response()
    host, _ = callbacks(budget=ExecutionBudget(max_context_tokens=13),
                        input_bound=7, transport=transport)
    host.attested = True
    host.dispatch(envelope(host, "model", parameters(), "first"))
    assert host.calls == len(seen) == 1 and host.input_reserved == 0


def test_unknown_native_input_keeps_reservation_and_blocks_retry():
    def failed(*_):
        raise TimeoutError("opaque provider error")
    host, _ = callbacks(budget=ExecutionBudget(max_context_tokens=14),
                        input_bound=7, transport=failed)
    host.attested = True
    with pytest.raises(RuntimeError):
        host.dispatch(envelope(host, "model", parameters(), "first"))
    assert host.calls == 1 and host.input_tokens == 0 and host.input_reserved == 7
    assert host.unknown_usage_calls == 1 and host.fatal
    with pytest.raises(ValueError):
        host.dispatch(envelope(host, "model", parameters(), "not-a-retry"))
    assert host.calls == 1


def test_false_native_input_bound_fails_closed_without_hiding_consumption():
    host, _ = callbacks(budget=ExecutionBudget(max_context_tokens=14), input_bound=6)
    host.attested = True
    with pytest.raises(RuntimeError):
        host.dispatch(envelope(host, "model", parameters(), "first"))
    assert host.input_tokens == 7 and host.output_tokens == 2
    assert host.input_reserved == 0 and host.unknown_usage_calls == 0
    assert host.fatal and host.calls == 1


@pytest.mark.parametrize("bound", [True, 0, -1, 1.5, "7"])
def test_invalid_native_input_bound_rejected_before_dispatch(bound):
    with pytest.raises(ValidationError, match="input token bound"):
        callbacks(budget=ExecutionBudget(max_context_tokens=14), input_bound=bound)


def test_native_generator_binding_changes_only_provider_seam(monkeypatch, tmp_path):
    source = tmp_path/"src/decomp/models/gpt3generator.py"
    source.parent.mkdir(parents=True)
    source.touch()
    original_generate = object()
    module = SimpleNamespace(__file__=str(source), GPT3Generator=SimpleNamespace(generate_text_sequence=original_generate))
    monkeypatch.setattr(driver.importlib,"import_module",lambda _:module)
    checkout = driver.Checkout(tmp_path,{"source_files":{"src/decomp/models/gpt3generator.py":"fixed"}},"fixed")
    def completed(**_):
        return response()
    driver._bind_native_completion(checkout,{"models":{"gen":{"name":"llmqadecomp","gen_model":"gpt3"}}},completed)
    assert module.openai_call is completed and module.openai.Completion.create is completed
    assert module.GPT3Generator.generate_text_sequence is original_generate
    with pytest.raises(driver.DriverError,match="gpt3"):
        driver._bind_native_completion(checkout,{"models":{"gen":{"name":"llmqadecomp","gen_model":"lm"}}},completed)


def test_controlled_native_runtime_rejects_implicit_instruction_history_projection():
    runtime = SimpleNamespace(fingerprint=lambda:{"backend_binding_mode":"controlled_native_completion_callbacks.v1"})
    method = DecompFullBridge(runtime,config_artifact_id="decomp.commaqa.coarse.p1")
    with pytest.raises(ValidationError,match="explicitly project"):
        method.run(FrozenAgentState(query="question",instruction="nonempty"),InterventionContext())


@pytest.mark.parametrize("consumed_calls", [0, 3])
def test_failed_native_runtime_settles_actual_usage_not_one_reserved_call(consumed_calls):
    class FailedRuntime:
        def fingerprint(self):
            return {"backend_binding_mode":"controlled_native_completion_callbacks.v1"}
        def run(self, request):
            assert request.execution_budget.max_model_calls == 8 and request.seed == 17
            return {"status":"error","failure_code":"FIXTURE_FAILURE","consumed":bool(consumed_calls),
                "usage":{"model_calls":consumed_calls,"retrieval_calls":0,"input_tokens":11*consumed_calls,
                         "output_tokens":7*consumed_calls,"unpriced_model_calls":consumed_calls}}
    monitor=RunMonitor()
    with pytest.raises(ExternalDependencyError,match="FIXTURE_FAILURE"):
        CompositeInterventionRunner(monitor=monitor).run(
            DecompFullBridge(None,config_artifact_id="decomp.commaqa.coarse.p1"),
            FrozenAgentState(query="question",budget=ExecutionBudget()),
            InterventionContext(backend=backend(),external_runtime=FailedRuntime(),seed=17))
    snapshot=monitor.list_snapshots()[0]
    assert snapshot.usage.model_calls == consumed_calls
    assert snapshot.usage.input_tokens == 11*consumed_calls
    assert snapshot.usage.output_tokens == 7*consumed_calls
    assert snapshot.usage.unpriced_model_calls == consumed_calls


@pytest.mark.parametrize("budget", [ExecutionBudget(max_context_tokens=10),ExecutionBudget(max_cost_usd=1)])
def test_unattested_native_budget_is_rejected_before_external_model_reservation(budget):
    class Runtime:
        def fingerprint(self):
            return {"backend_binding_mode":"controlled_native_completion_callbacks.v1"}
        def run(self, request):
            raise AssertionError("preflight must reject before native execution")
    monitor=RunMonitor()
    with pytest.raises(ValidationError,match="pre-call estimators"):
        CompositeInterventionRunner(monitor=monitor).run(
            DecompFullBridge(None,config_artifact_id="decomp.commaqa.coarse.p1"),
            FrozenAgentState(query="question",budget=budget),
            InterventionContext(backend=backend(),external_runtime=Runtime(),seed=17))
    assert monitor.list_snapshots()[0].usage.model_calls == 0


def test_public_factory_supports_context_injection_without_redundant_runtime_none_parameter():
    method=create_intervention("plan.decomp.full",config_artifact_id="decomp.commaqa.coarse.p1")
    assert isinstance(method,DecompFullBridge) and method.runtime is None


def test_source_failure_diagnostics_exclude_private_paths_and_raw_error_body(tmp_path):
    try:
        raise ValueError("SECRET provider body and credential")
    except ValueError as exc:
        failure=driver.DriverError("UPSTREAM_EXECUTION_FAILED","fixed public diagnostic")
        failure.__cause__=exc
    value=failure_response(failure,tmp_path)
    assert value["upstream_error_type"]=="ValueError" and value["upstream_source_frames"]==[]
    assert "SECRET" not in str(value) and str(tmp_path) not in str(value)


def test_native_wrapped_context_retains_only_source_locations_and_types(tmp_path):
    namespace = {}
    source = "def participant():\n    raise KeyError('PRIVATE generated output')\n"
    exec(compile(source, str(tmp_path / 'src/decomp/participant.py'), 'exec'), namespace)
    try:
        try:
            namespace['participant']()
        except KeyError:
            # Reproduce upstream's implicit wrapping, not an explicit cause.
            raise ValueError('SECRET wrapped provider response')  # noqa: B904
    except ValueError as exc:
        failure = driver.DriverError('UPSTREAM_EXECUTION_FAILED', 'fixed public diagnostic')
        failure.__cause__ = exc
    value = failure_response(failure, tmp_path)
    chain = value['upstream_error_chain']
    assert [item['error_type'] for item in chain] == ['ValueError', 'KeyError']
    assert chain[1]['source_frames'] == [
        dict(file='src/decomp/participant.py', line=2, function='participant')]
    assert all(text not in str(value) for text in ('PRIVATE', 'SECRET', str(tmp_path)))


def test_native_error_chain_respects_suppression_and_cycles(tmp_path):
    private = ValueError('PRIVATE')
    wrapped = RuntimeError('SECRET')
    wrapped.__context__ = private
    wrapped.__suppress_context__ = True
    failure = driver.DriverError('UPSTREAM_EXECUTION_FAILED', 'fixed public diagnostic')
    failure.__cause__ = wrapped
    assert len(failure_response(failure, tmp_path)['upstream_error_chain']) == 1
    wrapped.__suppress_context__ = False
    private.__context__ = wrapped
    assert len(failure_response(failure, tmp_path)['upstream_error_chain']) == 2


def test_native_error_chain_is_bounded(tmp_path):
    root = current = ValueError('PRIVATE')
    for _ in range(20):
        current.__context__ = ValueError('SECRET')
        current = current.__context__
    failure = driver.DriverError('UPSTREAM_EXECUTION_FAILED', 'fixed public diagnostic')
    failure.__cause__ = root
    assert len(failure_response(failure, tmp_path)['upstream_error_chain']) == 8
