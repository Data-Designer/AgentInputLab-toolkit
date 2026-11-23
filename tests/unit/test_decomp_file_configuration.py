"""Public native DecomP binding regressions; not model acceptance tests."""

import json
import sys
from dataclasses import replace

import pytest

from agentinputlab import FrozenAgentState, InterventionContext, RunConfig
from agentinputlab.contracts.errors import ValidationError


def configuration(tmp_path, profile="full"):
    return RunConfig(f"plan.decomp.{profile}",
        parameters={"config_artifact_id": "decomp.commaqa.coarse.p1"},
        backend={"kind": "decomp_completion", "model_id": "test-model",
                 "base_url": "http://127.0.0.1:1/v1",
                 "deployment_identity": {"revision": "test"}},
        runtime={"kind": "decomp_live", "checkout": str(tmp_path),
                 "python_executable": sys.executable, "nltk_data": str(tmp_path),
                 "per_call_output_cap": 512, "per_call_input_token_bound": 40960})


@pytest.mark.parametrize("profile", ["full", "planner_component"])
def test_native_binding_from_file_without_python_injection(tmp_path, profile):
    from agentinputlab.integrations.external_workers.decomp_live import DecompLiveRuntime
    from agentinputlab.integrations.external_workers.decomp_planner import DecompPlannerLiveRuntime

    config = configuration(tmp_path, profile)
    assert config.validate()["valid"]
    context = config._configured_context(None)
    bound = config._bind_case_context(context, FrozenAgentState(query="q"))
    expected = DecompLiveRuntime if profile == "full" else DecompPlannerLiveRuntime
    assert type(bound.external_runtime) is expected
    assert bound.backend is bound.external_runtime.backend
    assert bound.external_runtime.per_call_input_token_bound == 40960
    assert bound.external_runtime.per_call_output_cap == 512


@pytest.mark.parametrize("key,value", [("shell", "false"), ("per_call_output_cap", 0),
    ("per_call_input_token_bound", True), ("timeout_seconds", -1), ("nltk_data", "")])
def test_invalid_runtime_shape(tmp_path, key, value):
    config = configuration(tmp_path)
    with pytest.raises(ValidationError):
        replace(config, runtime={**config.runtime, key: value}).validate()


def test_no_silent_chat_conversion_or_final_channel_stop(tmp_path):
    config = configuration(tmp_path)
    for backend in ({**config.backend, "kind": "openai_compatible"},
                    {**config.backend, "stop_policy": "final_content"},
                    {**config.backend, "extra_body": {"temperature": 1}}):
        with pytest.raises(ValidationError):
            replace(config, backend=backend).validate()
    with pytest.raises(ValidationError):
        RunConfig("baseline.no_op", backend=config.backend, runtime=config.runtime).validate()
    from agentinputlab.benchmark_configuration import _CapturedBackend
    from tests.unit.test_benchmark_configuration import Backend

    assert not hasattr(_CapturedBackend(Backend([])), "complete_native")


def test_missing_worker_and_double_binding(tmp_path):
    config = configuration(tmp_path)
    context = config._configured_context(None)
    with pytest.raises(ValidationError, match="configured twice"):
        config._configured_context(InterventionContext(external_runtime=object()))
    bad = replace(config, runtime={**config.runtime, "python_executable": str(tmp_path / "gone")})
    assert bad.validate()["valid"]
    with pytest.raises(ValidationError, match="worker Python"):
        bad._bind_case_context(context, FrozenAgentState(query="q"))


def test_relative_nltk_and_worker_paths(tmp_path):
    config = configuration(tmp_path).to_dict()
    config["runtime"].update(nltk_data="nltk", checkout="source", python_executable="worker/bin/python")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    loaded = RunConfig.from_file(path)
    assert loaded.runtime["nltk_data"] == str(tmp_path / "nltk")
    assert loaded.runtime["checkout"] == str(tmp_path / "source")


def test_capture_native_parameters_and_responses_without_retry(tmp_path):
    from agentinputlab.benchmark_configuration import _CapturedBackend

    backend = configuration(tmp_path)._backend()
    response = {"model": "test-model", "choices": [{"text": "answer", "logprobs": {
        "tokens": ["answer"], "token_logprobs": [-0.3]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    calls = []

    def transport(url, headers, body, timeout):
        calls.append((url, body))
        return response

    backend.transport = transport
    captured = _CapturedBackend(backend)
    parameters = dict(prompt="native prompt", engine="original", temperature=0, max_tokens=100,
        top_p=1, frequency_penalty=0, presence_penalty=0, stop=["END"], n=1, logprobs=1)
    assert captured.complete_native(parameters, seed=17) == response
    assert len(calls) == 1 and calls[0][0].endswith("/completions")
    assert calls[0][1]["stop"] == ["END"] and calls[0][1]["max_tokens"] == 100
    assert captured.calls[0]["response"]["choices"] == response["choices"]

    def failed(*args):
        calls.append("failed")
        raise RuntimeError("fixture")

    backend.transport = failed
    with pytest.raises(RuntimeError):
        captured.complete_native(parameters, seed=17)
    assert len(calls) == 2 and captured.calls[-1]["status"] == "failed"


def test_evaluate_binds_case_capture_without_dropping_instructions(tmp_path, monkeypatch):
    from agentinputlab.benchmark_configuration import _CapturedBackend
    from tests.unit.test_benchmark_configuration import evaluate

    config = configuration(tmp_path)
    seen = []

    def check(self, context, state):
        from agentinputlab.decomp_configuration import bind_case

        bound = bind_case(self, context, state)
        assert isinstance(bound.external_runtime.backend, _CapturedBackend)
        seen.append(bound)
        raise ValidationError("fixture: stop before subprocess")

    monkeypatch.setattr(RunConfig, "_bind_case_context", check)
    backend = config._backend()
    result = evaluate(replace(config, backend={}), backend)
    assert result["failed_count"] == result["selected_count"] == len(seen) == 2
    assert seen[0].external_runtime is not seen[1].external_runtime
