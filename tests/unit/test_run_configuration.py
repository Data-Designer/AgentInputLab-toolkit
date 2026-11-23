"""Configuration usability checks. All model outputs here are synthetic."""

import json

import pytest

from agentinputlab import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.api import describe_intervention
from agentinputlab.backends.llm import GenerationRequest, OpenAICompatibleBackend
from agentinputlab.backends.retrieval import InMemoryBM25Retriever
from agentinputlab.cli.main import main
from agentinputlab.configuration import RunConfig
from agentinputlab.contracts.errors import BudgetExceededError, ValidationError
from agentinputlab.contracts.serialization import content_hash


def rrr_config(**changes):
    rewrite = "Rewrite {query}"
    reader = "Answer {query} using {background}"
    data = {
        "release_id": "rewrite.rrr.common_support_prompt",
        "parameters": {"config": {
            "rewrite_prompt": rewrite, "reader_prompt": reader,
            "prompt_source_sha256": "a" * 64,
            "rewrite_prompt_sha256": content_hash(rewrite),
            "reader_prompt_sha256": content_hash(reader),
            "temperature": 0.25, "max_rewriter_tokens": 32, "max_reader_tokens": 64,
            "top_k": 1,
        }},
        "budget": {"max_output_tokens": 96, "max_model_calls": 2, "max_retrieval_calls": 1},
        "retriever": {"kind": "bm25", "k1": 1.7, "b": 0.6, "documents": [
            {"doc_id": "one", "text": "Paris is the capital of France.", "corpus_revision": "synthetic"},
            {"doc_id": "two", "text": "Berlin is the capital of Germany.", "corpus_revision": "synthetic"},
        ]},
    }
    data.update(changes)
    return RunConfig.from_dict(data)


def test_config_reaches_model_and_retriever_and_records_budget():
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return {"model": "synthetic", "choices": [{"message": {"content": (
            "capital france" if len(calls) == 1 else "Paris")}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    backend = OpenAICompatibleBackend("synthetic", "http://localhost:9999/v1",
                                     deployment_identity={"revision": "synthetic"}, transport=transport)
    config = rrr_config()
    assert config.validate()["runtime_ready"] is None
    assert calls == []
    method = config.create_intervention()
    assert method.config.top_k == 1
    assert config._retriever().k1 == 1.7
    result = config.run("capital france", context=InterventionContext(backend=backend))
    assert [c["max_tokens"] for c in calls] == [32, 64]
    assert [c["temperature"] for c in calls] == [0.25, 0.25]
    # The native chain deliberately offsets the reader seed by one.
    assert [c["seed"] for c in calls] == [17, 18]
    assert result["configuration"]["budget"]["max_model_calls"] == 2
    assert result["scored"] is False
    assert "Berlin" not in calls[1]["messages"][0]["content"]


def test_budget_blocks_generation_without_hidden_retry():
    calls = []

    def transport(*args):
        calls.append(1)
        return {"choices": [{"message": {"content": "capital france"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    backend = OpenAICompatibleBackend("synthetic", "http://localhost/v1",
                                     deployment_identity={"revision": "synthetic"}, transport=transport)
    with pytest.raises(BudgetExceededError):
        rrr_config(budget={"max_model_calls": 0}).run("capital france", context=InterventionContext(backend=backend))
    assert calls == []


def test_nested_dataclass_parameters_are_discoverable():
    info = describe_intervention("rewrite.rrr.common_support_prompt")
    fields = next(p["fields"] for p in info["parameters"] if p["name"] == "config")
    assert {p["name"] for p in fields} >= {"temperature", "top_k", "max_reader_tokens"}


@pytest.mark.parametrize("key", ["budegt", "temperature", "max_length", "retriaver"])
def test_unknown_top_level_parameters_rejected(key):
    with pytest.raises(ValidationError, match="unknown configuration"):
        RunConfig.from_dict({"release_id": "baseline.no_op", key: 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_parameters_rejected(value):
    with pytest.raises(ValidationError, match="finite"):
        RunConfig("rewrite.query2doc.querygym_live", parameters={"temperature": value})


@pytest.mark.parametrize("params", [{"temperatur": 0.5}, {"max_length": 20}])
def test_method_typos_are_not_ignored(params):
    with pytest.raises(ValidationError, match="unknown parameters"):
        RunConfig("rewrite.query2doc.querygym_live", parameters=params).validate()


def test_nested_parameter_typo_rejected():
    data = rrr_config().to_dict()
    data["parameters"]["config"]["temperatur"] = 0.5
    with pytest.raises(ValidationError, match="unknown parameters.config"):
        RunConfig.from_dict(data).validate()


@pytest.mark.parametrize("body", [
    '{"release_id":"baseline.no_op","seed":1,"seed":2}',
    '{"release_id":"baseline.no_op","parameters":{"temperature":1e999}}',
    '[]', 'null', '{',
])
def test_bad_json_has_actionable_error(tmp_path, capsys, body):
    path = tmp_path / "config.json"
    path.write_text(body)
    assert main(["config", "validate", str(path)]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_config_roundtrip_cli_and_fresh_run(tmp_path, capsys):
    assert main(["config", "init"]) == 0
    raw = json.loads(capsys.readouterr().out)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    assert main(["config", "validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "configuration_only"
    assert main(["config", "run", str(path), "--query", "Find a train"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scored"] is False
    config = RunConfig.from_file(path)
    original = FrozenAgentState(query="Find a train", budget=ExecutionBudget(max_model_calls=3))
    config.run(original)
    assert original.budget.max_model_calls == 3


def test_context_conflicts_rejected():
    with pytest.raises(ValidationError, match="configured twice"):
        rrr_config().run("q", context=InterventionContext(retriever=object()))
    with pytest.raises(ValidationError, match="fresh context"):
        RunConfig("baseline.no_op").run("q", context=InterventionContext(budget_ledger=object()))


@pytest.mark.parametrize("key", ["temperature", "max_tokens", "max_length", "model", "seed", "stream", "max_completion_tokens"])
def test_extra_body_cannot_silently_override_owned_fields(key):
    with pytest.raises(ValidationError, match="extra_body"):
        OpenAICompatibleBackend("synthetic", "http://localhost/v1",
                                deployment_identity={"revision": "synthetic"}, extra_body={key: 1})


@pytest.mark.parametrize("k1,b", [(0, 0.5), (-1, 0.5), (True, 0.5), (1, -0.1), (1, 1.1), (1, float("nan"))])
def test_invalid_bm25_settings_fail_before_search(k1, b):
    with pytest.raises(ValidationError, match="BM25"):
        InMemoryBM25Retriever((), k1=k1, b=b)


def test_backend_config_never_inlines_credentials():
    config = RunConfig("baseline.no_op", backend={"api_key": "do-not-store"})
    with pytest.raises(ValidationError, match="api_key_env"):
        config.validate()


def test_generation_request_output_cap_has_no_total_length_alias():
    request = GenerationRequest(prompt="query", max_tokens=37, temperature=0.4)
    assert request.max_tokens == 37


def test_final_content_stop_policy_is_explicit_and_records_full_usage():
    from agentinputlab.backends.llm import FinalContentStopBackend

    settings = {"model_id": "synthetic", "base_url": "http://localhost/v1",
                "deployment_identity": {"revision": "synthetic"}}
    default = RunConfig("baseline.no_op", backend=settings)._backend()
    assert isinstance(default, OpenAICompatibleBackend)
    config = RunConfig("baseline.no_op", backend={**settings, "stop_policy": "final_content"})
    assert config.validate()["model_calls"] == 0
    backend = config._backend()
    assert isinstance(backend, FinalContentStopBackend)
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": "answer<END>tail",
                     "reasoning": "<END> in reasoning"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 512}}

    backend.inner.transport = transport
    response = backend.generate(GenerationRequest(prompt="synthetic", max_tokens=512,
                                                   seed=17, stop=("<END>",)))
    assert len(calls) == 1 and "stop" not in calls[0]
    assert calls[0]["max_tokens"] == 512 and calls[0]["seed"] == 17
    assert response.text == "answer" and response.usage.output_tokens == 512
    assert config.to_dict()["backend"]["stop_policy"] == "final_content"
    assert RunConfig.from_dict(config.to_dict())._backend().fingerprint() == config._backend().fingerprint()


@pytest.mark.parametrize("policy", [None, True, {}, [], "automatic", ""])
def test_invalid_stop_policy_fails_before_generation(policy):
    config = RunConfig("baseline.no_op", backend={"model_id": "synthetic",
        "base_url": "http://localhost/v1", "deployment_identity": {"revision": "synthetic"},
        "stop_policy": policy})
    with pytest.raises(ValidationError, match="stop_policy"):
        config.validate()


def test_config_file_size_bound(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValidationError, match="1 MiB"):
        RunConfig.from_file(path)


def test_config_init_rejects_unknown_profile(capsys):
    assert main(["config", "init", "--release-id", "missing.profile"]) == 2
    assert not capsys.readouterr().out


def test_config_cli_sanitizes_native_runtime_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(RunConfig("baseline.no_op").to_dict()))

    def fail(*args, **kwargs):
        raise RuntimeError("private-provider-material")

    monkeypatch.setattr(RunConfig, "run", fail)
    assert main(["config", "run", str(path), "--query", "q"]) == 2
    output = capsys.readouterr()
    assert "no automatic retry" in output.err
    assert "private-provider-material" not in output.err
    assert "Traceback" not in output.err
