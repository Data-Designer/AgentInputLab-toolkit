"""Public BFCL execution contracts; synthetic cases are not model acceptance."""
import json
from typing import ClassVar

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.backends.llm import GenerationResponse, ModelCapability, Usage
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import ValidationError

TOOLS = ({"type": "function", "function": {"name": "math_add", "description": "Add",
    "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}},)
CALLS = [{"id": "call_1", "type": "function", "function": {"name": "math_add", "arguments": '{"x":2}'}}]


class Backend:
    model_id = "fixture-model"
    capabilities = frozenset(ModelCapability)

    def __init__(self, outputs=None):
        self.requests = []
        self.outputs = list(outputs if outputs is not None else [CALLS])

    def fingerprint(self):
        return {"model_id": self.model_id, "revision": "fixture"}

    def generate(self, request):
        self.requests.append(request)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return GenerationResponse(text="", model=self.model_id, usage=Usage(input_tokens=12, output_tokens=5),
            raw={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": value}}],
                 "headers": {"Authorization": "DO_NOT_SAVE_HEADER"}})


class Adapter:
    benchmark_id = "bfcl.v3.simple_python"
    pairing_mode = "stateless_score_twice"
    evaluation_provenance: ClassVar = {"model_id": "fixture-model", "provenance_status": "source_verified"}

    def __init__(self):
        self.rows = tuple(BenchmarkExample(example_id=f"simple_{i}", query=f"Add {i}",
            instruction="Choose the tool", visible_tool_names=("math.add",),
            evaluator_gold={"ground_truth": "NEVER_SEND_GOLD"}) for i in range(2))
        self.predictions = []

    def verify_integrity(self):
        pass

    def examples(self, split):
        assert split == "test"
        return self.rows

    def function_tools(self, example):
        assert example in self.rows
        return TOOLS

    def score(self, example, prediction):
        self.predictions.append(prediction)
        return {"ast_accuracy": float(prediction == [{"math_add": '{"x":2}'}])}


def run(backend, **kwargs):
    config = kwargs.pop("config", RunConfig("baseline.no_op"))
    return config.evaluate("bfcl.v3.simple_python", split="test", example_ids=["simple_0"],
        adapter=kwargs.pop("adapter", Adapter()), context=InterventionContext(backend=backend), **kwargs)


def test_native_tools_score_and_capture_without_gold_or_headers():
    backend, adapter = Backend(), Adapter()
    result = run(backend, adapter=adapter)
    assert result["scored_count"] == 1 and result["failed_count"] == 0
    assert result["metrics_over_scored_cases"] == {"ast_accuracy": 1.0}
    request = backend.requests[0]
    assert request.tools == TOOLS and request.max_tokens == 512 and request.seed == 17
    assert "NEVER_SEND_GOLD" not in repr(request)
    assert "native_function_catalog" not in request.system  # block ID is not prompt text
    assert "math_add" in request.system
    row = result["records"][0]
    assert row["usage"]["model_calls"] == 1 and row["usage"]["output_tokens"] == 5
    assert row["model_responses"][0]["response"]["tool_calls"] == CALLS
    assert "DO_NOT_SAVE_HEADER" not in json.dumps(result)


def test_no_tool_call_is_a_real_zero_not_a_retry():
    backend = Backend([[]])
    result = run(backend)
    assert result["scored_count"] == 1 and result["metrics_over_scored_cases"] == {"ast_accuracy": 0.0}
    assert len(backend.requests) == 1


def test_provider_failure_preserved_without_retry():
    backend = Backend([RuntimeError("token=sk-private123456789")])
    result = run(backend)
    assert result["failed_count"] == 1 and len(backend.requests) == 1
    assert result["records"][0]["usage"]["model_calls"] == 1
    assert "sk-private123456789" not in json.dumps(result)


def test_explicit_cap_and_temperature_and_remaining_budget():
    backend = Backend()
    result = run(backend, config=RunConfig("baseline.no_op", budget=ExecutionBudget(max_output_tokens=80)),
        predictor_parameters={"schema_version": 1, "function_calling": {"max_output_tokens": 64, "temperature": 0.2}})
    assert result["scored_count"] == 1
    assert backend.requests[0].max_tokens == 64 and backend.requests[0].temperature == 0.2


def test_exhausted_budget_does_not_generate():
    backend = Backend()
    result = run(backend, config=RunConfig("baseline.no_op", budget=ExecutionBudget(max_model_calls=0)))
    assert result["failed_count"] == 1 and not backend.requests


def test_provider_overrun_retains_actual_usage_without_scoring():
    backend = Backend()
    result = run(backend, config=RunConfig("baseline.no_op", budget=ExecutionBudget(max_output_tokens=4)))
    assert result["failed_count"] == 1 and len(backend.requests) == 1
    assert backend.requests[0].max_tokens == 4
    assert result["records"][0]["usage"]["output_tokens"] == 5


def test_missing_source_archive_rejected_before_preparation(monkeypatch):
    import agentinputlab.benchmarks

    monkeypatch.setattr(agentinputlab.benchmarks, "prepare_benchmark", lambda *a, **kw: pytest.fail("download"))
    with pytest.raises(ValidationError, match="bfcl-source-archive"):
        RunConfig("baseline.no_op").evaluate("bfcl.v3.simple_python", split="test", limit=1,
            prepared_root="existing-prepared",
            context=InterventionContext(backend=Backend()))


def test_other_benchmarks_reject_bfcl_source_before_call():
    backend = Backend()
    with pytest.raises(ValidationError, match="only to bfcl"):
        RunConfig("baseline.no_op").evaluate("2wiki.multihop", split="dev", limit=1,
            bfcl_source_archive="original.tar.gz", context=InterventionContext(backend=backend))
    assert not backend.requests


def test_automatic_cache_preparation_binds_actual_model(monkeypatch, tmp_path):
    import agentinputlab.benchmarks

    backend, seen = Backend(), []
    def prepare(cell, **kwargs):
        seen.append((cell, kwargs))
        return {"prepared_root": str(tmp_path / "prepared"), "source_archive": "verified-cache.tar.gz"}
    monkeypatch.setattr(agentinputlab.benchmarks, "prepare_benchmark", prepare)
    bridges = []
    monkeypatch.setattr(agentinputlab.benchmarks, "SourceVerifiedBFCLV3ASTBridge",
                        lambda **kw: bridges.append(kw) or object())
    monkeypatch.setattr(agentinputlab.benchmarks, "load_bfcl_v3_simple_adapter", lambda *a, **kw: Adapter())
    result = RunConfig("baseline.no_op").evaluate("bfcl.v3.simple_python", split="test", limit=1,
        cache_dir=tmp_path, offline=True, context=InterventionContext(backend=backend))
    assert result["scored_count"] == 1
    assert seen == [("bfcl.v3.simple_python", {"cache_dir": tmp_path, "offline": True,
                                            "accept_license_review": False, "source_archive": None})]
    assert bridges == [{"source_archive": "verified-cache.tar.gz", "model_id": backend.model_id}]


@pytest.mark.parametrize("parameters", [
    {"schema_version": 1, "qa": {}},
    {"schema_version": 1, "function_calling": {"max_output_tokens": True}},
    {"schema_version": 1, "function_calling": {"temperature": -1}},
])
def test_bad_settings_rejected_before_call(parameters):
    backend = Backend()
    with pytest.raises(ValidationError):
        run(backend, predictor_parameters=parameters)
    assert not backend.requests


def test_cli_uses_source_verified_loader_and_preserves_output(tmp_path, monkeypatch, capsys):
    import agentinputlab.benchmarks

    backend, adapter = Backend(), Adapter()
    seen = []
    monkeypatch.setattr(RunConfig, "_backend", lambda self: backend)
    monkeypatch.setattr(agentinputlab.benchmarks, "SourceVerifiedBFCLV3ASTBridge",
        lambda **kw: seen.append(kw) or object())
    monkeypatch.setattr(agentinputlab.benchmarks, "load_bfcl_v3_simple_adapter", lambda *a, **kw: adapter)
    config = tmp_path / "run.json"
    config.write_text(json.dumps(RunConfig("baseline.no_op", backend={"fixture": True}).to_dict()))
    out = tmp_path / "rows.jsonl"
    args = ["config", "evaluate", str(config), "--benchmark", "bfcl.v3.simple_python", "--split", "test",
        "--case-id", "simple_0", "--prepared-root", str(tmp_path), "--bfcl-source-archive", "original.tar.gz",
        "--output", str(out)]
    assert main(args) == 0
    assert seen == [{"source_archive": "original.tar.gz", "model_id": backend.model_id}]
    assert json.loads(capsys.readouterr().out)["scored_count"] == 1
    before = out.read_bytes()
    assert main(args) == 2
    assert out.read_bytes() == before and len(backend.requests) == 1
