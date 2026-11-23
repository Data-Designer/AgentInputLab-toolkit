"""Public configuration regressions, not real-model acceptance evidence."""

import hashlib
import json
import sys
from dataclasses import replace

import pytest

from agentinputlab import FrozenAgentState, InterventionContext, RunConfig
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from tests.unit.test_benchmark_configuration import Backend


def configuration(tmp_path):
    prompts = {"planner": "fixture", "join": "fixture"}
    bundle = tmp_path / "prompts.json"
    bundle.write_text(json.dumps({"prompts": prompts}))
    (tmp_path / "upstream").mkdir()
    return RunConfig("plan.llm_compiler.full", parameters={"prompt_hash": content_hash(prompts)}, runtime={
        "kind": "llmcompiler_live", "checkout": str(tmp_path / "upstream"),
        "python_executable": sys.executable, "prompt_bundle": str(bundle),
        "prompt_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "per_call_output_cap": 512, "timeout_seconds": 400,
        "tools": {"kind": "visible_context_search", "source_revision": "fixture"}})


def test_public_compiler_configuration_accepts_original_worker_binding(tmp_path):
    config = configuration(tmp_path)
    assert config.validate()["valid"] is True
    context = config._configured_context(InterventionContext(backend=Backend([])))
    bound = config._bind_case_context(context, FrozenAgentState(query="visible question"))
    from agentinputlab.integrations.external_workers import LLMCompilerLiveWorker

    assert isinstance(bound.external_runtime, LLMCompilerLiveWorker)
    assert bound.external_runtime.tools is bound.tools
    assert bound.external_runtime.backend is bound.backend
    assert bound.external_runtime.per_call_output_cap == 512


def test_public_compiler_rejects_other_profile(tmp_path):
    config = configuration(tmp_path)
    with pytest.raises(ValidationError, match="supported only"):
        RunConfig("baseline.no_op", runtime=config.runtime).validate()


@pytest.mark.parametrize("key,value", [
    ("prompt_bundle_sha256", "bad"), ("per_call_output_cap", True),
    ("per_call_output_cap", 0), ("timeout_seconds", -1),
    ("argument_order", {"search": ["query", "query"]}),
    ("tools", {"kind": "shell", "source_revision": "fixture"}),
    ("tools", {"kind": "visible_context_search"}), ("shell_command", "false"),
])
def test_invalid_binding_is_rejected_without_io(tmp_path, key, value):
    config = configuration(tmp_path)
    with pytest.raises(ValidationError):
        replace(config, runtime={**config.runtime, key: value}).validate()


def test_prompt_binding_is_checked_before_any_worker_execution(tmp_path):
    config = configuration(tmp_path)
    context = InterventionContext(backend=Backend([]))
    state = FrozenAgentState(query="q")
    with pytest.raises(ArtifactError, match="checksum"):
        replace(config, runtime={**config.runtime, "prompt_bundle_sha256": "0" * 64})._bind_case_context(context, state)
    with pytest.raises(ArtifactError, match="prompt_hash"):
        replace(config, parameters={"prompt_hash": "0" * 64})._bind_case_context(context, state)
    with pytest.raises(ValidationError, match="backend"):
        config._bind_case_context(InterventionContext(), state)
    bound = config._bind_case_context(context, state)
    with pytest.raises(ValidationError, match="tools are configured twice"):
        config._bind_case_context(replace(context, tools=bound.tools), state)
    with pytest.raises(ValidationError, match="runtime is configured twice"):
        config._configured_context(bound)
    runtime = dict(config.runtime)
    del runtime["tools"]
    injected = replace(config, runtime=runtime)
    assert injected._bind_case_context(replace(context, tools=bound.tools), state).tools is bound.tools
    with pytest.raises(ValidationError, match="runtime.tools or context.tools"):
        injected._bind_case_context(context, state)


def test_file_paths_follow_configuration_not_working_directory(tmp_path, monkeypatch):
    from agentinputlab.contracts.serialization import to_jsonable

    config = configuration(tmp_path)
    value = to_jsonable(config)
    value["runtime"].update(checkout="source", prompt_bundle="prompts.json", python_executable="worker/bin/python")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value))
    monkeypatch.chdir(tmp_path.parent)
    loaded = RunConfig.from_file(path)
    assert loaded.runtime["prompt_bundle"] == str(tmp_path / "prompts.json")
    assert loaded.runtime["python_executable"] == str(tmp_path / "worker/bin/python")
    loaded.validate()  # Nonexistent source/interpreter are not accessed at validation.


def test_stale_worker_environment_has_actionable_preflight(tmp_path):
    config = configuration(tmp_path)
    config = replace(config, runtime={**config.runtime, "python_executable": str(tmp_path / "gone/bin/python")})
    assert config.validate()["valid"]  # Shape validation is not readiness.
    with pytest.raises(ValidationError, match="worker Python is missing"):
        config._bind_case_context(InterventionContext(backend=Backend([])), FrozenAgentState(query="q"))


def test_visible_search_is_case_local_and_has_no_gold_or_network(tmp_path):
    from agentinputlab.connectors.tools import ToolCall
    from agentinputlab.contracts import ContextBlock

    config = configuration(tmp_path)
    context = InterventionContext(backend=Backend([]))
    states = [FrozenAgentState(query="q", context=(ContextBlock(block_id=f"doc{i}", text=f"visible{i}"),)) for i in range(2)]
    bound = [config._bind_case_context(context, state) for state in states]
    assert bound[0].external_runtime is not bound[1].external_runtime
    assert bound[0].tools.fingerprint() != bound[1].tools.fingerprint()
    assert bound[0].tools.call(ToolCall("search", {"query": "visible0"})).output == "[doc0] visible0"
    assert bound[1].tools.calls == []
    assert bound[0].tools.list_tools()[0].read_only and not bound[0].tools.list_tools()[0].network
    with pytest.raises(ValidationError):
        bound[0].tools.call(ToolCall("shell", {"query": "q"}))


def test_run_binds_original_worker_without_python_object_injection(tmp_path, monkeypatch):
    from agentinputlab import api

    config = configuration(tmp_path)
    seen = []

    def execute(method, state, context):
        seen.append(context)
        assert context.external_runtime.tools is context.tools
        assert context.external_runtime.backend is context.backend
        assert state.instruction == "Keep this instruction"
        return {"synthetic_binding_check": True}

    monkeypatch.setattr(api, "run_composite_intervention", execute)
    result = config.run(FrozenAgentState(query="q", instruction="Keep this instruction"),
                        context=InterventionContext(backend=Backend([])))
    assert result["synthetic_binding_check"] and len(seen) == 1


def test_evaluate_binds_captured_backend_per_case_and_preserves_failure_denominator(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    seen = []
    original = RunConfig._bind_case_context

    def check(self, context, state):
        from agentinputlab.benchmark_configuration import _CapturedBackend

        bound = original(self, context, state)
        assert isinstance(bound.external_runtime.backend, _CapturedBackend)
        seen.append(bound)
        raise ValidationError("synthetic stop before native subprocess")

    monkeypatch.setattr(RunConfig, "_bind_case_context", check)
    from tests.unit.test_benchmark_configuration import evaluate

    result = evaluate(config, Backend([]))
    assert result["selected_count"] == result["failed_count"] == len(seen) == 2
    assert result["scored_count"] == 0
    assert seen[0].tools is not seen[1].tools
    assert seen[0].backend is not seen[1].backend
    assert all(row["model_responses"] == [] for row in result["records"])
