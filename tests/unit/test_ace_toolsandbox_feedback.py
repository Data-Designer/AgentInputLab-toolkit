"""Public API contract fixtures, never real ToolSandbox/model acceptance claims."""

import hashlib
import json
import types
from pathlib import Path

import pytest

from agentinputlab.api import _state
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.recipes import ACEToolSandboxFeedback, build_ace_toolsandbox_feedback_input
from agentinputlab.recipes.ace_native_plan import _save, source_file_pins
from agentinputlab.recipes.ace_toolsandbox import SCHEMA, render_toolsandbox_agent_visible


def state(key):
    return _state(
        dict(
            query="task " + key,
            task_id=key,
            instruction="Preserve the policy.",
            messages=[dict(role="user", content="visible user request")],
            tool_names=["add_contact", "end_conversation"],
            context=[
                dict(
                    block_id="official_tool_catalog",
                    text=json.dumps(
                        dict(
                            available_tools=["add_contact", "end_conversation"],
                            tool_schemas=[],
                            categories=["fixture"],
                        )
                    ),
                ),
                dict(block_id="evidence", text="keep this source"),
            ],
            budget=dict(max_tool_calls=1),
        )
    )


def schema():
    return [
        dict(type="function", function=dict(name="add_contact", parameters=dict(type="object")))
    ]


def test_role_projection_changes_only_catalog_and_keeps_visible_fields():
    s = state("a")
    original = to_jsonable(s)
    visible = render_toolsandbox_agent_visible(s, schema())
    assert visible["tool_names"] == ["add_contact"]
    for key in ("query", "instruction", "messages"):
        assert visible[key] == original[key]
    catalog = json.loads(visible["context"][0]["text"])
    assert catalog == dict(
        available_tools=["add_contact"], tool_schemas=schema(), categories=["fixture"]
    )
    assert visible["context"][1] == original["context"][1]
    assert to_jsonable(s) == original


@pytest.mark.parametrize("schemas", [[], schema() + schema(), [dict(name="end_conversation")]])
def test_invalid_role_catalog_not_silently_filtered(schemas):
    with pytest.raises(ValidationError, match="Agent-role"):
        render_toolsandbox_agent_visible(state("a"), schemas)


class Provider:
    benchmark_id = "toolsandbox.common_support"
    source_revision = "fixture-native-revision"
    scorer_id = "fixture-native-scorer"
    scorer_revision = "fixture-native-scorer-revision"

    def _validate_config(self, config):
        return config


def prepared(tmp_path):
    root = tmp_path / "native"
    root.mkdir()
    code = root / "native.py"
    code.write_text("# fixture identity, not native acceptance\n")
    rows = []
    for key, role in (("a", "train"), ("b", "validation")):
        s = state(key)
        context = dict(interactive_console=None)
        rows.append(
            dict(
                case_id=key,
                role=role,
                state=to_jsonable(s),
                state_hash=s.state_hash,
                protected_checkpoint=dict(
                    protected=dict(
                        starting_context=context,
                        starting_context_hash=content_hash(context),
                        hidden_requirement="DO_NOT_SEND_TO_MODEL",
                    )
                ),
            )
        )
    value = dict(
        schema=SCHEMA,
        benchmark=Provider.benchmark_id,
        native_source_revision=Provider.source_revision,
        source_files=source_file_pins(root, [code]),
        source_official_split=None,
        partition_source="caller-disjoint-fixture-families",
        fitting_rows=rows,
        heldout_ids=["heldout"],
        Agent_role_schemas={"a": schema(), "b": schema()},
        native_configuration=dict(agent_model="target", user_model="environment"),
        target_model=dict(model_id="target"),
        environment_model=dict(model_id="environment"),
    )
    p = tmp_path / "input.json"
    _save(p, value)
    return root, p, value


@pytest.fixture
def processor(tmp_path, monkeypatch):
    import agentinputlab.recipes.ace_native_plan as common
    import agentinputlab.recipes.ace_toolsandbox as sandbox

    root, path, value = prepared(tmp_path)
    monkeypatch.setattr(common, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(sandbox, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(sandbox, "_provider", Provider)
    monkeypatch.setattr(
        sandbox,
        "_native_inputs",
        lambda *args: (value["Agent_role_schemas"], value["source_files"]),
    )
    p = ACEToolSandboxFeedback(
        checkout=root,
        source_input=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        output_root=tmp_path / "feedback",
        mode="online",
    )
    return p


def test_public_model_input_excludes_protected_targets_and_simulator_schemas(processor):
    p = processor
    text = json.dumps(p.fit_examples("train")[0].as_dict()["input"])
    assert "DO_NOT_SEND_TO_MODEL" not in text
    # The instruction explains why this tool is forbidden, but it is absent
    # from the actual advertised role tool schema and name list.
    assert p.tool_names["a"] == {"add_contact"}
    assert "Preserve the policy." in text and "keep this source" in text
    stream = p.stream_examples()
    p.validate_online_examples([e.as_dict() for e in stream])
    assert all(e.metadata["source_official_split"] is None for e in stream)
    assert p.fingerprint()["partition_source"] == "caller-disjoint-fixture-families"


def test_user_tool_is_internal_invalid_feedback_not_native_zero(processor):
    p = processor
    assert (
        p.score_answer(
            '{"actions":[{"name":"end_conversation","arguments":{}}],"response":"done"}', "a"
        )
        == 0.0
    )
    assert p.events[-1]["native_metrics"] is None
    assert p.events[-1]["native_task_score_NOT_claimed"]
    assert not list((p.root / "native-feedback").glob("*"))


def test_valid_native_score_reuses_once_and_errors_are_not_repaired(processor):
    p = processor
    calls = []

    def scored(*args):
        calls.append(args)
        return dict(native_metrics=dict(official_similarity=0.75))

    p._native_score = scored
    output = '{"actions":[{"name":"add_contact","arguments":{}}],"response":"done"}'
    assert p.score_answer(output, "a") == p.score_answer(output, "a") == 0.75
    assert len(calls) == 1 and p.events[-1]["exact_feedback_reuse_NOT_new_model_draw"]
    with pytest.raises(ValidationError, match="outside"):
        p.score_answer(output, "heldout")


def test_builder_requires_explicit_nonempty_partition_before_native_import(tmp_path, monkeypatch):
    import agentinputlab.recipes.ace_toolsandbox as sandbox

    monkeypatch.setattr(sandbox, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(sandbox, "_provider", Provider)
    with pytest.raises(ValidationError, match="partition label"):
        build_ace_toolsandbox_feedback_input(
            checkout=tmp_path,
            captures=[],
            native_configuration={},
            train_ids=["a"],
            validation_ids=["b"],
            heldout_ids=[],
            partition_label="",
            target_model={},
            environment_model={},
            output_path=tmp_path / "no.json",
        )
    assert not (tmp_path / "no.json").exists()


def test_native_scenario_loader_failure_restores_original_global_context(tmp_path, monkeypatch):
    """Regression fixture for the native library's process-global context."""
    import sys

    import agentinputlab.recipes.ace_toolsandbox as sandbox

    original = object()
    current = [original]

    def set_context(value):
        current[0] = value

    execution = types.ModuleType("tool_sandbox.common.execution_context")
    execution.ExecutionContext = object
    execution.get_current_context = lambda: current[0]
    execution.set_current_context = set_context
    conversion = types.ModuleType("tool_sandbox.common.tool_conversion")
    conversion.convert_to_openai_tools = lambda value: value

    class NativeAgentFixture:
        @classmethod
        def get_available_tools(cls):
            return {}

    agent = types.ModuleType("tool_sandbox.roles.openai_api_agent")
    agent.OpenAIAPIAgent = NativeAgentFixture
    for module in (execution, conversion, agent):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class BrokenProvider:
        def _scenarios(self, config):
            set_context(object())
            raise RuntimeError("original native scenario load error")

    with pytest.raises(RuntimeError, match="original native scenario load error"):
        sandbox._native_inputs(tmp_path, {}, [{"case_id": "a"}], BrokenProvider())
    assert current[0] is original
