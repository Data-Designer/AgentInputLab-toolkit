"""Contract fixtures only, not evidence of real native/model acceptance."""

import hashlib
import json
from pathlib import Path

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes import ACEStateBenchFeedback, build_ace_statebench_feedback_input
from agentinputlab.recipes.ace_native_plan import (
    NativePlanFeedback,
    _save,
    native_process_scope,
    parse_native_tool_plan,
    source_file_pins,
    verify_source_files,
)
from agentinputlab.recipes.ace_statebench import SCHEMA


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source(tmp_path):
    from agentinputlab.api import _state
    from agentinputlab.contracts.serialization import to_jsonable

    native = tmp_path / "native"
    native.mkdir()
    code = native / "scorer.py"
    code.write_text("# fixture identity, never native acceptance\n")
    rows = []
    for key, role in (("train-id", "train"), ("validation-id", "validation")):
        state = _state(
            dict(
                query="query " + key,
                task_id=key,
                tool_names=["lookup"],
                budget=dict(max_tool_calls=1),
            )
        )
        rows.append(
            dict(
                case_id=key,
                role=role,
                state=to_jsonable(state),
                state_hash=state.state_hash,
                protected_checkpoint=dict(secret_reference="never in model input"),
            )
        )
    value = dict(
        schema=SCHEMA,
        benchmark="state_bench.common_support",
        native_source_revision="fixture",
        source_files=source_file_pins(native, [code]),
        fitting_rows=rows,
        heldout_ids=["test-id"],
        source_official_split="train",
        native_configuration=dict(
            split="train",
            agent_model="target",
            evaluator_model="judge",
            evaluator_base_url="http://127.0.0.1:1/v1",
        ),
        target_model=dict(model_id="target"),
        environment_model=dict(model_id="judge"),
    )
    path = tmp_path / "input.json"
    _save(path, value)
    return native, path, value


class FixtureFeedback(NativePlanFeedback):
    metric = "fixture_score"

    def __init__(self, *, checkout, source_input, output_root, mode="offline"):
        self._initialize(
            checkout=checkout, source_input=source_input, output_root=output_root, mode=mode
        )
        self.tool_names = {k: set(s.tool_names) for k, s in self.states.items()}
        self.calls = []
        self._create_output()

    def fingerprint(self):
        self._verify_source()
        return dict(source=self.source_input, mode=self.mode, fixture=True)

    def _example(self, key, split):
        from agentinputlab.recipes import ACEExample

        return ACEExample(
            example_id=key,
            input=dict(question=self.states[key].query),
            expected_output=key,
            split=split,
        )

    def _native_score(self, key, plan, folder):
        self.calls.append((key, plan))
        return dict(native_metrics=dict(fixture_score=0.0))


@pytest.fixture
def make(tmp_path, monkeypatch):
    import agentinputlab.recipes.ace_native_plan as common

    monkeypatch.setattr(common, "verify_native_checkout", lambda root, **kw: Path(root))
    native, path, value = source(tmp_path)

    def create(mode="offline"):
        return FixtureFeedback(
            checkout=native,
            source_input=pin(path),
            output_root=tmp_path / (mode + "-feedback"),
            mode=mode,
        )

    return create, native, path, value


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        '```json\n{"actions":[],"response":"ok"}\n```',
        '{"actions":[],"response":"ok","extra":1}',
        '{"actions":[],"actions":[],"response":"ok"}',
        '{"actions":[{"name":"hidden","arguments":{}}],"response":"ok"}',
        '{"actions":[{"name":[],"arguments":{}}],"response":"ok"}',
        '{"actions":[],"response":""}',
        '{"actions":[{"name":"lookup","arguments":{"x":NaN}}],"response":"ok"}',
        '{"actions":[{"name":"lookup","arguments":{"x":1e999}}],"response":"ok"}',
        '{"actions":[{"name":"lookup","arguments":{}},{"name":"lookup","arguments":{}}],"response":"ok"}',
    ],
)
def test_strict_complete_plan_contract(text):
    with pytest.raises((ValueError, TypeError)):
        parse_native_tool_plan(text, tool_names={"lookup"}, tool_cap=1)


def test_invalid_plan_is_internal_feedback_not_official_score_or_retry(make):
    p = make[0]()
    assert p.score_answer("bad", "train-id") == 0.0
    assert not p.calls and p.events[-1]["native_metrics"] is None
    assert p.events[-1]["native_task_score_NOT_claimed"]
    with pytest.raises(ValidationError, match="outside"):
        p.score_answer("bad", "test-id")


def test_valid_zero_is_real_score_and_exact_reuse_does_not_redraw(make):
    p = make[0]()
    plan = '{"actions":[{"name":"lookup","arguments":{"id":1}}],"response":"done"}'
    assert p.score_answer(plan, "train-id") == p.score_answer(plan, "train-id") == 0.0
    assert len(p.calls) == 1 and len(p.events) == 2
    assert p.events[-1]["exact_feedback_reuse_NOT_new_model_draw"]
    assert p.events[-1]["result"]["native_metrics"] == {"fixture_score": 0.0}


def test_scorer_failure_is_not_fabricated_zero_or_retried(make):
    p = make[0]()

    def broken(*args):
        raise RuntimeError("fixture native judge failed")

    p._native_score = broken
    with pytest.raises(RuntimeError, match="judge failed"):
        p.score_answer('{"actions":[],"response":"ok"}', "train-id")
    assert not p.events and not p.cache
    assert len(list((p.root / "native-feedback").iterdir())) == 1


def test_source_and_inmemory_drift_rejected_before_execution(make):
    p = make[0]()
    p.source["fitting_rows"][0]["state"]["query"] = "changed"
    with pytest.raises(ValidationError, match="in-memory"):
        p.score_answer("bad", "train-id")
    assert not p.calls


def test_source_file_drift_not_treated_as_model_failure(make):
    p = make[0]()
    (make[1] / "scorer.py").write_text("# changed\n")
    with pytest.raises(ValidationError, match="checksum"):
        p.score_answer("bad", "train-id")


def test_fitting_and_online_inputs_keep_disjoint_labels(make):
    p = make[0]("online")
    stream = p.stream_examples()
    assert [e.example_id for e in stream] == ["train-id", "validation-id"]
    assert all(e.split == "test" and e.metadata["source_official_split"] == "train" for e in stream)
    assert "never in model input" not in json.dumps([e.as_dict() for e in stream])
    p.validate_online_examples([e.as_dict() for e in stream])
    with pytest.raises(ValidationError, match="online order"):
        p.validate_online_examples([e.as_dict() for e in reversed(stream)])
    with pytest.raises(ValidationError, match="heldout"):
        p.fit_examples("test")


def test_offline_stream_requires_explicit_online_mode(make):
    with pytest.raises(ValidationError, match="explicit online"):
        make[0]().stream_examples()


def test_native_scope_restores_process_even_on_exception(tmp_path):
    import sys

    before, paths = Path.cwd(), list(sys.path)
    with (
        pytest.raises(RuntimeError),
        native_process_scope(tmp_path, package="fixture_no_real_import"),
    ):
        assert Path.cwd() == tmp_path and str(tmp_path) in sys.path
        raise RuntimeError("fixture")
    assert Path.cwd() == before and sys.path == paths


def test_native_scope_rejects_other_imported_checkout(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules, "fixture_native", SimpleNamespace(__file__="/elsewhere/file.py")
    )
    with (
        pytest.raises(ExternalDependencyError, match="different"),
        native_process_scope(tmp_path, package="fixture_native"),
    ):
        pass


@pytest.mark.parametrize("name", ["../scorer.py", "/tmp/scorer.py", "x/../scorer.py"])
def test_source_pin_cannot_escape_checkout(make, name):
    with pytest.raises(ValidationError, match="relative"):
        verify_source_files(make[1], {name: "0" * 64})


def test_public_state_constructor_needs_no_private_operator_modules(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import agentinputlab.recipes.ace_native_plan as common
    import agentinputlab.recipes.ace_statebench as state

    native, path, value = source(tmp_path)
    monkeypatch.setattr(state, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(common, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(
        state,
        "_provider",
        lambda: SimpleNamespace(
            benchmark_id="state_bench.common_support",
            source_revision="fixture",
            _validate_config=lambda v: v,
        ),
    )
    domains = {
        key: SimpleNamespace(tool_schemas=[dict(name="lookup")])
        for key in ("train-id", "validation-id")
    }
    monkeypatch.setattr(state, "_native_inputs", lambda *args: (domains, {}, value["source_files"]))
    p = ACEStateBenchFeedback(
        checkout=native, source_input=pin(path), output_root=tmp_path / "public-feedback"
    )
    assert p.root.is_dir() and p.metric == "task_completion_pass"
    text = json.dumps([e.as_dict()["input"] for e in p.fit_examples("train")])
    assert "lookup" in text and "never in model input" not in text


def test_state_builder_rejects_test_split_before_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import agentinputlab.recipes.ace_statebench as state

    monkeypatch.setattr(state, "verify_native_checkout", lambda root, **kw: Path(root))
    monkeypatch.setattr(
        state,
        "_provider",
        lambda: SimpleNamespace(source_revision="fixture", _validate_config=lambda v: v),
    )
    with pytest.raises(ValidationError, match="official train"):
        build_ace_statebench_feedback_input(
            checkout=tmp_path,
            captures=[],
            native_configuration=dict(split="test"),
            train_ids=["a"],
            validation_ids=["b"],
            heldout_ids=[],
            target_model={},
            environment_model={},
            output_path=tmp_path / "no.json",
        )
    assert not (tmp_path / "no.json").exists()


@pytest.mark.parametrize('extensions,expected', [
    ({}, dict(single_attempt=False, extra_body={})),
    ({'request_attempts': 1, 'environment_timeout_seconds': 17},
     dict(single_attempt=True, timeout=17, extra_body={})),
    ({'environment_extra_body': {'chat_template_kwargs': {'enable_thinking': False}}},
     dict(single_attempt=False, extra_body={'chat_template_kwargs': {'enable_thinking': False}})),
])
def test_state_ace_feedback_forwards_configured_judge_transport(monkeypatch, tmp_path, extensions, expected):
    """The ACE feedback path must honor the same validated transport as the arena."""
    import contextlib
    import sys
    from types import SimpleNamespace

    import agentinputlab.recipes.ace_statebench as module
    import workers.official_arenas.providers.statebench as provider

    class StopBeforeClient:
        pass

    class Captured(Exception):
        pass

    monkeypatch.setattr(module, 'native_process_scope', lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(module, '_save', lambda *a: None)
    diff = SimpleNamespace(to_dict=lambda: {})
    monkeypatch.setitem(sys.modules, 'state_bench.env_loader', SimpleNamespace(
        load_task_environment=lambda *a: (SimpleNamespace(deep_copy=lambda: {}), None)))
    monkeypatch.setitem(sys.modules, 'state_bench.protocol', SimpleNamespace(load_default_protocol=lambda: None))
    monkeypatch.setitem(sys.modules, 'state_bench.schemas', SimpleNamespace(
        StateDiff=SimpleNamespace(compute=lambda *a: diff)))
    monkeypatch.setitem(sys.modules, 'state_bench.scoring', SimpleNamespace(
        TaskRequirementsJudge=StopBeforeClient, combine_task_completion=lambda *a: None,
        evaluate_state_requirements=lambda *a: SimpleNamespace(score=0)))
    captured = {}
    def client(**kwargs):
        captured.update(kwargs)
        captured.setdefault('single_attempt', False)
        captured.setdefault('extra_body', {})
        raise Captured()
    monkeypatch.setattr(provider, '_openai_compatible_client', client)
    feedback = object.__new__(ACEStateBenchFeedback)
    feedback.checkout = tmp_path
    feedback.tasks = {'x': SimpleNamespace(now='fixture')}
    feedback.domains = {'x': SimpleNamespace(environment_class=lambda *a, **kw:
        SimpleNamespace(get_full_snapshot=lambda: {}))}
    feedback.states = {'x': SimpleNamespace(query='query', state_hash='fixture')}
    feedback.environment_base_url = 'http://127.0.0.1:1/v1'
    feedback.configuration = dict(evaluator_model='judge', evaluator_api_key_var='JUDGE_KEY', **extensions)
    with pytest.raises(Captured):
        feedback._native_score('x', dict(actions=[], response='done'), tmp_path)
    assert captured == dict(model='judge', base_url=feedback.environment_base_url,
                            api_key_var='JUDGE_KEY', **expected)
