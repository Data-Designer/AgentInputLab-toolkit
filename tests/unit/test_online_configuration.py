"""Online file-entry fixtures; these are not real-model acceptance results."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from agentinputlab.cli.main import main
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.online_configuration import OnlineConfig
from agentinputlab.recipes.ace_online import ACEOnlineControls, ACEOnlineRecipe
from tests.unit.test_fit_configuration import ace


def value():
    return dict(
        release_id="recipe.ace.online_transductive",
        checkout="upstream/ace",
        benchmark="hotpotqa.distractor",
        controls=asdict(ACEOnlineControls()),
        deployment=ace()["deployment"],
        allow_test_feedback=True,
        source_and_asset_acknowledged=True,
        stream=[dict(example_id="one", input="question", expected_output="answer", split="test")],
    )


def test_validate_cli_does_not_execute(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        ACEOnlineRecipe, "run_stream", lambda *a, **k: pytest.fail("unexpected generation")
    )
    path = tmp_path / "online.json"
    path.write_text(json.dumps(value()))
    assert main(["config", "online", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["validated"] and not result["executed"] and result["model_calls"] == 0
    assert OnlineConfig.from_file(path).value["checkout"] == str(tmp_path / "upstream/ace")


@pytest.mark.parametrize(
    "change",
    [
        {"allow_test_feedback": False},
        {"allow_test_feedback": 1},
        {"source_and_asset_acknowledged": False},
        {"benchmark": "tooltalk.full"},
        {"release_id": "recipe.ace.offline"},
        {"factory": "arbitrary:code"},
        {"stream": []},
        {"stream": "not an array"},
    ],
)
def test_invalid_protocol_rejected(change):
    v = value()
    v.update(change)
    with pytest.raises(ValidationError):
        OnlineConfig.from_dict(v)


def test_no_implicit_controls_or_roles():
    v = value()
    del v["controls"]["seed"]
    with pytest.raises(ValidationError, match="every native control"):
        OnlineConfig.from_dict(v)
    v = value()
    v["stream"][0]["split"] = "train"
    with pytest.raises(ValidationError, match="no implicit relabeling"):
        OnlineConfig.from_dict(v)
    v = value()
    v["deployment"]["max_tokens_per_call"] = 1024
    with pytest.raises(ValidationError, match="matching per-call"):
        OnlineConfig.from_dict(v)


def test_no_implicit_native_feedback_or_source_mix():
    v = value()
    v["feedback"] = {"kind": "tau"}
    with pytest.raises(ValidationError, match="not both"):
        OnlineConfig.from_dict(v)
    v = value()
    del v["stream"]
    v["feedback"] = dict(kind="retrieval", source_input=dict(path="a", sha256="a" * 64))
    assert OnlineConfig.from_dict(v).validate()["validated"]


def stub_native(monkeypatch, status="native_online_returned", answer="answer"):
    calls = []

    def run(self, stream, **kwargs):
        calls.append((stream, kwargs, self.controls))
        folder = kwargs["output_root"]
        folder.mkdir()
        terminal = dict(
            status=status,
            primary_outcomes=[
                dict(
                    case_id="one",
                    status="native_prediction_returned",
                    native_prediction=dict(final_answer=answer),
                )
            ],
            per_case_usage={"one": dict(model_calls=1, output_tokens=3)},
            usage=dict(model_calls=1),
        )
        (folder / "terminal.json").write_text(json.dumps(terminal))
        (folder / "manifest.json").write_text("{}")
        return SimpleNamespace(terminal=terminal)

    monkeypatch.setattr(ACEOnlineRecipe, "run_stream", run)
    monkeypatch.setenv("TEST_ACE_KEY", "fixture-not-a-secret")
    return calls


def test_original_stream_controls_and_failure_preserved(tmp_path, monkeypatch):
    calls = stub_native(monkeypatch, status="native_failed_terminal")
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value()))
    out = tmp_path / "result"
    assert main(["config", "online", str(path), "--execute", "--output-dir", str(out)]) == 2
    terminal = json.loads((out / "terminal.json").read_text())
    assert terminal["status"] == "native_failed_terminal" and terminal["usage"] == {
        "model_calls": 1
    }
    assert calls[0][2] == ACEOnlineControls()
    assert calls[0][1]["allow_test_feedback"] is True
    assert calls[0][0][0]["example_id"] == "one"
    assert "fixture-not-a-secret" not in (out / "terminal.json").read_text()
    with pytest.raises(ValidationError, match="new directory"):
        OnlineConfig.from_file(path).execute(out)
    assert len(calls) == 1


def test_failure_is_sanitized_and_not_retried(tmp_path, monkeypatch):
    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise RuntimeError("private-provider-body")

    monkeypatch.setattr(ACEOnlineRecipe, "run_stream", fail)
    monkeypatch.setenv("TEST_ACE_KEY", "local")
    with pytest.raises(ExternalDependencyError, match="no automatic retry"):
        OnlineConfig.from_dict(value()).execute(tmp_path / "result")
    raw = (tmp_path / "result/terminal.json").read_text()
    assert "private-provider-body" not in raw and len(calls) == 1
    assert json.loads(raw)["usage_may_have_been_consumed"] is True


def test_missing_key_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_ACE_KEY", raising=False)
    with pytest.raises(ExternalDependencyError, match="API-key"):
        OnlineConfig.from_dict(value()).execute(tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_native_toolsandbox_missing_credential_before_attempt(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setenv("TEST_ACE_KEY", "private-target-value")
    monkeypatch.delenv("TEST_NATIVE_SANDBOX_KEY", raising=False)
    source = tmp_path / "feedback.json"
    source.write_text(json.dumps(dict(benchmark="toolsandbox.common_support", native_configuration={
        "agent_key_env": "TEST_NATIVE_SANDBOX_KEY", "user_key_env": "TEST_NATIVE_SANDBOX_KEY",
    })))
    v = value()
    v.pop("stream")
    v["benchmark"] = "toolsandbox.common_support"
    v["feedback"] = dict(kind="toolsandbox", checkout="native/ToolSandbox",
        source_input=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    monkeypatch.setattr(ACEOnlineRecipe, "run_stream", lambda *a, **k: pytest.fail("unexpected generation"))
    with pytest.raises(ExternalDependencyError, match="missing online environment variable: TEST_NATIVE_SANDBOX_KEY"):
        OnlineConfig.from_dict(v).execute(tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_native_feedback_matches_primary_not_later_adapted_answer(tmp_path, monkeypatch):
    from agentinputlab.recipes import ace_tau

    stub_native(monkeypatch)
    v = value()
    del v["stream"]
    v["benchmark"] = "tau3.retail"
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(benchmark="tau3.retail")))
    import hashlib

    v["feedback"] = dict(
        kind="tau",
        checkout="native/tau",
        source_input=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    )
    events = [
        dict(
            target="one",
            prediction_sha256=content_hash("later-answer"),
            result=dict(native_metrics={"official_reward": 1}),
        ),
        dict(
            target="one",
            prediction_sha256=content_hash("answer"),
            result=dict(native_metrics={"official_reward": 0}),
        ),
    ]
    processor = SimpleNamespace(
        events=events,
        stream_examples=lambda: value()["stream"],
        validate_online_examples=lambda s: None,
    )
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return processor

    monkeypatch.setattr(ace_tau, "create_tau_feedback", create)
    result = OnlineConfig.from_dict(v).execute(tmp_path / "result")
    assert received["mode"] == "online"
    assert result["primary_native_feedback"][0]["feedback_event_index"] == 1
    assert result["primary_native_feedback"][0]["result"]["native_metrics"]["official_reward"] == 0
    assert result["feedback_events"]["sha256"]
