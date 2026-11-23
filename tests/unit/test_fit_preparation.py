"""Native preparation configuration fixtures, not real benchmark evidence."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinputlab.cli.main import main
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.fit_configuration import FitConfig
from agentinputlab.fit_preparation import _build, _configuration, prepare_fit
from tests.unit.test_fit_configuration import ace


def config(tmp_path, kind="statebench"):
    captures = tmp_path / "captures.json"
    captures.write_text('[{"checkpoint_id":"train-1"},{"checkpoint_id":"dev-1"}]')
    fit = ace()
    del fit["examples"]
    value = dict(
        fit=fit,
        feedback=dict(kind=kind, checkout="./native"),
        captures=dict(
            path="captures.json", sha256=hashlib.sha256(captures.read_bytes()).hexdigest()
        ),
        groups=dict(train=["train-1"], validation=["dev-1"], heldout=["test-1"]),
        native_configuration=dict(split="train"),
        target_model=dict(model_id="target"),
        environment_model=dict(model_id="judge"),
        timeout_seconds=30,
    )
    if kind != "statebench":
        value["partition_label"] = "caller-declared-development"
    if kind == "tau":
        del value["native_configuration"], value["environment_model"]
        value.update(domain="retail", native_configurations={"train-1": {}, "dev-1": {}})
    return value


def write(tmp_path, value):
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(value))
    return path


def fixture_native(monkeypatch, kind):
    import importlib

    from agentinputlab.fit_configuration import _save
    from agentinputlab.recipes import ACEExample

    module = importlib.import_module("agentinputlab.recipes.ace_" + kind)
    seen = {}

    def builder(**kwargs):
        seen["builder"] = kwargs
        output = kwargs["output_path"]
        _save(output, {"fixture": True})
        return dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())

    def factory(**kwargs):
        seen["factory"] = kwargs
        return SimpleNamespace(
            fit_examples=lambda split: tuple(
                ACEExample.from_value(e, split=split) for e in ace()["examples"][split]
            ),
            validate_examples=lambda train, validation: seen.update(validated=(train, validation)),
        )

    monkeypatch.setattr(module, "build_ace_" + kind + "_feedback_input", builder)
    monkeypatch.setattr(module, "create_" + kind + "_feedback", factory)
    return seen


@pytest.mark.parametrize("kind", ["statebench", "toolsandbox", "tau"])
def test_preparation_delegates_without_changing_source_or_groups(tmp_path, monkeypatch, kind):
    path = write(tmp_path, config(tmp_path, kind))
    value = _configuration(path)
    seen = fixture_native(monkeypatch, kind)
    out = tmp_path / "new"
    out.mkdir()
    result = _build(value, out)
    assert result["status"] == "prepared" and result["model_calls"] == 0
    assert not result["fit_executed"] and not result["scoring_executed"]
    assert seen["builder"]["train_ids"] == ["train-1"]
    assert seen["builder"]["validation_ids"] == ["dev-1"]
    assert seen["builder"]["heldout_ids"] == ["test-1"]
    assert seen["factory"]["mode"] == "offline"
    assert seen["factory"]["output_root"] == str(out / "input-verification")
    ready = FitConfig.from_file(result["fit_configuration"]["path"])
    assert ready.validate()["validated"]
    assert ready._ace_examples() == FitConfig.from_dict(ace())._ace_examples()
    assert ready.value["feedback"]["source_input"]["sha256"] == result["source_input"]["sha256"]


def test_cli_prepares_ready_file_in_selected_python_without_fitting(tmp_path, monkeypatch, capsys):
    import agentinputlab.fit_preparation as preparation
    from agentinputlab.fit_configuration import _save

    path = write(tmp_path, config(tmp_path))
    fixture_native(monkeypatch, "statebench")
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        request = Path(command[-1])
        _save(request.parent / "terminal.json", _build(_configuration(request), request.parent))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preparation.subprocess, "run", launch)
    output = tmp_path / "out"
    assert main(["config", "prepare-fit", str(path), "--output-dir", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((output / "terminal.json").read_text())
    assert calls[0][1:3] == ["-I", "-B"]
    assert calls[0][-2] == "--worker"
    assert 'run_module("agentinputlab.fit_preparation"' in calls[0][4]
    assert main(["config", "prepare-fit", str(path), "--output-dir", str(output)]) == 2
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change", ["overlap", "unknown", "test", "factory", "online", "source", "empty"]
)
def test_preparation_rejects_bad_groups_and_injected_feedback(tmp_path, change):
    value = config(tmp_path)
    if change == "overlap":
        value["groups"]["validation"] = ["train-1"]
    elif change == "unknown":
        value["groups"]["other"] = []
    elif change == "test":
        value["feedback"]["kind"] = "tooltalk"
    elif change == "factory":
        value["feedback"]["factory"] = "os:system"
    elif change == "online":
        value["feedback"]["mode"] = "online"
    elif change == "source":
        value["feedback"]["source_input"] = dict(path="x", sha256="a" * 64)
    else:
        value["groups"]["train"] = []
    with pytest.raises(ValidationError):
        prepare_fit(write(tmp_path, value), output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_captures_hash_and_output_location_checked_before_worker(tmp_path):
    value = config(tmp_path)
    value["captures"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="checksum"):
        prepare_fit(write(tmp_path, value), output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    value = config(tmp_path)
    with pytest.raises(ValidationError, match="outside"):
        prepare_fit(write(tmp_path, value), output_dir=tmp_path / "native/out")


def test_failed_preparation_does_not_retry_or_claim_fit(tmp_path, monkeypatch):
    import agentinputlab.fit_preparation as preparation

    path = write(tmp_path, config(tmp_path))

    def launch(*args, **kwargs):
        raise preparation.subprocess.TimeoutExpired("fixture-worker", 30)

    monkeypatch.setattr(preparation.subprocess, "run", launch)
    out = tmp_path / "out"
    with pytest.raises(ExternalDependencyError, match="no automatic retry"):
        prepare_fit(path, output_dir=out)
    result = json.loads((out / "terminal.json").read_text())
    assert result == dict(status="preparation_failed", model_calls=0, fit_executed=False)
    with pytest.raises(ValidationError, match="exists"):
        prepare_fit(path, output_dir=out)


def test_preparation_worker_python_symlink_is_not_resolved(tmp_path):
    import sys

    value = config(tmp_path)
    link = tmp_path / "worker-python"
    link.symlink_to(sys.executable)
    value["fit"]["python_executable"] = "./worker-python"
    parsed = _configuration(write(tmp_path, value))
    assert parsed["fit"]["python_executable"] == str(link)


def skillopt_config(tmp_path, monkeypatch):
    from tests.unit.test_skillopt_state_configuration import configured_state

    value = config(tmp_path)
    fit = configured_state(tmp_path, monkeypatch)
    environment = fit.pop("environment")
    environment.pop("source_input")
    environment["checkout"] = "./native"
    value.pop("feedback")
    value.update(fit=fit, environment=environment)
    return value


def test_skillopt_prepares_state_fit_without_ace_feedback(tmp_path, monkeypatch):
    value = skillopt_config(tmp_path, monkeypatch)
    seen = fixture_native(monkeypatch, "statebench")
    parsed = _configuration(write(tmp_path, value))
    out = tmp_path / "prepared"
    out.mkdir()
    result = _build(parsed, out)
    assert result["model_calls"] == 0 and not result["fit_executed"]
    assert not result["scoring_executed"] and "factory" not in seen
    assert seen["builder"]["train_ids"] == value["groups"]["train"]
    ready = FitConfig.from_file(result["fit_configuration"]["path"])
    assert ready.value["release_id"] == "recipe.skillopt.offline"
    assert ready.value["environment"]["source_input"] == result["source_input"]
    assert "examples" not in ready.value and "feedback" not in ready.value


@pytest.mark.parametrize("change", ["overlap", "source", "unsupported", "feedback"])
def test_skillopt_preparation_rejects_invalid_inputs(tmp_path, monkeypatch, change):
    value = skillopt_config(tmp_path, monkeypatch)
    if change == "overlap":
        value["groups"]["validation"] = value["groups"]["train"]
    elif change == "source":
        value["environment"]["source_input"] = {"path": "old.json", "sha256": "a" * 64}
    elif change == "unsupported":
        value["environment"]["kind"] = "tooltalk"
    else:
        value["feedback"] = {"kind": "statebench", "checkout": "./native"}
    with pytest.raises(ValidationError):
        _configuration(write(tmp_path, value))
