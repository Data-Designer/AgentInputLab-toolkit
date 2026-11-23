"""ToolTalk evidence survives scoring and never replaces a consumed attempt."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from types import ModuleType

import pytest

from workers.official_arenas.protocol import OfficialArenaError, content_hash
from workers.official_arenas.providers import tooltalk


@pytest.fixture
def episode(tmp_path, monkeypatch):
    conversation = {
        "conversation_id": "fixed-id",
        "conversation": [{"role": "user", "text": "测试 query"}],
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(conversation), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    calls = []
    metrics = dict(success=False, precision=0.25, recall=0.5, action_precision=0.0,
                   bad_action_rate=1.0, soft_success=0.125)

    class Predictor:
        hard_stops = 1
        history_usage = {}

        def __init__(self, **kwargs):
            calls.append(("predictor", kwargs))

        def usage(self):
            return {"model_calls": 1, "input_tokens": 9, "output_tokens": 4}

    class Executor:
        def __init__(self, **kwargs):
            calls.append(("executor", kwargs))

        def run_conversation(self, value, predictor):
            calls.append(("run", None))
            value["conversation"].append({"role": "assistant", "predictions": [
                {"api_name": "Example", "parameters": {"value": "原始 output"}}]})
            return value

        def evaluate_predictions(self, value):
            calls.append(("score", None))
            value["metrics"] = copy.deepcopy(metrics)
            return value

    package = ModuleType("tooltalk")
    evaluation = ModuleType("tooltalk.evaluation")
    executor = ModuleType("tooltalk.evaluation.tool_executor")
    executor.ToolExecutor = Executor
    for name, module in (("tooltalk", package), ("tooltalk.evaluation", evaluation),
                         ("tooltalk.evaluation.tool_executor", executor)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(tooltalk, "_Predictor", Predictor)
    monkeypatch.setattr(tooltalk, "_install_sent2vec_compatibility", lambda: None)
    monkeypatch.setattr(tooltalk, "materialized_state", lambda payload: {})
    monkeypatch.setattr(tooltalk, "require_materialized_budget", lambda state, config: {"max_model_calls": 8})
    monkeypatch.setattr(tooltalk, "intervention_usage", lambda payload: {"model_calls": 0})
    monkeypatch.setenv("TOOLTALK_TEST_KEY", "synthetic-only")
    provider = tooltalk.ToolTalkProvider()
    monkeypatch.setattr(provider, "_validate_config", lambda config: {
        "agent_key_env": "TOOLTALK_TEST_KEY", "agent_model": "fake-model",
        "agent_base_url": "http://not-contacted.invalid/v1",
    })
    replay = {"checkpoint_id": "fixed-checkpoint", "protected": {
        "relative_path": str(source), "conversation_sha256": tooltalk._sha256(source),
        "conversation_id": "fixed-id", "dataset_part": "easy",
    }}

    def run():
        return provider.run_arm({}, replay_record=replay, arm_payload={}, seed=17,
                                role="candidate", work_dir=work)

    run.provider = provider
    return work, calls, metrics, Executor, run, source, conversation


def test_full_native_episode_and_scores_are_retained_without_metric_changes(episode):
    work, calls, metrics, _, run, source, conversation = episode
    result = run()
    prediction = json.loads((work / "predicted-conversation.json").read_text())
    evaluated = json.loads((work / "native-evaluation.json").read_text())
    trace = json.loads((work / "trajectory.json").read_text())
    assert "metrics" not in prediction
    assert evaluated == dict(prediction, metrics=metrics)
    assert evaluated["conversation"][-1]["predictions"][0]["parameters"]["value"] == "原始 output"
    assert trace == result.trace
    assert trace["prediction_trace_sha256"] == content_hash(evaluated["conversation"])
    for artifact in trace["native_artifacts"].values():
        assert hashlib.sha256((work / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert result.metrics == dict(official_success=0.0, precision=0.25, recall=0.5,
                                  action_precision=0.0, bad_action_rate=1.0,
                                  soft_success=0.125, adapter_hard_stops=1.0)
    assert result.usage == {"model_calls": 1, "input_tokens": 9, "output_tokens": 4}
    assert json.loads(source.read_text()) == conversation
    assert [call[0] for call in calls] == ["predictor", "executor", "run", "score"]


def test_native_episode_survives_scorer_failure(episode, monkeypatch):
    work, calls, _, executor, run, _, _ = episode

    def fail(self, value):
        raise RuntimeError("native scorer failed")

    monkeypatch.setattr(executor, "evaluate_predictions", fail)
    with pytest.raises(RuntimeError, match="native scorer failed"):
        run()
    assert len(json.loads((work / "predicted-conversation.json").read_text())["conversation"]) == 2
    assert not (work / "native-evaluation.json").exists()
    assert not (work / "trajectory.json").exists()
    calls.clear()
    with pytest.raises(OfficialArenaError, match="fresh work directory"):
        run()
    assert not calls


def test_live_history_receipt_is_saved_and_bound_to_native_trace(episode):
    from types import SimpleNamespace
    work, _, _, _, run, _, _ = episode
    run.provider.history_transform = SimpleNamespace(
        fingerprint=lambda: {"scope": "fixture-not-neural-evidence"}, records=[{"status": "fixture"}])
    result = run()
    receipt = json.loads((work / "live-history.json").read_text())
    assert receipt["records"] == [{"status": "fixture"}]
    assert result.trace["live_history_artifact"] == {
        "path": "live-history.json", "sha256": tooltalk._sha256(work / "live-history.json")}


def test_fresh_run_installs_native_import_shim_before_predictor(episode, monkeypatch):
    _, _, _, _, run, _, _ = episode
    installed = []
    original = tooltalk._Predictor

    class Guarded(original):
        def __init__(self, **kwargs):
            assert installed == [True]
            super().__init__(**kwargs)

    monkeypatch.setattr(tooltalk, "_install_sent2vec_compatibility", lambda: installed.append(True))
    monkeypatch.setattr(tooltalk, "_Predictor", Guarded)
    run()
    assert installed == [True]


def test_live_history_receipt_survives_native_episode_failure(episode, monkeypatch):
    from types import SimpleNamespace
    work, calls, _, executor, run, _, _ = episode
    run.provider.history_transform = SimpleNamespace(
        fingerprint=lambda: {"scope": "fixture"}, records=[{"status": "failed_terminal"}])

    def fail(self, value, predictor):
        raise RuntimeError("fixture consumed native turn failed")

    monkeypatch.setattr(executor, "run_conversation", fail)
    with pytest.raises(RuntimeError, match="native turn failed"):
        run()
    assert json.loads((work / "live-history.json").read_text())["records"] == [{"status": "failed_terminal"}]
    calls.clear()
    with pytest.raises(OfficialArenaError, match="fresh work directory"):
        run()
    assert not calls


@pytest.mark.parametrize("name", ["predicted-conversation.json", "native-evaluation.json", "trajectory.json"])
def test_existing_artifact_rejected_before_model_initialization(episode, name):
    work, calls, _, _, run, _, _ = episode
    artifact = work / name
    artifact.write_bytes(b"original attempt\n")
    with pytest.raises(OfficialArenaError, match="fresh work directory"):
        run()
    assert artifact.read_bytes() == b"original attempt\n"
    assert not calls
