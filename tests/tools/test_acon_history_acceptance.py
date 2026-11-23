import json
from types import ModuleType, SimpleNamespace

import pytest
from run_acon_history_acceptance import implementation_retest_jobs, record_native_calls


def test_recorder_ignores_sdk_not_given_but_preserves_request_and_result(monkeypatch):
    module = ModuleType("openai")

    class NotGiven:
        pass

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
                model_dump=lambda **args: {"model": "real-alias", "choices": [], "usage": {"completion_tokens": 5}})))

    module.OpenAI, module.NotGiven = Client, NotGiven
    monkeypatch.setitem(__import__("sys").modules, "openai", module)
    calls = record_native_calls()
    client = module.OpenAI()
    client.chat.completions.create(model="real-alias", seed=17, max_tokens=512, extra_body=NotGiven(),
                                   api_key="do-not-save")
    assert len(calls) == 1
    assert "extra_body" not in calls[0]["request"]
    assert "do-not-save" not in json.dumps(calls)
    assert calls[0]["response"]["usage"]["completion_tokens"] == 5
    client.chat.completions.create(model="real-alias", seed=NotGiven(), max_tokens=512)
    assert "seed" not in calls[1]["request"]


def test_implementation_retests_do_not_select_actual_model_failures_or_scores(tmp_path):
    jobs = [{"id": "bad-recorder", "release": "baseline.no_op"},
            {"id": "real-failure", "release": "baseline.no_op"},
            {"id": "scored-zero", "release": "baseline.no_op"}]
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs, "model_bindings": {"fixture": True}}))
    (tmp_path / "complete.json").write_text("{}")
    for job in jobs:
        folder = tmp_path / job["id"]
        folder.mkdir()
        row = {"job": job, "status": "failed_terminal", "error": "real budget exhausted"}
        if job["id"] == "bad-recorder":
            row.update(error_type="TypeError", error="Object of type NotGiven is not JSON serializable",
                       traceback="instrumentation: digest(visible)")
        if job["id"] == "scored-zero":
            row.update(status="scored_terminal", metrics={"official_similarity": 0.0})
        (folder / "terminal.json").write_text(json.dumps(row))
    selected, sources, untouched = implementation_retest_jobs(tmp_path, {"fixture": True})
    assert len(selected) == 1 and selected[0]["prior_job_id"] == "bad-recorder"
    assert selected[0]["attempt_ordinal"] == 2
    assert len(untouched) == 2 and len(sources) == 1
    with pytest.raises(ValueError, match="identities differ"):
        implementation_retest_jobs(tmp_path, {"fixture": False})
