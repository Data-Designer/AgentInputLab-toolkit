import json

import pytest
from run_clariq_acceptance import select_zero_call_entry_repairs


def test_entry_repair_selects_only_proven_pre_call_guards(tmp_path):
    jobs = [{"id": "guard"}, {"id": "consumed"}]
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs, "source": {"fixture": True}}))
    directory = tmp_path / "terminals"
    directory.mkdir()
    guard = {"job": jobs[0], "error": "ReFormeR context retriever dependency is required",
             "status": "failed_terminal", "stage": "intervention", "calls": [],
             "monitor_snapshots": [{"usage": {"model_calls": 0}}]}
    (directory / "guard.json").write_text(json.dumps(guard))
    (directory / "consumed.json").write_text(json.dumps({
        "job": jobs[1], "error": "actual empty generation", "calls": [{"request": "real"}]}))
    selected, repairs, old = select_zero_call_entry_repairs(jobs, tmp_path, {"fixture": True})
    assert selected == [jobs[0]]
    assert set(repairs) == {"guard"}
    assert len(old) == 2
    guard["monitor_snapshots"][0]["usage"]["model_calls"] = 1
    (directory / "guard.json").write_text(json.dumps(guard))
    with pytest.raises(ValueError, match="consumed or ambiguous"):
        select_zero_call_entry_repairs(jobs, tmp_path, {"fixture": True})
