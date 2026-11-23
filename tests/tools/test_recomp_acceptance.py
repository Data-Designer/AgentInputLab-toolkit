from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest


@pytest.fixture
def module(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "recomp_acceptance_test", tools / "run_recomp_acceptance.py"
    )
    value = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, value)
    spec.loader.exec_module(value)
    return value


def test_capture_identity_is_explicitly_not_live_retrieval(module):
    source = module.CapturedContextSource("qa.test", "fixed-id", "a" * 64)
    assert source.fingerprint() == {
        "provider": "benchmark_visible_context_capture",
        "benchmark": "qa.test",
        "case_id": "fixed-id",
        "context_hash": "a" * 64,
        "live_retrieval_performed": False,
    }


def test_failed_transport_retains_unknown_consumption_and_resource(module, tmp_path):
    class Delegate:
        last_resource_receipt: ClassVar[dict] = {
            "provider_attempted": True,
            "tokens": "unknown",
        }

        def call(self, payload):
            raise RuntimeError("model process exited after start")

        def fingerprint(self):
            return {"worker": "real-deployment"}

    worker = module.RecordedWorker(Delegate(), tmp_path)
    with pytest.raises(RuntimeError, match="after start"):
        worker.call({"request_id": "fixed"})
    error = json.loads((tmp_path / "worker-01-transport-error.json").read_text())
    assert error["consumption"] == "unknown_if_process_failed_after_start"
    assert (
        json.loads((tmp_path / "worker-01-resource.json").read_text())
        == Delegate.last_resource_receipt
    )
    assert worker.responses == []
    assert not (tmp_path / "worker-01-response.json").exists()
