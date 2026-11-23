from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "adaptive_acceptance_test", tools / "run_adaptive_acceptance.py"
    )
    value = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, value)
    spec.loader.exec_module(value)
    return value


def wheel(path, *, router, predictor):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("agentinputlab/algorithms/acquire/adaptive_rag_common_support.py", router)
        archive.writestr("agentinputlab/experiments/production_builders.py", predictor)


def test_baseline_reuse_rejects_any_other_execution_code_change(module, tmp_path):
    old, new = tmp_path / "old.whl", tmp_path / "new.whl"
    wheel(old, router="old router", predictor="same predictor")
    wheel(new, router="strict artifact guard", predictor="same predictor")
    assert module.baseline_execution_code_unchanged(old, new) == [
        "agentinputlab/algorithms/acquire/adaptive_rag_common_support.py"
    ]
    wheel(new, router="strict artifact guard", predictor="changed predictor")
    with pytest.raises(ValueError, match="non-router package bytes changed"):
        module.baseline_execution_code_unchanged(old, new)


def test_saved_neural_route_cannot_be_used_for_another_question(module):
    value = object()
    router = module.FrozenQuestionRoute("exact visible question", value, {"source": "fixture"})
    assert router.route("exact visible question") is value
    with pytest.raises(ValueError, match="different visible question"):
        router.route("another question")
