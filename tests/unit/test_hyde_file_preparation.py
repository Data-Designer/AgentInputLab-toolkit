"""Controlled identity preparation, with no neural execution in these tests."""
import json

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts import ValidationError
from agentinputlab.hyde_configuration import prepare
from tests.unit.test_run_vector_configuration import VectorRuntime, reference


class Backend:
    def generate(self, request):
        pytest.fail("preparation must not generate")

    def fingerprint(self):
        return {"model": "test-generator", "revision": "frozen"}


def setup(tmp_path, monkeypatch):
    from agentinputlab.experiments.production_builders import PreparedCorpusRetrieverBuilder

    runtime = VectorRuntime()
    monkeypatch.setattr(runtime, "embed", lambda *a: pytest.fail("preparation must not embed"))
    monkeypatch.setattr(runtime, "search", lambda *a, **k: pytest.fail("preparation must not search"))
    monkeypatch.setattr(PreparedCorpusRetrieverBuilder, "build", lambda *a: runtime)
    monkeypatch.setattr(RunConfig, "_backend", lambda self: Backend())
    value = {"release_id": "acquire.hyde.controlled", "backend": {"kind": "test"},
             "vector_runtime": reference(tmp_path), "seed": 19,
             "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
             "parameters": {"config": {"num_hypotheses": 8, "max_tokens": 512, "temperature": 0.7}}}
    return value, runtime


def test_prepare_preserves_eight_calls_budget_and_identity(tmp_path, monkeypatch):
    value, runtime = setup(tmp_path, monkeypatch)
    proof = prepare(RunConfig.from_dict(value), tmp_path / "ready")
    assert proof["model_calls"] == proof["embedding_calls"] == proof["retrieval_calls"] == 0
    ready = RunConfig.from_file(proof["run_path"])
    method = ready.create_intervention()
    assert method.bindings.budget == ready.budget == RunConfig.from_dict(value).budget
    assert method.bindings.config.num_hypotheses == 8
    assert method.bindings.config.max_tokens == 512 and ready.seed == 19
    method.bindings.require_match(ready._configured_context(None))
    monkeypatch.setattr(runtime, "fingerprint", lambda: {"changed": "vector"})
    with pytest.raises(ValidationError, match="fingerprint differs"):
        method.bindings.require_match(ready._configured_context(None))
    with pytest.raises(ValidationError, match="new output"):
        prepare(RunConfig.from_dict(value), tmp_path / "ready")


@pytest.mark.parametrize("change", ["release", "backend", "vector", "settings", "profile"])
def test_preparation_rejects_invalid_inputs(tmp_path, monkeypatch, change):
    value, _ = setup(tmp_path, monkeypatch)
    if change == "release":
        value["release_id"] = "acquire.hyde.paper_reimplementation"
    elif change == "backend":
        value.pop("backend")
    elif change == "vector":
        value.pop("vector_runtime")
    elif change == "settings":
        value["parameters"]["config"]["unknown"] = True
    else:
        value["parameters"]["profile_kind"] = "native_parity"
    with pytest.raises(ValidationError):
        prepare(RunConfig.from_dict(value), tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_public_preparation_command(tmp_path, monkeypatch, capsys):
    from agentinputlab.cli.main import main

    value, _ = setup(tmp_path, monkeypatch)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    assert main(["config", "prepare-hyde", str(path), "--output-dir", str(tmp_path / "ready")]) == 0
    proof = json.loads(capsys.readouterr().out)
    assert RunConfig.from_file(proof["run_path"]).validate()["valid"]
