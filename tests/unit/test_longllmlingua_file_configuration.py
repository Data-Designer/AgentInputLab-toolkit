"""Public controlled compression bindings; native execution has separate evidence."""
import importlib
import json

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts import InterventionContext, ValidationError
from agentinputlab.integrations.external_workers.longllmlingua import LongLLMLinguaNativeRuntime


def runtime(tmp_path):
    return {"kind": "longllmlingua_native", "snapshot": str(tmp_path / "model"),
            "model_id": "test/model", "revision": "a" * 40,
            "file_sha256": {name: "b" * 64 for name in
                            ("config.json", "tokenizer_config.json", "special_tokens_map.json",
                             "tokenizer.json", "model.safetensors")},
            "device_map": "cpu", "torch_dtype": "float32"}


def config(tmp_path):
    return RunConfig.from_dict({"release_id": "curate.longllmlingua.controlled",
        "runtime": runtime(tmp_path), "seed": 19,
        "backend": {"kind": "openai_compatible", "model_id": "reader",
                    "base_url": "http://127.0.0.1:1/v1", "deployment_identity": {"revision": "test"}},
        "budget": {"max_output_tokens": 4096, "max_model_calls": 8},
        "parameters": {"config": {"rate": 0.7, "context_budget": "+100"}}})


def preparation():
    return importlib.import_module("agentinputlab.longllmlingua_configuration").prepare


def test_file_runtime_validation_does_not_load(tmp_path, monkeypatch):
    monkeypatch.setattr(LongLLMLinguaNativeRuntime, "__init__", lambda *a, **k: pytest.fail("load"))
    assert config(tmp_path)._runtime(load=False) is None


def test_file_runtime_uses_original_constructor(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    monkeypatch.setattr(LongLLMLinguaNativeRuntime, "_source", staticmethod(lambda: {"test": "source"}))
    result = config(tmp_path)._runtime(load=True)
    assert type(result) is LongLLMLinguaNativeRuntime
    assert result.snapshot == tmp_path / "model"
    assert result.fingerprint()["load"]["torch_dtype"] == "float32"
    assert result.fingerprint()["injected_factory"] is False


def test_relative_snapshot_follows_config_file(tmp_path, monkeypatch):
    value = config(tmp_path).to_dict()
    value["runtime"]["snapshot"] = "model"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value))
    monkeypatch.chdir(tmp_path.parent)
    assert RunConfig.from_file(path).runtime["snapshot"] == str(tmp_path / "model")


@pytest.mark.parametrize("field,value", [("device_map", "cuda:1"), ("torch_dtype", "int8"),
    ("revision", "main"), ("snapshot", ""), ("file_sha256", {"../model": "a" * 64}),
    ("compressor_factory", "arbitrary.factory")])
def test_runtime_shape_rejects_invalid_options(tmp_path, field, value):
    data = config(tmp_path).to_dict()
    data["runtime"][field] = value
    with pytest.raises(ValidationError):
        RunConfig.from_dict(data)._runtime(load=False)


def test_other_profile_cannot_use_controlled_runtime(tmp_path):
    data = config(tmp_path).to_dict()
    data["release_id"] = "curate.longllmlingua.official"
    with pytest.raises(ValidationError, match="controlled"):
        RunConfig.from_dict(data)._runtime(load=False)


def test_prepare_keeps_budget_settings_and_no_forward(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    monkeypatch.setattr(LongLLMLinguaNativeRuntime, "_source", staticmethod(lambda: {"test": "source"}))
    monkeypatch.setattr(LongLLMLinguaNativeRuntime, "run", lambda *a: pytest.fail("compression"))
    source = config(tmp_path)
    receipt = preparation()(source, tmp_path / "ready")
    assert receipt["model_calls"] == receipt["compression_calls"] == 0
    ready = RunConfig.from_file(receipt["run_path"])
    method = ready.create_intervention()
    assert method.bindings.budget == source.budget and method.bindings.seed == 19
    assert method.bindings.config.rate == 0.7 and method.bindings.config.context_budget == "+100"
    method.bindings.require_match(ready._configured_context(None))
    with pytest.raises(ValidationError, match="configured twice"):
        ready._configured_context(InterventionContext(external_runtime=object()))
    changed = ready.to_dict()
    changed["runtime"]["torch_dtype"] = "float16"
    with pytest.raises(ValidationError, match="compressor differs"):
        method.bindings.require_match(RunConfig.from_dict(changed)._configured_context(None))
    with pytest.raises(ValidationError, match="new output"):
        preparation()(source, tmp_path / "ready")


def test_cli_prepare(tmp_path, monkeypatch, capsys):
    from agentinputlab.cli.main import main

    (tmp_path / "model").mkdir()
    monkeypatch.setattr(LongLLMLinguaNativeRuntime, "_source", staticmethod(lambda: {"test": "source"}))
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config(tmp_path).to_dict()))
    assert main(["config", "prepare-longllmlingua", str(path), "--output-dir", str(tmp_path / "ready")]) == 0
    assert json.loads(capsys.readouterr().out)["compression_calls"] == 0
