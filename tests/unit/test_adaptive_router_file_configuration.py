"""Public file binding; these tests do not claim model-quality acceptance."""

import hashlib
import json
from dataclasses import asdict

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts import ValidationError
from tests.algorithms.unit.test_adaptive_rag_common_support import config as method_config


def run_config(path="router", digest="a" * 64, **options):
    return RunConfig(
        "acquire.adaptive_rag.common_support",
        parameters={"config": asdict(method_config())},
        artifacts={"router": {"path": str(path), "sha256": digest, **options}},
    )


def test_validate_router_binding_without_reading_or_loading(monkeypatch):
    from agentinputlab.algorithms.acquire import adaptive_rag_common_support as module

    def forbidden(*args, **kwargs):
        pytest.fail("shape validation must not read or construct the router")

    monkeypatch.setattr(module, "T5AdaptiveRAGRouter", forbidden)
    assert run_config().validate()["runtime_ready"] is None


def test_relative_router_binding_uses_original_constructor(tmp_path, monkeypatch):
    from agentinputlab.algorithms.acquire import adaptive_rag_common_support as module

    seen = []
    sentinel = object()
    monkeypatch.setattr(module, "T5AdaptiveRAGRouter", lambda **kw: seen.append(kw) or sentinel)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run_config(device="cpu").to_dict()))
    monkeypatch.chdir(tmp_path.parent)
    method = RunConfig.from_file(path).create_intervention()
    assert method.router is sentinel
    assert method.config == method_config()
    assert seen == [{"artifact_root": tmp_path / "router", "manifest_sha256": "a" * 64,
                     "device": "cpu"}]


@pytest.mark.parametrize("device", [None, 1, "", "cpu; run", "cuda:-1", "random"])
def test_invalid_router_device_is_rejected(device):
    with pytest.raises(ValidationError, match="device"):
        run_config(device=device).validate()


def test_other_profile_and_arbitrary_factory_rejected():
    with pytest.raises(ValidationError, match="not supported"):
        RunConfig("baseline.no_op", artifacts=run_config().artifacts).validate()
    with pytest.raises(ValidationError, match="requires"):
        run_config(factory="arbitrary.module").validate()


def test_original_manifest_checks_and_no_model_load(tmp_path):
    from agentinputlab.algorithms.acquire.adaptive_rag.runtime import ADAPTIVE_RAG_UPSTREAM_COMMIT

    files = {"config.json": "{}", "tokenizer_config.json": "{}", "model.safetensors": "fixture"}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    manifest = {
        "schema_version": "agentinputlab.adaptive-rag-router-artifact.v1",
        "upstream_commit": ADAPTIVE_RAG_UPSTREAM_COMMIT,
        "route_rule": "first_decoder_step_logit_argmax_A_first",
        "strategy_map": {"A": "no_retrieval", "B": "single_retrieval", "C": "ircot"},
        "upstream_router_checkpoint_parity_claimed": False,
        "paper_result_parity_claimed": False,
        "artifact_files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
    }
    raw = json.dumps(manifest).encode()
    (tmp_path / "artifact-manifest.json").write_bytes(raw)
    config = run_config(tmp_path, hashlib.sha256(raw).hexdigest())
    router = config.create_intervention().router
    assert router.device == "cpu"
    assert router._model is None and router._tokenizer is None
    (tmp_path / "model.safetensors").write_text("changed")
    with pytest.raises(ValidationError, match="closure mismatch"):
        config.create_intervention()


def test_duplicate_router_rejected():
    config = run_config().to_dict()
    config["parameters"]["router"] = {}
    with pytest.raises(ValidationError, match="same field"):
        RunConfig.from_dict(config).validate()
