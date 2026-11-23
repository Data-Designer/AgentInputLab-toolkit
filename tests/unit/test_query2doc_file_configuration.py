"""File-entry regressions; native model acceptance is recorded separately."""
import json
from dataclasses import replace

import pytest

from agentinputlab import RunConfig
from agentinputlab.algorithms.reformulate.query2doc import QUERYGYM_REVISION
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import to_jsonable


def configuration(tmp_path):
    return RunConfig("rewrite.query2doc.controlled_live", parameters={"config": {
        "mode": "zero_shot", "temperature": 0.7, "max_tokens": 256}},
        backend={"model_id": "test", "base_url": "http://127.0.0.1:1/v1",
                 "deployment_identity": {"revision": "fixture"}},
        retriever={"kind": "bm25", "documents": [{"doc_id": "d", "text": "evidence"}]},
        runtime={"kind": "querygym_query2doc", "checkout": str(tmp_path)})


def test_controlled_profile_can_bind_existing_querygym_runtime_without_loading(tmp_path):
    config = configuration(tmp_path)
    assert config._runtime(load=False) is None


def test_preparation_binds_actual_components_without_calls_and_is_create_only(tmp_path, monkeypatch):
    from agentinputlab.query2doc_configuration import prepare

    config = configuration(tmp_path)
    seen = []
    def runtime(self, *, load, backend=None):
        if not load:
            return None
        assert load and backend is not None
        seen.append(backend.fingerprint())
        class Runtime:
            def fingerprint(self):
                return {"backend": backend.fingerprint(), "revision": QUERYGYM_REVISION,
                        "source_closure_sha256": "a" * 64}
            def reformulate(self, request):
                pytest.fail("preparation must not generate")
        return Runtime()
    monkeypatch.setattr(RunConfig, "_runtime", runtime)
    out = tmp_path / "prepared"
    proof = prepare(config, out)
    assert proof["model_calls"] == proof["retrieval_calls"] == 0
    ready = RunConfig.from_file(out / "run.json")
    binding = ready.parameters["bindings"]
    assert to_jsonable(binding["generator"]) == to_jsonable(seen[0])
    assert to_jsonable(binding["retriever"]) == to_jsonable(config._retriever().fingerprint())
    assert binding["budget"] == to_jsonable(config.budget)
    assert binding["config"]["max_tokens"] == 256
    assert ready.seed == config.seed
    assert json.loads((out / "prepared.json").read_text())["run_sha256"] == proof["run_sha256"]
    with pytest.raises(ValidationError, match="new output directory"):
        prepare(config, out)
    assert len(seen) == 1


@pytest.mark.parametrize("change", [
    {"release_id": "rewrite.query2doc.querygym_live"}, {"backend": {}}, {"retriever": {}},
    {"runtime": {}}, {"parameters": {"bindings": {}}},
    {"parameters": {"config": {"mode": "few_shot", "demonstration_ids": ["dev1"]}}},
])
def test_preparation_rejects_missing_or_unsupported_binding_before_assets(tmp_path, change):
    from agentinputlab.query2doc_configuration import prepare
    with pytest.raises(ValidationError):
        prepare(replace(configuration(tmp_path), **change), tmp_path / "out")
    assert not (tmp_path / "out").exists()
