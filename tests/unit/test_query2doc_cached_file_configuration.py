"""File entry for exact author caches; fixtures are not scientific results."""
from dataclasses import replace

import pytest

from agentinputlab import FrozenAgentState, RunConfig
from agentinputlab.algorithms.reformulate.query2doc.official_cached import Query2DocCache
from agentinputlab.contracts import ArtifactError, ValidationError


def configuration(path):
    return RunConfig.from_dict({"release_id": "rewrite.query2doc.official_cached",
        "parameters": {"split": "trec_dl2019"}, "artifacts": {"cache": {
            "path": str(path), "split": "trec_dl2019", "accept_license_review": True}}})


def test_shape_validation_without_reading_or_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(Query2DocCache, "from_official_jsonl", lambda *a, **k: pytest.fail("read"))
    assert configuration(tmp_path / "absent.jsonl").validate()["valid"]


def test_file_binding_calls_original_official_loader(tmp_path, monkeypatch):
    calls = []
    def loader(path, **kwargs):
        calls.append((path, kwargs))
        raise ArtifactError("test loader reached")
    monkeypatch.setattr(Query2DocCache, "from_official_jsonl", loader)
    with pytest.raises(ArtifactError, match="test loader reached"):
        configuration(tmp_path / "cache.jsonl").create_intervention()
    assert calls == [(str(tmp_path / "cache.jsonl"),
                      {"split": "trec_dl2019", "accept_license_review": True})]


@pytest.mark.parametrize("field,value", [("split", "train"), ("split", []),
    ("accept_license_review", False), ("accept_license_review", "true"),
    ("path", ""), ("factory", "arbitrary.factory")])
def test_rejects_unsupported_cache_options(tmp_path, field, value):
    data = configuration(tmp_path / "cache").to_dict()
    data["artifacts"]["cache"][field] = value
    with pytest.raises(ValidationError):
        RunConfig.from_dict(data).validate()


def test_split_cannot_differ_from_native_cache(tmp_path):
    config = configuration(tmp_path / "cache")
    with pytest.raises(ValidationError, match="split"):
        replace(config, parameters={"split": "trec_dl2020"}).validate()


def test_synthetic_cache_cannot_impersonate_official(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text('{"query_id":"1","query":"q","pseudo_doc":"d"}\n')
    with pytest.raises(ArtifactError, match="size/checksum"):
        configuration(path).run(FrozenAgentState(task_id="1", query="q"))
