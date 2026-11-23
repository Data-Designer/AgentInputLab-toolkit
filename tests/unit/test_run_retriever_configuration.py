"""Public configuration binding tests, using synthetic corpora only."""

import hashlib
import json

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.production_builders import PreparedCorpusRetrieverBuilder


def corpus_config(tmp_path):
    path = tmp_path / "corpus.jsonl"
    raw = b'{"id":"a","title":"Alpha","text":"alpha evidence"}\n'
    path.write_bytes(raw)
    return {"kind": "prepared_corpus", "benchmark_id": "beir.arguana",
            "profile_id": "sparse.bm25.memory.v1", "corpus_path": str(path),
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "corpus_revision": "synthetic-v1", "expected_document_count": 1,
            "options": {"k1": 1.4, "b": 0.5}}


def test_prepared_corpus_config_builds_existing_retriever(tmp_path):
    config = RunConfig("baseline.no_op", retriever=corpus_config(tmp_path))
    assert config.validate()["runtime_ready"] is None
    retriever = config._configured_context(None).retriever
    assert retriever.retrieve("alpha", top_k=1)[0].document.doc_id == "a"
    fingerprint = to_jsonable(retriever.fingerprint())
    assert "synthetic-v1" in json.dumps(fingerprint)
    assert "1.4" in json.dumps(fingerprint)


def test_validation_does_not_open_corpus_or_load_dense_model(tmp_path, monkeypatch):
    reference = corpus_config(tmp_path)
    (tmp_path / "corpus.jsonl").unlink()
    monkeypatch.setattr(PreparedCorpusRetrieverBuilder, "build",
                        lambda *args: pytest.fail("validation loaded runtime"))
    config = RunConfig("baseline.no_op", retriever=reference)
    assert config.validate()["runtime_ready"] is None
    reference.update(profile_id="dense.e5.base_v2", options={"model_path": str(tmp_path / "absent")})
    assert RunConfig("baseline.no_op", retriever=reference).validate()["runtime_ready"] is None


@pytest.mark.parametrize("change", ["digest", "count", "duplicates"])
def test_wrong_corpus_is_rejected_before_model_execution(tmp_path, change):
    reference = corpus_config(tmp_path)
    path = tmp_path / "corpus.jsonl"
    if change == "digest":
        reference["corpus_sha256"] = "0" * 64
    elif change == "count":
        reference["expected_document_count"] = 2
    else:
        raw = path.read_bytes() * 2
        path.write_bytes(raw)
        reference["corpus_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ArtifactError):
        RunConfig("baseline.no_op", retriever=reference)._configured_context(None)


def test_relative_corpus_and_dense_model_files_follow_config(tmp_path, monkeypatch):
    reference = corpus_config(tmp_path)
    reference.update(corpus_path="corpus.jsonl", profile_id="dense.e5.base_v2",
                     options={"model_path": "model"},
                     artifact_files=[{"name": "config", "path": "model/config.json", "sha256": "a" * 64}])
    path = tmp_path / "run.json"
    path.write_text(json.dumps(RunConfig("baseline.no_op", retriever=reference).to_dict()))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    assert config.retriever["corpus_path"] == str(tmp_path / "corpus.jsonl")
    assert config.retriever["options"]["model_path"] == str(tmp_path / "model")
    assert config.retriever["artifact_files"][0]["path"] == str(tmp_path / "model/config.json")
    assert config.validate()["valid"] is True


@pytest.mark.parametrize("update", [
    {"factory": "os.system"}, {"schema_version": 3},
    {"options": {"unknown": 1}}, {"corpus_sha256": "not-a-digest"},
])
def test_invalid_reference_rejected_without_read(tmp_path, update):
    reference = corpus_config(tmp_path)
    reference.update(update)
    with pytest.raises((ValidationError, ArtifactError)):
        RunConfig("baseline.no_op", retriever=reference).validate()


def test_benchmark_evaluation_rejects_cross_dataset_corpus_before_loading(tmp_path):
    reference = corpus_config(tmp_path)
    (tmp_path / "corpus.jsonl").unlink()
    config = RunConfig("baseline.no_op", retriever=reference)
    with pytest.raises(ValidationError, match="retriever benchmark"):
        config.evaluate("bright.earth_science", split="test", limit=1)
