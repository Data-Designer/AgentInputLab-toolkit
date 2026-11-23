"""Synthetic local index tests; these are not real-model acceptance claims."""
import hashlib
import json

import pytest

from agentinputlab import RunConfig
from agentinputlab.backends.retrieval import Document, SQLiteFTS5Retriever
from agentinputlab.contracts import ArtifactError, ValidationError


def binding(tmp_path):
    path = tmp_path / "index.sqlite"
    identity = {"revision": "synthetic-v1", "sample": True}
    original = SQLiteFTS5Retriever.build(path, [Document("a", "apple evidence"),
        Document("b", "pear evidence")], corpus_identity=identity)
    return original, dict(kind="sqlite_fts5", index_path=str(path),
        index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        corpus_identity=identity, expected_document_count=2, benchmark_id="beir.arguana")


def test_file_binding_preserves_native_results_fingerprint_and_file(tmp_path, monkeypatch):
    original, reference = binding(tmp_path)
    path = tmp_path / "run.json"
    reference["index_path"] = "index.sqlite"
    path.write_text(json.dumps(RunConfig("baseline.no_op", retriever=reference).to_dict()))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    assert config.retriever["index_path"] == str(tmp_path / "index.sqlite")
    current = config._retriever()
    assert current.fingerprint() == original.fingerprint()
    assert current.retrieve("apple pear", top_k=2) == original.retrieve("apple pear", top_k=2)
    assert hashlib.sha256(original.path.read_bytes()).hexdigest() == reference["index_sha256"]


def test_shape_validation_never_opens_or_rebuilds_index(tmp_path, monkeypatch):
    _, reference = binding(tmp_path)
    (tmp_path / "index.sqlite").unlink()
    monkeypatch.setattr(SQLiteFTS5Retriever, "build", lambda *a, **kw: pytest.fail("rebuilt index"))
    config = RunConfig("baseline.no_op", retriever=reference)
    assert config.validate()["valid"]
    with pytest.raises(ArtifactError, match="existing regular"):
        config._retriever()
    assert not (tmp_path / "index.sqlite").exists()


@pytest.mark.parametrize("change", [{"index_sha256": "bad"}, {"expected_document_count": True},
    {"expected_document_count": 0}, {"index_path": ""}, {"corpus_identity": {}},
    {"corpus_identity": []}, {"benchmark_id": ""}, {"k1": 1.2}])
def test_invalid_configuration_is_rejected_without_loading(tmp_path, change):
    _, reference = binding(tmp_path)
    reference.update(change)
    with pytest.raises(ValidationError):
        RunConfig("baseline.no_op", retriever=reference).validate()


@pytest.mark.parametrize("change", [{"index_sha256": "0" * 64},
    {"expected_document_count": 3}, {"corpus_identity": {"revision": "other"}}])
def test_incorrect_frozen_binding_is_rejected(tmp_path, change):
    _, reference = binding(tmp_path)
    reference.update(change)
    with pytest.raises(ArtifactError):
        RunConfig("baseline.no_op", retriever=reference)._retriever()


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_unpinned_sidecars_are_rejected(tmp_path, suffix):
    _, reference = binding(tmp_path)
    (tmp_path / ("index.sqlite" + suffix)).write_bytes(b"unbound")
    with pytest.raises(ArtifactError, match="standalone"):
        RunConfig("baseline.no_op", retriever=reference)._retriever()


def test_symlink_and_cross_benchmark_rejected(tmp_path):
    _, reference = binding(tmp_path)
    link = tmp_path / "link.sqlite"
    link.symlink_to(tmp_path / "index.sqlite")
    reference["index_path"] = str(link)
    config = RunConfig("baseline.no_op", retriever=reference)
    with pytest.raises(ArtifactError, match="regular"):
        config._retriever()
    with pytest.raises(ValidationError, match="retriever benchmark"):
        config.evaluate("bright.earth_science", split="test", limit=1)
