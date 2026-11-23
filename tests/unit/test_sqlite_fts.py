import pytest

from agentinputlab.backends import Document
from agentinputlab.backends.retrieval.sqlite_fts import SQLiteFTS5Retriever
from agentinputlab.contracts import ArtifactError


def test_stream_build_reopen_and_safe_query(tmp_path):
    identity = {"archive_sha256": "a" * 64, "split": "test"}
    docs = (
        Document(doc_id=str(i), text=text, corpus_revision="source-v1")
        for i, text in enumerate(["apple orange", "orange pear", "a banana"])
    )
    r = SQLiteFTS5Retriever.build(tmp_path / "index.sqlite", docs, corpus_identity=identity)
    assert r.fingerprint()["document_count"] == 3
    assert len(r.retrieve("orange", top_k=5)) == 2
    assert r.retrieve('" OR * --', top_k=5) == ()
    assert r.retrieve("!?") == ()
    reopened = SQLiteFTS5Retriever(tmp_path / "index.sqlite", corpus_identity=identity)
    assert reopened.retrieve("apple")[0].document.doc_id == "0"
    with pytest.raises(ArtifactError, match="identity"):
        SQLiteFTS5Retriever(tmp_path / "index.sqlite", corpus_identity={"archive_sha256": "b" * 64})


def test_duplicate_ids_fail_without_publishing_partial_index(tmp_path):
    target = tmp_path / "index.sqlite"
    with pytest.raises(ArtifactError, match="duplicate"):
        SQLiteFTS5Retriever.build(
            target, [Document("x", "one"), Document("x", "two")], corpus_identity={"a": 1}
        )
    assert not target.exists()


@pytest.mark.parametrize("same_identity", [True, False])
def test_concurrent_builder_reuses_only_matching_published_index(tmp_path, monkeypatch, same_identity):
    import os

    target = tmp_path / "shared.sqlite"
    winner = tmp_path / "winner.sqlite"
    identity = {"archive_sha256": "a" * 64}
    winner_identity = identity if same_identity else {"archive_sha256": "b" * 64}
    SQLiteFTS5Retriever.build(winner, [Document("winner", "apple")], corpus_identity=winner_identity)
    winner_bytes = winner.read_bytes()
    link = os.link

    def publish_other_builder_first(source, destination):
        link(winner, destination)
        return link(source, destination)

    monkeypatch.setattr(os, "link", publish_other_builder_first)
    if same_identity:
        retriever = SQLiteFTS5Retriever.build(target, [Document("loser", "pear")], corpus_identity=identity)
        assert retriever.retrieve("apple")[0].document.doc_id == "winner"
    else:
        with pytest.raises(ArtifactError, match="identity"):
            SQLiteFTS5Retriever.build(target, [Document("loser", "pear")], corpus_identity=identity)
    assert target.read_bytes() == winner_bytes
    assert not list(tmp_path.glob(".fts-build-*"))
