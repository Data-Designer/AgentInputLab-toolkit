import hashlib
import json
import zipfile

import pytest

from agentinputlab.benchmarks.beir_archive import BEIR_ARCHIVE_LOCKS, BEIRArchiveAdapter
from agentinputlab.contracts import ArtifactError, ValidationError


def test_test_qrels_only_and_gold_separation(tmp_path, monkeypatch):
    p = tmp_path / "quora.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "quora/corpus.jsonl",
            json.dumps({"_id": "d", "text": "visible evidence", "title": ""}) + "\n",
        )
        z.writestr(
            "quora/queries.jsonl",
            "\n".join(
                json.dumps({"_id": i, "text": q})
                for i, q in [("q", "test request"), ("train", "unused training request")]
            )
            + "\n",
        )
        z.writestr("quora/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq\td\t1\n")
    monkeypatch.setitem(
        BEIR_ARCHIVE_LOCKS,
        "beir.quora",
        {
            "dataset": "quora",
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "upstream_md5": "test-fixture",
        },
    )
    a = BEIRArchiveAdapter(benchmark_id="beir.quora", source_archive=p, license_acknowledged=True)
    assert len(a.examples("test")) == 1
    e = a.examples("test")[0]
    assert "relevance" not in e.visible_state().metadata
    assert next(iter(a.iter_documents())).doc_id == "d"
    assert e.evaluator_gold["relevance"] == {"d": 1}
    with pytest.raises(ValidationError):
        a.examples("train")


def test_source_requires_acknowledgement_and_exact_bytes(tmp_path):
    with pytest.raises(ValidationError, match="license_acknowledged"):
        BEIRArchiveAdapter(benchmark_id="beir.quora", source_archive="unused")
    p = tmp_path / "invalid.zip"
    p.write_bytes(b"not verified source")
    with pytest.raises(ArtifactError, match="size"):
        BEIRArchiveAdapter(benchmark_id="beir.quora", source_archive=p, license_acknowledged=True)


@pytest.mark.parametrize("overlap", [False, True])
def test_fiqa_dev_uses_only_its_official_qrels(tmp_path, monkeypatch, overlap):
    path = tmp_path / "fiqa.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fiqa/corpus.jsonl", '{"_id":"doc","text":"visible evidence"}\n')
        archive.writestr(
            "fiqa/queries.jsonl",
            "".join(json.dumps({"_id": key, "text": key}) + "\n" for key in ("a", "b", "c")),
        )
        for split, key in (("train", "a" if overlap else "b"), ("dev", "a"), ("test", "c")):
            archive.writestr(f"fiqa/qrels/{split}.tsv", f"query-id\tcorpus-id\tscore\n{key}\tdoc\t1\n")
    monkeypatch.setitem(BEIR_ARCHIVE_LOCKS, "beir.fiqa_2018", {
        "dataset": "fiqa", "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "upstream_md5": "fixture",
    })
    args = dict(benchmark_id="beir.fiqa_2018", source_archive=path, license_acknowledged=True)
    if overlap:
        with pytest.raises(ArtifactError, match="dev query overlaps train"):
            BEIRArchiveAdapter(**args, split="dev")
        return
    dev = BEIRArchiveAdapter(**args, split="dev")
    assert [e.example_id for e in dev.examples("dev")] == ["a"]
    assert dev.examples("dev")[0].visible_metadata["source_split"] == "dev"
    assert dev.fingerprint()["dev_queries"] == 1
    assert [e.example_id for e in BEIRArchiveAdapter(**args).examples("test")] == ["c"]
    with pytest.raises(ValidationError):
        dev.examples("test")
    with pytest.raises(ValidationError, match="supports explicit dev"):
        BEIRArchiveAdapter(**args, split="train")
