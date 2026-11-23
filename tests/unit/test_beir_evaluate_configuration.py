"""Native archive/config routes; tiny fixtures are not model acceptance."""
import hashlib
import json
import zipfile

import pytest

from agentinputlab import InterventionContext, RunConfig
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.benchmarks.beir_archive import BEIR_ARCHIVE_LOCKS
from agentinputlab.contracts import ArtifactError, ValidationError


@pytest.fixture
def archive(tmp_path, monkeypatch):
    path = tmp_path / "fiqa.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("fiqa/corpus.jsonl", "\n".join(json.dumps(row) for row in [
            {"_id": "d1", "text": "savings account interest", "title": "Saving"},
            {"_id": "d2", "text": "market stock options", "title": "Stocks"}]) + "\n")
        z.writestr("fiqa/queries.jsonl", "\n".join(json.dumps(row) for row in [
            {"_id": "q1", "text": "savings interest"},
            {"_id": "qdev", "text": "stock options"},
            {"_id": "qtrain", "text": "financial training fixture"}]) + "\n")
        z.writestr("fiqa/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
        z.writestr("fiqa/qrels/dev.tsv", "query-id\tcorpus-id\tscore\nqdev\td2\t1\n")
        z.writestr("fiqa/qrels/train.tsv", "query-id\tcorpus-id\tscore\nqtrain\td2\t1\n")
    monkeypatch.setitem(BEIR_ARCHIVE_LOCKS, "beir.fiqa_2018", {
        "dataset": "fiqa", "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "upstream_md5": "fixture"})
    return path


def config(path, **options):
    return RunConfig("baseline.no_op", retriever={"kind": "beir_archive",
        "engine": "memory_bm25",
        "benchmark_id": "beir.fiqa_2018", "source_archive": str(path),
        "accept_license_review": True, **options})


def test_mounted_corpus_builds_existing_bm25_without_gold(archive):
    c = config(archive, k1=1.3, b=0.6)
    c.validate()
    r = c._retriever()
    expected = InMemoryBM25Retriever(tuple(r.adapter.iter_documents()), k1=1.3, b=0.6)
    assert r.retrieve("savings interest", top_k=2) == expected.retrieve("savings interest", top_k=2)
    assert r.fingerprint()["document_count"] == 2
    assert r.fingerprint()["corpus_scope"] == "complete_source"
    assert "relevance" not in json.dumps(r.fingerprint())


def test_validate_never_opens_archive_or_downloads(tmp_path):
    config(tmp_path / "not-mounted.zip").validate()


def test_corpus_cap_fails_not_silently_samples(archive):
    with pytest.raises(ArtifactError, match="max_documents"):
        config(archive, max_documents=1)._retriever()
    with pytest.raises(ArtifactError, match="max_text_bytes"):
        config(archive, max_text_bytes=1)._retriever()


def test_relative_archive_and_cache_paths_follow_config(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config("fiqa.zip").to_dict()))
    assert RunConfig.from_file(path).retriever["source_archive"] == str(tmp_path / "fiqa.zip")


def test_sqlite_uses_existing_native_index_and_read_only_reuse(archive, tmp_path):
    from agentinputlab.backends.retrieval import SQLiteFTS5Retriever
    from agentinputlab.benchmarks.beir_archive import BEIRArchiveAdapter
    adapter = BEIRArchiveAdapter(benchmark_id="beir.fiqa_2018", source_archive=archive, license_acknowledged=True)
    path = tmp_path / "index.sqlite"
    original = SQLiteFTS5Retriever.build(path, adapter.iter_documents(),
        corpus_identity={**adapter.fingerprint(), "forum": ""})
    before = path.read_bytes()
    candidate = config(archive, engine="sqlite_fts5", index_path=str(path))._retriever()
    assert candidate.retrieve("savings interest", top_k=2) == original.retrieve("savings interest", top_k=2)
    assert candidate.fingerprint()["retriever"] == original.fingerprint()
    assert path.read_bytes() == before


def test_sqlite_bounds_cleanup_partial_index(archive, tmp_path):
    path = tmp_path / "index.sqlite"
    with pytest.raises(ArtifactError, match="max_documents"):
        config(archive, engine="sqlite_fts5", index_path=str(path), max_documents=1)._retriever()
    assert not path.exists() and not list(tmp_path.glob(".fts-build-*"))


def test_default_engine_is_sqlite_and_memory_options_not_ignored(archive, tmp_path):
    c = config(archive, index_path=str(tmp_path / "index.sqlite"))
    data = c.to_dict()
    del data["retriever"]["engine"]
    default = RunConfig.from_dict(data)
    assert default._retriever().retriever_id == "sqlite.fts5.bm25.unicode61.v1"
    with pytest.raises(ValidationError, match="k1/b"):
        config(archive, engine="sqlite_fts5", k1=1.4).validate()


@pytest.mark.parametrize("options", [{"k1": -1}, {"b": 2}, {"max_documents": True},
    {"max_text_bytes": 0}, {"offline": "false"}, {"surprise": 1}])
def test_bad_options_rejected_without_io(tmp_path, options):
    with pytest.raises(ValidationError):
        config(tmp_path / "absent", **options).validate()


def test_wrong_benchmark_rejected_before_corpus_load(tmp_path):
    with pytest.raises(ValidationError, match="retriever benchmark"):
        config(tmp_path / "absent").evaluate("beir.arguana", split="test", limit=1)


def test_evaluate_offline_forbids_retriever_download(monkeypatch, tmp_path):
    from agentinputlab import beir_configuration
    seen = []
    def loader(cell, **kwargs):
        seen.append(kwargs)
        raise ArtifactError("no cached archive")
    monkeypatch.setattr(beir_configuration, "load_benchmark", loader)
    c = RunConfig("baseline.no_op", retriever={"kind": "beir_archive",
        "benchmark_id": "beir.fiqa_2018", "cache_dir": str(tmp_path),
        "accept_license_review": True})
    with pytest.raises(ArtifactError, match="no cached archive"):
        c.evaluate("beir.fiqa_2018", split="test", limit=1, offline=True)
    assert len(seen) == 1 and seen[0]["offline"] is True
    assert "offline" not in c.retriever  # do not mutate the caller's reusable config


def test_license_refusal_precedes_source_access(monkeypatch, tmp_path):
    from agentinputlab import beir_configuration
    def forbidden(*args, **kwargs):
        pytest.fail("source accessed before acknowledgement")
    monkeypatch.setattr(beir_configuration, "BEIRArchiveAdapter", forbidden)
    with pytest.raises(ArtifactError, match="review upstream"):
        config(tmp_path / "missing", accept_license_review=False)._retriever()


def test_retrieval_pipeline_uses_own_original_adapter(archive, monkeypatch):
    from agentinputlab.benchmarks.beir_archive import BEIRArchiveAdapter
    seen = []
    def score(self, example, prediction):
        assert example is self.examples("test")[0]
        seen.append(prediction.scores)
        return {"ndcg_at_10": 1.0}
    monkeypatch.setattr(BEIRArchiveAdapter, "score", score)
    result = config(archive).evaluate("beir.fiqa_2018", split="test", limit=1)
    assert result["scored_count"] == 1 and len(seen) == 1
    assert result["records"][0]["usage"]["model_calls"] == 0
    assert result["records"][0]["usage"]["retrieval_calls"] == 1
    assert set(seen[0]) == {"d1", "d2"}


def test_fiqa_dev_uses_its_native_case_and_prediction_without_switching_to_test(archive, monkeypatch):
    from agentinputlab.benchmarks.beir_archive import BEIRArchiveAdapter

    def score(self, example, prediction):
        assert self.split == "dev" and example.example_id == "qdev"
        assert next(iter(prediction.scores)) == "d2"
        return {"ndcg_at_10": 1.0}

    monkeypatch.setattr(BEIRArchiveAdapter, "score", score)
    result = config(archive, split="dev").evaluate("beir.fiqa_2018", split="dev", example_ids=["qdev"])
    assert result["records"][0]["evaluation"]["metrics"] == {"ndcg_at_10": 1.0}
    assert result["records"][0]["usage"]["model_calls"] == 0


def test_fiqa_dev_workflow_inspects_prepares_scores_and_resumes_offline(archive, monkeypatch, tmp_path):
    from agentinputlab import ExecutionBudget
    from agentinputlab.benchmarks.beir_archive import BEIRArchiveAdapter
    from agentinputlab.workflows import WorkflowConfig, inspect_workflow, prepare_workflow, run_workflow
    from agentinputlab.workflows import readiness

    seen = []
    def score(self, example, prediction):
        assert self.split == "dev" and example.example_id == "qdev"
        assert next(iter(prediction.scores)) == "d2"
        seen.append(example.example_id)
        return {"ndcg_at_10": 1.0}

    monkeypatch.setattr(BEIRArchiveAdapter, "score", score)
    monkeypatch.setattr(readiness, "interpreter_readiness", lambda *args, **kwargs: {})
    run = RunConfig("baseline.no_op", retriever=config(archive, split="dev").retriever,
                    budget=ExecutionBudget(max_model_calls=0, max_output_tokens=0))
    workflow = WorkflowConfig(run, {"kind": "benchmark", "benchmark_id": "beir.fiqa_2018",
                                   "split": "dev", "case_ids": ["qdev"]})
    report = inspect_workflow(workflow)
    assert report["ready_for_execution"], report["checks"]
    prep = prepare_workflow(workflow, output_dir=tmp_path / "prepared")
    prepared = WorkflowConfig.from_file(prep["workflow_path"])
    assert list(prepared.target["case_ids"]) == ["qdev"]
    out = tmp_path / "run"
    assert run_workflow(prepared, output_dir=out, execute=True)["outcomes"]["scored"] == 1
    assert run_workflow(prepared, output_dir=out, execute=True, resume=True)["reused_existing_attempts"]
    assert seen == ["qdev"]


def test_fiqa_workflow_rejects_mismatched_split_before_opening_archive(tmp_path):
    from agentinputlab.workflows import WorkflowConfig, inspect_workflow
    from agentinputlab.workflows.planning import load_adapter

    workflow = WorkflowConfig(config(tmp_path / "missing.zip", split="test"),
        {"kind": "benchmark", "benchmark_id": "beir.fiqa_2018", "split": "dev", "limit": 1})
    assert not inspect_workflow(workflow)["can_attempt"]
    with pytest.raises(ValidationError, match="must match the workflow split"):
        load_adapter(workflow, allow_network=True)


@pytest.mark.parametrize("cell", list(BEIR_ARCHIVE_LOCKS))
def test_native_ir_prediction_routes_with_caller_retriever(cell):
    class Adapter:
        benchmark_id = cell
        pairing_mode = "stateless_score_twice"
        def examples(self, split):
            assert split == "test"
            return [BenchmarkExample(example_id="q", query="savings", evaluator_gold={"secret": "gold"})]
        def score(self, example, prediction):
            assert set(prediction.scores) == {"d"}
            return {"ndcg_at_10": 1.0}
    r = InMemoryBM25Retriever([Document(doc_id="d", text="savings")])
    result = RunConfig("baseline.no_op").evaluate(cell, split="test", limit=1,
        adapter=Adapter(), context=InterventionContext(retriever=r))
    assert result["scored_count"] == 1
