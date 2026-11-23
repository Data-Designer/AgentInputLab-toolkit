"""Synthetic source-binding tests, separate from real author-cache acceptance."""
import json
from unittest.mock import patch

import pytest

from agentinputlab import RunConfig
from agentinputlab.cli.main import main
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.trec_sample_configuration import load, read_sources, sources
from tests.unit.test_trec_dl_sample import write_bundle


@pytest.mark.parametrize("year", [2019, 2020])
def test_relative_file_uses_native_sample_and_keeps_gold_separate(tmp_path, year):
    root = tmp_path / "sample"
    root.mkdir()
    digest, _ = write_bundle(root, year=year)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(dict(root="sample", manifest_sha256=digest)))
    with patch("socket.socket.connect", side_effect=AssertionError("no network")):
        adapter = load(read_sources(path), benchmark_id=f"trec_dl.{year}", split="test",
                       accept_license_review=True)
    assert adapter.benchmark_id == f"trec_dl.{year}"
    assert adapter.fingerprint()["full_dataset_score"] is False
    assert adapter.fingerprint()["binary_relevance_threshold"] == 2
    assert "relevance" not in repr(adapter.examples("test")[0].visible_state())
    with pytest.raises(ArtifactError, match="checksum"):
        load(dict(root=str(root), manifest_sha256="f" * 64),
             benchmark_id=f"trec_dl.{year}", split="test", accept_license_review=True)
    with pytest.raises(ValidationError, match="identity mismatch"):
        load(read_sources(path), benchmark_id=f"trec_dl.{4039-year}", split="test",
             accept_license_review=True)


@pytest.mark.parametrize("value", [None, [], {}, {"root": "a"},
    {"root": 1, "manifest_sha256": "a" * 64}, {"root": "a\x00b", "manifest_sha256": "a" * 64},
    {"root": "a", "manifest_sha256": "main"},
    {"root": "a", "manifest_sha256": "a" * 64, "guess": True}])
def test_invalid_sources_rejected(value):
    with pytest.raises(ValidationError):
        sources(value)


@pytest.mark.parametrize("other", [dict(prepared_root="data"), dict(cache_dir="cache"),
    dict(adapter=object()), dict(longbench_sources={}), dict(bfcl_source_archive="x"),
    dict(beir_sample_sources={})])
def test_conflicting_sources_fail_before_any_loading(other):
    with pytest.raises(ValidationError, match="owns its dataset source"):
        RunConfig("baseline.no_op").evaluate("trec_dl.2019", split="test", example_ids=["0"],
            trec_sample_sources=dict(root="/absent", manifest_sha256="a" * 64), **other)


@pytest.mark.parametrize("benchmark,split,ack", [("beir.fever", "test", True),
    ("trec_dl.2019", "train", True), ("trec_dl.2020", "test", False)])
def test_wrong_scope_rejected_before_loading(benchmark, split, ack):
    with pytest.raises(ValidationError):
        load(dict(root="/absent", manifest_sha256="a" * 64),
             benchmark_id=benchmark, split=split, accept_license_review=ack)


def test_invalid_json_and_oversized_config(tmp_path):
    path = tmp_path / "source.json"
    for content in ('{"root":"a","root":"b"}', '[]', '{', '"not a source"'):
        path.write_text(content)
        with pytest.raises(ValidationError):
            read_sources(path)
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValidationError, match="exceeds"):
        read_sources(path)


def test_cli_forwards_native_sample_configuration(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(root="sample", manifest_sha256="a" * 64)))
    config = tmp_path / "run.json"
    config.write_text('{"release_id":"baseline.no_op"}')
    with patch.object(RunConfig, "evaluate", return_value=dict(
            records=[], failed_count=0, awaiting_clarification_count=0, scored_count=0)) as evaluate:
        assert main(["config", "evaluate", str(config), "--benchmark", "trec_dl.2019",
            "--split", "test", "--case-id", "0", "--trec-sample-config", str(source),
            "--offline", "--accept-license-review", "--output", str(tmp_path / "out.jsonl")]) == 0
    assert evaluate.call_args.kwargs["trec_sample_sources"] == dict(
        root=str(tmp_path / "sample"), manifest_sha256="a" * 64)


@pytest.mark.parametrize("year", [2019, 2020])
def test_full_cli_reaches_native_scorer_without_model_or_download(tmp_path, capsys, year):
    import hashlib

    from agentinputlab.backends import SQLiteFTS5Retriever
    from agentinputlab.contracts.serialization import to_jsonable

    root = tmp_path / "sample"
    root.mkdir()
    digest, _ = write_bundle(root, year=year)
    adapter = load(dict(root=str(root), manifest_sha256=digest), benchmark_id=f"trec_dl.{year}",
                   split="test", accept_license_review=True)
    identity = {**to_jsonable(adapter.fingerprint()), "forum": ""}
    index = tmp_path / "index.sqlite"
    retriever = SQLiteFTS5Retriever.build(index, adapter.iter_documents(), corpus_identity=identity)
    config = tmp_path / "run.json"
    config.write_text(json.dumps(dict(release_id="baseline.no_op", retriever=dict(
        kind="sqlite_fts5", index_path=str(index), index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        corpus_identity=identity, expected_document_count=retriever.count, benchmark_id=f"trec_dl.{year}"))))
    source = tmp_path / "sample.json"
    source.write_text(json.dumps(dict(root=str(root), manifest_sha256=digest)))
    # Metrics provider is mocked only in this synthetic fixture test. The real
    # saved-author acceptance separately executes the installed native scorer.
    with patch("socket.socket.connect", side_effect=AssertionError("no network")), patch(
            "agentinputlab.benchmarks.trec_dl_sample.retrieval_aggregates", return_value={"ndcg": 0.0}) as scorer:
        code = main(["config", "evaluate", str(config), "--benchmark", f"trec_dl.{year}",
            "--split", "test", "--case-id", "0", "--trec-sample-config", str(source),
            "--offline", "--accept-license-review", "--output", str(tmp_path / "out.jsonl")])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["scored_count"] == 1
    assert scorer.call_count == 2  # Graded nDCG and threshold-2 binary metrics.
    records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert records[1]["usage"]["model_calls"] == 0
    assert records[1]["usage"]["retrieval_calls"] == 1
    assert records[1]["dataset_fingerprint"]["full_dataset_score"] is False
    assert records[-1]["dataset_fingerprint"]["metrics_scope"] == "sampled_candidate_pool_only"


@pytest.mark.parametrize("benchmark", ["trec_dl.2019", "trec_dl.2020", "beir.cqadupstack"])
def test_native_sample_predictor_routes_are_retrieval_not_qa(benchmark):
    from agentinputlab.backends import Document, InMemoryBM25Retriever
    from agentinputlab.experiments.drivers.static_production import (
        StaticPredictorBuildRequest,
        StaticRuntimeComponents,
    )
    from agentinputlab.experiments.production_builders import ProductionPredictorBuilder

    retriever = InMemoryBM25Retriever([Document(doc_id="d1", text="visible fact")])
    request = StaticPredictorBuildRequest(benchmark_id=benchmark, split="test", model_slot="none",
        seed=17, builder_parameters={"schema_version": 1},
        runtime=StaticRuntimeComponents(backend=None, retriever=retriever))
    assert ProductionPredictorBuilder().build(request).fingerprint()["task"] == "retrieval"
    from dataclasses import replace

    with pytest.raises(ValidationError, match="expects split"):
        ProductionPredictorBuilder().build(replace(request, split="train"))
    with pytest.raises(ValidationError, match="incompatible"):
        ProductionPredictorBuilder().build(replace(request, builder_parameters={"schema_version": 1, "task": "qa"}))
