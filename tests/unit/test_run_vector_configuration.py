"""File-config wiring for dense query-surrogate methods; not neural acceptance."""
import json

import pytest

from agentinputlab import InterventionContext, RunConfig
from agentinputlab.contracts import ValidationError
from agentinputlab.experiments.production_builders import PreparedCorpusRetrieverBuilder
from tests.unit.test_run_retriever_configuration import corpus_config


def reference(tmp_path):
    value = corpus_config(tmp_path)
    value.update(profile_id="dense.e5.base_v2", options={"model_path": str(tmp_path / "e5")})
    return value


def config(value):
    return RunConfig.from_dict({"release_id": "acquire.hyde.paper_reimplementation", "vector_runtime": value})


def test_vector_shape_validation_does_not_construct(tmp_path, monkeypatch):
    value = reference(tmp_path)
    (tmp_path / "corpus.jsonl").unlink()
    monkeypatch.setattr(PreparedCorpusRetrieverBuilder, "build", lambda *a: pytest.fail("must not load"))
    assert config(value).validate()["runtime_ready"] is None


class VectorRuntime:
    embedder_id = "test-embedder"
    searcher_id = "test-searcher"

    def embed(self, texts):
        return ()

    def search(self, vector, *, top_k):
        return ()

    def fingerprint(self):
        return {"test": "vector-binding"}


def test_same_original_vector_product_bound_once(tmp_path, monkeypatch):
    seen, runtime = [], VectorRuntime()
    monkeypatch.setattr(PreparedCorpusRetrieverBuilder, "build", lambda self, req: seen.append(req) or runtime)
    context = config(reference(tmp_path))._configured_context(None)
    assert context.dense_embedder is context.vector_searcher is runtime
    assert len(seen) == 1 and seen[0].seed == 17
    assert context.retriever is None


def test_vector_relative_assets_follow_file(tmp_path, monkeypatch):
    value = reference(tmp_path)
    value.update(corpus_path="corpus.jsonl", options={"model_path": "model"},
                 artifact_files=[{"name": "config", "path": "model/config.json", "sha256": "a" * 64}])
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config(value).to_dict()))
    monkeypatch.chdir(tmp_path.parent)
    loaded = RunConfig.from_file(path)
    assert loaded.vector_runtime["corpus_path"] == str(tmp_path / "corpus.jsonl")
    assert loaded.vector_runtime["options"]["model_path"] == str(tmp_path / "model")
    assert loaded.vector_runtime["artifact_files"][0]["path"] == str(tmp_path / "model/config.json")


@pytest.mark.parametrize("field", ["dense_embedder", "vector_searcher"])
def test_duplicate_context_rejected_before_build(tmp_path, monkeypatch, field):
    monkeypatch.setattr(PreparedCorpusRetrieverBuilder, "build", lambda *a: pytest.fail("must not load"))
    with pytest.raises(ValidationError, match="configured twice"):
        config(reference(tmp_path))._configured_context(InterventionContext(**{field: VectorRuntime()}))


def test_sparse_profile_cannot_be_used_as_vector_runtime(tmp_path):
    with pytest.raises(ValidationError, match="dense"):
        config(corpus_config(tmp_path)).validate()


def test_unexpected_builder_product_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(PreparedCorpusRetrieverBuilder, "build", lambda *a: object())
    with pytest.raises(ValidationError, match="DenseEmbedder"):
        config(reference(tmp_path))._configured_context(None)


def test_evaluation_rejects_other_benchmark_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(PreparedCorpusRetrieverBuilder, "build", lambda *a: pytest.fail("must not load"))
    with pytest.raises(ValidationError, match="vector runtime benchmark"):
        config(reference(tmp_path)).evaluate("bright.earth_science", split="test", limit=1)


def test_multilingual_e5_has_its_own_identity_not_english_e5():
    from agentinputlab.backends.retrieval.dense import retriever_profile

    profile = retriever_profile("dense.e5.multilingual_large")
    assert profile.model_id == "intfloat/multilingual-e5-large"
    assert profile.model_revision == "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
    assert profile.expected_dimensions == 1024
    assert (profile.query_prefix, profile.document_prefix) == ("query: ", "passage: ")
    assert profile.model_id != retriever_profile("dense.e5.large_v2").model_id
