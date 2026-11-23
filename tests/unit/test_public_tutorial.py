"""Keep the bilingual entry points and executable example aligned."""
import importlib.util
import re
from pathlib import Path

import pytest

from agentinputlab import InterventionContext, RunConfig
from agentinputlab.contracts.errors import ValidationError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md", "docs/tutorial.md", "docs/tutorial.zh-CN.md"])
def test_tutorial_local_links_exist(name):
    source = ROOT / name
    for link in re.findall(r"\]\(([^)]+)\)", source.read_text()):
        if "://" not in link and not link.startswith("#"):
            assert (source.parent / link.split("#", 1)[0]).exists(), link


def test_rrr_example_is_configurable_and_secret_free(monkeypatch):
    monkeypatch.setenv("AIL_MODEL", "example-model")
    monkeypatch.setenv("AIL_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AIL_MODEL_REVISION", "example-revision")
    spec = importlib.util.spec_from_file_location("rrr_example", ROOT / "examples/rrr_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.configuration()
    config = RunConfig.from_dict(data)
    config.validate()
    assert "api_key" not in data["backend"]
    assert data["backend"]["api_key_env"] == "AIL_API_KEY"
    assert config.budget.max_model_calls == 2
    assert data["parameters"]["config"]["max_reader_tokens"] == 512
    data["parameters"]["config"]["temperature"] = 0.3
    data["retriever"]["k1"] = 1.5
    data["budget"]["max_output_tokens"] = 2048
    edited = RunConfig.from_dict(data)
    edited.validate()
    assert edited.budget.max_output_tokens == 2048
    data["parameters"]["config"]["temperatur"] = 0.4
    with pytest.raises((ValidationError, TypeError)):
        RunConfig.from_dict(data).validate()


def test_inline_corpus_provenance_is_content_bound_and_preserves_explicit_revision():
    def retrieve(text="Paris is the capital of France.", revision=None):
        doc = {"doc_id": "france", "text": text}
        if revision is not None:
            doc["corpus_revision"] = revision
        cfg = RunConfig.from_dict({"release_id": "baseline.no_op", "retriever": {
            "kind": "bm25", "documents": [doc]}})
        return cfg._retriever().retrieve("France", top_k=1)[0].document
    first = retrieve().corpus_revision
    assert first.startswith("inline-sha256:")
    assert first == retrieve().corpus_revision
    assert first != retrieve("France has Paris as its capital.").corpus_revision
    assert retrieve(revision="user-pinned-v1").corpus_revision == "user-pinned-v1"


def test_rrr_tutorial_reaches_reader_with_unversioned_inline_documents(monkeypatch):
    # Synthetic regression only; live user acceptance is recorded separately.
    from tests.unit.test_benchmark_configuration import Backend
    monkeypatch.setenv("AIL_MODEL", "example-model")
    monkeypatch.setenv("AIL_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AIL_MODEL_REVISION", "example-revision")
    spec = importlib.util.spec_from_file_location("rrr_example", ROOT / "examples/rrr_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.configuration()
    data.pop("backend")
    backend = Backend([("capital of France", 3), ("Paris", 1)])
    result = RunConfig.from_dict(data).run("What is the capital of France?",
                                         context=InterventionContext(backend=backend))
    assert len(backend.requests) == 2
    assert result["record"]["usage"]["model_calls"] == 2
    assert result["record"]["usage"]["retrieval_calls"] == 1
    assert result["state"]["query"] == "capital of France"
    assert "Paris" in backend.requests[1].prompt


def test_workflow_ui_does_not_offer_unsupported_parallelism():
    html = (ROOT / "src/agentinputlab/studio/static/workflows.html").read_text()
    assert 'id="workers" type="number" min="1" max="1" value="1" readonly' in html
