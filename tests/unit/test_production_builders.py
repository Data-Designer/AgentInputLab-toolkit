from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest import SkipTest
from unittest.mock import patch

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.backends.llm import (
    GenerationResponse,
    ModelCapability,
    ReplayBackend,
    Usage,
)
from agentinputlab.backends.retrieval import (
    Document,
    InMemoryBM25Retriever,
    RetrievalResult,
)
from agentinputlab.benchmarks import QAPrediction, RetrievalPrediction
from agentinputlab.contracts.errors import ArtifactError, CapabilityError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash
from agentinputlab.contracts.state import ContextBlock, FrozenAgentState, Message
from agentinputlab.experiments.drivers.static_production import (
    StaticBackendBuildRequest,
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
    static_predictor_product_sha256,
    static_runtime_product_sha256,
)
from agentinputlab.experiments.production_builders import (
    PreparedCorpusRetrieverBuilder,
    ProductionBackendBuilder,
    ProductionPredictorBuilder,
    ProductionRetrieverBuildRequest,
    ProductionRetrieverRouter,
    builder_fingerprint_sha256,
    prepared_corpus_retriever_builder_factory,
    retriever_product_sha256,
    validate_predictor_product_sha256,
    validate_retriever_product_sha256,
    validate_runtime_product_sha256,
)

_RETRIEVER_ENTRYPOINT = (
    "agentinputlab.experiments.production_builders:prepared_corpus_retriever_builder_factory"
)


def _write_json(path: Path, value: object) -> str:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_corpus(path: Path) -> str:
    rows = (
        {"id": "doc-a", "title": "Alpha", "text": "alpha evidence"},
        {"id": "doc-b", "title": "Beta", "text": "beta evidence"},
    )
    raw = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows
    )
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _replay_parameters(tmp_path: Path, *, response: str = "answer") -> dict:
    fixture = tmp_path / "replay.json"
    digest = _write_json(
        fixture,
        {
            "schema_version": 1,
            "responses": [
                {
                    "text": response,
                    "model": "fixture-served",
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 2,
                        "cost_usd": 0.01,
                    },
                }
            ],
        },
    )
    return {
        "schema_version": 1,
        "backend": {
            "kind": "replay",
            "model_id": "fixture-replay",
            "fixture_path": str(fixture),
            "fixture_sha256": digest,
        },
        "required_capabilities": ["completion"],
        "run_config": {"generation": {"temperature": 0.0}},
    }


def _backend_request(parameters: dict, *, seed: int = 7) -> StaticBackendBuildRequest:
    return StaticBackendBuildRequest(
        model_slot="model-a",
        seed=seed,
        builder_parameters=parameters,
    )


def _predictor_request(
    runtime: StaticRuntimeComponents,
    *,
    benchmark_id: str,
    split: str,
    parameters: dict | None = None,
    seed: int = 7,
) -> StaticPredictorBuildRequest:
    return StaticPredictorBuildRequest(
        benchmark_id=benchmark_id,
        split=split,
        model_slot="model-a",
        seed=seed,
        builder_parameters=parameters or {"schema_version": 1},
        runtime=runtime,
    )


def _corpus_parameters(tmp_path: Path, *, benchmark_id: str = "beir.arguana") -> dict:
    corpus = tmp_path / f"{benchmark_id.replace('.', '-')}-corpus.jsonl"
    digest = _write_corpus(corpus)
    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "profile_id": "sparse.bm25.memory.v1",
        "corpus_path": str(corpus),
        "corpus_sha256": digest,
        "corpus_revision": "fixture-corpus-v1",
        "expected_document_count": 2,
        "options": {"retriever_id": f"bm25:{benchmark_id}"},
    }


class _FixtureVectorRuntime:
    retriever_id = "fixture-vector-runtime"
    embedder_id = "fixture-vector-runtime:embedder"
    searcher_id = "fixture-vector-runtime:index"

    def __init__(self, documents):
        self.documents = tuple(documents)

    def retrieve(self, query: str, *, top_k: int = 5):
        del query
        return self.search((1.0, 0.0), top_k=top_k)

    def embed(self, texts):
        return tuple((1.0, 0.0) for _ in texts)

    def search(self, vector, *, top_k: int):
        del vector
        return tuple(
            RetrievalResult(document=document, score=1.0, rank=rank)
            for rank, document in enumerate(self.documents[:top_k], start=1)
        )

    def fingerprint(self):
        return {
            "implementation": "tests.fixture_vector_runtime.v1",
            "corpus_ids": tuple(document.doc_id for document in self.documents),
        }


def test_builder_fingerprints_are_stable_and_secret_free() -> None:
    backend = ProductionBackendBuilder()
    predictor = ProductionPredictorBuilder()
    retriever = prepared_corpus_retriever_builder_factory()

    assert backend.builder_kind == "backend"
    assert predictor.builder_kind == "predictor"
    assert retriever.builder_kind == "retriever"
    assert builder_fingerprint_sha256(backend) == builder_fingerprint_sha256(
        ProductionBackendBuilder()
    )
    assert len(builder_fingerprint_sha256(predictor)) == 64
    assert len(builder_fingerprint_sha256(retriever)) == 64
    assert "sk-actual-value" not in json.dumps(backend.fingerprint()).lower()


def test_replay_backend_build_is_checksum_locked_and_offline(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    builder = ProductionBackendBuilder()

    assert builder.preflight(parameters) == ()
    runtime = builder.build(_backend_request(parameters))
    assert runtime.backend.model_id == "fixture-replay"
    assert ModelCapability.COMPLETION in runtime.backend.capabilities
    assert runtime.retriever is None
    assert runtime.run_config == {"generation": {"temperature": 0.0}}
    assert runtime.strict_intent is False
    assert builder.product_sha256(_backend_request(parameters)) == static_runtime_product_sha256(
        builder.build(_backend_request(parameters))
    )

    Path(parameters["backend"]["fixture_path"]).write_text("changed", encoding="utf-8")
    blockers = builder.preflight(parameters)
    assert len(blockers) == 1
    assert "checksum mismatch" in blockers[0]
    with pytest.raises(ArtifactError, match="preflight failed"):
        builder.build(_backend_request(parameters))


def test_backend_builder_binds_checksum_verified_querygym_runtime(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    checkout = tmp_path / "querygym"
    checkout.mkdir()
    parameters["external_runtime"] = {
        "kind": "querygym_query2doc",
        "checkout": str(checkout),
    }

    class FakePinnedRuntime:
        def __init__(self, observed_checkout, backend):
            self.checkout = observed_checkout
            self.backend = backend

        def fingerprint(self):
            return {
                "implementation": "agentinputlab.querygym.pinned_runtime.v1",
                "revision": "6d2d19213c7212c7940143e7b9b5f1d2af7ff47b",
                "source_closure_sha256": "fixture-source-closure",
                "backend": self.backend.fingerprint(),
            }

    with patch(
        "agentinputlab.algorithms.reformulate.query2doc.PinnedQueryGymRuntime",
        FakePinnedRuntime,
    ):
        builder = ProductionBackendBuilder()
        assert builder.preflight(parameters) == ()
        runtime = builder.build(_backend_request(parameters))

    assert isinstance(runtime.external_runtime, FakePinnedRuntime)
    assert runtime.external_runtime.checkout == checkout
    assert static_runtime_product_sha256(runtime) != static_runtime_product_sha256(
        ProductionBackendBuilder().build(
            _backend_request(
                {key: value for key, value in parameters.items() if key != "external_runtime"}
            )
        )
    )


def test_backend_builder_binds_ask_or_assume_worker_without_serializing_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = _replay_parameters(tmp_path)
    checkout = tmp_path / "ask-or-assume"
    checkout.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text("print('{}')\n", encoding="utf-8")
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text("{}\n", encoding="utf-8")
    executable = Path(sys.executable).resolve()

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    parameters["external_runtime"] = {
        "kind": "ask_or_assume_worker",
        "checkout": str(checkout),
        "driver_path": str(driver),
        "driver_sha256": digest(driver),
        "python_executable": str(executable),
        "python_executable_sha256": digest(executable),
        "compatibility_manifest_path": str(compatibility),
        "compatibility_manifest_sha256": digest(compatibility),
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "AILAB_ASK_TEST_KEY",
        "model_id": "fixture-replay",
        "timeout_seconds": 60.0,
        "max_output_tokens": 128,
    }
    builder = ProductionBackendBuilder()
    assert "AILAB_ASK_TEST_KEY" in builder.preflight(parameters)[0]

    monkeypatch.setenv("AILAB_ASK_TEST_KEY", "credential-value")
    assert builder.preflight(parameters) == ()
    runtime = builder.build(_backend_request(parameters))
    worker = runtime.external_runtime
    backend_config = json.loads(worker.command[-1])
    assert worker.pass_environment == ("AILAB_ASK_TEST_KEY",)
    assert backend_config["roles"]["intent"]["fingerprint"] == runtime.backend.fingerprint()
    assert backend_config["roles"]["coding"]["fingerprint"] == runtime.backend.fingerprint()
    assert "credential-value" not in str(worker.fingerprint())
    assert static_runtime_product_sha256(runtime)


def test_backend_builder_binds_reformer_worker_without_serializing_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = _replay_parameters(tmp_path)
    checkout = tmp_path / "reformer"
    checkout.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text("print('{}')\n", encoding="utf-8")
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text("{}\n", encoding="utf-8")
    executable = Path(sys.executable).resolve()

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    parameters["external_runtime"] = {
        "kind": "reformer_worker",
        "checkout": str(checkout),
        "driver_path": str(driver),
        "driver_sha256": digest(driver),
        "python_executable": str(executable),
        "python_executable_sha256": digest(executable),
        "compatibility_manifest_path": str(compatibility),
        "compatibility_manifest_sha256": digest(compatibility),
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "AILAB_REFORMER_TEST_KEY",
        "model_id": "fixture-replay",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "timeout_seconds": 60.0,
        "max_input_tokens_per_call": 4096,
        "max_output_tokens": 128,
        "temperature": 0.0,
    }
    builder = ProductionBackendBuilder()
    assert "AILAB_REFORMER_TEST_KEY" in builder.preflight(parameters)[0]

    monkeypatch.setenv("AILAB_REFORMER_TEST_KEY", "credential-value")
    assert builder.preflight(parameters) == ()
    runtime = builder.build(_backend_request(parameters))
    worker = runtime.external_runtime
    config_index = worker.command.index("--backend-config-json") + 1
    backend_config = json.loads(worker.command[config_index])
    assert worker.pass_environment == ("AILAB_REFORMER_TEST_KEY",)
    assert worker.environment == {"PYTHONDONTWRITEBYTECODE": "1"}
    assert backend_config["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert backend_config["fingerprint"] == runtime.backend.fingerprint()
    assert backend_config["temperature"] == 0.0
    assert "--acknowledge-noassertion-license" in worker.command
    assert "credential-value" not in str(worker.fingerprint())
    assert static_runtime_product_sha256(runtime)


def test_backend_builder_binds_infocqr_worker_without_serializing_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = _replay_parameters(tmp_path)
    checkout = tmp_path / "infocqr"
    checkout.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text("print('{}')\n", encoding="utf-8")
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text("{}\n", encoding="utf-8")
    executable = Path(sys.executable).resolve()

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    parameters["external_runtime"] = {
        "kind": "infocqr_worker",
        "checkout": str(checkout),
        "driver_path": str(driver),
        "driver_sha256": digest(driver),
        "python_executable": str(executable),
        "python_executable_sha256": digest(executable),
        "compatibility_manifest_path": str(compatibility),
        "compatibility_manifest_sha256": digest(compatibility),
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "AILAB_INFOCQR_TEST_KEY",
        "model_id": "fixture-replay",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "timeout_seconds": 60.0,
        "max_input_tokens_per_call": 4096,
        "max_output_tokens": 128,
        "temperature": 0.0,
    }
    builder = ProductionBackendBuilder()
    assert "AILAB_INFOCQR_TEST_KEY" in builder.preflight(parameters)[0]

    monkeypatch.setenv("AILAB_INFOCQR_TEST_KEY", "credential-value")
    assert builder.preflight(parameters) == ()
    runtime = builder.build(_backend_request(parameters))
    worker = runtime.external_runtime
    config_index = worker.command.index("--backend-config-json") + 1
    backend_config = json.loads(worker.command[config_index])
    assert worker.pass_environment == ("AILAB_INFOCQR_TEST_KEY",)
    assert worker.environment == {"PYTHONDONTWRITEBYTECODE": "1"}
    assert backend_config["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert backend_config["fingerprint"] == runtime.backend.fingerprint()
    assert "--manifest" in worker.command
    assert "--acknowledge-noassertion-license" in worker.command
    assert worker.worker_id == "infocqr-common-support-rewriter"
    assert "credential-value" not in str(worker.fingerprint())
    assert static_runtime_product_sha256(runtime)


def test_backend_builder_binds_qrag_common_support_runtime(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    for name in ("source", "checkpoint", "embedder"):
        (tmp_path / name).mkdir()
    compatibility = tmp_path / "qrag-compatibility.json"
    compatibility.write_text("{}", encoding="utf-8")
    parameters["external_runtime"] = {
        "kind": "qrag_common_support",
        "source_root": str(tmp_path / "source"),
        "checkpoint_root": str(tmp_path / "checkpoint"),
        "embedder_root": str(tmp_path / "embedder"),
        "compatibility_manifest_path": str(compatibility),
        "compatibility_manifest_sha256": "4" * 64,
        "embedder_manifest_sha256": "5" * 64,
        "device": "cpu",
    }

    class FakeRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def select(self, _request):
            raise AssertionError("builder preflight must not execute the Q-RAG policy")

        def fingerprint(self):
            return {
                "runtime": "qrag_released_pqn_common_support.v1",
                "embedder_manifest_sha256": self.kwargs["embedder_manifest_sha256"],
            }

    with patch(
        "agentinputlab.algorithms.acquire.qrag.LocalQRAGFrozenPolicyRuntime",
        FakeRuntime,
    ):
        builder = ProductionBackendBuilder()
        assert builder.preflight(parameters) == ()
        runtime = builder.build(_backend_request(parameters))

    assert runtime.external_runtime.fingerprint()["runtime"] == (
        "qrag_released_pqn_common_support.v1"
    )
    assert runtime.external_runtime.kwargs["device"] == "cpu"


def test_backend_builder_binds_recomp_extractive_runtime(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    checkout = tmp_path / "recomp-source"
    checkpoint = tmp_path / "recomp-checkpoint"
    checkout.mkdir()
    checkpoint.mkdir()
    compatibility = tmp_path / "recomp-compatibility.json"
    checkpoint_manifest = tmp_path / "recomp-checkpoint.json"
    compatibility_sha256 = _write_json(compatibility, {"kind": "fixture"})
    checkpoint_sha256 = _write_json(checkpoint_manifest, {"kind": "fixture"})
    parameters["external_runtime"] = {
        "kind": "recomp_extractive_worker",
        "checkout": str(checkout),
        "checkpoint_root": str(checkpoint),
        "checkpoint_manifest_path": str(checkpoint_manifest),
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "compatibility_manifest_path": str(compatibility),
        "compatibility_manifest_sha256": compatibility_sha256,
        "device": "cpu",
    }

    class FakeDriver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def call(self, _request):
            raise AssertionError("builder preflight must not execute RECOMP")

        def fingerprint(self):
            return {
                "worker_id": "recomp-pinned-inference",
                "mode": "extractive",
                "custom_model_loader": self.kwargs["model_loader"] is not None,
            }

    with patch("workers.recomp.driver.RECOMPDriver", FakeDriver):
        builder = ProductionBackendBuilder()
        assert builder.preflight(parameters) == ()
        runtime = builder.build(_backend_request(parameters))

    assert runtime.external_runtime.fingerprint()["mode"] == "extractive"
    assert runtime.external_runtime.kwargs["device"] == "cpu"
    assert runtime.external_runtime.kwargs["checkpoint_license_acknowledged"] is True


@pytest.mark.parametrize('aliases',[(),('verified-provider-family',)])
def test_backend_builder_binds_acon_observation_worker_without_network(tmp_path: Path, aliases) -> None:
    parameters = _replay_parameters(tmp_path)
    checkout = tmp_path / "acon-source"
    worker_root = tmp_path / "workers" / "acon"
    checkout.mkdir()
    worker_root.mkdir(parents=True)
    driver = worker_root / "driver.py"
    driver.write_text("raise AssertionError('builder must not execute worker')\n", encoding="utf-8")
    compatibility = worker_root / "compatibility_manifest.json"
    compatibility_sha256 = _write_json(compatibility, {"kind": "fixture"})
    driver_sha256 = hashlib.sha256(driver.read_bytes()).hexdigest()
    python = Path(sys.executable).resolve()
    python_sha256 = hashlib.sha256(python.read_bytes()).hexdigest()
    parameters["external_runtime"] = {
        "kind": "acon_observation_worker",
        "checkout": str(checkout),
        "driver_path": str(driver),
        "driver_sha256": driver_sha256,
        "python_executable": str(python),
        "python_executable_sha256": python_sha256,
        "compatibility_manifest_path": str(compatibility),
        "compatibility_manifest_sha256": compatibility_sha256,
        "base_url": "http://127.0.0.1:8000/v1",
        "model_id": "fixture-optimizer",
        "optimizer_model_revision": "fixture-optimizer",
        "target_model_revision": "fixture-replay",
        "tokenizer_revision": "tiktoken:fixture@1",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "max_output_tokens": 96,
        "temperature": 0.0,
    }
    if aliases:parameters['external_runtime']['response_model_aliases']=list(aliases)

    builder = ProductionBackendBuilder()
    assert builder.preflight(parameters) == ()
    runtime = builder.build(_backend_request(parameters))

    fingerprint = runtime.external_runtime.fingerprint()
    assert fingerprint["worker_id"] == "acon-common-support-observation"
    assert fingerprint["release_id"] == "curate.acon.common_support_observation"
    assert fingerprint["optimizer_model_revision"] == "fixture-optimizer"
    assert (
        fingerprint["optimizer_extra_body_hash"]
        == hashlib.sha256(b'{"chat_template_kwargs":{"enable_thinking":false}}').hexdigest()
    )
    assert "AGENTINPUTLAB_ACON_API_KEY" not in fingerprint
    if aliases:
        assert fingerprint['optimizer_response_model_aliases']==aliases
        assert json.loads(runtime.external_runtime.environment['AGENTINPUTLAB_ACON_RESPONSE_MODEL_ALIASES_JSON'])==list(aliases)
    else:
        assert 'optimizer_response_model_aliases' not in fingerprint
        assert 'AGENTINPUTLAB_ACON_RESPONSE_MODEL_ALIASES_JSON' not in runtime.external_runtime.environment


def test_backend_builder_binds_smartsearch_worker_without_network(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    checkout = tmp_path / "smartsearch-source"
    worker_root = tmp_path / "workers" / "smartsearch"
    checkout.mkdir()
    worker_root.mkdir(parents=True)
    driver = worker_root / "driver.py"
    driver.write_text("raise AssertionError('builder must not execute worker')\n", encoding="utf-8")
    compatibility = worker_root / "compatibility_manifest.json"
    compatibility_sha256 = _write_json(compatibility, {"kind": "fixture"})
    runtime_manifest = tmp_path / "smartsearch-runtime.json"
    runtime_manifest_sha256 = _write_json(runtime_manifest, {"kind": "fixture-runtime"})
    python = Path(sys.executable).resolve()
    parameters["external_runtime"] = {
        "kind": "smartsearch_worker",
        "checkout": str(checkout),
        "driver_path": str(driver),
        "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "python_executable": str(python),
        "python_executable_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "compatibility_manifest_path": str(compatibility),
        "compatibility_manifest_sha256": compatibility_sha256,
        "runtime_manifest_path": str(runtime_manifest),
        "runtime_manifest_sha256": runtime_manifest_sha256,
    }

    builder = ProductionBackendBuilder()
    assert builder.preflight(parameters) == ()
    runtime = builder.build(_backend_request(parameters))

    fingerprint = runtime.external_runtime.fingerprint()
    assert fingerprint["worker_id"] == "smartsearch-common-support-queries"
    assert fingerprint["release_id"] == "rewrite.smartsearch.common_support_queries"
    assert fingerprint["runtime_manifest_sha256"] == runtime_manifest_sha256
    runtime_manifest.write_text("changed", encoding="utf-8")
    assert "checksum mismatch" in builder.preflight(parameters)[0]


def test_backend_builder_binds_monitored_r2ag_common_support_runtime(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    for name in ("artifact", "source", "base", "feature"):
        (tmp_path / name).mkdir()
    parameters["external_runtime"] = {
        "kind": "r2ag_common_support",
        "artifact_root": str(tmp_path / "artifact"),
        "manifest_sha256": "1" * 64,
        "source_root": str(tmp_path / "source"),
        "base_model_root": str(tmp_path / "base"),
        "base_model_load_set_sha256": "2" * 64,
        "feature_model_root": str(tmp_path / "feature"),
        "feature_model_load_set_sha256": "3" * 64,
        "reader_device": "cpu",
        "feature_device": "cpu",
    }

    class FakeReader:
        def run(self, **_kwargs):
            raise AssertionError("builder preflight must not execute the reader")

        def fingerprint(self):
            return {"reader": "fixture", "manifest_sha256": "1" * 64}

    class FakeEncoder:
        def encode(self, **_kwargs):
            raise AssertionError("builder preflight must not execute the encoder")

        def fingerprint(self):
            return {"encoder": "fixture", "model_load_set_sha256": "3" * 64}

    with (
        patch(
            "agentinputlab.algorithms.acquire.r2ag.LocalR2AGCommonSupportReader",
            lambda **_kwargs: FakeReader(),
        ),
        patch(
            "agentinputlab.algorithms.acquire.r2ag.LocalSentenceTransformerR2AGFeatureEncoder",
            lambda **_kwargs: FakeEncoder(),
        ),
    ):
        builder = ProductionBackendBuilder()
        assert builder.preflight(parameters) == ()
        runtime = builder.build(_backend_request(parameters))

    assert runtime.external_runtime.fingerprint()["runtime"] == (
        "r2ag_common_support_context_runtime.v1"
    )
    assert runtime.external_runtime.fingerprint()["reader"]["manifest_sha256"] == "1" * 64


def test_backend_builder_rejects_incomplete_r2ag_runtime_block(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    parameters["external_runtime"] = {
        "kind": "r2ag_common_support",
        "artifact_root": str(tmp_path),
    }
    blockers = ProductionBackendBuilder().preflight(parameters)
    assert len(blockers) == 1
    assert "R²AG external runtime backend is missing fields" in blockers[0]


def test_backend_builder_rejects_unknown_external_runtime(tmp_path: Path) -> None:
    parameters = _replay_parameters(tmp_path)
    parameters["external_runtime"] = {
        "kind": "moving_unreviewed_runtime",
        "checkout": str(tmp_path),
    }
    blockers = ProductionBackendBuilder().preflight(parameters)
    assert len(blockers) == 1
    assert "unsupported production external runtime kind" in blockers[0]


def test_openai_preflight_uses_secret_indirection_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {
        "schema_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model_id": "served-model",
            "base_url": "http://127.0.0.1:8000/v1",
            "deployment_identity": {
                "revision": "weights-sha",
                "server": "vllm",
            },
            "api_key_env": "AILAB_TEST_API_KEY",
            "require_api_key": True,
        },
        "required_capabilities": ["completion", "seed"],
    }
    builder = ProductionBackendBuilder()

    monkeypatch.delenv("AILAB_TEST_API_KEY", raising=False)
    assert builder.preflight(parameters) == (
        "required environment variable 'AILAB_TEST_API_KEY' is not set",
    )

    secret = "sk-unit-test-value"
    monkeypatch.setenv("AILAB_TEST_API_KEY", secret)
    with patch("urllib.request.build_opener", side_effect=AssertionError("network attempted")):
        assert builder.preflight(parameters) == ()
        runtime = builder.build(_backend_request(parameters))
    fingerprint = str(runtime.backend.fingerprint())
    assert secret not in fingerprint
    assert "AILAB_TEST_API_KEY" in fingerprint


def test_openai_deployment_revision_can_be_machine_bound_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {
        "schema_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model_id": "served-model",
            "base_url": "http://127.0.0.1:8000/v1",
            "deployment_identity": {"revision": "a" * 64},
            "deployment_revision_env": "VLLM_MODEL_TREE_SHA256",
        },
        "required_capabilities": ["completion"],
    }
    builder = ProductionBackendBuilder()

    monkeypatch.delenv("VLLM_MODEL_TREE_SHA256", raising=False)
    assert "required environment variable" in builder.preflight(parameters)[0]
    monkeypatch.setenv("VLLM_MODEL_TREE_SHA256", "b" * 64)
    assert "differs from its environment lock" in builder.preflight(parameters)[0]
    monkeypatch.setenv("VLLM_MODEL_TREE_SHA256", "a" * 64)
    assert builder.preflight(parameters) == ()
    runtime = builder.build(_backend_request(parameters))
    assert runtime.backend.fingerprint()["binding"]["deployment_revision_source"] == {
        "kind": "environment",
        "name": "VLLM_MODEL_TREE_SHA256",
    }


def test_literal_secrets_and_sensitive_headers_fail_closed() -> None:
    literal = {
        "schema_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model_id": "m",
            "base_url": "https://example.test/v1",
            "deployment_identity": {"revision": "r"},
            "api_key": "sk-should-never-be-here",
        },
    }
    blocker = ProductionBackendBuilder().preflight(literal)[0]
    assert "environment-variable name" in blocker
    assert "sk-should-never-be-here" not in blocker

    sensitive_header = {
        "schema_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model_id": "m",
            "base_url": "https://example.test/v1",
            "deployment_identity": {"revision": "r"},
            "extra_headers": {"Authorization": "not-a-secret-looking-value"},
        },
    }
    blocker = ProductionBackendBuilder().preflight(sensitive_header)[0]
    assert "sensitive header" in blocker
    assert "use api_key_env" in blocker

    google_api_key_header = {
        "schema_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model_id": "m",
            "base_url": "https://example.test/v1",
            "deployment_identity": {"revision": "r"},
            "extra_headers": {"X-Goog-Api-Key": "AIzaSyA-unit-test-literal-that-must-be-rejected"},
        },
    }
    blocker = ProductionBackendBuilder().preflight(google_api_key_header)[0]
    assert "sensitive header" in blocker or "secret-like text" in blocker


def test_tool_schema_sensitive_property_name_is_not_treated_as_a_secret() -> None:
    parameters = {
        "schema_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model_id": "m",
            "base_url": "https://example.test/v1",
            "deployment_identity": {"revision": "r"},
        },
        "tools": {
            "kind": "schema_only",
            "connector_id": "fixture-tools",
            "definitions": [
                {
                    "type": "function",
                    "function": {
                        "name": "change_password",
                        "description": "Change an account password.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "new_password": {"type": "string"},
                            },
                            "required": ["new_password"],
                        },
                    },
                }
            ],
        },
    }

    assert ProductionBackendBuilder().preflight(parameters) == ()

    parameters["tools"]["definitions"][0]["function"]["parameters"]["properties"][
        "new_password"
    ]["default"] = "sk-unit-test-literal-that-must-be-rejected"
    blocker = ProductionBackendBuilder().preflight(parameters)[0]
    assert "secret-like text" in blocker
    assert "sk-unit-test-literal-that-must-be-rejected" not in blocker


def test_production_backend_rejects_a_claimed_intent_gate_without_a_contract(
    tmp_path: Path,
) -> None:
    parameters = _replay_parameters(tmp_path)
    parameters["strict_intent"] = True
    blockers = ProductionBackendBuilder().preflight(parameters)
    assert len(blockers) == 1
    assert "cannot infer a per-example IntentContract" in blockers[0]


def test_capability_mismatch_is_a_concrete_preflight_blocker() -> None:
    parameters = {
        "schema_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model_id": "m",
            "base_url": "https://example.test/v1",
            "deployment_identity": {"revision": "r"},
            "capabilities": ["chat"],
        },
        "required_capabilities": ["completion"],
    }
    blockers = ProductionBackendBuilder().preflight(parameters)
    assert len(blockers) == 1
    assert "lacks capabilities: completion" in blockers[0]


def test_transformers_build_is_lazy_local_only_and_artifact_bound(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    weights = model_path / "model.safetensors"
    weights.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    parameters = {
        "schema_version": 1,
        "backend": {
            "kind": "transformers",
            "model_id": "local-model",
            "model_path": str(model_path),
            "model_revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "artifact_files": [
                {
                    "name": "model.safetensors",
                    "path": str(weights),
                    "sha256": digest,
                }
            ],
            "local_files_only": True,
        },
        "required_capabilities": ["completion"],
    }
    builder = ProductionBackendBuilder()

    with (
        patch(
            "agentinputlab.experiments.production_builders.importlib.util.find_spec",
            return_value=object(),
        ),
        patch(
            "agentinputlab.experiments.production_builders.importlib.import_module",
            side_effect=AssertionError("Transformers imported during build"),
        ),
    ):
        assert builder.preflight(parameters) == ()
        runtime = builder.build(_backend_request(parameters))
    fingerprint = runtime.backend.fingerprint()
    assert fingerprint["backend"]["local_files_only"] is True
    assert fingerprint["binding"]["artifact_files"][0]["sha256"] == digest
    assert fingerprint["binding"]["snapshot_sha256"] == fingerprint["backend"]["snapshot_sha256"]
    assert str(model_path) not in str(fingerprint)

    metadata = weights.stat()
    weights.write_bytes(b"changed")
    os.utime(
        weights,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
    )
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        runtime.backend._inner._load()

    weights.write_bytes(b"weights")
    (model_path / "unlisted.json").write_text("{}", encoding="utf-8")
    with patch(
        "agentinputlab.experiments.production_builders.importlib.util.find_spec",
        return_value=object(),
    ):
        assert "complete model snapshot" in builder.preflight(parameters)[0]

    parameters["backend"]["local_files_only"] = False
    assert "local_files_only=true" in builder.preflight(parameters)[0]


def test_prepared_corpus_retriever_build_and_product_validation(tmp_path: Path) -> None:
    parameters = _corpus_parameters(tmp_path)
    builder = PreparedCorpusRetrieverBuilder()
    request = ProductionRetrieverBuildRequest(
        benchmark_id="beir.arguana",
        model_slot="model-a",
        seed=7,
        builder_parameters=parameters,
    )

    assert builder.preflight(parameters) == ()
    retriever = builder.build(request)
    results = retriever.retrieve("alpha", top_k=2)
    assert [item.document.doc_id for item in results] == ["doc-a", "doc-b"]
    product_hash = builder.product_sha256(request)
    assert product_hash == retriever_product_sha256(retriever)
    assert validate_retriever_product_sha256(retriever, product_hash) == product_hash
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        validate_retriever_product_sha256(retriever, "0" * 64)


@pytest.mark.parametrize(
    "benchmark_id",
    (
        "tau3.banking_knowledge",
        "state_bench.common_support",
        "toolsandbox.common_support",
    ),
)
def test_backend_builder_accepts_main_arena_bm25_routes(
    tmp_path: Path,
    benchmark_id: str,
) -> None:
    parameters = _corpus_parameters(tmp_path, benchmark_id=benchmark_id)
    retriever_builder = PreparedCorpusRetrieverBuilder()
    request = ProductionRetrieverBuildRequest(
        benchmark_id=benchmark_id,
        model_slot="model-a",
        seed=7,
        builder_parameters=parameters,
    )
    retriever_spec = {
        "entrypoint": _RETRIEVER_ENTRYPOINT,
        "fingerprint_sha256": builder_fingerprint_sha256(retriever_builder),
        "expected_product_sha256": retriever_builder.product_sha256(request),
        "parameters": parameters,
    }
    backend_parameters = _replay_parameters(tmp_path)
    backend_parameters["retrievers"] = {benchmark_id: retriever_spec}

    runtime = ProductionBackendBuilder().build(_backend_request(backend_parameters))

    assert isinstance(runtime.retriever, ProductionRetrieverRouter)
    runtime.retriever.bind(benchmark_id)
    assert runtime.retriever.retrieve("alpha", top_k=1)[0].document.doc_id == "doc-a"


def test_backend_builder_injects_checksum_locked_vector_runtime_for_external_arena(
    tmp_path: Path,
) -> None:
    benchmark_id = "toolsandbox.common_support"
    parameters = _corpus_parameters(tmp_path, benchmark_id=benchmark_id)
    parameters["profile_id"] = "dense.e5.base_v2"
    model_path = tmp_path / "e5"
    model_path.mkdir()
    artifact = model_path / "model.safetensors"
    artifact.write_bytes(b"fixture-weights")
    parameters["options"] = {
        "retriever_id": "e5:toolsandbox",
        "model_path": str(model_path),
    }
    parameters["artifact_files"] = [
        {
            "name": "model.safetensors",
            "path": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    ]
    retriever_builder = PreparedCorpusRetrieverBuilder()
    retriever_request = ProductionRetrieverBuildRequest(
        benchmark_id=benchmark_id,
        model_slot="model-a",
        seed=7,
        builder_parameters=parameters,
    )

    with (
        patch(
            "agentinputlab.experiments.production_builders.importlib.util.find_spec",
            return_value=object(),
        ),
        patch(
            "agentinputlab.experiments.production_builders.build_retriever",
            side_effect=lambda _profile, documents, **_options: _FixtureVectorRuntime(documents),
        ),
    ):
        expected_product = retriever_builder.product_sha256(retriever_request)
        backend_parameters = _replay_parameters(tmp_path)
        backend_parameters["vector_runtime"] = {
            "entrypoint": _RETRIEVER_ENTRYPOINT,
            "fingerprint_sha256": builder_fingerprint_sha256(retriever_builder),
            "expected_product_sha256": expected_product,
            "parameters": parameters,
        }
        backend_builder = ProductionBackendBuilder()
        assert backend_builder.preflight(backend_parameters) == ()
        runtime = backend_builder.build(_backend_request(backend_parameters))

    assert runtime.dense_embedder is runtime.vector_searcher
    assert runtime.dense_embedder.embedder_id == "fixture-vector-runtime:embedder"
    assert runtime.vector_searcher.searcher_id == "fixture-vector-runtime:index"
    assert runtime.vector_searcher.search((1.0, 0.0), top_k=1)[0].document.doc_id == "doc-a"


def test_corpus_retriever_reports_missing_assets_and_count_mismatch(tmp_path: Path) -> None:
    parameters = _corpus_parameters(tmp_path)
    parameters["expected_document_count"] = 3
    blockers = PreparedCorpusRetrieverBuilder().preflight(parameters)
    assert blockers == ("prepared corpus document count mismatch: expected 3, got 2",)

    parameters["corpus_path"] = str(tmp_path / "missing.jsonl")
    blockers = PreparedCorpusRetrieverBuilder().preflight(parameters)
    assert any("unavailable" in item for item in blockers)


def test_retriever_router_is_bound_by_predictor_and_hash_stays_stable(
    tmp_path: Path,
) -> None:
    corpus_parameters = _corpus_parameters(tmp_path)
    retriever_builder = PreparedCorpusRetrieverBuilder()
    retriever_request = ProductionRetrieverBuildRequest(
        benchmark_id="beir.arguana",
        model_slot="model-a",
        seed=7,
        builder_parameters=corpus_parameters,
    )
    retriever_hash = retriever_builder.product_sha256(retriever_request)
    parameters = _replay_parameters(tmp_path)
    parameters["retrievers"] = {
        "beir.arguana": {
            "entrypoint": _RETRIEVER_ENTRYPOINT,
            "fingerprint_sha256": builder_fingerprint_sha256(retriever_builder),
            "expected_product_sha256": retriever_hash,
            "parameters": corpus_parameters,
        }
    }
    backend_builder = ProductionBackendBuilder()
    assert backend_builder.preflight(parameters) == ()
    runtime = backend_builder.build(_backend_request(parameters))
    assert isinstance(runtime.retriever, ProductionRetrieverRouter)
    before = static_runtime_product_sha256(runtime)

    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="beir.arguana",
            split="test",
            parameters={
                "schema_version": 1,
                "retrieval": {"top_k": 2, "query_mode": "query_only"},
            },
        )
    )
    assert static_runtime_product_sha256(runtime) == before
    downstream = predictor.predict(FrozenAgentState(query="alpha"))
    assert isinstance(downstream.prediction, RetrievalPrediction)
    assert tuple(downstream.prediction.scores) == ("doc-a", "doc-b")
    assert downstream.usage.retrieval_calls == 1
    assert downstream.metadata["query_sha256"]
    assert "alpha" not in json.dumps(dict(downstream.metadata))


def test_retrieval_rank_score_mode_preserves_ties_and_default_fingerprint() -> None:
    retriever = InMemoryBM25Retriever(tuple(
        Document(doc_id=doc_id, text="same evidence") for doc_id in ("z", "a", "b")
    ))
    ranked = tuple(RetrievalResult(document=Document(doc_id=doc_id, text="same evidence"),
                                   score=2.0, rank=rank)
                   for rank, doc_id in enumerate(("z", "a", "b"), start=1))
    runtime = StaticRuntimeComponents(backend=ReplayBackend(["unused"]), retriever=retriever)
    builder = ProductionPredictorBuilder()
    predictions = {}
    hashes = {}
    with patch.object(type(retriever), "retrieve", return_value=ranked) as retrieve:
        for mode in (None, "raw", "rank"):
            parameters = {"schema_version": 1, "retrieval": {"top_k": 3}}
            if mode is not None:
                parameters["retrieval"]["score_mode"] = mode
            request = _predictor_request(runtime, benchmark_id="beir.arguana",
                                         split="test", parameters=parameters)
            predictor = builder.build(request)
            hashes[mode] = builder.product_sha256(request)
            predictions[mode] = predictor.predict(FrozenAgentState(query="same"))
        assert retrieve.call_count == 3
    assert hashes[None] == hashes["raw"] != hashes["rank"]
    assert dict(predictions[None].prediction.scores) == {"z": 2.0, "a": 2.0, "b": 2.0}
    assert predictions[None] == predictions["raw"]
    assert dict(predictions["rank"].prediction.scores) == {"z": 3.0, "a": 2.0, "b": 1.0}
    assert predictions["rank"].metadata["score_mode"] == "rank"
    assert "score_mode" not in predictions[None].metadata
    assert all(item.usage.retrieval_calls == 1 and item.usage.model_calls == 0
               for item in predictions.values())


@pytest.mark.parametrize("score_mode", ["best", "RAW", None, True, 3, [], {}])
def test_retrieval_score_mode_rejects_invalid_values(score_mode) -> None:
    errors = ProductionPredictorBuilder().preflight(
        {"schema_version": 1, "retrieval": {"score_mode": score_mode}}
    )
    assert len(errors) == 1 and "score_mode must be raw or rank" in errors[0]


def test_retrieval_predictor_prefers_visible_document_lineage_without_new_call() -> None:
    runtime = StaticRuntimeComponents(
        backend=ReplayBackend(["unused"]),
        retriever=InMemoryBM25Retriever((Document(doc_id="fallback", text="fallback evidence"),)),
    )
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="beir.arguana",
            split="test",
            parameters={
                "schema_version": 1,
                "retrieval": {
                    "top_k": 3,
                    "query_mode": "query_only",
                    "source_mode": "context_or_retriever",
                },
            },
        )
    )
    downstream = predictor.predict(
        FrozenAgentState(
            query="query that would miss",
            context=(
                ContextBlock(text="first", metadata={"document_id": "doc-a"}),
                ContextBlock(
                    text="compressed",
                    metadata={"source_block_ids": ("doc-b", "doc-c")},
                ),
            ),
        )
    )
    assert tuple(downstream.prediction.scores) == ("doc-a", "doc-b", "doc-c")
    assert downstream.usage.retrieval_calls == 0
    assert downstream.metadata["prediction_kind"] == "retrieval_visible_context"


def test_retrieval_predictor_requires_a_locked_runtime_retriever(tmp_path: Path) -> None:
    runtime = ProductionBackendBuilder().build(_backend_request(_replay_parameters(tmp_path)))
    with pytest.raises(CapabilityError, match="requires a locked runtime retriever"):
        ProductionPredictorBuilder().build(
            _predictor_request(
                runtime,
                benchmark_id="bright.economics",
                split="test",
            )
        )


def test_qa_predictor_uses_visible_state_only_and_accounts_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProductionBackendBuilder().build(_backend_request(_replay_parameters(tmp_path)))
    captured = {}

    def generate(_self, request):
        captured["request"] = request
        return GenerationResponse(
            text=json.dumps(
                {
                    "answer": "visible answer",
                    "support_ids": ["block-a"],
                    "evidence_ids": [],
                    "evidence_texts": ["visible context"],
                }
            ),
            model="served",
            finish_reason="stop",
            usage=Usage(input_tokens=13, output_tokens=2, cost_usd=None),
        )

    monkeypatch.setattr(type(runtime.backend), "generate", generate)
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="musique.answerable",
            split="dev",
            parameters={
                "schema_version": 1,
                "qa": {
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "prompt_revision": "fixture-v1",
                },
            },
        )
    )
    state = FrozenAgentState(
        task_id="secret-task-id",
        query="visible question",
        instruction="visible instruction",
        context=(ContextBlock(text="visible context", block_id="block-a"),),
        messages=(Message(role="user", content="visible history"),),
        metadata={"evaluator_gold": "secret answer"},
    )
    downstream = predictor.predict(state)

    assert downstream.prediction == QAPrediction(
        answer="visible answer",
        support_ids=("block-a",),
        evidence_texts=("visible context",),
    )
    assert downstream.usage.model_calls == 1
    assert downstream.usage.unpriced_model_calls == 1
    assert downstream.usage.input_tokens == 13
    assert downstream.usage.output_tokens == 2
    request_text = json.dumps(
        {
            "prompt": captured["request"].prompt,
            "system": captured["request"].system,
            "metadata": dict(captured["request"].metadata),
        }
    )
    assert "visible question" in request_text
    assert "visible context" in request_text
    assert "secret-task-id" not in request_text
    assert "secret answer" not in request_text
    assert captured["request"].response_format == {"type": "json_object"}


def test_structured_qa_predictor_filters_and_audits_hidden_support_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = StaticRuntimeComponents(backend=ReplayBackend(["unused"]))

    def generate(_self, request):
        del request
        return GenerationResponse(
            text=json.dumps(
                {
                    "answer": "answer",
                    "support_ids": ["gold-only-id"],
                    "evidence_ids": [],
                    "evidence_texts": [],
                }
            ),
            model="served",
        )

    monkeypatch.setattr(type(runtime.backend), "generate", generate)
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="musique.answerable",
            split="dev",
        )
    )
    state = FrozenAgentState(
        query="question",
        context=(ContextBlock(text="visible", block_id="visible-id"),),
    )
    downstream = predictor.predict(state)
    assert downstream.prediction == QAPrediction(answer="answer")
    assert downstream.metadata["support_id_validation"] == "filter_unknown_visible_only.v1"
    assert downstream.metadata["valid_support_id_count"] == 0
    assert downstream.metadata["invalid_support_id_count"] == 1
    assert downstream.metadata["invalid_support_ids_sha256"] == content_hash(("gold-only-id",))


def test_structured_qa_predictor_preserves_visible_support_order_when_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = StaticRuntimeComponents(backend=ReplayBackend(["unused"]))

    def generate(_self, request):
        del request
        return GenerationResponse(
            text=json.dumps(
                {
                    "answer": "answer",
                    "support_ids": ["visible-b", "hidden", "visible-a"],
                    "evidence_ids": [],
                    "evidence_texts": [],
                }
            ),
            model="served",
        )

    monkeypatch.setattr(type(runtime.backend), "generate", generate)
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="musique.answerable",
            split="dev",
        )
    )
    state = FrozenAgentState(
        query="question",
        context=(
            ContextBlock(text="first", block_id="visible-a"),
            ContextBlock(text="second", block_id="visible-b"),
        ),
    )

    downstream = predictor.predict(state)

    assert downstream.prediction == QAPrediction(
        answer="answer",
        support_ids=("visible-b", "visible-a"),
    )
    assert downstream.metadata["valid_support_id_count"] == 2
    assert downstream.metadata["invalid_support_id_count"] == 1
    assert downstream.metadata["invalid_support_ids_sha256"] == content_hash(("hidden",))


def test_structured_qa_predictor_deduplicates_and_audits_evidence_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = StaticRuntimeComponents(backend=ReplayBackend(["unused"]))

    def generate(_self, request):
        del request
        return GenerationResponse(
            text=json.dumps(
                {
                    "answer": "answer",
                    "support_ids": [],
                    "evidence_ids": [],
                    "evidence_texts": ["first", " second ", "first", "second"],
                }
            ),
            model="served",
        )

    monkeypatch.setattr(type(runtime.backend), "generate", generate)
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="musique.answerable",
            split="dev",
        )
    )

    downstream = predictor.predict(FrozenAgentState(query="question"))

    assert downstream.prediction == QAPrediction(
        answer="answer",
        evidence_texts=("first", "second"),
    )
    assert downstream.metadata["evidence_text_duplicate_policy"] == (
        "deduplicate_exact_trimmed_keep_first.v1"
    )
    assert downstream.metadata["duplicate_evidence_text_count"] == 2
    assert downstream.metadata["duplicate_evidence_texts_sha256"] == content_hash(
        ("first", "second")
    )


def test_two_wiki_predictor_filters_and_audits_malformed_evidence_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = StaticRuntimeComponents(backend=ReplayBackend(["unused"]))
    valid_evidence = canonical_json(["subject", "relation", "object"])

    def generate(_self, request):
        del request
        return GenerationResponse(
            text=json.dumps(
                {
                    "answer": "answer",
                    "support_ids": ["opaque-visible-block"],
                    "evidence_ids": [valid_evidence, "not-json"],
                    "evidence_texts": [],
                }
            ),
            model="served",
        )

    monkeypatch.setattr(type(runtime.backend), "generate", generate)
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="2wiki.multihop",
            split="dev",
        )
    )

    downstream = predictor.predict(
        FrozenAgentState(
            query="question",
            context=(ContextBlock(text="compressed", block_id="opaque-visible-block"),),
        )
    )

    assert downstream.prediction == QAPrediction(
        answer="answer",
        evidence_ids=(valid_evidence,),
    )
    assert downstream.metadata["evidence_id_validation"] == (
        "filter_invalid_2wiki_canonical_triples.v1"
    )
    assert downstream.metadata["valid_evidence_id_count"] == 1
    assert downstream.metadata["invalid_evidence_id_count"] == 1
    assert downstream.metadata["invalid_evidence_ids_sha256"] == content_hash(("not-json",))
    assert downstream.metadata["support_id_validation"] == (
        "filter_unknown_or_invalid_canonical_support.v1"
    )
    assert downstream.metadata["valid_support_id_count"] == 0
    assert downstream.metadata["invalid_support_id_count"] == 1
    assert downstream.metadata["invalid_support_ids_sha256"] == content_hash(
        ("opaque-visible-block",)
    )


def test_qasper_predictor_can_request_bounded_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = StaticRuntimeComponents(
        backend=ReplayBackend(
            ["unused"],
            capabilities=frozenset(
                {
                    ModelCapability.COMPLETION,
                    ModelCapability.JSON_SCHEMA,
                    ModelCapability.SEED,
                }
            ),
        )
    )
    captured = {}

    def generate(_self, request):
        captured["request"] = request
        return GenerationResponse(
            text=json.dumps(
                {
                    "answer": "yes",
                    "support_ids": [],
                    "evidence_ids": [],
                    "evidence_texts": ["visible evidence"],
                }
            ),
            model="served",
        )

    monkeypatch.setattr(type(runtime.backend), "generate", generate)
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="qasper.v0_3_dev",
            split="dev",
            parameters={
                "schema_version": 1,
                "qa": {
                    "structured_output_mode": "json_schema",
                    "use_seed": True,
                    "required_capabilities": ["completion", "json_schema", "seed"],
                },
            },
        )
    )
    downstream = predictor.predict(
        FrozenAgentState(
            query="question",
            context=(ContextBlock(text="visible evidence", block_id="block-a"),),
        )
    )

    assert downstream.prediction == QAPrediction(
        answer="yes",
        evidence_texts=("visible evidence",),
    )
    response_format = captured["request"].response_format
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["support_ids"]["maxItems"] == 0
    assert schema["properties"]["evidence_ids"]["maxItems"] == 0
    assert schema["properties"]["evidence_texts"]["maxItems"] == 3
    assert "uniqueItems" not in schema["properties"]["evidence_texts"]


@pytest.mark.parametrize(
    ("benchmark_id", "split", "task"),
    (
        ("beir.arguana", "test", "retrieval"),
        ("bright.earth_science", "test", "retrieval"),
        ("bright.economics", "test", "retrieval"),
        ("musique.answerable", "dev", "qa"),
        ("2wiki.multihop", "dev", "qa"),
        ("hotpotqa.distractor", "validation", "qa"),
        ("qasper.v0_3_dev", "dev", "qa"),
    ),
)
def test_predictor_routes_all_seven_production_cells(
    benchmark_id: str,
    split: str,
    task: str,
) -> None:
    runtime = StaticRuntimeComponents(
        backend=ReplayBackend(["answer"]),
        retriever=InMemoryBM25Retriever(
            [Document(doc_id="d", text="document")],
            retriever_id="fixture",
        ),
    )
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id=benchmark_id,
            split=split,
            parameters={"schema_version": 1, "task": "auto"},
        )
    )
    assert predictor.fingerprint()["task"] == task


def test_predictor_rejects_wrong_split_task_and_seed_capability(tmp_path: Path) -> None:
    runtime = ProductionBackendBuilder().build(_backend_request(_replay_parameters(tmp_path)))
    builder = ProductionPredictorBuilder()
    with pytest.raises(ValidationError, match="expects split"):
        builder.build(
            _predictor_request(
                runtime,
                benchmark_id="qasper.v0_3_dev",
                split="test",
            )
        )
    with pytest.raises(ValidationError, match="incompatible"):
        builder.build(
            _predictor_request(
                runtime,
                benchmark_id="qasper.v0_3_dev",
                split="dev",
                parameters={"schema_version": 1, "task": "retrieval"},
            )
        )

    no_seed_runtime = StaticRuntimeComponents(
        backend=ReplayBackend(
            ["answer"],
            capabilities=frozenset({ModelCapability.COMPLETION}),
        )
    )
    with pytest.raises(CapabilityError, match="seed"):
        builder.build(
            _predictor_request(
                no_seed_runtime,
                benchmark_id="qasper.v0_3_dev",
                split="dev",
                parameters={"schema_version": 1, "qa": {"use_seed": True}},
            )
        )


def test_runtime_and_predictor_product_hash_helpers(tmp_path: Path) -> None:
    backend_builder = ProductionBackendBuilder()
    runtime = backend_builder.build(_backend_request(_replay_parameters(tmp_path)))
    runtime_hash = static_runtime_product_sha256(runtime)
    assert validate_runtime_product_sha256(runtime, runtime_hash) == runtime_hash
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        validate_runtime_product_sha256(runtime, "0" * 64)

    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="qasper.v0_3_dev",
            split="dev",
        )
    )
    predictor_hash = static_predictor_product_sha256(predictor)
    assert validate_predictor_product_sha256(predictor, predictor_hash) == predictor_hash
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        validate_predictor_product_sha256(predictor, "0" * 64)


def test_predictor_recipe_hash_is_stable_across_locked_model_slots() -> None:
    runtime_a = StaticRuntimeComponents(backend=ReplayBackend(["answer-a"], model_id="model-a"))
    runtime_b = StaticRuntimeComponents(backend=ReplayBackend(["answer-b"], model_id="model-b"))
    builder = ProductionPredictorBuilder()
    predictor_a = builder.build(
        _predictor_request(
            runtime_a,
            benchmark_id="qasper.v0_3_dev",
            split="dev",
        )
    )
    predictor_b = builder.build(
        _predictor_request(
            runtime_b,
            benchmark_id="qasper.v0_3_dev",
            split="dev",
        )
    )

    assert static_runtime_product_sha256(runtime_a) != static_runtime_product_sha256(runtime_b)
    assert static_predictor_product_sha256(predictor_a) == static_predictor_product_sha256(
        predictor_b
    )
    assert predictor_a.fingerprint()["runtime_binding_slot"] == "backend"


def test_retriever_router_rejects_wrong_product_hash(tmp_path: Path) -> None:
    corpus_parameters = _corpus_parameters(tmp_path)
    spec = {
        "entrypoint": _RETRIEVER_ENTRYPOINT,
        "fingerprint_sha256": builder_fingerprint_sha256(PreparedCorpusRetrieverBuilder()),
        "expected_product_sha256": "0" * 64,
        "parameters": corpus_parameters,
    }
    parameters = _replay_parameters(tmp_path)
    parameters["retrievers"] = {"beir.arguana": spec}
    runtime = ProductionBackendBuilder().build(_backend_request(parameters))
    predictor = ProductionPredictorBuilder().build(
        _predictor_request(
            runtime,
            benchmark_id="beir.arguana",
            split="test",
        )
    )
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        predictor.predict(FrozenAgentState(query="alpha"))
