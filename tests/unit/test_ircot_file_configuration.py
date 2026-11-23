"""Public IRCoT configuration regressions; fixtures are not model acceptance."""

import json
import sys
from dataclasses import replace

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts.errors import ValidationError


def configuration(tmp_path, full=False):
    return RunConfig(
        "acquire.ircot.native_like_substituted" if full else "acquire.ircot.controlled",
        parameters=({"manifest_path": str(tmp_path / "manifest.json"),
                     "expected_manifest_sha256": "a" * 64} if full else
                    {"corpus_revision": "corpus-v1", "index_revision": "index-v1", "prompt_hash": "b" * 64}),
        backend={"model_id": "test-model", "base_url": "http://127.0.0.1:1/v1",
                 "deployment_identity": {"revision": "test"}},
        retriever={"kind": "bm25", "documents": [{"doc_id": "d", "text": "evidence"}]},
        runtime={"kind": "ircot_native", "source_root": str(tmp_path),
                 "python_executable": sys.executable, "tokenizer_root": str(tmp_path),
                 "tokenizer_manifest": str(tmp_path / "tokenizer.json"),
                 "tokenizer_manifest_sha256": "a" * 64, "nltk_data": str(tmp_path),
                 "corpus_revision": "corpus-v1", "index_revision": "index-v1",
                 "per_call_output_tokens": 512})


@pytest.mark.parametrize("full", [False, True])
def test_file_runtime_shape_accepts_both_native_profiles_without_io(tmp_path, full):
    assert configuration(tmp_path, full).validate()["valid"]


def test_relative_native_paths_and_substitution_manifest(tmp_path):
    config = configuration(tmp_path, True).to_dict()
    config["parameters"]["manifest_path"] = "ack.json"
    config["runtime"].update(tokenizer_root="tokenizer", tokenizer_manifest="closure.json")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config))
    loaded = RunConfig.from_file(path)
    assert loaded.runtime["tokenizer_root"] == str(tmp_path / "tokenizer")
    assert loaded.runtime["tokenizer_manifest"] == str(tmp_path / "closure.json")
    assert loaded.parameters["manifest_path"] == str(tmp_path / "ack.json")


@pytest.mark.parametrize("field,value", [("shell", "x"), ("per_call_output_tokens", True),
    ("per_call_output_tokens", 0), ("timeout_seconds", float("inf")),
    ("tokenizer_manifest_sha256", "unchecked"), ("index_revision", "")])
def test_invalid_native_runtime_rejected(tmp_path, field, value):
    config = configuration(tmp_path)
    with pytest.raises(ValidationError):
        replace(config, runtime={**config.runtime, field: value}).validate()


def test_native_profile_not_silently_used_for_common_support(tmp_path):
    config = configuration(tmp_path)
    with pytest.raises(ValidationError):
        replace(config, release_id="baseline.no_op", parameters={}).validate()


def test_full_worker_restores_public_typed_request_not_internal_parsed_request(tmp_path, monkeypatch):
    from agentinputlab.algorithms.acquire.ircot_native import IRCoTNativeQARequest
    from agentinputlab.contracts.serialization import to_jsonable
    from agentinputlab.integrations.external_workers.ircot import IRCoTControls, IRCoTWorkerRequest
    from workers.ircot import configured, native_qa

    config = configuration(tmp_path, True)
    identity = {"native": "fixture"}
    legacy = IRCoTWorkerRequest(request_id="request", state_hash="state", task_id="case",
        question="question", controls=IRCoTControls(),
        generator_fingerprint=config._backend().fingerprint(),
        retriever_fingerprint=config._retriever().fingerprint(),
        corpus_revision="corpus-v1", index_revision="index-v1", prompt_hash="b" * 64)
    request = IRCoTNativeQARequest(legacy, 8, 512, 17, "a" * 64)

    class Runtime:
        def __init__(self, **kwargs):
            pass

        def fingerprint(self):
            return identity

        def run(self, restored):
            # The actual native runtime immediately calls legacy.as_dict().
            assert restored.legacy.as_dict() == legacy.as_dict()
            assert isinstance(restored.legacy, IRCoTWorkerRequest)
            assert restored == request
            return {"status": "ok"}

    monkeypatch.setattr(native_qa, "PinnedIRCoTNativeQARuntime", Runtime)
    monkeypatch.setattr(configured, "_tokenizer", lambda value: (object(), {}))
    monkeypatch.setattr(configured, "_dependencies", lambda: {})
    payload = {"configuration": config.to_dict(), "operation": "full",
        "generator": to_jsonable(config._backend().fingerprint()),
        "retriever": to_jsonable(config._retriever().fingerprint()),
        "expected_identity": identity, "request": to_jsonable(request)}
    assert configured.execute(payload, []) == {"status": "ok"}


def test_native_worker_is_private_and_captures_responses_without_retry(tmp_path, monkeypatch):
    import os

    from agentinputlab.benchmark_configuration import _CapturedBackend
    from agentinputlab.integrations.external_workers.base import JsonLineSubprocessWorker
    from agentinputlab.ircot_configuration import FileIRCoTRuntime

    config = configuration(tmp_path)
    backend = _CapturedBackend(config._backend())
    retriever = config._retriever()
    original_home = os.environ.get("HOME")
    invoked = []
    identity = {"bound": True}
    saved = {"status": "responded", "response": {"text": "original"}}

    def call(worker, payload):
        invoked.append(payload)
        assert worker.command == (sys.executable, "-I", "-B", "-m", "workers.ircot.configured")
        assert worker.environment["HOME"] != original_home
        assert worker.environment["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ.get("HOME") == original_home
        assert payload["generator"] == backend.fingerprint()
        if payload["operation"] == "fingerprint":
            return {"status": "ok", "result": identity, "model_responses": []}
        assert payload["expected_identity"] == identity
        return {"status": "ok", "result": {"status": "error", "usage": {"model_calls": 1}},
                "model_responses": [saved]}

    monkeypatch.setattr(JsonLineSubprocessWorker, "call", call)
    runtime = FileIRCoTRuntime(config, backend, retriever)
    assert runtime.fingerprint() == identity
    assert runtime.fingerprint() == identity
    assert runtime.call({"request_id": "one"})["status"] == "error"
    assert len(invoked) == 2 and backend.calls == [saved]
    assert os.environ.get("HOME") == original_home


def test_worker_error_retains_capture_and_does_not_retry(tmp_path, monkeypatch):
    from agentinputlab.benchmark_configuration import _CapturedBackend
    from agentinputlab.contracts.errors import ExternalDependencyError
    from agentinputlab.integrations.external_workers.base import JsonLineSubprocessWorker
    from agentinputlab.ircot_configuration import FileIRCoTRuntime

    config = configuration(tmp_path)
    backend = _CapturedBackend(config._backend())
    calls = []

    def call(worker, payload):
        calls.append(payload)
        return {"status": "error", "error_type": "ValueError", "model_responses": [{"status": "failed"}]}

    monkeypatch.setattr(JsonLineSubprocessWorker, "call", call)
    runtime = FileIRCoTRuntime(config, backend, config._retriever())
    with pytest.raises(ExternalDependencyError, match="ValueError"):
        runtime.call({})
    assert len(calls) == 1 and backend.calls == [{"status": "failed"}]


def test_preparation_creates_controlled_binding_and_never_overwrites(tmp_path, monkeypatch):
    from agentinputlab.ircot_configuration import FileIRCoTRuntime, prepare

    config = replace(configuration(tmp_path), parameters={"controls": {"max_steps": 2}})
    identity = {"corpus_revision": "actual-corpus", "index_revision": "actual-index", "prompt_sha256": "c" * 64}
    invoked = []

    def fingerprint(runtime):
        invoked.append("fingerprint")
        return identity

    monkeypatch.setattr(FileIRCoTRuntime, "fingerprint", fingerprint)
    output = tmp_path / "prepared"
    with pytest.raises(ValidationError, match="acknowledge"):
        prepare(config, output)
    assert not invoked and not output.exists()
    proof = prepare(config, output, accept_substitutions=True)
    ready = RunConfig.from_file(proof["run_path"])
    assert ready.parameters["prompt_hash"] == "c" * 64
    assert ready.parameters["controls"]["max_steps"] == 2
    assert proof["model_calls"] == 0
    with pytest.raises(ValidationError, match="new output"):
        prepare(config, output, accept_substitutions=True)
    assert invoked == ["fingerprint"]


def test_missing_worker_and_nonserializable_injected_dependencies_fail(tmp_path):
    from agentinputlab.ircot_configuration import FileIRCoTRuntime

    config = configuration(tmp_path)
    config = replace(config, runtime={**config.runtime, "python_executable": str(tmp_path / "gone")})
    with pytest.raises(ValidationError, match="worker Python"):
        FileIRCoTRuntime(config, config._backend(), config._retriever())
    with pytest.raises(ValidationError, match="file backend"):
        replace(config, backend={}).validate()


def test_full_preparation_pins_distinct_six_role_manifest(tmp_path, monkeypatch):
    from agentinputlab.ircot_configuration import FileIRCoTRuntime, prepare
    from tests.unit.test_ircot_native_qa import RuntimeFixture

    config = replace(configuration(tmp_path, True), parameters={})
    fixture = RuntimeFixture(config._backend(), config._retriever())
    monkeypatch.setattr(FileIRCoTRuntime, "fingerprint", lambda runtime: fixture.fingerprint())
    proof = prepare(config, tmp_path / "full", accept_substitutions=True)
    ready = RunConfig.from_file(proof["run_path"])
    method = ready.create_intervention()
    assert method.release_id == "acquire.ircot.native_like_substituted"
    assert method._binding["original_controller_and_six_participants"] is True
    assert method._binding["shared_final_call_and_output_budget"] is True


def test_tokenizer_download_bookkeeping_is_not_model_configuration(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from workers.ircot.configured import _tokenizer

    tokenizer_type = type("Tokenizer", (), {"__module__": "transformers.models.gpt2.fixture"})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path, **kwargs: tokenizer_type())))
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text('{}')
    closure = tmp_path / "closure.json"
    closure.write_text(json.dumps({"file_sha256": {
        "tokenizer.json": hashlib.sha256(b'{}').hexdigest()}}))
    cache = root / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (cache / "tokenizer.json.lock").write_text('')
    value = {"tokenizer_root": str(root), "tokenizer_manifest": str(closure),
             "tokenizer_manifest_sha256": hashlib.sha256(closure.read_bytes()).hexdigest()}
    assert type(_tokenizer(value)[0]) is tokenizer_type
    (root / "added_tokens.json").write_text('{}')
    with pytest.raises(ValueError, match="closure"):
        _tokenizer(value)
