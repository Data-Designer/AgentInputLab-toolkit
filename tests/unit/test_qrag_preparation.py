"""Preparation fixtures are not neural compatibility evidence."""
import json
from dataclasses import replace

import pytest

from agentinputlab import RunConfig
from agentinputlab.contracts import ContextBlock, FrozenAgentState
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from tests.unit.test_qrag_file_configuration import configuration


def template(tmp_path):
    value = configuration()
    value["parameters"].pop("artifacts")
    from workers.qrag.driver import DEFAULT_MANIFEST
    value["runtime"]["compatibility_manifest_path"] = str(DEFAULT_MANIFEST)
    value["runtime"]["embedder_root"] = str(tmp_path)
    from agentinputlab.algorithms.acquire.qrag.bridge import (
        QRAG_EMBEDDER_REPOSITORY,
        QRAG_EMBEDDER_REVISION,
    )
    (tmp_path / "agentinputlab-manifest.json").write_text(json.dumps({
        "repository": QRAG_EMBEDDER_REPOSITORY, "revision": QRAG_EMBEDDER_REVISION}))
    return RunConfig.from_dict(value)


@pytest.mark.parametrize("profile", ["factorial_reader_transfer", "factorial_native_query_reader_transfer"])
def test_preparation_generates_identity_without_inference(tmp_path, monkeypatch, profile):
    from agentinputlab.qrag_configuration import prepare
    from workers.qrag.driver import QRAGFrozenDriver
    config = template(tmp_path)
    state = FrozenAgentState(query="question", context=(ContextBlock(block_id="d", text="text"),))
    verified = []
    for name in ("_verify_source", "_verify_checkpoint", "_verify_embedder"):
        monkeypatch.setattr(QRAGFrozenDriver, name, lambda self, a, n=name: verified.append(n))
    monkeypatch.setattr(QRAGFrozenDriver, "handle", lambda *_: pytest.fail("no inference"))
    out = tmp_path / "prepared"
    proof = prepare(config, state, out, reader_repository="local://fixture",
                    reader_revision="fixture", reader_profile=profile)
    assert len(verified) == 3
    assert proof["model_calls"] == proof["embedding_calls"] == 0
    ready = RunConfig.from_file(out / "run.json")
    identity = ready.create_intervention().artifacts
    assert identity.reader_profile == profile
    assert identity.corpus_sha256 == content_hash(state.context)
    assert identity.index_sha256 == content_hash(["d"])
    assert identity.reader_deployment_fingerprint_sha256 == content_hash(config._backend().fingerprint())
    assert ready.runtime["prepared_state_sha256"] == replace(state, budget=config.budget).state_hash
    assert ready.budget == config.budget and ready.seed == config.seed
    assert ready.validate()["valid"]
    from agentinputlab.configuration import load_run_state
    assert load_run_state(out / "state.json").state_hash == ready.runtime["prepared_state_sha256"]
    with pytest.raises(ValidationError, match="new output directory"):
        prepare(config, state, out, reader_repository="local://fixture",
                reader_revision="fixture", reader_profile=profile)
    assert len(verified) == 3


def test_preparation_rejects_real_missing_source_and_does_not_write(tmp_path):
    from agentinputlab.qrag_configuration import prepare
    config = template(tmp_path)
    with pytest.raises(ValidationError, match="assets"):
        prepare(config, FrozenAgentState(query="question"), tmp_path / "out",
                reader_repository="local://fixture", reader_revision="fixture",
                reader_profile="factorial_reader_transfer")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", [
    {"parameters": {"artifacts": {}}}, {"retriever": {"kind": "bm25"}},
    {"release_id": "acquire.qrag.common_support"}, {"runtime": {}},
])
def test_preparation_rejects_ambiguous_or_wrong_profile(tmp_path, change):
    from agentinputlab.qrag_configuration import prepare
    with pytest.raises(ValidationError):
        prepare(replace(template(tmp_path), **change), FrozenAgentState(query="question"),
                tmp_path / "out", reader_repository="local://fixture", reader_revision="fixture",
                reader_profile="factorial_reader_transfer")
    assert not (tmp_path / "out").exists()


def test_prepared_runtime_rejects_different_state_before_worker(tmp_path, monkeypatch):
    from agentinputlab.qrag_configuration import FileQRAGRuntime
    value = configuration()
    value["runtime"]["prepared_state_sha256"] = "b" * 64
    config = RunConfig.from_dict(value)
    backend = config._backend()
    value["parameters"]["artifacts"]["reader_deployment_fingerprint_sha256"] = content_hash(backend.fingerprint())
    runtime = FileQRAGRuntime(RunConfig.from_dict(value), backend)
    from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker
    monkeypatch.setattr(JsonLineSubprocessWorker, "call", lambda *_: pytest.fail("no worker"))
    result = runtime.call({"state_hash": "c" * 64, "payload": {"artifacts": value["parameters"]["artifacts"]}})
    assert result["consumed"] is False
    assert result["failure_code"] == "QRAG_PREPARED_STATE_MISMATCH"


@pytest.mark.parametrize("state", [FrozenAgentState(query="", instruction="instruction only"),
    FrozenAgentState(query="q", instruction="preserve instructions")])
def test_question_only_preparation_rejects_unsupported_state_before_assets(tmp_path, state):
    from agentinputlab.qrag_configuration import prepare
    with pytest.raises(ValidationError, match="visible state"):
        prepare(template(tmp_path), state, tmp_path / "out", reader_repository="fixture",
                reader_revision="fixture", reader_profile="factorial_reader_transfer")
    assert not (tmp_path / "out").exists()


def test_prepare_command_is_public_and_requires_explicit_inputs():
    from agentinputlab.cli.main import build_parser
    args = build_parser().parse_args(["config", "prepare-qrag", "template.json", "--state", "state.json",
        "--reader-repository", "fixture", "--reader-revision", "fixture",
        "--reader-profile", "factorial_reader_transfer", "--output-dir", "prepared"])
    assert args.func.__name__ == "_cmd_prepare_qrag"
