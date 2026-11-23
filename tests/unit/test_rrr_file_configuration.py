"""File/runtime wiring tests; synthetic source is not model acceptance."""
import hashlib
import json
from dataclasses import replace

import pytest

from agentinputlab import FrozenAgentState, InterventionContext, RunConfig
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from tests.unit.test_benchmark_configuration import Backend
from workers.rrr import driver


class Reader(Backend):
    def fingerprint(self):
        return {"model_id": self.model_id, "model_revision": "fixture-reader"}


@pytest.fixture
def configured(tmp_path, monkeypatch):
    state = FrozenAgentState(query="question", task_id="fixture")
    retriever = InMemoryBM25Retriever([Document(doc_id="doc", text="question evidence", corpus_revision="fixture")])
    backend = Reader([("answer**", 3)])
    rewrite, reader, manifest = (tmp_path / x for x in ("rewrite.txt", "reader.txt", "manifest.json"))
    rewrite.write_text("rewrite {query}")
    reader.write_text("read {query} {background}")
    manifest.write_text("{}")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    checkpoint = dict(role="rewriter", profile="warmup_checkpoint", revision="fixture-warmup",
                      manifest_sha256=sha(manifest), root=tmp_path, model_type="seq2seq")
    identity = {k: checkpoint[k] for k in ("role", "profile", "revision", "manifest_sha256")}
    saved = dict(schema="agentinputlab.rrr.rewriter-replay.v1",
        request=dict(prompt="rewrite question", seed=17, max_tokens=100, checkpoint=identity),
        response=dict(text="question**", input_tokens=5, output_tokens=2))
    replay = tmp_path / "rewrite.json"
    replay.write_text(json.dumps(saved))
    parameters = dict(source_license_acknowledged=True, checkpoint_license_acknowledged=True,
        prompt_hash=sha(rewrite), reader_prompt_hash=sha(reader),
        search_artifact_hash=content_hash(retriever.fingerprint()),
        corpus_revision=content_hash(dict(original_visible_state=state.state_hash, context=to_jsonable(state.context))),
        index_revision=content_hash(retriever.fingerprint()), rewriter_checkpoint_revision="fixture-warmup",
        reader_checkpoint_revision="fixture-reader", controls=dict(search_mode="controlled_static", rewriter_profile="warmup_checkpoint"))
    runtime = dict(kind="rrr_checkpoint", checkout=str(tmp_path), checkpoint_root=str(tmp_path),
        checkpoint_manifest_path=str(manifest), checkpoint_manifest_sha256=sha(manifest),
        rewrite_prompt_path=str(rewrite), reader_prompt_path=str(reader), acknowledge_credential_quarantine=True,
        rewriter_replay_path=str(replay), rewriter_replay_sha256=sha(replay))
    config = RunConfig("rewrite.rrr.warmup_external", parameters=parameters, runtime=runtime, seed=17, budget=state.budget)
    monkeypatch.setattr(driver, "verify_source_checkout", lambda p: driver.SourceEvidence("a" * 64, "fake_checkout_plumbing", 0))
    monkeypatch.setattr(driver, "verify_checkpoint", lambda *a: checkpoint)
    monkeypatch.setattr(driver, "LocalHFGenerator", lambda *a: pytest.fail("saved generation must not load or run a model"))
    return config, state, InterventionContext(backend=backend, retriever=retriever), saved, replay


def test_saved_rewriter_native_parser_search_and_reader(configured):
    config, state, context, _, _ = configured
    result = config.run(state, context=context)
    assert result["state"]["query"] == "question"
    assert result["record"]["usage"]["model_calls"] == 2
    assert result["record"]["usage"]["output_tokens"] == 5
    assert result["record"]["usage"]["retrieval_calls"] == 1
    assert len(context.backend.requests) == 1
    req = context.backend.requests[0]
    assert req.max_tokens == 300 and req.seed == 17 and req.temperature == 0 and req.stop == ()
    assert "question evidence" in req.prompt
    bound = config._bind_case_context(context, state)
    assert bound.external_runtime.fingerprint()["rewriter_execution"] == "explicit_saved_generation"
    with pytest.raises(ValidationError, match="third reader"):
        bound.backend.generate(req)


@pytest.mark.parametrize("field,value", [("seed", 18), ("max_tokens", 101), ("prompt", "changed")])
def test_saved_rewriter_request_mismatch_never_calls_reader(configured, field, value):
    config, state, context, saved, path = configured
    saved["request"][field] = value
    path.write_text(json.dumps(saved))
    config = replace(config, runtime={**config.runtime, "rewriter_replay_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(ValidationError, match="mismatch"):
        config.run(state, context=context)
    assert not context.backend.requests


def test_shape_validation_no_asset_io_and_relative_paths(configured, tmp_path):
    config, _, _, _, _ = configured
    value = {"release_id": config.release_id, "parameters": to_jsonable(config.parameters),
             "runtime": {**config.runtime, "checkout": "not-there", "rewriter_replay_path": "rewrite.json"}}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value))
    parsed = RunConfig.from_file(path)
    assert parsed.validate()["valid"]
    assert parsed.runtime["checkout"] == str(tmp_path / "not-there")
    assert parsed.runtime["rewriter_replay_path"] == str(tmp_path / "rewrite.json")


@pytest.mark.parametrize("change", [{"acknowledge_credential_quarantine": False},
    {"checkpoint_manifest_sha256": "bad"}, {"rewriter_replay_sha256": "bad"},
    {"shell": "anything"}, {"reader_prompt_path": ""}])
def test_invalid_runtime_shape(configured, change):
    config, *_ = configured
    with pytest.raises(ValidationError):
        replace(config, runtime={**config.runtime, **change}).validate()


def test_reader_and_state_identity_fail_before_model(configured):
    config, state, context, _, _ = configured
    for params in ({**config.parameters, "reader_checkpoint_revision": "wrong"},
                   {**config.parameters, "corpus_revision": "wrong"}):
        with pytest.raises(ValidationError):
            replace(config, parameters=params).run(state, context=context)
    assert not context.backend.requests


def test_checksum_failure_before_native_load(configured):
    config, state, context, _, _ = configured
    with pytest.raises(ValidationError, match="checksum"):
        replace(config, runtime={**config.runtime, "checkpoint_manifest_sha256": "0"*64}).run(state, context=context)
    assert not context.backend.requests


def test_asset_failure_has_actionable_error_before_reader(configured, monkeypatch):
    config, state, context, _, _ = configured

    def missing(*args):
        raise driver.DriverError("ARTIFACT_INVALID", "artifact must be a regular non-symlink file")

    monkeypatch.setattr(driver, "verify_checkpoint", missing)
    with pytest.raises(ValidationError, match="checkpoint_root must contain"):
        config.run(state, context=context)
    assert not context.backend.requests
