"""Explicit synthetic native-loop fixtures; not real-model acceptance evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agentinputlab.algorithms.acquire.ircot import IRCoTControlledWorker, IRCoTControls
from agentinputlab.backends import Document, GenerationResponse, InMemoryBM25Retriever, Usage
from agentinputlab.contracts import FrozenAgentState, InterventionContext
from tests.workers.test_ircot_worker_driver import FakeCheckout, _request, _sha256
from workers.ircot.controlled import PinnedIRCoTControlledRuntime


class Backend:
    def __init__(self, texts, *, fail=False):
        self.texts, self.fail, self.calls = texts, fail, []

    def fingerprint(self):
        return {"backend": "synthetic-test-only"}

    def generate(self, request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("synthetic transport failure; never retry")
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return GenerationResponse(text=text, usage=Usage(input_tokens=7, output_tokens=2))


class Tokenizer:
    def tokenize(self, text):
        return text.split()


def fixture_runtime(tmp_path, monkeypatch, texts, *, fail=False, max_calls=8):
    checkout = FakeCheckout(tmp_path)
    manifest = json.loads(checkout.manifest_path.read_text())
    profile = manifest["profiles"]["fixture.tiny.frozen.v1"]
    config_path = checkout.manifest_path.parent / "fixture_config.json"
    config = json.loads(config_path.read_text())
    config["models"]["step_by_step_cot_reasoning_gen"]["prompt_file"] = "native_prompt.txt"
    config_path.write_text(json.dumps(config))
    native_prompt = checkout.root / "native_prompt.txt"
    native_prompt.write_text((tmp_path / "fixture_prompt.txt").read_text())
    profile.update(
        prompt_scope="upstream",
        prompt_path=native_prompt.name,
        prompt_sha256=_sha256(native_prompt.read_bytes()),
        config_sha256=_sha256(config_path.read_bytes()),
    )
    module_path = checkout.root / "commaqa/inference/configurable_inference.py"
    source = module_path.read_text().replace(
        'ircot.safe_post_request("http://fixture.invalid/retrieve", {"query_text": query})',
        'ircot.safe_post_request("http://127.0.0.1:0/retrieve", {"query_text": query, '
        '"retrieval_method": "retrieve_from_elasticsearch", "max_hits_count": model_config["retrieval_count"], '
        '"corpus_name": model_config["source_corpus_name"], "document_type": "title_paragraph_text"})',
    )
    module_path.write_text(source)
    manifest["source_files"]["commaqa/inference/configurable_inference.py"] = _sha256(
        module_path.read_bytes()
    )
    prompt_reader = checkout.root / "commaqa/inference/prompt_reader.py"
    prompt_reader.write_text(
        "def fit_prompt_into_given_limit(original_prompt, **kwargs):\n"
        '    assert get_tokenizer("gpt2").tokenize(original_prompt)\n'
        "    return original_prompt\n"
    )
    manifest["source_files"]["commaqa/inference/prompt_reader.py"] = _sha256(
        prompt_reader.read_bytes()
    )
    checkout.manifest_path.write_text(json.dumps(manifest))
    home = tmp_path / "private-worker-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    backend = Backend(texts, fail=fail)
    retriever = InMemoryBM25Retriever(
        [
            Document(
                "d1",
                "Fixture entity has a relation.",
                title="Entity",
                corpus_revision="fixture-corpus-v1",
            ),
            Document(
                "d2",
                "A second fixture relation.",
                title="Other",
                corpus_revision="fixture-corpus-v1",
            ),
        ]
    )
    runtime = PinnedIRCoTControlledRuntime(
        source_root=checkout.root,
        backend=backend,
        retriever=retriever,
        tokenizer=Tokenizer(),
        tokenizer_fingerprint={"fixture": "tokenizer"},
        dependency_fingerprint={"fixture": "dependencies"},
        corpus_revision="fixture-corpus-v1",
        index_revision="fixture-index-v1",
        cache_home=home,
        max_model_calls=max_calls,
        per_call_output_tokens=16,
    )
    actual = checkout.driver()._checkout()
    monkeypatch.setattr(
        runtime,
        "_checkout",
        lambda: replace(
            actual,
            profile_id="synthetic-controlled-loop-test-only",
            profile={**actual.profile, "identity_kind": "controlled_model_transfer"},
        ),
    )
    request = _request(
        checkout.prompt_hash,
        generator_fingerprint=backend.fingerprint(),
        retriever_fingerprint=retriever.fingerprint(),
    )
    return runtime, backend, request


def test_complete_fixture_loop_records_real_observed_seam_usage(tmp_path, monkeypatch):
    runtime, backend, request = fixture_runtime(
        tmp_path, monkeypatch, ["A relation exists.", "The answer is fixture-answer."]
    )
    result = runtime.call(request)
    assert result["status"] == "ok", result
    assert result["worker_identity_kind"] == "controlled_model_transfer"
    assert result["usage"]["model_calls"] == len(backend.calls) == 2
    assert result["usage"]["input_tokens"] == 14 and result["usage"]["output_tokens"] == 4
    assert result["usage"]["retrieval_calls"] == 2
    assert result["unknown_token_usage_model_calls"] == 0
    assert result["usage"]["unpriced_model_calls"] == 2 and "cost_usd" not in result["usage"]
    assert [c.seed for c in backend.calls] == [17, 18]
    assert all(c.stop == ("\n",) and c.max_tokens <= 16 for c in backend.calls)
    assert len(result["native_controller_events"]) == 6


def test_binding_drift_rejected_before_model_or_retrieval(tmp_path, monkeypatch):
    runtime, backend, request = fixture_runtime(tmp_path, monkeypatch, ["not used"])
    request["generator_fingerprint"] = {"wrong": "reader"}
    result = runtime.call(request)
    assert result["failure_code"] == "BINDING_MISMATCH" and result["consumed"] is False
    assert backend.calls == [] and runtime.last_events == []


def test_budget_rejected_before_source_execution(tmp_path, monkeypatch):
    runtime, backend, request = fixture_runtime(tmp_path, monkeypatch, ["not used"], max_calls=1)
    result = runtime.call(request)
    assert result["failure_code"] == "CONTROL_BUDGET_EXCEEDED" and result["consumed"] is False
    assert backend.calls == []


def test_failed_transport_retains_unknown_attempt_and_retrieval(tmp_path, monkeypatch):
    runtime, backend, request = fixture_runtime(tmp_path, monkeypatch, ["not used"], fail=True)
    result = runtime.call(request)
    assert result["status"] == "error" and "consumed" not in result
    assert len(backend.calls) == result["usage"]["model_calls"] == 1
    assert result["unknown_token_usage_model_calls"] == 1
    assert result["usage_is_complete"] is False
    assert result["reported_tokens_are_observed_lower_bounds"] is True
    assert result["usage"]["retrieval_calls"] == 1
    assert "cost_usd" not in result["usage"]


def test_no_retry_or_budget_growth_when_fixture_does_not_answer(tmp_path, monkeypatch):
    runtime, backend, request = fixture_runtime(
        tmp_path, monkeypatch, ["A relation exists."], max_calls=2
    )
    result = runtime.call(request)
    assert result["status"] == "error" and result["failure_code"] == "MODEL_BUDGET_EXCEEDED"
    assert len(backend.calls) == result["usage"]["model_calls"] == 2
    assert result["unknown_token_usage_model_calls"] == 0


def test_request_cannot_inject_configuration(tmp_path, monkeypatch):
    runtime, backend, request = fixture_runtime(tmp_path, monkeypatch, ["not used"])
    result = runtime.call({**request, "endpoint": "https://untrusted.invalid"})
    assert result["failure_code"] == "INVALID_REQUEST" and result["consumed"] is False
    assert backend.calls == []


def test_existing_public_bridge_validates_controlled_loop_response(tmp_path, monkeypatch):
    runtime, backend, request = fixture_runtime(
        tmp_path, monkeypatch, ["A relation exists.", "The answer is fixture-answer."]
    )
    method = IRCoTControlledWorker(
        client=runtime,
        corpus_revision=runtime.corpus_revision,
        index_revision=runtime.index_revision,
        prompt_hash=request["prompt_hash"],
        controls=IRCoTControls(**request["controls"]),
    )
    state = FrozenAgentState(query=request["question"], task_id=request["task_id"])
    patch = method.propose(state, InterventionContext(backend=backend, retriever=runtime.retriever))
    assert patch.release_id == "acquire.ircot.controlled"
    assert patch.trace.counters["model_calls"] == 2
    assert patch.trace.counters["input_tokens"] == 14
    assert len(patch.apply(state).context) >= 1
    assert patch.metadata["worker_fingerprint"]["identity_kind"] == "controlled_model_transfer"


def test_startup_requires_own_home_and_nonsecret_fingerprints(tmp_path, monkeypatch):
    runtime, backend, _ = fixture_runtime(tmp_path, monkeypatch, ["not used"])
    arguments = dict(
        source_root=runtime.source_root,
        backend=backend,
        retriever=runtime.retriever,
        tokenizer=Tokenizer(),
        tokenizer_fingerprint={"fixture": "tokenizer"},
        dependency_fingerprint={"fixture": "deps"},
        corpus_revision="fixture-corpus-v1",
        index_revision="fixture-index-v1",
        cache_home=runtime.cache_home,
    )
    with pytest.raises(ValueError, match="seed"):
        PinnedIRCoTControlledRuntime(**arguments, seed=True)
    arguments["dependency_fingerprint"] = {"api_key": "must-not-be-copied"}
    with pytest.raises(Exception, match="must not contain secrets"):
        PinnedIRCoTControlledRuntime(**arguments)
    arguments["dependency_fingerprint"] = {"fixture": "deps"}
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ValueError, match="dedicated worker"):
        PinnedIRCoTControlledRuntime(**arguments)
