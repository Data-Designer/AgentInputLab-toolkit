"""Offline interface/security fixtures; NOT real-model GRPO certification."""
import hashlib
import json
import threading
import urllib.error
import urllib.request

import pytest

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.recipes import (
    SMARTSEARCH_GRPO_ENVIRONMENT,
    SmartSearchGRPORetrievalService,
    SmartSearchNativeGRPOReward,
    SmartSearchTeacherCompletion,
    SmartSearchTeacherControls,
    make_smartsearch_grpo_retrieval_server,
)
from workers.smartsearch.driver import DriverError


class Retrieval:
    def __init__(self):
        self.calls = []
        self.failure = False
        self.changed = False

    def fingerprint(self):
        return {"fixture_only": True, "corpus_sha256": "a" * 64,
                "index_sha256": ("c" if self.changed else "b") * 64}

    def search(self, query, *, top_n):
        self.calls.append((query, top_n))
        if self.failure:
            raise ConnectionError("fixture transport failure")
        return [{"id": "fixture", "contents": "Fixture document", "score": 1.0}]


def payload(call="wikipedia_search(query='fixture query', top_n=5)"):
    return {"env": SMARTSEARCH_GRPO_ENVIRONMENT, "call": call}


def test_fixed_search_format_identity_and_repeated_query_costs(tmp_path):
    retrieval = Retrieval()
    service = SmartSearchGRPORetrievalService(retrieval, output_root=tmp_path / "service", max_retrieval_calls=2)
    for _ in range(2):
        assert service.execute(payload()) == {"result": "Fixture document", "output": "", "error": ""}
    assert retrieval.calls == [("fixture query", 5)] * 2
    assert service.assert_healthy()["retrieval_responses"] == 2
    assert json.loads((service.root / "response-000001.json").read_text())["evidence"][0]["document_id"] == "fixture"
    with pytest.raises(BudgetExceededError):
        service.execute(payload())
    with pytest.raises(ExternalDependencyError):
        service.assert_healthy()
    assert len(retrieval.calls) == 2
    with pytest.raises(ValidationError, match="create-only"):
        SmartSearchGRPORetrievalService(retrieval, output_root=service.root, max_retrieval_calls=2)


@pytest.mark.parametrize("call", [
    "__import__('os').system('anything')", "wikipedia_search(query=f(), top_n=5)",
    "wikipedia_search('query', top_n=5)", "wikipedia_search(query='x', top_n=True)",
    "wikipedia_search(query='x', top_n=6)", "wikipedia_search(query='x', query='y')",
    "wikipedia_search(**{'query':'x','top_n':5})", "wikipedia_search(query='x', top_n=5); anything()",
    "wikipedia_search(query='', top_n=5)", "wikipedia_search(query='x', top_n=5, extra=1)",
])
def test_never_executes_generated_code_or_unbounded_search(tmp_path, call):
    retrieval = Retrieval()
    service = SmartSearchGRPORetrievalService(retrieval, output_root=tmp_path / "service", max_retrieval_calls=2)
    with pytest.raises(ValidationError):
        service.execute(payload(call))
    with pytest.raises(ValidationError):
        service.execute({**payload(), "env": "arbitrary Python environment"})
    assert retrieval.calls == [] and service.attempts == 0
    with pytest.raises(ExternalDependencyError):
        service.assert_healthy()


def test_retrieval_failure_latches_without_retry(tmp_path):
    retrieval = Retrieval()
    retrieval.failure = True
    service = SmartSearchGRPORetrievalService(retrieval, output_root=tmp_path / "service", max_retrieval_calls=2)
    with pytest.raises(DriverError, match="retriever request failed"):
        service.execute(payload())
    retrieval.failure = False
    with pytest.raises(ExternalDependencyError):
        service.execute(payload())
    assert len(retrieval.calls) == 1 and service.responses == 0
    assert json.loads((service.root / "failed-000001.json").read_text())["attempt_consumed"]


def test_loopback_native_route_authentication_and_duplicate_json_keys(tmp_path):
    service = SmartSearchGRPORetrievalService(Retrieval(), output_root=tmp_path / "service", max_retrieval_calls=2)
    server = make_smartsearch_grpo_retrieval_server(service, capability="a" * 32)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"

    def post(route, data):
        request = urllib.request.Request(base + route, data=data, headers={"Content-Type": "application/json"})
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=3)

    try:
        with post("a" * 32 + "/execute", json.dumps(payload()).encode()) as response:
            assert response.status == 200
            assert json.load(response)["result"] == "Fixture document"
        for route, data, code in [
            ("wrong/execute", b"{}", 401),
            ("a" * 32 + "/unknown", b"{}", 404),
            ("a" * 32 + "/execute", b'{"env":1,"env":2}', 400),
        ]:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(route, data)
            assert error.value.code == code
        assert service.attempts == 1
        with pytest.raises(ExternalDependencyError):
            service.assert_healthy()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture
def fixture_reward(tmp_path, monkeypatch):
    from agentinputlab.recipes import smartsearch_grpo_tools as module

    source = tmp_path / "source"
    path = source / module._REWARD_PATH
    path.parent.mkdir(parents=True)
    path.write_text("def compute_score_with_format(tokenizer, solution_str, ground_truth):\n"
                    "    return float(generate_response('explicit fixture teacher prompt').split('</')[0])\n")
    monkeypatch.setattr(module, "_REWARD_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "verify_smartsearch_sampling_source", lambda *_: (source, {"fixture_only": True}))
    return source


def reward(source, tmp_path, provider, **kwargs):
    return SmartSearchNativeGRPOReward(provider, checkout=source, references={"Fixture train": ["gold"]},
        model_fingerprint_sha256="a" * 64, controls=SmartSearchTeacherControls(),
        output_root=tmp_path / "reward", **kwargs)


SOLUTION = "<|im_start|>user\nFixture train<|im_end|>fixture response"


def test_reference_key_matches_native_question_whitespace_without_rewriting_solution(tmp_path, fixture_reward):
    scorer = SmartSearchNativeGRPOReward(
        lambda *args, **kwargs: SmartSearchTeacherCompletion("1", 20, 2, "a" * 64),
        checkout=fixture_reward, references={"  Fixture train  ": ["gold"]},
        model_fingerprint_sha256="a" * 64, controls=SmartSearchTeacherControls(), output_root=tmp_path / "spaced")
    raw = SOLUTION.replace("Fixture train", "  Fixture train  ")
    assert scorer(None, raw, ["gold"]) == 1.
    saved = json.loads((scorer.root / "score-000001-request.json").read_text())
    assert saved["solution_str"] == raw
    with pytest.raises(ValidationError, match="collide"):
        SmartSearchNativeGRPOReward(lambda *args, **kwargs: None, checkout=fixture_reward,
            references={"Fixture train": ["gold"], " Fixture train ": ["different"]},
            model_fingerprint_sha256="a" * 64, controls=SmartSearchTeacherControls(), output_root=tmp_path / "collision")
    assert not (tmp_path / "collision").exists()


def test_reward_metering_native_stop_append_and_split_rejection(tmp_path, fixture_reward):
    calls = []

    def provider(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return SmartSearchTeacherCompletion("1", 22, 3, "a" * 64)

    scorer = reward(fixture_reward, tmp_path, provider)
    with pytest.raises(ValidationError):
        scorer(None, SOLUTION.replace("Fixture train", "Fixture test"), ["gold"])
    with pytest.raises(ValidationError):
        scorer(None, SOLUTION, ["different gold"])
    assert not calls
    assert scorer(None, SOLUTION, ["gold"]) == 1.
    assert calls[0][1] == {"temperature": .6, "max_new_tokens": 2048, "stop": ("</answer>",), "seed": 17}
    assert scorer.usage["model_responses"] == scorer.usage["score_responses"] == 1
    assert scorer.usage["input_tokens"] == 22 and scorer.usage["output_tokens"] == 3
    with pytest.raises(ValidationError):
        reward(fixture_reward, tmp_path / "bad", provider, reference_split="test")


@pytest.mark.parametrize("kind", ["transport", "blank", "untyped", "identity", "cap"])
def test_reward_failed_helper_never_becomes_positive_or_retry(tmp_path, fixture_reward, kind):
    calls = []

    def provider(*args, **kwargs):
        calls.append(kwargs)
        if kind == "transport":
            raise ConnectionError("fixture failure")
        if kind == "untyped":
            return "fixture unmetered"
        return SmartSearchTeacherCompletion("" if kind == "blank" else "1", 20,
                                            2049 if kind == "cap" else 4,
                                            ("b" if kind == "identity" else "a") * 64)

    scorer = reward(fixture_reward, tmp_path, provider)
    with pytest.raises((ExternalDependencyError, BudgetExceededError, ConnectionError)):
        scorer(None, SOLUTION, ["gold"])
    with pytest.raises(ExternalDependencyError):
        scorer(None, SOLUTION, ["gold"])
    assert len(calls) == 1
    assert scorer.usage["score_responses"] == 0
    expected = 2048 if kind in {"transport", "untyped"} else 0
    assert scorer.usage["unknown_output_token_reservations"] == expected
    assert (scorer.root / "score-000001-failed.json").exists()
