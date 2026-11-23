"""Synthetic IR feedback fixtures; no neural fitting or benchmark claims."""
from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import pytest

from agentinputlab.backends import Document, InMemoryBM25Retriever, ReplayBackend
from agentinputlab.backends.llm import GenerationResponse, Usage
from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.ir_query import IRQueryOutputError
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import create_skillopt_retrieval_adapter


@pytest.fixture
def task(monkeypatch):
    base = types.ModuleType("skillopt.envs.base")
    base.EnvAdapter = type("EnvAdapter", (), {})
    monkeypatch.setitem(sys.modules, "skillopt.envs.base", base)
    rows = [BenchmarkExample(name, query=query, instruction="Preserve this task instruction.",
        evaluator_gold={"hidden_qrel": "LABEL-NOT-FOR-TARGET"}) for name,query in
        (("train/1", "find alpha"), ("train2", "find beta"), ("val", "find gamma"))]

    class Dataset:
        benchmark_id = "synthetic.IR"
        pairing_mode = "stateless_score_twice"
        def examples(self, split):return tuple(rows)
        def fingerprint(self):return {"fixture":"native-metric"}
        def score(self, example, prediction):
            assert example in rows
            return {"native": 0.75 if "a" in prediction.scores else 0.0}

    corpus = InMemoryBM25Retriever((Document("a", "alpha evidence"), Document("b", "beta")))
    return Dataset(), corpus, rows


class Backend:
    def __init__(self, text):
        self.delegate = ReplayBackend([GenerationResponse(text, usage=Usage(19,8))])
        self.requests = []
        self.model_id, self.capabilities = self.delegate.model_id, self.delegate.capabilities
    def fingerprint(self):return self.delegate.fingerprint()
    def generate(self, request):
        self.requests.append(request)
        return self.delegate.generate(request)


def build(task, backend, **kwargs):
    dataset,corpus,rows = task
    return create_skillopt_retrieval_adapter(adapter=dataset,retriever=corpus,target_backend=backend,
        train=rows[:2],validation=rows[2:],cfg={"seed":17},metric="native",**kwargs)


def test_native_scalar_metric_not_qa_score_and_labels_hidden(task, tmp_path):
    backend = Backend('{"query":"alpha"}')
    env = build(task,backend)
    batch = env.build_train_env(1,17)
    out = env.rollout(batch, "Source skill candidate", tmp_path / "rollout")
    assert out[0]["soft"] == 0.75 and out[0]["hard"] == 1
    assert out[0]["usage"]["model_calls"] == out[0]["usage"]["retrieval_calls"] == 1
    prompt = backend.requests[0].prompt
    assert "LABEL-NOT-FOR-TARGET" not in prompt and "hidden_qrel" not in prompt
    assert "Source skill candidate" in prompt and "Preserve this task instruction." in prompt
    assert backend.requests[0].max_tokens == 512 and backend.requests[0].seed == 17
    saved = json.loads((tmp_path / "rollout/predictions" / batch[0]["id"] / "conversation.json").read_text())
    assert '"score": 0.75' in saved[-1]["content"]
    assert "LABEL-NOT-FOR-TARGET" not in json.dumps(saved)


def test_query_format_failure_is_paid_feedback_zero_not_qa_conversion(task,tmp_path):
    backend = Backend('answer: alpha')
    env = build(task,backend)
    result = env.rollout(env.build_eval_env(1,"valid_seen",17),"skill",tmp_path)[0]
    assert result["hard"] == result["soft"] == 0 and result["native_metrics"] is None
    assert result["response"] == "answer: alpha" and result["query_or_retrieval_failure"]
    assert result["usage"]["model_calls"] == 1 and result["usage"]["retrieval_calls"] == 0
    assert len(backend.requests) == 1


def test_optional_fence_is_explicit_in_environment_fingerprint(task,tmp_path):
    raw = '```json\n{"query":"alpha"}\n```'
    env = build(task,Backend(raw),allow_json_fence=True)
    assert env.fingerprint()["allow_json_fence"]
    assert env.rollout(env.build_eval_env(1,"val",17),"skill",tmp_path)[0]["soft"] == 0.75


@pytest.mark.parametrize("split", ["test","valid_unseen","heldout","train"])
def test_fit_environment_cannot_fetch_heldout_or_alias_test_to_validation(task,split):
    backend=Backend('{"query":"alpha"}')
    with pytest.raises(ValidationError,match="validation only"):
        build(task,backend).build_eval_env(1,split,17)
    assert not backend.requests


def test_disjoint_ids_and_full_visible_inputs_are_checked(task):
    dataset,corpus,rows=task
    with pytest.raises(ValidationError,match="disjoint"):
        create_skillopt_retrieval_adapter(adapter=dataset,retriever=corpus,target_backend=Backend(''),
            train=rows[:1],validation=[replace(rows[0],example_id="different-id")],cfg={},metric="native")


def test_undeclared_or_mutated_fit_cases_fail_before_model(task,tmp_path):
    backend=Backend('{"query":"alpha"}')
    env=build(task,backend)
    batch=env.build_train_env(1,17)
    batch[0]["question"]="changed"
    with pytest.raises(ValidationError,match="changed"):
        env.rollout(batch,"skill",tmp_path)
    assert not backend.requests


def test_no_duplicate_draws_and_no_silent_budget_or_diagnostic_expansion(task,tmp_path):
    backend=Backend('{"query":"alpha"}')
    env=build(task,backend)
    batch=env.build_train_env(1,17)
    with pytest.raises(ValidationError,match="duplicate"):
        env.rollout(batch+batch,"skill",tmp_path)
    with pytest.raises(ValidationError,match="diagnostic"):
        env.rollout(batch,"skill",tmp_path,diagnostic_mode=True)
    with pytest.raises(ValidationError,match="exceeds"):
        env.build_train_env(3,17)
    assert not backend.requests


def test_retriever_execution_failure_is_not_a_format_zero_or_retry(task,tmp_path):
    dataset,corpus,rows=task
    class Broken:
        retriever_id=corpus.retriever_id
        def fingerprint(self):return corpus.fingerprint()
        def retrieve(self,*a,**kw):raise OSError("original corpus unavailable")
    backend=Backend('{"query":"alpha"}')
    env=create_skillopt_retrieval_adapter(adapter=dataset,retriever=Broken(),target_backend=backend,
        train=rows[:2],validation=rows[2:],cfg={},metric="native")
    with pytest.raises(IRQueryOutputError):
        env.rollout(env.build_train_env(1,17),"skill",tmp_path)
    assert len(backend.requests)==1
    failures=list(tmp_path.glob('predictions/*/consumed-execution-error.json'))
    assert len(failures)==1 and json.loads(failures[0].read_text())["usage"]["model_calls"]==1
