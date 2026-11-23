"""Offline sampling contracts with explicit fixture model/retrieval callbacks."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentinputlab.contracts import BudgetExceededError, ValidationError
from agentinputlab.recipes import (
    SmartSearchPolicySampler,
    SmartSearchSamplingControls,
    SmartSearchTrainingCase,
)
from workers.smartsearch.driver import ModelTurn


class FixtureModel:
    def __init__(self, failure=False):
        self.calls = []
        self.failure = failure

    def fingerprint(self):
        return {"fixture_only": True, "model_manifest_sha256": "a" * 64}

    def count_prompt_tokens(self, prompt):
        return 100

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.failure:
            raise ConnectionError("fixture model failure")
        return ModelTurn("fixture answer", "stop", "</answer>", 100, 10)


class FixtureRetrieval:
    def fingerprint(self):
        return {"fixture_only": True, "corpus_sha256": "b" * 64, "index_sha256": "c" * 64}

    def search(self, query, **kwargs):
        return [{"id": "fixture", "contents": "Fixture document", "score": 1.0}]


def run_sample(tmp_path, *, model=None, controls=None, **kwargs):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    module = SimpleNamespace()

    def run(self, env, schemas, question):
        response = module.requests.post("agentinputlab://model/generate", json={
            "text": "native fixture prompt: " + question,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 8192,
                                "stop": ["</search>", "</answer>"]}}).json()
        return response["text"]

    def prefix(self, env, schemas, question, prefix):
        response = module.requests.post("agentinputlab://model/generate", json={
            "text": "native fixture prefix: " + prefix,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 8192,
                                "stop": ["</search>"]}}).json()
        return response["text"]

    base = type("FixtureReCall", (), {"__init__": lambda *args: None,
                                     "run": SimpleNamespace(__wrapped__=run),
                                     "run_prefix": SimpleNamespace(__wrapped__=prefix)})
    surface = (base, lambda text: text, module, lambda _, pred, gold: {"f1": float(pred == gold[0])})
    model = model or FixtureModel()
    arguments = dict(checkout=source, case=SmartSearchTrainingCase("t", "Fixture question", "train"),
                     golden_answers=["fixture answer"], output_root=tmp_path / "out")
    arguments.update(kwargs)
    with patch("agentinputlab.recipes.smartsearch_sampling.verify_smartsearch_sampling_source",
               return_value=(source.resolve(), {"fixture_only": True})), patch(
                   "agentinputlab.recipes.smartsearch_sampling._native_surface", return_value=surface), patch(
                       "agentinputlab.recipes.smartsearch_sampling._SOURCE", {}):
        return SmartSearchPolicySampler(model, FixtureRetrieval(), controls=controls).sample(**arguments), model


def test_native_generation_and_scoring_seams_keep_gold_out_of_prompt(tmp_path):
    result, model = run_sample(tmp_path)
    assert result["model_responses"] == 1 and result["usage"]["model_calls"] == 1
    assert model.calls[0][1] == {"temperature": 0.0, "max_new_tokens": 8192,
                                "stop": ("</search>", "</answer>"), "seed": 17}
    assert "fixture answer" not in model.calls[0][0]
    assert not result["unsafe_sandbox_used"] and not result["neural_training_performed"]
    assert json.loads((tmp_path / "out/rollout.json").read_text())["output"]["metric_score"]["f1"] == 1.0
    with pytest.raises(ValidationError, match="create-only"):
        run_sample(tmp_path)


def test_native_prefix_stop_contract_is_different_and_preserved(tmp_path):
    _, model = run_sample(tmp_path, prefix="<think>fixture prefix", prefix_provenance={"fixture_only": True})
    assert model.calls[0][1]["stop"] == ("</search>",)


def test_provider_failure_not_retried_and_unknown_cost_not_zero(tmp_path):
    model = FixtureModel(failure=True)
    with pytest.raises(ConnectionError):
        run_sample(tmp_path, model=model)
    failed = json.loads((tmp_path / "out/failed.json").read_text())
    assert len(model.calls) == 1
    assert failed["usage"]["model_calls"] == 1 and failed["model_responses"] == 0
    assert failed["unknown_output_token_reservations"] == 8192


def test_too_small_budget_rejects_original_cap_before_call(tmp_path):
    model = FixtureModel()
    with pytest.raises(BudgetExceededError):
        run_sample(tmp_path, model=model, controls=SmartSearchSamplingControls(max_output_tokens=4096))
    assert model.calls == []


def test_test_split_is_not_a_curriculum_sampling_source(tmp_path):
    with pytest.raises(ValidationError, match="train/development"):
        run_sample(tmp_path, case=SmartSearchTrainingCase("t", "Fixture heldout", "test"))
    assert not (tmp_path / "out").exists()
