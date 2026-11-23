"""Held-out native inference boundary fixtures, not real neural acceptance."""

import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import (
    SmartSearchInferenceControls,
    SmartSearchNativeTrainingRecipe,
    SmartSearchSamplingControls,
    SmartSearchTrainedLocalModel,
    SmartSearchTrainedPolicyInference,
    SmartSearchTrainingCase,
)
from agentinputlab.recipes import smartsearch_inference as module
from workers.smartsearch.driver import DriverError, ModelTurn

NATIVE_VERIFY = module.verify_smartsearch_sampling_source
NATIVE_SURFACE = module._surface


class FixtureModel:
    def __init__(self, identity):
        self.identity, self.calls = identity, []
        self.failure = self.change_identity = False
        self.answer = "native fixture answer</answer>"
        self.output_tokens = 7
        self.input_tokens = 20

    def fingerprint(self):
        return dict(self.identity)

    def count_prompt_tokens(self, prompt):
        return 20

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.failure:
            raise ConnectionError("retained fixture transport failure")
        if self.change_identity:
            self.identity["serving_process"] = "changed"
        return ModelTurn(self.answer, "stop", "</answer>", self.input_tokens, self.output_tokens)


class FixtureRetriever:
    def fingerprint(self):
        return dict(corpus_sha256="b" * 64, index_sha256="c" * 64)

    def search(self, query, *, top_n):
        return [dict(id="fixture-doc", contents="Fixed visible fixture context", score=1.0)]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    chain, grpo, source = (tmp_path / name for name in ("chain", "grpo", "source"))
    for path in (chain, grpo, source):
        path.mkdir()
    (grpo / "model-manifest.json").write_text(
        json.dumps(
            dict(
                fixture_only=True,
                revision="fixture-trained-revision",
                files={"model.safetensors": "a" * 64},
            )
        )
    )
    (grpo / "fit-plan.json").write_text('{"controls":{"seed":17}}')
    references = dict(checkout=str(source), grpo_root=str(grpo))
    summary = dict(
        learned_grpo_manifest_sha256=module._sha(grpo / "model-manifest.json"),
        receipt_files=dict(grpo=dict(plan_sha256=module._sha(grpo / "fit-plan.json"))),
        artifacts=dict(grpo="fitted-grpo-id"),
        split_identity=dict(
            train={"train-id": module._question_hash("A fitted training question")},
            validation={"development-id": module._question_hash("A development question")},
            test={},
        ),
    )
    chain_record = dict(references=references, chain=summary)
    chain_record["artifact_id"] = content_hash(chain_record)
    (chain / "artifact.json").write_text(json.dumps(chain_record))
    # The separate chain tests verify the actual optimizer/checkpoint hash
    # closure. This fixture replaces only that expensive source-data boundary.
    monkeypatch.setattr(
        module.SmartSearchTrainingChainArtifact,
        "load",
        lambda _: SimpleNamespace(root=chain, artifact_id=chain_record["artifact_id"]),
    )
    monkeypatch.setattr(
        module, "verify_smartsearch_sampling_source", lambda _: (source, dict(fixture_source=True))
    )
    native_module = SimpleNamespace()

    def run(self, env, schemas, question):
        self.execute_tool_calls(env, ["fixture search"])
        response = native_module.requests.post(
            "agentinputlab://model/generate",
            json=dict(
                text="Native fixture prompt: " + question,
                sampling_params=dict(
                    temperature=0.0, max_new_tokens=8192, stop=["</search>", "</answer>"]
                ),
            ),
        ).json()
        # Upstream adds a stop suffix itself; the transport must not duplicate it.
        return response["text"] + response["meta_info"]["finish_reason"]["matched"]

    recall = type(
        "FixtureReCall", (), dict(__init__=lambda *args: None, run=SimpleNamespace(__wrapped__=run))
    )
    monkeypatch.setattr(
        module,
        "_surface",
        lambda *_: (recall, lambda text: text.removesuffix("</answer>"), native_module),
    )
    identity = dict(
        training_chain_artifact_id=chain_record["artifact_id"],
        trained_policy_artifact_id="fitted-grpo-id",
        model_manifest_sha256=summary["learned_grpo_manifest_sha256"],
        model_root=str(grpo / "exported-policy"),
    )
    model, retriever = FixtureModel(identity), FixtureRetriever()
    case = SmartSearchTrainingCase(
        "test-id",
        "A disjoint test question",
        "test",
        {"fixture_only": True, "never_forward_this_reference": "hidden-test-reference"},
    )
    return SimpleNamespace(
        chain=chain,
        grpo=grpo,
        source=source,
        model=model,
        retriever=retriever,
        case=case,
        output=tmp_path / "test-output",
    )


def predictor(setup, **kwargs):
    return SmartSearchTrainedPolicyInference(
        training_chain_root=setup.chain, model=setup.model, retriever=setup.retriever, **kwargs
    )


def test_trained_native_inference_has_no_reference_or_test_reward_parameter(setup):
    native = predictor(setup)
    with pytest.raises(TypeError):
        native.predict(case=setup.case, output_root=setup.output, golden_answers=["forbidden"])
    assert not setup.output.exists() and setup.model.calls == []
    result = native.predict(case=setup.case, output_root=setup.output)
    assert result["split"] == "test" and result["prediction"] == "native fixture answer"
    assert result["scored"] is False and result["benchmark_score"] is None
    assert result["usage"] == dict(
        model_calls=1, retrieval_calls=1, input_tokens=20, output_tokens=7
    )
    assert result["model_responses"] == 1 and result["optimizer_steps"] == 0
    assert result["unknown_output_token_reservations"] == 0
    assert result["training_sampler_used"] is result["test_reference_or_reward_used"] is False
    assert len(result["evidence"]) == len(result["trajectory"]) == 1
    prompt, sampling = setup.model.calls[0]
    assert "hidden-test-reference" not in prompt and "native fixture answer" not in prompt
    assert sampling == dict(
        temperature=0.0, max_new_tokens=8192, stop=("</search>", "</answer>"), seed=17
    )
    assert (setup.output / "transcript.json").exists()
    with pytest.raises(ValidationError, match="create-only"):
        native.predict(case=setup.case, output_root=setup.output)
    assert len(setup.model.calls) == 1


@pytest.mark.parametrize("split", ["train", "validation"])
def test_training_or_development_case_is_not_relabelled_as_test(setup, split):
    with pytest.raises(ValidationError, match="declared test"):
        predictor(setup).predict(
            case=SmartSearchTrainingCase("id", "question", split), output_root=setup.output
        )
    assert not setup.output.exists() and setup.model.calls == []


@pytest.mark.parametrize(
    "identity,question",
    [
        ("train-id", "Otherwise new question"),
        ("development-id", "Otherwise new question"),
        ("new-id", "  a FITTED  training question  "),
        ("new-id", "A development question"),
    ],
)
def test_train_dev_overlap_rejected_before_provider_or_output(setup, identity, question):
    with pytest.raises(ValidationError, match="overlaps"):
        predictor(setup).predict(
            case=SmartSearchTrainingCase(identity, question, "test"), output_root=setup.output
        )
    assert not setup.output.exists() and setup.model.calls == []


@pytest.mark.parametrize(
    "key",
    [
        "training_chain_artifact_id",
        "trained_policy_artifact_id",
        "model_manifest_sha256",
        "model_root",
    ],
)
def test_original_or_other_policy_cannot_impersonate_the_fitted_model(setup, key):
    setup.model.identity[key] = "not-the-trained-chain"
    with pytest.raises(ValidationError, match="verified trained policy"):
        predictor(setup)
    assert setup.model.calls == []


def test_provider_failure_is_retained_without_retry_and_unknown_cost_is_reserved(setup):
    setup.model.failure = True
    with pytest.raises(ConnectionError):
        predictor(setup).predict(case=setup.case, output_root=setup.output)
    failed = json.loads((setup.output / "failed.json").read_text())
    assert len(setup.model.calls) == failed["usage"]["model_calls"] == 1
    assert failed["model_responses"] == 0 and failed["unknown_output_token_reservations"] == 8192
    assert failed["no_automatic_retry"] is True


def test_budget_smaller_than_native_turn_does_not_admit_a_paid_call(setup):
    controls = SmartSearchInferenceControls(max_output_tokens=4096)
    with pytest.raises(DriverError, match="8192 output tokens"):
        predictor(setup, controls=controls).predict(case=setup.case, output_root=setup.output)
    failed = json.loads((setup.output / "failed.json").read_text())
    assert failed["usage"]["model_calls"] == 0 and setup.model.calls == []


@pytest.mark.parametrize("controls", [False, {}, SmartSearchSamplingControls()])
def test_controls_are_typed_not_silently_defaulted(setup, controls):
    with pytest.raises(ValidationError, match="typed inference"):
        predictor(setup, controls=controls)


def test_different_seed_rejected_before_provider(setup):
    with pytest.raises(ValidationError, match="seed differs"):
        predictor(setup, controls=SmartSearchInferenceControls(seed=18))
    assert setup.model.calls == []


@pytest.mark.parametrize("change", ["chain", "manifest", "provider"])
def test_binding_drift_before_dispatch_is_not_a_scientific_attempt(setup, change):
    native = predictor(setup)
    if change == "chain":
        (setup.chain / "artifact.json").write_text("{}")
    elif change == "manifest":
        (setup.grpo / "model-manifest.json").write_text("{}")
    else:
        setup.model.identity["new_process"] = "changed"
    with pytest.raises(ExternalDependencyError, match="binding changed"):
        native.predict(case=setup.case, output_root=setup.output)
    assert not setup.output.exists() and setup.model.calls == []


def test_binding_drift_after_paid_call_keeps_measured_consumption(setup):
    setup.model.change_identity = True
    with pytest.raises(ExternalDependencyError, match="binding changed"):
        predictor(setup).predict(case=setup.case, output_root=setup.output)
    failed = json.loads((setup.output / "failed.json").read_text())
    assert failed["usage"]["output_tokens"] == 7 and failed["model_responses"] == 1
    assert failed["unknown_output_token_reservations"] == 0 and len(setup.model.calls) == 1


def test_response_that_exceeds_native_turn_keeps_real_cost_and_fails(setup):
    setup.model.output_tokens = 8193
    with pytest.raises(BudgetExceededError, match="per-call token cap"):
        predictor(setup).predict(case=setup.case, output_root=setup.output)
    failed = json.loads((setup.output / "failed.json").read_text())
    assert failed["usage"]["output_tokens"] == 8193 and len(setup.model.calls) == 1


def test_native_empty_answer_is_preserved_but_not_scored_or_called_success(setup):
    setup.model.answer = "</answer>"
    result = predictor(setup).predict(case=setup.case, output_root=setup.output)
    assert result["prediction"] == "" and result["native_answer_present"] is False
    assert result["scored"] is False and result["benchmark_score"] is None
    assert result["empty_prediction_is_retained_not_success"] is True
    assert len(setup.model.calls) == 1


def test_prediction_output_does_not_overlap_source_or_trained_inputs(setup):
    with pytest.raises(ValidationError, match="overlaps protected"):
        predictor(setup).predict(case=setup.case, output_root=setup.grpo / "new-output")
    assert setup.model.calls == []


def test_recipe_exposes_the_fitted_inference_binding_not_the_sampling_api(setup):
    native = SmartSearchNativeTrainingRecipe().inference(
        training_chain_root=setup.chain, model=setup.model, retriever=setup.retriever
    )
    assert isinstance(native, SmartSearchTrainedPolicyInference)
    assert content_hash(native.model_identity) == content_hash(setup.model.identity)


def test_identity_drift_during_token_admission_rejects_before_paid_call(setup):
    def count(prompt):
        setup.model.identity["changed_during_tokenization"] = True
        return 20

    setup.model.count_prompt_tokens = count
    with pytest.raises(ExternalDependencyError, match="binding changed"):
        predictor(setup).predict(case=setup.case, output_root=setup.output)
    failed = json.loads((setup.output / "failed.json").read_text())
    assert failed["usage"]["model_calls"] == 0 and setup.model.calls == []


def test_input_admission_mismatch_retains_reported_usage_and_failure(setup):
    setup.model.input_tokens = 21
    with pytest.raises(DriverError, match="tokenizer admission"):
        predictor(setup).predict(case=setup.case, output_root=setup.output)
    failed = json.loads((setup.output / "failed.json").read_text())
    assert failed["usage"]["input_tokens"] == 21 and failed["usage"]["output_tokens"] == 7
    assert failed["unknown_output_token_reservations"] == 0 and len(setup.model.calls) == 1


def test_oversized_response_is_not_silently_truncated_or_retried(setup, monkeypatch):
    monkeypatch.setattr(module, "MAX_TEXT_CHARS", 512)
    setup.model.answer = "x" * 513
    with pytest.raises(ExternalDependencyError, match="text bound"):
        predictor(setup).predict(case=setup.case, output_root=setup.output)
    failed = json.loads((setup.output / "failed.json").read_text())
    response = json.loads((setup.output / "call-0001-response.json").read_text())
    assert response["oversized_text_not_truncated_for_scoring"] is True
    assert failed["usage"]["output_tokens"] == 7 and len(setup.model.calls) == 1


def test_grpo_plan_tampering_after_bind_is_rejected_before_paid_call(setup):
    native = predictor(setup)
    (setup.grpo / "fit-plan.json").write_text('{"controls":{"seed":18}}')
    with pytest.raises(ExternalDependencyError, match="binding changed"):
        native.predict(case=setup.case, output_root=setup.output)
    assert setup.model.calls == [] and not setup.output.exists()


@pytest.mark.parametrize("provider_failure", [False, True])
def test_pinned_native_loop_and_parser_not_only_fixture_algorithm(
    setup, monkeypatch, provider_failure
):
    source = os.environ.get("AGENTINPUTLAB_SMARTSEARCH_TRAINING_CHECKOUT")
    if not source:
        pytest.skip("mount pinned SmartSearch source for native held-out loop fixture")
    source = Path(source).resolve()
    record = json.loads((setup.chain / "artifact.json").read_text())
    record.pop("artifact_id")
    record["references"]["checkout"] = str(source)
    record["artifact_id"] = content_hash(record)
    (setup.chain / "artifact.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        module.SmartSearchTrainingChainArtifact,
        "load",
        lambda _: SimpleNamespace(root=setup.chain, artifact_id=record["artifact_id"]),
    )
    monkeypatch.setattr(module, "verify_smartsearch_sampling_source", NATIVE_VERIFY)
    monkeypatch.setattr(module, "_surface", NATIVE_SURFACE)
    setup.model.identity["training_chain_artifact_id"] = record["artifact_id"]

    def generate(prompt, **kwargs):
        setup.model.calls.append((prompt, kwargs))
        if provider_failure:
            raise ConnectionError("native-loop fixture provider failure")
        if len(setup.model.calls) == 1:
            return ModelTurn(
                "Need evidence</think><search>fixture query", "stop", "</search>", 20, 7
            )
        return ModelTurn(
            r"<think>Done</think><answer>\boxed{native fixture answer}", "stop", "</answer>", 20, 7
        )

    setup.model.generate = generate
    native = predictor(setup)
    if provider_failure:
        with pytest.raises(ConnectionError, match="native-loop fixture"):
            native.predict(case=setup.case, output_root=setup.output)
        # Actual upstream ReCall has a five-attempt outer decorator: bypassed,
        # so a charged failed request is never retried by this API.
        assert len(setup.model.calls) == 1
        failed = json.loads((setup.output / "failed.json").read_text())
        assert failed["unknown_output_token_reservations"] == 8192
    else:
        result = native.predict(case=setup.case, output_root=setup.output)
        assert (
            result["prediction"] == "native fixture answer" and result["usage"]["model_calls"] == 2
        )
        assert result["usage"]["retrieval_calls"] == 1 and len(setup.model.calls) == 2
        assert result["test_reference_or_reward_used"] is result["scored"] is False


@pytest.mark.parametrize("wrong_root", [False, True])
def test_concrete_learned_model_transport_uses_actual_loopback_root_and_seed(
    setup, wrong_root, monkeypatch
):
    requests = []
    root = str(setup.grpo / "exported-policy")
    # This process-local hostile proxy fixture must not receive loopback data.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            assert self.path == "/v1/models"
            assert self.headers.get("Authorization") is None
            self.send_json(
                dict(
                    data=[dict(id="fixture-trained", root=root + ("-wrong" if wrong_root else ""))]
                )
            )

        def do_POST(self):
            assert self.headers.get("Authorization") is None
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/tokenize":
                self.send_json(dict(count=20))
            else:
                assert self.path == "/v1/completions"
                self.send_json(
                    dict(
                        choices=[
                            dict(
                                text="native fixture answer</answer>",
                                finish_reason="stop",
                                stop_reason="</answer>",
                            )
                        ],
                        usage=dict(prompt_tokens=20, completion_tokens=7),
                    )
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        arguments = dict(
            training_chain_root=setup.chain,
            base_url=f"http://127.0.0.1:{server.server_port}",
            served_model_name="fixture-trained",
            timeout_seconds=5,
        )
        if wrong_root:
            with pytest.raises(DriverError, match="served model root differs"):
                SmartSearchTrainedLocalModel(**arguments)
            assert requests == []
        else:
            provider = SmartSearchTrainedLocalModel(**arguments)
            assert provider.fingerprint()["model_root"] == root
            assert provider.count_prompt_tokens("fixture prompt") == 20
            turn = provider.generate(
                "fixture prompt",
                temperature=0.0,
                max_new_tokens=8192,
                stop=("</search>", "</answer>"),
                seed=17,
            )
            assert turn.text == "native fixture answer" and turn.matched == "</answer>"
            assert turn.input_tokens == 20 and turn.output_tokens == 7
            assert requests[-1][1] == dict(
                model="fixture-trained",
                prompt="fixture prompt",
                temperature=0.0,
                max_tokens=8192,
                stop=["</search>", "</answer>"],
                seed=17,
            )
            with pytest.raises(ValidationError, match="native cap"):
                provider.generate(
                    "fixture prompt",
                    temperature=0.0,
                    max_new_tokens=512,
                    stop=("</search>", "</answer>"),
                    seed=17,
                )
            assert sum(path == "/v1/completions" for path, _ in requests) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_fitted_transport_inventory_uses_https_authentication_without_logging_token(tmp_path):
    captured = []

    class Opener:
        def open(self, request, timeout):
            captured.append((request, timeout))
            return io.BytesIO(
                json.dumps(dict(data=[dict(id="trained-fixture", root=str(tmp_path))])).encode()
            )

    transport = module._TrainedSnapshotTransport.__new__(module._TrainedSnapshotTransport)
    transport.base_url = "https://models.fixture.invalid"
    transport.model, transport.model_root = "trained-fixture", tmp_path
    transport.http = SimpleNamespace(api_key="fixture-local-only", timeout=5, opener=Opener())
    transport._verify_served_root()
    request, timeout = captured[0]
    assert timeout == 5 and request.get_header("Authorization") == "Bearer fixture-local-only"
    assert "fixture-local-only" not in request.full_url
    transport.base_url = "http://127.0.0.1:8800"
    with pytest.raises(DriverError, match="only be sent over HTTPS"):
        transport._verify_served_root()
    assert len(captured) == 1
