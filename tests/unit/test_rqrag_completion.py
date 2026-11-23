import copy
import math

import pytest

from agentinputlab.backends import GenerationRequest
from agentinputlab.contracts import CapabilityError, ValidationError
from agentinputlab.recipes import RQRAGCompletionBackend
from agentinputlab.recipes.rqrag_completion import echoed_path_score


def response():
    return {"model": "native", "choices": [{"text": "promptanswer", "finish_reason": "stop",
        "logprobs": {"token_logprobs": [None, -2.0, -1.0],
                     "top_logprobs": [None, {"p": -2.0}, {"a": -1.0, "b": -0.5}]}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


def test_native_full_path_ppl_includes_prompt_not_just_answer():
    text, score = echoed_path_score(response(), "prompt")
    assert text == "answer"
    assert score["perplexity"] == pytest.approx(math.exp(1.5))
    assert score["confidence"] == pytest.approx(math.exp(-0.5))
    assert score["scored_path_tokens"] == 2
    assert score["perplexity_scope"] == "full_path"


@pytest.mark.parametrize("change", [
    lambda d: d["choices"][0].update(text="differentanswer"),
    lambda d: d["choices"][0]["logprobs"].update(token_logprobs=[None, None, -1]),
    lambda d: d["choices"][0]["logprobs"].update(token_logprobs=[None, -2, float("nan")]),
    lambda d: d["choices"][0]["logprobs"].update(token_logprobs=[None, -2, True]),
    lambda d: d["choices"][0]["logprobs"].update(top_logprobs=[None, {"p": -2}, {}]),
    lambda d: d["usage"].update(prompt_tokens=3),
])
def test_incomplete_or_invalid_native_likelihoods_fail_closed(change):
    data = response()
    change(data)
    with pytest.raises((ValueError, TypeError)):
        echoed_path_score(data, "prompt")


def test_raw_backend_one_call_usage_and_security_boundaries():
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return response()

    backend = RQRAGCompletionBackend(model_id="model", base_url="http://localhost:1234/v1",
        deployment_identity={"revision": "immutable"}, transport=transport, api_key="secret-key")
    result = backend.generate(GenerationRequest(prompt="prompt", max_tokens=16, seed=17, stop=("END",)))
    assert result.text == "answer" and result.usage.output_tokens == 1
    assert len(calls) == 1 and calls[0][0].endswith("/completions")
    assert calls[0][2] == {"model": "model", "prompt": "prompt", "temperature": 0.0,
        "max_tokens": 16, "echo": True, "logprobs": 5, "add_special_tokens": False,
        "seed": 17, "stop": ["END"]}
    assert "secret-key" not in str(backend.fingerprint())
    with pytest.raises(CapabilityError):
        backend.generate(GenerationRequest(prompt="prompt", response_format={"type": "json_object"}))
    assert len(calls) == 1


def test_no_retry_and_secret_free_transport_diagnostic():
    calls = []

    def transport(*args):
        calls.append(args)
        raise RuntimeError("secret-key provider private body")

    backend = RQRAGCompletionBackend(model_id="model", base_url="http://localhost:1234/v1",
        deployment_identity={"revision": "immutable"}, transport=transport, api_key="secret-key")
    with pytest.raises(RuntimeError, match="no automatic retry") as error:
        backend.generate(GenerationRequest(prompt="prompt"))
    assert len(calls) == 1 and "secret-key" not in str(error.value)
    assert error.value.__context__ is None


def test_reject_reflected_credential_and_scoring_errors_without_context():
    data = copy.deepcopy(response())
    data["debug"] = "secret-key"
    backend = RQRAGCompletionBackend(model_id="model", base_url="http://localhost:1234/v1",
        deployment_identity={"revision": "immutable"}, transport=lambda *args: data, api_key="secret-key")
    with pytest.raises(RuntimeError, match="configured credential"):
        backend.generate(GenerationRequest(prompt="prompt"))
    data.pop("debug")
    data["choices"][0]["text"] = "invalid"
    with pytest.raises(RuntimeError, match="full-path") as error:
        backend.generate(GenerationRequest(prompt="prompt"))
    assert error.value.__context__ is None


def test_request_body_cannot_override_echo_or_budget():
    with pytest.raises(ValidationError, match="sampling extras"):
        RQRAGCompletionBackend(model_id="model", base_url="http://localhost:1234/v1",
            deployment_identity={"revision": "immutable"}, extra_body={"echo": False})
