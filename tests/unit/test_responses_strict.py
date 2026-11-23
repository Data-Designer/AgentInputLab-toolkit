"""The explicit strict route preserves the request and makes one attempt."""

from copy import deepcopy

import pytest

from agentinputlab.backends.llm import GenerationRequest, ModelCapability, OpenAICompatibleBackend
from agentinputlab.contracts.errors import CapabilityError, ValidationError
from agentinputlab.experiments.production_builders import _build_llm_backend, _parse_backend_block

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"credits": {"type": "integer"}}, "required": ["credits"]}
FORMAT = {"type": "json_schema", "json_schema": {"name": "assessment", "strict": True, "schema": SCHEMA}}
CAPABILITIES = frozenset({ModelCapability.CHAT, ModelCapability.JSON_SCHEMA, ModelCapability.SEED})


def backend(transport, **kwargs):
    return OpenAICompatibleBackend(model_id="exact-model", base_url="https://provider.example",
        deployment_identity={"revision": "pinned"}, capabilities=CAPABILITIES, transport=transport,
        structured_output_transport="responses_strict", **kwargs)


def raw(*, text='{"credits":7}', model="exact-model", status="completed"):
    return {"object": "response", "model": model, "status": status,
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 23, "output_tokens": 12, "total_tokens": 35}}


def request(**kwargs):
    return GenerationRequest(prompt="Assess this plan", system="Return the assessment.",
                             max_tokens=512, temperature=0, seed=18, response_format=FORMAT, **kwargs)


def test_strict_route_retains_schema_seed_budget_usage_and_raw():
    seen = []
    data = raw(model="response-alias")

    def transport(url, headers, payload, timeout):
        seen.append((url, deepcopy(payload)))
        return data

    b = backend(transport, response_model_aliases=("response-alias",),
                extra_body={"thinking": {"type": "disabled"}})
    out = b.generate(request())
    assert seen == [("https://provider.example/responses", {
        "model": "exact-model", "input": [
            {"role": "system", "content": "Return the assessment."},
            {"role": "user", "content": "Assess this plan"},
        ], "seed": 18, "temperature": 0, "max_output_tokens": 512, "reasoning": {"effort": "none"},
        "store": False, "stream": False, "text": {"format": {"type": "json_schema", **FORMAT["json_schema"]}},
    })]
    assert out.text == '{"credits":7}' and out.model == "response-alias" and out.finish_reason == "stop"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (23, 12)
    assert out.raw["object"] == "response"
    assert b.fingerprint()["structured_output_transport"] == "responses_strict"


def test_ordinary_chat_is_unchanged_when_strict_route_is_enabled():
    seen = []

    def transport(url, headers, payload, timeout):
        seen.append(url)
        assert "response_format" not in payload
        return {"choices": [{"message": {"content": "done"}}], "model": "exact-model",
                "usage": {"prompt_tokens": 4, "completion_tokens": 2}}

    assert backend(transport).generate(GenerationRequest(prompt="Plain chat")).text == "done"
    assert seen == ["https://provider.example/chat/completions"]


@pytest.mark.parametrize("data", [
    raw(text='{"credits":true}'), raw(text='{"credits":1,"credits":2}'), raw(model="other-model"),
    dict(raw(), usage={"input_tokens": 23}), dict(raw(), status="failed"),
])
def test_invalid_remote_response_is_not_retried(data):
    seen = []

    def transport(url, headers, payload, timeout):
        seen.append(url)
        return data

    with pytest.raises(RuntimeError, match="no automatic retry"):
        backend(transport).generate(request())
    assert seen == ["https://provider.example/responses"]


def test_unsupported_options_fail_before_transport():
    def unexpected(*args):
        pytest.fail("transport must not be called")

    with pytest.raises(CapabilityError, match="no model call"):
        backend(unexpected).generate(request(stop=("stop",)))


def test_token_limit_preserves_partial_text_and_length_terminal():
    data = dict(raw(text='{"credits":', status="incomplete"),
                incomplete_details={"reason": "max_output_tokens"})
    out = backend(lambda *args: data).generate(request())
    assert out.text == '{"credits":' and out.finish_reason == "length"


def test_production_configuration_is_explicit_and_bound():
    spec = {"kind": "openai_compatible", "model_id": "exact-model", "base_url": "https://provider.example",
            "deployment_identity": {"revision": "pinned"}, "capabilities": ["chat", "json_schema", "seed"],
            "structured_output_transport": "responses_strict"}
    parsed = _parse_backend_block(spec)
    assert _build_llm_backend(parsed).fingerprint()["backend"]["structured_output_transport"] == "responses_strict"
    with pytest.raises(ValidationError, match="json_schema"):
        _parse_backend_block(dict(spec, capabilities=["chat"]))
