import copy

import pytest

from agentinputlab import RunConfig
from agentinputlab.backends.llm import GenerationRequest
from agentinputlab.backends.llm.udcs_completion import XML_REMAINDER, UDCSFinalCompletionBackend
from agentinputlab.contracts import ValidationError

BASE = dict(model_id="gpt-oss-20b", base_url="http://127.0.0.1:123/v1",
            deployment_identity={"revision": "fixture"})
RELEASE = "clarify.udcs.repo_native_prompt_route"
TAIL = "assistant<|channel|>final<|message|><think>"


def backend(text="actual thought</think><u_request>0.2</u_request>"):
    calls = []
    def transport(url, headers, payload, timeout):
        calls.append((url, copy.deepcopy(payload)))
        if url.endswith("/tokenize"):
            return dict(tokens=[1, 2], token_strs=["prefix", TAIL])
        return dict(model="gpt-oss-20b", choices=[dict(text=text, finish_reason="stop")],
                    usage=dict(prompt_tokens=37, completion_tokens=29))
    return UDCSFinalCompletionBackend(**BASE, transport=transport), calls


def request(**kwargs):
    return GenerationRequest(prompt="original six-field prompt", system="original system",
        max_tokens=512, temperature=0, seed=17, metadata={"release_id": RELEASE}, **kwargs)


def test_original_controls_one_completion_and_raw_usage_preserved():
    b, calls = backend()
    result = b.generate(request())
    assert result.text.startswith("<think>actual thought</think>")
    assert result.usage.output_tokens == 29
    assert len(calls) == 2
    assert calls[0][0].endswith("/tokenize")
    assert calls[0][1]["messages"][-1] == {"role": "assistant", "content": "<think>"}
    assert calls[1][1] == dict(model="gpt-oss-20b", prompt=[1, 2],
        temperature=0, max_tokens=512, seed=17, skip_special_tokens=False,
        structured_outputs={"regex": XML_REMAINDER})
    assert result.raw["completion"]["choices"][0]["text"].startswith("actual thought")
    assert b.fingerprint()["analysis_promoted_to_answer"] is False


def test_only_terminal_eos_removed_not_content():
    b, _ = backend("real content</think><|return|>")
    assert b.generate(request()).text == "<think>real content</think>"


def test_channel_reentry_is_rejected_without_retry():
    b, calls = backend("<|channel|>analysis<|message|>private")
    with pytest.raises(RuntimeError, match="invalid envelope/channel"):
        b.generate(request())
    assert len(calls) == 2


def test_unknown_usage_is_not_zero():
    b, calls = backend()
    original = b.transport
    def broken(url, headers, payload, timeout):
        value = original(url, headers, payload, timeout)
        value.pop("usage", None)
        return value
    b.transport = broken
    with pytest.raises(RuntimeError, match="invalid envelope/channel"):
        b.generate(request())
    assert len(calls) == 2


def test_other_request_profiles_fail_before_network():
    b, calls = backend()
    with pytest.raises(ValidationError):
        b.generate(GenerationRequest(prompt="other algorithm", metadata={"release_id": "plan.other"}))
    with pytest.raises(ValidationError):
        b.generate(request(stop=("\n",)))
    assert not calls


def test_ordinary_downstream_uses_native_chat_without_XML_constraint():
    calls = []
    def transport(url, headers, payload, timeout):
        calls.append((url, payload))
        return dict(model="gpt-oss-20b", choices=[dict(message={"content": "real downstream"}, finish_reason="stop")],
                    usage=dict(prompt_tokens=10, completion_tokens=3))
    b = UDCSFinalCompletionBackend(**BASE, transport=transport, extra_body={"reasoning_effort": "low"})
    response = b.generate(GenerationRequest(prompt="downstream", max_tokens=512, seed=17))
    assert response.text == "real downstream" and len(calls) == 1
    assert calls[0][0].endswith("/chat/completions")
    assert calls[0][1]["reasoning_effort"] == "low"
    assert "structured_outputs" not in calls[0][1]


def test_config_variant_is_explicit_and_release_restricted():
    assert isinstance(RunConfig(RELEASE, backend={**BASE,
        "kind": "udcs_final_completion"})._backend(), UDCSFinalCompletionBackend)
    with pytest.raises(ValidationError):
        RunConfig("baseline.no_op", backend={**BASE,
            "kind": "udcs_final_completion"}).validate()
    with pytest.raises(ValidationError):
        UDCSFinalCompletionBackend(**{**BASE, "model_id": "qwen3-4b"})


def test_bad_token_template_never_generates():
    calls = []
    def transport(url, headers, payload, timeout):
        calls.append(url)
        return dict(tokens=[1], token_strs=["assistant analysis"])
    b = UDCSFinalCompletionBackend(**BASE, transport=transport)
    with pytest.raises(RuntimeError, match="transport failed"):
        b.generate(request())
    assert len(calls) == 1
