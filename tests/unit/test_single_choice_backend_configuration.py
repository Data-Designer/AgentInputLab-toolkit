"""Explicit n=1 is the existing single-response contract, not a budget override."""

import pytest

from agentinputlab.backends import GenerationRequest, OpenAICompatibleBackend
from agentinputlab.contracts.errors import ValidationError


def test_explicit_single_choice_preserves_request_and_charges_original_usage():
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return {"model": "model", "choices": [{"message": {"content": "answer"},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}

    backend = OpenAICompatibleBackend("model", "http://localhost/v1",
        deployment_identity={"revision": "test"}, extra_body={"n": 1}, transport=transport)
    response = backend.generate(GenerationRequest(prompt="question", max_tokens=300, seed=17))
    assert len(calls) == 1 and calls[0]["n"] == 1 and calls[0]["max_tokens"] == 300
    assert response.text == "answer" and response.usage.output_tokens == 7


@pytest.mark.parametrize("n", [0, 2, -1, True, 1.0, "1", None])
def test_branching_or_ambiguous_choice_count_still_rejected(n):
    with pytest.raises(ValidationError, match="extra_body"):
        OpenAICompatibleBackend("model", "http://localhost/v1",
            deployment_identity={"revision": "test"}, extra_body={"n": n})
