from __future__ import annotations

from dataclasses import replace

import pytest

from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.backends.llm.base import GenerationResponse, Usage
from workers.adaptive_rag.driver import RetrievedDocument, RuntimeIdentity
from workers.adaptive_rag.runtime import (
    LocalTransformersRouter,
    OpenAICompatibleGenerator,
    ToolkitRetriever,
)


def identity():
    return RuntimeIdentity(
        **{
            name: "a" * 64 if name.endswith("sha256") else "fixed-revision-v1"
            for name in RuntimeIdentity.__dataclass_fields__
            if name != "fixture_runtime"
        }
    )


@pytest.mark.parametrize("value", [0, -1, True, "384", 1.5])
def test_router_rejects_bad_input_limit_before_loading_optional_models(value):
    with pytest.raises(ValueError, match="positive integer"):
        LocalTransformersRouter(snapshot_root=None, identity=None, max_input_tokens=value)


@pytest.mark.parametrize("value", [None, 128, 384])
def test_router_estimation_and_generation_encoding_share_declared_limit(value):
    seen = []
    router = LocalTransformersRouter.__new__(LocalTransformersRouter)
    router.max_input_tokens = value

    class Tensor:
        def numel(self):
            return 17

    def tokenize(question, **kwargs):
        seen.append((question, kwargs))
        return {"input_ids": Tensor()}

    router.tokenizer = tokenize
    assert router.estimate_input_tokens("query") == 17
    router._encode_question("query")
    assert seen[0] == seen[1]
    assert seen[0][1] == dict(
        return_tensors="pt", truncation=True, **({"max_length": value} if value else {})
    )


class Backend:
    def __init__(self, response):
        self.response, self.calls = response, []

    def generate(self, request):
        self.calls.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def arguments():
    return dict(
        operation="answer.single_retrieval",
        question="Original question",
        documents=(RetrievedDocument("d1", "Visible evidence", "Title"),),
        reasoning=("Earlier step",),
        prompt="Locked controlled prompt",
        max_output_tokens=512,
        seed=17,
    )


def test_generator_preserves_seed_cap_final_channel_and_actual_cost():
    backend = Backend(GenerationResponse("Final", usage=Usage(18, 21), raw={"reasoning": "hidden"}))
    adapter = OpenAICompatibleGenerator(
        backend=backend, identity=identity(), input_token_upper_bound=32768
    )
    result = adapter.generate(**arguments())
    assert len(backend.calls) == 1
    request = backend.calls[0]
    assert request.seed == 17 and request.max_tokens == 512 and request.temperature == 0
    assert request.stop == () and request.response_format is None
    assert "hidden" not in request.prompt and result.text == "Final"
    assert result.input_tokens == 18 and result.output_tokens == 21 and result.cost_usd is None
    assert (
        adapter.estimate_input_tokens(
            **{k: v for k, v in arguments().items() if k not in {"seed", "max_output_tokens"}}
        )
        == 32768
    )
    assert adapter.component_manifest()["fixture_runtime"] is False


def test_generator_does_not_retry_errors_or_promote_empty_final():
    backend = Backend(RuntimeError("one failure"))
    adapter = OpenAICompatibleGenerator(
        backend=backend, identity=identity(), input_token_upper_bound=100
    )
    with pytest.raises(RuntimeError, match="one failure"):
        adapter.generate(**arguments())
    assert len(backend.calls) == 1
    backend.response = GenerationResponse("", usage=Usage(11, 512), raw={"reasoning": "not final"})
    result = adapter.generate(**arguments())
    assert result.text == "" and result.output_tokens == 512


def test_generator_marks_reasoning_query_without_changing_locked_prompt():
    values = {**arguments(), "operation": "ircot.reason.0"}
    request = OpenAICompatibleGenerator.request(**values)
    assert request.prompt.startswith("Locked controlled prompt")
    assert "last nonempty line is the retrieval query" in request.prompt
    assert "Give the final answer" not in request.prompt


def test_retriever_keeps_ids_scores_and_strips_all_document_metadata():
    retriever = InMemoryBM25Retriever(
        (Document("d1", "Visible evidence", "Title", metadata={"gold": "never forward"}),)
    )
    adapter = ToolkitRetriever(retriever=retriever, identity=identity())
    result = adapter.retrieve("Visible", top_k=1)
    assert len(result) == 1 and result[0].document_id == "d1"
    assert result[0].title == "Title" and result[0].metadata == {}
    assert result[0].text == "Visible evidence"
    assert adapter.component_manifest()["fixture_runtime"] is False


@pytest.mark.parametrize("bound", [True, False, 0, -1, 1.5])
def test_generator_rejects_invalid_reservation(bound):
    with pytest.raises(ValueError, match="positive integer"):
        OpenAICompatibleGenerator(
            backend=Backend(None), identity=identity(), input_token_upper_bound=bound
        )


def test_production_adapters_reject_fixture_identity():
    fixture = replace(identity(), fixture_runtime=True)
    with pytest.raises(ValueError, match="fixture"):
        OpenAICompatibleGenerator(
            backend=Backend(None), identity=fixture, input_token_upper_bound=1
        )
    with pytest.raises(ValueError, match="fixture"):
        ToolkitRetriever(
            retriever=InMemoryBM25Retriever((Document("d", "text"),)), identity=fixture
        )
