from __future__ import annotations

from agentinputlab.backends.llm.base import GenerationRequest, GenerationResponse, Usage
from agentinputlab.backends.retrieval.base import Document, RetrievalResult
from agentinputlab.contracts import (
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
)
from agentinputlab.recipes import (
    RQ_RAG_COMMON_SUPPORT_RELEASE_ID,
    RQRAGCommonSupportQueries,
    RQRAGMethodControls,
)


class _Backend:
    model_id = "rqrag-common-support-fixture"

    def __init__(self) -> None:
        self.calls: list[GenerationRequest] = []
        self.responses = [
            ("direct answer", 0.2),
            ("refined query", 0.4),
            ("evidence answer", 0.9),
        ]

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls.append(request)
        text, confidence = self.responses.pop(0)
        return GenerationResponse(
            text=text,
            model=self.model_id,
            usage=Usage(input_tokens=10, output_tokens=2),
            raw={"rqrag_score": {"confidence": confidence}},
        )

    def fingerprint(self):
        return {"backend": "fixture", "model_id": self.model_id, "revision": "v1"}


class _Retriever:
    retriever_id = "rqrag-common-support-retriever"

    def retrieve(self, query: str, *, top_k: int = 5):
        assert query == "refined query"
        assert top_k == 2
        return (
            RetrievalResult(
                document=Document(
                    doc_id="doc-1",
                    text="The visible evidence supports the evidence answer.",
                    title="Evidence",
                    corpus_revision="fixture-corpus-v1",
                ),
                score=1.0,
                rank=1,
            ),
        )

    def fingerprint(self):
        return {
            "retriever": "fixture",
            "retriever_id": self.retriever_id,
            "corpus_revision": "fixture-corpus-v1",
            "index_revision": "fixture-index-v1",
        }


def test_rqrag_common_support_preserves_captured_request_and_appends_one_block() -> None:
    backend = _Backend()
    state = FrozenAgentState(
        query="Original user request",
        task_id="arena-task",
        instruction="Follow the visible environment policy.",
        context=(ContextBlock("Original context", block_id="ctx-1"),),
        tool_names=("tool_a", "tool_b"),
        metadata={"gold_answer": "must-not-leak", "reward": 1.0},
        budget=ExecutionBudget(
            max_model_calls=3,
            max_retrieval_calls=1,
            max_output_tokens=32,
            max_context_tokens=128,
        ),
    )
    method = RQRAGCommonSupportQueries(
        controls=RQRAGMethodControls(
            max_depth=1,
            max_nodes=4,
            retrieval_top_k=2,
            max_output_tokens_per_call=8,
            query_actions=("rewrite",),
        )
    )

    patch = method.propose(
        state,
        InterventionContext(backend=backend, retriever=_Retriever(), seed=17),
    )
    output = patch.apply(state)

    assert patch.release_id == RQ_RAG_COMMON_SUPPORT_RELEASE_ID
    assert output.instruction == state.instruction
    assert output.tool_names == state.tool_names
    assert output.messages == state.messages
    assert output.metadata == state.metadata
    assert output.context == state.context
    assert output.query.startswith(state.query)
    assert "[rewrite] refined query" in output.query
    assert patch.metadata["tool_schemas_forwarded"] is False
    assert patch.metadata["state_metadata_forwarded"] is False
    assert patch.metadata["paper_checkpoint_parity"] is False
    assert patch.metadata["paper_result_parity_claimed"] is False
    assert patch.metadata["terminal_answer_forwarded"] is False
    assert patch.metadata["retrieved_evidence_forwarded"] is False
    assert patch.trace.counters["model_calls"] == 3
    assert patch.trace.counters["retrieval_calls"] == 1
    assert patch.trace.counters["optimizer_tool_schema_count"] == 0
    prompts = "\n".join(call.prompt for call in backend.calls)
    assert "must-not-leak" not in prompts
    assert "gold_answer" not in prompts
    assert "tool_a" not in prompts
    assert "tool_b" not in prompts


def test_rqrag_common_support_is_deterministic_for_identical_dependencies() -> None:
    state = FrozenAgentState(
        query="Question",
        budget=ExecutionBudget(
            max_model_calls=3,
            max_retrieval_calls=1,
            max_output_tokens=32,
            max_context_tokens=128,
        ),
    )
    controls = RQRAGMethodControls(
        max_depth=1,
        max_nodes=4,
        retrieval_top_k=2,
        max_output_tokens_per_call=8,
        query_actions=("rewrite",),
    )
    first = RQRAGCommonSupportQueries(controls).propose(
        state,
        InterventionContext(backend=_Backend(), retriever=_Retriever(), seed=17),
    )
    second = RQRAGCommonSupportQueries(controls).propose(
        state,
        InterventionContext(backend=_Backend(), retriever=_Retriever(), seed=17),
    )
    assert first.patch_hash == second.patch_hash
