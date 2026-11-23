"""RQ-RAG tree output mapped to the shared reformulation seam.

The controlled RQ-RAG implementation produces a terminal answer and selected
retrieval evidence. The paper taxonomy classifies RQ-RAG by its agent-visible
query refinement. This adapter therefore runs the complete controlled tree on
a read-only view of the visible request, keeps only query actions on the
model-selected path, and appends those refinements to the original query. It
never replaces the downstream agent, forwards retrieved evidence or terminal
answers, exposes tool schemas, or exposes scorer metadata to the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentinputlab.contracts import (
    CompositeResult,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
)
from agentinputlab.contracts.serialization import content_hash

from .rqrag import RQ_RAG_UPSTREAM_COMMIT, RQRAGMethodControls, RQRAGMethodTreeSearch

RQ_RAG_COMMON_SUPPORT_RELEASE_ID = "recipe.rqrag.common_support_queries"
RQ_RAG_COMMON_SUPPORT_MAPPING = "append_selected_query_refinements.v1"


@dataclass(slots=True)
class RQRAGCommonSupportQueries:
    """Run the controlled tree and append only selected-path query refinements."""

    controls: RQRAGMethodControls = field(default_factory=RQRAGMethodControls)
    method_id: str = "recipe.rqrag"
    release_id: str = RQ_RAG_COMMON_SUPPORT_RELEASE_ID
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        optimization_state = FrozenAgentState(
            query=state.query,
            task_id=state.task_id,
            instruction=state.instruction,
            context=state.context,
            messages=state.messages,
            tool_names=(),
            metadata={},
            budget=state.budget,
        )
        result = RQRAGMethodTreeSearch(controls=self.controls).run(
            optimization_state,
            context,
        )
        selected_path = tuple(result.metadata["selected_path"])
        refinements = _selected_query_refinements(result, selected_path=selected_path)
        appended_query = _serialize_refinements(refinements)
        selected_document_ids = tuple(result.metadata["selected_document_ids"])
        operation = (
            PatchOperation(
                PatchOp.APPEND_QUERY,
                appended_query,
                target="query",
                rationale=(
                    "Expose only RQ-RAG query actions on the model-selected path while "
                    "preserving the original request and downstream agent."
                ),
            )
            if appended_query
            else PatchOperation(
                PatchOp.NO_OP,
                rationale="RQ-RAG selected the root direct-response branch with no query action.",
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "rqrag_common_support_view",
                    {
                        "source_state_hash": state.state_hash,
                        "optimization_state_hash": optimization_state.state_hash,
                        "tool_schema_count": len(state.tool_names),
                        "tool_schemas_forwarded": False,
                        "metadata_forwarded": False,
                        "upstream_commit": RQ_RAG_UPSTREAM_COMMIT,
                    },
                ),
                *result.trace.events,
                TraceEvent(
                    TracePhase.FINALIZE,
                    "rqrag_selected_queries_appended",
                    {
                        "mapping": RQ_RAG_COMMON_SUPPORT_MAPPING,
                        "operation": operation.op.value,
                        "selected_path": selected_path,
                        "selected_query_refinements": refinements,
                        "selected_refinements_hash": content_hash(refinements),
                        "terminal_answer_forwarded": False,
                        "retrieved_evidence_forwarded": False,
                    },
                ),
            ),
            counters={
                **dict(result.trace.counters),
                "selected_query_refinements": len(refinements),
                "optimizer_tool_schema_count": 0,
            },
            warnings=(
                *result.trace.warnings,
                "controlled_query_only_mapping_not_released_7b_or_paper_result_parity",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=trace,
            implementation_identity=(
                f"controlled_method_tree:chanchimin/RQ-RAG@{RQ_RAG_UPSTREAM_COMMIT}+"
                f"{RQ_RAG_COMMON_SUPPORT_MAPPING}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "common_support_mapping": RQ_RAG_COMMON_SUPPORT_MAPPING,
                "method_tree_release_id": result.release_id,
                "method_tree_implementation_identity": result.implementation_identity,
                "selected_node_id": result.metadata["selected_node_id"],
                "selected_path": selected_path,
                "selected_query_refinements": refinements,
                "selected_refinements_hash": content_hash(refinements),
                "selected_document_ids_internal_only": selected_document_ids,
                "selection_mode": result.metadata["selection_mode"],
                "score_source": result.metadata["score_source"],
                "termination": result.metadata["termination"],
                "comparison_eligible": result.metadata["comparison_eligible"],
                "tree_decode": result.metadata["tree_decode"],
                "oracle": result.metadata["oracle"],
                "tool_schemas_forwarded": False,
                "state_metadata_forwarded": False,
                "terminal_answer_forwarded": False,
                "retrieved_evidence_forwarded": False,
                "paper_checkpoint_parity": False,
                "paper_result_parity_claimed": False,
            },
        )


def _selected_query_refinements(
    result: CompositeResult,
    *,
    selected_path: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    selected = set(selected_path)
    refinements: list[tuple[str, str]] = []
    for event in result.trace.events:
        if event.name != "tree_node":
            continue
        payload = event.payload
        node_id = payload.get("node_id")
        query_type = payload.get("query_type")
        query = payload.get("query")
        if (
            node_id in selected
            and query_type in {"rewrite", "decompose", "disambiguate"}
            and isinstance(query, str)
            and query.strip()
        ):
            refinements.append((str(query_type), query.strip()))
    return tuple(refinements)


def _serialize_refinements(refinements: tuple[tuple[str, str], ...]) -> str:
    if not refinements:
        return ""
    lines = ["RQ-RAG selected query refinement path:"]
    lines.extend(f"- [{kind}] {query}" for kind, query in refinements)
    return "\n".join(lines)


__all__ = [
    "RQ_RAG_COMMON_SUPPORT_MAPPING",
    "RQ_RAG_COMMON_SUPPORT_RELEASE_ID",
    "RQRAGCommonSupportQueries",
]
