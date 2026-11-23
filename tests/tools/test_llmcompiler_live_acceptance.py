from run_llmcompiler_live_acceptance import VisibleSearch

from agentinputlab.connectors.tools import ToolCall
from agentinputlab.contracts import ContextBlock, FrozenAgentState


def test_search_reads_the_public_retrieval_result_document_instead_of_nonexistent_outer_fields():
    state = FrozenAgentState(query="query", context=(
        ContextBlock(text="native QA text", block_id="doc-1"),))
    search = VisibleSearch(state, "source-version")
    result = search.call(ToolCall("search", {"query": "QA"}, "call-1"))
    assert result.output == "[doc-1] native QA text"
    assert search.selected == ["doc-1"] and len(search.calls) == 1
    assert search.calls[0]["results"][0]["document"]["doc_id"] == "doc-1"
