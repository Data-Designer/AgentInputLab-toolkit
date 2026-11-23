from types import SimpleNamespace

import pytest
from run_rqrag_fullpath_acceptance import NativeCalls, composite_qa_prediction

from agentinputlab.contracts import ContextBlock, FrozenAgentState


def test_leaf_reselection_reuses_only_exact_native_payload_and_cannot_use_network():
    saved = [{"url": "http://local/completions", "payload": {"seed": 17},
              "response": {"choices": []}}]
    replay = NativeCalls(saved)
    assert replay.transport(saved[0]["url"], {}, {"seed": 17}, 5) == saved[0]["response"]
    assert replay.calls == [] and replay.index == 1 and not replay.payload_mismatch
    with pytest.raises(ValueError, match="exhausted"):
        replay.transport(saved[0]["url"], {}, {"seed": 17}, 5)
    replay = NativeCalls(saved)
    with pytest.raises(ValueError, match="changed"):
        replay.transport(saved[0]["url"], {}, {"seed": 18}, 5)
    assert replay.payload_mismatch


def test_native_transport_failure_is_preserved_not_retried_on_reselection():
    replay = NativeCalls([{"url": "http://local/completions", "payload": {}, "error": "failed"}])
    with pytest.raises(RuntimeError, match="preserved original"):
        replay.transport("http://local/completions", {}, {}, 5)
    assert replay.calls == [] and replay.index == 1


def test_scoring_maps_only_visible_official_support_ids_not_kg_or_gold():
    state = FrozenAgentState(query="q", context=(ContextBlock("visible sentence", block_id="local-id",
        metadata={"official_support_id": '["Visible title",0]'}),))
    result = SimpleNamespace(final_answer="answer", metadata={"selected_document_ids": ("local-id",)})
    prediction = composite_qa_prediction(result, state)
    assert prediction.support_ids == ('["Visible title",0]',)
    assert prediction.evidence_ids == ()
    assert prediction.evidence_texts == ("visible sentence",)
    result.metadata["selected_document_ids"] = ("unknown",)
    with pytest.raises(ValueError, match="outside"):
        composite_qa_prediction(result, state)
