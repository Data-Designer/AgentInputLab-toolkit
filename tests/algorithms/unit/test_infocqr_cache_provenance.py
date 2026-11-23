"""Unit fixtures for caller-cache provenance, not real-model acceptance."""
import hashlib
import json

import pytest

from agentinputlab import ExecutionBudget, FrozenAgentState
from agentinputlab.algorithms.reformulate.infocqr import (
    InfoCQRCache,
    InfoCQRMode,
    InfoCQRRecord,
    conversation_view_hash,
)
from agentinputlab.api import ToolkitService, create_intervention
from agentinputlab.contracts.serialization import to_jsonable


@pytest.mark.parametrize(
    ("mode", "release"),
    (
        (InfoCQRMode.RW_ZSL, "rewrite.infocqr.cached_rw_zsl"),
        (InfoCQRMode.RW_FSL, "rewrite.infocqr.cached_rw_fsl"),
        (InfoCQRMode.EDIT, "rewrite.infocqr.cached_edit"),
    ),
)
def test_operator_jsonl_cache_does_not_assert_author_provenance(tmp_path, mode, release):
    state = FrozenAgentState(
        query="fixture source query", task_id="fixture-cache-id",
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096),
    )
    record = InfoCQRRecord(
        record_id=state.task_id, split="fixture-test", mode=mode,
        source_query=state.query, conversation_hash=conversation_view_hash(state),
        rewrite="fixture standalone rewrite", raw_candidate="fixture standalone rewrite",
        metadata={"cache_origin": "operator_capture", "fixture_only": True},
    )
    path = tmp_path / "operator-cache.jsonl"
    data = (json.dumps(to_jsonable(record), sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    cache = InfoCQRCache.from_jsonl(
        path, expected_sha256=hashlib.sha256(data).hexdigest(),
        artifact_id="fixture-operator-capture", dataset_revision="fixture-source-revision",
    )
    service = ToolkitService(factories={
        release: lambda: create_intervention(release, cache=cache, split="fixture-test"),
    })
    result = service.execute({"release_id": release, "state": to_jsonable(state), "seed": 17})
    assert result["state"]["query"] == record.rewrite
    assert result["state"]["budget"] == to_jsonable(state.budget)
    assert result["record"]["patch"]["operations"][0]["rationale"] == "checksum-locked InfoCQR cache replay"
    assert result["record"]["patch"]["metadata"]["record_metadata"] == record.metadata
    assert result["record"]["usage"]["model_calls"] == 0
    assert result["record"]["usage"]["output_tokens"] == 0
    assert result["scored"] is False and result["benchmark_score"] is None
