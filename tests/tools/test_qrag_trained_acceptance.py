"""Saved JSON fixtures; no real Q-RAG model or model service calls."""

import json
from dataclasses import replace

import pytest
from run_qrag_trained_acceptance import OFFSET, RELEASE, sha, verified_capture

from agentinputlab import ContextBlock, FrozenAgentState
from agentinputlab.contracts.serialization import to_jsonable


def capture(tmp_path):
    original = FrozenAgentState(query="fixture question", context=(ContextBlock(text="original"),))
    added = (ContextBlock(text="captured selected document"),)
    materialized = replace(original, context=original.context + added)
    job = dict(state_hash=original.state_hash)
    row = dict(
        job=job,
        trained_artifact_id="a" * 64,
        deployment_id="b" * 64,
        status="native_selected",
        record=dict(
            patch=dict(
                source_state_hash=original.state_hash,
                release_id=RELEASE,
                profile_kind="controlled_adapter",
                operations=[dict(op="add_context", value=to_jsonable(added))],
                metadata=dict(trained_artifact_id="a" * 64, deployment_id="b" * 64),
            )
        ),
        materialized_state_hash=materialized.state_hash,
        materialized_context=to_jsonable(materialized.context),
    )
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(row))
    return original, materialized, path, dict(path=str(path), sha256=sha(path), job=job)


def test_source_identity_and_add_context_materialization(tmp_path):
    original, expected, path, spec = capture(tmp_path)
    row, actual = verified_capture(tmp_path, spec, original, "a" * 64)
    assert actual == expected and row["status"] == "native_selected" and OFFSET == 226
    with pytest.raises(ValueError, match="source differs"):
        verified_capture(tmp_path, spec, original, "b" * 64)
    with pytest.raises(ValueError, match="source differs"):
        verified_capture(tmp_path, spec, replace(original, query="changed"), "a" * 64)
    path.write_text("changed")
    with pytest.raises(ValueError, match="path/hash"):
        verified_capture(tmp_path, spec, original, "a" * 64)


def test_frozen_alias_or_changed_materialization_rejected(tmp_path):
    original, _, path, spec = capture(tmp_path)
    row = json.loads(path.read_text())
    row["record"]["patch"]["metadata"]["deployment_id"] = "frozen-alias"
    path.write_text(json.dumps(row))
    spec["sha256"] = sha(path)
    with pytest.raises(ValueError, match="typed"):
        verified_capture(tmp_path, spec, original, "a" * 64)


def test_failed_helper_keeps_failure_without_noop_fallback(tmp_path):
    original, _, path, spec = capture(tmp_path)
    row = dict(
        job=spec["job"],
        trained_artifact_id="a" * 64,
        status="failed_helper_terminal",
        error_type="AlgorithmOutputError",
        error="native failure",
        physical_embedding_forward_attempts=1,
    )
    path.write_text(json.dumps(row))
    spec["sha256"] = sha(path)
    saved, materialized = verified_capture(tmp_path, spec, original, "a" * 64)
    assert materialized is None and saved["physical_embedding_forward_attempts"] == 1
