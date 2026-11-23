"""Saved-JSON fixtures only; never presented as native model acceptance."""

import json
from dataclasses import replace

import pytest
from run_llmlingua2_trained_acceptance import OFFSET, RELEASE, sha, verified_capture

from agentinputlab import ContextBlock, FrozenAgentState
from agentinputlab.contracts.serialization import to_jsonable


def capture(tmp_path):
    original = FrozenAgentState(query="fixture question", context=(ContextBlock(text="original"),))
    context = (ContextBlock(text="actual captured fixture"),)
    materialized = replace(original, context=context)
    job = dict(state_hash=original.state_hash)
    row = dict(
        job=job,
        trained_artifact_id="a" * 64,
        status="native_compressed",
        record=dict(
            patch=dict(
                source_state_hash=original.state_hash,
                release_id=RELEASE,
                profile_kind="controlled_adapter",
                operations=[dict(op="replace_context", value=to_jsonable(context))],
                metadata=dict(
                    trained_artifact_id="a" * 64, model_revision="native-fit-" + "a" * 64
                ),
            )
        ),
        materialized_state_hash=materialized.state_hash,
        materialized_context=to_jsonable(context),
    )
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(row))
    return original, materialized, path, dict(path=str(path), sha256=sha(path), job=job)


def test_exact_source_artifact_and_materialized_context_required(tmp_path):
    original, expected, path, spec = capture(tmp_path)
    row, actual = verified_capture(tmp_path, spec, original, "a" * 64)
    assert actual == expected and row["status"] == "native_compressed" and OFFSET == 224
    with pytest.raises(ValueError, match="source differs"):
        verified_capture(tmp_path, spec, original, "b" * 64)
    with pytest.raises(ValueError, match="source differs"):
        verified_capture(tmp_path, spec, replace(original, query="another"), "a" * 64)
    path.write_text("changed")
    with pytest.raises(ValueError, match="path/hash"):
        verified_capture(tmp_path, spec, original, "a" * 64)


def test_wrong_patch_revision_or_output_hash_not_accepted(tmp_path):
    original, _, path, spec = capture(tmp_path)
    row = json.loads(path.read_text())
    row["record"]["patch"]["metadata"]["model_revision"] = "released-not-trained"
    path.write_text(json.dumps(row))
    spec["sha256"] = sha(path)
    with pytest.raises(ValueError, match="typed"):
        verified_capture(tmp_path, spec, original, "a" * 64)


def test_real_failed_helper_is_preserved_without_noop_fallback(tmp_path):
    original, _, path, spec = capture(tmp_path)
    row = dict(
        job=spec["job"],
        trained_artifact_id="a" * 64,
        status="failed_native_helper",
        error_type="AlgorithmOutputError",
        error="native empty context",
        physical_classifier_forward_attempts=1,
    )
    path.write_text(json.dumps(row))
    spec["sha256"] = sha(path)
    saved, materialized = verified_capture(tmp_path, spec, original, "a" * 64)
    assert materialized is None and saved["physical_classifier_forward_attempts"] == 1
