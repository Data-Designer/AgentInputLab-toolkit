from __future__ import annotations

import hashlib
import json
import unittest

import pytest

from agentinputlab.contracts import (
    ContextBlock,
    FrozenAgentState,
    InterventionContext,
    PatchOp,
    ValidationError,
)
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers.ace import (
    ACE_UPSTREAM_COMMIT,
    ACEPlaybook,
    PlaybookBullet,
)
from agentinputlab.recipes import (
    ACE_COMMON_SUPPORT_PLACEMENT_RULE,
    ACE_COMMON_SUPPORT_RELEASE,
    ACEContextArtifact,
    ACEControls,
    ACEPlaybookCommonSupport,
)
from agentinputlab.runner import InterventionRunner


def _artifact(*, empty: bool = False) -> ACEContextArtifact:
    playbook = ACEPlaybook()
    if not empty:
        playbook = ACEPlaybook(
            (
                PlaybookBullet(
                    "ctx-00001",
                    "Prefer explicit evidence over unsupported assumptions.",
                    metadata={"section": "CONTEXT CLUES & INDICATORS"},
                ),
            )
        )
    values = dict(
        release_id="recipe.ace.offline",
        playbook=playbook,
        selected_iteration=0,
        split_audit_hash="b" * 64,
        upstream_commit=ACE_UPSTREAM_COMMIT,
        worker_fingerprint={"worker_id": "ace-fixture"},
        model_fingerprints={
            role: {"provider": "fixture", "model": "m", "revision": "r"}
            for role in ("target", "reflector", "curator")
        },
        controls={"seed": 17},
        usage={"model_calls": 5, "tokens": 4016, "cost_usd": 0.0},
        iteration_hashes=("c" * 64,),
    )
    identity = {k: v for k, v in values.items() if k not in {"usage", "playbook"}}
    identity["playbook_hash"] = playbook.playbook_hash
    return ACEContextArtifact(artifact_id=content_hash(identity), **values)


class ACECommonSupportTests(unittest.TestCase):
    def test_appends_frozen_playbook_and_preserves_captured_state(self) -> None:
        source = FrozenAgentState(
            query="Which action should be taken?",
            instruction="Follow the visible policy.",
            context=(ContextBlock("Visible policy text", block_id="policy"),),
            tool_names=("search",),
        )
        record = InterventionRunner().run(
            ACEPlaybookCommonSupport(_artifact()),
            source,
            InterventionContext(seed=17),
        )
        materialized = record.patch.apply(source)

        self.assertEqual(record.release_id, ACE_COMMON_SUPPORT_RELEASE)
        self.assertEqual(record.patch.operations[0].op, PatchOp.ADD_CONTEXT)
        self.assertEqual(materialized.query, source.query)
        self.assertEqual(materialized.instruction, source.instruction)
        self.assertEqual(materialized.context[0], source.context[0])
        self.assertEqual(materialized.tool_names, source.tool_names)
        self.assertEqual(len(materialized.context), 2)
        self.assertIn("Prefer explicit evidence", materialized.context[-1].text)
        self.assertEqual(
            record.patch.metadata["common_support_placement_rule"],
            ACE_COMMON_SUPPORT_PLACEMENT_RULE,
        )
        self.assertEqual(record.usage.model_calls, 0)
        self.assertEqual(record.patch.trace.counters["offline_training_model_calls"], 5)

    def test_native_empty_selected_playbook_is_explicit_no_op(self) -> None:
        source = FrozenAgentState(query="Question", context=(ContextBlock("Evidence"),))
        record = InterventionRunner().run(
            ACEPlaybookCommonSupport(_artifact(empty=True)), source, InterventionContext(seed=17)
        )
        self.assertEqual([op.op for op in record.patch.operations], [PatchOp.NO_OP])
        self.assertEqual(record.patch.apply(source), source)
        self.assertTrue(record.patch.metadata["native_empty_playbook_no_op"])
        self.assertEqual(record.patch.trace.counters["output_context_blocks"], 1)
        self.assertEqual(record.patch.trace.counters["playbook_characters"], 0)


def test_ace_artifact_json_roundtrip_and_full_file_usage_pin(tmp_path):
    original = _artifact()
    raw = json.dumps(original.as_dict(), ensure_ascii=False).encode()
    source = tmp_path / "artifact.json"
    source.write_bytes(raw)
    loaded = ACEContextArtifact.from_json(source, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert loaded.as_dict() == original.as_dict()
    assert loaded.artifact_id == original.artifact_id
    # Historical ACE IDs omit usage. The trusted whole-file pin protects it.
    changed = original.as_dict()
    changed["usage"]["tokens"] += 1
    assert ACEContextArtifact.from_mapping(changed).artifact_id == original.artifact_id
    source.write_text(json.dumps(changed))
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        ACEContextArtifact.from_json(source, expected_sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("field", ["controls", "model_fingerprints", "worker_fingerprint"])
def test_ace_artifact_detects_mutated_identity(field):
    value = _artifact().as_dict()
    value[field]["unexpected"] = {"model": "changed"} if field == "model_fingerprints" else True
    with pytest.raises(ValidationError, match="content identity"):
        ACEContextArtifact.from_mapping(value)


def test_ace_artifact_freezes_nested_metadata_and_fingerprints():
    value = _artifact().as_dict()
    value["playbook"]["bullets"][0]["metadata"]["nested"] = {"values": ["original"]}
    value["model_fingerprints"]["target"]["nested"] = {"values": ["model"]}
    identity = {k: v for k, v in value.items() if k not in {"usage", "playbook", "artifact_id"}}
    identity["playbook_hash"] = content_hash(value["playbook"])
    value["artifact_id"] = content_hash(identity)
    loaded = ACEContextArtifact.from_mapping(value)
    value["playbook"]["bullets"][0]["metadata"]["nested"]["values"].append("changed")
    value["model_fingerprints"]["target"]["nested"]["values"].append("changed")
    assert loaded.playbook.bullets[0].metadata["nested"]["values"] == ("original",)
    assert loaded.model_fingerprints["target"]["nested"]["values"] == ("model",)
    with pytest.raises(TypeError):
        loaded.playbook.bullets[0].metadata["nested"]["values"] = ()


@pytest.mark.parametrize(
    "field,bad",
    [
        ("iteration_hashes", "not-an-array"),
        ("controls", None),
        ("selected_iteration", True),
        ("playbook", {}),
        ("unknown", 1),
    ],
)
def test_ace_artifact_rejects_malformed_values(field, bad):
    value = _artifact().as_dict()
    value[field] = bad
    with pytest.raises(ValidationError):
        ACEContextArtifact.from_mapping(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rounds": True},
        {"max_tokens": 0},
        {"max_model_calls": 1.5},
        {"seed": True},
        {"seed": -1},
        {"max_cost_usd": float("nan")},
        {"max_cost_usd": float("inf")},
        {"max_cost_usd": True},
        {"max_cost_usd": -1},
    ],
)
def test_ace_controls_reject_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        ACEControls(**kwargs)


if __name__ == "__main__":
    unittest.main()
