from __future__ import annotations

import hashlib
import json

import pytest

from agentinputlab.contracts import (
    ContextBlock,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import (
    SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE,
    SKILLOPT_COMMON_SUPPORT_RELEASE,
    SKILLOPT_PROTOCOL,
    SKILLOPT_UPSTREAM_COMMIT,
    SkillOptArtifact,
    SkillOptSkillCommonSupport,
)


def _artifact() -> SkillOptArtifact:
    skill = "- Verify the request against visible evidence before acting."
    digest = hashlib.sha256(skill.encode()).hexdigest()
    values = dict(
        release_id="recipe.skillopt.offline",
        best_skill=skill,
        best_skill_sha256=digest,
        upstream_commit=SKILLOPT_UPSTREAM_COMMIT,
        protocol=SKILLOPT_PROTOCOL,
        config_sha256="a" * 64,
        source_hashes={"source.py": "b" * 64},
        output_hashes={"best_skill.md": digest},
        controls={"seed": 17},
        model_fingerprints={"target": {"model": "fixture"}, "optimizer": {"model": "fixture"}},
        environment_keys=("OPENAI_COMPATIBLE_BASE_URL",),
        command_sha256="c" * 64,
    )
    identity = {k: v for k, v in values.items() if k != "best_skill"}
    return SkillOptArtifact(artifact_id=content_hash(identity), **values)


def test_skillopt_common_support_appends_frozen_skill_only() -> None:
    state = FrozenAgentState(
        query="Original request",
        instruction="Original instruction",
        context=(ContextBlock("Original evidence", block_id="evidence-1"),),
        tool_names=("tool_a",),
        metadata={"gold": "must-stay-hidden"},
    )
    patch = SkillOptSkillCommonSupport(_artifact()).propose(state, InterventionContext(seed=17))
    output = patch.apply(state)

    assert patch.release_id == SKILLOPT_COMMON_SUPPORT_RELEASE
    assert output.query == state.query
    assert output.instruction == state.instruction
    assert output.tool_names == state.tool_names
    assert output.metadata == state.metadata
    assert output.context[:-1] == state.context
    assert "Verify the request" in output.context[-1].text
    assert patch.metadata["common_support_placement_rule"] == (
        SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE
    )
    assert patch.metadata["paper_result_parity_claimed"] is False
    assert patch.trace.counters["model_calls"] == 0
    assert patch.trace.counters["output_context_blocks"] == 2


def test_skillopt_common_support_is_deterministic() -> None:
    state = FrozenAgentState(query="Question")
    method = SkillOptSkillCommonSupport(_artifact())
    first = method.propose(state, InterventionContext(seed=17))
    second = method.propose(state, InterventionContext(seed=17))
    assert first.patch_hash == second.patch_hash


def test_skillopt_artifact_pinned_json_roundtrip(tmp_path):
    original = _artifact()
    source = tmp_path / "artifact.json"
    raw = json.dumps(original.as_dict()).encode()
    source.write_bytes(raw)
    loaded = SkillOptArtifact.from_json(source, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert loaded.as_dict() == original.as_dict()
    assert loaded.artifact_id == original.artifact_id


@pytest.mark.parametrize("field", ["controls", "model_fingerprints", "output_hashes"])
def test_skillopt_artifact_rejects_identity_changes(field):
    value = _artifact().as_dict()
    value[field]["unexpected"] = {"model": "changed"} if field == "model_fingerprints" else True
    with pytest.raises(ValidationError, match="content identity"):
        SkillOptArtifact.from_mapping(value)


def test_skillopt_changed_skill_with_new_text_digest_still_rejects_old_artifact_id():
    value = _artifact().as_dict()
    value["best_skill"] = "Changed instructions"
    value["best_skill_sha256"] = hashlib.sha256(value["best_skill"].encode()).hexdigest()
    value["output_hashes"]["best_skill.md"] = value["best_skill_sha256"]
    with pytest.raises(ValidationError, match="content identity"):
        SkillOptArtifact.from_mapping(value)


def test_skillopt_rejects_inconsistent_output_manifest_even_with_recomputed_id():
    value = _artifact().as_dict()
    value["output_hashes"]["best_skill.md"] = "d" * 64
    value["artifact_id"] = content_hash(
        {k: v for k, v in value.items() if k not in {"artifact_id", "best_skill"}}
    )
    with pytest.raises(ValidationError, match="output manifest"):
        SkillOptArtifact.from_mapping(value)


def test_skillopt_nested_fingerprints_are_detached_and_immutable():
    value = _artifact().as_dict()
    value["model_fingerprints"]["target"]["nested"] = {"models": ["original"]}
    value["artifact_id"] = content_hash(
        {k: v for k, v in value.items() if k not in {"artifact_id", "best_skill"}}
    )
    loaded = SkillOptArtifact.from_mapping(value)
    value["model_fingerprints"]["target"]["nested"]["models"].append("changed")
    assert loaded.model_fingerprints["target"]["nested"]["models"] == ("original",)
    with pytest.raises(TypeError):
        loaded.model_fingerprints["target"]["nested"]["models"] = ()


@pytest.mark.parametrize(
    "field,bad",
    [
        ("best_skill", None),
        ("source_hashes", []),
        ("model_fingerprints", {"target": None}),
        ("environment_keys", "OPENAI_API_KEY"),
        ("controls", {"x": float("nan")}),
        ("unknown", 1),
    ],
)
def test_skillopt_artifact_malformed_values_are_typed(field, bad):
    value = _artifact().as_dict()
    value[field] = bad
    with pytest.raises(ValidationError):
        SkillOptArtifact.from_mapping(value)
