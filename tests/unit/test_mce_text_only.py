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
from agentinputlab.recipes import (
    MCE_TEXT_ARTIFACT_RELEASE,
    MCETextArtifact,
    MCETextArtifactCommonSupport,
)


def _artifact(*, selected_iteration: int = 1) -> MCETextArtifact:
    return MCETextArtifact.build(
        selected_iteration=selected_iteration,
        skill_markdown=(
            "## Skill Overview\nGround decisions in visible evidence." if selected_iteration else ""
        ),
        context_files=(
            {"context/guidance.md": "Prefer precise evidence over generic advice."}
            if selected_iteration
            else {}
        ),
        source_hashes={"mce/main.py": "a" * 64},
        iteration_scores=((0.5, 0.0) if selected_iteration == 0 else (0.0, 0.5)),
        discovery_split_sha256="b" * 64,
        model_fingerprint={"model": "fixture"},
        training_usage={"model_calls": 5, "input_tokens": 20, "output_tokens": 10},
    )


def test_mce_text_artifact_appends_only_visible_text() -> None:
    state = FrozenAgentState(
        query="Original request",
        instruction="Original instruction",
        context=(ContextBlock("Original evidence", block_id="evidence-1"),),
        tool_names=("tool_a",),
        metadata={"gold": "must-stay-hidden"},
    )
    patch = MCETextArtifactCommonSupport(_artifact()).propose(
        state,
        InterventionContext(seed=17),
    )
    output = patch.apply(state)

    assert patch.release_id == MCE_TEXT_ARTIFACT_RELEASE
    assert output.query == state.query
    assert output.instruction == state.instruction
    assert output.tool_names == state.tool_names
    assert output.metadata == state.metadata
    assert output.context[:-1] == state.context
    assert "Ground decisions" in output.context[-1].text
    assert "Prefer precise evidence" in output.context[-1].text
    assert patch.metadata["generated_code_executed"] is False
    assert patch.metadata["paper_result_parity_claimed"] is False
    assert patch.trace.counters["model_calls"] == 0
    assert patch.trace.counters["output_context_blocks"] == 2


def test_mce_baseline_selection_emits_explicit_no_op() -> None:
    state = FrozenAgentState(query="Question")
    patch = MCETextArtifactCommonSupport(_artifact(selected_iteration=0)).propose(
        state,
        InterventionContext(seed=17),
    )
    assert patch.apply(state) == state
    assert patch.operations[0].op.value == "no_op"
    assert patch.trace.counters["output_context_blocks"] == 0


def test_mce_text_artifact_is_deterministic_and_rejects_python() -> None:
    state = FrozenAgentState(query="Question")
    method = MCETextArtifactCommonSupport(_artifact())
    assert (
        method.propose(state, InterventionContext(seed=17)).patch_hash
        == method.propose(
            state,
            InterventionContext(seed=17),
        ).patch_hash
    )

    try:
        MCETextArtifact.build(
            selected_iteration=1,
            skill_markdown="skill",
            context_files={"interfaces/get_context.py": "def get_context(): return ''"},
            source_hashes={"mce/main.py": "a" * 64},
            iteration_scores=(0.0, 0.5),
            discovery_split_sha256="b" * 64,
            model_fingerprint={"model": "fixture"},
            training_usage={"model_calls": 1},
        )
    except ValidationError as exc:
        assert "only allow context" in str(exc)
    else:
        raise AssertionError("generated Python must be rejected")


def test_mce_artifact_enforces_first_best_selection_and_empty_baseline() -> None:
    common = {
        "source_hashes": {"mce/main.py": "a" * 64},
        "discovery_split_sha256": "b" * 64,
        "model_fingerprint": {"model": "fixture"},
        "training_usage": {"model_calls": 1},
    }
    try:
        MCETextArtifact.build(
            selected_iteration=1,
            skill_markdown="skill",
            context_files={},
            iteration_scores=(0.5, 0.1),
            **common,
        )
    except ValidationError as exc:
        assert "first-best" in str(exc)
    else:
        raise AssertionError("a non-best MCE iteration must not be frozen")

    try:
        MCETextArtifact.build(
            selected_iteration=0,
            skill_markdown="unexpected skill",
            context_files={},
            iteration_scores=(0.5, 0.1),
            **common,
        )
    except ValidationError as exc:
        assert "empty artifact" in str(exc)
    else:
        raise AssertionError("baseline selection must not hide a context mutation")


@pytest.mark.parametrize("iteration", [0, 1])
def test_text_artifact_pinned_file_roundtrip(tmp_path, iteration):
    artifact = _artifact(selected_iteration=iteration)
    raw = json.dumps(artifact.as_dict()).encode()
    path = tmp_path / "artifact.json"
    path.write_bytes(raw)
    loaded = MCETextArtifact.from_json(path, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert loaded.as_dict() == artifact.as_dict()
    from agentinputlab import RunConfig
    config = RunConfig(MCE_TEXT_ARTIFACT_RELEASE, artifacts={"artifact": {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}})
    result = config.run("A new question")
    assert result["record"]["usage"]["model_calls"] == 0
    assert bool(result["state"]["context"]) == bool(iteration)
    path.write_bytes(raw + b" ")
    from agentinputlab.contracts.errors import ArtifactError
    with pytest.raises(ArtifactError, match="checksum"):
        config.create_intervention()


@pytest.mark.parametrize("field,value", [
    ("skill_markdown", None), ("context_files", []), ("context_files", {"context/a.py": "code"}),
    ("iteration_scores", "1"), ("iteration_scores", [False, True]),
    ("iteration_scores", [0, float("nan")]), ("training_usage", None),
    ("artifact_id", "changed"), ("release_id", "recipe.mce.final_artifact_eval"),
    ("selected_iteration", True), ("context_files", {"context/../a.md": "unsafe"}),
])
def test_text_artifact_mapping_rejects_malformed_or_changed_values(field, value):
    payload = _artifact().as_dict()
    payload[field] = value
    with pytest.raises(ValidationError):
        MCETextArtifact.from_mapping(payload)


@pytest.mark.parametrize("payload", [{}, [], {"factory": "os.system"}])
def test_text_artifact_mapping_rejects_schema(payload):
    with pytest.raises(ValidationError, match="fields"):
        MCETextArtifact.from_mapping(payload)
