"""Frozen SkillOpt skill replay for the common-support paper arenas.

The skill is produced offline by the pinned upstream ``scripts/train.py``
pipeline.  This adapter does not re-run or approximate that optimizer.  It
only appends the validated, content-addressed ``best_skill.md`` to the visible
context used by the unchanged downstream agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentinputlab.contracts import (
    ContextBlock,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash

from .skillopt import SkillOptArtifact

SKILLOPT_COMMON_SUPPORT_RELEASE = "recipe.skillopt.eval_only"
SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE = "append_frozen_skill_context.v1"


@dataclass(slots=True)
class SkillOptSkillCommonSupport:
    """Replay one audited SkillOpt artifact through the shared patch seam."""

    artifact: SkillOptArtifact
    method_id: str = "skillopt"
    release_id: str = SKILLOPT_COMMON_SUPPORT_RELEASE
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, SkillOptArtifact):
            raise ValidationError("SkillOpt common support requires a SkillOptArtifact")
        if not self.artifact.best_skill.strip():
            raise ValidationError("SkillOpt common-support skill must not be empty")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        skill_text = _render_skill(self.artifact)
        model_fingerprints_hash = content_hash(self.artifact.model_fingerprints)
        block = ContextBlock(
            text=skill_text,
            block_id=f"skillopt-skill:{self.artifact.artifact_id[:20]}",
            source="SkillOpt frozen best_skill.md",
            metadata={
                "artifact_id": self.artifact.artifact_id,
                "best_skill_sha256": self.artifact.best_skill_sha256,
                "upstream_commit": self.artifact.upstream_commit,
                "placement_rule": SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE,
            },
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "skillopt_frozen_artifact_verified",
                    {
                        "artifact_id": self.artifact.artifact_id,
                        "best_skill_sha256": self.artifact.best_skill_sha256,
                        "upstream_commit": self.artifact.upstream_commit,
                        "model_fingerprints_hash": model_fingerprints_hash,
                        "command_sha256": self.artifact.command_sha256,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "skillopt_skill_appended_to_visible_context",
                    {
                        "placement_rule": SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE,
                        "source_context_blocks": len(state.context),
                        "output_context_blocks": len(state.context) + 1,
                        "skill_characters": len(skill_text),
                    },
                ),
            ),
            counters={
                "model_calls": 0,
                "retrieval_calls": 0,
                "tool_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "source_context_blocks": len(state.context),
                "output_context_blocks": len(state.context) + 1,
                "skill_characters": len(skill_text),
            },
            warnings=(
                "offline_training_cost_is_reported_once_not_charged_per_eval_unit",
                "controlled_common_support_skill_mapping_not_paper_result_parity",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    block,
                    target="context",
                    rationale=(
                        "Expose the frozen upstream SkillOpt best_skill.md without changing "
                        "the captured request or evidence."
                    ),
                ),
            ),
            trace=trace,
            implementation_identity=(
                "pinned_upstream_skillopt_offline+" + SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE
            ),
            profile_kind="controlled_adapter",
            metadata={
                "artifact_id": self.artifact.artifact_id,
                "best_skill_sha256": self.artifact.best_skill_sha256,
                "upstream_commit": self.artifact.upstream_commit,
                "config_sha256": self.artifact.config_sha256,
                "command_sha256": self.artifact.command_sha256,
                "model_fingerprints_hash": model_fingerprints_hash,
                "common_support_placement_rule": SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE,
                "source_state_hash": state.state_hash,
                "paper_result_parity_claimed": False,
            },
        )


def _render_skill(artifact: SkillOptArtifact) -> str:
    return "\n".join(
        (
            "SKILLOPT LEARNED SKILL (frozen offline artifact)",
            "Use these learned strategies when relevant; the user request and visible "
            "evidence remain authoritative.",
            artifact.best_skill.strip(),
        )
    )


__all__ = [
    "SKILLOPT_COMMON_SUPPORT_PLACEMENT_RULE",
    "SKILLOPT_COMMON_SUPPORT_RELEASE",
    "SkillOptSkillCommonSupport",
]
