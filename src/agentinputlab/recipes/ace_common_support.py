"""Frozen ACE playbook replay for the three common-support paper arenas.

ACE learns its playbook offline through the unchanged pinned upstream training
loop. This controlled adapter performs no further optimization at evaluation
time: it appends the frozen, hash-addressed playbook as one visible context
block while preserving the captured request, tools, messages, and evidence.
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

from .ace import ACEContextArtifact

ACE_COMMON_SUPPORT_RELEASE = "recipe.ace.eval_only"
ACE_COMMON_SUPPORT_PLACEMENT_RULE = "append_frozen_playbook_context.v1"


@dataclass(slots=True)
class ACEPlaybookCommonSupport:
    """Replay one leakage-audited ACE artifact through the unified patch seam."""

    artifact: ACEContextArtifact
    method_id: str = "ace"
    release_id: str = ACE_COMMON_SUPPORT_RELEASE
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ACEContextArtifact):
            raise ValidationError("ACE common support requires an ACEContextArtifact")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        playbook_text = _render_playbook(self.artifact)
        applied_characters = len(playbook_text) if self.artifact.playbook.bullets else 0
        playbook_hash = self.artifact.playbook.playbook_hash
        model_fingerprints_hash = content_hash(self.artifact.model_fingerprints)
        worker_fingerprint_hash = content_hash(self.artifact.worker_fingerprint)
        block = ContextBlock(
            text=playbook_text,
            block_id=f"ace-playbook:{self.artifact.artifact_id[:20]}",
            source="ACE frozen offline playbook",
            metadata={
                "artifact_id": self.artifact.artifact_id,
                "playbook_hash": playbook_hash,
                "split_audit_hash": self.artifact.split_audit_hash,
                "upstream_commit": self.artifact.upstream_commit,
                "placement_rule": ACE_COMMON_SUPPORT_PLACEMENT_RULE,
            },
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "ace_frozen_artifact_verified",
                    {
                        "artifact_id": self.artifact.artifact_id,
                        "playbook_hash": playbook_hash,
                        "split_audit_hash": self.artifact.split_audit_hash,
                        "upstream_commit": self.artifact.upstream_commit,
                        "selected_iteration": self.artifact.selected_iteration,
                        "model_fingerprints_hash": model_fingerprints_hash,
                        "worker_fingerprint_hash": worker_fingerprint_hash,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "ace_playbook_appended_to_visible_context"
                    if self.artifact.playbook.bullets
                    else "ace_native_empty_playbook_no_op",
                    {
                        "placement_rule": ACE_COMMON_SUPPORT_PLACEMENT_RULE,
                        "source_context_blocks": len(state.context),
                        "output_context_blocks": len(state.context)
                        + bool(self.artifact.playbook.bullets),
                        "playbook_bullets": len(self.artifact.playbook.bullets),
                        "playbook_characters": applied_characters,
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
                "output_context_blocks": len(state.context) + bool(self.artifact.playbook.bullets),
                "playbook_bullets": len(self.artifact.playbook.bullets),
                "playbook_characters": applied_characters,
                "offline_training_model_calls": int(self.artifact.usage.get("model_calls", 0)),
                "offline_training_tokens": int(self.artifact.usage.get("tokens", 0)),
            },
            warnings=(
                "offline_training_cost_is_reported_on_the_artifact_not_charged_per_eval_unit",
                "controlled_common_support_playbook_mapping_not_paper_result_parity",
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
                    rationale="Expose the frozen ACE playbook without changing captured evidence.",
                ),
            )
            if self.artifact.playbook.bullets
            else (
                PatchOperation(PatchOp.NO_OP, rationale="Native ACE selected an empty playbook."),
            ),
            trace=trace,
            implementation_identity=(
                "pinned_upstream_ace_offline+" + ACE_COMMON_SUPPORT_PLACEMENT_RULE
            ),
            profile_kind="controlled_adapter",
            metadata={
                "artifact_id": self.artifact.artifact_id,
                "playbook_hash": playbook_hash,
                "split_audit_hash": self.artifact.split_audit_hash,
                "upstream_commit": self.artifact.upstream_commit,
                "selected_iteration": self.artifact.selected_iteration,
                "model_fingerprints_hash": model_fingerprints_hash,
                "worker_fingerprint_hash": worker_fingerprint_hash,
                "offline_training_usage": dict(self.artifact.usage),
                "common_support_placement_rule": ACE_COMMON_SUPPORT_PLACEMENT_RULE,
                "source_state_hash": state.state_hash,
                "paper_result_parity_claimed": False,
                "native_empty_playbook_no_op": not self.artifact.playbook.bullets,
            },
        )


def _render_playbook(artifact: ACEContextArtifact) -> str:
    lines = [
        "ACE LEARNED PLAYBOOK (frozen offline artifact)",
        "Use these learned strategies when relevant; the user request and visible evidence "
        "remain authoritative.",
    ]
    for bullet in artifact.playbook.bullets:
        lines.append(f"- [{bullet.bullet_id}] {bullet.content}")
    return "\n".join(lines)


__all__ = [
    "ACE_COMMON_SUPPORT_PLACEMENT_RULE",
    "ACE_COMMON_SUPPORT_RELEASE",
    "ACEPlaybookCommonSupport",
]
