"""Safe text-artifact profile for Meta Context Engineering (MCE).

The official MCE loop can evolve executable context functions.  This profile
keeps its pinned ``run_iteration`` orchestration but restricts the deployable
artifact to UTF-8 Markdown produced by a text-only runtime.  Generated Python
is neither accepted nor imported.  The frozen artifact is replayed through the
same visible-context seam used by the other common-support interventions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from agentinputlab.artifacts import load_pinned_json_mapping
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
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

from .mce import MCE_UPSTREAM_COMMIT

MCE_TEXT_ARTIFACT_RELEASE = "recipe.mce.text_artifact_eval"
MCE_TEXT_ARTIFACT_PROTOCOL = "agentinputlab.mce.text-artifact.v1"
MCE_TEXT_ARTIFACT_PLACEMENT_RULE = "append_frozen_text_artifact.v1"


def _text_artifact_identity(
    *,
    selected_iteration: int,
    skill_markdown: str,
    context_files: Mapping[str, str],
    source_hashes: Mapping[str, str],
    iteration_scores: tuple[float, ...],
    discovery_split_sha256: str,
    model_fingerprint: Mapping[str, Any],
    training_usage: Mapping[str, int | float],
) -> str:
    return content_hash(
        {
            "protocol": MCE_TEXT_ARTIFACT_PROTOCOL,
            "release_id": MCE_TEXT_ARTIFACT_RELEASE,
            "upstream_commit": MCE_UPSTREAM_COMMIT,
            "selected_iteration": selected_iteration,
            "skill_markdown": skill_markdown,
            "context_files": context_files,
            "source_hashes": source_hashes,
            "iteration_scores": iteration_scores,
            "discovery_split_sha256": discovery_split_sha256,
            "model_fingerprint": model_fingerprint,
            "training_usage": training_usage,
        }
    )


@dataclass(frozen=True, slots=True)
class MCETextArtifact:
    """Content-addressed, non-executable result of one pinned MCE run."""

    artifact_id: str
    selected_iteration: int
    skill_markdown: str
    context_files: Mapping[str, str]
    source_hashes: Mapping[str, str]
    iteration_scores: tuple[float, ...]
    discovery_split_sha256: str
    model_fingerprint: Mapping[str, Any]
    training_usage: Mapping[str, int | float]
    release_id: str = MCE_TEXT_ARTIFACT_RELEASE
    protocol: str = MCE_TEXT_ARTIFACT_PROTOCOL
    upstream_commit: str = MCE_UPSTREAM_COMMIT

    def __post_init__(self) -> None:
        if self.release_id != MCE_TEXT_ARTIFACT_RELEASE:
            raise ValidationError("MCE text artifact release identity mismatch")
        if self.protocol != MCE_TEXT_ARTIFACT_PROTOCOL:
            raise ValidationError("MCE text artifact protocol mismatch")
        if self.upstream_commit != MCE_UPSTREAM_COMMIT:
            raise ValidationError("MCE text artifact upstream revision mismatch")
        if (
            isinstance(self.selected_iteration, bool)
            or not isinstance(self.selected_iteration, int)
            or self.selected_iteration < 0
        ):
            raise ValidationError("MCE selected iteration must be non-negative")
        scores = tuple(float(value) for value in self.iteration_scores)
        if len(scores) <= self.selected_iteration:
            raise ValidationError("MCE iteration scores omit the selected iteration")
        if not scores or not all(isfinite(value) for value in scores):
            raise ValidationError("MCE iteration scores must be finite")
        expected_iteration = max(range(len(scores)), key=scores.__getitem__)
        if self.selected_iteration != expected_iteration:
            raise ValidationError("MCE selected iteration must follow first-best validation score")
        files: dict[str, str] = {}
        for raw_path, text in self.context_files.items():
            path = PurePosixPath(raw_path)
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 2
                or path.parts[0] != "context"
                or path.suffix.lower() not in {".md", ".txt"}
            ):
                raise ValidationError("MCE text artifacts only allow context/*.md or *.txt")
            if not isinstance(text, str) or not text.strip():
                raise ValidationError("MCE text artifact files must contain non-empty UTF-8 text")
            files[path.as_posix()] = text
        if self.selected_iteration == 0 and (self.skill_markdown.strip() or files):
            raise ValidationError("MCE baseline selection must materialize an empty artifact")
        if self.selected_iteration > 0 and not (self.skill_markdown.strip() or files):
            raise ValidationError("an evolved MCE text artifact must not be empty")
        object.__setattr__(self, "context_files", MappingProxyType(dict(sorted(files.items()))))
        object.__setattr__(self, "source_hashes", freeze_mapping(self.source_hashes))
        object.__setattr__(self, "iteration_scores", scores)
        object.__setattr__(self, "model_fingerprint", freeze_mapping(self.model_fingerprint))
        object.__setattr__(self, "training_usage", freeze_mapping(self.training_usage))
        expected = _text_artifact_identity(
            selected_iteration=self.selected_iteration,
            skill_markdown=self.skill_markdown,
            context_files=self.context_files,
            source_hashes=self.source_hashes,
            iteration_scores=self.iteration_scores,
            discovery_split_sha256=self.discovery_split_sha256,
            model_fingerprint=self.model_fingerprint,
            training_usage=self.training_usage,
        )
        if self.artifact_id != expected:
            raise ValidationError("MCE text artifact identity does not match its payload")

    @classmethod
    def build(
        cls,
        *,
        selected_iteration: int,
        skill_markdown: str,
        context_files: Mapping[str, str],
        source_hashes: Mapping[str, str],
        iteration_scores: tuple[float, ...],
        discovery_split_sha256: str,
        model_fingerprint: Mapping[str, Any],
        training_usage: Mapping[str, int | float],
    ) -> MCETextArtifact:
        artifact_id = _text_artifact_identity(
            selected_iteration=selected_iteration,
            skill_markdown=skill_markdown,
            context_files=context_files,
            source_hashes=source_hashes,
            iteration_scores=iteration_scores,
            discovery_split_sha256=discovery_split_sha256,
            model_fingerprint=model_fingerprint,
            training_usage=training_usage,
        )
        return cls(
            artifact_id=artifact_id,
            selected_iteration=selected_iteration,
            skill_markdown=skill_markdown,
            context_files=context_files,
            source_hashes=source_hashes,
            iteration_scores=iteration_scores,
            discovery_split_sha256=discovery_split_sha256,
            model_fingerprint=model_fingerprint,
            training_usage=training_usage,
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MCETextArtifact:
        """Load the exact non-executable artifact emitted by the offline recipe."""
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise ValidationError("MCE text artifact fields are invalid")
        if (any(not isinstance(value[k], str) for k in (
                "artifact_id", "skill_markdown", "discovery_split_sha256",
                "release_id", "protocol", "upstream_commit"))
                or any(not isinstance(value[k], Mapping) for k in (
                    "context_files", "source_hashes", "model_fingerprint", "training_usage"))
                or not isinstance(value["iteration_scores"], (list, tuple))
                or any(type(v) not in (int, float) for v in value["iteration_scores"])):
            raise ValidationError("MCE text artifact values are malformed")
        try:
            return cls(**dict(value))
        except ValidationError:
            raise
        except (AttributeError, TypeError, ValueError, RecursionError) as exc:
            raise ValidationError("MCE text artifact values are malformed") from exc

    @classmethod
    def from_json(
        cls, path: str | Path, *, expected_sha256: str, max_bytes: int = 4 * 1024 * 1024
    ) -> MCETextArtifact:
        return cls.from_mapping(
            load_pinned_json_mapping(path, expected_sha256=expected_sha256, max_bytes=max_bytes)
        )


@dataclass(slots=True)
class MCETextArtifactCommonSupport:
    """Replay a frozen MCE text artifact without executing generated code."""

    artifact: MCETextArtifact
    method_id: str = "recipe.mce"
    release_id: str = MCE_TEXT_ARTIFACT_RELEASE
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, MCETextArtifact):
            raise ValidationError("MCE common support requires an MCETextArtifact")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        rendered = _render_artifact(self.artifact)
        if not rendered:
            operation = PatchOperation(
                PatchOp.NO_OP,
                rationale="Pinned MCE validation selected its baseline iteration.",
            )
            output_blocks = len(state.context)
        else:
            operation = PatchOperation(
                PatchOp.ADD_CONTEXT,
                ContextBlock(
                    text=rendered,
                    block_id=f"mce-text:{self.artifact.artifact_id[:20]}",
                    source="MCE frozen text-only artifact",
                    metadata={
                        "artifact_id": self.artifact.artifact_id,
                        "selected_iteration": self.artifact.selected_iteration,
                        "placement_rule": MCE_TEXT_ARTIFACT_PLACEMENT_RULE,
                    },
                ),
                target="context",
                rationale="Expose the validation-selected MCE text artifact to the agent.",
            )
            output_blocks = len(state.context) + 1
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "mce_text_artifact_verified",
                    {
                        "artifact_id": self.artifact.artifact_id,
                        "selected_iteration": self.artifact.selected_iteration,
                        "upstream_commit": self.artifact.upstream_commit,
                        "generated_code_executed": False,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "mce_text_artifact_materialized",
                    {
                        "placement_rule": MCE_TEXT_ARTIFACT_PLACEMENT_RULE,
                        "source_context_blocks": len(state.context),
                        "output_context_blocks": output_blocks,
                        "artifact_characters": len(rendered),
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
                "output_context_blocks": output_blocks,
                "artifact_characters": len(rendered),
            },
            warnings=(
                "offline_training_cost_is_reported_once_not_charged_per_eval_unit",
                "controlled_text_artifact_profile_not_official_e2b_or_paper_result_parity",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=trace,
            implementation_identity=(
                "pinned_upstream_mce_run_iteration+text_artifact_no_generated_code.v1"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "artifact_id": self.artifact.artifact_id,
                "selected_iteration": self.artifact.selected_iteration,
                "upstream_commit": self.artifact.upstream_commit,
                "discovery_split_sha256": self.artifact.discovery_split_sha256,
                "common_support_placement_rule": MCE_TEXT_ARTIFACT_PLACEMENT_RULE,
                "generated_code_executed": False,
                "paper_result_parity_claimed": False,
            },
        )


def _render_artifact(artifact: MCETextArtifact) -> str:
    sections: list[str] = []
    if artifact.skill_markdown.strip():
        sections.extend(("MCE EVOLVED CONTEXT-ENGINEERING SKILL", artifact.skill_markdown.strip()))
    for path, text in artifact.context_files.items():
        sections.extend((f"MCE CONTEXT FILE: {path}", text.strip()))
    return "\n\n".join(sections)


__all__ = [
    "MCE_TEXT_ARTIFACT_PLACEMENT_RULE",
    "MCE_TEXT_ARTIFACT_PROTOCOL",
    "MCE_TEXT_ARTIFACT_RELEASE",
    "MCETextArtifact",
    "MCETextArtifactCommonSupport",
]
