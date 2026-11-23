"""Checksum-bound replay of externally supplied InfoCQR rewrites.

No upstream source or cache is redistributed.  A caller supplies a checksum-
verified artifact and the adapter validates task, split, conversation view,
profile, post-processing identity, and leakage policy before applying it.
Byte verification does not establish authorship or the generating model;
retain those identities and historical costs in the source receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
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
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

INFOCQR_UPSTREAM_COMMIT = "2953c461aa0c6817b1b7656dbd5168050227b31d"
_GOLD_KEYS = frozenset({"answer", "gold", "qrels", "reference", "truth_rewrite"})


class InfoCQRMode(str, Enum):
    RW_ZSL = "rw_zsl"
    RW_FSL = "rw_fsl"
    EDIT = "edit"


@dataclass(frozen=True, slots=True)
class InfoCQRRecord:
    record_id: str
    split: str
    mode: InfoCQRMode
    source_query: str
    conversation_hash: str
    rewrite: str
    raw_candidate: str = ""
    postprocess_id: str = "upstream_post_processing"
    fallback_reason: str = ""
    first_turn_truth_rewrite: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip() or not self.split.strip():
            raise ArtifactError("InfoCQR records require record_id and split")
        if not isinstance(self.mode, InfoCQRMode):
            try:
                object.__setattr__(self, "mode", InfoCQRMode(str(self.mode)))
            except ValueError as exc:
                raise ArtifactError("InfoCQR record has an unsupported mode") from exc
        if not self.source_query.strip() or not self.rewrite.strip():
            raise ArtifactError("InfoCQR source_query and rewrite must not be empty")
        if len(self.conversation_hash) != 64:
            raise ArtifactError("InfoCQR conversation_hash must be a SHA-256 hex digest")
        try:
            int(self.conversation_hash, 16)
        except ValueError as exc:
            raise ArtifactError("InfoCQR conversation_hash must be hexadecimal") from exc
        if not self.postprocess_id.strip():
            raise ArtifactError("InfoCQR postprocess_id must not be empty")
        if not isinstance(self.first_turn_truth_rewrite, bool):
            raise ArtifactError("InfoCQR first_turn_truth_rewrite must be boolean")
        lowered = {str(key).lower() for key in self.metadata}
        if lowered & _GOLD_KEYS:
            raise ArtifactError("InfoCQR solver-visible metadata contains a gold field")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> InfoCQRRecord:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ArtifactError("InfoCQR record metadata must be a mapping")
        try:
            return cls(
                record_id=str(value["record_id"]),
                split=str(value["split"]),
                mode=InfoCQRMode(str(value["mode"])),
                source_query=str(value["source_query"]),
                conversation_hash=str(value["conversation_hash"]),
                rewrite=str(value["rewrite"]),
                raw_candidate=str(value.get("raw_candidate", "")),
                postprocess_id=str(value.get("postprocess_id", "upstream_post_processing")),
                fallback_reason=str(value.get("fallback_reason", "")),
                first_turn_truth_rewrite=value.get("first_turn_truth_rewrite", False),
                metadata=dict(metadata),
            )
        except (KeyError, ValueError) as exc:
            raise ArtifactError("InfoCQR record is missing a required field") from exc


class InfoCQRCache:
    """Immutable keyed view of one externally supplied cache artifact."""

    def __init__(
        self,
        records: Sequence[InfoCQRRecord],
        *,
        artifact_id: str,
        artifact_sha256: str,
        dataset_revision: str,
    ) -> None:
        if not artifact_id.strip() or not dataset_revision.strip():
            raise ArtifactError("InfoCQR cache requires artifact and dataset identities")
        if len(artifact_sha256) != 64:
            raise ArtifactError("InfoCQR artifact_sha256 must be a SHA-256 digest")
        try:
            int(artifact_sha256, 16)
        except ValueError as exc:
            raise ArtifactError("InfoCQR artifact_sha256 must be hexadecimal") from exc
        indexed: dict[tuple[str, str, str], InfoCQRRecord] = {}
        for record in records:
            if not isinstance(record, InfoCQRRecord):
                raise ArtifactError("InfoCQR cache entries must be InfoCQRRecord instances")
            key = (record.mode.value, record.split, record.record_id)
            if key in indexed:
                raise ArtifactError(f"duplicate InfoCQR cache key: {key}")
            indexed[key] = record
        if not indexed:
            raise ArtifactError("InfoCQR cache must not be empty")
        self._records = MappingProxyType(dict(indexed))
        self.artifact_id = artifact_id
        self.artifact_sha256 = artifact_sha256.lower()
        self.dataset_revision = dataset_revision

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        artifact_id: str,
        dataset_revision: str,
    ) -> InfoCQRCache:
        raw = Path(path).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256.lower():
            raise ArtifactError("InfoCQR cache checksum mismatch")
        records: list[InfoCQRRecord] = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"invalid InfoCQR JSONL line {line_number}") from exc
            if not isinstance(value, Mapping):
                raise ArtifactError(f"InfoCQR JSONL line {line_number} must be an object")
            records.append(InfoCQRRecord.from_mapping(value))
        return cls(
            records,
            artifact_id=artifact_id,
            artifact_sha256=actual,
            dataset_revision=dataset_revision,
        )

    def get(self, mode: InfoCQRMode, split: str, record_id: str) -> InfoCQRRecord:
        try:
            return self._records[(mode.value, split, record_id)]
        except KeyError as exc:
            raise ArtifactError(
                f"InfoCQR cache has no {mode.value}/{split}/{record_id} record"
            ) from exc


def conversation_view_hash(state: FrozenAgentState) -> str:
    """Hash only the conversation fields consumed by the InfoCQR serializer."""

    return content_hash(
        {
            "messages": state.messages,
            "current_query": state.query,
            "new_context": state.metadata.get("new_context", ""),
        }
    )


@dataclass(slots=True)
class _InfoCQRReplay:
    cache: InfoCQRCache
    split: str
    mode: InfoCQRMode
    release_id: str
    allow_first_turn_truth_rewrite: bool = False
    expected_postprocess_id: str = "upstream_post_processing"
    method_id: str = "infocqr"
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def __post_init__(self) -> None:
        if not isinstance(self.cache, InfoCQRCache):
            raise ValidationError("InfoCQR replay requires an InfoCQRCache")
        if not self.split.strip() or not self.expected_postprocess_id.strip():
            raise ValidationError("InfoCQR split and postprocess identity must not be empty")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        record_id = state.task_id or str(state.metadata.get("record_id", ""))
        if not record_id:
            raise ArtifactError("InfoCQR replay requires task_id or metadata.record_id")
        record = self.cache.get(self.mode, self.split, record_id)
        if record.source_query != state.query:
            raise ArtifactError("InfoCQR cached source query mismatch")
        observed_conversation_hash = conversation_view_hash(state)
        if record.conversation_hash != observed_conversation_hash:
            raise ArtifactError("InfoCQR cached conversation view mismatch")
        if record.postprocess_id != self.expected_postprocess_id:
            raise ArtifactError("InfoCQR post-processing identity mismatch")
        if record.first_turn_truth_rewrite and not self.allow_first_turn_truth_rewrite:
            raise ArtifactError(
                "InfoCQR first-turn Truth_rewrite is gold-derived and disabled by policy"
            )
        if not record.rewrite.strip():
            raise AlgorithmOutputError("InfoCQR cache produced an empty final rewrite")
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "cache_identity",
                    {
                        "artifact_id": self.cache.artifact_id,
                        "artifact_sha256": self.cache.artifact_sha256,
                        "dataset_revision": self.cache.dataset_revision,
                        "mode": self.mode.value,
                        "split": self.split,
                        "conversation_hash": observed_conversation_hash,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "cached_candidate",
                    {
                        "raw_candidate": record.raw_candidate,
                        "postprocess_id": record.postprocess_id,
                        "fallback_reason": record.fallback_reason,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "final_rewrite",
                    {
                        "rewrite": record.rewrite,
                        "first_turn_truth_rewrite": record.first_turn_truth_rewrite,
                    },
                ),
            ),
            counters={"historical_model_calls": 0, "replay_records": 1},
            warnings=(
                ("gold_derived_first_turn_rewrite_explicitly_allowed",)
                if record.first_turn_truth_rewrite
                else ()
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_QUERY,
                    record.rewrite,
                    target="query",
                    rationale="checksum-locked InfoCQR cache replay",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"artifact_replay:smartyfh/InfoCQR@{INFOCQR_UPSTREAM_COMMIT}:"
                f"{self.cache.artifact_sha256}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "compatibility_replay",
                "artifact_id": self.cache.artifact_id,
                "artifact_sha256": self.cache.artifact_sha256,
                "dataset_revision": self.cache.dataset_revision,
                "historical_generation_not_rebilled": True,
                "gold_derived_first_turn": record.first_turn_truth_rewrite,
                "record_metadata": dict(record.metadata),
            },
        )


class InfoCQRZeroShotReplay(_InfoCQRReplay):
    def __init__(
        self,
        cache: InfoCQRCache,
        *,
        split: str,
        allow_first_turn_truth_rewrite: bool = False,
    ) -> None:
        super().__init__(
            cache=cache,
            split=split,
            mode=InfoCQRMode.RW_ZSL,
            release_id="rewrite.infocqr.cached_rw_zsl",
            allow_first_turn_truth_rewrite=allow_first_turn_truth_rewrite,
        )


class InfoCQRFewShotReplay(_InfoCQRReplay):
    def __init__(
        self,
        cache: InfoCQRCache,
        *,
        split: str,
        allow_first_turn_truth_rewrite: bool = False,
    ) -> None:
        super().__init__(
            cache=cache,
            split=split,
            mode=InfoCQRMode.RW_FSL,
            release_id="rewrite.infocqr.cached_rw_fsl",
            allow_first_turn_truth_rewrite=allow_first_turn_truth_rewrite,
        )


class InfoCQREditReplay(_InfoCQRReplay):
    def __init__(
        self,
        cache: InfoCQRCache,
        *,
        split: str,
        allow_first_turn_truth_rewrite: bool = False,
    ) -> None:
        super().__init__(
            cache=cache,
            split=split,
            mode=InfoCQRMode.EDIT,
            release_id="rewrite.infocqr.cached_edit",
            allow_first_turn_truth_rewrite=allow_first_turn_truth_rewrite,
        )


__all__ = [
    "INFOCQR_UPSTREAM_COMMIT",
    "InfoCQRCache",
    "InfoCQREditReplay",
    "InfoCQRFewShotReplay",
    "InfoCQRMode",
    "InfoCQRRecord",
    "InfoCQRZeroShotReplay",
    "conversation_view_hash",
]
