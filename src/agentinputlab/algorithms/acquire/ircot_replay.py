"""Checksum-bound replay of original IRCoT ID-to-answer prediction files.

The archive remains operator-owned. Loading verifies bytes and declared input/
model bindings, not authorship or published-score parity. No prediction is
passed off as corpus evidence and no model request is made during replay.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    CompositeResult,
    ExecutionTrace,
    FrozenAgentState,
    InterventionContext,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

IRCOT_REPLAY_RELEASE = "acquire.ircot.prediction_replay"
IRCOT_REPLAY_SCHEMA = "agentinputlab.ircot.prediction_replay.v1"


def _sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ArtifactError(f"IRCoT replay {label} must be a lowercase SHA-256")
    return value


def _mounted_json(root, spec):
    if not isinstance(spec, Mapping) or set(spec) != {"path", "sha256"}:
        raise ArtifactError("IRCoT replay files require a relative path and trusted checksum")
    name = spec["path"]
    if not isinstance(name, str) or not name or "\\" in name:
        raise ArtifactError("IRCoT replay file path is invalid")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArtifactError("IRCoT replay file path must remain under its manifest directory")
    target = root.joinpath(*path.parts)
    if not target.parent.resolve().is_relative_to(root):
        raise ArtifactError("IRCoT replay file parent escapes its manifest directory")
    return load_pinned_json_mapping(
        target, expected_sha256=_sha(spec["sha256"], "file checksum"), max_bytes=8 * 1024 * 1024
    )


@dataclass(frozen=True, slots=True)
class IRCoTPredictionArchive:
    """Immutable selected scope, including empty predictions and historical cost."""

    manifest_sha256: str
    manifest: Mapping[str, Any]
    records: Mapping[str, Any]
    source_prediction_count: int

    def __post_init__(self):
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))
        object.__setattr__(self, "records", freeze_mapping(self.records))

    @property
    def generator_fingerprint_sha256(self):
        return content_hash(self.manifest["generator_fingerprint"])


def load_ircot_prediction_archive(manifest_path, *, expected_sha256):
    """Load a bounded operator manifest plus an unchanged upstream-format file.

    ``inputs`` is a label-free binding map, never a ground-truth answer file.
    It may select a documented ID-based scope from a larger prediction file.
    Unselected predictions remain counted in ``source_prediction_count``.
    """
    pin = _sha(expected_sha256, "manifest checksum")
    path = Path(manifest_path).expanduser().absolute()
    manifest = load_pinned_json_mapping(path, expected_sha256=pin, max_bytes=1024 * 1024)
    required = {
        "schema",
        "origin_kind",
        "run_revision",
        "benchmark",
        "split",
        "generator_fingerprint",
        "dependency_fingerprint",
        "provenance",
        "selection_rule",
        "predictions",
        "inputs",
    }
    if set(manifest) != required or manifest["schema"] != IRCOT_REPLAY_SCHEMA:
        raise ArtifactError("IRCoT replay manifest schema/fields differ")
    if manifest["origin_kind"] not in ("author_release", "controlled_capture"):
        raise ArtifactError(
            "IRCoT replay origin must be an author release or separately labelled controlled capture"
        )
    revision = manifest["run_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ArtifactError(
            "IRCoT replay run_revision must be the actual full immutable Git revision"
        )
    for key in ("benchmark", "split", "selection_rule"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ArtifactError(f"IRCoT replay requires {key}")
    for key in ("generator_fingerprint", "dependency_fingerprint"):
        if not isinstance(manifest[key], Mapping) or not manifest[key]:
            raise ArtifactError(f"IRCoT replay requires {key}")
        content_hash(manifest[key])
    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "source_uri",
        "source_artifact_sha256",
        "run_config_sha256",
        "query_source_sha256",
    }:
        raise ArtifactError(
            "IRCoT replay requires explicit source, configuration and query provenance"
        )
    if not isinstance(provenance["source_uri"], str) or not provenance["source_uri"].strip():
        raise ArtifactError("IRCoT replay source_uri is required")
    for key in ("source_artifact_sha256", "run_config_sha256", "query_source_sha256"):
        _sha(provenance[key], key)
    root = path.parent.resolve()
    predictions = _mounted_json(root, manifest["predictions"])
    bindings = _mounted_json(root, manifest["inputs"])
    if not predictions or not bindings or not set(bindings) <= set(predictions):
        raise ArtifactError("IRCoT replay selected IDs must exist in the original prediction file")
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str) for k, v in predictions.items()
    ):
        raise ArtifactError(
            "IRCoT answer replay requires the original ID-to-text format, not retrieval IDs or gold answer lists"
        )
    records = {}
    for key, binding in bindings.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "query",
            "input_state_hash",
            "historical_usage",
            "origin_receipt_sha256",
        }:
            raise ArtifactError("IRCoT replay input binding fields differ; labels are not allowed")
        if not isinstance(binding["query"], str) or not binding["query"].strip():
            raise ArtifactError("IRCoT replay input binding requires its exact source query")
        for name in ("input_state_hash", "origin_receipt_sha256"):
            if binding[name] is not None:
                _sha(binding[name], name)
        historical = binding["historical_usage"]
        if historical is not None:
            if not isinstance(historical, Mapping) or not historical:
                raise ArtifactError(
                    "IRCoT replay historical_usage must be measured data or null, never a guessed zero"
                )
            from agentinputlab.observability import normalize_external_usage

            normalize_external_usage(historical)
        if (
            manifest["origin_kind"] == "controlled_capture"
            and binding["origin_receipt_sha256"] is None
        ):
            raise ArtifactError(
                "IRCoT controlled capture requires the original inference receipt checksum"
            )
        records[key] = {
            **dict(binding),
            "answer": predictions[key],
            "status": "archived_prediction"
            if predictions[key].strip()
            else "archived_empty_prediction",
        }
    return IRCoTPredictionArchive(pin, manifest, records, len(predictions))


@dataclass(frozen=True, slots=True)
class IRCoTPredictionReplay:
    manifest_path: str | Path
    expected_manifest_sha256: str
    expected_generator_fingerprint_sha256: str
    benchmark: str
    split: str
    method_id: str = "ircot"
    release_id: str = IRCOT_REPLAY_RELEASE
    family: SemanticFamily = SemanticFamily.ACQUIRE
    _archive: IRCoTPredictionArchive = field(init=False, repr=False)

    def __post_init__(self):
        if (
            self.method_id != "ircot"
            or self.release_id != IRCOT_REPLAY_RELEASE
            or self.family is not SemanticFamily.ACQUIRE
        ):
            raise ValidationError("IRCoT replay identity differs from the registered profile")
        archive = load_ircot_prediction_archive(
            self.manifest_path, expected_sha256=self.expected_manifest_sha256
        )
        if archive.generator_fingerprint_sha256 != _sha(
            self.expected_generator_fingerprint_sha256, "expected generator"
        ):
            raise ArtifactError("IRCoT replay belongs to a different model/deployment")
        if (
            archive.manifest["benchmark"] != self.benchmark
            or archive.manifest["split"] != self.split
        ):
            raise ArtifactError("IRCoT replay benchmark/split does not match the requested scope")
        object.__setattr__(self, "_archive", archive)

    @property
    def archive(self):
        """Inspect original empty outcomes/provenance without fabricating answers."""
        return self._archive

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        # Recheck trusted files before reuse; there is never a live fallback.
        archive = load_ircot_prediction_archive(
            self.manifest_path, expected_sha256=self.expected_manifest_sha256
        )
        if archive != self._archive:
            raise ArtifactError("IRCoT replay archive changed after construction")
        record = archive.records.get(state.task_id)
        if record is None or record["query"] != state.query:
            raise ArtifactError("IRCoT replay exact task ID/query mismatch; no generation fallback")
        if (
            record["input_state_hash"] is not None
            and record["input_state_hash"] != state.state_hash
        ):
            raise ArtifactError("IRCoT replay original input-state hash differs")
        for key, value in (("benchmark_cell", self.benchmark), ("source_split", self.split)):
            if key in state.metadata and state.metadata[key] != value:
                raise ArtifactError("IRCoT replay visible dataset scope differs")
        if (
            context.backend is not None
            and content_hash(context.backend.fingerprint()) != archive.generator_fingerprint_sha256
        ):
            raise ArtifactError(
                "IRCoT replay cannot relabel an archived answer as another live backbone"
            )
        if not record["answer"].strip():
            raise AlgorithmOutputError(
                "IRCoT archived empty prediction preserved; no retry or replacement answer"
            )
        manifest = archive.manifest
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=record["answer"],
            evidence=(),
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                events=(
                    TraceEvent(
                        TracePhase.PREPARE,
                        "prediction_archive_binding",
                        {
                            "manifest_sha256": archive.manifest_sha256,
                            "source_prediction_count": archive.source_prediction_count,
                            "selected_record_count": len(archive.records),
                            "task_id": state.task_id,
                            "generator_fingerprint_sha256": archive.generator_fingerprint_sha256,
                        },
                    ),
                    TraceEvent(
                        TracePhase.FINALIZE,
                        "archived_prediction",
                        {
                            "answer": record["answer"],
                            "origin_receipt_sha256": record["origin_receipt_sha256"],
                            "historical_usage": record["historical_usage"],
                        },
                    ),
                ),
                counters={
                    "model_calls": 0,
                    "retrieval_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            ),
            implementation_identity=IRCOT_REPLAY_SCHEMA,
            profile_kind="official_native"
            if manifest["origin_kind"] == "author_release"
            else "controlled_adapter",
            metadata={
                "artifact_replay": True,
                "origin_kind": manifest["origin_kind"],
                "run_revision": manifest["run_revision"],
                "manifest_sha256": archive.manifest_sha256,
                "generator_fingerprint": manifest["generator_fingerprint"],
                "dependency_fingerprint": manifest["dependency_fingerprint"],
                "provenance": manifest["provenance"],
                "historical_usage": record["historical_usage"],
                "historical_cost_recharged": False,
                "historical_usage_unknown": record["historical_usage"] is None,
                "new_provider_requests": 0,
                "current_seed_or_budget_rerun_claimed": False,
                "input_state_hash_verified": record["input_state_hash"] is not None,
                "answers_are_not_corpus_evidence": True,
                "checksum_proves_authorship": False,
                "benchmark_scoring": "caller_owned_independent_scorer",
                "automatic_retry": False,
            },
        )


__all__ = [
    "IRCOT_REPLAY_RELEASE",
    "IRCOT_REPLAY_SCHEMA",
    "IRCoTPredictionArchive",
    "IRCoTPredictionReplay",
    "load_ircot_prediction_archive",
]
