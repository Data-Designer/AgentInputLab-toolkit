"""Strict conversion from frozen discovery runs to selector training artifacts.

This module is a producer boundary, not an experiment runner.  It consumes only
completed, checksum-verified discovery run directories and never invokes a
driver, model, retriever, tool, or benchmark.  Raw state text remains in the
source experiment artifact; the exported selector discovery freeze contains
only hashes, visible-only feature observations, evaluator outcomes, and the
derived training table.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash
from agentinputlab.contracts.state import (
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    Message,
)
from agentinputlab.experiments import (
    ExperimentManifest,
    ExperimentStage,
    load_experiment_manifest,
    load_verified_cell_results,
    verify_experiment_run_directory,
)
from agentinputlab.metrics import assess_hard_harm_from_metrics

from .base import SelectionFeatures
from .features import VisibleFeatureConfig, VisibleStateFeatureExtractor
from .learned import SelectorExample
from .outcome_table import (
    EvaluatedArmOutcome,
    EvaluatedArmStatus,
    SelectorTrainingTable,
    UtilityDefinition,
    build_selector_training_table,
)
from .runtime import FeatureAccessDomain, FeatureObservation

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CORE_FAMILIES = frozenset(
    {
        SemanticFamily.REFORMULATE,
        SemanticFamily.DECOMPOSE,
        SemanticFamily.ACQUIRE,
        SemanticFamily.CLARIFY,
        SemanticFamily.CURATE,
    }
)
_SELECTOR_SPLIT_BY_EXPERIMENT_SPLIT = {
    "discovery_train": "train",
    "discovery_calibration": "calibration",
}


@dataclass(frozen=True, slots=True)
class SelectorDiscoverySource:
    """Content identities for one verified source experiment run."""

    study_id: str
    identity_hash: str
    manifest_hash: str
    plan_hash: str
    result_set_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValidationError("selector discovery source study_id must be non-empty")
        for label in (
            "identity_hash",
            "manifest_hash",
            "plan_hash",
            "result_set_hash",
        ):
            if not isinstance(getattr(self, label), str) or not _SHA256.fullmatch(
                getattr(self, label)
            ):
                raise ValidationError(f"selector discovery source {label} must be SHA-256")

    @property
    def source_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class SelectorDiscoveryIdentity:
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.revision, str)
            or not _SAFE_SEGMENT.fullmatch(self.revision)
            or ".." in self.revision
        ):
            raise ValidationError("selector discovery revision must be a safe flat identifier")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValidationError("selector discovery identity requires a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SelectorDiscoveryFreeze:
    """Immutable, content-addressed selector producer input."""

    revision: str
    source_runs: tuple[SelectorDiscoverySource, ...]
    observations: tuple[FeatureObservation, ...]
    outcomes: tuple[EvaluatedArmOutcome, ...]
    training_table: SelectorTrainingTable
    utility_definition: UtilityDefinition
    representative_by_family: Mapping[SemanticFamily, str]
    extractor_id: str
    extractor_revision: str
    extractor_hash: str
    feature_schema_id: str
    feature_schema_hash: str
    comparison_profile_hash: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValidationError("unsupported selector discovery schema")
        if (
            not isinstance(self.revision, str)
            or not _SAFE_SEGMENT.fullmatch(self.revision)
            or ".." in self.revision
        ):
            raise ValidationError("selector discovery revision must be a safe flat identifier")
        sources = tuple(self.source_runs)
        if not sources or any(not isinstance(item, SelectorDiscoverySource) for item in sources):
            raise ValidationError("selector discovery freeze requires source run identities")
        if len({item.identity_hash for item in sources}) != len(sources):
            raise ValidationError("selector discovery source run identities must be unique")
        observations = tuple(self.observations)
        outcomes = tuple(self.outcomes)
        if not observations or any(
            not isinstance(item, FeatureObservation) for item in observations
        ):
            raise ValidationError("selector discovery freeze requires feature observations")
        if not outcomes or any(not isinstance(item, EvaluatedArmOutcome) for item in outcomes):
            raise ValidationError("selector discovery freeze requires evaluated arm outcomes")
        if len({item.state_hash for item in observations}) != len(observations):
            raise ValidationError("selector discovery observations require unique state hashes")
        if not isinstance(self.training_table, SelectorTrainingTable):
            raise ValidationError("selector discovery freeze requires SelectorTrainingTable")
        if not isinstance(self.utility_definition, UtilityDefinition):
            raise ValidationError("selector discovery freeze requires UtilityDefinition")
        representatives = _validate_representatives(self.representative_by_family)
        for label in ("extractor_id", "extractor_revision", "feature_schema_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(f"selector discovery {label} must be trimmed and non-empty")
        for label in (
            "extractor_hash",
            "feature_schema_hash",
            "comparison_profile_hash",
        ):
            if not _SHA256.fullmatch(getattr(self, label)):
                raise ValidationError(f"selector discovery {label} must be SHA-256")
        if {item.state_hash for item in observations} != {item.state_hash for item in outcomes}:
            raise ValidationError(
                "selector discovery observations and outcomes cover different states"
            )
        for observation in observations:
            if (
                observation.extractor_id != self.extractor_id
                or observation.extractor_revision != self.extractor_revision
                or observation.provenance.get("extractor_hash") != self.extractor_hash
                or observation.provenance.get("schema_id") != self.feature_schema_id
                or observation.provenance.get("schema_hash") != self.feature_schema_hash
            ):
                raise ValidationError(
                    "selector discovery feature observation differs from frozen extractor/schema"
                )
        if {item.comparison_profile_hash for item in outcomes} != {self.comparison_profile_hash}:
            raise ValidationError("selector discovery outcomes must share one comparison profile")
        rebuilt = build_selector_training_table(
            outcomes,
            {item.state_hash: item for item in observations},
            utility_definition=self.utility_definition,
            representative_by_family=representatives,
        )
        if rebuilt.table_hash != self.training_table.table_hash:
            raise ValidationError(
                "selector discovery exported training table differs from its source rows"
            )
        object.__setattr__(
            self,
            "source_runs",
            tuple(sorted(sources, key=lambda item: item.identity_hash)),
        )
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(observations, key=lambda item: item.state_hash)),
        )
        object.__setattr__(
            self,
            "outcomes",
            tuple(sorted(outcomes, key=_outcome_sort_key)),
        )
        object.__setattr__(
            self,
            "representative_by_family",
            MappingProxyType(representatives),
        )

    @property
    def split_provenance_hash(self) -> str:
        return content_hash(
            {
                "group_split_digests": self.training_table.group_split_digests,
            }
        )

    @property
    def discovery_hash(self) -> str:
        return content_hash(self.to_artifact())

    @property
    def identity(self) -> SelectorDiscoveryIdentity:
        return SelectorDiscoveryIdentity(self.revision, self.discovery_hash)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "source_runs": [_source_to_artifact(item) for item in self.source_runs],
            "extractor": {
                "id": self.extractor_id,
                "revision": self.extractor_revision,
                "sha256": self.extractor_hash,
            },
            "feature_schema": {
                "id": self.feature_schema_id,
                "sha256": self.feature_schema_hash,
            },
            "comparison_profile_hash": self.comparison_profile_hash,
            "utility_definition": _utility_to_artifact(self.utility_definition),
            "representative_by_family": _encode_representatives(self.representative_by_family),
            "observations": [_observation_to_artifact(item) for item in self.observations],
            "outcomes": [_outcome_to_artifact(item) for item in self.outcomes],
            "training_table": _training_table_to_artifact(self.training_table),
        }

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any]) -> SelectorDiscoveryFreeze:
        artifact = _strict_object(
            value,
            allowed={
                "schema_version",
                "revision",
                "source_runs",
                "extractor",
                "feature_schema",
                "comparison_profile_hash",
                "utility_definition",
                "representative_by_family",
                "observations",
                "outcomes",
                "training_table",
            },
            required={
                "schema_version",
                "revision",
                "source_runs",
                "extractor",
                "feature_schema",
                "comparison_profile_hash",
                "utility_definition",
                "representative_by_family",
                "observations",
                "outcomes",
                "training_table",
            },
            label="selector discovery artifact",
        )
        extractor = _strict_object(
            artifact["extractor"],
            allowed={"id", "revision", "sha256"},
            required={"id", "revision", "sha256"},
            label="selector discovery extractor",
        )
        feature_schema = _strict_object(
            artifact["feature_schema"],
            allowed={"id", "sha256"},
            required={"id", "sha256"},
            label="selector discovery feature schema",
        )
        return cls(
            schema_version=_integer(
                artifact["schema_version"],
                label="schema_version",
            ),
            revision=_text(artifact["revision"], label="revision"),
            source_runs=tuple(
                _source_from_artifact(item)
                for item in _array(artifact["source_runs"], label="source_runs")
            ),
            observations=tuple(
                _observation_from_artifact(item)
                for item in _array(artifact["observations"], label="observations")
            ),
            outcomes=tuple(
                _outcome_from_artifact(item)
                for item in _array(artifact["outcomes"], label="outcomes")
            ),
            training_table=_training_table_from_artifact(artifact["training_table"]),
            utility_definition=_utility_from_artifact(artifact["utility_definition"]),
            representative_by_family=_decode_representatives(artifact["representative_by_family"]),
            extractor_id=_text(extractor["id"], label="extractor.id"),
            extractor_revision=_text(
                extractor["revision"],
                label="extractor.revision",
            ),
            extractor_hash=_sha256(
                extractor["sha256"],
                label="extractor.sha256",
            ),
            feature_schema_id=_text(
                feature_schema["id"],
                label="feature_schema.id",
            ),
            feature_schema_hash=_sha256(
                feature_schema["sha256"],
                label="feature_schema.sha256",
            ),
            comparison_profile_hash=_sha256(
                artifact["comparison_profile_hash"],
                label="comparison_profile_hash",
            ),
        )


class SelectorDiscoveryStore:
    """Append-only, content-addressed storage for discovery freezes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, identity: SelectorDiscoveryIdentity) -> Path:
        if not isinstance(identity, SelectorDiscoveryIdentity):
            raise TypeError("selector discovery path requires SelectorDiscoveryIdentity")
        destination = (
            self.root
            / "selector_discovery"
            / identity.revision
            / identity.sha256
            / "discovery.json"
        ).resolve()
        if self.root not in destination.parents:
            raise ArtifactError("selector discovery path escaped store root")
        return destination

    def save(self, freeze: SelectorDiscoveryFreeze) -> SelectorDiscoveryIdentity:
        if not isinstance(freeze, SelectorDiscoveryFreeze):
            raise TypeError("selector discovery store requires SelectorDiscoveryFreeze")
        identity = freeze.identity
        destination = self.path_for(identity)
        revision_root = destination.parent.parent
        if revision_root.exists():
            existing_hashes = {
                item.name
                for item in revision_root.iterdir()
                if item.is_dir() and _SHA256.fullmatch(item.name)
            }
            if existing_hashes - {identity.sha256}:
                raise ArtifactError(
                    "selector discovery revision is already bound to different content"
                )
        payload = (canonical_json(freeze.to_artifact()) + "\n").encode("utf-8")
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != payload:
                raise ArtifactError("stored selector discovery bytes differ from identity")
            self.load(identity)
            return identity
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".partial-",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != payload:
                    raise ArtifactError(
                        "concurrent selector discovery materialization differs"
                    ) from None
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.load(identity)
        return identity

    def load(self, identity: SelectorDiscoveryIdentity) -> SelectorDiscoveryFreeze:
        path = self.path_for(identity)
        if not path.is_file():
            raise ArtifactError(
                f"selector discovery artifact missing: {identity.revision}@{identity.sha256}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("selector discovery artifact is not valid UTF-8 JSON") from exc
        if not isinstance(raw, Mapping):
            raise ArtifactError("selector discovery artifact root must be an object")
        try:
            freeze = SelectorDiscoveryFreeze.from_artifact(raw)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("selector discovery artifact failed validation") from exc
        if freeze.identity != identity:
            raise ArtifactError("selector discovery identity does not match stored content")
        return freeze


def build_selector_discovery_freeze(
    run_dirs: Sequence[str | Path],
    *,
    revision: str,
    extractor: VisibleStateFeatureExtractor,
    utility_definition: UtilityDefinition,
    representative_by_family: Mapping[SemanticFamily, str],
) -> SelectorDiscoveryFreeze:
    """Convert explicit, verified discovery runs into a frozen training artifact."""

    roots = tuple(Path(item).expanduser().resolve() for item in run_dirs)
    if not roots:
        raise ValidationError("selector discovery conversion requires run directories")
    if not isinstance(extractor, VisibleStateFeatureExtractor):
        raise TypeError("selector discovery production requires VisibleStateFeatureExtractor")
    if not isinstance(utility_definition, UtilityDefinition):
        raise TypeError("utility_definition must be UtilityDefinition")
    representatives = _validate_representatives(representative_by_family)

    observations: dict[str, FeatureObservation] = {}
    outcomes: list[EvaluatedArmOutcome] = []
    sources: list[SelectorDiscoverySource] = []
    source_identities: set[str] = set()
    profile_hashes: set[str] = set()
    backend_by_model_slot: dict[str, str] = {}

    for root in roots:
        verification = verify_experiment_run_directory(root)
        if not verification["complete"] or not verification["all_succeeded"]:
            raise ArtifactError(
                "selector discovery conversion requires a complete, all-succeeded run"
            )
        manifest = load_experiment_manifest(root / "manifest.json")
        _validate_discovery_manifest(manifest)
        if manifest.metadata.get("full_five_family_selector") is True:
            if set(representatives) != _CORE_FAMILIES:
                raise ValidationError(
                    "full-five-family selector discovery requires exactly one "
                    "representative for every core intervention family"
                )
            manifest_families = {
                arm.family for arm in manifest.arms if arm.family is not SemanticFamily.NO_OP
            }
            if manifest_families != _CORE_FAMILIES:
                raise ValidationError(
                    "full-five-family selector discovery manifest must contain "
                    "all and only the five core intervention families"
                )
        records = load_verified_cell_results(root)
        source = SelectorDiscoverySource(
            study_id=manifest.study_id,
            identity_hash=_sha256(
                verification["identity_hash"],
                label="source identity_hash",
            ),
            manifest_hash=_sha256(
                verification["manifest_hash"],
                label="source manifest_hash",
            ),
            plan_hash=_sha256(
                verification["plan_hash"],
                label="source plan_hash",
            ),
            result_set_hash=content_hash(records),
        )
        if source.identity_hash in source_identities:
            raise ValidationError("selector discovery source run was supplied more than once")
        source_identities.add(source.identity_hash)
        sources.append(source)
        arm_by_id = {item.arm_id: item for item in manifest.arms}
        arm_by_release: dict[str, list[str]] = {}
        for arm in manifest.arms:
            arm_by_release.setdefault(arm.release_id, []).append(arm.arm_id)
        no_op_arm_id = _unique_arm_for_release(
            arm_by_release,
            "baseline.no_op",
            label="NO_OP",
        )
        representative_arm_ids = {
            family: _unique_arm_for_release(
                arm_by_release,
                release_id,
                label=family.value,
            )
            for family, release_id in representatives.items()
        }
        for record in records:
            converted, observation, profile_hash, model_slot, backend_hash = _convert_result_record(
                record,
                arm_by_id=arm_by_id,
                no_op_arm_id=no_op_arm_id,
                representative_arm_ids=representative_arm_ids,
                representative_by_family=representatives,
                extractor=extractor,
            )
            existing = observations.setdefault(observation.state_hash, observation)
            if existing != observation:
                raise ValidationError("one frozen state produced different feature observations")
            outcomes.extend(converted)
            profile_hashes.add(profile_hash)
            previous_backend = backend_by_model_slot.setdefault(
                model_slot,
                backend_hash,
            )
            if previous_backend != backend_hash:
                raise ValidationError(
                    f"model slot {model_slot!r} changed backend identity across discovery runs"
                )

    if len(profile_hashes) != 1:
        raise ValidationError("selector discovery inputs must share one comparison profile hash")
    comparison_profile_hash = next(iter(profile_hashes))
    table = build_selector_training_table(
        outcomes,
        observations,
        utility_definition=utility_definition,
        representative_by_family=representatives,
    )
    return SelectorDiscoveryFreeze(
        revision=revision,
        source_runs=tuple(sources),
        observations=tuple(observations.values()),
        outcomes=tuple(outcomes),
        training_table=table,
        utility_definition=utility_definition,
        representative_by_family=representatives,
        extractor_id=extractor.extractor_id,
        extractor_revision=extractor.extractor_revision,
        extractor_hash=extractor.extractor_hash,
        feature_schema_id=extractor.feature_schema_id,
        feature_schema_hash=extractor.feature_schema_hash,
        comparison_profile_hash=comparison_profile_hash,
    )


def load_discovery_export_config(
    path: str | Path,
) -> tuple[
    str,
    VisibleStateFeatureExtractor,
    UtilityDefinition,
    Mapping[SemanticFamily, str],
]:
    """Load the explicit, strict configuration used by the export script."""

    return parse_discovery_export_config(
        _read_json_object(path, label="selector discovery export config")
    )


def parse_discovery_export_config(
    raw: Mapping[str, Any],
) -> tuple[
    str,
    VisibleStateFeatureExtractor,
    UtilityDefinition,
    Mapping[SemanticFamily, str],
]:
    """Parse an already authenticated selector discovery export config."""

    config = _strict_object(
        raw,
        allowed={
            "schema_version",
            "revision",
            "feature_extractor",
            "utility_definition",
            "representative_by_family",
        },
        required={
            "schema_version",
            "revision",
            "feature_extractor",
            "utility_definition",
            "representative_by_family",
        },
        label="selector discovery export config",
    )
    if _integer(config["schema_version"], label="schema_version") != 1:
        raise ValidationError("unsupported selector discovery export config")
    extractor_raw = _strict_object(
        config["feature_extractor"],
        allowed={
            "extractor_id",
            "extractor_revision",
            "schema_id",
            "tokenization_revision",
            "context_soft_limit_tokens",
        },
        required={
            "extractor_id",
            "extractor_revision",
            "schema_id",
            "tokenization_revision",
            "context_soft_limit_tokens",
        },
        label="feature_extractor",
    )
    extractor = VisibleStateFeatureExtractor(
        config=VisibleFeatureConfig(
            schema_id=_text(extractor_raw["schema_id"], label="schema_id"),
            tokenization_revision=_text(
                extractor_raw["tokenization_revision"],
                label="tokenization_revision",
            ),
            context_soft_limit_tokens=_integer(
                extractor_raw["context_soft_limit_tokens"],
                label="context_soft_limit_tokens",
            ),
        ),
        extractor_id=_text(
            extractor_raw["extractor_id"],
            label="extractor_id",
        ),
        extractor_revision=_text(
            extractor_raw["extractor_revision"],
            label="extractor_revision",
        ),
    )
    return (
        _text(config["revision"], label="revision"),
        extractor,
        _utility_from_artifact(config["utility_definition"]),
        _decode_representatives(config["representative_by_family"]),
    )


def _validate_discovery_manifest(manifest: ExperimentManifest) -> None:
    if manifest.stage is not ExperimentStage.DISCOVERY:
        raise ValidationError("selector producer accepts discovery-stage runs only")
    if manifest.metadata.get("selector_training_allowed") is not True:
        raise ValidationError(
            "discovery manifest must explicitly set selector_training_allowed=true"
        )


def _convert_result_record(
    record: Mapping[str, Any],
    *,
    arm_by_id: Mapping[str, Any],
    no_op_arm_id: str,
    representative_arm_ids: Mapping[SemanticFamily, str],
    representative_by_family: Mapping[SemanticFamily, str],
    extractor: VisibleStateFeatureExtractor,
    selector_split_override: str | None = None,
) -> tuple[
    tuple[EvaluatedArmOutcome, ...],
    FeatureObservation,
    str,
    str,
    str,
]:
    cell = _mapping(record.get("cell"), label="result cell")
    result = _mapping(record.get("result"), label="result payload")
    if result.get("status") != "succeeded":
        raise ArtifactError("selector discovery cannot consume a failed experiment cell")
    payload = _mapping(result.get("payload"), label="successful result.payload")
    selector_record = _strict_object(
        payload.get("selector_record"),
        allowed={
            "schema_version",
            "state",
            "group_id",
            "eligible_families",
        },
        required={
            "schema_version",
            "state",
            "group_id",
            "eligible_families",
        },
        label="result.payload.selector_record",
    )
    if _integer(selector_record["schema_version"], label="selector_record.schema_version") != 1:
        raise ArtifactError("unsupported selector_record schema")
    state = _state_from_artifact(selector_record["state"])
    observation = extractor.extract(state)
    _validate_pre_action_receipt(
        payload.get("pre_action"),
        observation=observation,
        extractor=extractor,
    )
    experiment_split = _text(cell.get("split"), label="cell.split")
    if selector_split_override is None:
        try:
            expected_selector_split = _SELECTOR_SPLIT_BY_EXPERIMENT_SPLIT[experiment_split]
        except KeyError as exc:
            allowed = ", ".join(sorted(_SELECTOR_SPLIT_BY_EXPERIMENT_SPLIT))
            raise ValidationError(
                "selector discovery cell.split must freeze the producer partition "
                f"({allowed}); got {experiment_split!r}"
            ) from exc
        split = expected_selector_split
    else:
        if selector_split_override != "test":
            raise ValidationError(
                "selector result conversion override is reserved for untouched test"
            )
        split = selector_split_override
    raw_group_id = _text(selector_record["group_id"], label="group_id")
    benchmark_id = _text(cell.get("benchmark_id"), label="cell.benchmark_id")
    unit_id = _text(cell.get("unit_id"), label="cell.unit_id")
    model_slot = _text(cell.get("model_slot"), label="cell.model_slot")
    seed = _integer(cell.get("seed"), label="cell.seed")
    example_id = "example_" + content_hash({"benchmark_id": benchmark_id, "unit_id": unit_id})
    group_id = "group_" + content_hash({"benchmark_id": benchmark_id, "group_id": raw_group_id})

    cell_arm_ids = tuple(
        _text(item, label="cell.arm_ids item")
        for item in _array(cell.get("arm_ids"), label="cell.arm_ids")
    )
    if len(cell_arm_ids) != len(set(cell_arm_ids)):
        raise ArtifactError("selector discovery cell contains duplicate arm IDs")
    try:
        cell_specs = {arm_id: arm_by_id[arm_id] for arm_id in cell_arm_ids}
    except KeyError as exc:
        raise ArtifactError("selector discovery cell references an unknown manifest arm") from exc
    if no_op_arm_id not in cell_specs:
        raise ValidationError("selector discovery cell requires an explicit NO_OP arm")
    dependencies = _dependency_hashes(
        payload.get("observed_dependency_hashes"),
        label="result.payload.observed_dependency_hashes",
        expected_arm_ids=cell_arm_ids,
    )
    backend_hash = _selector_backend_hash(
        dependencies,
        model_slot=model_slot,
    )
    profile_hash = _selector_comparison_profile_hash(
        cell_arm_ids,
        cell_specs,
    )

    eligible_families = _family_array(
        selector_record["eligible_families"],
        label="eligible_families",
    )
    observed_families = {
        spec.family for spec in cell_specs.values() if spec.family is not SemanticFamily.NO_OP
    }
    if eligible_families != observed_families:
        raise ValidationError(
            "selector_record eligible_families differ from the frozen cell arm families"
        )
    for family, arm_id in representative_arm_ids.items():
        if family in eligible_families and arm_id not in cell_specs:
            raise ValidationError(
                f"eligible family {family.value} lacks its frozen representative arm"
            )

    raw_arms = _array(payload.get("arms"), label="result.payload.arms")
    arm_results: dict[str, Mapping[str, Any]] = {}
    for index, raw_arm in enumerate(raw_arms):
        arm = _mapping(raw_arm, label=f"result.payload.arms[{index}]")
        arm_id = _text(arm.get("arm_id"), label=f"arms[{index}].arm_id")
        if arm_id in arm_results:
            raise ArtifactError("selector discovery result contains duplicate arms")
        arm_results[arm_id] = arm
    if set(arm_results) != set(cell_arm_ids):
        raise ArtifactError("selector discovery result arms differ from the frozen cell inventory")

    normalized_arms = {
        arm_id: _normalized_selector_arm(
            raw_arm,
            arm_id=arm_id,
        )
        for arm_id, raw_arm in arm_results.items()
    }
    _baseline_status, _baseline_metrics, baseline_full_cost, _baseline_harm = normalized_arms[
        no_op_arm_id
    ]

    selected: list[tuple[SemanticFamily, str, str, bool]] = [
        (SemanticFamily.NO_OP, "baseline.no_op", no_op_arm_id, False)
    ]
    selected.extend(
        (
            family,
            representative_by_family[family],
            arm_id,
            family not in eligible_families,
        )
        for family, arm_id in sorted(
            representative_arm_ids.items(),
            key=lambda item: item[0].value,
        )
    )
    converted: list[EvaluatedArmOutcome] = []
    for family, release_id, arm_id, structural in selected:
        if structural:
            converted.append(
                EvaluatedArmOutcome(
                    example_id=example_id,
                    group_id=group_id,
                    state_hash=state.state_hash,
                    split=split,
                    benchmark_id=benchmark_id,
                    family=family,
                    release_id=release_id,
                    seed=seed,
                    backend_hash=backend_hash,
                    comparison_profile_hash=profile_hash,
                    status=EvaluatedArmStatus.STRUCTURAL_INELIGIBLE,
                )
            )
            continue
        converted_status, metrics, full_cost, hard_harm = normalized_arms[arm_id]
        is_no_op = family is SemanticFamily.NO_OP
        cost = 0.0 if is_no_op else max(0.0, full_cost - baseline_full_cost)
        # Strict harm is independent of whether utility falls below NO_OP.
        # Utility regression remains available from the arm utility and NO_OP
        # utility stored together in the selector outcome table.
        risk = 0.0 if is_no_op else float(hard_harm)
        converted.append(
            EvaluatedArmOutcome(
                example_id=example_id,
                group_id=group_id,
                state_hash=state.state_hash,
                split=split,
                benchmark_id=benchmark_id,
                family=family,
                release_id=release_id,
                seed=seed,
                backend_hash=backend_hash,
                comparison_profile_hash=profile_hash,
                status=converted_status,
                metrics=metrics,
                cost=cost,
                risk=risk,
            )
        )
    return tuple(converted), observation, profile_hash, model_slot, backend_hash


def _validate_pre_action_receipt(
    value: object,
    *,
    observation: FeatureObservation,
    extractor: VisibleStateFeatureExtractor,
) -> None:
    """Bind selector features to the state observed before any intervention.

    The experiment driver creates this receipt at the benchmark/checkpoint
    boundary.  The selector producer independently reconstructs the visible
    state and feature observation, then requires every content identity and
    score to match.  A driver payload cannot silently relabel a post-action
    state or swap a train/calibration row after execution.
    """

    receipt = _strict_object(
        value,
        allowed={
            "state_hash",
            "feature_hash",
            "feature_schema_id",
            "feature_schema_hash",
            "provenance_hash",
            "defect_scores",
        },
        required={
            "state_hash",
            "feature_hash",
            "feature_schema_id",
            "feature_schema_hash",
            "provenance_hash",
            "defect_scores",
        },
        label="result.payload.pre_action",
    )
    expected_text = {
        "state_hash": observation.state_hash,
        "feature_hash": observation.feature_hash,
        "feature_schema_id": extractor.feature_schema_id,
        "feature_schema_hash": extractor.feature_schema_hash,
        "provenance_hash": content_hash(observation.provenance),
    }
    for key, expected in expected_text.items():
        observed = _text(receipt[key], label=f"pre_action.{key}")
        if observed != expected:
            raise ArtifactError(f"pre_action.{key} does not authenticate selector_record.state")

    features = observation.features
    expected_scores = {
        SemanticFamily.REFORMULATE.value: features.reformulation_need_score,
        SemanticFamily.DECOMPOSE.value: features.multi_step_score,
        SemanticFamily.ACQUIRE.value: features.retrieval_need_score,
        SemanticFamily.CLARIFY.value: features.ambiguity_score,
        SemanticFamily.CURATE.value: features.context_pressure,
    }
    raw_scores = _mapping(receipt["defect_scores"], label="pre_action.defect_scores")
    if set(raw_scores) != set(expected_scores):
        raise ArtifactError("pre_action.defect_scores must contain exactly five families")
    for family, expected in expected_scores.items():
        observed = _number(
            raw_scores[family],
            label=f"pre_action.defect_scores.{family}",
        )
        if observed != expected:
            raise ArtifactError(
                "pre_action defect scores do not authenticate selector_record.state"
            )


def _state_from_artifact(value: object) -> FrozenAgentState:
    state = _strict_object(
        value,
        allowed={
            "query",
            "task_id",
            "instruction",
            "context",
            "messages",
            "tool_names",
            "metadata",
            "budget",
        },
        required={
            "query",
            "task_id",
            "instruction",
            "context",
            "messages",
            "tool_names",
            "metadata",
            "budget",
        },
        label="selector_record.state",
    )
    context = tuple(
        _context_from_artifact(item, index=index)
        for index, item in enumerate(_array(state["context"], label="state.context"))
    )
    messages = tuple(
        _message_from_artifact(item, index=index)
        for index, item in enumerate(_array(state["messages"], label="state.messages"))
    )
    tool_names = tuple(
        _text(item, label="state.tool_names item")
        for item in _array(state["tool_names"], label="state.tool_names")
    )
    metadata = _mapping(state["metadata"], label="state.metadata")
    budget_raw = _strict_object(
        state["budget"],
        allowed={
            "max_model_calls",
            "max_retrieval_calls",
            "max_tool_calls",
            "max_output_tokens",
            "max_context_tokens",
            "max_cost_usd",
        },
        required={
            "max_model_calls",
            "max_retrieval_calls",
            "max_tool_calls",
            "max_output_tokens",
            "max_context_tokens",
            "max_cost_usd",
        },
        label="state.budget",
    )
    return FrozenAgentState(
        query=_string(state["query"], label="state.query"),
        task_id=_string(state["task_id"], label="state.task_id"),
        instruction=_string(state["instruction"], label="state.instruction"),
        context=context,
        messages=messages,
        tool_names=tool_names,
        metadata=metadata,
        budget=ExecutionBudget(
            max_model_calls=_integer(
                budget_raw["max_model_calls"],
                label="budget.max_model_calls",
            ),
            max_retrieval_calls=_integer(
                budget_raw["max_retrieval_calls"],
                label="budget.max_retrieval_calls",
            ),
            max_tool_calls=_integer(
                budget_raw["max_tool_calls"],
                label="budget.max_tool_calls",
            ),
            max_output_tokens=_integer(
                budget_raw["max_output_tokens"],
                label="budget.max_output_tokens",
            ),
            max_context_tokens=_optional_integer(
                budget_raw["max_context_tokens"],
                label="budget.max_context_tokens",
            ),
            max_cost_usd=_optional_number(
                budget_raw["max_cost_usd"],
                label="budget.max_cost_usd",
            ),
        ),
    )


def _context_from_artifact(value: object, *, index: int) -> ContextBlock:
    context = _strict_object(
        value,
        allowed={"text", "block_id", "source", "score", "metadata"},
        required={"text", "block_id", "source", "score", "metadata"},
        label=f"state.context[{index}]",
    )
    return ContextBlock(
        text=_string(context["text"], label=f"context[{index}].text"),
        block_id=_string(
            context["block_id"],
            label=f"context[{index}].block_id",
        ),
        source=_string(context["source"], label=f"context[{index}].source"),
        score=_optional_number(
            context["score"],
            label=f"context[{index}].score",
        ),
        metadata=_mapping(
            context["metadata"],
            label=f"context[{index}].metadata",
        ),
    )


def _message_from_artifact(value: object, *, index: int) -> Message:
    message = _strict_object(
        value,
        allowed={"role", "content", "name"},
        required={"role", "content", "name"},
        label=f"state.messages[{index}]",
    )
    name = message["name"]
    if name is not None and not isinstance(name, str):
        raise ValidationError(f"state.messages[{index}].name must be a string or null")
    return Message(
        role=_text(message["role"], label=f"messages[{index}].role"),
        content=_string(
            message["content"],
            label=f"messages[{index}].content",
        ),
        name=name,
    )


def _source_to_artifact(value: SelectorDiscoverySource) -> dict[str, Any]:
    return {
        "study_id": value.study_id,
        "identity_hash": value.identity_hash,
        "manifest_hash": value.manifest_hash,
        "plan_hash": value.plan_hash,
        "result_set_hash": value.result_set_hash,
        "source_hash": value.source_hash,
    }


def _source_from_artifact(value: object) -> SelectorDiscoverySource:
    raw = _strict_object(
        value,
        allowed={
            "study_id",
            "identity_hash",
            "manifest_hash",
            "plan_hash",
            "result_set_hash",
            "source_hash",
        },
        required={
            "study_id",
            "identity_hash",
            "manifest_hash",
            "plan_hash",
            "result_set_hash",
            "source_hash",
        },
        label="selector discovery source",
    )
    source = SelectorDiscoverySource(
        study_id=_text(raw["study_id"], label="source.study_id"),
        identity_hash=_sha256(
            raw["identity_hash"],
            label="source.identity_hash",
        ),
        manifest_hash=_sha256(
            raw["manifest_hash"],
            label="source.manifest_hash",
        ),
        plan_hash=_sha256(raw["plan_hash"], label="source.plan_hash"),
        result_set_hash=_sha256(
            raw["result_set_hash"],
            label="source.result_set_hash",
        ),
    )
    if _sha256(raw["source_hash"], label="source.source_hash") != source.source_hash:
        raise ValidationError("selector discovery source hash mismatch")
    return source


def _utility_to_artifact(value: UtilityDefinition) -> dict[str, Any]:
    return {
        "utility_metric": value.utility_metric,
        "algorithm_failure_utility": value.algorithm_failure_utility,
        "aggregate": value.aggregate,
        "definition_hash": value.definition_hash,
    }


def _utility_from_artifact(value: object) -> UtilityDefinition:
    raw = _strict_object(
        value,
        allowed={
            "utility_metric",
            "algorithm_failure_utility",
            "aggregate",
            "definition_hash",
        },
        required={
            "utility_metric",
            "algorithm_failure_utility",
            "aggregate",
        },
        label="utility_definition",
    )
    utility = UtilityDefinition(
        utility_metric=_text(
            raw["utility_metric"],
            label="utility_metric",
        ),
        algorithm_failure_utility=_number(
            raw["algorithm_failure_utility"],
            label="algorithm_failure_utility",
        ),
        aggregate=_text(raw["aggregate"], label="aggregate"),
    )
    if (
        "definition_hash" in raw
        and _sha256(
            raw["definition_hash"],
            label="definition_hash",
        )
        != utility.definition_hash
    ):
        raise ValidationError("utility definition hash mismatch")
    return utility


def _observation_to_artifact(value: FeatureObservation) -> dict[str, Any]:
    return {
        "state_hash": value.state_hash,
        "extractor_id": value.extractor_id,
        "extractor_revision": value.extractor_revision,
        "access_domains": [item.value for item in value.access_domains],
        "features": _features_to_artifact(value.features),
        "provenance": value.provenance,
        "feature_hash": value.feature_hash,
    }


def _observation_from_artifact(value: object) -> FeatureObservation:
    raw = _strict_object(
        value,
        allowed={
            "state_hash",
            "extractor_id",
            "extractor_revision",
            "access_domains",
            "features",
            "provenance",
            "feature_hash",
        },
        required={
            "state_hash",
            "extractor_id",
            "extractor_revision",
            "access_domains",
            "features",
            "provenance",
            "feature_hash",
        },
        label="feature observation",
    )
    try:
        domains = tuple(
            FeatureAccessDomain(_text(item, label="access domain"))
            for item in _array(raw["access_domains"], label="access_domains")
        )
    except ValueError as exc:
        raise ValidationError("unknown feature access domain") from exc
    observation = FeatureObservation(
        features=_features_from_artifact(raw["features"]),
        state_hash=_sha256(raw["state_hash"], label="observation.state_hash"),
        extractor_id=_text(raw["extractor_id"], label="observation.extractor_id"),
        extractor_revision=_text(
            raw["extractor_revision"],
            label="observation.extractor_revision",
        ),
        access_domains=domains,
        provenance=_mapping(raw["provenance"], label="observation.provenance"),
    )
    if _sha256(raw["feature_hash"], label="feature_hash") != observation.feature_hash:
        raise ValidationError("feature observation hash mismatch")
    return observation


def _features_to_artifact(value: SelectionFeatures) -> dict[str, float]:
    return {
        "ambiguity_score": value.ambiguity_score,
        "multi_step_score": value.multi_step_score,
        "retrieval_need_score": value.retrieval_need_score,
        "context_pressure": value.context_pressure,
        "reformulation_need_score": value.reformulation_need_score,
    }


def _features_from_artifact(value: object) -> SelectionFeatures:
    raw = _strict_object(
        value,
        allowed={
            "ambiguity_score",
            "multi_step_score",
            "retrieval_need_score",
            "context_pressure",
            "reformulation_need_score",
        },
        required={
            "ambiguity_score",
            "multi_step_score",
            "retrieval_need_score",
            "context_pressure",
            "reformulation_need_score",
        },
        label="selection features",
    )
    return SelectionFeatures(
        ambiguity_score=_number(raw["ambiguity_score"], label="ambiguity_score"),
        multi_step_score=_number(raw["multi_step_score"], label="multi_step_score"),
        retrieval_need_score=_number(
            raw["retrieval_need_score"],
            label="retrieval_need_score",
        ),
        context_pressure=_number(
            raw["context_pressure"],
            label="context_pressure",
        ),
        reformulation_need_score=_number(
            raw["reformulation_need_score"],
            label="reformulation_need_score",
        ),
    )


def _outcome_to_artifact(value: EvaluatedArmOutcome) -> dict[str, Any]:
    payload = {
        "example_id": value.example_id,
        "group_id": value.group_id,
        "state_hash": value.state_hash,
        "split": value.split,
        "benchmark_id": value.benchmark_id,
        "family": value.family.value,
        "release_id": value.release_id,
        "seed": value.seed,
        "backend_hash": value.backend_hash,
        "comparison_profile_hash": value.comparison_profile_hash,
        "status": value.status.value,
        "metrics": value.metrics,
        "cost": value.cost,
        "risk": value.risk,
    }
    payload["outcome_hash"] = content_hash(payload)
    return payload


def _outcome_from_artifact(value: object) -> EvaluatedArmOutcome:
    raw = _strict_object(
        value,
        allowed={
            "example_id",
            "group_id",
            "state_hash",
            "split",
            "benchmark_id",
            "family",
            "release_id",
            "seed",
            "backend_hash",
            "comparison_profile_hash",
            "status",
            "metrics",
            "cost",
            "risk",
            "outcome_hash",
        },
        required={
            "example_id",
            "group_id",
            "state_hash",
            "split",
            "benchmark_id",
            "family",
            "release_id",
            "seed",
            "backend_hash",
            "comparison_profile_hash",
            "status",
            "metrics",
            "cost",
            "risk",
            "outcome_hash",
        },
        label="evaluated arm outcome",
    )
    without_hash = {key: item for key, item in raw.items() if key != "outcome_hash"}
    if _sha256(raw["outcome_hash"], label="outcome_hash") != content_hash(without_hash):
        raise ValidationError("evaluated arm outcome hash mismatch")
    try:
        family = SemanticFamily(_text(raw["family"], label="outcome.family"))
        status = EvaluatedArmStatus(_text(raw["status"], label="outcome.status"))
    except ValueError as exc:
        raise ValidationError("unknown outcome family or status") from exc
    return EvaluatedArmOutcome(
        example_id=_text(raw["example_id"], label="outcome.example_id"),
        group_id=_text(raw["group_id"], label="outcome.group_id"),
        state_hash=_sha256(raw["state_hash"], label="outcome.state_hash"),
        split=_text(raw["split"], label="outcome.split"),
        benchmark_id=_text(raw["benchmark_id"], label="outcome.benchmark_id"),
        family=family,
        release_id=_text(raw["release_id"], label="outcome.release_id"),
        seed=_integer(raw["seed"], label="outcome.seed"),
        backend_hash=_sha256(raw["backend_hash"], label="outcome.backend_hash"),
        comparison_profile_hash=_sha256(
            raw["comparison_profile_hash"],
            label="outcome.comparison_profile_hash",
        ),
        status=status,
        metrics=_metrics(raw["metrics"], label="outcome.metrics", allow_empty=True),
        cost=_number(raw["cost"], label="outcome.cost"),
        risk=_number(raw["risk"], label="outcome.risk"),
    )


def _training_table_to_artifact(value: SelectorTrainingTable) -> dict[str, Any]:
    return {
        "examples": [_example_to_artifact(item) for item in value.examples],
        "representative_by_family": _encode_representatives(value.representative_by_family),
        "group_split_digests": value.group_split_digests,
        "extractor_id": value.extractor_id,
        "extractor_revision": value.extractor_revision,
        "feature_table_sha256": value.feature_table_sha256,
        "outcome_table_sha256": value.outcome_table_sha256,
        "utility_definition_sha256": value.utility_definition_sha256,
        "table_hash": value.table_hash,
    }


def _training_table_from_artifact(value: object) -> SelectorTrainingTable:
    raw = _strict_object(
        value,
        allowed={
            "examples",
            "representative_by_family",
            "group_split_digests",
            "extractor_id",
            "extractor_revision",
            "feature_table_sha256",
            "outcome_table_sha256",
            "utility_definition_sha256",
            "table_hash",
        },
        required={
            "examples",
            "representative_by_family",
            "group_split_digests",
            "extractor_id",
            "extractor_revision",
            "feature_table_sha256",
            "outcome_table_sha256",
            "utility_definition_sha256",
            "table_hash",
        },
        label="selector training table",
    )
    digests = _mapping(raw["group_split_digests"], label="group_split_digests")
    table = SelectorTrainingTable(
        examples=tuple(
            _example_from_artifact(item)
            for item in _array(raw["examples"], label="training examples")
        ),
        representative_by_family=_decode_representatives(raw["representative_by_family"]),
        group_split_digests={
            _text(split, label="group split"): _sha256(
                digest,
                label=f"group split digest {split}",
            )
            for split, digest in digests.items()
        },
        extractor_id=_text(raw["extractor_id"], label="table.extractor_id"),
        extractor_revision=_text(
            raw["extractor_revision"],
            label="table.extractor_revision",
        ),
        feature_table_sha256=_sha256(
            raw["feature_table_sha256"],
            label="feature_table_sha256",
        ),
        outcome_table_sha256=_sha256(
            raw["outcome_table_sha256"],
            label="outcome_table_sha256",
        ),
        utility_definition_sha256=_sha256(
            raw["utility_definition_sha256"],
            label="utility_definition_sha256",
        ),
    )
    if _sha256(raw["table_hash"], label="table_hash") != table.table_hash:
        raise ValidationError("selector training table hash mismatch")
    return table


def _example_to_artifact(value: SelectorExample) -> dict[str, Any]:
    return {
        "example_id": value.example_id,
        "features": _features_to_artifact(value.features),
        "utilities": _encode_family_values(value.utilities),
        "split": value.split,
        "costs": _encode_family_values(value.costs),
        "risks": _encode_family_values(value.risks),
    }


def _example_from_artifact(value: object) -> SelectorExample:
    raw = _strict_object(
        value,
        allowed={"example_id", "features", "utilities", "split", "costs", "risks"},
        required={"example_id", "features", "utilities", "split", "costs", "risks"},
        label="selector training example",
    )
    return SelectorExample(
        example_id=_text(raw["example_id"], label="example_id"),
        features=_features_from_artifact(raw["features"]),
        utilities=_decode_family_values(
            raw["utilities"],
            label="example.utilities",
            allow_no_op=True,
        ),
        split=_text(raw["split"], label="example.split"),
        costs=_decode_family_values(
            raw["costs"],
            label="example.costs",
            allow_no_op=False,
        ),
        risks=_decode_family_values(
            raw["risks"],
            label="example.risks",
            allow_no_op=False,
        ),
    )


def _normalized_selector_arm(
    value: Mapping[str, Any],
    *,
    arm_id: str,
) -> tuple[EvaluatedArmStatus, Mapping[str, float], float, bool]:
    status = _text(value.get("status"), label=f"arm {arm_id} status")
    if status == "succeeded":
        metrics = _metrics(value.get("metrics"), label=f"arm {arm_id} metrics")
        converted_status = EvaluatedArmStatus.SUCCEEDED
    elif status == "failed":
        failure = _mapping(value.get("failure"), label=f"arm {arm_id} failure")
        if failure.get("scope") != "algorithm":
            raise ArtifactError(f"failed selector arm {arm_id} must be algorithm-scoped")
        if value.get("metrics") not in (None, {}):
            raise ArtifactError(f"failed selector arm {arm_id} cannot carry evaluator metrics")
        metrics = {}
        converted_status = EvaluatedArmStatus.ALGORITHM_FAILURE
    else:
        raise ArtifactError(f"selector arm {arm_id} status must be succeeded or failed")

    usage = _mapping(value.get("usage"), label=f"arm {arm_id} usage")
    unpriced_calls = _integer(
        usage.get("unpriced_model_calls"),
        label=f"arm {arm_id} usage.unpriced_model_calls",
    )
    if unpriced_calls:
        raise ArtifactError(
            f"selector arm {arm_id} has unpriced model calls; freeze a pricing or "
            "self-hosted compute-cost policy before selector production"
        )
    total_cost = _number(
        usage.get("total_cost_usd"),
        label=f"arm {arm_id} usage.total_cost_usd",
    )
    if total_cost < 0:
        raise ArtifactError(f"selector arm {arm_id} total cost must be non-negative")
    assessment = assess_hard_harm_from_metrics(
        {
            **metrics,
            # A typed algorithm-scoped failure terminates this intervention arm
            # before a valid outcome exists and is therefore the generic
            # adapter-hard-stop component of the strict protocol.
            "adapter_hard_stops": float(converted_status is EvaluatedArmStatus.ALGORITHM_FAILURE),
        }
    )
    return converted_status, metrics, total_cost, assessment.hard_harm


def _dependency_hashes(
    value: object,
    *,
    label: str,
    expected_arm_ids: Sequence[str],
) -> Mapping[str, str]:
    """Normalize both supported execution-driver dependency receipts.

    External replay emits one common ``slot -> digest`` map.  The static
    multi-arm runner emits ``arm_id -> {slot -> digest}``; those maps are
    admissible only when every planned arm is present and all common nuisance
    dependencies are byte-for-byte identical.  The arm IDs are deliberately
    not folded into the selector backend identity.
    """

    raw = _mapping(value, label=label)
    if not raw:
        raise ArtifactError(f"{label} must not be empty")
    value_kinds = {"mapping" if isinstance(item, Mapping) else "scalar" for item in raw.values()}
    if value_kinds == {"mapping"}:
        if set(raw) != set(expected_arm_ids):
            raise ArtifactError(f"{label} arm inventory differs from the frozen experiment cell")
        by_arm = {
            _text(arm_id, label=f"{label} arm_id"): _flat_dependency_hashes(
                dependencies,
                label=f"{label}.{arm_id}",
            )
            for arm_id, dependencies in raw.items()
        }
        common = next(iter(by_arm.values()))
        if any(dependencies != common for dependencies in by_arm.values()):
            raise ArtifactError(
                f"{label} differs across arms; selector discovery requires balanced "
                "nuisance dependencies"
            )
        return common
    if value_kinds != {"scalar"}:
        raise ArtifactError(f"{label} cannot mix flat and per-arm dependency receipts")
    return _flat_dependency_hashes(raw, label=label)


def _flat_dependency_hashes(value: object, *, label: str) -> Mapping[str, str]:
    raw = _mapping(value, label=label)
    if not raw:
        raise ArtifactError(f"{label} must not be empty")
    normalized: dict[str, str] = {}
    for slot, digest in raw.items():
        name = _text(slot, label=f"{label} slot")
        normalized[name] = _sha256(digest, label=f"{label}.{name}")
    return normalized


def _selector_comparison_profile_hash(
    arm_ids: Sequence[str],
    arm_specs: Mapping[str, Any],
) -> str:
    """Derive the selector's comparison profile from the frozen manifest arms."""

    return content_hash(
        {
            "schema": "agentinputlab.selector-comparison-profile.v1",
            "arms": [
                {
                    "arm_id": arm_id,
                    "release_id": arm_specs[arm_id].release_id,
                    "family": arm_specs[arm_id].family.value,
                    "profile_kind": arm_specs[arm_id].profile_kind.value,
                    "arm_kind": arm_specs[arm_id].arm_kind.value,
                    "parameters": arm_specs[arm_id].parameters,
                }
                for arm_id in arm_ids
            ],
        }
    )


def _selector_backend_hash(
    dependencies: Mapping[str, str],
    *,
    model_slot: str,
) -> str:
    """Bind selector rows to the shared agent model, not arena infrastructure."""

    identity_slots = tuple(slot for slot in ("agent_model", "backend") if slot in dependencies)
    if len(identity_slots) != 1:
        raise ArtifactError(
            "selector dependency receipt must contain exactly one model identity "
            "slot ('agent_model' for external arenas or 'backend' for static runs)"
        )
    slot = identity_slots[0]
    return content_hash(
        {
            "schema": "agentinputlab.selector-backend.v2",
            "model_slot": model_slot,
            "identity_slot": slot,
            "identity_sha256": dependencies[slot],
        }
    )


def _metrics(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> dict[str, float]:
    raw = _mapping(value, label=label)
    metrics = {
        _text(name, label=f"{label} name"): _number(item, label=f"{label}.{name}")
        for name, item in raw.items()
    }
    if not metrics and not allow_empty:
        raise ArtifactError(f"{label} must not be empty")
    return metrics


def _family_array(value: object, *, label: str) -> frozenset[SemanticFamily]:
    raw = _array(value, label=label)
    families: list[SemanticFamily] = []
    for item in raw:
        try:
            family = SemanticFamily(_text(item, label=f"{label} item"))
        except ValueError as exc:
            raise ValidationError(f"{label} contains an unknown family") from exc
        if family not in _CORE_FAMILIES:
            raise ValidationError(f"{label} may contain core intervention families only")
        families.append(family)
    if len(families) != len(set(families)):
        raise ValidationError(f"{label} contains duplicate families")
    return frozenset(families)


def _unique_arm_for_release(
    arm_by_release: Mapping[str, list[str]],
    release_id: str,
    *,
    label: str,
) -> str:
    matches = arm_by_release.get(release_id, [])
    if len(matches) != 1:
        raise ValidationError(
            f"selector discovery manifest requires exactly one {label} arm for {release_id}"
        )
    return matches[0]


def _validate_representatives(
    value: Mapping[SemanticFamily, str],
) -> dict[SemanticFamily, str]:
    representatives = dict(value)
    if not representatives or any(
        family not in _CORE_FAMILIES
        or not isinstance(release, str)
        or not release.strip()
        or release != release.strip()
        for family, release in representatives.items()
    ):
        raise ValidationError(
            "selector discovery representatives require core families and release IDs"
        )
    if len(set(representatives.values())) != len(representatives):
        raise ValidationError("selector discovery representatives must use unique releases")
    return representatives


def _encode_representatives(
    value: Mapping[SemanticFamily, str],
) -> dict[str, str]:
    return {
        family.value: release
        for family, release in sorted(value.items(), key=lambda item: item[0].value)
    }


def _decode_representatives(value: object) -> dict[SemanticFamily, str]:
    raw = _mapping(value, label="representative_by_family")
    result: dict[SemanticFamily, str] = {}
    for name, release in raw.items():
        try:
            family = SemanticFamily(_text(name, label="representative family"))
        except ValueError as exc:
            raise ValidationError("unknown representative family") from exc
        result[family] = _text(release, label=f"representative {name}")
    return _validate_representatives(result)


def _encode_family_values(
    value: Mapping[SemanticFamily, float],
) -> dict[str, float]:
    return {
        family.value: float(item)
        for family, item in sorted(value.items(), key=lambda pair: pair[0].value)
    }


def _decode_family_values(
    value: object,
    *,
    label: str,
    allow_no_op: bool,
) -> dict[SemanticFamily, float]:
    raw = _mapping(value, label=label)
    result: dict[SemanticFamily, float] = {}
    for name, item in raw.items():
        try:
            family = SemanticFamily(_text(name, label=f"{label} family"))
        except ValueError as exc:
            raise ValidationError(f"{label} contains an unknown family") from exc
        allowed = _CORE_FAMILIES | ({SemanticFamily.NO_OP} if allow_no_op else set())
        if family not in allowed:
            raise ValidationError(f"{label} contains an unsupported family")
        result[family] = _number(item, label=f"{label}.{name}")
    return result


def _outcome_sort_key(
    row: EvaluatedArmOutcome,
) -> tuple[str, str, str, str, str, str, int]:
    return (
        row.split,
        row.group_id,
        row.state_hash,
        row.backend_hash,
        row.comparison_profile_hash,
        row.family.value,
        row.seed,
    )


def _read_json_object(path: str | Path, *, label: str) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(raw, Mapping):
        raise ArtifactError(f"{label} must be a JSON object")
    return raw


def _strict_object(
    value: object,
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> dict[str, Any]:
    raw = _mapping(value, label=label)
    unknown = set(raw) - allowed
    missing = required - set(raw)
    if unknown:
        raise ValidationError(f"{label} contains unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValidationError(f"{label} is missing fields: {', '.join(sorted(missing))}")
    return raw


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{label} must be an object with string keys")
    return dict(value)


def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a JSON array")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a string")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed non-empty string")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label=label)


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{label} must be a finite number")
    return float(value)


def _optional_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _number(value, label=label)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValidationError(f"{label} must be a lowercase SHA-256")
    return value


__all__ = [
    "SelectorDiscoveryFreeze",
    "SelectorDiscoveryIdentity",
    "SelectorDiscoverySource",
    "SelectorDiscoveryStore",
    "build_selector_discovery_freeze",
    "load_discovery_export_config",
    "parse_discovery_export_config",
]
