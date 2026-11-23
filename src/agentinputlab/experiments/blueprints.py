"""Materialize compact study blueprints against one content-bound arm catalog."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .schema import ExperimentManifest

_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_GIT_OBJECT_ID_LENGTHS = frozenset({40, 64})


def load_experiment_blueprint(
    blueprint_path: str | Path,
    *,
    arm_catalog_path: str | Path,
    environment: Mapping[str, str] | None = None,
) -> ExperimentManifest:
    blueprint = _read_object(blueprint_path, label="experiment blueprint")
    catalog = _read_object(arm_catalog_path, label="experiment arm catalog")
    if set(blueprint) != {"schema_version", "arm_set", "manifest"}:
        raise ValidationError("experiment blueprint fields are invalid")
    if blueprint["schema_version"] != 1:
        raise ValidationError("unsupported experiment blueprint schema")
    arm_set_id = blueprint["arm_set"]
    if not isinstance(arm_set_id, str) or not arm_set_id.strip():
        raise ValidationError("experiment blueprint arm_set must be non-empty")
    if set(catalog) != {"schema_version", "arm_sets"} or catalog["schema_version"] != 1:
        raise ValidationError("unsupported experiment arm catalog schema")
    arm_sets = catalog["arm_sets"]
    if not isinstance(arm_sets, Mapping):
        raise ValidationError("experiment arm_sets must be an object")
    try:
        arms = arm_sets[arm_set_id]
    except KeyError as exc:
        raise ValidationError(f"unknown experiment arm set: {arm_set_id}") from exc
    manifest = blueprint["manifest"]
    if not isinstance(manifest, Mapping):
        raise ValidationError("experiment blueprint manifest must be an object")
    if "arms" in manifest:
        raise ValidationError("experiment blueprint cannot override catalog arms")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValidationError("experiment blueprint metadata must be an object")
    metadata = dict(metadata)
    manifest_fields = dict(manifest)
    benchmarks = manifest_fields.get("benchmarks")
    env = os.environ if environment is None else environment
    revision_environment = metadata.get("source_revision_placeholder_must_match_env")
    if revision_environment is not None:
        if not isinstance(revision_environment, str) or not revision_environment:
            raise ValidationError(
                "source_revision_placeholder_must_match_env must be a non-empty "
                "environment-variable name"
            )
        revision = env.get(revision_environment)
        if revision:
            if not isinstance(benchmarks, list):
                raise ValidationError("experiment blueprint benchmarks must be an array")
            bound_benchmarks: list[Any] = []
            for target in benchmarks:
                if not isinstance(target, Mapping):
                    raise ValidationError("experiment blueprint benchmark targets must be objects")
                bound_target = dict(target)
                if bound_target.get("external") is True:
                    bound_target["source_revision"] = revision
                bound_benchmarks.append(bound_target)
            manifest_fields["benchmarks"] = bound_benchmarks

    revision_bindings = metadata.get("checkout_expected_revision_from_env")
    if revision_bindings is not None:
        if not isinstance(revision_bindings, Mapping) or not revision_bindings:
            raise ValidationError("checkout_expected_revision_from_env must be a non-empty object")
        literal_expectations = metadata.get("checkout_expected_revisions", {})
        if not isinstance(literal_expectations, Mapping):
            raise ValidationError("checkout_expected_revisions must be an object")
        materialized_expectations = dict(literal_expectations)
        for checkout_environment, source_revision_environment in revision_bindings.items():
            if (
                not isinstance(checkout_environment, str)
                or _ENVIRONMENT_NAME.fullmatch(checkout_environment) is None
                or not isinstance(source_revision_environment, str)
                or _ENVIRONMENT_NAME.fullmatch(source_revision_environment) is None
            ):
                raise ValidationError(
                    "checkout_expected_revision_from_env must map environment-variable "
                    "names to environment-variable names"
                )
            revision = env.get(source_revision_environment)
            if not revision:
                continue
            _require_git_object_id(
                revision,
                label=f"environment {source_revision_environment}",
            )
            existing = materialized_expectations.get(checkout_environment)
            if existing is not None and existing != revision:
                raise ValidationError(
                    f"checkout revision for {checkout_environment} conflicts with "
                    "checkout_expected_revisions"
                )
            materialized_expectations[checkout_environment] = revision
        if materialized_expectations:
            metadata["checkout_expected_revisions"] = materialized_expectations

    merged: dict[str, Any] = {
        **manifest_fields,
        "arms": arms,
        "metadata": {
            **metadata,
            "blueprint_hash": content_hash(blueprint),
            "arm_set_id": arm_set_id,
            "arm_set_hash": content_hash({"arm_set_id": arm_set_id, "arms": arms}),
        },
    }
    return ExperimentManifest.from_mapping(merged)


def _require_git_object_id(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in _GIT_OBJECT_ID_LENGTHS
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{label} must be a full lowercase Git object ID")


def _read_object(path: str | Path, *, label: str) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return value


__all__ = ["load_experiment_blueprint"]
