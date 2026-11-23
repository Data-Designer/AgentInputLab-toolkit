"""Fail-closed validation for a table's run-provenance index.

The schema binds references to checksum-verified run directories and enforces
one broad profile kind.  It does not itself prove that multiple runs share the
same model/retriever/scorer setting, bind a rendered/serialized table,
authenticate the process that produced a run, or establish scientific validity.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentinputlab.contracts.enums import ProfileKind
from agentinputlab.contracts.errors import RegistryError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_text(data: Mapping[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{path}.{key} must be a non-empty string")
    return value


def _profile_kind(value: Any, path: str) -> ProfileKind:
    if not isinstance(value, str):
        raise RegistryError(f"{path} must be a string")
    try:
        return ProfileKind(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ProfileKind)
        raise RegistryError(f"{path} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class ResultStudyReference:
    """One checksum-consistent run reference contributing rows to a table.

    These are operator-supplied references.  The beta gate re-opens ``run_path``
    with :func:`verify_experiment_run_directory` and requires every identity to
    match the run artifact exactly.  This detects mutation or substitution
    relative to those identities; it is not proof that an external model,
    scorer, or worker actually executed as claimed.
    """

    study_id: str
    profile_kind: ProfileKind
    profile_hash: str
    run_path: str
    run_identity_hash: str
    manifest_hash: str
    plan_hash: str
    driver_hash: str

    @classmethod
    def from_mapping(cls, raw: Any, path: str) -> ResultStudyReference:
        required = {
            "study_id",
            "profile_kind",
            "profile_hash",
            "run_path",
            "run_identity_hash",
            "manifest_hash",
            "plan_hash",
            "driver_hash",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise RegistryError(f"{path} fields must be exactly: {', '.join(sorted(required))}")
        digests: dict[str, str] = {}
        for key in (
            "profile_hash",
            "run_identity_hash",
            "manifest_hash",
            "plan_hash",
            "driver_hash",
        ):
            value = _required_text(raw, key, path)
            if not _SHA256.fullmatch(value):
                raise RegistryError(f"{path}.{key} must be a lowercase SHA-256 digest")
            digests[key] = value
        return cls(
            study_id=_required_text(raw, "study_id", path),
            profile_kind=_profile_kind(raw.get("profile_kind"), f"{path}.profile_kind"),
            profile_hash=digests["profile_hash"],
            run_path=_required_text(raw, "run_path", path),
            run_identity_hash=digests["run_identity_hash"],
            manifest_hash=digests["manifest_hash"],
            plan_hash=digests["plan_hash"],
            driver_hash=digests["driver_hash"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "study_id": self.study_id,
            "profile_kind": self.profile_kind.value,
            "profile_hash": self.profile_hash,
            "run_path": self.run_path,
            "run_identity_hash": self.run_identity_hash,
            "manifest_hash": self.manifest_hash,
            "plan_hash": self.plan_hash,
            "driver_hash": self.driver_hash,
        }


@dataclass(frozen=True, slots=True)
class ResultTableManifest:
    """Run-provenance index and profile boundary for one prospective table.

    Version 2 has no table-artifact path/hash or row-derivation mapping, so a
    separately rendered table is not authenticated by this object alone.
    """

    table_id: str
    profile_kind: ProfileKind
    studies: tuple[ResultStudyReference, ...]
    schema_version: int = 2

    @classmethod
    def from_mapping(cls, raw: Any) -> ResultTableManifest:
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema_version",
            "table_id",
            "profile_kind",
            "studies",
        }:
            raise RegistryError("result_table_manifest must be an object")
        if raw.get("schema_version") != 2:
            raise RegistryError("result_table_manifest.schema_version must equal 2")
        raw_studies = raw.get("studies")
        if not isinstance(raw_studies, list) or not raw_studies:
            raise RegistryError("result_table_manifest.studies must be a non-empty array")
        studies = tuple(
            ResultStudyReference.from_mapping(
                item,
                f"result_table_manifest.studies[{index}]",
            )
            for index, item in enumerate(raw_studies)
        )
        study_ids = [item.study_id for item in studies]
        if len(study_ids) != len(set(study_ids)):
            raise RegistryError("result_table_manifest.studies contains duplicate study IDs")

        declared_kind = _profile_kind(
            raw.get("profile_kind"),
            "result_table_manifest.profile_kind",
        )
        observed_kinds = {item.profile_kind for item in studies}
        if observed_kinds != {declared_kind}:
            rendered = ", ".join(sorted(item.value for item in observed_kinds))
            raise RegistryError(
                "one result table must not mix official_native, controlled_adapter, "
                f"or factorial_adapter profiles; declared={declared_kind.value}, "
                f"observed={rendered}"
            )
        return cls(
            table_id=_required_text(raw, "table_id", "result_table_manifest"),
            profile_kind=declared_kind,
            studies=studies,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "table_id": self.table_id,
            "profile_kind": self.profile_kind.value,
            "studies": [item.to_dict() for item in self.studies],
        }


def validate_result_table_manifest(raw: Any) -> ResultTableManifest:
    """Validate an immutable, single-profile-kind run-provenance index."""

    return ResultTableManifest.from_mapping(raw)


__all__ = [
    "ResultStudyReference",
    "ResultTableManifest",
    "validate_result_table_manifest",
]
