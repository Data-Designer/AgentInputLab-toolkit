"""Artifact and upstream-source identity schemas."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.enums import StrEnum
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import freeze_mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactKind(StrEnum):
    DATASET = "dataset"
    MODEL = "model"
    INDEX = "index"
    CACHE = "cache"
    FEATURE = "feature"
    PROMPT = "prompt"
    TRACE = "trace"
    RECIPE_OUTPUT = "recipe_output"


class AcquisitionPolicy(StrEnum):
    BUNDLED = "bundled"
    DOWNLOAD = "download"
    USER_PROVIDED = "user_provided"
    GATED = "gated"
    EXTERNAL_MOUNT = "external_mount"


@dataclass(frozen=True, slots=True)
class UpstreamLock:
    repository: str
    commit: str
    license_id: str
    vendoring_allowed: bool
    subpath: str = ""
    branch: str = ""

    def __post_init__(self) -> None:
        if self.commit and not _GIT_SHA.fullmatch(self.commit):
            raise ValidationError("upstream git commit must be a full 40-character SHA")
        if self.license_id == "NOASSERTION" and self.vendoring_allowed:
            raise ValidationError("NOASSERTION source cannot be marked vendoring_allowed")


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifact_id: str
    kind: ArtifactKind
    revision: str
    sha256: str
    acquisition: AcquisitionPolicy
    license_id: str
    source_uri: str = ""
    size_bytes: int | None = None
    lineage: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SAFE_SEGMENT.fullmatch(self.artifact_id) or ".." in self.artifact_id:
            raise ValidationError("artifact_id must be a flat safe identifier")
        if not _SAFE_SEGMENT.fullmatch(self.revision) or ".." in self.revision:
            raise ValidationError("artifact revision must be a flat safe identifier")
        if self.sha256 and not _SHA256.fullmatch(self.sha256):
            raise ValidationError("artifact sha256 must contain 64 lowercase hex characters")
        if self.acquisition is AcquisitionPolicy.DOWNLOAD:
            if not self.sha256:
                raise ValidationError("download artifacts require a pinned sha256")
            if not self.source_uri:
                raise ValidationError("download artifacts require a source_uri")
        if self.size_bytes is not None and (type(self.size_bytes) is not int or self.size_bytes < 0):
            raise ValidationError("artifact size must be a non-negative integer")
        lineage = dict(self.lineage)
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in lineage.items()
        ):
            raise ValidationError("artifact lineage requires non-empty string keys and values")
        try:
            frozen_lineage = freeze_mapping(lineage)
            frozen_metadata = freeze_mapping(self.metadata)
        except TypeError as exc:
            raise ValidationError(
                f"artifact metadata must be canonically serializable: {exc}"
            ) from exc
        object.__setattr__(self, "lineage", frozen_lineage)
        object.__setattr__(self, "metadata", frozen_metadata)
