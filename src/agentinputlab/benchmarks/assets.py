"""Versioned, checksum-pinned benchmark smoke assets.

The benchmark inventory describes experimental coverage.  This module
is intentionally separate: it lists only source files that have been fetched,
byte-verified, and assigned a deterministic preprocessing recipe.  A registry
cell therefore does not imply that its full data is already materializable.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any

from agentinputlab.artifacts import AcquisitionPolicy, ArtifactKind, ArtifactManifest
from agentinputlab.contracts.errors import RegistryError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_KNOWN_STATUSES = {"verified_smoke", "license_review_required"}
_KNOWN_REDISTRIBUTION = {"downloader_only", "not_redistributed"}
_KNOWN_PROCESSORS = {
    "beir_zip_v1",
    "bfcl_v3_simple_tar_v1",
    "bright_corpus_parquet_v1",
    "bright_examples_parquet_v1",
    "clariq_tsv_v1",
    "hotpotqa_parquet_v1",
    "musique_zip_v1",
    "qasper_tgz_v1",
    "tau3_banking_tar_v1",
    "two_wiki_zip_v1",
}


def _string_mapping(value: Any, *, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{label} must be a mapping")
    converted: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise RegistryError(f"{label} keys and values must be strings")
        converted[key] = item
    return converted


def _count_mapping(value: Any, *, label: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{label} must be a mapping")
    converted: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, int) or isinstance(item, bool):
            raise RegistryError(f"{label} keys must be strings and values must be integers")
        if item < 0:
            raise RegistryError(f"{label} values must be non-negative")
        converted[key] = item
    return converted


@dataclass(frozen=True, slots=True)
class BenchmarkSmokeAsset:
    asset_id: str
    cell_id: str
    source_url: str
    source_revision: str
    sha256: str
    size_bytes: int
    license_id: str
    redistribution: str
    status: str
    processor: str
    expected_counts: Mapping[str, int]
    prepared_sha256: Mapping[str, str]
    allowed_hosts: tuple[str, ...]
    provenance: Mapping[str, str]
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_hosts",
            tuple(sorted({host.strip().lower() for host in self.allowed_hosts if host.strip()})),
        )
        object.__setattr__(self, "expected_counts", MappingProxyType(dict(self.expected_counts)))
        object.__setattr__(self, "prepared_sha256", MappingProxyType(dict(self.prepared_sha256)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        parsed = urllib.parse.urlsplit(self.source_url)
        if not _SAFE_SEGMENT.fullmatch(self.asset_id) or ".." in self.asset_id:
            raise RegistryError("benchmark asset_id must be a flat safe identifier")
        if not _SAFE_SEGMENT.fullmatch(self.source_revision) or ".." in self.source_revision:
            raise RegistryError("benchmark source_revision must be a flat safe identifier")
        if parsed.scheme != "https" or not parsed.hostname:
            raise RegistryError(f"benchmark asset {self.asset_id!r} requires an HTTPS source")
        if parsed.username is not None or parsed.password is not None:
            raise RegistryError("benchmark asset URLs must not contain user information")
        if not _SHA256.fullmatch(self.sha256):
            raise RegistryError(f"benchmark asset {self.asset_id!r} has an invalid SHA-256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise RegistryError(f"benchmark asset {self.asset_id!r} size must be an integer")
        if self.size_bytes < 1:
            raise RegistryError(f"benchmark asset {self.asset_id!r} has an invalid size")
        if self.status not in _KNOWN_STATUSES:
            raise RegistryError(f"benchmark asset {self.asset_id!r} has an unknown status")
        if self.redistribution not in _KNOWN_REDISTRIBUTION:
            raise RegistryError(
                f"benchmark asset {self.asset_id!r} has an unknown redistribution policy"
            )
        if self.processor not in _KNOWN_PROCESSORS:
            raise RegistryError(f"benchmark asset {self.asset_id!r} has an unknown processor")
        if any(not _SHA256.fullmatch(digest) for digest in self.prepared_sha256.values()):
            raise RegistryError(
                f"benchmark asset {self.asset_id!r} has an invalid prepared SHA-256"
            )
        if not set(self.prepared_sha256) <= {
            "audit.jsonl",
            "inputs.jsonl",
            "gold.jsonl",
            "corpus.jsonl",
        }:
            raise RegistryError(f"benchmark asset {self.asset_id!r} has unknown prepared files")
        if not self.allowed_hosts or parsed.hostname.lower() not in self.allowed_hosts:
            raise RegistryError(
                f"benchmark asset {self.asset_id!r} must allowlist its source hostname"
            )
        if self.license_id == "NOASSERTION" and self.status != "license_review_required":
            raise RegistryError("NOASSERTION benchmark assets must remain license-review gated")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkSmokeAsset:
        try:
            hosts = value["allowed_hosts"]
            if not isinstance(hosts, list) or not all(isinstance(item, str) for item in hosts):
                raise RegistryError("benchmark asset allowed_hosts must be a string list")
            expected_counts = _count_mapping(
                value["expected_counts"], label="benchmark asset expected_counts"
            )
            provenance = _string_mapping(
                value.get("provenance", {}), label="benchmark asset provenance"
            )
            prepared_sha256 = _string_mapping(
                value.get("prepared_sha256", {}), label="benchmark asset prepared_sha256"
            )
            return cls(
                asset_id=str(value["asset_id"]),
                cell_id=str(value["cell_id"]),
                source_url=str(value["source_url"]),
                source_revision=str(value["source_revision"]),
                sha256=str(value["sha256"]),
                size_bytes=int(value["size_bytes"]),
                license_id=str(value["license_id"]),
                redistribution=str(value["redistribution"]),
                status=str(value["status"]),
                processor=str(value["processor"]),
                expected_counts=expected_counts,
                prepared_sha256=prepared_sha256,
                allowed_hosts=tuple(sorted({item.lower() for item in hosts})),
                provenance=provenance,
                note=str(value.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"invalid benchmark smoke asset: {value!r}") from exc

    def artifact_manifest(self) -> ArtifactManifest:
        """Project the source lock into the shared artifact-store contract."""

        return ArtifactManifest(
            artifact_id=self.asset_id,
            kind=ArtifactKind.DATASET,
            revision=self.source_revision,
            sha256=self.sha256,
            acquisition=AcquisitionPolicy.DOWNLOAD,
            license_id=self.license_id,
            source_uri=self.source_url,
            size_bytes=self.size_bytes,
            lineage={"benchmark_cell": self.cell_id, "processor": self.processor},
            metadata={
                "redistribution": self.redistribution,
                "status": self.status,
                "provenance": dict(self.provenance),
                "prepared_sha256": dict(self.prepared_sha256),
            },
        )


class BenchmarkSmokeAssetRegistry:
    def __init__(self, assets: tuple[BenchmarkSmokeAsset, ...]) -> None:
        if not assets:
            raise RegistryError("benchmark smoke asset registry must not be empty")
        if len({asset.asset_id for asset in assets}) != len(assets):
            raise RegistryError("benchmark smoke asset IDs must be unique")
        self._assets = assets
        self._by_id = {asset.asset_id: asset for asset in assets}

    @classmethod
    def default(cls) -> BenchmarkSmokeAssetRegistry:
        resource = resources.files("agentinputlab.benchmarks").joinpath("data", "smoke_assets.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise RegistryError("unsupported benchmark smoke asset registry snapshot")
        raw_assets = payload.get("assets")
        if not isinstance(raw_assets, list) or payload.get("asset_count") != len(raw_assets):
            raise RegistryError("benchmark smoke asset count does not match its payload")
        return cls(tuple(BenchmarkSmokeAsset.from_mapping(item) for item in raw_assets))

    def get(self, asset_id: str) -> BenchmarkSmokeAsset:
        try:
            return self._by_id[asset_id]
        except KeyError as exc:
            raise RegistryError(f"unknown benchmark smoke asset: {asset_id}") from exc

    def list(
        self,
        *,
        include_license_review: bool = False,
    ) -> tuple[BenchmarkSmokeAsset, ...]:
        if include_license_review:
            return self._assets
        return tuple(asset for asset in self._assets if asset.status == "verified_smoke")

    def __len__(self) -> int:
        return len(self._assets)
