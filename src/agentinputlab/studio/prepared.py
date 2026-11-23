"""Verified prepared-benchmark loading for the local Studio.

This module never downloads data and never accepts paths from a browser
request.  The server process supplies one prepared-data root, the benchmark
layer re-verifies the pinned cache, and only then is a bounded deterministic
slice projected into Studio execution objects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.backends.retrieval import Document
from agentinputlab.benchmarks import (
    BenchmarkSmokeAsset,
    BenchmarkSmokeAssetRegistry,
    PreparedBenchmark,
    load_prepared_benchmark,
    production_benchmark_specs,
)
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import canonical_json

SUPPORTED_PREPARED_CELLS = frozenset({"beir.arguana"})
_PRODUCTION_DATA_SPECS = {
    spec.cell_id: spec
    for spec in production_benchmark_specs()
    if spec.adapter_kind in {"qa", "retrieval"}
}
PRODUCTION_DATA_CELLS = frozenset(_PRODUCTION_DATA_SPECS)
MAX_PREPARED_SAMPLE_LIMIT = 4


@dataclass(frozen=True, slots=True)
class PreparedStudioExample:
    """Optimizer-visible input paired with evaluator-only relevance labels."""

    example_id: str
    query: str
    instruction: str
    visible_metadata: Mapping[str, Any]
    relevant_document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "visible_metadata", MappingProxyType(dict(self.visible_metadata)))

    @property
    def primary_relevant_document_id(self) -> str | None:
        return self.relevant_document_ids[0] if self.relevant_document_ids else None


@dataclass(frozen=True, slots=True)
class PreparedStudioDataset:
    cell_id: str
    asset_id: str
    source_sha256: str
    processor: str
    processor_revision: str
    counts: Mapping[str, int]
    file_sha256: Mapping[str, str]
    examples: tuple[PreparedStudioExample, ...]
    documents: tuple[Document, ...]
    selection_rule: str
    selection_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "file_sha256", MappingProxyType(dict(self.file_sha256)))

    def audit_payload(self) -> dict[str, Any]:
        """Return cache identity without disclosing the server filesystem path."""

        return {
            "cell_id": self.cell_id,
            "asset_id": self.asset_id,
            "source_sha256": self.source_sha256,
            "processor": self.processor,
            "processor_revision": self.processor_revision,
            "counts": dict(self.counts),
            "file_sha256": dict(self.file_sha256),
            "gold_separated": True,
            "selection_rule": self.selection_rule,
            "selection_fingerprint": self.selection_fingerprint,
            "selected_examples": len(self.examples),
        }


@dataclass(frozen=True, slots=True)
class PreparedProductionCell:
    """Metadata-only proof that every locked source component is prepared."""

    cell_id: str
    split: str
    adapter_kind: str
    assets: Mapping[str, PreparedBenchmark]

    def __post_init__(self) -> None:
        if self.cell_id not in PRODUCTION_DATA_CELLS:
            raise ValueError("prepared production cell is not a stateless production cell")
        if self.adapter_kind not in {"qa", "retrieval"}:
            raise ValueError("prepared production cell adapter_kind must be qa or retrieval")
        spec = _PRODUCTION_DATA_SPECS[self.cell_id]
        if self.split != spec.split or self.adapter_kind != spec.adapter_kind:
            raise ValueError("prepared production cell identity differs from its locked spec")
        assets = dict(self.assets)
        if not assets:
            raise ValueError("prepared production cell requires at least one asset")
        expected_asset_ids = {spec.query_asset_id}
        if spec.corpus_asset_id is not None:
            expected_asset_ids.add(spec.corpus_asset_id)
        if set(assets) != expected_asset_ids:
            raise ValueError("prepared production cell does not contain its complete asset set")
        if any(prepared.cell_id != self.cell_id for prepared in assets.values()):
            raise ValueError("prepared production asset cell identity differs from its cell")
        if any(asset_id != prepared.asset_id for asset_id, prepared in assets.items()):
            raise ValueError("prepared production asset key differs from its asset identity")
        object.__setattr__(self, "assets", MappingProxyType(assets))

    def audit_payload(self) -> dict[str, Any]:
        """Expose hashes and counts without loading rows or disclosing paths."""

        return {
            "cell_id": self.cell_id,
            "split": self.split,
            "adapter_kind": self.adapter_kind,
            "complete_asset_set": True,
            "gold_separated": True,
            "assets": {
                asset_id: {
                    "source_sha256": prepared.source_sha256,
                    "processor": prepared.processor,
                    "processor_revision": prepared.processor_revision,
                    "counts": dict(prepared.counts),
                    "file_sha256": dict(prepared.file_sha256),
                }
                for asset_id, prepared in sorted(self.assets.items())
            },
        }


def discover_prepared_production_cells(
    prepared_root: str | Path | None,
) -> dict[str, PreparedProductionCell]:
    """Verify complete asset sets for the seven stateless production cells.

    This performs cache integrity checks only.  It deliberately does not
    construct adapters, load full QA rows, build a BRIGHT index, or claim that
    Studio has a downstream predictor for the cell.
    """

    if prepared_root is None:
        return {}
    root = Path(prepared_root).expanduser().resolve()
    registry = BenchmarkSmokeAssetRegistry.default()
    discovered: dict[str, PreparedProductionCell] = {}
    for spec in production_benchmark_specs():
        if spec.cell_id not in PRODUCTION_DATA_CELLS:
            continue
        asset_ids = (
            (spec.query_asset_id,)
            if spec.corpus_asset_id is None
            else (spec.query_asset_id, spec.corpus_asset_id)
        )
        try:
            prepared_assets = {
                asset_id: load_prepared_benchmark(registry.get(asset_id), root)
                for asset_id in asset_ids
            }
        except (ArtifactError, OSError):
            continue
        discovered[spec.cell_id] = PreparedProductionCell(
            cell_id=spec.cell_id,
            split=spec.split,
            adapter_kind=spec.adapter_kind,
            assets=prepared_assets,
        )
    return discovered


def discover_prepared_benchmarks(
    prepared_root: str | Path | None,
) -> dict[str, PreparedBenchmark]:
    """Discover only checksum-verified, license-cleared smoke assets.

    Missing or tampered caches are intentionally treated as unavailable.  The
    detailed filesystem error is not put in the browser catalog because it can
    reveal local paths.
    """

    if prepared_root is None:
        return {}
    root = Path(prepared_root).expanduser().resolve()
    discovered: dict[str, PreparedBenchmark] = {}
    for asset in BenchmarkSmokeAssetRegistry.default().list():
        if asset.cell_id not in SUPPORTED_PREPARED_CELLS:
            continue
        try:
            discovered[asset.cell_id] = load_prepared_benchmark(asset, root)
        except (ArtifactError, OSError):
            continue
    return discovered


def load_verified_studio_dataset(
    *,
    cell_id: str,
    prepared_root: str | Path,
    sample_limit: int,
) -> PreparedStudioDataset:
    """Re-verify a known cache and load a deterministic bounded sample."""

    if cell_id not in SUPPORTED_PREPARED_CELLS:
        raise ArtifactError(f"Studio has no prepared execution adapter for {cell_id}")
    asset = _asset_for_cell(cell_id)
    prepared = load_prepared_benchmark(asset, prepared_root)
    return project_prepared_benchmark(asset, prepared, sample_limit=sample_limit)


def project_prepared_benchmark(
    asset: BenchmarkSmokeAsset,
    prepared: PreparedBenchmark,
    *,
    sample_limit: int,
) -> PreparedStudioDataset:
    """Project an already verified cache into gold-separated runtime objects."""

    if asset.cell_id not in SUPPORTED_PREPARED_CELLS:
        raise ArtifactError(f"Studio has no prepared execution adapter for {asset.cell_id}")
    if prepared.asset_id != asset.asset_id or prepared.source_sha256 != asset.sha256:
        raise ArtifactError("prepared benchmark identity does not match its audited asset")
    if not 1 <= sample_limit <= MAX_PREPARED_SAMPLE_LIMIT:
        raise ValueError(
            f"prepared Studio sample_limit must be in [1, {MAX_PREPARED_SAMPLE_LIMIT}]"
        )
    corpus_path = prepared.corpus_path
    if corpus_path is None:
        raise ArtifactError(f"prepared benchmark has no retrieval corpus: {asset.asset_id}")

    inputs = _records_by_id(prepared.inputs_path, label="prepared inputs")
    gold = _records_by_id(prepared.gold_path, label="prepared gold")
    if set(inputs) != set(gold):
        raise ArtifactError("prepared input and gold example IDs do not match")

    selected_ids = tuple(sorted(inputs)[:sample_limit])
    examples: list[PreparedStudioExample] = []
    for example_id in selected_ids:
        input_row = inputs[example_id]
        gold_row = gold[example_id]
        query = _required_text(input_row, "query", label="prepared input")
        instruction = _required_text(input_row, "instruction", label="prepared input")
        visible_metadata = input_row.get("visible_metadata", {})
        if not isinstance(visible_metadata, dict):
            raise ArtifactError("prepared input visible_metadata must be an object")
        relevance = gold_row.get("relevance")
        if not isinstance(relevance, dict):
            raise ArtifactError("prepared retrieval gold must contain a relevance object")
        relevant_ids = tuple(
            sorted(
                document_id
                for document_id, score in relevance.items()
                if isinstance(document_id, str)
                and document_id
                and isinstance(score, int)
                and not isinstance(score, bool)
                and score > 0
            )
        )
        if not relevant_ids:
            raise ArtifactError(f"prepared example {example_id!r} has no positive relevance label")
        examples.append(
            PreparedStudioExample(
                example_id=example_id,
                query=query,
                instruction=instruction,
                visible_metadata=visible_metadata,
                relevant_document_ids=relevant_ids,
            )
        )

    revision = f"{asset.asset_id}:{prepared.source_sha256}"
    documents: list[Document] = []
    for row in _read_jsonl(corpus_path, label="prepared corpus"):
        document_id = _required_text(row, "id", label="prepared corpus")
        documents.append(
            Document(
                doc_id=document_id,
                title=_optional_text(row, "title", label="prepared corpus"),
                text=_required_text(row, "text", label="prepared corpus"),
                corpus_revision=revision,
            )
        )
    if len(documents) != prepared.counts.get("corpus"):
        raise ArtifactError("prepared corpus count changed after cache verification")

    selection_rule = "lexicographic_example_id_first_n"
    selection_fingerprint = hashlib.sha256(
        canonical_json(
            {
                "asset_id": asset.asset_id,
                "sample_limit": sample_limit,
                "selection_rule": selection_rule,
                "selected_ids": selected_ids,
            }
        ).encode("utf-8")
    ).hexdigest()
    return PreparedStudioDataset(
        cell_id=asset.cell_id,
        asset_id=asset.asset_id,
        source_sha256=prepared.source_sha256,
        processor=prepared.processor,
        processor_revision=prepared.processor_revision,
        counts=prepared.counts,
        file_sha256=prepared.file_sha256,
        examples=tuple(examples),
        documents=tuple(documents),
        selection_rule=selection_rule,
        selection_fingerprint=selection_fingerprint,
    )


def prepared_catalog_payload(
    asset: BenchmarkSmokeAsset,
    prepared: PreparedBenchmark,
) -> dict[str, Any]:
    """Summarize a verified cache for the UI without local path disclosure."""

    return {
        "asset_id": asset.asset_id,
        "source_sha256": prepared.source_sha256,
        "processor": prepared.processor,
        "processor_revision": prepared.processor_revision,
        "counts": dict(prepared.counts),
        "file_sha256": dict(prepared.file_sha256),
        "gold_separated": True,
        "sample_limit": MAX_PREPARED_SAMPLE_LIMIT,
    }


def _asset_for_cell(cell_id: str) -> BenchmarkSmokeAsset:
    matches = [
        asset for asset in BenchmarkSmokeAssetRegistry.default().list() if asset.cell_id == cell_id
    ]
    if len(matches) != 1:
        raise ArtifactError(
            f"expected one verified smoke asset for {cell_id}, found {len(matches)}"
        )
    return matches[0]


def _records_by_id(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path, label=label):
        row_id = _required_text(row, "id", label=label)
        if row_id in records:
            raise ArtifactError(f"{label} contains duplicate ID {row_id!r}")
        records[row_id] = row
    return records


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise ArtifactError(f"{label} contains a blank line at {line_number}")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ArtifactError(f"{label} record {line_number} must be an object")
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not valid UTF-8 JSONL") from exc
    if not records:
        raise ArtifactError(f"{label} contains no records")
    return tuple(records)


def _required_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} field {key!r} must be a non-empty string")
    return value


def _optional_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = record.get(key, "")
    if not isinstance(value, str):
        raise ArtifactError(f"{label} field {key!r} must be a string")
    return value


__all__ = [
    "MAX_PREPARED_SAMPLE_LIMIT",
    "PRODUCTION_DATA_CELLS",
    "SUPPORTED_PREPARED_CELLS",
    "PreparedProductionCell",
    "PreparedStudioDataset",
    "PreparedStudioExample",
    "discover_prepared_benchmarks",
    "discover_prepared_production_cells",
    "load_verified_studio_dataset",
    "prepared_catalog_payload",
    "project_prepared_benchmark",
]
