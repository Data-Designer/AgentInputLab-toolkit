"""Locked constructors for checksum-verified production benchmark cells."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agentinputlab.contracts.errors import ArtifactError, ValidationError

from .assets import BenchmarkSmokeAsset, BenchmarkSmokeAssetRegistry
from .preprocess import load_prepared_benchmark
from .qa import PreparedQABenchmarkAdapter
from .retrieval import PreparedRetrievalBenchmarkAdapter
from .tau3 import (
    PreparedTau3BankingBundle,
    Tau3BankingCheckpoint,
    Tau3BankingCheckpointAdapter,
    Tau3EvaluationBridge,
)


@dataclass(frozen=True, slots=True)
class ProductionBenchmarkSpec:
    """One public cell bound to exact prepared asset identities."""

    cell_id: str
    split: str
    adapter_kind: str
    query_asset_id: str
    corpus_asset_id: str | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.cell_id, self.split, self.adapter_kind, self.query_asset_id)
        ):
            raise ValueError("production benchmark spec identities must be non-empty strings")
        if self.adapter_kind not in {"checkpoint", "qa", "retrieval"}:
            raise ValueError(
                "production benchmark adapter_kind must be checkpoint, qa, or retrieval"
            )
        if self.corpus_asset_id is not None and (
            not isinstance(self.corpus_asset_id, str) or not self.corpus_asset_id.strip()
        ):
            raise ValueError("production benchmark corpus_asset_id must be non-empty")
        if self.adapter_kind == "qa" and self.corpus_asset_id is not None:
            raise ValueError("production QA specs cannot declare a corpus asset")
        if self.adapter_kind == "checkpoint" and self.corpus_asset_id is not None:
            raise ValueError("production checkpoint specs cannot declare a corpus asset")


_PRODUCTION_SPECS = (
    ProductionBenchmarkSpec(
        cell_id="beir.arguana",
        split="test",
        adapter_kind="retrieval",
        query_asset_id="beir.arguana.official",
    ),
    ProductionBenchmarkSpec(
        cell_id="bright.earth_science",
        split="test",
        adapter_kind="retrieval",
        query_asset_id="bright.earth_science.examples",
        corpus_asset_id="bright.earth_science.corpus",
    ),
    ProductionBenchmarkSpec(
        cell_id="bright.economics",
        split="test",
        adapter_kind="retrieval",
        query_asset_id="bright.economics.examples",
        corpus_asset_id="bright.economics.corpus",
    ),
    ProductionBenchmarkSpec(
        cell_id="bright.robotics",
        split="test",
        adapter_kind="retrieval",
        query_asset_id="bright.robotics.examples",
        corpus_asset_id="bright.robotics.corpus",
    ),
    ProductionBenchmarkSpec(
        cell_id="bright.stackoverflow",
        split="test",
        adapter_kind="retrieval",
        query_asset_id="bright.stackoverflow.examples",
        corpus_asset_id="bright.stackoverflow.corpus",
    ),
    ProductionBenchmarkSpec(
        cell_id="musique.answerable",
        split="dev",
        adapter_kind="qa",
        query_asset_id="musique.answerable.dev",
    ),
    ProductionBenchmarkSpec(
        cell_id="2wiki.multihop",
        split="dev",
        adapter_kind="qa",
        query_asset_id="2wiki.multihop.dev",
    ),
    ProductionBenchmarkSpec(
        cell_id="hotpotqa.distractor",
        split="validation",
        adapter_kind="qa",
        query_asset_id="hotpotqa.distractor.validation",
    ),
    ProductionBenchmarkSpec(
        cell_id="qasper.v0_3_dev",
        split="dev",
        adapter_kind="qa",
        query_asset_id="qasper.dev.official",
    ),
    ProductionBenchmarkSpec(
        cell_id="tau3.banking_knowledge",
        split="base",
        adapter_kind="checkpoint",
        query_asset_id="tau3.banking_knowledge.v1_0_1",
    ),
)
_PRODUCTION_BY_CELL = {spec.cell_id: spec for spec in _PRODUCTION_SPECS}


def _require_registered_identity(
    asset: BenchmarkSmokeAsset,
    *,
    expected_asset_id: str,
    expected_cell_id: str,
    role: str,
) -> None:
    if asset.asset_id != expected_asset_id or asset.cell_id != expected_cell_id:
        raise ArtifactError(f"production {role} asset identity does not match its spec")


def production_benchmark_specs() -> tuple[ProductionBenchmarkSpec, ...]:
    """Return the immutable production inventory."""

    return _PRODUCTION_SPECS


def load_production_benchmark_adapter(
    cell_id: str,
    prepared_root: str | Path,
    *,
    checkpoints: Iterable[Tau3BankingCheckpoint] | None = None,
    evaluation_bridge: Tau3EvaluationBridge | None = None,
) -> PreparedQABenchmarkAdapter | PreparedRetrievalBenchmarkAdapter | Tau3BankingCheckpointAdapter:
    """Load a named production cell through its locked asset and split mapping.

    This is the preferred public entry point for official cells.  It resolves
    assets from the packaged registry and calls :func:`load_prepared_benchmark`,
    which reverifies cache identity, counts, and every prepared-file hash.
    Direct adapter construction remains available for custom datasets.
    """

    if not isinstance(cell_id, str) or not cell_id.strip():
        raise ValidationError("production benchmark cell_id must be non-empty")
    try:
        spec = _PRODUCTION_BY_CELL[cell_id]
    except KeyError as exc:
        available = ", ".join(item.cell_id for item in _PRODUCTION_SPECS)
        raise ValidationError(
            f"unknown production benchmark cell {cell_id!r}; available: {available}"
        ) from exc

    if cell_id != "tau3.banking_knowledge" and (
        checkpoints is not None or evaluation_bridge is not None
    ):
        raise ValidationError("captured checkpoints and evaluation_bridge are only valid for tau3")
    if spec.adapter_kind == "checkpoint" and (checkpoints is None or evaluation_bridge is None):
        raise ValidationError(
            "tau3 loading requires externally captured checkpoints and a typed "
            "caller-asserted evaluation_bridge; official runtime capture, clone, and "
            "continuation are not implemented"
        )

    registry = BenchmarkSmokeAssetRegistry.default()
    query_asset = registry.get(spec.query_asset_id)
    _require_registered_identity(
        query_asset,
        expected_asset_id=spec.query_asset_id,
        expected_cell_id=spec.cell_id,
        role="query",
    )
    query_prepared = load_prepared_benchmark(query_asset, prepared_root)
    if spec.adapter_kind == "checkpoint":
        return Tau3BankingCheckpointAdapter(
            PreparedTau3BankingBundle(query_prepared),
            checkpoints=checkpoints,
            evaluation_bridge=evaluation_bridge,
        )
    if spec.adapter_kind == "qa":
        return PreparedQABenchmarkAdapter(
            query_prepared,
            benchmark_id=spec.cell_id,
            split=spec.split,
        )

    corpus_prepared = None
    if spec.corpus_asset_id is not None:
        corpus_asset = registry.get(spec.corpus_asset_id)
        _require_registered_identity(
            corpus_asset,
            expected_asset_id=spec.corpus_asset_id,
            expected_cell_id=spec.cell_id,
            role="corpus",
        )
        corpus_prepared = load_prepared_benchmark(corpus_asset, prepared_root)
    return PreparedRetrievalBenchmarkAdapter(
        query_prepared,
        benchmark_id=spec.cell_id,
        split=spec.split,
        corpus_prepared=corpus_prepared,
    )


def load_tau3_banking_bundle(
    prepared_root: str | Path,
) -> PreparedTau3BankingBundle:
    """Load the pinned static tau3 source bundle without inventing a checkpoint."""

    registry = BenchmarkSmokeAssetRegistry.default()
    asset = registry.get("tau3.banking_knowledge.v1_0_1")
    _require_registered_identity(
        asset,
        expected_asset_id="tau3.banking_knowledge.v1_0_1",
        expected_cell_id="tau3.banking_knowledge",
        role="query",
    )
    return PreparedTau3BankingBundle(load_prepared_benchmark(asset, prepared_root))


__all__ = [
    "ProductionBenchmarkSpec",
    "load_production_benchmark_adapter",
    "load_tau3_banking_bundle",
    "production_benchmark_specs",
]
