"""Metadata-only discovery: no downloads, cache creation or runtime execution."""

from __future__ import annotations

from agentinputlab.contracts.serialization import to_jsonable

from .assets import BenchmarkSmokeAssetRegistry
from .beir_archive import BEIR_ARCHIVE_LOCKS
from .cache import BFCL_SIMPLE_ASSET, BFCL_SIMPLE_CELL, DEFAULT_DOWNLOAD_LIMIT
from .longbench import _SUBSETS as LONG_BENCH_CELLS
from .longbench_sources import manifests as longbench_manifests
from .production import production_benchmark_specs
from .registry import BenchmarkRegistry


def describe_benchmark(cell_id: str) -> dict:
    """Return source sizes, prerequisites and an explicit preparation route.

    Source size is for a cold cache, not an estimate of normalized storage or
    GPU memory. An automatic loader is not proof of algorithm compatibility.
    """
    cell = BenchmarkRegistry.default().get(cell_id)
    result = {
        **to_jsonable(cell),
        "automatic_loader": False,
        "loader_kind": "native_adapter",
        "split": None,
        "sources": [],
        "source_download_bytes": None,
        "exceeds_default_download_limit": None,
        "runtime_parameters": [],
        "prepare_command": None,
        "next_step": "Use the documented native adapter and its runtime prerequisites; "
        "this cell is not supported by load_benchmark(). See docs/benchmark-setup.md.",
    }
    specs = {spec.cell_id: spec for spec in production_benchmark_specs()}
    if cell_id in specs:
        spec = specs[cell_id]
        assets = BenchmarkSmokeAssetRegistry.default()
        sources = [assets.get(key) for key in (spec.query_asset_id, spec.corpus_asset_id) if key]
        result.update(
            loader_kind=spec.adapter_kind,
            split=spec.split,
            sources=[{
                "asset_id": asset.asset_id, "url": asset.source_url,
                "size_bytes": asset.size_bytes, "sha256": asset.sha256,
            } for asset in sources],
            license_review_required=any(a.status == "license_review_required" for a in sources),
            source_download_bytes=sum(a.size_bytes for a in sources),
        )
        if spec.adapter_kind == "checkpoint":
            result["runtime_parameters"] = ["checkpoints", "evaluation_bridge"]
    elif cell_id == BFCL_SIMPLE_CELL:
        asset = BenchmarkSmokeAssetRegistry.default().get(BFCL_SIMPLE_ASSET)
        result.update(
            loader_kind="bfcl_v3_simple", split="test", runtime_parameters=["model_id"],
            license_review_required=asset.status == "license_review_required",
            source_download_bytes=asset.size_bytes,
            sources=[{"asset_id": asset.asset_id, "url": asset.source_url,
                      "size_bytes": asset.size_bytes, "sha256": asset.sha256}],
        )
    elif cell_id in LONG_BENCH_CELLS:
        sources = longbench_manifests()
        result.update(loader_kind="longbench_v1", split="test", license_review_required=True,
            source_download_bytes=sum(m.size_bytes for m in sources.values()),
            sources=[{"asset_id": m.artifact_id, "url": m.source_uri,
                      "size_bytes": m.size_bytes, "sha256": m.sha256} for m in sources.values()])
    elif cell_id in BEIR_ARCHIVE_LOCKS:
        lock = BEIR_ARCHIVE_LOCKS[cell_id]
        result.update(
            loader_kind="beir_archive", split="test", license_review_required=True,
            source_download_bytes=lock["size_bytes"],
            sources=[{
                "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
                + lock["dataset"] + ".zip",
                "size_bytes": lock["size_bytes"], "sha256": lock["sha256"],
            }],
        )
    else:
        return result
    result["automatic_loader"] = True
    size = result["source_download_bytes"]
    result["exceeds_default_download_limit"] = size > DEFAULT_DOWNLOAD_LIMIT
    command = f"agentinputlab datasets prepare {cell_id}"
    if result["license_review_required"]:
        command += " --accept-license-review"
    result["prepare_command"] = command
    result["next_step"] = (
        "Review upstream terms before acknowledging them. Prepare a verified cache, then "
        "use load_benchmark() with the listed runtime parameters. Large sources require "
        "an explicit download limit; preparation can need additional disk space. "
        "See docs/benchmark-setup.md."
    )
    return result


def list_benchmarks(*, search: str = "", family: str | None = None) -> list[dict]:
    """Search the packaged inventory by ID, suite or task (case insensitive)."""
    query = search.casefold()
    return [
        describe_benchmark(cell.cell_id)
        for cell in BenchmarkRegistry.default().list(family=family)
        if query in f"{cell.cell_id} {cell.suite} {cell.task}".casefold()
    ]
