"""Cache-first benchmark preparation with bounded, opt-out network acquisition.

Only source-locked loaders are exposed here. Inventory entries without a locked
loader are not downloaded under a guessed URL. Large datasets require an explicit
larger byte limit; loading a small test sample never silently downloads a full
multi-GB corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

from agentinputlab.artifacts import AcquisitionPolicy, ArtifactKind, ArtifactManifest, ArtifactStore
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.data import (
    DownloadPolicy,
    PinnedHTTPSDownloadTransport,
    download_artifact_streamed,
)
from agentinputlab.data.storage import (
    DEFAULT_PREPARED_LIMIT,
    positive_byte_limit,
    require_storage_capacity,
)

from .acquisition import materialize_benchmark_asset
from .assets import BenchmarkSmokeAssetRegistry
from .beir_archive import BEIR_ARCHIVE_LOCKS, BEIRArchiveAdapter
from .longbench import _SUBSETS as LONG_BENCH_CELLS
from .longbench_sources import ALLOWED_HOSTS as LONG_BENCH_HOSTS
from .longbench_sources import manifests as _longbench_manifests
from .preprocess import _prepared_path, load_prepared_benchmark, prepare_smoke_asset
from .production import load_production_benchmark_adapter, production_benchmark_specs
from .registry import BenchmarkRegistry

DEFAULT_DOWNLOAD_LIMIT = 512 * 1024 * 1024
BFCL_SIMPLE_CELL = "bfcl.v3.simple_python"
BFCL_SIMPLE_ASSET = "bfcl.v3.simple_python.v1_3"


def benchmark_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve a toolkit-owned cache, without changing HF or research caches."""
    value = cache_dir or os.environ.get("AGENTINPUTLAB_CACHE")
    return (
        (Path(value) if value else Path.home() / ".cache" / "agentinputlab").expanduser().resolve()
    )


def _source_manifest(cell_id: str) -> ArtifactManifest:
    lock = BEIR_ARCHIVE_LOCKS[cell_id]
    return ArtifactManifest(
        artifact_id=cell_id,
        kind=ArtifactKind.DATASET,
        revision=lock["sha256"],
        sha256=lock["sha256"],
        acquisition=AcquisitionPolicy.DOWNLOAD,
        license_id="NOASSERTION",
        source_uri="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
        + lock["dataset"]
        + ".zip",
        size_bytes=lock["size_bytes"],
    )


def _require_download_capacity(root: Path, total_bytes: int, limit: int) -> None:
    positive_byte_limit(limit, name="max_download_bytes")
    if total_bytes > limit:
        raise ArtifactError(
            f"missing sources need {total_bytes:,} bytes, above max_download_bytes={limit:,}; "
            "use a documented source-side sample or explicitly raise the limit for a full download"
        )
    require_storage_capacity([(root, total_bytes)])


def prepare_benchmark(
    cell_id: str,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    accept_license_review: bool = False,
    max_download_bytes: int = DEFAULT_DOWNLOAD_LIMIT,
    max_prepared_bytes: int = DEFAULT_PREPARED_LIMIT,
    source_archive: str | Path | None = None,
) -> dict:
    """Reuse a verified cache or download missing pinned sources automatically.

    ``offline=True`` prohibits all network calls. The default download budget is
    512 MiB across missing source files, not an unbounded per-file allowance.
    License-review sources require explicit acknowledgement, including cache use.
    Corrupt caches raise an integrity error and are never silently overwritten.
    """
    BenchmarkRegistry.default().get(cell_id)
    if source_archive is not None and cell_id != BFCL_SIMPLE_CELL:
        raise ValidationError("source_archive is supported only for BFCL simple preparation")
    root = benchmark_cache_dir(cache_dir)
    positive_byte_limit(max_download_bytes, name="max_download_bytes")
    positive_byte_limit(max_prepared_bytes, name="max_prepared_bytes")
    if cell_id in LONG_BENCH_CELLS:
        if not accept_license_review:
            raise ArtifactError("review upstream dataset terms and pass accept_license_review=True")
        manifests = _longbench_manifests()
        store = ArtifactStore(root / "sources")
        missing = []
        paths = {}
        acquired = {}
        # Verify every present file before acquiring any missing file. A damaged
        # scorer cache is never overwritten, executed, or hidden by a redownload.
        for key, manifest in manifests.items():
            path = store.path_for(manifest)
            paths[key] = str(path)
            if path.exists():
                store.verify(manifest)
                acquired[key] = "cache"
            else:
                missing.append((key, manifest))
        if missing and offline:
            raise ArtifactError("LongBench source files are not cached; offline mode prohibits downloads")
        if missing:
            _require_download_capacity(root, sum(m.size_bytes for _, m in missing), max_download_bytes)
        for key, manifest in missing:
            download_artifact_streamed(manifest, store,
                policy=DownloadPolicy(allow_network=True, allowed_hosts=LONG_BENCH_HOSTS,
                                      max_bytes=manifest.size_bytes),
                transport=PinnedHTTPSDownloadTransport())
            acquired[key] = "network"
        return {"cell_id": cell_id, "cache_dir": str(root), "kind": "longbench_v1",
                "longbench_sources": paths, "acquired_from": acquired,
                "source_sha256": {key: m.sha256 for key, m in manifests.items()}}
    if cell_id == BFCL_SIMPLE_CELL:
        asset = BenchmarkSmokeAssetRegistry.default().get(BFCL_SIMPLE_ASSET)
        if asset.status == "license_review_required" and not accept_license_review:
            raise ArtifactError("review upstream dataset terms and pass accept_license_review=True")
        prepared = root / "prepared"
        acquired = "mounted"
        if source_archive is not None:
            path = Path(source_archive).expanduser().resolve()
        else:
            # BFCL needs the original archive for its native scorer, even when
            # normalized inputs already exist. Never accept prepared data alone.
            manifest = asset.artifact_manifest()
            store = ArtifactStore(root / "sources")
            path = store.path_for(manifest)
            acquired = "cache"
            if path.exists():
                store.verify(manifest)
            else:
                if offline:
                    raise ArtifactError(f"{cell_id} scorer source is not cached; offline mode prohibits downloads")
                _require_download_capacity(root, asset.size_bytes, max_download_bytes)
                if not _prepared_path(asset, prepared).exists():
                    require_storage_capacity([(root, asset.size_bytes), (prepared, max_prepared_bytes)])
                path = Path(download_artifact_streamed(
                    manifest, store,
                    policy=DownloadPolicy(allow_network=True, allowed_hosts=asset.allowed_hosts,
                                          max_bytes=min(asset.size_bytes, max_download_bytes)),
                    transport=PinnedHTTPSDownloadTransport(),
                ))
                acquired = "network"
        # Also verifies a mounted archive and an existing normalized cache;
        # neither is copied or overwritten if its identity differs.
        prepare_smoke_asset(asset, path, prepared, max_prepared_bytes=max_prepared_bytes)
        return {
            "cell_id": cell_id, "cache_dir": str(root), "kind": "bfcl_v3_simple",
            "prepared_root": str(prepared), "source_archive": str(path),
            "source_sha256": asset.sha256, "acquired_from": acquired,
        }
    specs = {s.cell_id: s for s in production_benchmark_specs()}
    if cell_id in specs:
        spec = specs[cell_id]
        registry = BenchmarkSmokeAssetRegistry.default()
        assets = [registry.get(x) for x in (spec.query_asset_id, spec.corpus_asset_id) if x]
        if not accept_license_review and any(a.status == "license_review_required" for a in assets):
            raise ArtifactError("review upstream dataset terms and pass accept_license_review=True")
        prepared = root / "prepared"
        missing = []
        for asset in assets:
            # A prepared cache remains usable after its raw download is cleaned.
            if _prepared_path(asset, prepared).exists():
                load_prepared_benchmark(asset, prepared)
            else:
                missing.append(asset)
        store = ArtifactStore(root / "sources")
        uncached = []
        for asset in missing:
            manifest = asset.artifact_manifest()
            if store.path_for(manifest).exists():
                store.verify(manifest)
            else:
                uncached.append(asset)
        if uncached and offline:
            raise ArtifactError(f"{cell_id} is not cached; offline mode prohibits downloads")
        if uncached:
            _require_download_capacity(
                root, sum(a.size_bytes for a in uncached), max_download_bytes
            )
        if missing:
            require_storage_capacity(
                [
                    (root / "sources", sum(a.size_bytes for a in uncached)),
                    (prepared, len(missing) * max_prepared_bytes),
                ]
            )
        reports = [
            materialize_benchmark_asset(
                a.asset_id,
                store_root=root / "sources",
                prepared_root=prepared,
                allow_network=not offline,
                accept_license_review=accept_license_review,
                max_download_bytes=max_download_bytes,
                max_prepared_bytes=max_prepared_bytes,
            )
            for a in missing
        ]
        return {
            "cell_id": cell_id,
            "cache_dir": str(root),
            "kind": "production",
            "prepared_root": str(prepared),
            "acquisitions": reports,
        }
    if cell_id in BEIR_ARCHIVE_LOCKS:
        if not accept_license_review:
            raise ArtifactError("review upstream dataset terms and pass accept_license_review=True")
        manifest = _source_manifest(cell_id)
        store = ArtifactStore(root / "sources")
        path = store.path_for(manifest)
        acquired = "cache"
        if path.exists():
            store.verify(manifest)
        else:
            if offline:
                raise ArtifactError(f"{cell_id} is not cached; offline mode prohibits downloads")
            _require_download_capacity(root, manifest.size_bytes, max_download_bytes)
            path = Path(
                download_artifact_streamed(
                    manifest,
                    store,
                    policy=DownloadPolicy(
                        allow_network=True,
                        allowed_hosts=("public.ukp.informatik.tu-darmstadt.de",),
                        max_bytes=max_download_bytes,
                    ),
                    transport=PinnedHTTPSDownloadTransport(),
                )
            )
            acquired = "network"
        return {
            "cell_id": cell_id,
            "cache_dir": str(root),
            "kind": "beir_archive",
            "source_archive": str(path),
            "source_sha256": manifest.sha256,
            "acquired_from": acquired,
        }
    raise ValidationError(
        f"{cell_id!r} has no automatic source-locked loader yet; use its documented native adapter"
    )


def load_benchmark(
    cell_id: str,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    accept_license_review: bool = False,
    max_download_bytes: int = DEFAULT_DOWNLOAD_LIMIT,
    max_prepared_bytes: int = DEFAULT_PREPARED_LIMIT,
    source_archive: str | Path | None = None,
    **runtime,
):
    """Prepare missing data and construct the corresponding official adapter."""
    # Check dynamic-runtime requirements before spending bandwidth.
    allowed = ({"split"} if cell_id in BEIR_ARCHIVE_LOCKS else
               {"model_id"} if cell_id == BFCL_SIMPLE_CELL else
               {"checkpoints", "evaluation_bridge"} if cell_id == "tau3.banking_knowledge" else set())
    unexpected = set(runtime) - allowed
    if unexpected:
        raise ValidationError(
            f"unsupported runtime arguments for {cell_id}: {', '.join(sorted(unexpected))}; "
            "check agentinputlab datasets show before preparing data"
        )
    if cell_id == "tau3.banking_knowledge" and (
        runtime.get("checkpoints") is None or runtime.get("evaluation_bridge") is None
    ):
        raise ValidationError(
            "tau3 requires captured checkpoints and an evaluation_bridge before downloading"
        )
    if cell_id == BFCL_SIMPLE_CELL and (
        not isinstance(runtime.get("model_id"), str) or not runtime["model_id"].strip()
    ):
        raise ValidationError("BFCL requires the actual model_id before downloading")
    if cell_id in BEIR_ARCHIVE_LOCKS:
        from .beir_archive import validate_split

        validate_split(cell_id, runtime.get("split", "test"))
    report = prepare_benchmark(
        cell_id,
        cache_dir=cache_dir,
        offline=offline,
        accept_license_review=accept_license_review,
        max_download_bytes=max_download_bytes,
        max_prepared_bytes=max_prepared_bytes,
        source_archive=source_archive,
    )
    if report["kind"] == "bfcl_v3_simple":
        from .bfcl_native import SourceVerifiedBFCLV3ASTBridge
        from .bfcl_v3 import load_bfcl_v3_simple_adapter

        return load_bfcl_v3_simple_adapter(
            report["prepared_root"], evaluation_bridge=SourceVerifiedBFCLV3ASTBridge(
                source_archive=report["source_archive"], model_id=runtime["model_id"],
            ),
        )
    if report["kind"] == "longbench_v1":
        from .longbench import LongBenchAdapter

        return LongBenchAdapter(benchmark_id=cell_id, **report["longbench_sources"])
    if report["kind"] == "beir_archive":
        return BEIRArchiveAdapter(
            benchmark_id=cell_id, source_archive=report["source_archive"], license_acknowledged=True,
            split=runtime.get("split", "test"),
        )
    return load_production_benchmark_adapter(cell_id, report["prepared_root"], **runtime)


__all__ = ["DEFAULT_DOWNLOAD_LIMIT", "benchmark_cache_dir", "load_benchmark", "prepare_benchmark"]
