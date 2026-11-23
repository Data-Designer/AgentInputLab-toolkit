"""Explicit acquisition service for checksum-pinned existing benchmark assets."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

from agentinputlab.artifacts import ArtifactStore
from agentinputlab.contracts.errors import ArtifactError
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

from .assets import BenchmarkSmokeAssetRegistry
from .preprocess import _prepared_path, load_prepared_benchmark, prepare_smoke_asset


def materialize_benchmark_asset(
    asset_id: str,
    *,
    store_root: str | Path,
    prepared_root: str | Path,
    allow_network: bool,
    accept_license_review: bool = False,
    max_download_bytes: int = 512 * 1024 * 1024,
    max_prepared_bytes: int = DEFAULT_PREPARED_LIMIT,
) -> dict[str, Any]:
    """Acquire and preprocess one exact registered source asset.

    Network use and local processing of ``NOASSERTION`` sources are both
    separately opt-in.  The function cannot acquire an inventory-only cell
    because only accepted sources exist in :class:`BenchmarkSmokeAssetRegistry`.
    """

    asset = BenchmarkSmokeAssetRegistry.default().get(asset_id)
    positive_byte_limit(max_download_bytes, name="max_download_bytes")
    positive_byte_limit(max_prepared_bytes, name="max_prepared_bytes")
    if asset.status == "license_review_required" and not accept_license_review:
        raise ArtifactError(
            f"{asset.asset_id} has NOASSERTION licensing; pass accept_license_review=True "
            "only after reviewing the upstream terms"
        )
    # A verified prepared cache needs no download or new output allowance.
    prepared_destination = _prepared_path(asset, Path(prepared_root).expanduser().resolve())
    if prepared_destination.exists():
        prepared = load_prepared_benchmark(asset, Path(prepared_root))
        acquired = "prepared_cache"
    else:
        store = ArtifactStore(store_root)
        manifest = asset.artifact_manifest()
        try:
            source = store.verify(manifest)
            acquired = "cache"
        except ArtifactError:
            if not allow_network:
                raise ArtifactError(
                    f"{asset.asset_id} is not cached and network acquisition is disabled"
                ) from None
            if asset.size_bytes > max_download_bytes:
                raise ArtifactError("source exceeds max_download_bytes before download") from None
            require_storage_capacity(
                [(store.root, asset.size_bytes), (prepared_root, max_prepared_bytes)]
            )
            source = Path(
                download_artifact_streamed(
                    manifest,
                    store,
                    policy=DownloadPolicy(
                        allow_network=True,
                        allowed_hosts=asset.allowed_hosts,
                        max_bytes=min(asset.size_bytes, max_download_bytes),
                        timeout_seconds=120.0,
                        max_redirects=3,
                    ),
                    transport=PinnedHTTPSDownloadTransport(),
                )
            )
            acquired = "network"
        prepared = prepare_smoke_asset(
            asset, source, Path(prepared_root), max_prepared_bytes=max_prepared_bytes
        )
    return {
        "asset_id": asset.asset_id,
        "cell_id": asset.cell_id,
        "status": asset.status,
        "license_id": asset.license_id,
        "acquired_from": acquired,
        "source_sha256": asset.sha256,
        "source_size_bytes": asset.size_bytes,
        "processor": asset.processor,
        "prepared_root": str(prepared.root),
        "counts": dict(prepared.counts),
        "file_sha256": dict(prepared.file_sha256),
        "source_host": urllib.parse.urlsplit(asset.source_url).hostname,
    }


__all__ = ["materialize_benchmark_asset"]
