from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentinputlab.artifacts import ArtifactStore
from agentinputlab.benchmarks import BenchmarkSmokeAsset, materialize_benchmark_asset
from agentinputlab.benchmarks.preprocess import prepare_smoke_asset
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.data.storage import DEFAULT_DISK_RESERVE, require_storage_capacity


def small_asset():
    content = (
        b"topic_id\tinitial_request\ttopic_desc\tclarification_need\tfacet_id\t"
        b"facet_desc\tquestion_id\tquestion\tanswer\n"
        b"1\tjaguar\tCars\t3\tF1\tCars\tQ1\tDo you mean cars?\tYes\n"
    )
    return BenchmarkSmokeAsset(
        asset_id="fixture.storage.clariq",
        cell_id="clariq.main",
        source_url="https://example.com/small.tsv",
        source_revision="fixture-v1",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        redistribution="not_redistributed",
        status="verified_smoke",
        processor="clariq_tsv_v1",
        expected_counts={"inputs": 1, "gold": 1, "topics": 1},
        prepared_sha256={},
        allowed_hosts=("example.com",),
        provenance={"fixture": "unit-test"},
    ), content


class StorageCapacityTests(unittest.TestCase):
    def test_checks_actual_ancestor_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            destination = root / "missing" / "data"
            with patch(
                "agentinputlab.data.storage.shutil.disk_usage",
                return_value=SimpleNamespace(free=100),
            ) as disk:
                require_storage_capacity([(destination, 20)], reserve_bytes=80)
            disk.assert_called_once_with(root)
            self.assertFalse(destination.parent.exists())

    def test_sums_allocations_on_same_device_and_keeps_reserve(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "agentinputlab.data.storage.shutil.disk_usage",
                return_value=SimpleNamespace(free=100),
            ) as disk,
        ):
            root = Path(directory)
            with self.assertRaisesRegex(ArtifactError, "insufficient storage"):
                require_storage_capacity(
                    [(root / "source", 40), (root / "prepared", 41)], reserve_bytes=20
                )
            self.assertEqual(disk.call_count, 1)

    def test_invalid_or_unknown_sizes_are_not_unbounded_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            for size in (None, True, -1, 2.5):
                with self.subTest(size=size), self.assertRaises(ValidationError):
                    require_storage_capacity([(directory, size)])
            for reserve in (None, False, 0, -1):
                with self.subTest(reserve=reserve), self.assertRaises(ValidationError):
                    require_storage_capacity([(directory, 1)], reserve_bytes=reserve)


class BoundedPreparationTests(unittest.TestCase):
    def test_low_space_rejected_before_normalization(self):
        asset, content = small_asset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            source.write_bytes(content)
            with (
                patch(
                    "agentinputlab.data.storage.shutil.disk_usage",
                    return_value=SimpleNamespace(free=DEFAULT_DISK_RESERVE + 99),
                ),
                patch("agentinputlab.benchmarks.preprocess._normalize_clariq_tsv") as normalize,
                self.assertRaisesRegex(ArtifactError, "insufficient storage"),
            ):
                prepare_smoke_asset(asset, source, root / "prepared", max_prepared_bytes=100)
            normalize.assert_not_called()
            self.assertFalse(list(root.rglob(".prepare-*")))

    def test_output_limit_cleans_partial_and_preserves_source(self):
        asset, content = small_asset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            source.write_bytes(content)
            with self.assertRaisesRegex(ArtifactError, "max_prepared_bytes"):
                prepare_smoke_asset(asset, source, root / "prepared", max_prepared_bytes=10)
            self.assertFalse(list(root.rglob(".prepare-*")))
            self.assertFalse(list((root / "prepared").rglob("metadata.json")))
            self.assertEqual(source.read_bytes(), content)

    def test_complete_cache_reused_without_requiring_new_output_space(self):
        asset, content = small_asset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            source.write_bytes(content)
            first = prepare_smoke_asset(asset, source, root / "prepared")
            with patch(
                "agentinputlab.data.storage.shutil.disk_usage",
                side_effect=AssertionError("verified cache requires no new allocation"),
            ):
                repeated = prepare_smoke_asset(
                    asset, source, root / "prepared", max_prepared_bytes=1
                )
            self.assertEqual(first, repeated)

    def test_acquisition_sums_source_and_prepared_before_network(self):
        asset, _ = small_asset()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "agentinputlab.benchmarks.acquisition.BenchmarkSmokeAssetRegistry.default",
                return_value=SimpleNamespace(get=lambda _: asset),
            ),
            patch(
                "agentinputlab.data.storage.shutil.disk_usage",
                return_value=SimpleNamespace(free=DEFAULT_DISK_RESERVE + asset.size_bytes + 99),
            ),
            patch("agentinputlab.benchmarks.acquisition.download_artifact_streamed") as download,
        ):
            with self.assertRaisesRegex(ArtifactError, "insufficient storage"):
                materialize_benchmark_asset(
                    asset.asset_id,
                    store_root=Path(directory) / "source",
                    prepared_root=Path(directory) / "prepared",
                    allow_network=True,
                    max_prepared_bytes=100,
                )
            download.assert_not_called()

    def test_cached_raw_source_still_checks_preparation_space(self):
        asset, content = small_asset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ArtifactStore(root / "source").put_bytes(asset.artifact_manifest(), content)
            with (
                patch(
                    "agentinputlab.benchmarks.acquisition.BenchmarkSmokeAssetRegistry.default",
                    return_value=SimpleNamespace(get=lambda _: asset),
                ),
                patch(
                    "agentinputlab.data.storage.shutil.disk_usage",
                    return_value=SimpleNamespace(free=DEFAULT_DISK_RESERVE + 99),
                ),
                patch(
                    "agentinputlab.benchmarks.acquisition.download_artifact_streamed"
                ) as download,
                self.assertRaisesRegex(ArtifactError, "insufficient storage"),
            ):
                materialize_benchmark_asset(
                    asset.asset_id,
                    store_root=root / "source",
                    prepared_root=root / "prepared",
                    allow_network=False,
                    max_prepared_bytes=100,
                )
            download.assert_not_called()

    def test_prepared_cache_does_not_redownload_cleaned_raw_source(self):
        asset, content = small_asset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            source.write_bytes(content)
            prepare_smoke_asset(asset, source, root / "prepared")
            source.unlink()
            with (
                patch(
                    "agentinputlab.benchmarks.acquisition.BenchmarkSmokeAssetRegistry.default",
                    return_value=SimpleNamespace(get=lambda _: asset),
                ),
                patch(
                    "agentinputlab.data.storage.shutil.disk_usage",
                    side_effect=AssertionError("no new storage needed"),
                ),
                patch("agentinputlab.benchmarks.acquisition.download_artifact_streamed") as download,
            ):
                report = materialize_benchmark_asset(
                    asset.asset_id,
                    store_root=root / "not-created",
                    prepared_root=root / "prepared",
                    allow_network=False,
                    max_download_bytes=1,
                    max_prepared_bytes=1,
                )
            self.assertEqual(report["acquired_from"], "prepared_cache")
            self.assertFalse((root / "not-created").exists())
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
