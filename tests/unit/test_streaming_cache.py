from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentinputlab.artifacts import AcquisitionPolicy, ArtifactKind, ArtifactManifest, ArtifactStore
from agentinputlab.benchmarks import BenchmarkSmokeAssetRegistry, prepare_benchmark
from agentinputlab.benchmarks.beir_archive import BEIR_ARCHIVE_LOCKS
from agentinputlab.benchmarks.preprocess import _prepared_path
from agentinputlab.contracts import ArtifactError
from agentinputlab.data import DownloadPolicy, StreamingDownloadResponse, download_artifact_streamed


class ModelDownloadSettingsTests(unittest.TestCase):
    def test_native_hf_loading_preserves_pins_cache_and_offline_flag(self):
        from agentinputlab.backends import TransformersBackend

        tokenizer, model = MagicMock(), MagicMock()
        module = SimpleNamespace(AutoTokenizer=tokenizer, AutoModelForCausalLM=model)
        with patch.dict(sys.modules, {"transformers": module}):
            backend = TransformersBackend(
                model_id="upstream/model",
                model_revision="model-sha",
                tokenizer_revision="tokenizer-sha",
                cache_dir="/fixture/cache",
                offline=True,
            )
            backend._load()
        tokenizer.from_pretrained.assert_called_once_with(
            "upstream/model",
            revision="tokenizer-sha",
            trust_remote_code=False,
            cache_dir="/fixture/cache",
            local_files_only=True,
        )
        model.from_pretrained.assert_called_once_with(
            "upstream/model",
            revision="model-sha",
            trust_remote_code=False,
            cache_dir="/fixture/cache",
            local_files_only=True,
        )


def manifest(data=b"abcdef", **changes):
    args = dict(
        artifact_id="stream-fixture",
        kind=ArtifactKind.DATASET,
        revision="v1",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        acquisition=AcquisitionPolicy.DOWNLOAD,
        license_id="CC0-1.0",
        source_uri="https://example.com/file",
    )
    return ArtifactManifest(**(args | changes))


class Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.closed = 0
        self.targets = []

    @contextmanager
    def stream(self, target, **_kwargs):
        self.targets.append(target)
        try:
            yield next(self.responses)
        finally:
            self.closed += 1


class StreamingStoreTests(unittest.TestCase):
    def test_chunks_and_import_are_idempotent_without_read_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("must stream")):
                path = store.put_stream(manifest(), iter([b"abc", b"def"]), max_bytes=6)
                self.assertEqual(store.put_stream(manifest(), [b"abcdef"]), path)
                self.assertEqual(store.import_file(manifest(), path), path)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertFalse(list(Path(directory).rglob(".partial-*")))

    def test_checksum_size_and_interruption_leave_no_payload(self):
        def interrupted():
            yield b"ab"
            raise OSError("lost connection")

        for chunks, pattern, error in [
            ([b"xxxxxx"], "checksum", ArtifactError),
            ([b"a"], "size mismatch", ArtifactError),
            ([b"abcdefg"], "size limit", ArtifactError),
            (interrupted(), "lost connection", OSError),
        ]:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                store = ArtifactStore(directory)
                with self.assertRaisesRegex(error, pattern):
                    store.put_stream(manifest(), chunks)
                self.assertFalse(store.path_for(manifest()).exists())
                self.assertFalse(list(Path(directory).rglob(".partial-*")))

    def test_existing_revision_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            path = store.put_stream(manifest(), [b"abcdef"])
            with self.assertRaisesRegex(ArtifactError, "overwrite"):
                store.put_stream(manifest(b"change"), [b"change"])
            self.assertEqual(path.read_bytes(), b"abcdef")


@patch(
    "agentinputlab.data.downloader.socket.getaddrinfo",
    return_value=[(2, 1, 6, "", ("93.184.216.34", 443))],
)
class StreamingDownloadTests(unittest.TestCase):
    def test_chunks_follow_validated_redirect_and_close(self, _dns):
        t = Transport(
            [
                StreamingDownloadResponse(302, {"Location": "/actual"}, (), "93.184.216.34"),
                StreamingDownloadResponse(
                    200, {"Content-Length": "6"}, [b"abc", b"def"], "93.184.216.34"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = download_artifact_streamed(
                manifest(),
                ArtifactStore(directory),
                policy=DownloadPolicy(allow_network=True, allowed_hosts=("example.com",)),
                transport=t,
            )
            self.assertEqual(Path(path).read_bytes(), b"abcdef")
        self.assertEqual(t.closed, 2)
        self.assertTrue(t.targets[1].url.endswith("/actual"))

    def test_peer_and_content_encoding_are_validated_before_chunks(self, _dns):
        def chunks():
            raise AssertionError("must not consume")
            yield b""

        for peer, headers, pattern in [
            ("127.0.0.1", {}, "peer"),
            ("93.184.216.34", {"Content-Encoding": "gzip"}, "encoded"),
            ("93.184.216.34", {"Content-Length": "7"}, "declared size"),
        ]:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                t = Transport([StreamingDownloadResponse(200, headers, chunks(), peer)])
                with self.assertRaisesRegex(ArtifactError, pattern):
                    download_artifact_streamed(
                        manifest(),
                        ArtifactStore(directory),
                        policy=DownloadPolicy(allow_network=True, allowed_hosts=("example.com",)),
                        transport=t,
                    )
                self.assertEqual(t.closed, 1)

    def test_oversized_manifest_rejected_before_resolution_or_transport(self, dns):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ArtifactError, "before download"):
                download_artifact_streamed(
                    manifest(),
                    ArtifactStore(directory),
                    policy=DownloadPolicy(
                        allow_network=True, allowed_hosts=("example.com",), max_bytes=5
                    ),
                    transport=Transport([]),
                )
            dns.assert_not_called()


class CacheLoaderTests(unittest.TestCase):
    def test_default_size_budget_blocks_large_source_before_network(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agentinputlab.benchmarks.cache.download_artifact_streamed") as download,
        ):
            with self.assertRaisesRegex(ArtifactError, "max_download_bytes"):
                prepare_benchmark("beir.fever", cache_dir=directory, accept_license_review=True)
            download.assert_not_called()

    def test_offline_missing_and_license_review_do_not_network(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agentinputlab.benchmarks.cache.download_artifact_streamed") as download,
        ):
            with self.assertRaisesRegex(ArtifactError, "upstream dataset terms"):
                prepare_benchmark("beir.fiqa_2018", cache_dir=directory)
            with self.assertRaisesRegex(ArtifactError, "offline mode"):
                prepare_benchmark(
                    "beir.fiqa_2018", cache_dir=directory, offline=True, accept_license_review=True
                )
            download.assert_not_called()

    def test_cached_archive_is_verified_and_used_offline(self):
        from agentinputlab.benchmarks.cache import _source_manifest

        tiny = {
            "dataset": "fiqa",
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
            "size_bytes": 7,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(BEIR_ARCHIVE_LOCKS, {"beir.fiqa_2018": tiny}),
            patch("agentinputlab.benchmarks.cache.download_artifact_streamed") as download,
        ):
            store = ArtifactStore(Path(directory) / "sources")
            source = store.put_stream(_source_manifest("beir.fiqa_2018"), [b"fixture"])
            result = prepare_benchmark(
                "beir.fiqa_2018", cache_dir=directory, offline=True, accept_license_review=True
            )
            self.assertEqual(result["source_archive"], str(source))
            self.assertEqual(result["acquired_from"], "cache")
            source.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ArtifactError, "checksum mismatch"):
                prepare_benchmark("beir.fiqa_2018", cache_dir=directory, accept_license_review=True)
            download.assert_not_called()

    def test_prepared_cache_survives_raw_archive_cleanup(self):
        asset = BenchmarkSmokeAssetRegistry.default().get("qasper.dev.official")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agentinputlab.benchmarks.cache.load_prepared_benchmark") as prepared,
            patch("agentinputlab.benchmarks.cache.materialize_benchmark_asset") as acquire,
        ):
            path = _prepared_path(asset, Path(directory) / "prepared")
            path.mkdir(parents=True)
            result = prepare_benchmark(
                asset.cell_id, cache_dir=directory, offline=True, accept_license_review=True
            )
            self.assertEqual(result["acquisitions"], [])
            prepared.assert_called_once()
            acquire.assert_not_called()

    def test_missing_small_source_is_automatically_streamed(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "agentinputlab.benchmarks.cache.download_artifact_streamed",
                return_value="/fixture/payload",
            ) as download,
        ):
            result = prepare_benchmark(
                "beir.fiqa_2018", cache_dir=directory, accept_license_review=True
            )
            self.assertEqual(result["acquired_from"], "network")
            self.assertTrue(download.call_args.kwargs["policy"].allow_network)


if __name__ == "__main__":
    unittest.main()
