from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentinputlab.artifacts import (
    AcquisitionPolicy,
    ArtifactKind,
    ArtifactManifest,
    ArtifactStore,
    UpstreamLock,
)
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.data import (
    DownloadPolicy,
    DownloadResponse,
    download_artifact,
    validate_download_url,
)


class FixtureDownloadTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.targets = []

    def request(self, target, *, timeout_seconds, max_bytes):
        self.targets.append((target, timeout_seconds, max_bytes))
        return self.responses.pop(0)


class ArtifactTests(unittest.TestCase):
    def _manifest(self, content: bytes) -> ArtifactManifest:
        return ArtifactManifest(
            artifact_id="tiny-fixture",
            kind=ArtifactKind.DATASET,
            revision="v1",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            acquisition=AcquisitionPolicy.BUNDLED,
            license_id="CC0-1.0",
        )

    def test_atomic_put_and_verify(self) -> None:
        content = b'{"id":"1"}\n'
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            path = store.put_bytes(self._manifest(content), content)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(store.verify(self._manifest(content)), path)

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            with self.assertRaises(ArtifactError):
                store.put_bytes(self._manifest(b"expected"), b"other")

    def test_same_revision_is_append_only_but_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            original = self._manifest(b"first")
            first = store.put_bytes(original, b"first")
            self.assertEqual(store.put_bytes(original, b"first"), first)
            replacement = self._manifest(b"other")
            with self.assertRaisesRegex(ArtifactError, "refusing to overwrite"):
                store.put_bytes(replacement, b"other")
            self.assertEqual(first.read_bytes(), b"first")

    def test_manifest_lineage_and_metadata_are_recursively_frozen(self) -> None:
        lineage = {"parent": "source@v1"}
        metadata = {"nested": {"values": [1, 2]}}
        manifest = ArtifactManifest(
            artifact_id="frozen-fixture",
            kind=ArtifactKind.MODEL,
            revision="v1",
            sha256="",
            acquisition=AcquisitionPolicy.USER_PROVIDED,
            license_id="MIT",
            lineage=lineage,
            metadata=metadata,
        )
        lineage["parent"] = "changed"
        metadata["nested"]["values"].append(3)
        self.assertEqual(manifest.lineage["parent"], "source@v1")
        self.assertEqual(manifest.metadata["nested"]["values"], (1, 2))
        with self.assertRaises((AttributeError, TypeError)):
            manifest.metadata["new"] = "value"  # type: ignore[index]

    def test_manifest_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValidationError):
            ArtifactManifest(
                artifact_id="safe",
                revision="../escape",
                kind=ArtifactKind.CACHE,
                sha256="",
                acquisition=AcquisitionPolicy.USER_PROVIDED,
                license_id="NOASSERTION",
            )

    def test_noassertion_cannot_be_vendored(self) -> None:
        with self.assertRaises(ValidationError):
            UpstreamLock(
                repository="owner/repo",
                commit="a" * 40,
                license_id="NOASSERTION",
                vendoring_allowed=True,
            )

    def test_network_is_disabled_by_default(self) -> None:
        with self.assertRaises(ArtifactError):
            validate_download_url("https://example.com/data", DownloadPolicy())

    def test_network_opt_in_still_requires_an_explicit_allowlist(self) -> None:
        with self.assertRaisesRegex(ArtifactError, "non-empty host allowlist"):
            validate_download_url(
                "https://example.com/data",
                DownloadPolicy(allow_network=True),
            )

    @patch(
        "agentinputlab.data.downloader.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    def test_download_uses_validated_peer_and_pinned_checksum(self, _resolve) -> None:
        content = b"verified artifact"
        manifest = ArtifactManifest(
            artifact_id="download-fixture",
            kind=ArtifactKind.DATASET,
            revision="v1",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            acquisition=AcquisitionPolicy.DOWNLOAD,
            license_id="CC0-1.0",
            source_uri="https://example.com/data",
        )
        transport = FixtureDownloadTransport(
            (
                DownloadResponse(
                    status_code=200,
                    headers={"Content-Length": str(len(content))},
                    body=content,
                    peer_address="93.184.216.34",
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = download_artifact(
                manifest,
                ArtifactStore(directory),
                policy=DownloadPolicy(
                    allow_network=True,
                    allowed_hosts=("example.com",),
                ),
                transport=transport,
            )
            self.assertEqual(Path(path).read_bytes(), content)

    @patch(
        "agentinputlab.data.downloader.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    def test_download_rejects_peer_not_in_validated_dns_set(self, _resolve) -> None:
        content = b"x"
        manifest = ArtifactManifest(
            artifact_id="peer-fixture",
            kind=ArtifactKind.DATASET,
            revision="v1",
            sha256=hashlib.sha256(content).hexdigest(),
            acquisition=AcquisitionPolicy.DOWNLOAD,
            license_id="CC0-1.0",
            source_uri="https://example.com/data",
        )
        transport = FixtureDownloadTransport(
            (
                DownloadResponse(
                    status_code=200,
                    headers={},
                    body=content,
                    peer_address="8.8.8.8",
                ),
            )
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ArtifactError, "validated DNS"),
        ):
            download_artifact(
                manifest,
                ArtifactStore(directory),
                policy=DownloadPolicy(
                    allow_network=True,
                    allowed_hosts=("example.com",),
                ),
                transport=transport,
            )

    @patch(
        "agentinputlab.data.downloader.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    def test_redirect_is_revalidated_against_allowlist(self, _resolve) -> None:
        content = b"x"
        manifest = ArtifactManifest(
            artifact_id="redirect-fixture",
            kind=ArtifactKind.DATASET,
            revision="v1",
            sha256=hashlib.sha256(content).hexdigest(),
            acquisition=AcquisitionPolicy.DOWNLOAD,
            license_id="CC0-1.0",
            source_uri="https://example.com/data",
        )
        transport = FixtureDownloadTransport(
            (
                DownloadResponse(
                    status_code=302,
                    headers={"Location": "https://internal.example/secret"},
                    body=b"",
                    peer_address="93.184.216.34",
                ),
            )
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ArtifactError, "not allowlisted"),
        ):
            download_artifact(
                manifest,
                ArtifactStore(directory),
                policy=DownloadPolicy(
                    allow_network=True,
                    allowed_hosts=("example.com",),
                ),
                transport=transport,
            )


if __name__ == "__main__":
    unittest.main()
