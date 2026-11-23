"""Checksum-verified local artifact store with atomic writes."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agentinputlab.contracts.errors import ArtifactError

from .manifest import ArtifactManifest


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ArtifactStore:
    root: Path

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(
                f"cannot create artifact cache directory {self.root}: {exc.strerror}; "
                "choose a writable directory with cache_dir / --cache-dir or "
                "AGENTINPUTLAB_CACHE; do not change permissions on shared research data"
            ) from exc

    def path_for(self, manifest: ArtifactManifest) -> Path:
        candidate = (self.root / manifest.artifact_id / manifest.revision / "payload").resolve()
        if self.root not in candidate.parents:
            raise ArtifactError("artifact path escaped store root")
        return candidate

    def verify(self, manifest: ArtifactManifest) -> Path:
        path = self.path_for(manifest)
        if not path.is_file():
            raise ArtifactError(f"artifact missing: {manifest.artifact_id}@{manifest.revision}")
        if manifest.size_bytes is not None and path.stat().st_size != manifest.size_bytes:
            raise ArtifactError(f"artifact size mismatch: {manifest.artifact_id}")
        if manifest.sha256 and sha256_file(path) != manifest.sha256:
            raise ArtifactError(f"artifact checksum mismatch: {manifest.artifact_id}")
        return path

    def put_bytes(self, manifest: ArtifactManifest, content: bytes) -> Path:
        digest = hashlib.sha256(content).hexdigest()
        if manifest.sha256 and digest != manifest.sha256:
            raise ArtifactError(
                f"refusing materialization: checksum mismatch for {manifest.artifact_id}"
            )
        if manifest.size_bytes is not None and len(content) != manifest.size_bytes:
            raise ArtifactError(
                f"refusing materialization: size mismatch for {manifest.artifact_id}"
            )
        destination = self.path_for(manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_file() and destination.read_bytes() == content:
                return self.verify(manifest)
            raise ArtifactError(
                "refusing to overwrite an existing artifact revision with different content: "
                f"{manifest.artifact_id}@{manifest.revision}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".partial-",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Linking a fully flushed temporary file is atomic and, unlike
                # os.replace(), cannot overwrite a concurrently materialized
                # artifact revision.
                os.link(temporary_name, destination)
            except FileExistsError:
                if not destination.is_file() or destination.read_bytes() != content:
                    raise ArtifactError(
                        "refusing concurrent overwrite of artifact revision: "
                        f"{manifest.artifact_id}@{manifest.revision}"
                    ) from None
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return self.verify(manifest)

    def import_file(self, manifest: ArtifactManifest, source: str | Path) -> Path:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise ArtifactError(f"source artifact is not a regular file: {source_path}")
        with source_path.open("rb") as handle:
            return self.put_stream(manifest, iter(lambda: handle.read(1024 * 1024), b""))

    def put_stream(
        self,
        manifest: ArtifactManifest,
        chunks: Iterable[bytes],
        *,
        max_bytes: int | None = None,
    ) -> Path:
        """Hash bounded chunks to a temporary file and publish create-only.

        A network interruption, size violation, or bad checksum never leaves a
        usable payload. Concurrent downloads cannot replace an existing revision.
        Neither the incoming payload nor an existing payload is buffered in RAM.
        """
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ArtifactError("stream size limit must be a non-negative integer")
        destination = self.path_for(manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".partial-", dir=destination.parent)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise ArtifactError("artifact stream must yield bytes")
                    size += len(chunk)
                    if (max_bytes is not None and size > max_bytes) or (
                        manifest.size_bytes is not None and size > manifest.size_bytes
                    ):
                        raise ArtifactError("artifact stream exceeds size limit")
                    digest.update(chunk)
                    handle.write(chunk)
                if manifest.size_bytes is not None and size != manifest.size_bytes:
                    raise ArtifactError(
                        f"refusing materialization: size mismatch for {manifest.artifact_id}"
                    )
                if manifest.sha256 and digest.hexdigest() != manifest.sha256:
                    raise ArtifactError(
                        f"refusing materialization: checksum mismatch for {manifest.artifact_id}"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_name, destination)
            except FileExistsError:
                if (
                    not destination.is_file()
                    or destination.stat().st_size != size
                    or sha256_file(destination) != digest.hexdigest()
                ):
                    raise ArtifactError(
                        "refusing concurrent overwrite of artifact revision: "
                        f"{manifest.artifact_id}@{manifest.revision}"
                    ) from None
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination
