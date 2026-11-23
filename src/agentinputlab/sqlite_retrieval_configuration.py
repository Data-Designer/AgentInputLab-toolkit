"""Explicit, checksum-bound reuse of an existing native SQLite FTS5 index."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.backends.retrieval import SQLiteFTS5Retriever
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json


def build(config: Mapping[str, Any], *, load: bool = True):
    allowed = {"kind", "index_path", "index_sha256", "corpus_identity",
               "expected_document_count", "benchmark_id"}
    if set(config) - allowed:
        raise ValidationError("unknown sqlite_fts5 retriever fields: " +
                              ", ".join(sorted(set(config) - allowed)))
    if not isinstance(config.get("index_path"), str) or not config["index_path"].strip():
        raise ValidationError("sqlite_fts5 index_path must be a nonempty path string")
    digest = config.get("index_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValidationError("sqlite_fts5 index_sha256 must be a lowercase SHA-256 digest")
    identity = config.get("corpus_identity")
    if not isinstance(identity, Mapping) or not identity:
        raise ValidationError("sqlite_fts5 corpus_identity must be a nonempty object")
    canonical_json(identity)
    count = config.get("expected_document_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValidationError("sqlite_fts5 expected_document_count must be a positive integer")
    if "benchmark_id" in config and (
        not isinstance(config["benchmark_id"], str) or not config["benchmark_id"].strip()
    ):
        raise ValidationError("sqlite_fts5 benchmark_id must be nonempty text")
    if not load:
        return None  # Shape validation never opens an index or prepares a corpus.
    path = Path(config["index_path"]).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ArtifactError("sqlite_fts5 requires an existing regular index; no automatic rebuild")
    # The digest binds a standalone database, not unpinned WAL/journal contents.
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-journal")):
        raise ArtifactError("sqlite_fts5 requires a closed standalone index without WAL/journal files")
    try:
        before = path.stat()
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        after = path.stat()
        def signature(st):
            return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
        if signature(before) != signature(after) or hasher.hexdigest() != digest:
            raise ArtifactError("sqlite_fts5 index SHA-256 differs or file changed during verification")
        retriever = SQLiteFTS5Retriever(path, corpus_identity=identity)
    except OSError as exc:
        raise ArtifactError("sqlite_fts5 index cannot be read") from exc
    except (KeyError, TypeError) as exc:
        raise ArtifactError("sqlite_fts5 index corpus binding is incomplete") from exc
    if isinstance(retriever.count, bool) or retriever.count != count:
        raise ArtifactError("sqlite_fts5 index document count differs")
    return retriever  # Preserve native rankings and fingerprint; no substitute engine.
