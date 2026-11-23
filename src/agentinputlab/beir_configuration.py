"""Bounded, complete BEIR archive corpora for public retrieval configurations."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentinputlab.backends.retrieval import InMemoryBM25Retriever, SQLiteFTS5Retriever
from agentinputlab.benchmarks.beir_archive import (
    BEIR_ARCHIVE_LOCKS,
    BEIRArchiveAdapter,
    validate_split,
)
from agentinputlab.benchmarks.cache import (
    DEFAULT_DOWNLOAD_LIMIT,
    benchmark_cache_dir,
    load_benchmark,
)
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.data.storage import require_storage_capacity


def build(config: Mapping[str, Any], *, load: bool = True):
    allowed = {"kind", "benchmark_id", "source_archive", "cache_dir", "offline",
               "accept_license_review", "max_download_bytes", "max_documents", "max_text_bytes", "k1", "b",
               "engine", "index_path", "split"}
    if set(config) - allowed:
        raise ValidationError("unknown BEIR retriever fields: " + ", ".join(sorted(set(config) - allowed)))
    cell = config.get("benchmark_id")
    if not isinstance(cell, str) or cell not in BEIR_ARCHIVE_LOCKS:
        raise ValidationError("beir_archive retriever requires a supported locked BEIR benchmark_id")
    split = config.get("split", "test")
    validate_split(cell, split)
    for name in ("offline", "accept_license_review"):
        if not isinstance(config.get(name, False), bool):
            raise ValidationError(f"BEIR {name} must be boolean")
    limits = {"max_download_bytes": DEFAULT_DOWNLOAD_LIMIT, "max_documents": 100_000,
              "max_text_bytes": 256 * 1024 * 1024}
    for key, default in limits.items():
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(f"BEIR {key} must be a positive integer")
        limits[key] = value
    for key in ("source_archive", "cache_dir", "index_path"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValidationError(f"BEIR {key} must be a nonempty path string")
    if "source_archive" in config and "cache_dir" in config:
        raise ValidationError("choose BEIR source_archive or cache_dir, not both")
    engine = config.get("engine", "sqlite_fts5")
    if engine not in ("sqlite_fts5", "memory_bm25"):
        raise ValidationError("BEIR engine must be sqlite_fts5 or memory_bm25")
    if engine == "sqlite_fts5" and ("k1" in config or "b" in config):
        raise ValidationError("k1/b belong to memory_bm25; SQLite uses its original native BM25")
    if engine == "memory_bm25" and "index_path" in config:
        raise ValidationError("memory_bm25 does not use index_path")
    options = {"k1": config.get("k1", 1.2), "b": config.get("b", .75)}
    InMemoryBM25Retriever((), **options)  # validate numbers, no file/network access
    if not load:
        return None
    if not config.get("accept_license_review", False):
        raise ArtifactError("review upstream dataset terms and set accept_license_review=true")
    if "source_archive" in config:
        adapter = BEIRArchiveAdapter(benchmark_id=cell,
            source_archive=Path(config["source_archive"]).expanduser(), license_acknowledged=True, split=split)
    else:
        adapter = load_benchmark(cell, cache_dir=config.get("cache_dir"), offline=config.get("offline", False),
            accept_license_review=True, max_download_bytes=limits["max_download_bytes"], split=split)
    def bounded_documents():
        size = 0
        for count, doc in enumerate(adapter.iter_documents(), start=1):
            if count > limits["max_documents"]:
                raise ArtifactError("complete BEIR corpus exceeds max_documents; no truncated index was created")
            size += len(doc.doc_id.encode()) + len(doc.title.encode()) + len(doc.text.encode())
            if size > limits["max_text_bytes"]:
                raise ArtifactError("complete BEIR corpus exceeds max_text_bytes; no truncated index was created")
            yield doc
    if engine == "sqlite_fts5":
        identity = {**adapter.fingerprint(), "forum": ""}
        path = (Path(config["index_path"]).expanduser() if "index_path" in config else
                benchmark_cache_dir(config.get("cache_dir")) / "indices" / (content_hash(identity) + ".sqlite"))
        if not path.exists():
            require_storage_capacity([(path.parent, 4 * limits["max_text_bytes"])])
        inner = SQLiteFTS5Retriever.build(path, bounded_documents(), corpus_identity=identity)
        if inner.count > limits["max_documents"]:
            raise ArtifactError("cached complete BEIR corpus exceeds max_documents")
        return BEIRCorpusRetriever(adapter, inner, inner.count)
    documents = list(bounded_documents())
    if len({doc.doc_id for doc in documents}) != len(documents):
        raise ArtifactError("BEIR corpus contains duplicate document IDs")
    if not documents:
        raise ArtifactError("BEIR corpus is empty")
    return BEIRCorpusRetriever(adapter, InMemoryBM25Retriever(documents, **options), len(documents))


class BEIRCorpusRetriever:
    """Reuse the original owned adapter for scoring; no qrels enter retrieval."""

    def __init__(self, adapter, inner, count):
        self.adapter, self._inner, self._count = adapter, inner, count

    @property
    def retriever_id(self):
        return self._inner.retriever_id

    def retrieve(self, query, *, top_k=5):
        return self._inner.retrieve(query, top_k=top_k)

    def fingerprint(self):
        return {"implementation": "agentinputlab.beir_archive_retriever.v1", "corpus_scope": "complete_source",
                "source": self.adapter.fingerprint(), "document_count": self._count,
                "retriever": self._inner.fingerprint()}
