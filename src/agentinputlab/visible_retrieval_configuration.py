"""Explicit per-case BM25 binding; no corpus download or hidden task data."""

from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.visible_chunks import split_visible_text


class VisibleContextBM25:
    """Unbound configuration placeholder, replaced before each case executes."""

    retriever_id = "visible-context-bm25"

    def __init__(self, value):
        from agentinputlab.configuration import _known

        _known(value, {"kind", "k1", "b", "document_projection", "corpus_revision", "chunk_words", "chunk_boundary"}, "visible-context retriever")
        self.k1 = value.get("k1", 1.2)
        self.b = value.get("b", 0.75)
        self.document_projection = value.get("document_projection", "visible_metadata")
        if self.document_projection not in ("visible_metadata", "text_only"):
            raise ValidationError("visible-context document_projection must be visible_metadata or text_only")
        self.corpus_revision = value.get("corpus_revision")
        if "corpus_revision" in value and (not isinstance(self.corpus_revision, str) or not self.corpus_revision.strip()):
            raise ValidationError("visible-context corpus_revision must be non-empty text")
        self.chunk_words = value.get("chunk_words")
        if "chunk_words" in value and (type(self.chunk_words) is not int or self.chunk_words < 1):
            raise ValidationError("visible-context chunk_words must be a positive integer")
        if self.chunk_words is not None and self.document_projection == "text_only":
            raise ValidationError("visible-context chunks require traceable source metadata")
        self.chunk_boundary = value.get("chunk_boundary", "word")
        if "chunk_boundary" in value and (self.chunk_words is None or self.chunk_boundary not in ("word", "line")):
            raise ValidationError("visible-context chunk_boundary requires chunk_words and word or line")
        # Reuse the actual implementation's validation; no documents or I/O.
        InMemoryBM25Retriever((), k1=self.k1, b=self.b)

    def fingerprint(self):
        return {"retriever_id": self.retriever_id, "binding": "pending_visible_case",
                "k1": self.k1, "b": self.b, "document_projection": self.document_projection,
                "corpus_revision": self.corpus_revision,
                **({"chunk_words": self.chunk_words, "chunk_boundary": self.chunk_boundary}
                   if self.chunk_words is not None else {})}

    def retrieve(self, query, *, top_k=5):
        raise ValidationError("visible_context_bm25 must be bound to a case before retrieval")

    def bind(self, state):
        if not state.context:
            raise ValidationError("visible_context_bm25 requires non-empty visible case context; no fallback corpus")
        ids = [block.block_id for block in state.context]
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValidationError("visible_context_bm25 requires unique non-empty visible block IDs")
        revision = self.corpus_revision or content_hash(state.context)
        documents = []
        for block in state.context:
            if self.document_projection == "text_only":
                documents.append(Document(block.block_id, block.text, corpus_revision=revision))
                continue
            title = block.metadata.get("title", "" if self.chunk_words is not None else block.block_id)
            if not isinstance(title, str):
                raise ValidationError("visible_context_bm25 block metadata.title must be text")
            # A new retrieval assigns its own rank/features/provenance. Preserve
            # old annotations in a namespace rather than spoofing new fields.
            reserved = {"title", "document_id", "corpus_revision", "rank",
                        "retrieval_features", "evidence_kind"}
            metadata = {k: v for k, v in block.metadata.items() if k not in reserved}
            provenance = {k: v for k, v in block.metadata.items() if k in reserved - {"title"}}
            if provenance:
                if "visible_context_provenance" in metadata:
                    raise ValidationError("visible context contains conflicting provenance namespace")
                metadata["visible_context_provenance"] = provenance
            if self.chunk_words is None:
                documents.append(Document(block.block_id, block.text, title=title,
                                          corpus_revision=revision, metadata=metadata))
                continue
            if "visible_chunk_source" in metadata:
                raise ValidationError("visible context contains conflicting chunk provenance")
            for chunk in split_visible_text(block.text, words_per_chunk=self.chunk_words, boundary=self.chunk_boundary):
                # The ID encodes the complete source ID followed by integer offsets.
                doc_id = f"{len(block.block_id)}:{block.block_id}:{chunk.start}:{chunk.end}"
                documents.append(Document(doc_id, chunk.text, title=title, corpus_revision=revision,
                    metadata={**metadata, "visible_chunk_source": {
                        "block_id": block.block_id, "char_start": chunk.start,
                        "char_end": chunk.end, "block_sha256": chunk.source_sha256,
                        "words_per_chunk": self.chunk_words, "chunk_boundary": self.chunk_boundary,
                    }}))
        return InMemoryBM25Retriever(tuple(documents), k1=self.k1, b=self.b)
