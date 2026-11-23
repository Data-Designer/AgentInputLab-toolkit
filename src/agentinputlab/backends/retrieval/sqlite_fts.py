"""Disk-backed, deterministic SQLite FTS5 BM25 for mounted large corpora.

This is an explicit lexical retrieval profile, not a claim to reproduce a
different BM25 implementation's rankings. Building streams the corpus once;
concurrent queries use independent read-only connections.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import closing, suppress
from pathlib import Path
from typing import Any

from agentinputlab.contracts import ArtifactError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash

from .base import Document, RetrievalResult


class SQLiteFTS5Retriever:
    retriever_id = "sqlite.fts5.bm25.unicode61.v1"

    def __init__(self, path: str | Path, *, corpus_identity: Mapping[str, Any]):
        self.path = Path(path).expanduser().absolute()
        self.identity = json.loads(canonical_json(corpus_identity))
        if not self.identity:
            raise ValidationError("disk corpus identity is required")
        if self.path.is_symlink() or not self.path.is_file():
            raise ArtifactError("FTS index must be a regular existing file")
        try:
            with closing(self._connection()) as db:
                row = db.execute("SELECT value FROM metadata WHERE key=?", ("binding",)).fetchone()
                if row is None:
                    raise ArtifactError("FTS index has no corpus binding")
                metadata = json.loads(row[0])
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise ArtifactError("FTS index is invalid or incomplete") from exc
        if (
            metadata["corpus_identity"] != self.identity
            or metadata["retriever_id"] != self.retriever_id
        ):
            raise ArtifactError("FTS index identity differs")
        self.count = metadata["document_count"]

    def _connection(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)

    @classmethod
    def build(
        cls, path: str | Path, documents: Iterable[Document], *, corpus_identity: Mapping[str, Any]
    ):
        target = Path(path).expanduser().absolute()
        if target.exists():
            return cls(target, corpus_identity=corpus_identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".fts-build-", suffix=".sqlite", dir=target.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            with closing(sqlite3.connect(temporary)) as db:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA synchronous=OFF")
                db.execute(
                    "CREATE TABLE documents (doc_id TEXT UNIQUE NOT NULL,title TEXT NOT NULL,text TEXT NOT NULL,revision TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE VIRTUAL TABLE passages USING fts5(title,text,content='documents',content_rowid='rowid',tokenize='unicode61')"
                )
                count = 0
                for doc in documents:
                    if not isinstance(doc, Document):
                        raise ValidationError("FTS corpus requires Documents")
                    cursor = db.execute(
                        "INSERT INTO documents VALUES (?,?,?,?)",
                        (doc.doc_id, doc.title, doc.text, doc.corpus_revision),
                    )
                    db.execute(
                        "INSERT INTO passages(rowid,title,text) VALUES (?,?,?)",
                        (cursor.lastrowid, doc.title, doc.text),
                    )
                    count += 1
                    if count % 10000 == 0:
                        db.commit()
                if not count:
                    raise ValidationError("FTS corpus cannot be empty")
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
                binding = {
                    "retriever_id": cls.retriever_id,
                    "corpus_identity": json.loads(canonical_json(corpus_identity)),
                    "document_count": count,
                }
                db.execute(
                    "INSERT INTO metadata VALUES (?,?)", ("binding", canonical_json(binding))
                )
                db.execute("INSERT INTO passages(passages) VALUES ('optimize')")
                db.commit()
            # A concurrent builder cannot replace an existing index.
            # Another builder may publish first. Reopen below and validate
            # its identity; never overwrite it or mask a mismatched cache.
            with suppress(FileExistsError):
                os.link(temporary, target)
        except sqlite3.IntegrityError as exc:
            raise ArtifactError("FTS corpus contains duplicate document IDs") from exc
        except sqlite3.OperationalError as exc:
            if "no such module: fts5" in str(exc):
                raise ExternalDependencyError("SQLite must be compiled with FTS5 support") from exc
            raise ArtifactError(f"FTS index build failed: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return cls(target, corpus_identity=corpus_identity)

    def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValidationError("top_k must be positive")
        if not isinstance(query, str):
            raise ValidationError("query must be text")
        terms = list(dict.fromkeys(re.findall(r"\w+", query, flags=re.UNICODE)))[:256]
        if not terms:
            return ()
        expression = " OR ".join('"' + term + '"' for term in terms)
        with closing(self._connection()) as db:
            rows = db.execute(
                "SELECT d.doc_id,d.title,d.text,d.revision,bm25(passages) AS score "
                "FROM passages JOIN documents d ON d.rowid=passages.rowid WHERE passages MATCH ? "
                "ORDER BY score ASC,d.doc_id ASC LIMIT ?",
                (expression, top_k),
            ).fetchall()
        return tuple(
            RetrievalResult(
                Document(doc_id=r[0], title=r[1], text=r[2], corpus_revision=r[3]),
                score=-r[4],
                rank=i + 1,
            )
            for i, r in enumerate(rows)
        )

    def fingerprint(self):
        return {
            "retriever_id": self.retriever_id,
            "sqlite_version": sqlite3.sqlite_version,
            "corpus_identity": self.identity,
            "corpus_hash": content_hash(self.identity),
            "document_count": self.count,
            "tokenizer": "unicode61",
            "query_operator": "OR",
            "max_unique_query_terms": 256,
            "title_weight": 1.0,
            "text_weight": 1.0,
        }


__all__ = ["SQLiteFTS5Retriever"]
