"""Official cached Query2doc profile.

This module consumes the author-released pseudo-document artifact.  It does not
generate pseudo-documents and never treats them as retrieved evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType

from agentinputlab.contracts import (
    ArtifactError,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash

from .models import OfficialQuery2DocRow

OFFICIAL_DATASET_REVISION = "3019b760cd849830aec6d0654dbe8a4a03e791d6"
# No single checksum covers the author's full multi-split release. Only the
# two independently audited, unmodified native TREC files are supported below.
OFFICIAL_ARTIFACT_SHA256: str | None = None
OFFICIAL_TREC_ARTIFACTS = MappingProxyType({
    "trec_dl2019": (
        "d2aec536829539cbb1bcb8e3a93d1580ff8d9667cd9182c3d09f976592d7d5fe", 21995, 43,
    ),
    "trec_dl2020": (
        "97edbf0b4ae1f2367fd90a6f365030e72667f9c9e5da9219e8ff23e1d8e7c782", 26973, 54,
    ),
})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def compose_sparse_expansion(
    query: str,
    pseudo_doc: str,
    *,
    query_repeat: int = 5,
) -> str:
    """Materialize the official sparse query: query repeated five times + pseudo-doc."""

    if not query:
        raise ValidationError("Query2doc expansion requires a non-empty query")
    if not pseudo_doc:
        raise ValidationError("Query2doc expansion requires a non-empty pseudo_doc")
    if query_repeat < 1:
        raise ValidationError("query_repeat must be positive")
    return " ".join((*([query] * query_repeat), pseudo_doc))


class Query2DocCache:
    """In-memory view over an already materialized, hash-checked cache artifact."""

    __slots__ = ("_artifact_hash", "_revision", "_rows", "_verification")

    def __init__(
        self,
        rows: Iterable[OfficialQuery2DocRow],
        *,
        revision: str = OFFICIAL_DATASET_REVISION,
        artifact_hash: str = "",
        _verification: str = "fixture_unverified",
    ) -> None:
        self._revision = revision
        self._artifact_hash = artifact_hash
        self._verification = _verification
        keyed: dict[tuple[str, str], OfficialQuery2DocRow] = {}
        for row in rows:
            key = (row.split, row.query_id)
            if key in keyed:
                raise ArtifactError(f"duplicate Query2doc cache key: {key!r}")
            keyed[key] = row
        if not keyed:
            raise ArtifactError("Query2doc cache cannot be empty")
        self._rows = keyed

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def artifact_hash(self) -> str:
        return self._artifact_hash

    @property
    def verification(self) -> str:
        return self._verification

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object] | OfficialQuery2DocRow],
        *,
        revision: str = OFFICIAL_DATASET_REVISION,
        artifact_hash: str = "",
    ) -> Query2DocCache:
        normalized = []
        for value in rows:
            if isinstance(value, OfficialQuery2DocRow):
                normalized.append(value)
            else:
                normalized.append(
                    OfficialQuery2DocRow(
                        query_id=str(value["query_id"]),
                        query=str(value["query"]),
                        pseudo_doc=str(value["pseudo_doc"]),
                        split=str(value.get("split", "")),
                    )
                )
        return cls(normalized, revision=revision, artifact_hash=artifact_hash)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        revision: str = OFFICIAL_DATASET_REVISION,
    ) -> Query2DocCache:
        if not _SHA256.fullmatch(expected_sha256):
            raise ArtifactError("official Query2doc cache requires a pinned SHA-256")
        data = Path(path).read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_sha256:
            raise ArtifactError(
                f"Query2doc cache checksum mismatch: expected {expected_sha256}, got {actual_hash}"
            )
        rows = []
        for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"invalid Query2doc JSONL at line {line_number}") from exc
        normalized = cls.from_rows(rows, revision=revision, artifact_hash=actual_hash)
        return cls(
            normalized._rows.values(),
            revision=revision,
            artifact_hash=actual_hash,
            _verification="caller_sha256_verified",
        )

    @classmethod
    def from_official_jsonl(
        cls, path: str | Path, *, split: str, accept_license_review: bool = False,
    ) -> Query2DocCache:
        """Read an exact author TREC file, offline, after explicit terms review.

        Native files do not contain a ``split`` field. Bind it here without
        rewriting any source query or pseudo-document. Generation used the
        author's text-davinci-003, not the caller's current target backbone.
        """
        if accept_license_review is not True:
            raise ArtifactError("review Query2doc CC-BY-4.0 and underlying MS MARCO terms first")
        if split not in OFFICIAL_TREC_ARTIFACTS:
            raise ArtifactError("only audited trec_dl2019 and trec_dl2020 author caches are supported")
        expected_hash, expected_size, expected_rows = OFFICIAL_TREC_ARTIFACTS[split]
        with Path(path).open("rb") as handle:
            raw = handle.read(expected_size + 1)
        actual = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_size or actual != expected_hash:
            raise ArtifactError("official Query2doc native file size/checksum mismatch")
        rows = []
        try:
            for line in raw.decode("utf-8").splitlines():
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != {"query_id", "query", "pseudo_doc"}:
                    raise ArtifactError("official Query2doc native row fields differ")
                if not all(isinstance(v, str) and v.strip() for v in value.values()):
                    raise ArtifactError("official Query2doc native row values must be non-empty strings")
                rows.append(OfficialQuery2DocRow(**value, split=split))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("invalid official Query2doc native JSONL") from exc
        if len(rows) != expected_rows:
            raise ArtifactError("official Query2doc native row count differs")
        return cls(
            rows,
            revision=OFFICIAL_DATASET_REVISION,
            artifact_hash=actual,
            _verification="official_manifest_verified",
        )

    def get(self, split: str, query_id: str) -> OfficialQuery2DocRow:
        try:
            return self._rows[(split, query_id)]
        except KeyError as exc:
            raise ArtifactError(
                f"Query2doc cache has no row for split={split!r}, query_id={query_id!r}"
            ) from exc


class _Query2DocCachedBase:
    method_id = "query2doc"
    family = SemanticFamily.REFORMULATE
    release_id = ""
    implementation_origin = ""
    implementation_identity = ""
    profile_kind = ""
    cache_event = ""
    patch_rationale = ""
    requires_verified_artifact = False

    def __init__(
        self,
        cache: Query2DocCache,
        *,
        split: str,
        strict_query_match: bool = True,
    ) -> None:
        if self.requires_verified_artifact and cache.revision != OFFICIAL_DATASET_REVISION:
            raise ArtifactError(
                "official_cached requires the pinned Query2doc dataset revision "
                f"{OFFICIAL_DATASET_REVISION}, got {cache.revision}"
            )
        if self.requires_verified_artifact and cache.verification != "official_manifest_verified":
            raise ArtifactError(
                "official_cached requires the audited Query2DocCache.from_official_jsonl asset"
            )
        if self.requires_verified_artifact and strict_query_match is not True:
            raise ArtifactError("official_cached requires exact source query matching")
        self.cache = cache
        self.split = split
        self.strict_query_match = strict_query_match

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        query_id = state.task_id or str(state.metadata.get("query_id", ""))
        if not query_id:
            raise ValidationError(
                "Query2doc official cache requires state.task_id or metadata.query_id"
            )
        row = self.cache.get(self.split, query_id)
        if self.strict_query_match and row.query != state.query:
            raise ArtifactError(
                f"Query2doc cached query mismatch for {query_id!r}; refusing text lookup/fallback"
            )
        expanded = compose_sparse_expansion(state.query, row.pseudo_doc, query_repeat=5)
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    self.cache_event,
                    {
                        "query_id": query_id,
                        "split": self.split,
                        "dataset_revision": self.cache.revision,
                        "artifact_hash": self.cache.artifact_hash,
                        "artifact_verification": self.cache.verification,
                        "source_generation_model": (
                            "text-davinci-003" if self.requires_verified_artifact else "unverified"
                        ),
                        "historical_generation_cost": "unknown_not_free",
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "pseudo_document_expansion",
                    {
                        "pseudo_doc": row.pseudo_doc,
                        "pseudo_doc_hash": content_hash(row.pseudo_doc),
                        "query_repeat": 5,
                        "evidence_role": "synthetic_query_surrogate",
                        "expanded_query": expanded,
                    },
                ),
            ),
            counters={"model_calls": 0, "query_repeat": 5},
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_QUERY,
                    value=expanded,
                    target="query",
                    rationale=self.patch_rationale,
                ),
            ),
            trace=trace,
            implementation_identity=self.implementation_identity,
            profile_kind=self.profile_kind,
            metadata={
                "implementation_origin": self.implementation_origin,
                "pseudo_document_is_evidence": False,
                "dataset_revision": self.cache.revision,
                "artifact_verification": self.cache.verification,
                "source_generation_model": (
                    "text-davinci-003" if self.requires_verified_artifact else "unverified"
                ),
                "historical_generation_cost": "unknown_not_free",
            },
        )


class Query2DocOfficialCached(_Query2DocCachedBase):
    """Use a checksum-verified author-released pseudo-document cache."""

    release_id = "rewrite.query2doc.official_cached"
    implementation_origin = "official_artifact"
    implementation_identity = (
        f"official_artifact:intfloat/query2doc_msmarco@{OFFICIAL_DATASET_REVISION}"
    )
    profile_kind = "official_native"
    cache_event = "official_cache_lookup"
    patch_rationale = "official Query2doc cached pseudo-document expansion"
    requires_verified_artifact = True


class Query2DocFixtureCached(_Query2DocCachedBase):
    """Controlled local-fixture profile used only for offline examples and tests."""

    release_id = "fixture.query2doc.cached"
    implementation_origin = "fixture_unverified"
    implementation_identity = "fixture:query2doc_cached.v1"
    profile_kind = "fixture_unverified"
    cache_event = "fixture_cache_lookup"
    patch_rationale = "controlled Query2doc-style cached pseudo-document expansion"
