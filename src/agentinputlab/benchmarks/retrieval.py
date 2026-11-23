"""Prepared retrieval benchmark adapter with evaluator-only qrels.

The adapter consumes the checksum-verified normalized representation produced
by :mod:`agentinputlab.benchmarks.preprocess`.  Prediction callbacks see only
the query-side state; qrels remain attached to ``BenchmarkExample`` and cross
the boundary only inside :meth:`score`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, freeze_mapping
from agentinputlab.metrics import retrieval_aggregates

from .base import BenchmarkExample
from .preprocess import PreparedBenchmark


@dataclass(frozen=True, slots=True)
class _NamedRetrievalProfile:
    query_asset_id: str
    query_processor: str
    split: str
    corpus_asset_id: str | None = None
    corpus_processor: str | None = None


_NAMED_RETRIEVAL_PROFILES = {
    "beir.arguana": _NamedRetrievalProfile(
        query_asset_id="beir.arguana.official",
        query_processor="beir_zip_v1",
        split="test",
    ),
    "bright.earth_science": _NamedRetrievalProfile(
        query_asset_id="bright.earth_science.examples",
        query_processor="bright_examples_parquet_v1",
        split="test",
        corpus_asset_id="bright.earth_science.corpus",
        corpus_processor="bright_corpus_parquet_v1",
    ),
    "bright.economics": _NamedRetrievalProfile(
        query_asset_id="bright.economics.examples",
        query_processor="bright_examples_parquet_v1",
        split="test",
        corpus_asset_id="bright.economics.corpus",
        corpus_processor="bright_corpus_parquet_v1",
    ),
}


@dataclass(frozen=True, slots=True)
class RetrievalPrediction:
    """One ranked retrieval run represented by document scores."""

    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        converted: dict[str, float] = {}
        for document_id, score in self.scores.items():
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValidationError("retrieval document IDs must be non-empty strings")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValidationError("retrieval scores must be numeric")
            numeric = float(score)
            if not math.isfinite(numeric):
                raise ValidationError("retrieval scores must be finite")
            converted[document_id] = numeric
        object.__setattr__(self, "scores", MappingProxyType(converted))

    @classmethod
    def from_ranked_ids(cls, document_ids: Sequence[str]) -> RetrievalPrediction:
        if isinstance(document_ids, str):
            raise ValidationError("ranked document IDs must be a sequence, not one string")
        ids = tuple(document_ids)
        if len(ids) != len(set(ids)):
            raise ValidationError("ranked retrieval predictions must not contain duplicates")
        size = len(ids)
        return cls({document_id: float(size - rank) for rank, document_id in enumerate(ids)})


class PreparedRetrievalBenchmarkAdapter:
    """Production adapter for a verified, normalized retrieval benchmark."""

    pairing_mode = "stateless_score_twice"

    def __init__(
        self,
        prepared: PreparedBenchmark,
        *,
        benchmark_id: str,
        split: str = "test",
        corpus_prepared: PreparedBenchmark | None = None,
    ) -> None:
        if not isinstance(prepared, PreparedBenchmark):
            raise TypeError("prepared must be a PreparedBenchmark")
        _validated_identity(benchmark_id, label="benchmark_id")
        _validated_identity(split, label="split")
        if corpus_prepared is not None and not isinstance(corpus_prepared, PreparedBenchmark):
            raise TypeError("corpus_prepared must be a PreparedBenchmark")
        named_profile = _NAMED_RETRIEVAL_PROFILES.get(benchmark_id)
        if named_profile is None:
            _validate_prepared_identity(
                prepared,
                benchmark_id=benchmark_id,
                split=split,
                label="prepared retrieval query/gold",
                require_complete=prepared.processor == "bright_examples_parquet_v1",
            )
            if corpus_prepared is not None:
                _validate_prepared_identity(
                    corpus_prepared,
                    benchmark_id=benchmark_id,
                    split=split,
                    label="prepared retrieval corpus",
                    require_complete=True,
                )
        else:
            _validate_named_prepared_identity(
                prepared,
                benchmark_id=benchmark_id,
                split=split,
                expected_asset_id=named_profile.query_asset_id,
                expected_processor=named_profile.query_processor,
                expected_split=named_profile.split,
                label="prepared retrieval query/gold",
            )
            if named_profile.corpus_asset_id is None:
                if corpus_prepared is not None:
                    raise ArtifactError(
                        f"named retrieval profile {benchmark_id!r} does not accept "
                        "a separate corpus"
                    )
            else:
                if corpus_prepared is None:
                    raise ArtifactError(
                        f"named retrieval profile {benchmark_id!r} requires its locked corpus"
                    )
                _validate_named_prepared_identity(
                    corpus_prepared,
                    benchmark_id=benchmark_id,
                    split=split,
                    expected_asset_id=named_profile.corpus_asset_id,
                    expected_processor=named_profile.corpus_processor,
                    expected_split=named_profile.split,
                    label="prepared retrieval corpus",
                )

        input_rows = _records_by_id(prepared.inputs_path, label="prepared retrieval inputs")
        gold_rows = _records_by_id(prepared.gold_path, label="prepared retrieval gold")
        if set(input_rows) != set(gold_rows):
            raise ArtifactError("prepared retrieval input and gold IDs do not match")
        effective_corpus = corpus_prepared or prepared
        corpus_path = effective_corpus.corpus_path
        if corpus_path is None:
            raise ArtifactError("prepared retrieval benchmark requires a corpus")

        examples: list[BenchmarkExample] = []
        for example_id in sorted(input_rows):
            visible = input_rows[example_id]
            evaluator = gold_rows[example_id]
            query = _required_text(visible, "query", label="prepared retrieval input")
            instruction = _required_text(
                visible,
                "instruction",
                label="prepared retrieval input",
            )
            metadata = _visible_metadata(
                visible.get("visible_metadata", {}),
                benchmark_id=benchmark_id,
                split=split,
                require_source_split=(
                    prepared.processor == "bright_examples_parquet_v1"
                    or (named_profile is not None and named_profile.corpus_asset_id is not None)
                ),
            )
            relevance = evaluator.get("relevance")
            if not isinstance(relevance, Mapping) or not relevance:
                raise ArtifactError("prepared retrieval gold requires non-empty relevance labels")
            normalized_relevance: dict[str, int] = {}
            for document_id, value in relevance.items():
                if not isinstance(document_id, str) or not document_id:
                    raise ArtifactError("prepared retrieval qrel IDs must be non-empty strings")
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ArtifactError("prepared retrieval qrels must be non-negative integers")
                normalized_relevance[document_id] = value
            if not any(value > 0 for value in normalized_relevance.values()):
                raise ArtifactError("prepared retrieval example requires a positive qrel")

            evaluator_metadata: Mapping[str, Any] | None = None
            excluded_document_ids: tuple[str, ...] = ()
            if "evaluator_metadata" in evaluator:
                raw_evaluator_metadata = evaluator["evaluator_metadata"]
                if not isinstance(raw_evaluator_metadata, Mapping):
                    raise ArtifactError("prepared retrieval evaluator_metadata must be an object")
                excluded_document_ids = _excluded_document_ids(raw_evaluator_metadata)
                if set(excluded_document_ids) & set(normalized_relevance):
                    raise ArtifactError(
                        "prepared retrieval excluded document IDs overlap relevance labels"
                    )
                evaluator_metadata = _frozen_canonical_mapping(
                    raw_evaluator_metadata,
                    label="prepared retrieval evaluator_metadata",
                )
            evaluator_gold: dict[str, Any] = {"relevance": normalized_relevance}
            if evaluator_metadata is not None:
                evaluator_gold["evaluator_metadata"] = evaluator_metadata
            examples.append(
                BenchmarkExample(
                    example_id=example_id,
                    query=query,
                    instruction=instruction,
                    visible_metadata=metadata,
                    evaluator_gold=_frozen_canonical_mapping(
                        evaluator_gold,
                        label="prepared retrieval evaluator gold",
                    ),
                )
            )
        self.benchmark_id = benchmark_id
        self.split = split
        self.prepared = prepared
        self.query_prepared = prepared
        self.corpus_prepared = effective_corpus
        self.corpus_path = corpus_path
        self._ignore_identical_ids = prepared.processor == "beir_zip_v1"
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self.split:
            raise ValidationError(
                f"prepared retrieval adapter exposes split {self.split!r}, not {split!r}"
            )
        return self._examples

    def example(self, split: str, example_id: str) -> BenchmarkExample:
        """Resolve one verified unit without rescanning the full split."""

        if split != self.split:
            raise ValidationError(
                f"prepared retrieval adapter exposes split {self.split!r}, not {split!r}"
            )
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValidationError("prepared retrieval example_id must be non-empty")
        try:
            return self._examples_by_id[example_id]
        except KeyError as exc:
            raise ValidationError(
                f"prepared retrieval split does not contain example {example_id!r}"
            ) from exc

    def fingerprint(self) -> Mapping[str, Any]:
        """Portable prepared-source identities without exposing qrel contents.

        Hashes were verified while loading the immutable prepared cache. Do not
        rescan a potentially large corpus each time a caller binds a runtime.
        Evaluation dependencies are resolved only by ``score``, not this method.
        """
        def prepared(value: PreparedBenchmark) -> dict[str, Any]:
            return dict(
                asset_id=value.asset_id,
                processor=value.processor,
                processor_revision=value.processor_revision,
                source_sha256=value.source_sha256,
                source_revision=value.source_revision,
                cell_id=value.cell_id,
                split=value.split,
                counts=dict(value.counts),
                file_sha256=dict(value.file_sha256),
            )

        return dict(
            benchmark_id=self.benchmark_id,
            split=self.split,
            query_preparation=prepared(self.query_prepared),
            corpus_preparation=prepared(self.corpus_prepared),
            test_queries=len(self._examples),
            scorer="ir_measures:nDCG@10,R@100",
            ignore_identical_ids=self._ignore_identical_ids,
            excluded_documents_policy="evaluator_metadata.excluded_document_ids_except_N/A",
        )

    def score(
        self,
        example: BenchmarkExample,
        prediction: RetrievalPrediction | Mapping[str, float] | Sequence[str],
    ) -> Mapping[str, float]:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("example must be a BenchmarkExample")
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError(
                "benchmark example does not belong to this prepared retrieval adapter"
            )
        if isinstance(prediction, RetrievalPrediction):
            normalized = prediction
        elif isinstance(prediction, Mapping):
            normalized = RetrievalPrediction(prediction)
        else:
            normalized = RetrievalPrediction.from_ranked_ids(prediction)
        relevance = example.evaluator_gold.get("relevance")
        if not isinstance(relevance, Mapping):
            raise ArtifactError("benchmark example does not contain retrieval qrels")
        evaluator_metadata = example.evaluator_gold.get("evaluator_metadata", {})
        if not isinstance(evaluator_metadata, Mapping):  # pragma: no cover - construction invariant
            raise ArtifactError("benchmark example evaluator metadata is invalid")
        raw_excluded = evaluator_metadata.get("excluded_document_ids", ())
        if not isinstance(raw_excluded, tuple):  # pragma: no cover - construction invariant
            raise ArtifactError("benchmark example excluded document IDs are invalid")
        excluded = frozenset(item for item in raw_excluded if item != "N/A")
        # BRIGHT's official get_scores removes excluded documents before
        # truncation/evaluation.  Apply the same boundary here so callers may
        # submit raw retriever scores without accidentally changing ranks.
        # BEIR's default evaluator also removes query/document pairs with an
        # identical ID; this is material for ArguAna, whose corpus reuses most
        # query IDs.  Keep this profile-specific rather than applying it to
        # unrelated retrieval suites.
        evaluation_scores = {
            document_id: score
            for document_id, score in normalized.scores.items()
            if document_id not in excluded
            and not (self._ignore_identical_ids and document_id == example.example_id)
        }
        return retrieval_aggregates(
            {example.example_id: relevance},
            {example.example_id: evaluation_scores},
            measures={
                "ndcg_at_10": "nDCG@10",
                "recall": "R@100",
            },
        )


def _records_by_id(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise ArtifactError(f"{label} contains a blank line at {line_number}")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ArtifactError(f"{label} row {line_number} must be an object")
                row_id = _required_text(value, "id", label=label)
                if row_id in records:
                    raise ArtifactError(f"{label} contains duplicate ID {row_id!r}")
                records[row_id] = value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{label} is not valid UTF-8 JSONL") from exc
    if not records:
        raise ArtifactError(f"{label} contains no records")
    return records


def _required_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} field {key!r} must be a non-empty string")
    return value


def _validated_identity(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty")


def _validate_prepared_identity(
    prepared: PreparedBenchmark,
    *,
    benchmark_id: str,
    split: str,
    label: str,
    require_complete: bool,
) -> None:
    if require_complete and not prepared.cell_id:
        raise ArtifactError(f"{label} requires an explicit cell_id")
    if require_complete and not prepared.split:
        raise ArtifactError(f"{label} requires an explicit split")
    if prepared.cell_id and prepared.cell_id != benchmark_id:
        raise ArtifactError(f"{label} cell_id does not match adapter benchmark_id")
    if prepared.split and prepared.split != split:
        raise ArtifactError(f"{label} split does not match adapter split")


def _validate_named_prepared_identity(
    prepared: PreparedBenchmark,
    *,
    benchmark_id: str,
    split: str,
    expected_asset_id: str,
    expected_processor: str | None,
    expected_split: str,
    label: str,
) -> None:
    if split != expected_split:
        raise ArtifactError(
            f"named retrieval profile {benchmark_id!r} requires adapter split {expected_split!r}"
        )
    expected_fields = {
        "asset_id": expected_asset_id,
        "processor": expected_processor,
        "cell_id": benchmark_id,
        "split": expected_split,
    }
    for field, expected_value in expected_fields.items():
        if getattr(prepared, field) != expected_value:
            raise ArtifactError(f"{label} {field} does not match named profile {benchmark_id!r}")


def _visible_metadata(
    value: Any,
    *,
    benchmark_id: str,
    split: str,
    require_source_split: bool,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError("prepared retrieval visible_metadata must be an object")
    if value.get("benchmark_cell") != benchmark_id:
        raise ArtifactError("prepared retrieval benchmark_cell does not match adapter benchmark_id")
    source_split = value.get("source_split")
    if require_source_split and source_split is None:
        raise ArtifactError("prepared retrieval visible_metadata requires source_split")
    if source_split is not None and source_split != split:
        raise ArtifactError("prepared retrieval source_split does not match adapter split")
    return _frozen_canonical_mapping(value, label="prepared retrieval visible_metadata")


def _excluded_document_ids(evaluator_metadata: Mapping[str, Any]) -> tuple[str, ...]:
    value = evaluator_metadata.get("excluded_document_ids", [])
    if not isinstance(value, list):
        raise ArtifactError("prepared retrieval excluded_document_ids must be a JSON array")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ArtifactError(
            "prepared retrieval excluded_document_ids must contain non-empty strings"
        )
    normalized = tuple(item for item in value if item != "N/A")
    if len(normalized) != len(set(normalized)):
        raise ArtifactError("prepared retrieval excluded_document_ids must not contain duplicates")
    return normalized


def _frozen_canonical_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    try:
        frozen = freeze_mapping(value)
        canonical_json(frozen)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} is not canonical: {exc}") from exc
    return frozen


__all__ = ["PreparedRetrievalBenchmarkAdapter", "RetrievalPrediction"]
