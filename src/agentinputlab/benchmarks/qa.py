"""Prepared question-answering adapter with evaluator-only references.

The normalized input schema is intentionally small and dataset-neutral.  A
visible row contains the user question plus optional context blocks, while its
matching gold row contains either one reference group or a set of annotation-
aligned reference groups.  Keeping those files separate prevents an
intervention or prediction callback from observing answers during execution.
"""

from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, freeze_mapping
from agentinputlab.contracts.state import ContextBlock
from agentinputlab.metrics import (
    best_reference_score,
    english_qa_exact_match,
    english_qa_f1,
    normalize_english_answer,
    set_f1,
)

from .base import BenchmarkExample
from .preprocess import PreparedBenchmark

_INPUT_FIELDS = {"id", "query", "instruction", "visible_context", "visible_metadata"}
_CONTEXT_FIELDS = {"block_id", "text", "source", "score", "metadata"}
_REFERENCE_FIELDS = {
    "answer_type",
    "answers",
    "support_ids",
    "evidence_ids",
    "evidence_texts",
}
_BENCHMARK_PROFILES = {
    "musique.answerable": "musique",
    "hotpotqa.distractor": "hotpotqa",
    "2wiki.multihop": "2wiki",
    "qasper.v0_3_dev": "qasper",
}
_NAMED_PROFILE_IDENTITIES = {
    "musique.answerable": ("musique.answerable.dev", "musique_zip_v1", "dev"),
    "hotpotqa.distractor": (
        "hotpotqa.distractor.validation",
        "hotpotqa_parquet_v1",
        "validation",
    ),
    "2wiki.multihop": ("2wiki.multihop.dev", "two_wiki_zip_v1", "dev"),
    "qasper.v0_3_dev": ("qasper.dev.official", "qasper_tgz_v1", "dev"),
}
_CATEGORICAL_ANSWERS = frozenset({"yes", "no", "noanswer"})
_QASPER_ANSWER_TYPES = frozenset({"extractive", "abstractive", "boolean", "none"})
_PROFILE_METRIC_KEYS = {
    "musique": ("answer_em", "answer_f1", "support_f1"),
    "hotpotqa": (
        "answer_em",
        "answer_f1",
        "answer_precision",
        "answer_recall",
        "support_em",
        "support_f1",
        "support_precision",
        "support_recall",
        "joint_em",
        "joint_f1",
        "joint_precision",
        "joint_recall",
    ),
    "2wiki": (
        "answer_em",
        "answer_f1",
        "answer_precision",
        "answer_recall",
        "support_em",
        "support_f1",
        "support_precision",
        "support_recall",
        "evidence_em",
        "evidence_f1",
        "evidence_precision",
        "evidence_recall",
        "joint_em",
        "joint_f1",
        "joint_precision",
        "joint_recall",
    ),
    "qasper": ("answer_em", "answer_f1", "evidence_f1"),
}


@dataclass(frozen=True, slots=True)
class QAPrediction:
    """One answer with optional support IDs and ID- or text-based evidence."""

    answer: str
    support_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_texts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise ValidationError("QA prediction answer must be a string")
        object.__setattr__(
            self,
            "support_ids",
            _prediction_identifiers(self.support_ids, label="QA prediction support_ids"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _prediction_identifiers(self.evidence_ids, label="QA prediction evidence_ids"),
        )
        object.__setattr__(
            self,
            "evidence_texts",
            _prediction_identifiers(self.evidence_texts, label="QA prediction evidence_texts"),
        )


class PreparedQABenchmarkAdapter:
    """Production adapter for a verified, normalized QA benchmark.

    Input rows must use this exact schema::

        {
          "id": "...",
          "query": "...",
          "instruction": "...",
          "visible_context": [
            {"block_id": "...", "text": "...", "source": "...",
             "score": null, "metadata": {}}
          ],
          "visible_metadata": {
            "benchmark_cell": "...",
            "source_split": "..."
          }
        }

    A gold row uses either ``answers`` with optional ``support_ids`` and one of
    ``evidence_ids``/``evidence_texts``, or ``reference_groups`` containing
    those same fields.  The latter preserves annotation alignment for datasets
    such as QASPER; its official evidence score uses the text-based variant.
    """

    pairing_mode = "stateless_score_twice"

    def __init__(
        self,
        prepared: PreparedBenchmark,
        *,
        benchmark_id: str,
        split: str,
    ) -> None:
        if not isinstance(prepared, PreparedBenchmark):
            raise TypeError("prepared must be a PreparedBenchmark")
        _validated_identity(benchmark_id, label="benchmark_id")
        _validated_identity(split, label="split")
        _validate_prepared_identity(
            prepared,
            benchmark_id=benchmark_id,
            split=split,
        )

        input_rows = _records_by_id(prepared.inputs_path, label="prepared QA inputs")
        gold_rows = _records_by_id(prepared.gold_path, label="prepared QA gold")
        if set(input_rows) != set(gold_rows):
            raise ArtifactError("prepared QA input and gold IDs do not match")

        examples: list[BenchmarkExample] = []
        benchmark_profile = _BENCHMARK_PROFILES.get(benchmark_id, "generic")
        metric_profile: tuple[bool, str | None] | None = None
        for example_id in sorted(input_rows):
            visible = input_rows[example_id]
            evaluator = gold_rows[example_id]
            _require_exact_fields(visible, _INPUT_FIELDS, label="prepared QA input")
            query = _required_text(visible, "query", label="prepared QA input")
            instruction = _required_text(visible, "instruction", label="prepared QA input")
            context = _visible_context(visible["visible_context"], example_id=example_id)
            metadata = _visible_metadata(
                visible["visible_metadata"],
                benchmark_id=benchmark_id,
                split=split,
            )
            references, evaluator_metadata, profile = _reference_groups(evaluator)
            _validate_benchmark_profile(
                benchmark_profile,
                references=references,
                evaluator_metadata=evaluator_metadata,
            )
            if metric_profile is None:
                metric_profile = profile
            elif metric_profile != profile:
                raise ArtifactError(
                    "prepared QA gold rows must expose one consistent support/evidence schema"
                )
            examples.append(
                BenchmarkExample(
                    example_id=example_id,
                    query=query,
                    instruction=instruction,
                    visible_context=context,
                    visible_metadata=metadata,
                    evaluator_gold=_frozen_canonical_mapping(
                        {
                            "reference_groups": references,
                            "evaluator_metadata": evaluator_metadata,
                        },
                        label="prepared QA evaluator gold",
                    ),
                )
            )

        self.benchmark_id = benchmark_id
        self.split = split
        self.prepared = prepared
        self.metric_profile = benchmark_profile
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}
        self._has_support, self._evidence_field = metric_profile or (False, None)
        self.metric_keys = _PROFILE_METRIC_KEYS.get(
            benchmark_profile,
            (
                "exact_match",
                "answer_f1",
                *(("support_f1",) if self._has_support else ()),
                *(("evidence_f1",) if self._evidence_field is not None else ()),
            ),
        )

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self.split:
            raise ValidationError(
                f"prepared QA adapter exposes split {self.split!r}, not {split!r}"
            )
        return self._examples

    def example(self, split: str, example_id: str) -> BenchmarkExample:
        """Resolve one verified unit without rescanning the full split."""

        if split != self.split:
            raise ValidationError(
                f"prepared QA adapter exposes split {self.split!r}, not {split!r}"
            )
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValidationError("prepared QA example_id must be non-empty")
        try:
            return self._examples_by_id[example_id]
        except KeyError as exc:
            raise ValidationError(
                f"prepared QA split does not contain example {example_id!r}"
            ) from exc

    def score(
        self,
        example: BenchmarkExample,
        prediction: QAPrediction | Mapping[str, Any],
    ) -> Mapping[str, float]:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("example must be a BenchmarkExample")
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this prepared QA adapter")
        normalized = (
            prediction
            if isinstance(prediction, QAPrediction)
            else _prediction_from_mapping(prediction)
        )
        raw_groups = example.evaluator_gold.get("reference_groups")
        if not isinstance(raw_groups, tuple) or not raw_groups:
            raise ArtifactError("benchmark example does not contain QA reference groups")

        groups = _validated_runtime_groups(raw_groups)
        evaluator_metadata = example.evaluator_gold.get("evaluator_metadata")
        if not isinstance(evaluator_metadata, Mapping):  # pragma: no cover - invariant
            raise ArtifactError("benchmark example does not contain QA evaluator metadata")

        if self.metric_profile == "musique":
            return _score_musique(normalized, groups)
        if self.metric_profile == "hotpotqa":
            return _score_hotpotqa(normalized, groups)
        if self.metric_profile == "2wiki":
            return _score_two_wiki(normalized, groups, evaluator_metadata)
        if self.metric_profile == "qasper":
            return _score_qasper(normalized, groups)
        return _score_generic(
            normalized,
            groups,
            has_support=self._has_support,
            evidence_field=self._evidence_field,
        )


def _score_generic(
    prediction: QAPrediction,
    groups: tuple[Mapping[str, Any], ...],
    *,
    has_support: bool,
    evidence_field: str | None,
) -> Mapping[str, float]:
    answer_exact: list[float] = []
    answer_f1: list[float] = []
    support_f1: list[float] = []
    evidence_f1: list[float] = []
    for group in groups:
        answers = group["answers"]
        answer_exact.append(
            best_reference_score(
                prediction.answer,
                answers,
                scorer=english_qa_exact_match,
            )
        )
        answer_f1.append(best_reference_score(prediction.answer, answers, scorer=english_qa_f1))
        if has_support:
            support_f1.append(set_f1(prediction.support_ids, group["support_ids"]))
        if evidence_field == "evidence_ids":
            evidence_f1.append(set_f1(prediction.evidence_ids, group["evidence_ids"]))
        elif evidence_field == "evidence_texts":
            evidence_f1.append(set_f1(prediction.evidence_texts, group["evidence_texts"]))

    metrics = {
        "exact_match": max(answer_exact),
        "answer_f1": max(answer_f1),
    }
    if has_support:
        metrics["support_f1"] = max(support_f1)
    if evidence_field is not None:
        metrics["evidence_f1"] = max(evidence_f1)
    return metrics


def _score_musique(
    prediction: QAPrediction,
    groups: tuple[Mapping[str, Any], ...],
) -> Mapping[str, float]:
    """Match the official answerable-split answer and paragraph metrics."""

    answer_em = max(
        best_reference_score(
            prediction.answer,
            group["answers"],
            scorer=english_qa_exact_match,
        )
        for group in groups
    )
    answer_f1 = max(
        best_reference_score(prediction.answer, group["answers"], scorer=english_qa_f1)
        for group in groups
    )
    support_f1 = max(set_f1(prediction.support_ids, group["support_ids"]) for group in groups)
    return {
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "support_f1": support_f1,
    }


def _score_hotpotqa(
    prediction: QAPrediction,
    groups: tuple[Mapping[str, Any], ...],
) -> Mapping[str, float]:
    answer = _best_answer_scores(prediction.answer, groups, categorical_guard=True)
    support = _best_support_scores(prediction.support_ids, groups, lowercase_titles=False)
    return _joint_metrics(answer, support)


def _score_two_wiki(
    prediction: QAPrediction,
    groups: tuple[Mapping[str, Any], ...],
    evaluator_metadata: Mapping[str, Any],
) -> Mapping[str, float]:
    answer = _best_answer_scores(prediction.answer, groups, categorical_guard=True)
    support = _best_support_scores(prediction.support_ids, groups, lowercase_titles=True)
    evidence = _two_wiki_evidence_scores(
        prediction.evidence_ids,
        evaluator_metadata.get("evidence_options"),
    )
    return _joint_metrics(answer, support, evidence)


def _score_qasper(
    prediction: QAPrediction,
    groups: tuple[Mapping[str, Any], ...],
) -> Mapping[str, float]:
    """Match QASPER's independent max-over-annotations scoring rule."""

    return {
        "answer_em": max(
            best_reference_score(
                prediction.answer,
                group["answers"],
                scorer=english_qa_exact_match,
            )
            for group in groups
        ),
        "answer_f1": max(
            best_reference_score(prediction.answer, group["answers"], scorer=english_qa_f1)
            for group in groups
        ),
        "evidence_f1": max(
            set_f1(prediction.evidence_texts, group["evidence_texts"]) for group in groups
        ),
    }


def _best_answer_scores(
    prediction: str,
    groups: tuple[Mapping[str, Any], ...],
    *,
    categorical_guard: bool,
) -> Mapping[str, float]:
    scores = [
        _answer_scores(prediction, answer, categorical_guard=categorical_guard)
        for group in groups
        for answer in group["answers"]
    ]
    return {key: max(score[key] for score in scores) for key in ("em", "f1", "precision", "recall")}


def _answer_scores(
    prediction: str,
    reference: str,
    *,
    categorical_guard: bool,
) -> Mapping[str, float]:
    normalized_prediction = _normalized_answer(prediction)
    normalized_reference = _normalized_answer(reference)
    exact_match = float(normalized_prediction == normalized_reference)
    if categorical_guard and (
        normalized_prediction != normalized_reference
        and (
            normalized_prediction in _CATEGORICAL_ANSWERS
            or normalized_reference in _CATEGORICAL_ANSWERS
        )
    ):
        return {"em": exact_match, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    prediction_tokens = normalized_prediction.split()
    reference_tokens = normalized_reference.split()
    common = set(prediction_tokens) & set(reference_tokens)
    # Token multiplicity matters in the official SQuAD/Hotpot scorer.
    matched = sum(
        min(prediction_tokens.count(token), reference_tokens.count(token)) for token in common
    )
    if matched == 0:
        return {"em": exact_match, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    precision = matched / len(prediction_tokens)
    recall = matched / len(reference_tokens)
    return {
        "em": exact_match,
        "f1": _harmonic_mean(precision, recall),
        "precision": precision,
        "recall": recall,
    }


def _best_support_scores(
    prediction_ids: tuple[str, ...],
    groups: tuple[Mapping[str, Any], ...],
    *,
    lowercase_titles: bool,
) -> Mapping[str, float]:
    predicted = _support_pairs(
        prediction_ids,
        lowercase_titles=lowercase_titles,
        prediction=True,
    )
    scores = [
        _set_scores(
            predicted,
            _support_pairs(
                group["support_ids"],
                lowercase_titles=lowercase_titles,
                prediction=False,
            ),
        )
        for group in groups
    ]
    return {key: max(score[key] for score in scores) for key in ("em", "f1", "precision", "recall")}


def _support_pairs(
    identifiers: Sequence[str],
    *,
    lowercase_titles: bool,
    prediction: bool,
) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for identifier in identifiers:
        value = _canonical_identifier(
            identifier,
            expected_length=2,
            label="support ID",
            prediction=prediction,
        )
        title, sentence_index = value
        if not isinstance(title, str) or not title.strip():
            _identifier_error("support ID title must be a non-empty string", prediction)
        if (
            isinstance(sentence_index, bool)
            or not isinstance(sentence_index, int)
            or sentence_index < 0
        ):
            _identifier_error("support ID index must be a non-negative integer", prediction)
        pairs.add((title.lower() if lowercase_titles else title, sentence_index))
    return pairs


def _two_wiki_evidence_scores(
    prediction_ids: tuple[str, ...],
    raw_options: Any,
) -> Mapping[str, float]:
    if not isinstance(raw_options, tuple) or not raw_options:
        raise ArtifactError("2Wiki evaluator metadata requires non-empty evidence_options")
    accepted_groups: list[set[tuple[str, str, str]]] = []
    for group_index, raw_group in enumerate(raw_options):
        if not isinstance(raw_group, tuple) or not raw_group:
            raise ArtifactError(f"2Wiki evidence_options group {group_index} must be non-empty")
        accepted_groups.append(
            {
                _normalized_evidence_triple(
                    option,
                    prediction=False,
                )
                for option in raw_group
            }
        )

    # The official v1.1 evaluator normalizes first and then converts predictions
    # to a set.  Without that second deduplication, case/punctuation variants of
    # one triple can be counted more than once and push recall/F1 above 1.
    predicted = {
        _normalized_evidence_triple(identifier, prediction=True) for identifier in prediction_ids
    }
    matched = sum(
        1 for candidate in predicted if any(candidate in group for group in accepted_groups)
    )
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(accepted_groups) if accepted_groups else 0.0
    return {
        "em": float(matched == len(predicted) == len(accepted_groups)),
        "f1": _harmonic_mean(precision, recall),
        "precision": precision,
        "recall": recall,
    }


def _normalized_evidence_triple(
    identifier: str,
    *,
    prediction: bool,
) -> tuple[str, str, str]:
    value = _canonical_identifier(
        identifier,
        expected_length=3,
        label="evidence ID",
        prediction=prediction,
    )
    if not all(isinstance(item, str) and item.strip() for item in value):
        _identifier_error("evidence ID must contain three non-empty strings", prediction)
    subject, relation, object_ = value
    return (
        _normalize_evidence_component(subject),
        _normalize_evidence_component(relation),
        _normalize_evidence_component(object_),
    )


def _canonical_identifier(
    identifier: str,
    *,
    expected_length: int,
    label: str,
    prediction: bool,
) -> list[Any]:
    try:
        value = json.loads(identifier)
    except json.JSONDecodeError:
        _identifier_error(f"{label} must be canonical JSON", prediction)
    if (
        not isinstance(value, list)
        or len(value) != expected_length
        or canonical_json(value) != identifier
    ):
        _identifier_error(
            f"{label} must be a canonical JSON array of length {expected_length}",
            prediction,
        )
    return value


def _identifier_error(message: str, prediction: bool) -> None:
    if prediction:
        raise ValidationError(f"QA prediction {message}")
    raise ArtifactError(f"prepared QA gold {message}")


def _set_scores(prediction: set[Any], reference: set[Any]) -> Mapping[str, float]:
    matched = len(prediction & reference)
    precision = matched / len(prediction) if prediction else 0.0
    recall = matched / len(reference) if reference else 0.0
    return {
        "em": float(prediction == reference),
        "f1": _harmonic_mean(precision, recall),
        "precision": precision,
        "recall": recall,
    }


def _joint_metrics(*components: Mapping[str, float]) -> Mapping[str, float]:
    prefixes = ("answer", "support", "evidence")
    metrics: dict[str, float] = {}
    for prefix, component in zip(prefixes[: len(components)], components, strict=True):
        for metric in ("em", "f1", "precision", "recall"):
            metrics[f"{prefix}_{metric}"] = component[metric]
    joint_em = math.prod(component["em"] for component in components)
    joint_precision = math.prod(component["precision"] for component in components)
    joint_recall = math.prod(component["recall"] for component in components)
    metrics.update(
        {
            "joint_em": joint_em,
            "joint_f1": _harmonic_mean(joint_precision, joint_recall),
            "joint_precision": joint_precision,
            "joint_recall": joint_recall,
        }
    )
    return metrics


def _harmonic_mean(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _normalized_answer(value: str) -> str:
    # Keep one source of truth for SQuAD/Hotpot answer normalization.
    return normalize_english_answer(value)


def _normalize_evidence_component(value: str) -> str:
    without_punctuation = "".join(
        character for character in value.lower() if character not in string.punctuation
    )
    return " ".join(without_punctuation.split())


def _validated_runtime_groups(value: tuple[Any, ...]) -> tuple[Mapping[str, Any], ...]:
    groups: list[Mapping[str, Any]] = []
    for group in value:
        if not isinstance(group, Mapping):  # pragma: no cover - construction invariant
            raise ArtifactError("benchmark QA reference group is invalid")
        groups.append(group)
    return tuple(groups)


def _validate_prepared_identity(
    prepared: PreparedBenchmark,
    *,
    benchmark_id: str,
    split: str,
) -> None:
    expected = _NAMED_PROFILE_IDENTITIES.get(benchmark_id)
    if expected is None:
        # Direct construction remains available for custom datasets.  Preserve
        # the original compatibility contract: custom identities may be absent,
        # but identities that are supplied must agree with the adapter request.
        if prepared.cell_id and prepared.cell_id != benchmark_id:
            raise ArtifactError("prepared QA cell_id does not match adapter benchmark_id")
        if prepared.split and prepared.split != split:
            raise ArtifactError("prepared QA split does not match adapter split")
        return

    expected_asset_id, expected_processor, expected_split = expected
    if split != expected_split:
        raise ArtifactError(
            f"named QA profile {benchmark_id!r} requires adapter split {expected_split!r}"
        )
    expected_fields = {
        "asset_id": expected_asset_id,
        "processor": expected_processor,
        "cell_id": benchmark_id,
        "split": expected_split,
    }
    for field, expected_value in expected_fields.items():
        if getattr(prepared, field) != expected_value:
            raise ArtifactError(
                f"prepared QA {field} does not match named profile {benchmark_id!r}"
            )


def _validate_benchmark_profile(
    profile: str,
    *,
    references: tuple[Mapping[str, Any], ...],
    evaluator_metadata: Mapping[str, Any],
) -> None:
    if profile in {"musique", "hotpotqa"}:
        if any(
            "support_ids" not in group or "evidence_ids" in group or "evidence_texts" in group
            for group in references
        ):
            raise ArtifactError(f"{profile} gold requires support_ids and no evidence")
    elif profile == "2wiki":
        if any("support_ids" not in group or "evidence_ids" not in group for group in references):
            raise ArtifactError("2Wiki gold requires support_ids and evidence_ids")
        options = evaluator_metadata.get("evidence_options")
        if not isinstance(options, tuple) or not options:
            raise ArtifactError("2Wiki evaluator metadata requires non-empty evidence_options")
        if any(len(options) != len(group["evidence_ids"]) for group in references):
            raise ArtifactError("2Wiki evidence_options must align with evidence_ids")
    elif profile == "qasper" and any(
        group.get("answer_type") not in _QASPER_ANSWER_TYPES
        or "evidence_texts" not in group
        or "support_ids" in group
        for group in references
    ):
        raise ArtifactError(
            "QASPER gold requires a valid answer_type and evidence_texts per annotation"
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


def _visible_context(value: Any, *, example_id: str) -> tuple[ContextBlock, ...]:
    if not isinstance(value, list):
        raise ArtifactError("prepared QA visible_context must be a JSON array")
    blocks: list[ContextBlock] = []
    block_ids: set[str] = set()
    for index, item in enumerate(value):
        label = f"prepared QA context {example_id!r} item {index}"
        if not isinstance(item, dict):
            raise ArtifactError(f"{label} must be an object")
        unknown = set(item) - _CONTEXT_FIELDS
        if unknown:
            raise ArtifactError(f"{label} has unknown fields: {sorted(unknown)!r}")
        if not {"block_id", "text"} <= set(item):
            raise ArtifactError(f"{label} requires block_id and text")
        block_id = _required_text(item, "block_id", label=label)
        if block_id in block_ids:
            raise ArtifactError(f"prepared QA context contains duplicate block ID {block_id!r}")
        block_ids.add(block_id)
        text = _required_text(item, "text", label=label)
        source = item.get("source", "")
        if not isinstance(source, str):
            raise ArtifactError(f"{label} field 'source' must be a string")
        score = item.get("score")
        numeric_score: float | None = None
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ArtifactError(f"{label} field 'score' must be finite numeric or null")
            try:
                numeric_score = float(score)
            except OverflowError as exc:
                raise ArtifactError(
                    f"{label} field 'score' must be finite numeric or null"
                ) from exc
            if not math.isfinite(numeric_score):
                raise ArtifactError(f"{label} field 'score' must be finite numeric or null")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ArtifactError(f"{label} field 'metadata' must be an object")
        try:
            block = ContextBlock(
                block_id=block_id,
                text=text,
                source=source,
                score=numeric_score,
                metadata=_frozen_canonical_mapping(metadata, label=f"{label} metadata"),
            )
        except (TypeError, ValidationError) as exc:
            raise ArtifactError(f"{label} is invalid: {exc}") from exc
        blocks.append(block)
    return tuple(blocks)


def _visible_metadata(
    value: Any,
    *,
    benchmark_id: str,
    split: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError("prepared QA visible_metadata must be an object")
    if value.get("benchmark_cell") != benchmark_id:
        raise ArtifactError("prepared QA benchmark_cell does not match adapter benchmark_id")
    if value.get("source_split") != split:
        raise ArtifactError("prepared QA source_split does not match adapter split")
    return _frozen_canonical_mapping(value, label="prepared QA visible_metadata")


def _reference_groups(
    evaluator: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any], tuple[bool, str | None]]:
    raw_evaluator_metadata = evaluator.get("evaluator_metadata", {})
    if not isinstance(raw_evaluator_metadata, Mapping):
        raise ArtifactError("prepared QA evaluator_metadata must be an object")
    evaluator_metadata = _frozen_canonical_mapping(
        raw_evaluator_metadata,
        label="prepared QA evaluator_metadata",
    )
    if "reference_groups" in evaluator:
        if not set(evaluator) <= {"id", "reference_groups", "evaluator_metadata"}:
            raise ArtifactError(
                "prepared QA grouped gold permits only id, reference_groups, and evaluator_metadata"
            )
        raw_groups = evaluator["reference_groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ArtifactError("prepared QA reference_groups must be a non-empty JSON array")
    else:
        fields = set(evaluator)
        allowed = {"id", "evaluator_metadata"} | _REFERENCE_FIELDS
        if not {"id", "answers"} <= fields or not fields <= allowed:
            raise ArtifactError("prepared QA gold has an invalid flat reference schema")
        raw_groups = [
            {key: evaluator[key] for key in fields if key not in {"id", "evaluator_metadata"}}
        ]

    groups: list[Mapping[str, Any]] = []
    profile: tuple[bool, str | None] | None = None
    for index, raw_group in enumerate(raw_groups):
        label = f"prepared QA reference group {index}"
        if not isinstance(raw_group, dict):
            raise ArtifactError(f"{label} must be an object")
        fields = set(raw_group)
        if "answers" not in fields or not fields <= _REFERENCE_FIELDS:
            raise ArtifactError(f"{label} has an invalid schema")
        answers = _answers(raw_group["answers"], label=label)
        normalized: dict[str, Any] = {"answers": answers}
        if "answer_type" in fields:
            normalized["answer_type"] = _required_text(
                raw_group,
                "answer_type",
                label=label,
            )
        has_support = "support_ids" in fields
        evidence_fields = fields & {"evidence_ids", "evidence_texts"}
        if len(evidence_fields) > 1:
            raise ArtifactError(f"{label} cannot mix evidence_ids and evidence_texts")
        evidence_field = next(iter(evidence_fields), None)
        if has_support:
            normalized["support_ids"] = _gold_identifiers(
                raw_group["support_ids"], label=f"{label} support_ids"
            )
        if evidence_field is not None:
            normalized[evidence_field] = _gold_identifiers(
                raw_group[evidence_field], label=f"{label} {evidence_field}"
            )
        current_profile = (has_support, evidence_field)
        if profile is None:
            profile = current_profile
        elif profile != current_profile:
            raise ArtifactError("prepared QA reference groups must use one support/evidence schema")
        groups.append(_frozen_canonical_mapping(normalized, label=label))
    return tuple(groups), evaluator_metadata, profile or (False, None)


def _answers(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ArtifactError(f"{label} answers must be a non-empty JSON array")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ArtifactError(f"{label} answers must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ArtifactError(f"{label} answers must not contain duplicates")
    return tuple(value)


def _gold_identifiers(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ArtifactError(f"{label} must be a JSON array")
    try:
        return _identifier_tuple(value, label=label)
    except ValidationError as exc:
        raise ArtifactError(str(exc)) from exc


def _prediction_identifiers(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValidationError(f"{label} must be a sequence, not one string")
    return _identifier_tuple(value, label=label)


def _identifier_tuple(value: Sequence[Any], *, label: str) -> tuple[str, ...]:
    identifiers = tuple(value)
    if not all(isinstance(item, str) and item.strip() for item in identifiers):
        raise ValidationError(f"{label} must contain non-empty strings")
    if len(identifiers) != len(set(identifiers)):
        raise ValidationError(f"{label} must not contain duplicates")
    return identifiers


def _prediction_from_mapping(value: Mapping[str, Any]) -> QAPrediction:
    if not isinstance(value, Mapping):
        raise TypeError("QA prediction must be QAPrediction or a mapping")
    fields = set(value)
    if "answer" not in fields or not fields <= {
        "answer",
        "support_ids",
        "evidence_ids",
        "evidence_texts",
    }:
        raise ValidationError("QA prediction mapping has an invalid schema")
    return QAPrediction(
        answer=value["answer"],
        support_ids=value.get("support_ids", ()),
        evidence_ids=value.get("evidence_ids", ()),
        evidence_texts=value.get("evidence_texts", ()),
    )


def _require_exact_fields(record: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    if set(record) != fields:
        missing = sorted(fields - set(record))
        unknown = sorted(set(record) - fields)
        raise ArtifactError(f"{label} schema mismatch: missing={missing!r}, unknown={unknown!r}")


def _required_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} field {key!r} must be a non-empty string")
    return value


def _validated_identity(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty")


def _frozen_canonical_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    try:
        frozen = freeze_mapping(value)
        canonical_json(frozen)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} is not canonical: {exc}") from exc
    return frozen


__all__ = ["PreparedQABenchmarkAdapter", "QAPrediction"]
