"""Lazy adapters to established metric implementations.

The core package stays dependency-free.  Install ``agentinputlab[evaluation]``
to use these adapters; benchmark runs record the resolved package versions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from agentinputlab.contracts.errors import ExternalDependencyError

_IR_EXPRESSION = re.compile(r"^(nDCG|R|AP)(?:@([1-9][0-9]*))?$")


@dataclass(frozen=True, slots=True)
class RougeScore:
    precision: float
    recall: float
    f1: float


def retrieval_aggregates(
    qrels: Mapping[str, Mapping[str, int | float]],
    run: Mapping[str, Mapping[str, int | float]],
    *,
    measures: Mapping[str, str] | None = None,
) -> Mapping[str, float]:
    """Compute aggregate IR measures through ``ir-measures`` providers.

    ``measures`` maps stable output names to ir-measures expressions, for
    example ``{"ndcg_at_10": "nDCG@10", "recall_at_100": "R@100"}``.
    """

    ir_measures = _optional_import("ir_measures", extra="evaluation")
    normalized_qrels = _validate_nested_scores(qrels, label="qrels", require_integer=True)
    normalized_run = _validate_nested_scores(run, label="run", require_integer=False)
    if not normalized_qrels:
        raise ValueError("qrels must contain at least one query")
    specifications = dict(
        measures
        or {
            "ndcg_at_10": "nDCG@10",
            "recall_at_100": "R@100",
            "map": "AP",
        }
    )
    if not specifications:
        raise ValueError("at least one retrieval measure is required")
    parsed: dict[str, Any] = {}
    for name, expression in specifications.items():
        if not isinstance(name, str) or not name:
            raise ValueError("retrieval measure names must be non-empty strings")
        if not isinstance(expression, str) or not expression:
            raise ValueError("retrieval measure expressions must be non-empty strings")
        parsed[name] = _parse_ir_measure(ir_measures, expression)
    aggregate = ir_measures.calc_aggregate(
        list(parsed.values()),
        normalized_qrels,
        normalized_run,
    )
    return {name: float(aggregate[measure]) for name, measure in parsed.items()}


def rouge_l(
    prediction: str,
    references: Sequence[str],
    *,
    use_stemmer: bool = True,
) -> RougeScore:
    """Return the best-reference ROUGE-L score via Google's ``rouge-score``."""

    rouge_scorer = _optional_import("rouge_score.rouge_scorer", extra="evaluation")
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if isinstance(references, str):
        raise TypeError("ROUGE references must be a sequence of strings, not one string")
    if not references:
        raise ValueError("at least one ROUGE reference is required")
    if not all(isinstance(reference, str) for reference in references):
        raise TypeError("ROUGE references must be strings")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
    candidates = [scorer.score(reference, prediction)["rougeL"] for reference in references]
    best = max(candidates, key=lambda score: (score.fmeasure, score.recall, score.precision))
    return RougeScore(
        precision=float(best.precision),
        recall=float(best.recall),
        f1=float(best.fmeasure),
    )


def evaluation_dependency_versions() -> Mapping[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("ir-measures", "pytrec-eval-terrier", "rouge-score"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _optional_import(module: str, *, extra: str) -> Any:
    try:
        return __import__(module, fromlist=[module.rsplit(".", 1)[-1]])
    except ImportError as exc:
        raise ExternalDependencyError(
            f"optional metric dependency {module!r} is unavailable; install agentinputlab[{extra}]"
        ) from exc


def _parse_ir_measure(ir_measures: Any, expression: str) -> Any:
    """Parse the audited IR subset without relying on Python AST internals.

    ``ir_measures.parse_measure`` versions through 0.4.3 still reference
    ``ast.Num``, which Python 3.14 removed.  Constructing the public measure
    objects directly is both narrower and cross-version stable.
    """

    match = _IR_EXPRESSION.fullmatch(expression)
    if match is None:
        raise ValueError(
            f"invalid ir-measures expression: {expression}; supported: nDCG, R, AP with optional @k"
        )
    measure = getattr(ir_measures, match.group(1))
    cutoff = match.group(2)
    return measure if cutoff is None else measure @ int(cutoff)


def _validate_nested_scores(
    value: Mapping[str, Mapping[str, int | float]],
    *,
    label: str,
    require_integer: bool,
) -> dict[str, dict[str, int | float]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, dict[str, int | float]] = {}
    for query_id, documents in value.items():
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{label} query IDs must be non-empty strings")
        if not isinstance(documents, Mapping):
            raise TypeError(f"{label} values must be document-score mappings")
        converted: dict[str, int | float] = {}
        for document_id, score in documents.items():
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"{label} document IDs must be non-empty strings")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"{label} scores must be numeric")
            if require_integer and not isinstance(score, int):
                raise TypeError(f"{label} relevance labels must be integers")
            numeric = float(score)
            if not math.isfinite(numeric):
                raise ValueError(f"{label} scores must be finite")
            converted[document_id] = score if require_integer else numeric
        result[query_id] = converted
    return result
