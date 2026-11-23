"""Audited first-party QA metrics matching common English QA scorers."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_english_answer(text: str) -> str:
    """Apply the SQuAD/Hotpot-style English normalization pipeline."""

    if not isinstance(text, str):
        raise TypeError("answer text must be a string")
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def english_qa_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_english_answer(prediction) == normalize_english_answer(reference))


def english_qa_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_english_answer(prediction).split()
    reference_tokens = normalize_english_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    matched = sum(overlap.values())
    if matched == 0:
        return 0.0
    precision = matched / len(prediction_tokens)
    recall = matched / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def set_f1(prediction: Iterable[str], reference: Iterable[str]) -> float:
    """F1 over unique string identifiers, suitable for support/evidence IDs."""

    if isinstance(prediction, str) or isinstance(reference, str):
        raise TypeError("set_f1 inputs must be collections of string identifiers, not strings")
    predicted = set(prediction)
    expected = set(reference)
    if not all(isinstance(item, str) for item in predicted | expected):
        raise TypeError("set_f1 inputs must contain only strings")
    if not predicted or not expected:
        return float(predicted == expected)
    matched = len(predicted & expected)
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(expected)
    return 2 * precision * recall / (precision + recall)


def best_reference_score(
    prediction: str,
    references: Iterable[str],
    *,
    scorer: Callable[[str, str], float] = english_qa_f1,
) -> float:
    values = tuple(references)
    if isinstance(references, str):
        raise TypeError("references must be a collection of answers, not one string")
    if not values:
        raise ValueError("at least one reference answer is required")
    if not all(isinstance(value, str) for value in values):
        raise TypeError("reference answers must be strings")
    return max(scorer(prediction, reference) for reference in values)
