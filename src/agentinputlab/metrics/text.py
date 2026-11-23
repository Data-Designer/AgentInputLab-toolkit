"""Dependency-free patch quality and intent-overlap metrics."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def token_set(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text)}


def intent_token_recall(original: str, transformed: str) -> float:
    source = token_set(original)
    if not source:
        return 1.0
    return len(source & token_set(transformed)) / len(source)


def token_jaccard(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def compression_ratio(original: str, transformed: str) -> float:
    original_tokens = _TOKEN.findall(original)
    transformed_tokens = _TOKEN.findall(transformed)
    if not original_tokens:
        return 1.0
    return len(transformed_tokens) / len(original_tokens)
