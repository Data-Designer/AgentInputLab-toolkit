"""Deterministic, evaluator-blind features for selector inference.

The extractor in this module deliberately implements a small, transparent
feature schema.  It reads only visible text plus conversation, context, and
tool-list shape from :class:`FrozenAgentState`.  In particular,
``task_id`` and ``metadata`` are never consulted: both may contain benchmark
or source identities that are useful for audit but unsafe as selector inputs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.contracts.state import FrozenAgentState

from .base import SelectionFeatures
from .runtime import FeatureAccessDomain, FeatureObservation

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]|[^\W_]+", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"[.!?\u3002\uff01\uff1f]+")
_AMBIGUITY_TERMS = frozenset(
    {
        "appropriate",
        "best",
        "better",
        "few",
        "it",
        "maybe",
        "nearby",
        "optimal",
        "reasonable",
        "recent",
        "several",
        "some",
        "something",
        "that",
        "them",
        "there",
        "they",
        "this",
        "those",
        "various",
        "合适",
        "最好",
        "最近",
        "一些",
        "这个",
        "那个",
        "它",
        "他们",
    }
)
_MULTI_STEP_TERMS = frozenset(
    {
        "after",
        "and",
        "before",
        "compare",
        "contrast",
        "finally",
        "first",
        "next",
        "then",
        "versus",
        "vs",
        "之后",
        "以及",
        "并且",
        "然后",
        "比较",
        "首先",
    }
)
_RETRIEVAL_TERMS = frozenset(
    {
        "according",
        "cite",
        "citation",
        "current",
        "evidence",
        "latest",
        "paper",
        "reference",
        "report",
        "research",
        "source",
        "statistics",
        "study",
        "today",
        "verify",
        "when",
        "where",
        "who",
        "year",
        "今天",
        "何时",
        "哪里",
        "引用",
        "数据",
        "最新",
        "来源",
        "研究",
        "论文",
        "证据",
    }
)
_QUESTION_TERMS = frozenset(
    {
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "什么",
        "为什么",
        "何时",
        "哪里",
        "如何",
        "谁",
    }
)


@dataclass(frozen=True, slots=True)
class VisibleFeatureConfig:
    """Frozen configuration for the five-family visible feature schema."""

    schema_id: str = "family_defect_features.v1"
    tokenization_revision: str = "unicode_word_cjk_v1"
    context_soft_limit_tokens: int = 8192

    def __post_init__(self) -> None:
        for label, value in (
            ("schema_id", self.schema_id),
            ("tokenization_revision", self.tokenization_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"visible feature {label} must be non-empty")
        if (
            isinstance(self.context_soft_limit_tokens, bool)
            or not isinstance(self.context_soft_limit_tokens, int)
            or self.context_soft_limit_tokens < 1
        ):
            raise ValidationError("context_soft_limit_tokens must be a positive integer")

    @property
    def config_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class VisibleStateFeatureExtractor:
    """Extract five normalized scores without consulting identities or metadata."""

    config: VisibleFeatureConfig = VisibleFeatureConfig()
    extractor_id: str = "visible_state_heuristic"
    extractor_revision: str = "v1"

    def __post_init__(self) -> None:
        if not isinstance(self.config, VisibleFeatureConfig):
            raise ValidationError("visible feature extractor requires VisibleFeatureConfig")
        for label, value in (
            ("extractor_id", self.extractor_id),
            ("extractor_revision", self.extractor_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"feature extractor {label} must be non-empty")

    @property
    def extractor_hash(self) -> str:
        return content_hash(
            {
                "extractor_id": self.extractor_id,
                "extractor_revision": self.extractor_revision,
                "config": self.config,
            }
        )

    @property
    def feature_schema_id(self) -> str:
        return self.config.schema_id

    @property
    def feature_schema_hash(self) -> str:
        return content_hash(
            {
                "schema_id": self.config.schema_id,
                "ordered_features": (
                    "ambiguity_score",
                    "multi_step_score",
                    "retrieval_need_score",
                    "context_pressure",
                    "reformulation_need_score",
                ),
                "tokenization_revision": self.config.tokenization_revision,
            }
        )

    def extract(self, state: FrozenAgentState) -> FeatureObservation:
        if not isinstance(state, FrozenAgentState):
            raise TypeError("visible feature extraction requires FrozenAgentState")

        query_tokens = _tokens(state.query)
        instruction_tokens = _tokens(state.instruction)
        message_tokens = tuple(
            token for message in state.messages for token in _tokens(message.content)
        )
        context_tokens = tuple(token for block in state.context for token in _tokens(block.text))
        visible_tokens = (*query_tokens, *instruction_tokens, *message_tokens, *context_tokens)
        request_text = "\n".join(part for part in (state.instruction, state.query) if part)
        request_lower = request_text.casefold()
        request_tokens = (*instruction_tokens, *query_tokens)
        request_term_set = frozenset(request_tokens)

        ambiguity_hits = _term_hits(request_lower, request_term_set, _AMBIGUITY_TERMS)
        dangling_reference = int(
            bool(request_term_set & {"it", "that", "them", "there", "they", "this", "those"})
            and not state.messages
            and not state.context
        )
        alternative_signal = int(
            " or " in f" {request_lower} "
            or "either" in request_term_set
            or "或者" in request_text
            or "还是" in request_text
        )
        ambiguity = _saturating(
            ambiguity_hits + dangling_reference + 0.5 * alternative_signal,
            scale=3.0,
        )

        multi_step_hits = _term_hits(
            request_lower,
            request_term_set,
            _MULTI_STEP_TERMS,
        )
        sentence_count = len(_SENTENCE_BOUNDARY.findall(request_text))
        list_signal = int(bool(re.search(r"(?:^|\s)(?:\d+[.)]|[-*])\s", request_text)))
        multi_step = _saturating(
            multi_step_hits + max(0, sentence_count - 1) + list_signal,
            scale=4.0,
        )

        retrieval_hits = _term_hits(
            request_lower,
            request_term_set,
            _RETRIEVAL_TERMS,
        )
        interrogative = int(
            bool(_term_hits(request_lower, request_term_set, _QUESTION_TERMS))
            or "?" in request_text
            or "\uff1f" in request_text
        )
        context_relief = min(0.6, len(context_tokens) / 256.0)
        retrieval = _clamp(
            _saturating(retrieval_hits + 0.5 * interrogative, scale=3.0) - context_relief
        )

        visible_token_count = len(visible_tokens)
        context_pressure = _clamp(
            visible_token_count / float(self.config.context_soft_limit_tokens)
        )

        nonspace = [character for character in request_text if not character.isspace()]
        noisy = sum(
            not character.isalnum()
            and character
            not in {
                "?",
                "!",
                ".",
                ",",
                ":",
                ";",
                "-",
                "_",
                "'",
                '"',
                "\u3001",
                "\u3002",
                "\uff01",
                "\uff08",
                "\uff09",
                "\uff0c",
                "\uff1a",
                "\uff1b",
                "\uff1f",
            }
            for character in nonspace
        )
        noise_ratio = noisy / max(1, len(nonspace))
        fragment = int(0 < len(request_tokens) <= 3 and not _SENTENCE_BOUNDARY.search(request_text))
        extreme_length = int(len(request_tokens) >= 64)
        keyword_style = int(
            len(request_tokens) >= 4
            and not _SENTENCE_BOUNDARY.search(request_text)
            and not _term_hits(request_lower, request_term_set, _QUESTION_TERMS)
        )
        reformulation = _clamp(
            0.45 * fragment
            + 0.35 * keyword_style
            + 0.35 * extreme_length
            + min(0.5, noise_ratio * 2.0)
        )

        features = SelectionFeatures(
            ambiguity_score=ambiguity,
            multi_step_score=multi_step,
            retrieval_need_score=retrieval,
            context_pressure=context_pressure,
            reformulation_need_score=reformulation,
        )
        provenance = {
            "schema_id": self.config.schema_id,
            "schema_hash": self.feature_schema_hash,
            "extractor_hash": self.extractor_hash,
            "tokenization_revision": self.config.tokenization_revision,
            "input_fields": (
                "query",
                "instruction",
                "context.text",
                "messages.content",
                "tool_names.count",
            ),
            "ignored_fields": ("task_id", "metadata"),
            "counts": {
                "query_tokens": len(query_tokens),
                "instruction_tokens": len(instruction_tokens),
                "message_tokens": len(message_tokens),
                "context_tokens": len(context_tokens),
                "messages": len(state.messages),
                "context_blocks": len(state.context),
                "tools": len(state.tool_names),
            },
            "signals": {
                "ambiguity": ambiguity_hits + dangling_reference + alternative_signal,
                "multi_step": multi_step_hits + max(0, sentence_count - 1) + list_signal,
                "retrieval": retrieval_hits + interrogative,
                "reformulation": fragment + keyword_style + extreme_length + int(noisy > 0),
            },
        }
        return FeatureObservation(
            features=features,
            state_hash=state.state_hash,
            extractor_id=self.extractor_id,
            extractor_revision=self.extractor_revision,
            access_domains=(FeatureAccessDomain.VISIBLE_PRE_ACTION,),
            provenance=provenance,
        )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(text))


def _term_hits(
    text: str,
    tokens: frozenset[str],
    terms: frozenset[str],
) -> int:
    return sum((term in tokens) if term.isascii() else (term in text) for term in terms)


def _saturating(value: float, *, scale: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("feature signal counts must be finite and non-negative")
    return _clamp(value / scale)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "VisibleFeatureConfig",
    "VisibleStateFeatureExtractor",
]
