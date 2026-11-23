"""Reversible access chunks for an original visible text block."""

import re
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise

from agentinputlab.contracts import ValidationError


@dataclass(frozen=True)
class VisibleChunk:
    start: int
    end: int
    text: str
    source_sha256: str


def split_visible_text(text: str, *, words_per_chunk: int = 350, boundary: str = "word") -> tuple[VisibleChunk, ...]:
    """Partition original Unicode text without dropping whitespace or adding content."""
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("a nonempty original visible document is required")
    if type(words_per_chunk) is not int or words_per_chunk < 1:
        raise ValidationError("words_per_chunk must be a positive integer")
    if boundary not in ("word", "line"):
        raise ValidationError("visible chunk boundary must be word or line")
    boundaries = [0]
    if boundary == "word":
        boundaries.extend(
            match.start()
            for index, match in enumerate(re.finditer(r"\S+", text))
            if index and index % words_per_chunk == 0
        )
    else:
        # A meeting turn exceeding the target remains one intact source line.
        used = offset = 0
        for line in text.splitlines(keepends=True):
            count = len(line.split())
            if used and count and used + count > words_per_chunk:
                boundaries.append(offset)
                used = 0
            used += count
            offset += len(line)
    boundaries.append(len(text))
    digest = sha256(text.encode("utf-8")).hexdigest()
    return tuple(VisibleChunk(a, b, text[a:b], digest) for a, b in pairwise(boundaries))
