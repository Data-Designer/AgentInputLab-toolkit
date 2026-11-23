"""Conservative exact-span provenance for compressed context."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from agentinputlab.contracts import ContextBlock

# Character-level difflib is useful for short inputs but quadratic on the long,
# repetitive contexts these compressors are meant to handle.  Longer pairs use
# literal indexed anchors instead; this changes provenance choices, not text.
_DIFF_PRODUCT_LIMIT = 2_000_000
_TOKENS = re.compile(r"[A-Za-z0-9_]+|[^\s]")


def _indexed_matches(
    source: str, output: str, output_tokens: Sequence[tuple[int, int]], minimum_match: int,
) -> list[tuple[int, int, int]]:
    """Find exact runs without a source-by-output character comparison matrix.

    One-, two- and three-token anchors include the original intervening spaces.
    Sorted occurrence lists keep even repeated anchors cheap.  Reordered source
    text is allowed; partial words fall back to indexed character runs.  Neither
    path normalizes text or attributes a span to a merely similar substring.
    """

    source_tokens = [match.span() for match in _TOKENS.finditer(source)]
    anchors: list[dict[str, list[int]]] = [{}, {}, {}]
    for width, index in enumerate(anchors, start=1):
        for offset in range(len(source_tokens) - width + 1):
            start = source_tokens[offset][0]
            end = source_tokens[offset + width - 1][1]
            index.setdefault(source[start:end], []).append(start)
    character_anchors: dict[str, list[int]] | None = None
    character_width = min(3, minimum_match)
    matches: list[tuple[int, int, int]] = []

    def append(source_start: int, output_start: int, size: int) -> None:
        if matches:
            last_source, last_output, last_size = matches[-1]
            source_end, output_end = last_source + last_size, last_output + last_size
            if (source_start >= source_end and output_start >= output_end
                    and source_start - source_end == output_start - output_end
                    and source[source_end:source_start] == output[output_end:output_start]):
                matches[-1] = (last_source, last_output, output_start + size - last_output)
                return
        matches.append((source_start, output_start, size))

    source_cursor = 0
    offset = 0
    while offset < len(output_tokens):
        for width in range(min(3, len(output_tokens) - offset), 0, -1):
            start, end = output_tokens[offset][0], output_tokens[offset + width - 1][1]
            positions = anchors[width - 1].get(output[start:end])
            if positions:
                position_index = bisect_left(positions, source_cursor)
                source_start = positions[position_index if position_index < len(positions) else 0]
                append(source_start, start, end - start)
                source_cursor = source_start + end - start
                offset += width
                break
        else:
            # Token compression can keep only part of a word.  Short literal
            # anchors locate those runs without rescanning the source per word.
            if character_anchors is None:
                character_anchors = {}
                for position in range(len(source) - character_width + 1):
                    key = source[position:position + character_width]
                    character_anchors.setdefault(key, []).append(position)
            start, end = output_tokens[offset]
            output_start = start
            while output_start + character_width <= end:
                positions = character_anchors.get(output[output_start:output_start + character_width])
                if not positions:
                    output_start += 1
                    continue
                position_index = bisect_left(positions, source_cursor)
                source_start = positions[position_index if position_index < len(positions) else 0]
                size = character_width
                while (output_start + size < end and source_start + size < len(source)
                       and output[output_start + size] == source[source_start + size]):
                    size += 1
                append(source_start, output_start, size)
                source_cursor = source_start + size
                output_start += size
            offset += 1
    return matches


@dataclass(frozen=True, slots=True)
class SpanMapping:
    output_start: int
    output_end: int
    source_block_id: str
    source_start: int
    source_end: int
    exact: bool = True


def build_span_map(
    compressed_text: str,
    source_blocks: Sequence[ContextBlock],
    *,
    minimum_match: int = 3,
) -> tuple[SpanMapping, ...]:
    """Map only exact matching spans and leave generated/unmatched text unmapped.

    Per-block matching supports context reorder.  Overlapping output matches are
    resolved longest-first; the function never invents provenance for a token
    that cannot be located exactly in a source block.  Large pairs use indexed
    literal anchors instead of quadratic character diffing; ambiguous locations
    can therefore differ from the short-input matcher.  This is provenance, not
    a reconstruction of the compressor's internal token-selection decisions.
    """

    if minimum_match < 1:
        raise ValueError("minimum_match must be positive")
    if not compressed_text:
        return ()
    output_tokens: list[tuple[int, int]] | None = None
    candidates: list[SpanMapping] = []
    for index, block in enumerate(source_blocks):
        block_id = block.block_id or f"context-{index}"
        contained = block.text.find(compressed_text)
        if contained >= 0:
            matches = [(contained, 0, len(compressed_text))]
        elif len(block.text) * len(compressed_text) <= _DIFF_PRODUCT_LIMIT:
            matcher = SequenceMatcher(None, block.text, compressed_text, autojunk=False)
            matches = matcher.get_matching_blocks()
        else:
            if output_tokens is None:
                output_tokens = [match.span() for match in _TOKENS.finditer(compressed_text)]
            matches = _indexed_matches(block.text, compressed_text, output_tokens, minimum_match)
        for source_start, output_start, size in matches:
            if size < minimum_match:
                continue
            candidates.append(
                SpanMapping(
                    output_start=output_start,
                    output_end=output_start + size,
                    source_block_id=block_id,
                    source_start=source_start,
                    source_end=source_start + size,
                )
            )
    candidates.sort(
        key=lambda item: (
            -(item.output_end - item.output_start),
            item.output_start,
            item.source_block_id,
            item.source_start,
        )
    )
    occupied = bytearray(len(compressed_text))
    selected: list[SpanMapping] = []
    for candidate in candidates:
        if any(occupied[candidate.output_start : candidate.output_end]):
            continue
        occupied[candidate.output_start : candidate.output_end] = b"\x01" * (
            candidate.output_end - candidate.output_start
        )
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: (item.output_start, item.output_end)))
