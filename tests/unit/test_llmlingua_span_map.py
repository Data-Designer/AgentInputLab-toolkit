"""Exact provenance and bounded-work regression, without any neural model."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from agentinputlab.algorithms.curate.llmlingua import build_span_map
from agentinputlab.contracts import ContextBlock

MODULE = "agentinputlab.algorithms.curate.llmlingua.span_map"


class ExactSpanTests(unittest.TestCase):
    def assert_exact(self, output, blocks, spans):
        sources = {block.block_id or f"context-{index}": block.text
                   for index, block in enumerate(blocks)}
        previous_end = 0
        for span in spans:
            self.assertTrue(span.exact)
            self.assertGreaterEqual(span.output_start, previous_end)
            self.assertGreaterEqual(span.source_start, 0)
            self.assertLessEqual(span.output_end, len(output))
            self.assertLessEqual(span.source_end, len(sources[span.source_block_id]))
            self.assertGreaterEqual(span.output_end - span.output_start, 3)
            self.assertEqual(output[span.output_start:span.output_end],
                             sources[span.source_block_id][span.source_start:span.source_end])
            previous_end = span.output_end

    def test_contained_output_never_needs_quadratic_diff(self):
        output = "alpha beta " * 4_000
        blocks = (ContextBlock(block_id="source", text="PREFIX " + output + " SUFFIX"),)
        with patch(MODULE + ".SequenceMatcher", side_effect=AssertionError("quadratic")):
            spans = build_span_map(output, blocks)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].source_start, 7)
        self.assert_exact(output, blocks, spans)

    def test_repetitive_long_deletions_keep_literal_provenance(self):
        blocks = (ContextBlock(block_id="source", text="alpha beta gamma omitted " * 1_000),)
        output = "alpha beta gamma " * 500
        with patch(MODULE + ".SequenceMatcher", side_effect=AssertionError("quadratic")):
            spans = build_span_map(output, blocks)
        self.assert_exact(output, blocks, spans)
        self.assertGreater(sum(s.output_end - s.output_start for s in spans), len(output) * .9)

    def test_long_reordered_blocks_keep_ids_and_no_overlap(self):
        blocks = (
            ContextBlock(block_id="a", text="ALPHA detail omitted " * 500),
            ContextBlock(block_id="b", text="BETA evidence deleted " * 500),
        )
        output = "BETA evidence " * 100 + "ALPHA detail " * 100
        with patch(MODULE + ".SequenceMatcher", side_effect=AssertionError("quadratic")):
            spans = build_span_map(output, blocks)
        self.assert_exact(output, blocks, spans)
        self.assertEqual({span.source_block_id for span in spans}, {"a", "b"})
        self.assertGreater(sum(s.output_end - s.output_start for s in spans), len(output) * .8)

    def test_cjk_and_unicode_offsets_are_not_bytes_or_normalized(self):
        blocks = (ContextBlock(block_id="文", text="臺灣研究測試。保留原始字元\uff01 café 🙂 " * 300),)
        output = "臺灣測試。原始字元\uff01 café 🙂 " * 150 + "UNSEEN🧪"
        with patch(MODULE + ".SequenceMatcher", side_effect=AssertionError("quadratic")):
            spans = build_span_map(output, blocks)
        self.assert_exact(output, blocks, spans)
        self.assertGreater(sum(s.output_end - s.output_start for s in spans), len(output) * .6)
        self.assertFalse(any(s.output_end > output.index("UNSEEN") for s in spans))

    def test_partial_words_use_exact_character_runs(self):
        blocks = (ContextBlock(block_id="a", text="prefixresearchsuffix " * 1_000),)
        output = "research " * 500
        with patch(MODULE + ".SequenceMatcher", side_effect=AssertionError("quadratic")):
            spans = build_span_map(output, blocks)
        self.assert_exact(output, blocks, spans)
        self.assertEqual(len(spans), 500)
        self.assertTrue(all(output[s.output_start:s.output_end] == "research" for s in spans))

    def test_reordered_spans_inside_one_block(self):
        blocks = (ContextBlock(block_id="a", text="first phrase omitted second evidence"),)
        output = "second evidence first phrase"
        with patch(MODULE + "._DIFF_PRODUCT_LIMIT", 0):
            spans = build_span_map(output, blocks)
        self.assert_exact(output, blocks, spans)
        self.assertEqual(len(spans), 2)
        self.assertEqual([output[s.output_start:s.output_end] for s in spans],
                         ["second evidence", "first phrase"])

    def test_small_pair_retains_original_difflib_choices(self):
        blocks = (ContextBlock(block_id="a", text="first phrase omitted second evidence"),)
        with patch(MODULE + "._indexed_matches", side_effect=AssertionError("large only")):
            spans = build_span_map("first phrase second evidence", blocks)
        self.assert_exact("first phrase second evidence", blocks, spans)

    def test_generated_text_empty_inputs_and_minimum_match(self):
        self.assertEqual(build_span_map("entirely new", ()), ())
        self.assertEqual(build_span_map("", (ContextBlock(text="source"),)), ())
        self.assertEqual(build_span_map("XYZ", (ContextBlock(text="abcdef"),)), ())
        self.assertEqual(build_span_map("ab", (ContextBlock(text="abc"),)), ())
        with self.assertRaisesRegex(ValueError, "positive"):
            build_span_map("text", (), minimum_match=0)

    def test_short_partial_word_respects_explicit_minimum(self):
        blocks = (ContextBlock(text="xyzABxyz"),)
        with patch(MODULE + "._DIFF_PRODUCT_LIMIT", 0):
            spans = build_span_map("AB XY", blocks, minimum_match=2)
        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0].output_start, spans[0].output_end), (0, 2))

    def test_indexed_mapping_invariants_for_generated_deletions_and_reorder(self):
        rng = random.Random(17)
        for _ in range(150):
            words = ["".join(rng.choices("abcdef世界é🙂", k=rng.randrange(2, 12)))
                     for _ in range(40)]
            blocks = (ContextBlock(block_id="a", text=" ".join(words)),)
            selected = words[::rng.randrange(2, 5)]
            rng.shuffle(selected)
            output = " ".join(selected) + " NEW🧪"
            with patch(MODULE + "._DIFF_PRODUCT_LIMIT", 0):
                spans = build_span_map(output, blocks)
                self.assertEqual(spans, build_span_map(output, blocks))
            self.assert_exact(output, blocks, spans)


if __name__ == "__main__":
    unittest.main()
