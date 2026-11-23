from __future__ import annotations

import importlib.util
import math
import unittest

from agentinputlab.benchmarks import BenchmarkRegistry
from agentinputlab.contracts.errors import ExternalDependencyError
from agentinputlab.metrics import (
    METRIC_CATALOG,
    best_reference_score,
    compression_ratio,
    english_qa_exact_match,
    english_qa_f1,
    evaluation_dependency_versions,
    intent_token_recall,
    metric_definition,
    normalize_english_answer,
    retrieval_aggregates,
    rouge_l,
    set_f1,
    summarize_pairs,
    token_jaccard,
)

_HAS_IR_MEASURES = importlib.util.find_spec("ir_measures") is not None
_HAS_ROUGE_SCORE = importlib.util.find_spec("rouge_score") is not None


class FirstPartyMetricTests(unittest.TestCase):
    def test_english_qa_normalization_matches_known_pipeline(self) -> None:
        self.assertEqual(normalize_english_answer("The, QUICK fox!"), "quick fox")
        self.assertEqual(english_qa_exact_match("An answer.", "answer"), 1.0)
        self.assertEqual(english_qa_exact_match("answer one", "answer two"), 0.0)

    def test_english_qa_f1_known_values_and_repeated_tokens(self) -> None:
        self.assertEqual(english_qa_f1("", ""), 1.0)
        self.assertEqual(english_qa_f1("", "answer"), 0.0)
        self.assertEqual(english_qa_f1("red blue", "green"), 0.0)
        self.assertAlmostEqual(english_qa_f1("red red blue", "red blue blue"), 2 / 3)
        self.assertAlmostEqual(english_qa_f1("the red book", "red book on desk"), 2 / 3)

    def test_best_reference_and_set_f1_known_values(self) -> None:
        self.assertEqual(best_reference_score("NYC", ("London", "NYC")), 1.0)
        self.assertAlmostEqual(set_f1(("a", "b"), ("b", "c", "d")), 0.4)
        self.assertEqual(set_f1((), ()), 1.0)
        self.assertEqual(set_f1(("a",), ()), 0.0)
        with self.assertRaises(TypeError):
            set_f1("ab", ("a", "b"))
        with self.assertRaises(ValueError):
            best_reference_score("answer", ())
        with self.assertRaises(TypeError):
            best_reference_score("answer", "answer")

    def test_text_metrics_have_stable_boundaries_and_symmetry(self) -> None:
        cases = ("", "red", "red blue", "RED blue!", "你好 世界")
        for left in cases:
            self.assertEqual(token_jaccard(left, left), 1.0)
            self.assertEqual(intent_token_recall(left, left), 1.0)
            for right in cases:
                score = token_jaccard(left, right)
                self.assertTrue(0.0 <= score <= 1.0)
                self.assertEqual(score, token_jaccard(right, left))
                recall = intent_token_recall(left, right)
                self.assertTrue(0.0 <= recall <= 1.0)
        self.assertEqual(compression_ratio("", "anything"), 1.0)
        self.assertEqual(compression_ratio("one two", "one"), 0.5)
        self.assertEqual(compression_ratio("one", "one two"), 2.0)

    def test_paired_summary_known_values_threshold_and_nonfinite_rejection(self) -> None:
        summary = summarize_pairs(
            (0.0, 1.0, 0.5, 1.0),
            (1.0, 0.5, 0.5, 0.8),
            harm_threshold=0.25,
        )
        self.assertEqual(summary.count, 4)
        self.assertAlmostEqual(summary.mean_effect, 0.075)
        self.assertEqual(summary.win_rate, 0.25)
        self.assertEqual(summary.tie_rate, 0.25)
        self.assertEqual(summary.harm_rate, 0.25)
        for invalid in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                summarize_pairs((0.0,), (invalid,))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            summarize_pairs((0.0,), (1.0,), harm_threshold=-0.1)

    def test_metric_catalog_covers_every_benchmark_metric(self) -> None:
        declared = {
            metric for cell in BenchmarkRegistry.default().list() for metric in cell.metrics
        }
        catalogued = {definition.name for definition in METRIC_CATALOG}
        self.assertEqual(len(catalogued), len(METRIC_CATALOG))
        self.assertEqual(declared, catalogued)
        self.assertEqual(metric_definition("ndcg_at_10").owner, "third_party")
        self.assertEqual(metric_definition("intent_preservation").status, "adapter_required")
        self.assertEqual(
            metric_definition("official_reward").status,
            "caller_asserted_external_bridge",
        )
        self.assertEqual(
            metric_definition("harm_rate").owner,
            "first_party_paired_estimator",
        )
        self.assertEqual(metric_definition("cost").status, "composition_required")
        self.assertEqual(
            metric_definition("valid_success").status,
            "study_composition_required",
        )
        with self.assertRaises(KeyError):
            metric_definition("made_up")


class OptionalMetricDependencyTests(unittest.TestCase):
    @unittest.skipIf(_HAS_IR_MEASURES, "dependency-missing behavior only")
    def test_retrieval_metric_has_actionable_missing_dependency_error(self) -> None:
        with self.assertRaisesRegex(ExternalDependencyError, r"agentinputlab\[evaluation\]"):
            retrieval_aggregates({"q": {"d": 1}}, {"q": {"d": 1.0}})

    @unittest.skipUnless(_HAS_IR_MEASURES, "install agentinputlab[evaluation]")
    def test_ir_measures_adapter_matches_hand_calculated_binary_example(self) -> None:
        qrels = {
            "q1": {"d1": 1, "d2": 1},
            "q2": {"d4": 1},
        }
        run = {
            "q1": {"d1": 3.0, "d3": 2.0, "d2": 1.0},
            "q2": {"d5": 2.0, "d4": 1.0},
        }
        first = retrieval_aggregates(
            qrels,
            run,
            measures={"ndcg_at_2": "nDCG@2", "recall_at_2": "R@2", "map": "AP"},
        )
        second = retrieval_aggregates(
            qrels,
            run,
            measures={"ndcg_at_2": "nDCG@2", "recall_at_2": "R@2", "map": "AP"},
        )
        expected_ndcg = ((1 / (1 + 1 / math.log2(3))) + (1 / math.log2(3))) / 2
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["ndcg_at_2"], expected_ndcg)
        self.assertAlmostEqual(first["recall_at_2"], 0.75)
        self.assertAlmostEqual(first["map"], 2 / 3)
        self.assertEqual(qrels["q1"], {"d1": 1, "d2": 1})

    @unittest.skipUnless(_HAS_IR_MEASURES, "install agentinputlab[evaluation]")
    def test_ir_measures_adapter_rejects_nonfinite_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            retrieval_aggregates({"q": {"d": 1}}, {"q": {"d": math.nan}})

    @unittest.skipIf(_HAS_ROUGE_SCORE, "dependency-missing behavior only")
    def test_rouge_has_actionable_missing_dependency_error(self) -> None:
        with self.assertRaisesRegex(ExternalDependencyError, r"agentinputlab\[evaluation\]"):
            rouge_l("answer", ("answer",))

    @unittest.skipUnless(_HAS_ROUGE_SCORE, "install agentinputlab[evaluation]")
    def test_rouge_score_adapter_known_boundaries_and_repeatability(self) -> None:
        identity = rouge_l("the red book", ("the red book",), use_stemmer=False)
        disjoint = rouge_l("red", ("blue",), use_stemmer=False)
        repeated = rouge_l("the red book", ("the red book",), use_stemmer=False)
        self.assertEqual(identity, repeated)
        self.assertEqual(identity.f1, 1.0)
        self.assertEqual(disjoint.f1, 0.0)
        with self.assertRaises(TypeError):
            rouge_l("answer", "answer")

    def test_dependency_versions_are_always_explicit(self) -> None:
        versions = evaluation_dependency_versions()
        self.assertEqual(
            set(versions),
            {"ir-measures", "pytrec-eval-terrier", "rouge-score"},
        )
        self.assertTrue(all(value for value in versions.values()))


if __name__ == "__main__":
    unittest.main()
