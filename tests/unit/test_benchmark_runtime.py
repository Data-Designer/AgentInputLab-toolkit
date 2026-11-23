from __future__ import annotations

import math
import unittest
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.runtime import (
    BenchmarkBatchRecord,
    BenchmarkBatchRunner,
    BenchmarkExampleRecord,
    PairedBenchmarkRunner,
)
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.state import ExecutionBudget, FrozenAgentState


class AnswerAdapter:
    benchmark_id = "fixture.answers"

    def __init__(self, examples: tuple[BenchmarkExample, ...]) -> None:
        self._examples = examples
        self.requested_splits: list[str] = []
        self.scored_examples: list[BenchmarkExample] = []

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        self.requested_splits.append(split)
        return self._examples

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        self.scored_examples.append(example)
        answer = example.evaluator_gold["answer"]
        return {
            "exact_match": float(prediction == answer),
            "prediction_length": float(len(str(prediction))),
        }


class MetricAdapter:
    benchmark_id = "fixture.metrics"

    def __init__(
        self,
        metrics: tuple[object, ...],
        *,
        examples: tuple[BenchmarkExample, ...] | None = None,
    ) -> None:
        self._metrics = metrics
        self._examples = examples or tuple(
            BenchmarkExample(example_id=f"e{index}", query=f"q{index}")
            for index in range(len(metrics))
        )
        self._score_index = 0

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        del split
        return self._examples

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        del example, prediction
        value = self._metrics[self._score_index]
        self._score_index += 1
        return value  # type: ignore[return-value]


class ValueAdapter:
    benchmark_id = "fixture.values"
    pairing_mode = "stateless_score_twice"

    def __init__(self) -> None:
        self._examples = (
            BenchmarkExample(example_id="a", query="1", evaluator_gold={"target": 1}),
            BenchmarkExample(example_id="b", query="2", evaluator_gold={"target": 2}),
        )

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        self.split = split
        return self._examples

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        target = float(example.evaluator_gold["target"])
        predicted = float(prediction)
        return {
            "value": predicted,
            "negative_error": -abs(predicted - target),
        }


class BenchmarkBatchRunnerTests(unittest.TestCase):
    def test_batch_isolates_gold_and_aggregates_metrics(self) -> None:
        examples = (
            BenchmarkExample(
                example_id="one",
                query="alpha",
                visible_metadata={"domain": "public"},
                evaluator_gold={"answer": "ALPHA", "secret": "gold-only-one"},
            ),
            BenchmarkExample(
                example_id="two",
                query="beta",
                evaluator_gold={"answer": "wrong", "secret": "gold-only-two"},
            ),
        )
        adapter = AnswerAdapter(examples)
        observed_states: list[FrozenAgentState] = []

        def predict(state: FrozenAgentState) -> str:
            observed_states.append(state)
            serialized = str(state.canonical_dict())
            self.assertNotIn("gold-only", serialized)
            self.assertNotIn("answer", serialized)
            return state.query.upper()

        result = BenchmarkBatchRunner().run(adapter, "test", predict)

        self.assertEqual(adapter.requested_splits, ["test"])
        self.assertEqual(result.benchmark_id, "fixture.answers")
        self.assertEqual(result.split, "test")
        self.assertEqual(result.example_count, 2)
        self.assertEqual([state.task_id for state in observed_states], ["one", "two"])
        self.assertEqual([record.prediction for record in result.examples], ["ALPHA", "BETA"])
        self.assertEqual(result.aggregate_metrics["exact_match"], 0.5)
        self.assertEqual(result.aggregate_metrics["prediction_length"], 4.5)
        self.assertIs(adapter.scored_examples[0], examples[0])
        self.assertEqual(adapter.scored_examples[0].evaluator_gold["answer"], "ALPHA")

    def test_budget_is_visible_without_gold(self) -> None:
        example = BenchmarkExample(
            example_id="one",
            query="q",
            evaluator_gold={"answer": "q"},
        )
        budget = ExecutionBudget(max_model_calls=2, max_output_tokens=17)
        seen: list[ExecutionBudget] = []

        def predict(state: FrozenAgentState) -> str:
            seen.append(state.budget)
            return state.query

        BenchmarkBatchRunner().run(AnswerAdapter((example,)), "dev", predict, budget=budget)
        self.assertEqual(seen, [budget])

    def test_visible_tool_names_reach_the_frozen_state(self) -> None:
        example = BenchmarkExample(
            example_id="one",
            query="q",
            visible_tool_names=("KB_search", "transfer_to_human_agents"),
            evaluator_gold={"answer": "q"},
        )
        seen: list[tuple[str, ...]] = []

        def predict(state: FrozenAgentState) -> str:
            seen.append(state.tool_names)
            return state.query

        BenchmarkBatchRunner().run(AnswerAdapter((example,)), "dev", predict)
        self.assertEqual(seen, [("KB_search", "transfer_to_human_agents")])

    def test_prediction_record_is_recursively_immutable(self) -> None:
        adapter = MetricAdapter(({"score": 1.0},))
        prediction = {"tokens": ["a", "b"]}
        result = BenchmarkBatchRunner().run(adapter, "test", lambda state: prediction)
        prediction["tokens"].append("changed")
        self.assertEqual(result.examples[0].prediction["tokens"], ("a", "b"))
        with self.assertRaises(TypeError):
            result.examples[0].prediction["new"] = "value"
        with self.assertRaises(TypeError):
            result.aggregate_metrics["score"] = 0.0

    def test_adapter_evaluation_provenance_is_captured_and_recursively_immutable(
        self,
    ) -> None:
        adapter = MetricAdapter(({"score": 1.0},))
        provenance = {
            "official_runtime_verified": False,
            "scorer": {"identity": "fixture", "revisions": ["v1"]},
        }
        adapter.evaluation_provenance = provenance

        result = BenchmarkBatchRunner().run(adapter, "test", lambda state: state.query)
        record = result.examples[0]
        provenance["scorer"]["revisions"].append("changed")

        self.assertEqual(record.evaluation_provenance["scorer"]["revisions"], ("v1",))
        with self.assertRaises(TypeError):
            record.evaluation_provenance["scorer"]["identity"] = "changed"

    def test_invalid_adapter_evaluation_provenance_fails_closed(self) -> None:
        cases = (
            (["not", "a", "mapping"], "adapter.evaluation_provenance must be a mapping"),
            ({"unsupported": {"set"}}, "must be canonically serializable"),
        )
        for provenance, message in cases:
            adapter = MetricAdapter(({"score": 1.0},))
            adapter.evaluation_provenance = provenance
            with (
                self.subTest(provenance=provenance),
                self.assertRaisesRegex(ValidationError, message),
            ):
                BenchmarkBatchRunner().run(adapter, "test", lambda state: state.query)
            self.assertEqual(adapter._score_index, 0)

    def test_nonserializable_prediction_fails_closed(self) -> None:
        for prediction in (object(), math.nan):
            with (
                self.subTest(prediction=prediction),
                self.assertRaisesRegex(ValidationError, "canonically serializable"),
            ):
                BenchmarkBatchRunner().run(
                    MetricAdapter(({"score": 1.0},)),
                    "test",
                    lambda state, value=prediction: value,
                )

    def test_invalid_metric_outputs_fail_closed(self) -> None:
        invalid_metrics = (
            ({}, "must not be empty"),
            ({"": 1.0}, "metric key"),
            ({" score": 1.0}, "metric key"),
            ({"bad\nkey": 1.0}, "metric key"),
            ({"score": True}, "must be numeric"),
            ({"score": "1"}, "must be numeric"),
            ({"score": math.nan}, "must be finite"),
            ({"score": math.inf}, "must be finite"),
            ([1.0], "must be a mapping"),
        )
        for metrics, message in invalid_metrics:
            with (
                self.subTest(metrics=metrics),
                self.assertRaisesRegex(ValidationError, message),
            ):
                BenchmarkBatchRunner().run(
                    MetricAdapter((metrics,)),
                    "test",
                    lambda state: "prediction",
                )

    def test_metric_schema_must_be_consistent_across_examples(self) -> None:
        adapter = MetricAdapter(({"accuracy": 1.0}, {"f1": 1.0}))
        with self.assertRaisesRegex(ValidationError, "differ from the batch schema"):
            BenchmarkBatchRunner().run(adapter, "test", lambda state: state.query)

    def test_empty_split_and_duplicate_ids_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one example"):
            BenchmarkBatchRunner().run(MetricAdapter((), examples=()), "test", lambda state: "x")

        duplicated = (
            BenchmarkExample(example_id="same", query="q1"),
            BenchmarkExample(example_id="same", query="q2"),
        )
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            BenchmarkBatchRunner().run(
                MetricAdapter(({"score": 1.0}, {"score": 1.0}), examples=duplicated),
                "test",
                lambda state: "x",
            )

    def test_invalid_runtime_configuration_fails_before_prediction(self) -> None:
        adapter = MetricAdapter(({"score": 1.0},))
        adapter.benchmark_id = ""
        with self.assertRaisesRegex(ValidationError, "adapter.benchmark_id"):
            BenchmarkBatchRunner().run(adapter, "test", lambda state: "x")
        adapter.benchmark_id = "fixture.metrics"
        with self.assertRaisesRegex(ValidationError, "split"):
            BenchmarkBatchRunner().run(adapter, " ", lambda state: "x")
        with self.assertRaisesRegex(TypeError, "must be callable"):
            BenchmarkBatchRunner().run(adapter, "test", None)  # type: ignore[arg-type]

    def test_record_rejects_forged_aggregate(self) -> None:
        state_hash = FrozenAgentState(query="q").state_hash
        record = BenchmarkExampleRecord(
            example_id="one",
            state_hash=state_hash,
            prediction="x",
            metrics={"score": 1.0},
        )
        with self.assertRaisesRegex(ValidationError, "batch aggregate metric"):
            BenchmarkBatchRecord(
                benchmark_id="fixture",
                split="test",
                examples=(record,),
                aggregate_metrics={"score": 0.0},
            )


class PairedBenchmarkRunnerTests(unittest.TestCase):
    def test_paired_arms_share_exact_state_and_aggregate_deltas(self) -> None:
        adapter = ValueAdapter()
        baseline_states: list[FrozenAgentState] = []
        treatment_states: list[FrozenAgentState] = []

        def baseline(state: FrozenAgentState) -> float:
            baseline_states.append(state)
            return 0.0

        def treatment(state: FrozenAgentState) -> float:
            treatment_states.append(state)
            return float(state.query)

        result = PairedBenchmarkRunner().run(adapter, "test", baseline, treatment)

        self.assertEqual(result.example_count, 2)
        self.assertEqual(result.baseline_metrics, {"value": 0.0, "negative_error": -1.5})
        self.assertEqual(result.treatment_metrics, {"value": 1.5, "negative_error": 0.0})
        self.assertEqual(result.metric_deltas, {"value": 1.5, "negative_error": 1.5})
        for index, record in enumerate(result.examples):
            self.assertIs(baseline_states[index], treatment_states[index])
            self.assertEqual(record.state_hash, baseline_states[index].state_hash)
            self.assertEqual(record.baseline.state_hash, record.treatment.state_hash)
            self.assertEqual(
                record.metric_deltas["value"],
                record.treatment.metrics["value"] - record.baseline.metrics["value"],
            )

        with self.assertRaisesRegex(ValidationError, "baseline aggregate metric"):
            replace(
                result,
                baseline_metrics={"value": 99.0, "negative_error": -1.5},
                metric_deltas={"value": -97.5, "negative_error": 1.5},
            )

    def test_paired_metric_schema_mismatch_fails_closed(self) -> None:
        class ArmDependentAdapter:
            benchmark_id = "fixture.arm-dependent"
            pairing_mode = "stateless_score_twice"

            def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
                del split
                return (BenchmarkExample(example_id="one", query="q"),)

            def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
                del example
                return {str(prediction): 1.0}

        with self.assertRaisesRegex(ValidationError, "differ from the batch schema"):
            PairedBenchmarkRunner().run(
                ArmDependentAdapter(),
                "test",
                lambda state: "baseline_metric",
                lambda state: "treatment_metric",
            )

    def test_invalid_treatment_runner_fails_before_examples_execute(self) -> None:
        adapter = MetricAdapter(({"score": 1.0},))
        baseline_calls: list[str] = []

        def baseline(state: FrozenAgentState) -> str:
            baseline_calls.append(state.task_id)
            return "x"

        with self.assertRaisesRegex(TypeError, "treatment prediction runner"):
            PairedBenchmarkRunner().run(
                adapter,
                "test",
                baseline,
                None,  # type: ignore[arg-type]
            )
        self.assertEqual(baseline_calls, [])

    def test_stateful_external_adapter_cannot_fake_pairing_with_visible_hash_only(
        self,
    ) -> None:
        adapter = MetricAdapter(({"score": 1.0},))
        adapter.pairing_mode = "unsupported_external_replay"

        with self.assertRaisesRegex(
            ValidationError,
            "atomic restore/evaluate-pair bridge",
        ):
            PairedBenchmarkRunner().run(
                adapter,
                "test",
                lambda state: "baseline",
                lambda state: "treatment",
            )

    def test_missing_pairing_mode_fails_closed(self) -> None:
        adapter = MetricAdapter(({"score": 1.0},))

        with self.assertRaisesRegex(
            ValidationError,
            "requires an explicit pairing_mode",
        ):
            PairedBenchmarkRunner().run(
                adapter,
                "test",
                lambda state: "baseline",
                lambda state: "treatment",
            )


if __name__ == "__main__":
    unittest.main()
