from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agentinputlab.benchmarks import PreparedQABenchmarkAdapter, QAPrediction
from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.preprocess import PreparedBenchmark
from agentinputlab.benchmarks.runtime import BenchmarkBatchRunner
from agentinputlab.contracts.errors import ArtifactError, ValidationError


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _input(
    example_id: str = "q1",
    *,
    benchmark_cell: str = "qa.multihop",
    source_split: str = "validation",
) -> dict[str, object]:
    return {
        "id": example_id,
        "query": "Who wrote the novel?",
        "instruction": "Answer using the supplied passages.",
        "visible_context": [
            {
                "block_id": "doc-1#sent-0",
                "text": "Ada wrote the novel.",
                "source": "doc-1",
                "score": 0.75,
                "metadata": {"paragraph": 0},
            },
            {
                "block_id": "doc-2#sent-1",
                "text": "The novel won a prize.",
            },
        ],
        "visible_metadata": {
            "benchmark_cell": benchmark_cell,
            "source_split": source_split,
            "source_id": example_id,
        },
    }


def _prepared(
    root: Path,
    *,
    inputs: list[dict[str, object]] | None = None,
    gold: list[dict[str, object]] | None = None,
    asset_id: str = "qa.multihop.official",
    processor: str = "qa_json_v1",
    cell_id: str = "",
    split: str = "",
) -> PreparedBenchmark:
    input_rows = inputs or [_input()]
    gold_rows = gold or [
        {
            "id": "q1",
            "answers": ["Ada", "Ada Lovelace"],
            "support_ids": ["doc-1#sent-0"],
        }
    ]
    inputs_hash = _write_jsonl(root / "inputs.jsonl", input_rows)
    gold_hash = _write_jsonl(root / "gold.jsonl", gold_rows)
    return PreparedBenchmark(
        asset_id=asset_id,
        processor=processor,
        processor_revision="1",
        source_sha256="a" * 64,
        root=root,
        counts={"inputs": len(input_rows), "gold": len(gold_rows)},
        file_sha256={"inputs.jsonl": inputs_hash, "gold.jsonl": gold_hash},
        cell_id=cell_id,
        split=split,
    )


class PreparedQABenchmarkAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)

    def test_flat_references_run_with_gold_physically_isolated(self) -> None:
        adapter = PreparedQABenchmarkAdapter(
            _prepared(self.root),
            benchmark_id="qa.multihop",
            split="validation",
        )
        seen = []

        def predict(state):
            payload = state.canonical_dict()
            self.assertNotIn("Ada Lovelace", json.dumps(payload))
            self.assertNotIn("answers", json.dumps(payload))
            self.assertEqual(state.context[0].block_id, "doc-1#sent-0")
            self.assertEqual(state.context[0].metadata["paragraph"], 0)
            seen.append(state.task_id)
            return QAPrediction(answer="The Ada.", support_ids=("doc-1#sent-0",))

        result = BenchmarkBatchRunner().run(adapter, "validation", predict)

        self.assertEqual(seen, ["q1"])
        self.assertEqual(
            result.aggregate_metrics,
            {"exact_match": 1.0, "answer_f1": 1.0, "support_f1": 1.0},
        )

    def test_annotation_groups_remain_aligned_and_score_by_group(self) -> None:
        prepared = _prepared(
            self.root,
            gold=[
                {
                    "id": "q1",
                    "evaluator_metadata": {
                        "annotation_ids": ["ann-1", "ann-2"],
                        "private_note": "gold-only-secret",
                    },
                    "reference_groups": [
                        {"answers": ["Ada"], "evidence_texts": ["Paragraph one."]},
                        {
                            "answers": ["Lovelace"],
                            "evidence_texts": ["Paragraph two.", "Paragraph three."],
                        },
                    ],
                }
            ],
        )
        adapter = PreparedQABenchmarkAdapter(
            prepared,
            benchmark_id="qa.multihop",
            split="validation",
        )
        example = adapter.examples("validation")[0]

        groups = example.evaluator_gold["reference_groups"]
        evaluator_metadata = example.evaluator_gold["evaluator_metadata"]
        self.assertEqual(evaluator_metadata["annotation_ids"], ("ann-1", "ann-2"))
        self.assertEqual(groups[0]["answers"], ("Ada",))
        self.assertEqual(groups[0]["evidence_texts"], ("Paragraph one.",))
        self.assertEqual(groups[1]["answers"], ("Lovelace",))
        self.assertEqual(
            groups[1]["evidence_texts"],
            ("Paragraph two.", "Paragraph three."),
        )
        with self.assertRaises(TypeError):
            evaluator_metadata["late"] = True
        self.assertNotIn(
            "gold-only-secret",
            json.dumps(example.visible_state().canonical_dict()),
        )
        metrics = adapter.score(
            example,
            {"answer": "Lovelace", "evidence_texts": ["Paragraph two."]},
        )
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["answer_f1"], 1.0)
        self.assertAlmostEqual(metrics["evidence_f1"], 2 / 3)
        self.assertNotIn("support_f1", metrics)

    def test_split_cell_and_example_identity_fail_closed(self) -> None:
        prepared = _prepared(self.root)
        adapter = PreparedQABenchmarkAdapter(
            prepared,
            benchmark_id="qa.multihop",
            split="validation",
        )
        with self.assertRaisesRegex(ValidationError, "exposes split"):
            adapter.examples("test")
        with self.assertRaisesRegex(ValidationError, "does not belong"):
            adapter.score(
                BenchmarkExample(example_id="q1", query="impostor"),
                QAPrediction("Ada"),
            )

        with self.assertRaisesRegex(ArtifactError, "cell_id"):
            PreparedQABenchmarkAdapter(
                _prepared(self.root, cell_id="qa.other"),
                benchmark_id="qa.multihop",
                split="validation",
            )
        with self.assertRaisesRegex(ArtifactError, "prepared QA split"):
            PreparedQABenchmarkAdapter(
                _prepared(self.root, split="test"),
                benchmark_id="qa.multihop",
                split="validation",
            )

        _write_jsonl(self.root / "inputs.jsonl", [_input(benchmark_cell="qa.other")])
        with self.assertRaisesRegex(ArtifactError, "benchmark_cell"):
            PreparedQABenchmarkAdapter(
                prepared,
                benchmark_id="qa.multihop",
                split="validation",
            )
        _write_jsonl(self.root / "inputs.jsonl", [_input(source_split="test")])
        with self.assertRaisesRegex(ArtifactError, "source_split"):
            PreparedQABenchmarkAdapter(
                prepared,
                benchmark_id="qa.multihop",
                split="validation",
            )

    def test_named_profile_locks_prepared_asset_processor_cell_and_split(self) -> None:
        prepared = _prepared(
            self.root,
            inputs=[_input(benchmark_cell="musique.answerable", source_split="dev")],
            gold=[
                {
                    "id": "q1",
                    "answers": ["Ada"],
                    "support_ids": ["paragraph:0"],
                }
            ],
            asset_id="musique.answerable.dev",
            processor="musique_zip_v1",
            cell_id="musique.answerable",
            split="dev",
        )
        PreparedQABenchmarkAdapter(
            prepared,
            benchmark_id="musique.answerable",
            split="dev",
        )

        invalid_fields = {
            "asset_id": "custom.asset",
            "processor": "qa_json_v1",
            "cell_id": "",
            "split": "",
        }
        for field, value in invalid_fields.items():
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ArtifactError, rf"prepared QA {field}"),
            ):
                PreparedQABenchmarkAdapter(
                    replace(prepared, **{field: value}),
                    benchmark_id="musique.answerable",
                    split="dev",
                )

        with self.assertRaisesRegex(ArtifactError, "requires adapter split 'dev'"):
            PreparedQABenchmarkAdapter(
                prepared,
                benchmark_id="musique.answerable",
                split="validation",
            )

    def test_input_and_context_schema_are_strict(self) -> None:
        row = _input()
        row["unexpected"] = True
        with self.assertRaisesRegex(ArtifactError, "schema mismatch"):
            PreparedQABenchmarkAdapter(
                _prepared(self.root, inputs=[row]),
                benchmark_id="qa.multihop",
                split="validation",
            )

        row = _input()
        context = row["visible_context"]
        assert isinstance(context, list)
        assert isinstance(context[0], dict)
        context[0]["title"] = "not a ContextBlock field"
        with self.assertRaisesRegex(ArtifactError, "unknown fields"):
            PreparedQABenchmarkAdapter(
                _prepared(self.root, inputs=[row]),
                benchmark_id="qa.multihop",
                split="validation",
            )

        row = _input()
        context = row["visible_context"]
        assert isinstance(context, list)
        assert isinstance(context[1], dict)
        context[1]["block_id"] = "doc-1#sent-0"
        with self.assertRaisesRegex(ArtifactError, "duplicate block ID"):
            PreparedQABenchmarkAdapter(
                _prepared(self.root, inputs=[row]),
                benchmark_id="qa.multihop",
                split="validation",
            )

    def test_gold_schema_and_metric_profile_are_strict(self) -> None:
        with self.assertRaisesRegex(ArtifactError, "input and gold IDs"):
            PreparedQABenchmarkAdapter(
                _prepared(self.root, gold=[{"id": "other", "answers": ["Ada"]}]),
                benchmark_id="qa.multihop",
                split="validation",
            )

        with self.assertRaisesRegex(ArtifactError, "one support/evidence schema"):
            PreparedQABenchmarkAdapter(
                _prepared(
                    self.root,
                    gold=[
                        {
                            "id": "q1",
                            "reference_groups": [
                                {"answers": ["Ada"], "evidence_texts": ["one"]},
                                {"answers": ["Ada"]},
                            ],
                        }
                    ],
                ),
                benchmark_id="qa.multihop",
                split="validation",
            )

        with self.assertRaisesRegex(ArtifactError, "duplicates"):
            PreparedQABenchmarkAdapter(
                _prepared(self.root, gold=[{"id": "q1", "answers": ["Ada", "Ada"]}]),
                benchmark_id="qa.multihop",
                split="validation",
            )
        with self.assertRaisesRegex(ArtifactError, "cannot mix"):
            PreparedQABenchmarkAdapter(
                _prepared(
                    self.root,
                    gold=[
                        {
                            "id": "q1",
                            "answers": ["Ada"],
                            "evidence_ids": ["p1"],
                            "evidence_texts": ["Paragraph one."],
                        }
                    ],
                ),
                benchmark_id="qa.multihop",
                split="validation",
            )

    def test_id_based_evidence_is_preserved_for_non_qasper_datasets(self) -> None:
        adapter = PreparedQABenchmarkAdapter(
            _prepared(
                self.root,
                gold=[{"id": "q1", "answers": ["Ada"], "evidence_ids": ["p1", "p2"]}],
            ),
            benchmark_id="qa.multihop",
            split="validation",
        )
        example = adapter.examples("validation")[0]

        metrics = adapter.score(example, QAPrediction("Ada", evidence_ids=("p1",)))

        self.assertAlmostEqual(metrics["evidence_f1"], 2 / 3)

    def test_prediction_schema_is_immutable_and_strict(self) -> None:
        source = ["s1"]
        prediction = QAPrediction("Ada", support_ids=source)
        source.append("late")
        self.assertEqual(prediction.support_ids, ("s1",))
        with self.assertRaisesRegex(ValidationError, "duplicates"):
            QAPrediction("Ada", support_ids=("s1", "s1"))
        with self.assertRaisesRegex(ValidationError, "invalid schema"):
            adapter = PreparedQABenchmarkAdapter(
                _prepared(self.root),
                benchmark_id="qa.multihop",
                split="validation",
            )
            adapter.score(adapter.examples("validation")[0], {"answer": "Ada", "extra": 1})

    def test_musique_profile_uses_official_answerable_metric_keys(self) -> None:
        prepared = _prepared(
            self.root,
            inputs=[_input(benchmark_cell="musique.answerable", source_split="dev")],
            gold=[
                {
                    "id": "q1",
                    "answers": ["Ada", "Ada Lovelace"],
                    "support_ids": ["paragraph:0", "paragraph:2"],
                }
            ],
            asset_id="musique.answerable.dev",
            processor="musique_zip_v1",
            cell_id="musique.answerable",
            split="dev",
        )
        adapter = PreparedQABenchmarkAdapter(
            prepared,
            benchmark_id="musique.answerable",
            split="dev",
        )

        metrics = adapter.score(
            adapter.examples("dev")[0],
            QAPrediction("The Ada.", support_ids=("paragraph:0",)),
        )

        self.assertEqual(set(metrics), {"answer_em", "answer_f1", "support_f1"})
        self.assertEqual(metrics["answer_em"], 1.0)
        self.assertEqual(metrics["answer_f1"], 1.0)
        self.assertAlmostEqual(metrics["support_f1"], 2 / 3)

    def test_hotpot_profile_applies_categorical_hard_zero_and_joint_metrics(self) -> None:
        first_support = _canonical(["Document", 0])
        second_support = _canonical(["Other", 1])
        prepared = _prepared(
            self.root,
            inputs=[
                _input(
                    benchmark_cell="hotpotqa.distractor",
                    source_split="validation",
                )
            ],
            gold=[
                {
                    "id": "q1",
                    "answers": ["yes"],
                    "support_ids": [first_support, second_support],
                }
            ],
            asset_id="hotpotqa.distractor.validation",
            processor="hotpotqa_parquet_v1",
            cell_id="hotpotqa.distractor",
            split="validation",
        )
        adapter = PreparedQABenchmarkAdapter(
            prepared,
            benchmark_id="hotpotqa.distractor",
            split="validation",
        )

        metrics = adapter.score(
            adapter.examples("validation")[0],
            QAPrediction("yes, perhaps", support_ids=(first_support,)),
        )

        self.assertEqual(
            set(metrics),
            {
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
            },
        )
        self.assertEqual(metrics["answer_f1"], 0.0)
        self.assertEqual(metrics["answer_precision"], 0.0)
        self.assertEqual(metrics["answer_recall"], 0.0)
        self.assertEqual(metrics["support_precision"], 1.0)
        self.assertEqual(metrics["support_recall"], 0.5)
        self.assertAlmostEqual(metrics["support_f1"], 2 / 3)
        self.assertEqual(metrics["joint_f1"], 0.0)

    def test_two_wiki_profile_accepts_alias_evidence_and_lowercases_support_title(self) -> None:
        support_one = _canonical(["Document", 0])
        support_two = _canonical(["Other", 1])
        evidence_one = _canonical(["Ada Lovelace", "occupation", "mathematician"])
        evidence_alias = _canonical(["Augusta Ada King", "occupation", "Mathematician"])
        evidence_two = _canonical(["Ada Lovelace", "born in", "London"])
        prepared = _prepared(
            self.root,
            inputs=[_input(benchmark_cell="2wiki.multihop", source_split="dev")],
            gold=[
                {
                    "id": "q1",
                    "answers": ["yes"],
                    "support_ids": [support_one, support_two],
                    "evidence_ids": [evidence_one, evidence_two],
                    "evaluator_metadata": {
                        "evidence_options": [
                            [evidence_one, evidence_alias],
                            [evidence_two],
                        ]
                    },
                }
            ],
            asset_id="2wiki.multihop.dev",
            processor="two_wiki_zip_v1",
            cell_id="2wiki.multihop",
            split="dev",
        )
        adapter = PreparedQABenchmarkAdapter(
            prepared,
            benchmark_id="2wiki.multihop",
            split="dev",
        )
        example = adapter.examples("dev")[0]

        metrics = adapter.score(
            example,
            QAPrediction(
                "yes",
                support_ids=(_canonical(["document", 0]),),
                evidence_ids=(evidence_alias,),
            ),
        )

        self.assertEqual(metrics["answer_em"], 1.0)
        self.assertEqual(metrics["support_precision"], 1.0)
        self.assertEqual(metrics["support_recall"], 0.5)
        self.assertEqual(metrics["evidence_precision"], 1.0)
        self.assertEqual(metrics["evidence_recall"], 0.5)
        self.assertEqual(metrics["joint_precision"], 1.0)
        self.assertEqual(metrics["joint_recall"], 0.25)
        self.assertEqual(metrics["joint_f1"], 0.4)

        normalized_duplicate = _canonical(["ADA LOVELACE", "occupation", "MATHEMATICIAN"])
        deduplicated = adapter.score(
            example,
            QAPrediction(
                "yes",
                support_ids=(_canonical(["document", 0]),),
                evidence_ids=(evidence_one, normalized_duplicate),
            ),
        )
        self.assertEqual(deduplicated["evidence_precision"], 1.0)
        self.assertEqual(deduplicated["evidence_recall"], 0.5)
        self.assertLessEqual(deduplicated["evidence_f1"], 1.0)
        with self.assertRaisesRegex(ValidationError, "canonical JSON"):
            adapter.score(
                example,
                QAPrediction("yes", evidence_ids=(' ["Ada","r","B"]',)),
            )

    def test_qasper_profile_maximizes_answer_and_evidence_independently(self) -> None:
        prepared = _prepared(
            self.root,
            inputs=[_input(benchmark_cell="qasper.v0_3_dev", source_split="dev")],
            gold=[
                {
                    "id": "q1",
                    "reference_groups": [
                        {
                            "answers": ["alpha"],
                            "answer_type": "extractive",
                            "evidence_texts": ["Evidence one."],
                        },
                        {
                            "answers": ["beta"],
                            "answer_type": "abstractive",
                            "evidence_texts": ["Evidence two.", "Evidence three."],
                        },
                    ],
                }
            ],
            asset_id="qasper.dev.official",
            processor="qasper_tgz_v1",
            cell_id="qasper.v0_3_dev",
            split="dev",
        )
        adapter = PreparedQABenchmarkAdapter(
            prepared,
            benchmark_id="qasper.v0_3_dev",
            split="dev",
        )
        example = adapter.examples("dev")[0]
        self.assertEqual(
            adapter.metric_keys,
            ("answer_em", "answer_f1", "evidence_f1"),
        )

        metrics = adapter.score(
            example,
            QAPrediction("alpha", evidence_texts=("Evidence two.",)),
        )

        self.assertEqual(metrics["answer_em"], 1.0)
        self.assertEqual(metrics["answer_f1"], 1.0)
        self.assertAlmostEqual(metrics["evidence_f1"], 2 / 3)
        groups = example.evaluator_gold["reference_groups"]
        self.assertEqual(groups[0]["answer_type"], "extractive")
        self.assertEqual(groups[1]["answer_type"], "abstractive")
        with self.assertRaises(TypeError):
            groups[0]["answer_type"] = "late mutation"

    def test_named_profiles_reject_incompatible_gold_schema(self) -> None:
        with self.assertRaisesRegex(ArtifactError, "QASPER gold requires"):
            PreparedQABenchmarkAdapter(
                _prepared(
                    self.root,
                    inputs=[_input(benchmark_cell="qasper.v0_3_dev", source_split="dev")],
                    gold=[
                        {
                            "id": "q1",
                            "answers": ["Ada"],
                            "evidence_texts": ["Evidence."],
                        }
                    ],
                    asset_id="qasper.dev.official",
                    processor="qasper_tgz_v1",
                    cell_id="qasper.v0_3_dev",
                    split="dev",
                ),
                benchmark_id="qasper.v0_3_dev",
                split="dev",
            )


if __name__ == "__main__":
    unittest.main()
