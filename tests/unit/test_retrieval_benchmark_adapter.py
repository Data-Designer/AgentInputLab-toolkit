from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.benchmarks.preprocess import PreparedBenchmark
from agentinputlab.benchmarks.retrieval import (
    PreparedRetrievalBenchmarkAdapter,
    RetrievalPrediction,
)
from agentinputlab.benchmarks.runtime import BenchmarkBatchRunner
from agentinputlab.contracts.errors import ArtifactError, ValidationError

_HAS_IR_MEASURES = importlib.util.find_spec("ir_measures") is not None


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _prepared(
    tmp_path: Path,
    *,
    asset_id: str = "beir.arguana.official",
    processor: str = "beir_zip_v1",
    cell_id: str = "beir.arguana",
    split: str = "test",
    benchmark_cell: str = "beir.arguana",
) -> PreparedBenchmark:
    inputs_hash = _write_jsonl(
        tmp_path / "inputs.jsonl",
        [
            {
                "id": "q1",
                "query": "counterargument one",
                "instruction": "Retrieve the most relevant counterargument.",
                "visible_metadata": {"benchmark_cell": benchmark_cell},
            },
            {
                "id": "q2",
                "query": "counterargument two",
                "instruction": "Retrieve the most relevant counterargument.",
                "visible_metadata": {"benchmark_cell": benchmark_cell},
            },
        ],
    )
    gold_hash = _write_jsonl(
        tmp_path / "gold.jsonl",
        [
            {"id": "q1", "relevance": {"d1": 1}},
            {"id": "q2", "relevance": {"d2": 1}},
        ],
    )
    corpus_hash = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"id": "d1", "title": "one", "text": "document one"},
            {"id": "d2", "title": "two", "text": "document two"},
            {"id": "noise", "title": "noise", "text": "irrelevant"},
        ],
    )
    return PreparedBenchmark(
        asset_id=asset_id,
        processor=processor,
        processor_revision="1",
        source_sha256="a" * 64,
        root=tmp_path,
        counts={"inputs": 2, "gold": 2, "corpus": 3, "qrels": 2},
        file_sha256={
            "inputs.jsonl": inputs_hash,
            "gold.jsonl": gold_hash,
            "corpus.jsonl": corpus_hash,
        },
        cell_id=cell_id,
        split=split,
    )


def _bright_query_prepared(
    root: Path,
    *,
    cell_id: str = "bright.earth_science",
    split: str = "test",
    visible_cell: str = "bright.earth_science",
    visible_split: str | None = "test",
    evaluator_metadata: object | None = None,
    asset_id: str | None = None,
    processor: str = "bright_examples_parquet_v1",
) -> PreparedBenchmark:
    root.mkdir(parents=True, exist_ok=True)
    visible_metadata: dict[str, object] = {
        "benchmark_cell": visible_cell,
        "source_query_id": "bright-q1",
    }
    if visible_split is not None:
        visible_metadata["source_split"] = visible_split
    inputs_hash = _write_jsonl(
        root / "inputs.jsonl",
        [
            {
                "id": "bright-q1",
                "query": "Why does this geological process occur?",
                "instruction": "Retrieve evidence needed to answer the query.",
                "visible_metadata": visible_metadata,
            }
        ],
    )
    if evaluator_metadata is None:
        evaluator_metadata = {
            "excluded_document_ids": ["excluded-doc"],
            "gold_answer": "evaluator-only-secret",
        }
    gold_hash = _write_jsonl(
        root / "gold.jsonl",
        [
            {
                "id": "bright-q1",
                "relevance": {"relevant-doc": 1},
                "evaluator_metadata": evaluator_metadata,
            }
        ],
    )
    return PreparedBenchmark(
        asset_id=asset_id or f"{cell_id}.examples",
        processor=processor,
        processor_revision="1",
        source_sha256="b" * 64,
        root=root,
        counts={"inputs": 1, "gold": 1, "qrels": 1},
        file_sha256={"inputs.jsonl": inputs_hash, "gold.jsonl": gold_hash},
        cell_id=cell_id,
        split=split,
    )


def _bright_corpus_prepared(
    root: Path,
    *,
    cell_id: str = "bright.earth_science",
    split: str = "test",
    asset_id: str | None = None,
    processor: str = "bright_corpus_parquet_v1",
) -> PreparedBenchmark:
    root.mkdir(parents=True, exist_ok=True)
    corpus_hash = _write_jsonl(
        root / "corpus.jsonl",
        [
            {"id": "relevant-doc", "title": "Relevant", "text": "Evidence."},
            {"id": "excluded-doc", "title": "Excluded", "text": "Source prompt."},
        ],
    )
    return PreparedBenchmark(
        asset_id=asset_id or f"{cell_id}.corpus",
        processor=processor,
        processor_revision="1",
        source_sha256="c" * 64,
        root=root,
        counts={"corpus": 2},
        file_sha256={"corpus.jsonl": corpus_hash},
        cell_id=cell_id,
        split=split,
    )


class PreparedRetrievalBenchmarkAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.tmp_path = Path(self._temporary_directory.name)

    def test_prepared_fingerprint_is_portable_source_bound_and_does_not_rescan(self):
        source = _prepared(self.tmp_path)
        adapter = PreparedRetrievalBenchmarkAdapter(source, benchmark_id="beir.arguana")
        with patch.object(Path, "open", side_effect=AssertionError("no file IO in fingerprint")):
            identity = adapter.fingerprint()
        self.assertEqual(identity["query_preparation"]["file_sha256"], dict(source.file_sha256))
        self.assertEqual(identity["corpus_preparation"]["source_sha256"], source.source_sha256)
        self.assertTrue(identity["ignore_identical_ids"])
        self.assertEqual(identity["test_queries"], 2)
        self.assertNotIn(str(self.tmp_path), json.dumps(identity))
        self.assertNotIn("counterargument one", json.dumps(identity))
        self.assertNotIn("relevance", json.dumps(identity))
        identity["query_preparation"]["file_sha256"]["inputs.jsonl"] = "0" * 64
        self.assertEqual(adapter.fingerprint()["query_preparation"]["file_sha256"],
                         dict(source.file_sha256))

    @unittest.skipUnless(_HAS_IR_MEASURES, "install agentinputlab[evaluation]")
    def test_prepared_retrieval_adapter_runs_official_ir_measures(self) -> None:
        adapter = PreparedRetrievalBenchmarkAdapter(
            _prepared(self.tmp_path),
            benchmark_id="beir.arguana",
        )
        seen: list[str] = []

        def predict(state):
            serialized = str(state.canonical_dict())
            self.assertNotIn("relevance", serialized)
            seen.append(state.task_id)
            target = "d1" if state.task_id == "q1" else "d2"
            return RetrievalPrediction.from_ranked_ids((target, "noise"))

        result = BenchmarkBatchRunner().run(adapter, "test", predict)

        self.assertEqual(seen, ["q1", "q2"])
        self.assertEqual(
            result.aggregate_metrics,
            {"ndcg_at_10": 1.0, "recall": 1.0},
        )

    def test_retrieval_prediction_and_split_fail_closed(self) -> None:
        adapter = PreparedRetrievalBenchmarkAdapter(
            _prepared(self.tmp_path),
            benchmark_id="beir.arguana",
        )
        with self.assertRaisesRegex(ValidationError, "exposes split"):
            adapter.examples("train")
        with self.assertRaisesRegex(ValidationError, "duplicates"):
            RetrievalPrediction.from_ranked_ids(("d1", "d1"))
        with self.assertRaisesRegex(ValidationError, "finite"):
            RetrievalPrediction({"d1": float("nan")})

    def test_named_arguana_profile_locks_prepared_identity(self) -> None:
        prepared = _prepared(self.tmp_path)
        PreparedRetrievalBenchmarkAdapter(prepared, benchmark_id="beir.arguana")

        invalid_fields = {
            "asset_id": "custom.asset",
            "processor": "custom_processor_v1",
            "cell_id": "",
            "split": "",
        }
        for field, value in invalid_fields.items():
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    ArtifactError,
                    rf"query/gold {field}",
                ),
            ):
                PreparedRetrievalBenchmarkAdapter(
                    replace(prepared, **{field: value}),
                    benchmark_id="beir.arguana",
                )

        with self.assertRaisesRegex(ArtifactError, "requires adapter split 'test'"):
            PreparedRetrievalBenchmarkAdapter(
                prepared,
                benchmark_id="beir.arguana",
                split="dev",
            )
        with self.assertRaisesRegex(ArtifactError, "does not accept a separate corpus"):
            PreparedRetrievalBenchmarkAdapter(
                prepared,
                benchmark_id="beir.arguana",
                corpus_prepared=prepared,
            )

    def test_unknown_retrieval_profile_keeps_generic_identity_compatibility(self) -> None:
        prepared = _prepared(
            self.tmp_path,
            asset_id="custom.retrieval.asset",
            processor="custom_retrieval_v1",
            cell_id="",
            split="",
            benchmark_cell="custom.retrieval",
        )

        adapter = PreparedRetrievalBenchmarkAdapter(
            prepared,
            benchmark_id="custom.retrieval",
            split="test",
        )

        self.assertEqual(len(adapter.examples("test")), 2)

    @patch("agentinputlab.benchmarks.retrieval.retrieval_aggregates")
    def test_beir_filters_identical_query_document_id_before_scoring(
        self,
        retrieval_aggregates,
    ) -> None:
        retrieval_aggregates.return_value = {"ndcg_at_10": 1.0, "recall": 1.0}
        adapter = PreparedRetrievalBenchmarkAdapter(
            _prepared(self.tmp_path),
            benchmark_id="beir.arguana",
        )
        example = adapter.examples("test")[0]

        adapter.score(example, RetrievalPrediction.from_ranked_ids(("q1", "d1")))

        qrels, run = retrieval_aggregates.call_args.args
        self.assertEqual(qrels, {"q1": {"d1": 1}})
        self.assertEqual(run, {"q1": {"d1": 1.0}})

    def test_prepared_retrieval_adapter_rejects_missing_qrels(self) -> None:
        prepared = _prepared(self.tmp_path)
        _write_jsonl(
            self.tmp_path / "gold.jsonl",
            [{"id": "q1", "relevance": {"d1": 1}}],
        )
        with self.assertRaisesRegex(ArtifactError, "IDs do not match"):
            PreparedRetrievalBenchmarkAdapter(prepared, benchmark_id="beir.arguana")

    @patch("agentinputlab.benchmarks.retrieval.retrieval_aggregates")
    def test_bright_uses_separate_corpus_and_filters_excluded_before_scoring(
        self,
        retrieval_aggregates,
    ) -> None:
        retrieval_aggregates.return_value = {"ndcg_at_10": 1.0, "recall": 1.0}
        query_prepared = _bright_query_prepared(self.tmp_path / "bright-query")
        corpus_prepared = _bright_corpus_prepared(self.tmp_path / "bright-corpus")
        adapter = PreparedRetrievalBenchmarkAdapter(
            query_prepared,
            benchmark_id="bright.earth_science",
            split="test",
            corpus_prepared=corpus_prepared,
        )
        example = adapter.examples("test")[0]

        self.assertIs(adapter.query_prepared, query_prepared)
        self.assertIs(adapter.corpus_prepared, corpus_prepared)
        self.assertEqual(adapter.corpus_path, corpus_prepared.corpus_path)
        evaluator_metadata = example.evaluator_gold["evaluator_metadata"]
        self.assertEqual(
            evaluator_metadata["excluded_document_ids"],
            ("excluded-doc",),
        )
        with self.assertRaises(TypeError):
            evaluator_metadata["late"] = True
        visible_payload = json.dumps(example.visible_state().canonical_dict())
        self.assertNotIn("excluded-doc", visible_payload)
        self.assertNotIn("evaluator-only-secret", visible_payload)

        metrics = adapter.score(
            example,
            RetrievalPrediction.from_ranked_ids(("excluded-doc", "relevant-doc")),
        )

        self.assertEqual(metrics, {"ndcg_at_10": 1.0, "recall": 1.0})
        qrels, run = retrieval_aggregates.call_args.args
        self.assertEqual(qrels, {"bright-q1": {"relevant-doc": 1}})
        self.assertEqual(run, {"bright-q1": {"relevant-doc": 1.0}})

    def test_bright_component_and_row_identities_fail_closed(self) -> None:
        corpus = _bright_corpus_prepared(self.tmp_path / "corpus")
        with self.assertRaisesRegex(ArtifactError, "query/gold cell_id"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(
                    self.tmp_path / "query-wrong-cell",
                    cell_id="bright.economics",
                    asset_id="bright.earth_science.examples",
                ),
                benchmark_id="bright.earth_science",
                corpus_prepared=corpus,
            )
        with self.assertRaisesRegex(ArtifactError, "query/gold split"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(self.tmp_path / "query-wrong-split", split="dev"),
                benchmark_id="bright.earth_science",
                corpus_prepared=corpus,
            )
        with self.assertRaisesRegex(ArtifactError, "corpus cell_id"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(self.tmp_path / "query-corpus-cell"),
                benchmark_id="bright.earth_science",
                corpus_prepared=_bright_corpus_prepared(
                    self.tmp_path / "corpus-wrong-cell",
                    cell_id="bright.economics",
                    asset_id="bright.earth_science.corpus",
                ),
            )
        with self.assertRaisesRegex(ArtifactError, "corpus split"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(self.tmp_path / "query-corpus-split"),
                benchmark_id="bright.earth_science",
                corpus_prepared=_bright_corpus_prepared(
                    self.tmp_path / "corpus-wrong-split",
                    split="dev",
                ),
            )
        with self.assertRaisesRegex(ArtifactError, "corpus cell_id"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(self.tmp_path / "query-corpus-no-cell"),
                benchmark_id="bright.earth_science",
                corpus_prepared=_bright_corpus_prepared(
                    self.tmp_path / "corpus-no-cell",
                    cell_id="",
                    asset_id="bright.earth_science.corpus",
                ),
            )
        with self.assertRaisesRegex(ArtifactError, "benchmark_cell"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(
                    self.tmp_path / "query-visible-cell",
                    visible_cell="bright.economics",
                ),
                benchmark_id="bright.earth_science",
                corpus_prepared=corpus,
            )
        with self.assertRaisesRegex(ArtifactError, "source_split"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(
                    self.tmp_path / "query-visible-split",
                    visible_split="dev",
                ),
                benchmark_id="bright.earth_science",
                corpus_prepared=corpus,
            )
        with self.assertRaisesRegex(ArtifactError, "requires source_split"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(
                    self.tmp_path / "query-no-visible-split",
                    visible_split=None,
                ),
                benchmark_id="bright.earth_science",
                corpus_prepared=corpus,
            )

    def test_bright_named_profile_locks_query_and_corpus_components(self) -> None:
        query = _bright_query_prepared(self.tmp_path / "query")
        corpus = _bright_corpus_prepared(self.tmp_path / "corpus")
        PreparedRetrievalBenchmarkAdapter(
            query,
            benchmark_id="bright.earth_science",
            corpus_prepared=corpus,
        )

        invalid_query_fields = {
            "asset_id": "custom.examples",
            "processor": "custom_examples_v1",
            "cell_id": "",
            "split": "",
        }
        for field, value in invalid_query_fields.items():
            with (
                self.subTest(component="query", field=field),
                self.assertRaisesRegex(
                    ArtifactError,
                    rf"query/gold {field}",
                ),
            ):
                PreparedRetrievalBenchmarkAdapter(
                    replace(query, **{field: value}),
                    benchmark_id="bright.earth_science",
                    corpus_prepared=corpus,
                )

        invalid_corpus_fields = {
            "asset_id": "custom.corpus",
            "processor": "custom_corpus_v1",
            "cell_id": "",
            "split": "",
        }
        for field, value in invalid_corpus_fields.items():
            with (
                self.subTest(component="corpus", field=field),
                self.assertRaisesRegex(ArtifactError, rf"corpus {field}"),
            ):
                PreparedRetrievalBenchmarkAdapter(
                    query,
                    benchmark_id="bright.earth_science",
                    corpus_prepared=replace(corpus, **{field: value}),
                )

        with self.assertRaisesRegex(ArtifactError, "requires its locked corpus"):
            PreparedRetrievalBenchmarkAdapter(
                query,
                benchmark_id="bright.earth_science",
            )
        with self.assertRaisesRegex(ArtifactError, "requires adapter split 'test'"):
            PreparedRetrievalBenchmarkAdapter(
                query,
                benchmark_id="bright.earth_science",
                split="dev",
                corpus_prepared=corpus,
            )

    def test_bright_economics_named_profile_accepts_only_its_locked_pair(self) -> None:
        query = _bright_query_prepared(
            self.tmp_path / "query-economics",
            cell_id="bright.economics",
            visible_cell="bright.economics",
        )
        corpus = _bright_corpus_prepared(
            self.tmp_path / "corpus-economics",
            cell_id="bright.economics",
        )

        adapter = PreparedRetrievalBenchmarkAdapter(
            query,
            benchmark_id="bright.economics",
            corpus_prepared=corpus,
        )

        self.assertEqual(adapter.corpus_path, corpus.corpus_path)

    def test_bright_gold_and_example_identity_fail_closed(self) -> None:
        corpus = _bright_corpus_prepared(self.tmp_path / "corpus")
        with self.assertRaisesRegex(ArtifactError, "JSON array"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(
                    self.tmp_path / "query-bad-exclusions",
                    evaluator_metadata={"excluded_document_ids": "excluded-doc"},
                ),
                benchmark_id="bright.earth_science",
                corpus_prepared=corpus,
            )
        with self.assertRaisesRegex(ArtifactError, "overlap relevance"):
            PreparedRetrievalBenchmarkAdapter(
                _bright_query_prepared(
                    self.tmp_path / "query-overlap",
                    evaluator_metadata={"excluded_document_ids": ["relevant-doc"]},
                ),
                benchmark_id="bright.earth_science",
                corpus_prepared=corpus,
            )

        adapter = PreparedRetrievalBenchmarkAdapter(
            _bright_query_prepared(self.tmp_path / "query-valid"),
            benchmark_id="bright.earth_science",
            corpus_prepared=corpus,
        )
        with self.assertRaisesRegex(ValidationError, "does not belong"):
            adapter.score(
                BenchmarkExample(example_id="bright-q1", query="impostor"),
                ("relevant-doc",),
            )


if __name__ == "__main__":
    unittest.main()
