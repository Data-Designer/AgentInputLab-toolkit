from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from agentinputlab.benchmarks import BenchmarkSmokeAsset, prepare_smoke_asset
from agentinputlab.contracts.errors import ArtifactError

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installations
    pa = None
    pq = None

_HAS_PYARROW = pa is not None and pq is not None


def _asset(
    source: Path,
    *,
    asset_id: str,
    cell_id: str,
    processor: str,
    expected_counts: dict[str, int],
) -> BenchmarkSmokeAsset:
    content = source.read_bytes()
    return BenchmarkSmokeAsset(
        asset_id=asset_id,
        cell_id=cell_id,
        source_url="https://example.com/official-source",
        source_revision="fixture-v1",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC-BY-4.0",
        redistribution="not_redistributed",
        status="verified_smoke",
        processor=processor,
        expected_counts=expected_counts,
        prepared_sha256={},
        allowed_hosts=("example.com",),
        provenance={"fixture": "unit-test"},
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    assert pa is not None
    assert pq is not None
    pq.write_table(pa.Table.from_pylist(rows), path)


def _musique_zip(*, unsafe_member: str | None = None) -> bytes:
    example = {
        "id": "musique-q1",
        "question": "Which public entity satisfies the two-hop relation?",
        "answer": "Gold-only answer",
        "answer_aliases": ["Gold-only alias"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 1,
                "title": "Distractor",
                "paragraph_text": "This paragraph is visible but is not supporting.",
                "is_supporting": False,
            },
            {
                "idx": 0,
                "title": "Evidence",
                "paragraph_text": "This public paragraph contains relevant evidence.",
                "is_supporting": True,
            },
        ],
        "question_decomposition": [
            {
                "id": 0,
                "question": "Private decomposition question",
                "answer": "Private decomposition answer",
            }
        ],
    }
    audit = {
        "nq": [
            {
                "id": 17,
                "question": "Potentially overlapping single-hop question",
            }
        ]
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "data/musique_ans_v1.0_dev.jsonl",
            json.dumps(example) + "\n",
        )
        archive.writestr(
            "data/dev_test_singlehop_questions_v1.0.json",
            json.dumps(audit),
        )
        if unsafe_member is not None:
            archive.writestr(unsafe_member, "escape")
    return stream.getvalue()


def _two_wiki_zip() -> bytes:
    example = {
        "_id": "2wiki-q1",
        "answer": "Answer Canonical",
        "answer_id": "QSUB",
        "context": [
            ["Page B", ["A second public sentence."]],
            ["Page A", ["A first public sentence."]],
        ],
        "entity_ids": '["QSUB","QOBJ"]',
        "evidences": [["Subject Literal", "relates_to", "Object Literal"]],
        "evidences_id": [["QSUB", "relates_to", "QOBJ"]],
        "question": "Which entity satisfies the public relation?",
        "supporting_facts": [["Page A", 0], ["Page B", 0]],
        "type": "compositional",
    }
    aliases = [
        {"Q_id": "QOBJ", "aliases": ["Object Alias"], "demonyms": []},
        {"Q_id": "QSUB", "aliases": ["Answer Alias"], "demonyms": []},
    ]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dev.json", json.dumps([example]))
        archive.writestr(
            "id_aliases.json",
            "".join(json.dumps(row) + "\n" for row in aliases),
        )
    return stream.getvalue()


def _qasper_tgz() -> bytes:
    papers = {
        "paper-1": {
            "full_text": [
                {
                    "section_name": "Introduction",
                    "paragraphs": ["A public paragraph.\nIt remains visible."],
                }
            ],
            "figures_and_tables": [{"caption": "A public figure caption.", "file": "figure-1.png"}],
            "qas": [
                {
                    "question_id": "qasper-q1",
                    "question": "What conclusion do the authors report?",
                    "answers": [
                        {
                            "annotation_id": "annotation-1",
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["Extractive gold secret"],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": ["  Evidence one.\n"],
                            },
                        },
                        {
                            "annotation_id": "annotation-2",
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": [],
                                "free_form_answer": "Abstractive gold secret",
                                "yes_no": None,
                                "evidence": ["Evidence two."],
                            },
                        },
                    ],
                }
            ],
        }
    }
    payload = json.dumps(papers).encode("utf-8")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo("qasper-dev-v0.3.json")
        member.size = len(payload)
        member.mtime = 0
        archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def _tau3_banking_tar(*, unsafe: bool = False) -> bytes:
    revision = "fixture-v1"
    root = f"tau2-bench-{revision}"
    documents = [
        {"id": "doc_public_001", "title": "Public policy A", "content": "Public A."},
        {"id": "doc_public_002", "title": "Public policy B", "content": "Public B."},
    ]
    tasks = []
    for index, document in enumerate(documents, start=1):
        task_id = f"task_{index:03d}"
        tasks.append(
            {
                "annotations": None,
                "description": {"purpose": task_id},
                "evaluation_criteria": {
                    "actions": [
                        {
                            "arguments": {"private": f"gold-action-{index}"},
                            "name": "submit_private_action",
                        }
                    ],
                    "communicate_info": [],
                    "reward_basis": ["DB"],
                },
                "id": task_id,
                "initial_state": (
                    {"initialization_data": {"private": "initial-db-secret"}}
                    if index == 1
                    else None
                ),
                "required_documents": [document["id"]],
                "user_scenario": {
                    "instructions": f"hidden-user-simulator-script-{index}",
                    "persona": None,
                },
                "user_tools": ["private_user_tool"],
            }
        )
    legacy = json.loads(json.dumps(tasks))
    legacy[1]["description"]["purpose"] = "stale legacy task"

    members: dict[str, bytes] = {
        "LICENSE": (
            b"MIT License\n\nPermission is hereby granted, free of charge, to any person "
            b"obtaining a copy.\n"
        ),
        "pyproject.toml": b'[project]\nname = "tau2"\nversion = "1.0.1"\n',
        "data/tau2/domains/banking_knowledge/db.json": json.dumps(
            {"private_database_secret": "evaluator-only"}
        ).encode(),
        "data/tau2/domains/banking_knowledge/tasks.json": json.dumps(legacy).encode(),
        "data/tau2/domains/banking_knowledge/prompts/policy.md": b"Public runtime policy.",
        "src/tau2/domains/banking_knowledge/environment.py": b"# environment\n",
        "src/tau2/domains/banking_knowledge/retrieval.py": b"# retrieval\n",
        "src/tau2/domains/banking_knowledge/tools.py": b"# tools\n",
        "src/tau2/data_model/tasks.py": b"# task model\n",
        "src/tau2/environment/environment.py": b"# shared environment\n",
        "src/tau2/evaluator/evaluator.py": b"# evaluator\n",
        "src/tau2/orchestrator/orchestrator.py": b"# orchestrator\n",
        "src/tau2/runner/checkpoint.py": b"# checkpoint\n",
        "src/tau2/user/user_simulator.py": b"# user simulator\n",
    }
    for task in tasks:
        members[f"data/tau2/domains/banking_knowledge/tasks/{task['id']}.json"] = json.dumps(
            task
        ).encode()
    for document in documents:
        members[f"data/tau2/domains/banking_knowledge/documents/{document['id']}.json"] = (
            json.dumps(document).encode()
        )
    if unsafe:
        members["../escape"] = b"unsafe"

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for relative, payload in members.items():
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


class DatasetPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)

    @unittest.skipUnless(_HAS_PYARROW, "install agentinputlab[benchmarks]")
    def test_bright_components_are_deterministic_and_keep_gold_separate(self) -> None:
        examples_source = self.root / "bright-examples.parquet"
        _write_parquet(
            examples_source,
            [
                {
                    "excluded_ids": ["excluded-doc", "N/A"],
                    "gold_answer": "Gold answer two",
                    "gold_ids": ["doc-2", "doc-3"],
                    "gold_ids_long": ["doc-2", "doc-3", "doc-4"],
                    "id": "query-2",
                    "query": "Second visible query",
                    "reasoning": "Private reasoning two",
                },
                {
                    "excluded_ids": [],
                    "gold_answer": "Gold answer one",
                    "gold_ids": ["doc-1"],
                    "gold_ids_long": ["doc-1"],
                    "id": "query-1",
                    "query": "First visible query",
                    "reasoning": "Private reasoning one",
                },
            ],
        )
        corpus_source = self.root / "bright-corpus.parquet"
        _write_parquet(
            corpus_source,
            [
                {"content": "Second document.", "id": "doc-2"},
                {"content": "First document.", "id": "doc-1"},
            ],
        )
        examples_asset = _asset(
            examples_source,
            asset_id="fixture.bright.examples",
            cell_id="bright.earth_science",
            processor="bright_examples_parquet_v1",
            expected_counts={
                "excluded_documents": 1,
                "gold": 2,
                "inputs": 2,
                "qrels": 3,
            },
        )
        corpus_asset = _asset(
            corpus_source,
            asset_id="fixture.bright.corpus",
            cell_id="bright.earth_science",
            processor="bright_corpus_parquet_v1",
            expected_counts={"corpus": 2},
        )

        examples = prepare_smoke_asset(
            examples_asset,
            examples_source,
            self.root / "prepared-examples-a",
        )
        repeated = prepare_smoke_asset(
            examples_asset,
            examples_source,
            self.root / "prepared-examples-b",
        )
        corpus = prepare_smoke_asset(
            corpus_asset,
            corpus_source,
            self.root / "prepared-corpus",
        )

        self.assertEqual(dict(examples.counts), dict(examples_asset.expected_counts))
        self.assertEqual(dict(examples.file_sha256), dict(repeated.file_sha256))
        self.assertEqual(examples.inputs_path.read_bytes(), repeated.inputs_path.read_bytes())
        self.assertIsNone(examples.corpus_path)
        self.assertIsNotNone(corpus.corpus_path)
        visible = examples.inputs_path.read_text(encoding="utf-8")
        gold = examples.gold_path.read_text(encoding="utf-8")
        self.assertNotIn("Gold answer", visible)
        self.assertNotIn("Private reasoning", visible)
        self.assertNotIn("excluded-doc", visible)
        self.assertIn("Gold answer", gold)
        self.assertIn("Private reasoning", gold)
        self.assertIn("excluded-doc", gold)
        assert corpus.corpus_path is not None
        self.assertEqual(
            _read_jsonl(corpus.corpus_path),
            [
                {"id": "doc-1", "text": "First document.", "title": ""},
                {"id": "doc-2", "text": "Second document.", "title": ""},
            ],
        )

        for cell, label in (
            ("bright.robotics", "Robotics"),
            ("bright.stackoverflow", "Stack Overflow"),
        ):
            with self.subTest(cell=cell):
                query_prepared = prepare_smoke_asset(
                    replace(examples_asset, cell_id=cell),
                    examples_source,
                    self.root / cell / "queries",
                )
                corpus_prepared = prepare_smoke_asset(
                    replace(corpus_asset, cell_id=cell),
                    corpus_source,
                    self.root / cell / "corpus",
                )
                query = _read_jsonl(query_prepared.inputs_path)[0]
                self.assertIn(label, query["instruction"])
                self.assertEqual(query["visible_metadata"]["benchmark_cell"], cell)
                self.assertEqual(
                    query_prepared.gold_path.read_bytes(), examples.gold_path.read_bytes()
                )
                self.assertEqual(
                    corpus_prepared.corpus_path.read_bytes(), corpus.corpus_path.read_bytes()
                )
                self.assertNotIn("Gold answer", query_prepared.inputs_path.read_text())

    def test_musique_zip_preserves_leakage_audit_and_hides_labels(self) -> None:
        source = self.root / "musique.zip"
        source.write_bytes(_musique_zip())
        expected_counts = {
            "context_blocks": 2,
            "gold": 1,
            "inputs": 1,
            "leakage_audit_records": 1,
            "leakage_audit_sources": 1,
            "support_labels": 1,
        }
        asset = _asset(
            source,
            asset_id="fixture.musique",
            cell_id="musique.answerable",
            processor="musique_zip_v1",
            expected_counts=expected_counts,
        )

        prepared = prepare_smoke_asset(asset, source, self.root / "prepared-musique-a")
        repeated = prepare_smoke_asset(asset, source, self.root / "prepared-musique-b")

        self.assertEqual(dict(prepared.counts), expected_counts)
        self.assertEqual(dict(prepared.file_sha256), dict(repeated.file_sha256))
        self.assertIsNotNone(prepared.audit_path)
        visible = prepared.inputs_path.read_text(encoding="utf-8")
        gold = prepared.gold_path.read_text(encoding="utf-8")
        self.assertNotIn("Gold-only answer", visible)
        self.assertNotIn("Gold-only alias", visible)
        self.assertNotIn("Private decomposition", visible)
        self.assertNotIn("is_supporting", visible)
        self.assertIn("Gold-only answer", gold)
        self.assertIn("Private decomposition", gold)
        self.assertIn('"support_ids":["paragraph:0"]', gold)
        assert prepared.audit_path is not None
        audit = _read_jsonl(prepared.audit_path)
        self.assertEqual(audit[0]["source"], "nq")
        self.assertEqual(audit[0]["source_id"], 17)
        self.assertIn("Potentially overlapping", str(audit[0]["question"]))

    def test_two_wiki_zip_preserves_aliases_evidence_and_sentence_support_ids(self) -> None:
        source = self.root / "2wiki.zip"
        source.write_bytes(_two_wiki_zip())
        expected_counts = {
            "alias_entities": 2,
            "alias_strings": 2,
            "context_blocks": 2,
            "duplicate_context_sentences": 0,
            "empty_context_sentences": 0,
            "evidence_triples": 1,
            "gold": 1,
            "inputs": 1,
            "orphan_support_labels": 0,
            "support_labels": 2,
        }
        asset = _asset(
            source,
            asset_id="fixture.2wiki",
            cell_id="2wiki.multihop",
            processor="two_wiki_zip_v1",
            expected_counts=expected_counts,
        )

        prepared = prepare_smoke_asset(asset, source, self.root / "prepared-2wiki")

        self.assertEqual(dict(prepared.counts), expected_counts)
        self.assertIsNotNone(prepared.audit_path)
        inputs = _read_jsonl(prepared.inputs_path)
        gold = _read_jsonl(prepared.gold_path)
        visible = json.dumps(inputs, sort_keys=True)
        self.assertNotIn("Answer Canonical", visible)
        self.assertNotIn("Answer Alias", visible)
        self.assertNotIn("relates_to", visible)
        blocks = inputs[0]["visible_context"]
        assert isinstance(blocks, list)
        support_ids = {
            block["metadata"]["official_support_id"]
            for block in blocks
            if isinstance(block, dict) and isinstance(block.get("metadata"), dict)
        }
        self.assertEqual(support_ids, {'["Page A",0]', '["Page B",0]'})
        self.assertEqual(
            gold[0]["support_ids"],
            ['["Page A",0]', '["Page B",0]'],
        )
        self.assertEqual(gold[0]["answers"], ["Answer Canonical", "Answer Alias"])
        self.assertEqual(
            gold[0]["evidence_ids"],
            ['["Subject Literal","relates_to","Object Literal"]'],
        )
        evaluator_metadata = gold[0]["evaluator_metadata"]
        assert isinstance(evaluator_metadata, dict)
        options = evaluator_metadata["evidence_options"]
        self.assertIn(
            '["Answer Alias","relates_to","Object Alias"]',
            options[0],
        )
        assert prepared.audit_path is not None
        audit = _read_jsonl(prepared.audit_path)
        self.assertEqual([row["id"] for row in audit], ["QOBJ", "QSUB"])

    @unittest.skipUnless(_HAS_PYARROW, "install agentinputlab[benchmarks]")
    def test_hotpot_parquet_preserves_orphans_without_gold_label_leakage(self) -> None:
        source = self.root / "hotpot.parquet"
        _write_parquet(
            source,
            [
                {
                    "answer": "Gold-only Hotpot answer",
                    "context": {
                        "title": ["Page A", "Distractor"],
                        "sentences": [
                            ["Visible supporting sentence.", "Visible non-support sentence."],
                            ["Visible distractor sentence."],
                        ],
                    },
                    "id": "hotpot-q1",
                    "level": "hard",
                    "question": "What follows from the visible passages?",
                    "supporting_facts": {
                        "title": ["Page A", "Missing Page"],
                        "sent_id": [0, 902],
                    },
                    "type": "bridge",
                }
            ],
        )
        expected_counts = {
            "context_blocks": 3,
            "duplicate_context_sentences": 0,
            "empty_context_sentences": 0,
            "gold": 1,
            "inputs": 1,
            "orphan_support_labels": 1,
            "support_labels": 2,
        }
        asset = _asset(
            source,
            asset_id="fixture.hotpot",
            cell_id="hotpotqa.distractor",
            processor="hotpotqa_parquet_v1",
            expected_counts=expected_counts,
        )

        prepared = prepare_smoke_asset(asset, source, self.root / "prepared-hotpot")

        self.assertEqual(dict(prepared.counts), expected_counts)
        inputs = _read_jsonl(prepared.inputs_path)
        gold = _read_jsonl(prepared.gold_path)
        visible = json.dumps(inputs, sort_keys=True)
        self.assertNotIn("Gold-only Hotpot answer", visible)
        self.assertNotIn("Missing Page", visible)
        self.assertNotIn('"difficulty"', visible)
        self.assertNotIn('"question_type"', visible)
        blocks = inputs[0]["visible_context"]
        assert isinstance(blocks, list)
        first_block = blocks[0]
        assert isinstance(first_block, dict)
        metadata = first_block["metadata"]
        assert isinstance(metadata, dict)
        self.assertEqual(metadata["official_support_id"], '["Page A",0]')
        evaluator_metadata = gold[0]["evaluator_metadata"]
        assert isinstance(evaluator_metadata, dict)
        self.assertEqual(
            evaluator_metadata["orphan_support_ids"],
            ['["Missing Page",902]'],
        )
        self.assertEqual(evaluator_metadata["difficulty"], "hard")
        self.assertEqual(evaluator_metadata["question_type"], "bridge")

    def test_qasper_tgz_preserves_grouped_references_outside_visible_inputs(self) -> None:
        source = self.root / "qasper.tgz"
        source.write_bytes(_qasper_tgz())
        expected_counts = {
            "annotations": 2,
            "context_blocks": 3,
            "evidence_labels": 2,
            "gold": 1,
            "inputs": 1,
            "papers": 1,
            "unique_paper_context_blocks": 3,
        }
        asset = _asset(
            source,
            asset_id="fixture.qasper",
            cell_id="qasper.v0_3_dev",
            processor="qasper_tgz_v1",
            expected_counts=expected_counts,
        )

        prepared = prepare_smoke_asset(asset, source, self.root / "prepared-qasper")

        self.assertEqual(dict(prepared.counts), expected_counts)
        inputs = _read_jsonl(prepared.inputs_path)
        gold = _read_jsonl(prepared.gold_path)
        visible = json.dumps(inputs, sort_keys=True)
        self.assertNotIn("Extractive gold secret", visible)
        self.assertNotIn("Abstractive gold secret", visible)
        self.assertNotIn("Evidence one", visible)
        self.assertNotIn("annotation-1", visible)
        groups = gold[0]["reference_groups"]
        assert isinstance(groups, list)
        self.assertEqual(
            groups,
            [
                {
                    "answer_type": "extractive",
                    "answers": ["Extractive gold secret"],
                    "evidence_texts": ["Evidence one."],
                },
                {
                    "answer_type": "abstractive",
                    "answers": ["Abstractive gold secret"],
                    "evidence_texts": ["Evidence two."],
                },
            ],
        )
        self.assertEqual(
            gold[0]["evaluator_metadata"],
            {"annotation_ids": ["annotation-1", "annotation-2"]},
        )

    def test_tau3_banking_tar_requires_checkpoints_and_separates_hidden_state(self) -> None:
        source = self.root / "tau3-banking.tar.gz"
        source.write_bytes(_tau3_banking_tar())
        expected_counts = {
            "audit": 1,
            "corpus": 2,
            "documents": 2,
            "gold": 2,
            "gold_actions": 2,
            "inputs": 2,
            "legacy_combined_task_mismatches": 1,
            "required_document_labels": 2,
            "required_document_unique": 2,
            "tasks": 2,
            "tasks_with_initial_state": 1,
            "tasks_with_initialization_data": 1,
            "tasks_with_message_history": 0,
            "user_simulator_scripts": 2,
        }
        asset = _asset(
            source,
            asset_id="fixture.tau3.banking",
            cell_id="tau3.banking_knowledge",
            processor="tau3_banking_tar_v1",
            expected_counts=expected_counts,
        )

        first = prepare_smoke_asset(asset, source, self.root / "prepared-tau3-a")
        second = prepare_smoke_asset(asset, source, self.root / "prepared-tau3-b")

        self.assertEqual(dict(first.counts), expected_counts)
        self.assertEqual(dict(first.file_sha256), dict(second.file_sha256))
        inputs = _read_jsonl(first.inputs_path)
        gold = _read_jsonl(first.gold_path)
        corpus_path = first.corpus_path
        audit_path = first.audit_path
        assert corpus_path is not None
        assert audit_path is not None
        corpus = _read_jsonl(corpus_path)
        audit = _read_jsonl(audit_path)
        visible = json.dumps(inputs, sort_keys=True)
        self.assertEqual(
            inputs[0],
            {
                "checkpoint_required": True,
                "id": "task_001",
                "visible_metadata": {
                    "benchmark_cell": "tau3.banking_knowledge",
                    "source_split": "base",
                    "source_task_id": "task_001",
                },
            },
        )
        for hidden in (
            "hidden-user-simulator-script",
            "gold-action",
            "initial-db-secret",
            "required_documents",
            "evaluation_criteria",
        ):
            self.assertNotIn(hidden, visible)
        self.assertIn("hidden-user-simulator-script-1", json.dumps(gold))
        self.assertEqual([row["id"] for row in corpus], ["doc_public_001", "doc_public_002"])
        self.assertEqual(audit[0]["package_version"], "1.0.1")
        self.assertEqual(audit[0]["legacy_combined_task_mismatch_ids"], ["task_002"])
        self.assertEqual(
            audit[0]["database"],
            {"private_database_secret": "evaluator-only"},
        )

    def test_tau3_banking_tar_rejects_path_traversal(self) -> None:
        source = self.root / "unsafe-tau3.tar.gz"
        source.write_bytes(_tau3_banking_tar(unsafe=True))
        asset = _asset(
            source,
            asset_id="fixture.tau3.unsafe",
            cell_id="tau3.banking_knowledge",
            processor="tau3_banking_tar_v1",
            expected_counts={},
        )

        with self.assertRaisesRegex(ArtifactError, "unsafe path"):
            prepare_smoke_asset(asset, source, self.root / "prepared-unsafe-tau3")

    def test_archive_path_traversal_fails_closed(self) -> None:
        source = self.root / "unsafe-musique.zip"
        source.write_bytes(_musique_zip(unsafe_member="../escape.json"))
        asset = _asset(
            source,
            asset_id="fixture.musique.unsafe",
            cell_id="musique.answerable",
            processor="musique_zip_v1",
            expected_counts={},
        )

        with self.assertRaisesRegex(ArtifactError, "unsafe path"):
            prepare_smoke_asset(asset, source, self.root / "prepared-unsafe")

    @unittest.skipUnless(_HAS_PYARROW, "install agentinputlab[benchmarks]")
    def test_parquet_schema_drift_fails_closed(self) -> None:
        source = self.root / "bright-schema-drift.parquet"
        _write_parquet(
            source,
            [
                {
                    "excluded_ids": [],
                    "gold_answer": "answer",
                    "gold_ids": ["doc-1"],
                    "gold_ids_long": ["doc-1"],
                    "id": "query-1",
                    "query": "query",
                    "renamed_reasoning": "schema drift",
                }
            ],
        )
        asset = _asset(
            source,
            asset_id="fixture.bright.schema-drift",
            cell_id="bright.earth_science",
            processor="bright_examples_parquet_v1",
            expected_counts={},
        )

        with self.assertRaisesRegex(ArtifactError, "columns do not match"):
            prepare_smoke_asset(asset, source, self.root / "prepared-schema-drift")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
