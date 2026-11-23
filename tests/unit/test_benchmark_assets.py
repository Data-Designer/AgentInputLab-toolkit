from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from agentinputlab.benchmarks import (
    BenchmarkRegistry,
    BenchmarkSmokeAsset,
    BenchmarkSmokeAssetRegistry,
    load_prepared_benchmark,
    prepare_smoke_asset,
)
from agentinputlab.contracts.errors import ArtifactError, RegistryError


def _asset(
    content: bytes,
    *,
    processor: str,
    expected_counts: dict[str, int],
) -> BenchmarkSmokeAsset:
    return BenchmarkSmokeAsset(
        asset_id=f"fixture.{processor}",
        cell_id="beir.arguana" if processor == "beir_zip_v1" else "clariq.main",
        source_url="https://example.com/data",
        source_revision="fixture-v1",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        redistribution="not_redistributed",
        status="verified_smoke",
        processor=processor,
        expected_counts=expected_counts,
        prepared_sha256={},
        allowed_hosts=("example.com",),
        provenance={"fixture": "unit-test"},
    )


def _beir_zip(*, unsafe_member: str | None = None) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "arguana/corpus.jsonl",
            "\n".join(
                (
                    json.dumps({"_id": "d2", "title": "B", "text": "second"}),
                    json.dumps({"_id": "d1", "title": "A", "text": "first"}),
                )
            )
            + "\n",
        )
        archive.writestr(
            "arguana/queries.jsonl",
            "\n".join(
                (
                    json.dumps({"_id": "q2", "text": "query two"}),
                    json.dumps({"_id": "q1", "text": "query one"}),
                )
            )
            + "\n",
        )
        archive.writestr(
            "arguana/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nq2\td2\t1\nq1\td1\t1\n",
        )
        if unsafe_member:
            archive.writestr(unsafe_member, "escape")
    return stream.getvalue()


def _clariq_tsv() -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        delimiter="\t",
        lineterminator="\n",
        fieldnames=(
            "topic_id",
            "initial_request",
            "topic_desc",
            "clarification_need",
            "facet_id",
            "facet_desc",
            "question_id",
            "question",
            "answer",
        ),
    )
    writer.writeheader()
    writer.writerow(
        {
            "topic_id": "2",
            "initial_request": "jaguar",
            "topic_desc": "Find information about Jaguar cars",
            "clarification_need": "4",
            "facet_id": "F2",
            "facet_desc": "Jaguar cars",
            "question_id": "Q2",
            "question": "Do you mean the car?",
            "answer": "Yes",
        }
    )
    writer.writerow(
        {
            "topic_id": "1",
            "initial_request": "weather today",
            "topic_desc": "Today's local weather",
            "clarification_need": "1",
            "facet_id": "F1",
            "facet_desc": "Local weather",
            "question_id": "Q00001",
            "question": "",
            "answer": "",
        }
    )
    return stream.getvalue().encode("utf-8")


class BenchmarkSmokeAssetTests(unittest.TestCase):
    def test_registry_only_defaults_to_license_verified_assets(self) -> None:
        registry = BenchmarkSmokeAssetRegistry.default()
        all_assets = registry.list(include_license_review=True)
        self.assertEqual(len(registry), 16)
        self.assertEqual(
            {asset.asset_id for asset in registry.list()},
            {
                "beir.arguana.official",
                "bfcl.v3.simple_python.v1_3",
                "bright.earth_science.examples",
                "bright.earth_science.corpus",
                "bright.economics.examples",
                "bright.economics.corpus",
                "bright.robotics.examples",
                "bright.robotics.corpus",
                "bright.stackoverflow.examples",
                "bright.stackoverflow.corpus",
                "musique.answerable.dev",
                "2wiki.multihop.dev",
                "hotpotqa.distractor.validation",
                "qasper.dev.official",
                "tau3.banking_knowledge.v1_0_1",
            },
        )
        self.assertEqual(
            {asset.status for asset in all_assets},
            {
                "verified_smoke",
                "license_review_required",
            },
        )
        known_cells = {cell.cell_id for cell in BenchmarkRegistry.default().list()}
        self.assertTrue({asset.cell_id for asset in all_assets} <= known_cells)

    def test_asset_collections_are_deeply_immutable(self) -> None:
        asset = BenchmarkSmokeAssetRegistry.default().get("beir.arguana.official")
        with self.assertRaises(TypeError):
            asset.expected_counts["inputs"] = 0  # type: ignore[index]
        with self.assertRaises(TypeError):
            asset.provenance["new"] = "value"  # type: ignore[index]
        with self.assertRaises(TypeError):
            asset.prepared_sha256["inputs.jsonl"] = "0" * 64  # type: ignore[index]

    def test_asset_id_cannot_escape_prepared_root(self) -> None:
        content = _clariq_tsv()
        with self.assertRaisesRegex(RegistryError, "flat safe identifier"):
            replace(
                _asset(
                    content,
                    processor="clariq_tsv_v1",
                    expected_counts={"inputs": 2, "gold": 2, "topics": 2},
                ),
                asset_id="../escape",
            )

    def test_beir_preprocessing_is_deterministic_and_separates_gold(self) -> None:
        content = _beir_zip()
        asset = _asset(
            content,
            processor="beir_zip_v1",
            expected_counts={
                "inputs": 2,
                "gold": 2,
                "corpus": 2,
                "qrels": 2,
                "orphan_qrels": 0,
                "orphan_queries": 0,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zip"
            source.write_bytes(content)
            first = prepare_smoke_asset(asset, source, Path(directory) / "prepared-a")
            second = prepare_smoke_asset(asset, source, Path(directory) / "prepared-b")
            loaded = load_prepared_benchmark(asset, Path(directory) / "prepared-a")
            self.assertEqual(dict(first.file_sha256), dict(second.file_sha256))
            self.assertEqual(dict(first.file_sha256), dict(loaded.file_sha256))
            self.assertEqual(first.inputs_path.read_bytes(), second.inputs_path.read_bytes())
            inputs = first.inputs_path.read_text(encoding="utf-8")
            gold = first.gold_path.read_text(encoding="utf-8")
            self.assertNotIn('"relevance"', inputs)
            self.assertIn('"relevance"', gold)
            self.assertEqual(
                first.corpus_path.read_text(encoding="utf-8").splitlines()[0],
                '{"id":"d1","text":"first","title":"A"}',
            )

    def test_existing_prepared_cache_is_reverified(self) -> None:
        content = _beir_zip()
        asset = _asset(
            content,
            processor="beir_zip_v1",
            expected_counts={
                "inputs": 2,
                "gold": 2,
                "corpus": 2,
                "qrels": 2,
                "orphan_qrels": 0,
                "orphan_queries": 0,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zip"
            source.write_bytes(content)
            prepared = prepare_smoke_asset(asset, source, Path(directory) / "prepared")
            prepared.inputs_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "cache verification"):
                prepare_smoke_asset(asset, source, Path(directory) / "prepared")
            with self.assertRaisesRegex(ArtifactError, "cache verification"):
                load_prepared_benchmark(asset, Path(directory) / "prepared")

    def test_loading_missing_prepared_cache_fails_closed(self) -> None:
        content = _clariq_tsv()
        asset = _asset(
            content,
            processor="clariq_tsv_v1",
            expected_counts={"inputs": 2, "gold": 2, "topics": 2},
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ArtifactError, "cache is missing"),
        ):
            load_prepared_benchmark(asset, directory)

    def test_clariq_preprocessing_keeps_latent_intent_out_of_inputs(self) -> None:
        content = _clariq_tsv()
        asset = _asset(
            content,
            processor="clariq_tsv_v1",
            expected_counts={"inputs": 2, "gold": 2, "topics": 2},
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dev.tsv"
            source.write_bytes(content)
            prepared = prepare_smoke_asset(asset, source, Path(directory) / "prepared")
            inputs = prepared.inputs_path.read_text(encoding="utf-8")
            gold = prepared.gold_path.read_text(encoding="utf-8")
            self.assertNotIn("Jaguar cars", inputs)
            self.assertNotIn("clarification_need", inputs)
            self.assertIn("Jaguar cars", gold)
            self.assertIn('"clarification_need":4', gold)
            self.assertIn('"reference_question":""', gold)

    def test_source_checksum_and_expected_counts_are_enforced(self) -> None:
        content = _clariq_tsv()
        asset = _asset(
            content,
            processor="clariq_tsv_v1",
            expected_counts={"inputs": 99, "gold": 99, "topics": 2},
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dev.tsv"
            source.write_bytes(content)
            with self.assertRaisesRegex(ArtifactError, "count mismatch"):
                prepare_smoke_asset(asset, source, Path(directory) / "prepared")
            wrong = replace(asset, sha256="0" * 64)
            with self.assertRaisesRegex(ArtifactError, "checksum mismatch"):
                prepare_smoke_asset(wrong, source, Path(directory) / "other")

    def test_unsafe_or_unexpected_zip_member_is_rejected(self) -> None:
        content = _beir_zip(unsafe_member="../escape")
        asset = _asset(
            content,
            processor="beir_zip_v1",
            expected_counts={
                "inputs": 2,
                "gold": 2,
                "corpus": 2,
                "qrels": 2,
                "orphan_qrels": 0,
                "orphan_queries": 0,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zip"
            source.write_bytes(content)
            with self.assertRaisesRegex(ArtifactError, "unsafe path"):
                prepare_smoke_asset(asset, source, Path(directory) / "prepared")


if __name__ == "__main__":
    unittest.main()
