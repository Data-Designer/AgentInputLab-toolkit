from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentinputlab.benchmarks import TRECDLSampleAdapter
from agentinputlab.contracts import ArtifactError, ValidationError


def write_manifest(root, manifest):
    raw = json.dumps(manifest, sort_keys=True).encode()
    (root / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def write_bundle(root, *, year=2019, edit=None):
    corpus = [{"id": str(i), "text": "passage " + str(i), "title": "", "forum": ""}
              for i in range(4)]
    inputs = [{"id": "0", "query": "Native question", "instruction": "Retrieve passages.",
               "visible_metadata": {"benchmark_cell": f"trec_dl.{year}", "source_split": "test",
                                    "forum": ""}}]
    gold = [{"id": "0", "relevance": {str(i): 3 - i for i in range(4)}}]
    files = {}
    for name, rows in [("inputs.jsonl", inputs), ("gold.jsonl", gold), ("corpus.jsonl", corpus)]:
        # Snapshot native rows before tests alter the normalized copy.
        values = json.loads(json.dumps(rows))
        if edit:
            edit(name, values)
        raw = b"".join((json.dumps(row) + "\n").encode() for row in values)
        (root / name).write_bytes(raw)
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    (root / "resources").mkdir()
    sources = [
        ("queries_gzip", gzip.compress(b"0\tNative question\n"),
         "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
         f"msmarco-test{year}-queries.tsv.gz"),
        ("qrels", b"".join(f"0 {'0' if year == 2020 else 'Q0'} {i} {3-i}\n".encode()
                           for i in range(4)),
         f"https://trec.nist.gov/data/deep/{year}qrels-pass.txt"),
        ("corpus_rows", json.dumps({"rows": [
            {"row": {"_id": row["id"], "title": "", "text": row["text"]},
             "truncated_cells": []} for row in corpus
        ]}).encode(), "https://datasets-server.huggingface.co/rows?"
         "dataset=BeIR%2Fmsmarco&config=corpus&split=corpus&offset=0&length=4"),
    ]
    resources = []
    for index, (role, raw, url) in enumerate(sources):
        name, sha = f"resources/{index}.raw", hashlib.sha256(raw).hexdigest()
        (root / name).write_bytes(raw)
        resources.append({"path": name, "role": role, "url": url, "size_bytes": len(raw),
                          "sha256": sha, "source_revision": "a" * 40 if role == "corpus_rows"
                          else sha})
    manifest = {
        "schema_version": "agentinputlab.trec_dl.sample.v1", "benchmark_id": f"trec_dl.{year}",
        "split": "test", "corpus_scope": "fixed_gold_enriched_compatibility_subset",
        "full_dataset_score": False, "full_benchmark_corpus_download": False,
        "binary_relevance_threshold": 2, "corpus_repository": "BeIR/msmarco",
        "corpus_revision": "a" * 40,
        "license_ids": {"microsoft/msmarco": "CC-BY-4.0", "BeIR/msmarco": "CC-BY-SA-4.0"},
        "files": files, "counts": {"inputs": 1, "gold": 1, "corpus": 4}, "resources": resources,
    }
    return write_manifest(root, manifest), manifest


class TRECDLSampleTests(unittest.TestCase):
    def test_both_years_are_offline_sample_only_and_gold_separated(self):
        for year in (2019, 2020):
            with self.subTest(year=year), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sha, _ = write_bundle(root, year=year)
                with patch("socket.socket.connect", side_effect=AssertionError("offline")):
                    a = TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
                    self.assertNotIn("relevance", repr(a.examples("test")[0].visible_state()))
                self.assertEqual(len(list(a.iter_documents())), 4)
                self.assertFalse(a.fingerprint()["full_dataset_score"])
                self.assertEqual(a.fingerprint()["binary_relevance_threshold"], 2)
                self.assertFalse(a.fingerprint()["ignore_identical_ids"])

    def test_grade1_is_not_binary_relevant_but_is_preserved_in_graded_ndcg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root)
            a = TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
            with patch("agentinputlab.benchmarks.trec_dl_sample.retrieval_aggregates",
                       side_effect=[{"ndcg": 0.8}, {"map": 0.5, "recall": 1.0}]) as native:
                self.assertEqual(a.score(a.examples("test")[0], ["2", "1", "0", "3"]),
                                 {"ndcg": 0.8, "map": 0.5, "recall": 1.0})
            self.assertEqual(native.call_args_list[0].args[0]["0"]["2"], 1)
            self.assertEqual(native.call_args_list[1].args[0]["0"],
                             {"0": 1, "1": 1, "2": 0, "3": 0})
            # Query and passage IDs occupy different namespaces; do not remove equal IDs.
            self.assertIn("0", native.call_args_list[0].args[1]["0"])

    @unittest.skipUnless(importlib.util.find_spec("ir_measures"), "evaluation extra unavailable")
    def test_actual_native_metric_provider_threshold_and_empty_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root)
            a = TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
            e = a.examples("test")[0]
            measured = a.score(e, ["2", "1", "0", "3"])
            self.assertAlmostEqual(measured["map"], 7 / 12)
            self.assertEqual(measured["recall"], 1)
            self.assertGreater(measured["ndcg"], 0)
            self.assertLess(measured["ndcg"], 1)
            self.assertEqual(a.score(e, {}), {"ndcg": 0, "map": 0, "recall": 0})

    def test_normalized_source_mismatch_is_rejected_even_with_fresh_checksum(self):
        for target in ("inputs.jsonl", "gold.jsonl", "corpus.jsonl"):
            def edit(name, rows, target=target):
                if name == target:
                    if name == "inputs.jsonl":
                        rows[0]["query"] = "changed query"
                    elif name == "gold.jsonl":
                        rows[0]["relevance"]["2"] = 2
                    else:
                        rows[0]["text"] = "changed passage"

            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sha, _ = write_bundle(root, edit=edit)
                with self.assertRaisesRegex(ArtifactError, "differ"):
                    TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)

    def test_scope_revision_threshold_and_source_url_refused(self):
        changes = [{"full_dataset_score": True}, {"binary_relevance_threshold": 1},
                   {"binary_relevance_threshold": True}, {"corpus_revision": "main"}]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest = write_bundle(root)
                sha = write_manifest(root, manifest | change)
                with self.assertRaises(ArtifactError):
                    TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = write_bundle(root)
            manifest["resources"][0]["url"] = "https://example.com/queries.gz"
            sha = write_manifest(root, manifest)
            with self.assertRaisesRegex(ArtifactError, "source identity"):
                TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)

    def test_unused_qrels_column_is_not_a_relevance_or_identity_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = write_bundle(root, year=2020)
            qrel = next(r for r in manifest["resources"] if r["role"] == "qrels")
            raw = (root / qrel["path"]).read_bytes()
            self.assertIn(b"0 0 0 3", raw)
            qrel.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                        source_revision=hashlib.sha256(raw).hexdigest())
            sha = write_manifest(root, manifest)
            a = TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
            self.assertEqual(dict(a.examples("test")[0].evaluator_gold["relevance"]),
                             {"0": 3, "1": 2, "2": 1, "3": 0})
            changed = raw.replace(b"0 0 0 3", b"0 unknown 0 3")
            (root / qrel["path"]).write_bytes(changed)
            qrel.update(size_bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest(),
                        source_revision=hashlib.sha256(changed).hexdigest())
            sha = write_manifest(root, manifest)
            with self.assertRaisesRegex(ArtifactError, "qrel duplicate/label"):
                TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)

    def test_license_checksum_symlink_and_unknown_prediction_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root)
            with self.assertRaisesRegex(ValidationError, "terms"):
                TRECDLSampleAdapter(root, manifest_sha256=sha)
            with self.assertRaisesRegex(ArtifactError, "manifest checksum"):
                TRECDLSampleAdapter(root, manifest_sha256="f" * 64, license_acknowledged=True)
            a = TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
            with self.assertRaisesRegex(ValidationError, "outside"):
                a.score(a.examples("test")[0], ["absent"])
            path = root / "resources/0.raw"
            path.rename(root / "separate.raw")
            path.symlink_to(root / "separate.raw")
            with self.assertRaisesRegex(ArtifactError, "symlink"):
                TRECDLSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)


if __name__ == "__main__":
    unittest.main()
