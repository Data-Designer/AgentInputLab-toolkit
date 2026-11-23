from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentinputlab.benchmarks import BEIRSampleAdapter
from agentinputlab.contracts import ArtifactError, ValidationError


def write_bundle(root, *, cqa=False, edit=None):
    groups = ["android", "english"] if cqa else [""]
    inputs, gold, corpus = [], [], []
    for group in groups:
        ns = (lambda value, prefix=group: f"{prefix}::{value}") if group else (lambda value: value)
        inputs.append(
            {
                "id": ns("q1"),
                "query": "Visible query",
                "instruction": "Retrieve.",
                "visible_metadata": {
                    "forum": group,
                    "benchmark_cell": "beir.cqadupstack" if cqa else "beir.fever",
                    "source_split": "test",
                },
            }
        )
        gold.append({"id": ns("q1"), "relevance": {ns("d1"): 1}})
        corpus.extend(
            [
                {"id": ns(did), "title": "", "text": "visible " + did, "forum": group}
                for did in ("q1", "d1", "d2")
            ]
        )
    files = {}
    for name, rows in [("inputs.jsonl", inputs), ("gold.jsonl", gold), ("corpus.jsonl", corpus)]:
        if edit:
            edit(name, rows)
        raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        (root / name).write_bytes(raw)
        files[name] = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (root / "resources").mkdir()
    (root / "resources/source.raw").write_bytes(b"fixture")
    repo = "BeIR/cqadupstack" if cqa else "BeIR/fever"
    manifest = {
        "schema_version": "agentinputlab.beir.sample.v1",
        "benchmark_id": "beir.cqadupstack" if cqa else "beir.fever",
        "split": "test",
        "license_id": "CC-BY-SA-4.0",
        "groups": groups,
        "source_revisions": {repo: "a" * 40, repo + "-qrels": "b" * 40},
        "corpus_scope": "fixed_gold_enriched_compatibility_subset",
        "full_dataset_score": False,
        "full_benchmark_corpus_download": False,
        "files": files,
        "counts": {"inputs": len(inputs), "gold": len(gold), "corpus": len(corpus)},
        "resources": [
            {
                "path": "resources/source.raw",
                "url": "https://example.com/fixture",
                "sha256": hashlib.sha256(b"fixture").hexdigest(),
                "size_bytes": 7,
                "source_revision": "a" * 40,
            }
        ],
    }
    return write_manifest(root, manifest), manifest


def write_manifest(root, manifest):
    raw = json.dumps(manifest, sort_keys=True).encode()
    (root / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class BEIRSampleTests(unittest.TestCase):
    def test_source_locks_and_gold_separation_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root)
            with patch("socket.socket.connect", side_effect=AssertionError("must be offline")):
                adapter = BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
                state = adapter.examples("test")[0].visible_state()
            self.assertNotIn("relevance", repr(state))
            self.assertEqual(len(list(adapter.iter_documents())), 3)
            fp = adapter.fingerprint()
            self.assertFalse(fp["full_dataset_score"])
            self.assertEqual(fp["metrics_scope"], "sampled_candidate_pool_only")
            self.assertEqual(fp["sample_manifest_sha256"], sha)

    def test_forums_with_same_raw_ids_remain_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root, cqa=True)
            adapter = BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
            self.assertEqual(len(adapter.examples("test")), 2)
            self.assertEqual(
                {d.doc_id for d in adapter.iter_documents(forum="android")},
                {"android::q1", "android::d1", "android::d2"},
            )
            example = adapter.example("test", "android::q1")
            with self.assertRaisesRegex(ValidationError, "same-forum"):
                adapter.score(example, ["english::d1"])

    def test_native_metrics_remove_self_id_not_gold_and_do_not_invent_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root)
            adapter = BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)
            example = adapter.examples("test")[0]
            with patch(
                "agentinputlab.benchmarks.retrieval.retrieval_aggregates",
                return_value={"ndcg_at_10": 1},
            ) as native:
                self.assertEqual(adapter.score(example, ["q1", "d1"]), {"ndcg_at_10": 1})
            qrels, run = native.call_args.args
            self.assertEqual(qrels["q1"], {"d1": 1})
            self.assertEqual(set(run["q1"]), {"d1"})

    def test_missing_positive_document_or_cross_forum_qrel_refused(self):
        def edit(name, rows):
            if name == "gold.jsonl":
                rows[0]["relevance"] = {"absent": 1}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root, edit=edit)
            with self.assertRaisesRegex(ArtifactError, "qrel document"):
                BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)

    def test_corrupt_resources_and_normalized_files_are_rejected(self):
        for filename in ("inputs.jsonl", "resources/source.raw"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sha, _ = write_bundle(root)
                (root / filename).write_bytes(b"changed")
                with self.assertRaisesRegex(ArtifactError, "checksum/size"):
                    BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)

    def test_full_corpus_claim_or_ambiguous_revision_is_rejected(self):
        for change in ({"full_dataset_score": True}, {"source_revisions": {"BeIR/fever": "main"}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest = write_bundle(root)
                sha = write_manifest(root, manifest | change)
                with self.assertRaises(ArtifactError):
                    BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)

    def test_license_pin_unsafe_paths_and_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, manifest = write_bundle(root)
            with self.assertRaisesRegex(ValidationError, "terms"):
                BEIRSampleAdapter(root, manifest_sha256=sha)
            with self.assertRaisesRegex(ArtifactError, "manifest checksum"):
                BEIRSampleAdapter(root, manifest_sha256="f" * 64, license_acknowledged=True)
            manifest["resources"][0]["path"] = "resources/../../outside"
            sha = write_manifest(root, manifest)
            with self.assertRaisesRegex(ArtifactError, "unsafe"):
                BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)

    def test_symlink_resource_is_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha, _ = write_bundle(root)
            target = root / "resources/source.raw"
            target.rename(root / "separate.raw")
            target.symlink_to(root / "separate.raw")
            with self.assertRaisesRegex(ArtifactError, "symlink"):
                BEIRSampleAdapter(root, manifest_sha256=sha, license_acknowledged=True)


if __name__ == "__main__":
    unittest.main()
