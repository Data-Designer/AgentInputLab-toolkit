"""Pinned, small BEIR candidate pools for compatibility, not full-corpus scores.

The operator supplies a source-bound bundle. This adapter never accesses the
network, guesses URLs, downloads a corpus, or treats a sample as a leaderboard
run. Reference judgments are evaluator-only; candidate enrichment is explicit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from agentinputlab.backends import Document
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.metrics import evaluation_dependency_versions

from .preprocess import PreparedBenchmark
from .retrieval import PreparedRetrievalBenchmarkAdapter, RetrievalPrediction, _records_by_id

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_SCOPE = "fixed_gold_enriched_compatibility_subset"
_FORUMS = {
    "android",
    "english",
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress",
}
_MAX_BUNDLE_BYTES = 16 * 1024**2


class BEIRSampleAdapter(PreparedRetrievalBenchmarkAdapter):
    """Load a byte-locked source-side sample and use native IR metric providers.

    Positive qrels plus fixed background rows form the candidate pool. That
    enrichment tests the protocol and implementation, not retrieval quality on
    the complete corpus. CQADupstack IDs are namespaced per forum and retrieval
    must remain within that forum. No upstream data is distributed by this API.
    """

    def __init__(self, root: str | Path, *, manifest_sha256: str, license_acknowledged=False):
        if not license_acknowledged:
            raise ValidationError(
                "review upstream sample data terms and set license_acknowledged=True"
            )
        if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
            raise ValidationError("sample requires a pinned manifest SHA-256")
        supplied = Path(root).expanduser()
        if supplied.is_symlink():
            raise ArtifactError("sample root must not be a symlink")
        self.root = supplied.resolve()
        self._read_bytes = 0
        raw = self._read_file("manifest.json")
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ArtifactError("sample manifest checksum differs")
        manifest = json.loads(raw)
        if manifest.get("schema_version") != "agentinputlab.beir.sample.v1":
            raise ArtifactError("unknown BEIR sample schema")
        benchmark = manifest.get("benchmark_id")
        if benchmark not in {"beir.fever", "beir.cqadupstack"}:
            raise ArtifactError("sample benchmark has no reviewed source profile")
        if (
            manifest.get("split") != "test"
            or manifest.get("corpus_scope") != _SCOPE
            or manifest.get("full_dataset_score") is not False
            or manifest.get("full_benchmark_corpus_download") is not False
        ):
            raise ArtifactError("sample must explicitly exclude full-corpus evaluation")
        expected_repos = (
            {"BeIR/fever", "BeIR/fever-qrels"}
            if benchmark == "beir.fever"
            else {"BeIR/cqadupstack", "BeIR/cqadupstack-qrels"}
        )
        revisions = manifest.get("source_revisions")
        if (
            not isinstance(revisions, dict)
            or set(revisions) != expected_repos
            or any(
                not isinstance(value, str) or not _REVISION.fullmatch(value)
                for value in revisions.values()
            )
        ):
            raise ArtifactError("sample source repository/revision identities differ")
        groups = manifest.get("groups")
        if (
            not isinstance(groups, list)
            or not groups
            or any(not isinstance(group, str) for group in groups)
            or len(set(groups)) != len(groups)
            or (groups != [""] if benchmark == "beir.fever" else not set(groups) <= _FORUMS)
        ):
            raise ArtifactError("sample forum identities differ")
        if manifest.get("license_id") != "CC-BY-SA-4.0":
            raise ArtifactError("sample must retain its declared upstream data license")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {
            "inputs.jsonl",
            "gold.jsonl",
            "corpus.jsonl",
        }:
            raise ArtifactError("sample needs exact input/gold/corpus files")
        for name, lock in files.items():
            self._verified_file(name, lock)
        resources = manifest.get("resources")
        if not isinstance(resources, list) or not resources or len(resources) > 128:
            raise ArtifactError("sample source evidence resources are required and bounded")
        seen = set()
        for resource in resources:
            if not isinstance(resource, dict):
                raise ArtifactError("sample source resource must be an object")
            name = resource.get("path")
            if not isinstance(name, str) or not name.startswith("resources/") or name in seen:
                raise ArtifactError("sample source evidence paths differ")
            seen.add(name)
            self._verified_file(name, resource)
            if resource.get("source_revision") not in revisions.values():
                raise ArtifactError("sample source evidence revision is undeclared")
            if not isinstance(resource.get("url"), str) or not resource["url"].startswith(
                "https://"
            ):
                raise ArtifactError("sample source evidence needs its HTTPS origin")
        corpus = _records_by_id(self.root / "corpus.jsonl", label="sample corpus")
        for did, row in corpus.items():
            group = row.get("forum")
            if group not in groups or (group and not did.startswith(group + "::")):
                raise ArtifactError("sample document forum/ID namespace differs")
            if (
                not isinstance(row.get("title"), str)
                or not isinstance(row.get("text"), str)
                or not row["text"].strip()
            ):
                raise ArtifactError("sample document title/text is invalid")
        counts = manifest.get("counts")
        if (
            not isinstance(counts, dict)
            or set(counts) != {"inputs", "gold", "corpus"}
            or any(type(value) is not int or value < 1 for value in counts.values())
        ):
            raise ArtifactError("sample counts are missing")
        prepared = PreparedBenchmark(
            asset_id=benchmark + ".compatibility-sample",
            processor="beir_source_sample_v1",
            processor_revision="1",
            source_sha256=manifest_sha256,
            root=self.root,
            counts=counts,
            file_sha256={name: lock["sha256"] for name, lock in files.items()},
            cell_id=benchmark,
            source_revision=manifest_sha256,
            split="test",
        )
        super().__init__(prepared, benchmark_id=benchmark, split="test")
        self._ignore_identical_ids = True
        if counts != {
            "inputs": len(self._examples),
            "gold": len(self._examples),
            "corpus": len(corpus),
        }:
            raise ArtifactError("sample counts differ from locked files")
        observed = set()
        for example in self._examples:
            group = example.visible_metadata.get("forum")
            if group not in groups or (group and not example.example_id.startswith(group + "::")):
                raise ArtifactError("sample query forum/ID namespace differs")
            observed.add(group)
            for did in example.evaluator_gold["relevance"]:
                if did not in corpus or corpus[did]["forum"] != group:
                    raise ArtifactError("sample qrel document missing or from another forum")
        if observed != set(groups):
            raise ArtifactError("sample claims a forum without query evidence")
        self.manifest_sha256, self.manifest = manifest_sha256, manifest
        self.groups = tuple(groups)
        self._corpus = corpus

    def _read_file(self, name):
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name:
            raise ArtifactError("sample file path is unsafe")
        path = self.root / name
        current = self.root
        for part in parts.parts:
            current = current / part
            if current.is_symlink():
                raise ArtifactError("sample files must not use symlinks")
        if not path.is_file() or path.stat().st_size + self._read_bytes > _MAX_BUNDLE_BYTES:
            raise ArtifactError("sample file absent or bundle exceeds 16 MiB")
        with path.open("rb") as handle:
            raw = handle.read(_MAX_BUNDLE_BYTES - self._read_bytes + 1)
        if len(raw) + self._read_bytes > _MAX_BUNDLE_BYTES:
            raise ArtifactError("sample bundle exceeds 16 MiB")
        self._read_bytes += len(raw)
        return raw

    def _verified_file(self, name, lock):
        if not isinstance(lock, dict):
            raise ArtifactError("sample file lock must be an object")
        raw = self._read_file(name)
        if len(raw) != lock.get("size_bytes") or hashlib.sha256(raw).hexdigest() != lock.get(
            "sha256"
        ):
            raise ArtifactError("sample file checksum/size differs")

    def iter_documents(self, *, forum=None):
        if forum is not None and forum not in self.groups:
            raise ValidationError("unknown sample forum")
        for did in sorted(self._corpus):
            row = self._corpus[did]
            if forum is None or forum == row["forum"]:
                yield Document(
                    doc_id=did,
                    title=row["title"],
                    text=row["text"],
                    corpus_revision=self.manifest_sha256,
                    metadata={"forum": row["forum"]},
                )

    def score(self, example, prediction):
        if isinstance(prediction, RetrievalPrediction):
            normalized = prediction
        elif isinstance(prediction, Mapping):
            normalized = RetrievalPrediction(prediction)
        else:
            normalized = RetrievalPrediction.from_ranked_ids(prediction)
        group = example.visible_metadata.get("forum")
        if any(
            did not in self._corpus or self._corpus[did]["forum"] != group
            for did in normalized.scores
        ):
            raise ValidationError(
                "sample prediction must contain only same-forum candidate documents"
            )
        return super().score(example, normalized)

    def fingerprint(self):
        return {
            "benchmark_id": self.benchmark_id,
            "sample_manifest_sha256": self.manifest_sha256,
            "source_revisions": dict(self.manifest["source_revisions"]),
            "groups": list(self.groups),
            "split": "test",
            "test_queries": len(self._examples),
            "corpus_documents": len(self._corpus),
            "corpus_scope": _SCOPE,
            "metrics_scope": "sampled_candidate_pool_only",
            "query_selection": self.manifest.get("query_selection"),
            "background_selection": self.manifest.get("background_selection"),
            "full_dataset_score": False,
            "full_benchmark_corpus_download": False,
            "downloaded_source_scope": self.manifest.get(
                "downloaded_source_scope", "source_side_rows"
            ),
            "scorer": "ir_measures:nDCG@10,R@100",
            "ignore_identical_ids": True,
            "evaluation_dependencies": dict(evaluation_dependency_versions()),
        }


__all__ = ["BEIRSampleAdapter"]
