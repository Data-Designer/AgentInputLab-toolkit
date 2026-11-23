"""Bounded TREC DL passage samples with graded and threshold-2 native metrics.

The caller acquires the source-side bundle separately. Gold-enriched candidate
pools exercise compatibility; their scores are not full-corpus TREC results.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit

from agentinputlab.backends import Document
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.metrics import evaluation_dependency_versions, retrieval_aggregates

from .preprocess import PreparedBenchmark
from .retrieval import PreparedRetrievalBenchmarkAdapter, RetrievalPrediction, _records_by_id

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_SCOPE = "fixed_gold_enriched_compatibility_subset"
_MAX_BUNDLE_BYTES = 16 * 1024**2
_MAX_COMPONENT_BYTES = 256 * 1024


class TRECDLSampleAdapter(PreparedRetrievalBenchmarkAdapter):
    """Read a byte-locked, explicitly sampled 2019/2020 passage task.

    Native source query text, passage text/IDs and every retained qrel are
    reconciled before exposing examples. No network or full-corpus index is
    created. Binary measures treat grade 1 (related, not relevant) as zero.
    """

    def __init__(self, root: str | Path, *, manifest_sha256: str, license_acknowledged=False):
        if not license_acknowledged:
            raise ValidationError("review upstream data terms and set license_acknowledged=True")
        if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
            raise ValidationError("sample requires a pinned manifest SHA-256")
        supplied = Path(root).expanduser()
        if supplied.is_symlink():
            raise ArtifactError("sample root must not be a symlink")
        self.root, self._read_bytes = supplied.resolve(), 0
        raw = self._read_file("manifest.json")
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ArtifactError("sample manifest checksum differs")
        manifest = json.loads(raw)
        benchmark = manifest.get("benchmark_id")
        if (
            manifest.get("schema_version") != "agentinputlab.trec_dl.sample.v1"
            or benchmark not in {"trec_dl.2019", "trec_dl.2020"}
            or manifest.get("split") != "test"
            or manifest.get("corpus_scope") != _SCOPE
            or manifest.get("full_dataset_score") is not False
            or manifest.get("full_benchmark_corpus_download") is not False
            or type(manifest.get("binary_relevance_threshold")) is not int
            or manifest["binary_relevance_threshold"] != 2
        ):
            raise ArtifactError("TREC sample scope/threshold differs")
        revision = manifest.get("corpus_revision")
        if (
            manifest.get("corpus_repository") != "BeIR/msmarco"
            or not isinstance(revision, str)
            or not _REVISION.fullmatch(revision)
            or manifest.get("license_ids")
            != {"microsoft/msmarco": "CC-BY-4.0", "BeIR/msmarco": "CC-BY-SA-4.0"}
        ):
            raise ArtifactError("TREC source revision/terms identity differs")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {
            "inputs.jsonl", "gold.jsonl", "corpus.jsonl"
        }:
            raise ArtifactError("sample requires exact input/gold/corpus files")
        for name, lock in files.items():
            self._verified_file(name, lock)
        resources = manifest.get("resources")
        if not isinstance(resources, list) or not 3 <= len(resources) <= 128:
            raise ArtifactError("bounded source evidence resources are required")
        sources, seen, documents = {}, set(), {}
        year = benchmark.rsplit(".", 1)[1]
        urls = {
            "qrels": f"https://trec.nist.gov/data/deep/{year}qrels-pass.txt",
            "queries_gzip": "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
            f"msmarco-test{year}-queries.tsv.gz",
        }
        for resource in resources:
            if not isinstance(resource, dict):
                raise ArtifactError("source evidence must be an object")
            name = resource.get("path")
            if not isinstance(name, str) or not name.startswith("resources/") or name in seen:
                raise ArtifactError("source evidence path differs")
            seen.add(name)
            body = self._verified_file(name, resource)
            if len(body) > _MAX_COMPONENT_BYTES:
                raise ArtifactError("source component exceeds bounded size")
            role = resource.get("role")
            if role in urls:
                if role in sources or resource.get("url") != urls[role]:
                    raise ArtifactError("official query/qrel source identity differs")
                if resource.get("source_revision") != hashlib.sha256(body).hexdigest():
                    raise ArtifactError("official component content revision differs")
                sources[role] = body
            elif role == "corpus_rows":
                if resource.get("source_revision") != revision:
                    raise ArtifactError("corpus row revision differs")
                parts = urlsplit(resource.get("url", ""))
                params = parse_qs(parts.query)
                if (
                    parts.scheme != "https"
                    or parts.netloc != "datasets-server.huggingface.co"
                    or parts.path != "/rows"
                    or parts.fragment
                    or set(params) != {"dataset", "config", "split", "offset", "length"}
                    or params["dataset"] != ["BeIR/msmarco"]
                    or params["config"] != ["corpus"]
                    or params["split"] != ["corpus"]
                    or len(params["offset"]) != 1
                    or len(params["length"]) != 1
                ):
                    raise ArtifactError("corpus source-side row request differs")
                offset, length = int(params["offset"][0]), int(params["length"][0])
                rows = json.loads(body)["rows"]
                if offset < 0 or not 1 <= length <= 100 or len(rows) != length:
                    raise ArtifactError("corpus source-side row bounds differ")
                for index, entry in enumerate(rows):
                    row = entry["row"]
                    did = str(row["_id"])
                    if entry["truncated_cells"] or did != str(offset + index):
                        raise ArtifactError("native passage ID missing/truncated/different")
                    value = {"id": did, "title": row.get("title", ""),
                             "text": row["text"], "forum": ""}
                    if did in documents and documents[did] != value:
                        raise ArtifactError("conflicting source passage text")
                    documents[did] = value
            else:
                raise ArtifactError("unknown source evidence role")
        if set(sources) != set(urls) or not documents:
            raise ArtifactError("official query/qrel/corpus evidence missing")
        with gzip.GzipFile(fileobj=io.BytesIO(sources["queries_gzip"])) as stream:
            query_raw = stream.read(_MAX_COMPONENT_BYTES + 1)
        if len(query_raw) > _MAX_COMPONENT_BYTES:
            raise ArtifactError("decoded query component exceeds bound")
        queries = {}
        for line in query_raw.decode().splitlines():
            qid, text = line.split("\t", 1)
            if qid in queries:
                raise ArtifactError("duplicate official query")
            queries[qid] = text
        qrels = {}
        for line in sources["qrels"].decode().splitlines():
            qid, unused, did, label = line.split()
            mapping = qrels.setdefault(qid, {})
            # NIST2019 uses Q0; NIST2020 uses 0 in this unused qrels column.
            # It is not a relevance label, query ID or passage ID.
            if unused not in {"0", "Q0"} or did in mapping or int(label) not in range(4):
                raise ArtifactError("official qrel duplicate/label differs")
            mapping[did] = int(label)
        corpus = _records_by_id(self.root / "corpus.jsonl", label="TREC sample corpus")
        if corpus != documents or any(
            not isinstance(row["text"], str) or not row["text"].strip()
            or not isinstance(row["title"], str) for row in corpus.values()
        ):
            raise ArtifactError("sample corpus differs from source-native passages")
        counts = manifest.get("counts")
        if (
            not isinstance(counts, dict) or set(counts) != {"inputs", "gold", "corpus"}
            or any(type(v) is not int or v < 1 for v in counts.values())
        ):
            raise ArtifactError("sample counts are invalid")
        prepared = PreparedBenchmark(
            asset_id=benchmark + ".compatibility-sample", processor="trec_dl_source_sample_v1",
            processor_revision="1", source_sha256=manifest_sha256, root=self.root,
            counts=counts, file_sha256={name: lock["sha256"] for name, lock in files.items()},
            cell_id=benchmark, source_revision=manifest_sha256, split="test",
        )
        super().__init__(prepared, benchmark_id=benchmark, split="test")
        if counts != {"inputs": len(self._examples), "gold": len(self._examples),
                      "corpus": len(corpus)}:
            raise ArtifactError("sample counts differ from locked files")
        for example in self._examples:
            labels = {d: v for d, v in qrels.get(example.example_id, {}).items() if d in corpus}
            if (
                example.query != queries.get(example.example_id)
                or dict(example.evaluator_gold["relevance"]) != labels
                or not any(label >= 2 for label in labels.values())
            ):
                raise ArtifactError("query text or retained source qrels differ")
        self.manifest, self.manifest_sha256, self._corpus = manifest, manifest_sha256, corpus

    def _read_file(self, name):
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name:
            raise ArtifactError("sample file path is unsafe")
        current = self.root
        for part in parts.parts:
            current /= part
            if current.is_symlink():
                raise ArtifactError("sample files must not use symlinks")
        if not current.is_file() or current.stat().st_size + self._read_bytes > _MAX_BUNDLE_BYTES:
            raise ArtifactError("sample file absent or bundle exceeds 16 MiB")
        with current.open("rb") as stream:
            raw = stream.read(_MAX_BUNDLE_BYTES - self._read_bytes + 1)
        self._read_bytes += len(raw)
        if self._read_bytes > _MAX_BUNDLE_BYTES:
            raise ArtifactError("sample bundle exceeds 16 MiB")
        return raw

    def _verified_file(self, name, lock):
        if not isinstance(lock, dict) or type(lock.get("size_bytes")) is not int:
            raise ArtifactError("sample file byte lock is missing")
        raw = self._read_file(name)
        if lock["size_bytes"] != len(raw) or lock.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise ArtifactError("sample file checksum/size differs")
        return raw

    def iter_documents(self, *, forum=None):
        if forum not in (None, ""):
            raise ValidationError("TREC passage sample has no forum groups")
        for did, row in self._corpus.items():
            yield Document(doc_id=did, title=row["title"], text=row["text"],
                           corpus_revision=self.manifest_sha256)

    def score(self, example, prediction):
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this TREC adapter")
        if isinstance(prediction, RetrievalPrediction):
            normalized = prediction
        elif isinstance(prediction, Mapping):
            normalized = RetrievalPrediction(prediction)
        else:
            normalized = RetrievalPrediction.from_ranked_ids(prediction)
        if not set(normalized.scores) <= self._corpus.keys():
            raise ValidationError("prediction contains documents outside the sample")
        qid, labels = example.example_id, example.evaluator_gold["relevance"]
        run = {qid: normalized.scores}
        graded = retrieval_aggregates({qid: labels}, run, measures={"ndcg": "nDCG@10"})
        binary = {d: int(label >= 2) for d, label in labels.items()}
        return {**graded, **retrieval_aggregates(
            {qid: binary}, run, measures={"map": "AP", "recall": "R@100"}
        )}

    def fingerprint(self):
        return {
            "benchmark_id": self.benchmark_id, "sample_manifest_sha256": self.manifest_sha256,
            "corpus_repository": "BeIR/msmarco", "corpus_revision": self.manifest["corpus_revision"],
            "split": "test", "test_queries": len(self._examples),
            "corpus_documents": len(self._corpus), "corpus_scope": _SCOPE,
            "metrics_scope": "sampled_candidate_pool_only", "full_dataset_score": False,
            "full_benchmark_corpus_download": False, "binary_relevance_threshold": 2,
            "query_selection": self.manifest.get("query_selection"),
            "candidate_selection": self.manifest.get("candidate_selection"),
            "scorer": "ir_measures:nDCG@10,AP(rel>=2),R(rel>=2)@100",
            "ignore_identical_ids": False,
            "evaluation_dependencies": dict(evaluation_dependency_versions()),
        }


__all__ = ["TRECDLSampleAdapter"]
