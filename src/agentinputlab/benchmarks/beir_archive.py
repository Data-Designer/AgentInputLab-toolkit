"""Checksum-pinned BEIR archives with streamed corpus access and held-out qrels.

No dataset is bundled. This loader preserves the complete corpus and official
split judgments, including queries whose IDs overlap corpus IDs. CQADupstack
uses explicit forum namespaces and never merges equal IDs from different forums.
"""

from __future__ import annotations

import csv
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from agentinputlab.artifacts import sha256_file
from agentinputlab.backends import Document
from agentinputlab.contracts import ArtifactError, ValidationError

from .base import BenchmarkExample
from .retrieval import PreparedRetrievalBenchmarkAdapter

# Raw source hashes independently match the upstream BEIR download table's MD5.
BEIR_ARCHIVE_LOCKS = {
    "beir.fiqa_2018": {
        "dataset": "fiqa",
        "size_bytes": 17948027,
        "sha256": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
        "upstream_md5": "17918ed23cd04fb15047f73e6c3bd9d9",
    },
    "beir.quora": {
        "dataset": "quora",
        "size_bytes": 15853968,
        "sha256": "56aacd9dcc4d9c093b63f175afdda0e21cbc8442ecf6c717d09de7b358d77531",
        "upstream_md5": "18fb154900ba42a600f84b839c173167",
    },
    "beir.dbpedia_entity": {
        "dataset": "dbpedia-entity",
        "size_bytes": 639285131,
        "sha256": "e4c13080b5399a19e02b04c908c485f8997ac9e265a06ef5f9f28c3f1134c472",
        "upstream_md5": "c2a39eb420a3164af735795df012ac2c",
    },
    "beir.fever": {
        "dataset": "fever",
        "size_bytes": 1236988269,
        "sha256": "58b3faee0c152158b27970380f778ea56e4e34482034404226c10e11eadc7b72",
        "upstream_md5": "5a818580227bfb4b35bb6fa46d9b6c03",
    },
}
_FORUMS = (
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
)


def validate_split(benchmark_id: str, split: str) -> None:
    if not isinstance(split, str) or (
        split != "test" and not (benchmark_id == "beir.fiqa_2018" and split == "dev")
    ):
        raise ValidationError("BEIR archive exposes test; FiQA also supports explicit dev")


class BEIRArchiveAdapter(PreparedRetrievalBenchmarkAdapter):
    """Official held-out cases, native IR measures, and a streaming complete corpus."""

    def __init__(
        self, *, benchmark_id: str, source_archive: str | Path, license_acknowledged: bool = False,
        split: str = "test",
    ):
        if not license_acknowledged:
            raise ValidationError(
                "mounted BEIR dataset requires license_acknowledged=True; data licenses are separate from code"
            )
        if benchmark_id not in BEIR_ARCHIVE_LOCKS:
            raise ValidationError("BEIR archive cell has no verified source lock")
        validate_split(benchmark_id, split)
        self.lock = dict(BEIR_ARCHIVE_LOCKS[benchmark_id])
        self.path = Path(source_archive)
        if not self.path.is_file() or self.path.stat().st_size != self.lock["size_bytes"]:
            raise ArtifactError("BEIR archive size differs")
        sha = sha256_file(self.path)
        if sha != self.lock["sha256"]:
            raise ArtifactError("BEIR archive checksum differs")
        self.benchmark_id = benchmark_id
        self.split = split
        self._ignore_identical_ids = True
        with zipfile.ZipFile(self.path) as archive:
            self.groups = self._validate(archive)
            examples = []
            for group, prefix in self.groups:
                qrels = {}
                with archive.open(prefix + f"qrels/{split}.tsv") as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
                    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                        raise ArtifactError("BEIR qrels schema differs")
                    for row in reader:
                        qid = self._id(group, row["query-id"])
                        did = self._id(group, row["corpus-id"])
                        score = int(row["score"])
                        if score < 0 or did in qrels.setdefault(qid, {}):
                            raise ArtifactError("BEIR invalid or duplicate judgment")
                        qrels[qid][did] = score
                if split == "dev":
                    if not qrels:
                        raise ArtifactError("BEIR dev judgments are empty")
                    for excluded in ("train", "test"):
                        with archive.open(prefix + f"qrels/{excluded}.tsv") as raw:
                            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
                            if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                                raise ArtifactError("BEIR qrels schema differs")
                            if any(self._id(group, row["query-id"]) in qrels for row in reader):
                                raise ArtifactError(f"BEIR dev query overlaps {excluded}")
                queries = {}
                for row in self._rows(archive, prefix + "queries.jsonl"):
                    qid = self._id(group, row["_id"])
                    if qid in queries:
                        raise ArtifactError("BEIR duplicate query ID")
                    if not isinstance(row["text"], str) or not row["text"].strip():
                        raise ArtifactError("BEIR query text is empty")
                    queries[qid] = row["text"]
                if set(qrels) - set(queries):
                    raise ArtifactError("BEIR qrels refer to absent queries")
                for qid in sorted(qrels):
                    if not any(s > 0 for s in qrels[qid].values()):
                        raise ArtifactError("BEIR selected query lacks positive judgment")
                    examples.append(
                        BenchmarkExample(
                            example_id=qid,
                            query=queries[qid],
                            instruction="Retrieve the most relevant "
                            + (
                                "duplicate questions."
                                if self.lock["dataset"] in ("quora", "cqadupstack")
                                else "documents."
                            ),
                            visible_metadata=MappingProxyType(
                                {
                                    "benchmark_cell": benchmark_id,
                                    "source_split": split,
                                    "forum": group,
                                }
                            ),
                            evaluator_gold=MappingProxyType(
                                {"relevance": MappingProxyType(qrels[qid])}
                            ),
                        )
                    )
        self._examples = tuple(examples)
        self._examples_by_id = {e.example_id: e for e in examples}

    def _validate(self, archive):
        infos = archive.infolist()
        names = set()
        total = 0
        if len(infos) > 256:
            raise ArtifactError("BEIR archive has too many members")
        for info in infos:
            path = PurePosixPath(info.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or info.filename in names
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ArtifactError("BEIR archive contains unsafe/duplicate members")
            names.add(info.filename)
            total += info.file_size
            if info.file_size > 12 * 1024**3 or info.file_size / max(info.compress_size, 1) > 400:
                raise ArtifactError("BEIR archive member is oversized")
        if total > 24 * 1024**3:
            raise ArtifactError("BEIR archive is oversized")
        dataset = self.lock["dataset"]
        if dataset == "cqadupstack":
            groups = tuple((f, "cqadupstack/" + f + "/") for f in _FORUMS)
        else:
            groups = (("", dataset + "/"),)
        for _, prefix in groups:
            required = {prefix + "corpus.jsonl", prefix + "queries.jsonl", prefix + "qrels/test.tsv"}
            if self.split == "dev":
                required.update(prefix + f"qrels/{split}.tsv" for split in ("dev", "train"))
            if not required <= names:
                raise ArtifactError("BEIR required corpus, query, or judgment members missing")
        return groups

    @staticmethod
    def _id(group, value):
        if not isinstance(value, str) or not value:
            raise ArtifactError("BEIR source ID must be nonempty text")
        return group + "::" + value if group else value

    @staticmethod
    def _rows(archive, name):
        with archive.open(name) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ArtifactError("BEIR JSONL record must be object")
                yield row

    def iter_documents(self, *, forum: str | None = None):
        """Yield all source documents, without loading full corpora into RAM."""
        if forum is not None and forum not in {g for g, _ in self.groups}:
            raise ValidationError("unknown BEIR forum")
        with zipfile.ZipFile(self.path) as archive:
            for group, prefix in self.groups:
                if forum is not None and group != forum:
                    continue
                for row in self._rows(archive, prefix + "corpus.jsonl"):
                    text = row["text"]
                    title = row.get("title", "")
                    if not isinstance(text, str) or not isinstance(title, str):
                        raise ArtifactError("BEIR corpus title/text must be strings")
                    yield Document(
                        doc_id=self._id(group, row["_id"]),
                        text=text,
                        title=title,
                        corpus_revision=self.lock["sha256"],
                        metadata={"forum": group},
                    )

    def fingerprint(self):
        return {
            "benchmark_id": self.benchmark_id,
            "source_archive_sha256": self.lock["sha256"],
            "source_size_bytes": self.lock["size_bytes"],
            "upstream_md5": self.lock["upstream_md5"],
            "source_url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
            + self.lock["dataset"]
            + ".zip",
            "split": self.split,
            f"{self.split}_queries": len(self._examples),
            "ignore_identical_ids": True,
            "corpus_scope": "complete_source",
            "scorer": "ir_measures:nDCG@10,R@100",
            "cqadupstack_query_corpus": "same_forum_only",
            "aggregate_leaderboard_claimed": False,
            "licenses_bundled": False,
        }


__all__ = ["BEIR_ARCHIVE_LOCKS", "BEIRArchiveAdapter"]
