"""Deterministic, gold-separated preprocessing for audited smoke assets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from agentinputlab.artifacts import sha256_file
from agentinputlab.contracts.errors import ArtifactError
from agentinputlab.contracts.serialization import canonical_json
from agentinputlab.data.storage import (
    DEFAULT_PREPARED_LIMIT,
    positive_byte_limit,
    require_storage_capacity,
)

from .assets import BenchmarkSmokeAsset

_PROCESSOR_REVISIONS = {
    "beir_zip_v1": "1",
    "bfcl_v3_simple_tar_v1": "1",
    "bright_corpus_parquet_v1": "1",
    "bright_examples_parquet_v1": "1",
    "clariq_tsv_v1": "1",
    "hotpotqa_parquet_v1": "1",
    "musique_zip_v1": "1",
    "qasper_tgz_v1": "1",
    "tau3_banking_tar_v1": "1",
    "two_wiki_zip_v1": "1",
}
_PROCESSOR_FILES = {
    "beir_zip_v1": {"inputs.jsonl", "gold.jsonl", "corpus.jsonl"},
    "bfcl_v3_simple_tar_v1": {"inputs.jsonl", "gold.jsonl", "audit.jsonl"},
    "bright_corpus_parquet_v1": {"corpus.jsonl"},
    "bright_examples_parquet_v1": {"inputs.jsonl", "gold.jsonl"},
    "clariq_tsv_v1": {"inputs.jsonl", "gold.jsonl"},
    "hotpotqa_parquet_v1": {"inputs.jsonl", "gold.jsonl"},
    "musique_zip_v1": {"inputs.jsonl", "gold.jsonl", "audit.jsonl"},
    "qasper_tgz_v1": {"inputs.jsonl", "gold.jsonl"},
    "tau3_banking_tar_v1": {
        "inputs.jsonl",
        "gold.jsonl",
        "corpus.jsonl",
        "audit.jsonl",
    },
    "two_wiki_zip_v1": {"inputs.jsonl", "gold.jsonl", "audit.jsonl"},
}
_BEIR_MEMBERS = {
    "arguana/corpus.jsonl",
    "arguana/queries.jsonl",
    "arguana/qrels/test.tsv",
}
_CLARIQ_FIELDS = (
    "topic_id",
    "initial_request",
    "topic_desc",
    "clarification_need",
    "facet_id",
    "facet_desc",
    "question_id",
    "question",
    "answer",
)
_MAX_ARCHIVE_MEMBERS = 16
_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200.0
_MUSIQUE_MEMBERS = {
    "data/.DS_Store",
    "data/dev_test_singlehop_questions_v1.0.json",
    "data/musique_ans_v1.0_dev.jsonl",
    "data/musique_ans_v1.0_test.jsonl",
    "data/musique_ans_v1.0_train.jsonl",
    "data/musique_full_v1.0_dev.jsonl",
    "data/musique_full_v1.0_test.jsonl",
    "data/musique_full_v1.0_train.jsonl",
}
_QASPER_MEMBERS = {
    "README.md",
    "qasper-dev-v0.3.json",
    "qasper-train-v0.3.json",
}
_TWO_WIKI_MEMBERS = {
    "dev.json",
    "id_aliases.json",
    "test.json",
    "train.json",
}
_TAU3_BANKING_CELL = "tau3.banking_knowledge"
_TAU3_BANKING_DATA_ROOT = "data/tau2/domains/banking_knowledge"
_TAU3_BANKING_SOURCE_ROOT = "src/tau2/domains/banking_knowledge"
_TAU3_RUNTIME_SOURCE_PREFIXES = (
    "src/tau2/agent/base/",
    "src/tau2/data_model/",
    "src/tau2/domains/banking_knowledge/",
    "src/tau2/environment/",
    "src/tau2/evaluator/",
    "src/tau2/knowledge/",
    "src/tau2/orchestrator/",
    "src/tau2/runner/",
    "src/tau2/user/",
)
_TAU3_TASK_FIELDS = {
    "annotations",
    "description",
    "evaluation_criteria",
    "id",
    "initial_state",
    "required_documents",
    "user_scenario",
    "user_tools",
}
_TAU3_DOCUMENT_FIELDS = {"content", "id", "title"}
_TAU3_MAX_ARCHIVE_MEMBERS = 5_000
_TAU3_MAX_SELECTED_FILES = 2_048
_TAU3_MAX_SELECTED_MEMBER_BYTES = 8 * 1024 * 1024
_TAU3_MAX_SELECTED_BYTES = 64 * 1024 * 1024
_TAU3_VERSION_PATTERN = re.compile(
    r'^version\s*=\s*["\'](?P<version>[^"\']+)["\']\s*$', re.MULTILINE
)
_BFCL_V3_CELL = "bfcl.v3.simple_python"
_BFCL_V3_ARCHIVE_ROOT = "gorilla-1.3"
_BFCL_V3_SOURCE_ROOT = "berkeley-function-call-leaderboard"
_BFCL_V3_PROMPT_PATH = f"{_BFCL_V3_SOURCE_ROOT}/bfcl_eval/data/BFCL_v3_simple.json"
_BFCL_V3_GOLD_PATH = f"{_BFCL_V3_SOURCE_ROOT}/bfcl_eval/data/possible_answer/BFCL_v3_simple.json"
_BFCL_V3_REQUIRED_FILES = {
    "LICENSE",
    f"{_BFCL_V3_SOURCE_ROOT}/pyproject.toml",
    _BFCL_V3_PROMPT_PATH,
    _BFCL_V3_GOLD_PATH,
    f"{_BFCL_V3_SOURCE_ROOT}/bfcl_eval/constants/category_mapping.py",
    f"{_BFCL_V3_SOURCE_ROOT}/bfcl_eval/eval_checker/ast_eval/ast_checker.py",
    f"{_BFCL_V3_SOURCE_ROOT}/bfcl_eval/eval_checker/eval_runner.py",
    f"{_BFCL_V3_SOURCE_ROOT}/bfcl_eval/model_handler/base_handler.py",
}
_BFCL_V3_MAX_ARCHIVE_MEMBERS = 1_000
_BFCL_V3_MAX_SELECTED_MEMBER_BYTES = 2 * 1024 * 1024
_BFCL_V3_MAX_SELECTED_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PreparedBenchmark:
    asset_id: str
    processor: str
    processor_revision: str
    source_sha256: str
    root: Path
    counts: Mapping[str, int]
    file_sha256: Mapping[str, str]
    cell_id: str = ""
    source_revision: str = ""
    split: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "file_sha256", MappingProxyType(dict(self.file_sha256)))

    @property
    def inputs_path(self) -> Path:
        return self.root / "inputs.jsonl"

    @property
    def gold_path(self) -> Path:
        return self.root / "gold.jsonl"

    @property
    def corpus_path(self) -> Path | None:
        path = self.root / "corpus.jsonl"
        return path if path.is_file() else None

    @property
    def audit_path(self) -> Path | None:
        path = self.root / "audit.jsonl"
        return path if path.is_file() else None


def prepare_smoke_asset(
    asset: BenchmarkSmokeAsset,
    source: str | Path,
    output_root: str | Path,
    *,
    max_prepared_bytes: int = DEFAULT_PREPARED_LIMIT,
) -> PreparedBenchmark:
    """Verify and normalize one complete source artifact into an immutable cache.

    The normalized cache always stores optimizer-visible inputs separately from
    evaluator-only gold.  Existing caches are verified before reuse and never
    silently overwritten.
    """

    positive_byte_limit(max_prepared_bytes, name="max_prepared_bytes")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise ArtifactError(f"benchmark source is missing: {source_path}")
    if source_path.stat().st_size != asset.size_bytes:
        raise ArtifactError(f"benchmark source size mismatch: {asset.asset_id}")
    if sha256_file(source_path) != asset.sha256:
        raise ArtifactError(f"benchmark source checksum mismatch: {asset.asset_id}")

    revision = _PROCESSOR_REVISIONS.get(asset.processor)
    if revision is None:
        raise ArtifactError(f"unsupported benchmark processor: {asset.processor}")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = _prepared_path(asset, root)
    if destination.exists():
        return _load_prepared(destination, expected_asset=asset)
    require_storage_capacity([(root, max_prepared_bytes)])

    if asset.processor == "beir_zip_v1":
        normalized = _normalize_beir_zip(source_path, asset)
    elif asset.processor == "bfcl_v3_simple_tar_v1":
        normalized = _normalize_bfcl_v3_simple_tar(source_path, asset)
    elif asset.processor == "bright_examples_parquet_v1":
        normalized = _normalize_bright_examples(source_path, asset)
    elif asset.processor == "bright_corpus_parquet_v1":
        normalized = _normalize_bright_corpus(source_path, asset)
    elif asset.processor == "clariq_tsv_v1":
        normalized = _normalize_clariq_tsv(source_path, asset)
    elif asset.processor == "musique_zip_v1":
        normalized = _normalize_musique_zip(source_path, asset)
    elif asset.processor == "two_wiki_zip_v1":
        normalized = _normalize_two_wiki_zip(source_path, asset)
    elif asset.processor == "hotpotqa_parquet_v1":
        normalized = _normalize_hotpotqa_parquet(source_path, asset)
    elif asset.processor == "qasper_tgz_v1":
        normalized = _normalize_qasper_tgz(source_path, asset)
    elif asset.processor == "tau3_banking_tar_v1":
        normalized = _normalize_tau3_banking_tar(source_path, asset)
    else:  # pragma: no cover - guarded above and by the asset registry
        raise ArtifactError(f"unsupported benchmark processor: {asset.processor}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=parent))
    try:
        file_sha256: dict[str, str] = {}
        written_bytes = 0

        def write_bounded(path: Path, content: bytes) -> None:
            nonlocal written_bytes
            if written_bytes + len(content) > max_prepared_bytes:
                raise ArtifactError("normalized benchmark exceeds max_prepared_bytes")
            require_storage_capacity([(temporary, len(content))])
            _write_bytes(path, content)
            written_bytes += len(content)

        for name in ("inputs.jsonl", "gold.jsonl", "corpus.jsonl", "audit.jsonl"):
            records = normalized.get(name)
            if records is None:
                continue
            content = _jsonl(records)
            path = temporary / name
            write_bounded(path, content)
            file_sha256[name] = hashlib.sha256(content).hexdigest()
        if asset.prepared_sha256 and dict(asset.prepared_sha256) != file_sha256:
            raise ArtifactError(
                f"prepared content hash mismatch for {asset.asset_id}: "
                "the processor output differs from the audited lock"
            )
        counts = dict(normalized["counts"])
        _require_expected_counts(asset, counts)
        metadata = {
            "schema_version": 1,
            "asset_id": asset.asset_id,
            "cell_id": asset.cell_id,
            "source_revision": asset.source_revision,
            "source_sha256": asset.sha256,
            "source_size_bytes": asset.size_bytes,
            "processor": asset.processor,
            "processor_revision": revision,
            "gold_separated": True,
            "counts": counts,
            "files": file_sha256,
        }
        if asset.processor not in {"beir_zip_v1", "clariq_tsv_v1"}:
            metadata["split"] = _processor_split(asset.processor)
        write_bounded(
            temporary / "metadata.json",
            (canonical_json(metadata) + "\n").encode("utf-8"),
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _load_prepared(destination, expected_asset=asset)


def load_prepared_benchmark(
    asset: BenchmarkSmokeAsset,
    prepared_root: str | Path,
) -> PreparedBenchmark:
    """Load and reverify a deterministic prepared cache without source bytes."""

    root = Path(prepared_root).expanduser().resolve()
    destination = _prepared_path(asset, root)
    if not destination.is_dir():
        raise ArtifactError(f"prepared benchmark cache is missing: {asset.asset_id}")
    return _load_prepared(destination, expected_asset=asset)


def _prepared_path(asset: BenchmarkSmokeAsset, root: Path) -> Path:
    revision = _PROCESSOR_REVISIONS.get(asset.processor)
    if revision is None:
        raise ArtifactError(f"unsupported benchmark processor: {asset.processor}")
    cache_key = hashlib.sha256(f"{asset.sha256}:{asset.processor}:{revision}".encode()).hexdigest()[
        :20
    ]
    return root / asset.asset_id / cache_key


def _normalize_beir_zip(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ArtifactError("BEIR source is not a valid ZIP archive") from exc
    with archive:
        _validate_zip(archive)
        corpus = _read_jsonl_member(archive, "arguana/corpus.jsonl")
        queries = _read_jsonl_member(archive, "arguana/queries.jsonl")
        qrel_text = _read_text_member(archive, "arguana/qrels/test.tsv")

    corpus_by_id = _records_by_id(corpus, label="BEIR corpus")
    query_by_id = _records_by_id(queries, label="BEIR queries")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    reader = csv.DictReader(io.StringIO(qrel_text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != ("query-id", "corpus-id", "score"):
        raise ArtifactError("BEIR qrels schema does not match query-id/corpus-id/score")
    qrel_count = 0
    for row in reader:
        query_id = _required_text(row, "query-id", label="BEIR qrels")
        corpus_id = _required_text(row, "corpus-id", label="BEIR qrels")
        try:
            score = int(_required_text(row, "score", label="BEIR qrels"))
        except ValueError as exc:
            raise ArtifactError("BEIR qrels score must be an integer") from exc
        if score < 0:
            raise ArtifactError("BEIR qrels score must be non-negative")
        if corpus_id in qrels[query_id]:
            raise ArtifactError("BEIR qrels contain a duplicate query/document pair")
        qrels[query_id][corpus_id] = score
        qrel_count += 1
    if set(qrels) != set(query_by_id):
        raise ArtifactError("BEIR qrels and query IDs do not match")
    orphan_queries = {
        query_id for query_id, judgments in qrels.items() if set(judgments) - set(corpus_by_id)
    }

    inputs = []
    gold = []
    for query_id in sorted(query_by_id):
        query = query_by_id[query_id]
        text = _required_text(query, "text", label="BEIR query")
        inputs.append(
            {
                "id": query_id,
                "query": text,
                "instruction": "Retrieve the most relevant counterargument.",
                "visible_metadata": {
                    "benchmark_cell": asset.cell_id,
                    "source_query_id": query_id,
                },
            }
        )
        gold.append(
            {
                "id": query_id,
                "relevance": {
                    doc_id: qrels[query_id][doc_id] for doc_id in sorted(qrels[query_id])
                },
                "orphan_relevant_document_ids": sorted(set(qrels[query_id]) - set(corpus_by_id)),
            }
        )
    normalized_corpus = []
    for document_id in sorted(corpus_by_id):
        document = corpus_by_id[document_id]
        normalized_corpus.append(
            {
                "id": document_id,
                "title": _optional_text(document, "title", label="BEIR corpus"),
                "text": _required_text(document, "text", label="BEIR corpus"),
            }
        )
    return {
        "inputs.jsonl": inputs,
        "gold.jsonl": gold,
        "corpus.jsonl": normalized_corpus,
        "counts": {
            "inputs": len(inputs),
            "gold": len(gold),
            "corpus": len(normalized_corpus),
            "qrels": qrel_count,
            "orphan_qrels": sum(
                doc_id not in corpus_by_id for judgments in qrels.values() for doc_id in judgments
            ),
            "orphan_queries": len(orphan_queries),
        },
    }


def _normalize_bright_examples(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    rows = _read_parquet_rows(
        source,
        expected_columns={
            "excluded_ids",
            "gold_answer",
            "gold_ids",
            "gold_ids_long",
            "id",
            "query",
            "reasoning",
        },
        label="BRIGHT examples",
    )
    instructions = {
        "bright.earth_science": (
            "Retrieve short documents that provide the evidence needed to answer this "
            "reasoning-intensive Earth Science query."
        ),
        "bright.economics": (
            "Retrieve short documents that provide the evidence needed to answer this "
            "reasoning-intensive Economics query."
        ),
        "bright.robotics": (
            "Retrieve short documents that provide the evidence needed to answer this "
            "reasoning-intensive Robotics query."
        ),
        "bright.stackoverflow": (
            "Retrieve short documents that provide the evidence needed to answer this "
            "reasoning-intensive Stack Overflow query."
        ),
    }
    try:
        instruction = instructions[asset.cell_id]
    except KeyError as exc:
        raise ArtifactError(f"unsupported BRIGHT examples cell: {asset.cell_id}") from exc
    normalized: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    qrel_count = 0
    excluded_count = 0
    for row in rows:
        example_id = _required_text(row, "id", label="BRIGHT example")
        query = _required_text(row, "query", label="BRIGHT example")
        gold_ids = _string_list(row.get("gold_ids"), label="BRIGHT gold_ids", non_empty=True)
        excluded_ids = _string_list(
            row.get("excluded_ids"), label="BRIGHT excluded_ids", non_empty=False
        )
        relevance = {document_id: 1 for document_id in sorted(set(gold_ids))}
        if len(relevance) != len(gold_ids):
            raise ArtifactError("BRIGHT gold_ids contain duplicates")
        qrel_count += len(relevance)
        excluded = sorted({item for item in excluded_ids if item != "N/A"})
        excluded_count += len(excluded)
        visible = {
            "id": example_id,
            "query": query,
            "instruction": instruction,
            "visible_metadata": {
                "benchmark_cell": asset.cell_id,
                "source_split": "test",
                "source_query_id": example_id,
            },
        }
        evaluator = {
            "id": example_id,
            "relevance": relevance,
            "evaluator_metadata": {
                "excluded_document_ids": excluded,
                "gold_answer": _optional_text(row, "gold_answer", label="BRIGHT example"),
                "gold_ids_long": _string_list(
                    row.get("gold_ids_long"),
                    label="BRIGHT gold_ids_long",
                    non_empty=False,
                ),
                "reasoning": _optional_text(row, "reasoning", label="BRIGHT example"),
            },
        }
        normalized.append((example_id, visible, evaluator))
    normalized.sort(key=lambda item: item[0])
    _require_unique_ids((item[0] for item in normalized), label="BRIGHT examples")
    return {
        "inputs.jsonl": [item[1] for item in normalized],
        "gold.jsonl": [item[2] for item in normalized],
        "counts": {
            "excluded_documents": excluded_count,
            "gold": len(normalized),
            "inputs": len(normalized),
            "qrels": qrel_count,
        },
    }


def _normalize_bright_corpus(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    if asset.cell_id not in {
        "bright.earth_science",
        "bright.economics",
        "bright.robotics",
        "bright.stackoverflow",
    }:
        raise ArtifactError(f"unsupported BRIGHT corpus cell: {asset.cell_id}")
    rows = _read_parquet_rows(
        source,
        expected_columns={"content", "id"},
        label="BRIGHT corpus",
    )
    corpus: list[dict[str, str]] = []
    ids: list[str] = []
    for row in rows:
        document_id = _required_text(row, "id", label="BRIGHT corpus")
        ids.append(document_id)
        corpus.append(
            {
                "id": document_id,
                "title": "",
                "text": _required_text(row, "content", label="BRIGHT corpus"),
            }
        )
    _require_unique_ids(ids, label="BRIGHT corpus")
    corpus.sort(key=lambda item: item["id"])
    return {
        "corpus.jsonl": corpus,
        "counts": {"corpus": len(corpus)},
    }


def _normalize_musique_zip(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    if asset.cell_id != "musique.answerable":
        raise ArtifactError(f"unsupported MuSiQue cell: {asset.cell_id}")
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ArtifactError("MuSiQue source is not a valid ZIP archive") from exc
    with archive:
        _validate_named_zip(
            archive,
            allowed_files=_MUSIQUE_MEMBERS,
            required_files={
                "data/dev_test_singlehop_questions_v1.0.json",
                "data/musique_ans_v1.0_dev.jsonl",
            },
            max_members=16,
            max_member_bytes=512 * 1024 * 1024,
            max_uncompressed_bytes=1024 * 1024 * 1024,
            label="MuSiQue",
        )
        rows = _read_jsonl_member(archive, "data/musique_ans_v1.0_dev.jsonl")
        leakage_payload = _read_json_member(
            archive,
            "data/dev_test_singlehop_questions_v1.0.json",
            label="MuSiQue leakage audit",
        )
    audit_rows = _normalize_musique_leakage_audit(leakage_payload)
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    context_count = 0
    support_count = 0
    for row in rows:
        example_id = _required_text(row, "id", label="MuSiQue example")
        if row.get("answerable") is not True:
            raise ArtifactError("MuSiQue answerable dev contains a non-answerable row")
        paragraphs = row.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ArtifactError("MuSiQue example requires a non-empty paragraphs list")
        blocks: list[dict[str, Any]] = []
        support_ids: list[str] = []
        paragraph_indices: list[int] = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                raise ArtifactError("MuSiQue paragraph must be an object")
            index = paragraph.get("idx")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ArtifactError("MuSiQue paragraph idx must be a non-negative integer")
            paragraph_indices.append(index)
            block_id = f"paragraph:{index}"
            title = _required_text(paragraph, "title", label="MuSiQue paragraph")
            blocks.append(
                {
                    "block_id": block_id,
                    "text": _required_text(
                        paragraph,
                        "paragraph_text",
                        label="MuSiQue paragraph",
                    ),
                    "source": title,
                    "metadata": {"paragraph_index": index, "title": title},
                }
            )
            is_supporting = paragraph.get("is_supporting")
            if not isinstance(is_supporting, bool):
                raise ArtifactError("MuSiQue is_supporting must be boolean")
            if is_supporting:
                support_ids.append(block_id)
        _require_unique_ids(
            (str(index) for index in paragraph_indices),
            label="MuSiQue paragraph indices",
        )
        if not support_ids:
            raise ArtifactError("MuSiQue answerable row requires supporting paragraphs")
        answer = _required_text(row, "answer", label="MuSiQue example")
        aliases = _string_list(
            row.get("answer_aliases"), label="MuSiQue answer_aliases", non_empty=False
        )
        answers = _ordered_unique((answer, *aliases))
        decomposition = row.get("question_decomposition")
        if not isinstance(decomposition, list):
            raise ArtifactError("MuSiQue question_decomposition must be a list")
        inputs.append(
            {
                "id": example_id,
                "query": _required_text(row, "question", label="MuSiQue example"),
                "instruction": (
                    "Answer the multi-hop question using the provided paragraphs and identify "
                    "the supporting paragraphs."
                ),
                "visible_context": blocks,
                "visible_metadata": {
                    "benchmark_cell": asset.cell_id,
                    "source_split": "dev",
                },
            }
        )
        gold.append(
            {
                "id": example_id,
                "answers": answers,
                "support_ids": sorted(support_ids),
                "evaluator_metadata": {"question_decomposition": decomposition},
            }
        )
        context_count += len(blocks)
        support_count += len(support_ids)
    inputs.sort(key=lambda item: item["id"])
    gold.sort(key=lambda item: item["id"])
    _require_unique_ids((item["id"] for item in inputs), label="MuSiQue examples")
    return {
        "audit.jsonl": audit_rows,
        "inputs.jsonl": inputs,
        "gold.jsonl": gold,
        "counts": {
            "context_blocks": context_count,
            "gold": len(gold),
            "inputs": len(inputs),
            "leakage_audit_records": len(audit_rows),
            "leakage_audit_sources": len(leakage_payload),
            "support_labels": support_count,
        },
    }


def _normalize_two_wiki_zip(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    if asset.cell_id != "2wiki.multihop":
        raise ArtifactError(f"unsupported 2Wiki cell: {asset.cell_id}")
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ArtifactError("2WikiMultiHopQA source is not a valid ZIP archive") from exc
    with archive:
        _validate_named_zip(
            archive,
            allowed_files=_TWO_WIKI_MEMBERS,
            required_files={"dev.json", "id_aliases.json"},
            max_members=8,
            max_member_bytes=768 * 1024 * 1024,
            max_uncompressed_bytes=900 * 1024 * 1024,
            label="2WikiMultiHopQA",
        )
        rows = _read_json_member(archive, "dev.json", label="2WikiMultiHopQA dev")
        alias_rows = _read_jsonl_member(archive, "id_aliases.json")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ArtifactError("2WikiMultiHopQA dev must be a non-empty JSON array of objects")
    expected_fields = {
        "_id",
        "answer",
        "answer_id",
        "context",
        "entity_ids",
        "evidences",
        "evidences_id",
        "question",
        "supporting_facts",
        "type",
    }
    for row in rows:
        if set(row) != expected_fields:
            raise ArtifactError("2WikiMultiHopQA dev row schema differs from the audited release")
    aliases, alias_audit, alias_count = _normalize_two_wiki_aliases(alias_rows)
    normalized = _normalize_hotpot_style_rows(
        rows,
        asset=asset,
        source_split="dev",
        two_wiki=True,
        two_wiki_aliases=aliases,
    )
    normalized["audit.jsonl"] = alias_audit
    normalized["counts"]["alias_entities"] = len(alias_audit)
    normalized["counts"]["alias_strings"] = alias_count
    return normalized


def _normalize_hotpotqa_parquet(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    if asset.cell_id != "hotpotqa.distractor":
        raise ArtifactError(f"unsupported HotpotQA cell: {asset.cell_id}")
    rows = _read_parquet_rows(
        source,
        expected_columns={
            "answer",
            "context",
            "id",
            "level",
            "question",
            "supporting_facts",
            "type",
        },
        label="HotpotQA distractor validation",
    )
    return _normalize_hotpot_style_rows(
        rows,
        asset=asset,
        source_split="validation",
        two_wiki=False,
    )


def _normalize_hotpot_style_rows(
    rows: list[dict[str, Any]],
    *,
    asset: BenchmarkSmokeAsset,
    source_split: str,
    two_wiki: bool,
    two_wiki_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    context_count = 0
    support_count = 0
    orphan_support_count = 0
    duplicate_context_count = 0
    empty_context_count = 0
    evidence_count = 0
    ids: list[str] = []
    label = "2WikiMultiHopQA" if two_wiki else "HotpotQA"
    for row in rows:
        id_key = "_id" if two_wiki else "id"
        example_id = _required_text(row, id_key, label=f"{label} example")
        ids.append(example_id)
        raw_context = (
            row.get("context")
            if two_wiki
            else _hotpot_struct_context(row.get("context"), label=label)
        )
        raw_support = (
            row.get("supporting_facts")
            if two_wiki
            else _hotpot_struct_support(row.get("supporting_facts"), label=label)
        )
        if not isinstance(raw_context, list) or not raw_context:
            raise ArtifactError(f"{label} context must be a non-empty list")
        blocks: list[dict[str, Any]] = []
        available_supports: set[str] = set()
        visible_sentence_text: dict[str, str] = {}
        for paragraph_index, item in enumerate(raw_context):
            if not isinstance(item, list) or len(item) != 2:
                raise ArtifactError(f"{label} context entries must be [title, sentences]")
            title, sentences = item
            if not isinstance(title, str) or not title.strip():
                raise ArtifactError(f"{label} context title must be non-empty")
            if not isinstance(sentences, list) or not sentences:
                raise ArtifactError(f"{label} context sentences must be a non-empty list")
            if not all(isinstance(sentence, str) for sentence in sentences):
                raise ArtifactError(f"{label} context sentences must be strings")
            for sentence_index, sentence in enumerate(sentences):
                support_id = _support_identifier(title, sentence_index)
                if not sentence.strip():
                    empty_context_count += 1
                    continue
                if support_id in visible_sentence_text:
                    if visible_sentence_text[support_id] != sentence:
                        raise ArtifactError(f"{label} duplicate sentence ID has conflicting text")
                    duplicate_context_count += 1
                else:
                    visible_sentence_text[support_id] = sentence
                available_supports.add(support_id)
                blocks.append(
                    {
                        "block_id": f"paragraph:{paragraph_index}:sentence:{sentence_index}",
                        "text": sentence,
                        "source": title,
                        "metadata": {
                            "official_support_id": support_id,
                            "paragraph_index": paragraph_index,
                            "sentence_index": sentence_index,
                            "title": title,
                        },
                    }
                )
        if not isinstance(raw_support, list) or not raw_support:
            raise ArtifactError(f"{label} supporting facts must be a non-empty list")
        support_ids: list[str] = []
        orphan_support_ids: list[str] = []
        for item in raw_support:
            if not isinstance(item, list) or len(item) != 2:
                raise ArtifactError(f"{label} supporting facts must contain [title, sent_id]")
            title, sentence_index = item
            if not isinstance(title, str) or not title.strip():
                raise ArtifactError(f"{label} supporting-fact title must be non-empty")
            if (
                isinstance(sentence_index, bool)
                or not isinstance(sentence_index, int)
                or sentence_index < 0
            ):
                raise ArtifactError(f"{label} supporting-fact index must be non-negative")
            support_id = _support_identifier(title, sentence_index)
            if support_id not in available_supports:
                # The official HotpotQA validation release contains a small
                # number of source annotation defects (for example sentence
                # index 902 in a five-sentence paragraph).  Preserve these
                # labels for scorer parity and record them explicitly instead
                # of silently dropping or repairing upstream gold.
                orphan_support_ids.append(support_id)
            support_ids.append(support_id)
        if len(set(support_ids)) != len(support_ids):
            raise ArtifactError(f"{label} supporting facts contain duplicates")
        evaluator_metadata: dict[str, Any] = {
            "question_type": _required_text(row, "type", label=f"{label} example")
        }
        if not two_wiki:
            evaluator_metadata["difficulty"] = _required_text(
                row,
                "level",
                label=f"{label} example",
            )
        if orphan_support_ids:
            evaluator_metadata["orphan_support_ids"] = sorted(orphan_support_ids)
        if two_wiki:
            if two_wiki_aliases is None:  # pragma: no cover - caller invariant
                raise ArtifactError("2Wiki aliases are required")
            evidences, evidence_ids, evidence_options = _two_wiki_evidence_groups(
                row,
                aliases=two_wiki_aliases,
            )
            answer_id = row.get("answer_id")
            if answer_id is not None and not isinstance(answer_id, str):
                raise ArtifactError("2Wiki answer_id must be a string or null")
            entity_ids = _required_text(row, "entity_ids", label="2Wiki example")
            evaluator_metadata.update(
                {
                    "answer_id": answer_id,
                    "entity_ids": entity_ids,
                    "evidence_options": evidence_options,
                }
            )
            evidence_count += len(evidences)
            answer = _required_text(row, "answer", label=f"{label} example")
            answers = _ordered_unique(
                (
                    answer,
                    *(two_wiki_aliases.get(answer_id, ()) if answer_id is not None else ()),
                )
            )
        else:
            answers = [_required_text(row, "answer", label=f"{label} example")]
        inputs.append(
            {
                "id": example_id,
                "query": _required_text(row, "question", label=f"{label} example"),
                "instruction": (
                    "Answer the multi-hop question using the provided paragraphs and identify "
                    "the supporting facts."
                ),
                "visible_context": blocks,
                "visible_metadata": {
                    "benchmark_cell": asset.cell_id,
                    "source_split": source_split,
                },
            }
        )
        gold.append(
            {
                "id": example_id,
                "answers": answers,
                "support_ids": sorted(support_ids),
                **({"evidence_ids": evidence_ids} if two_wiki else {}),
                "evaluator_metadata": evaluator_metadata,
            }
        )
        context_count += len(blocks)
        support_count += len(support_ids)
        orphan_support_count += len(orphan_support_ids)
    _require_unique_ids(ids, label=f"{label} examples")
    inputs.sort(key=lambda item: item["id"])
    gold.sort(key=lambda item: item["id"])
    counts = {
        "context_blocks": context_count,
        "duplicate_context_sentences": duplicate_context_count,
        "empty_context_sentences": empty_context_count,
        "gold": len(gold),
        "inputs": len(inputs),
        "orphan_support_labels": orphan_support_count,
        "support_labels": support_count,
    }
    if two_wiki:
        counts["evidence_triples"] = evidence_count
    return {"inputs.jsonl": inputs, "gold.jsonl": gold, "counts": counts}


def _normalize_qasper_tgz(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    if asset.cell_id != "qasper.v0_3_dev":
        raise ArtifactError(f"unsupported QASPER cell: {asset.cell_id}")
    try:
        with tarfile.open(source, mode="r:gz") as archive:
            _validate_named_tar(
                archive,
                allowed_files=_QASPER_MEMBERS,
                required_files={"qasper-dev-v0.3.json"},
                max_members=8,
                max_member_bytes=64 * 1024 * 1024,
                max_uncompressed_bytes=128 * 1024 * 1024,
                label="QASPER",
            )
            member = archive.getmember("qasper-dev-v0.3.json")
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - validated regular member
                raise ArtifactError("QASPER dev member cannot be read")
            try:
                papers = json.load(handle)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactError("QASPER dev member is not valid UTF-8 JSON") from exc
    except tarfile.TarError as exc:
        raise ArtifactError("QASPER source is not a valid gzip tar archive") from exc
    if not isinstance(papers, dict) or not papers:
        raise ArtifactError("QASPER dev data must be a non-empty object")
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    question_ids: list[str] = []
    annotation_count = 0
    evidence_count = 0
    context_count = 0
    unique_context_count = 0
    for paper_id in sorted(papers):
        paper = papers[paper_id]
        if not isinstance(paper, Mapping):
            raise ArtifactError("QASPER paper must be an object")
        blocks = _qasper_context_blocks(paper, paper_id=paper_id)
        unique_context_count += len(blocks)
        qas = paper.get("qas")
        if not isinstance(qas, list) or not qas:
            raise ArtifactError("QASPER paper requires a non-empty qas list")
        for qa in qas:
            if not isinstance(qa, Mapping):
                raise ArtifactError("QASPER qa entry must be an object")
            question_id = _required_text(qa, "question_id", label="QASPER question")
            question_ids.append(question_id)
            annotations = qa.get("answers")
            if not isinstance(annotations, list) or not annotations:
                raise ArtifactError("QASPER question requires answer annotations")
            groups: list[dict[str, Any]] = []
            annotation_ids: list[str] = []
            for annotation in annotations:
                if not isinstance(annotation, Mapping):
                    raise ArtifactError("QASPER answer annotation must be an object")
                answer_payload = annotation.get("answer")
                if not isinstance(answer_payload, Mapping):
                    raise ArtifactError("QASPER answer annotation requires an answer object")
                answer_value, answer_type = _qasper_answer(answer_payload)
                raw_evidence = _string_list(
                    answer_payload.get("evidence"),
                    label="QASPER evidence",
                    non_empty=False,
                    allow_empty_items=True,
                )
                evidence = [item for item in (_normalize_space(x) for x in raw_evidence) if item]
                group = {
                    "answers": [answer_value],
                    "answer_type": answer_type,
                    "evidence_texts": evidence,
                }
                groups.append(group)
                annotation_id = _required_text(
                    annotation,
                    "annotation_id",
                    label="QASPER answer annotation",
                )
                annotation_ids.append(annotation_id)
                annotation_count += 1
                evidence_count += len(evidence)
            inputs.append(
                {
                    "id": question_id,
                    "query": _required_text(qa, "question", label="QASPER question"),
                    "instruction": (
                        "Answer the question using the provided scientific paper and return the "
                        "supporting evidence passages."
                    ),
                    "visible_context": blocks,
                    "visible_metadata": {
                        "benchmark_cell": asset.cell_id,
                        "paper_id": paper_id,
                        "source_split": "dev",
                    },
                }
            )
            context_count += len(blocks)
            gold.append(
                {
                    "id": question_id,
                    "reference_groups": groups,
                    "evaluator_metadata": {"annotation_ids": annotation_ids},
                }
            )
    _require_unique_ids(question_ids, label="QASPER question IDs")
    inputs.sort(key=lambda item: item["id"])
    gold.sort(key=lambda item: item["id"])
    return {
        "inputs.jsonl": inputs,
        "gold.jsonl": gold,
        "counts": {
            "annotations": annotation_count,
            "context_blocks": context_count,
            "evidence_labels": evidence_count,
            "gold": len(gold),
            "inputs": len(inputs),
            "papers": len(papers),
            "unique_paper_context_blocks": unique_context_count,
        },
    }


def _qasper_context_blocks(
    paper: Mapping[str, Any],
    *,
    paper_id: str,
) -> list[dict[str, Any]]:
    full_text = paper.get("full_text")
    if not isinstance(full_text, list) or not full_text:
        raise ArtifactError("QASPER paper requires non-empty full_text")
    blocks: list[dict[str, Any]] = []
    for section_index, section in enumerate(full_text):
        if not isinstance(section, Mapping):
            raise ArtifactError("QASPER full_text sections must be objects")
        section_name = section.get("section_name")
        if section_name is not None and not isinstance(section_name, str):
            raise ArtifactError("QASPER section_name must be a string or null")
        normalized_section = _normalize_space(section_name or "")
        if normalized_section:
            blocks.append(
                {
                    "block_id": f"section:{section_index}:heading",
                    "text": normalized_section,
                    "source": paper_id,
                    "metadata": {
                        "kind": "section_heading",
                        "section_index": section_index,
                    },
                }
            )
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise ArtifactError("QASPER section paragraphs must be a list")
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, str):
                raise ArtifactError("QASPER paragraphs must be strings")
            text = _normalize_space(paragraph)
            if not text:
                continue
            blocks.append(
                {
                    "block_id": f"section:{section_index}:paragraph:{paragraph_index}",
                    "text": text,
                    "source": normalized_section or paper_id,
                    "metadata": {
                        "kind": "paragraph",
                        "section_index": section_index,
                        "paragraph_index": paragraph_index,
                    },
                }
            )
    figures = paper.get("figures_and_tables")
    if not isinstance(figures, list):
        raise ArtifactError("QASPER figures_and_tables must be a list")
    for figure_index, figure in enumerate(figures):
        if not isinstance(figure, Mapping):
            raise ArtifactError("QASPER figure/table entries must be objects")
        caption = _normalize_space(_required_text(figure, "caption", label="QASPER figure/table"))
        blocks.append(
            {
                "block_id": f"float:{figure_index}",
                "text": f"FLOAT SELECTED: {caption}",
                "source": paper_id,
                "metadata": {
                    "file": _optional_text(figure, "file", label="QASPER figure/table"),
                    "kind": "figure_or_table",
                },
            }
        )
    if not blocks:
        raise ArtifactError("QASPER paper produced no visible context blocks")
    _require_unique_ids((item["block_id"] for item in blocks), label="QASPER context blocks")
    return blocks


def _qasper_answer(answer: Mapping[str, Any]) -> tuple[str, str]:
    unanswerable = answer.get("unanswerable")
    if not isinstance(unanswerable, bool):
        raise ArtifactError("QASPER unanswerable must be boolean")
    if unanswerable:
        return "Unanswerable", "none"
    extractive = _string_list(
        answer.get("extractive_spans"),
        label="QASPER extractive_spans",
        non_empty=False,
    )
    free_form = answer.get("free_form_answer")
    if not isinstance(free_form, str):
        raise ArtifactError("QASPER free_form_answer must be a string")
    yes_no = answer.get("yes_no")
    if yes_no is not None and not isinstance(yes_no, bool):
        raise ArtifactError("QASPER yes_no must be boolean or null")
    populated = int(bool(extractive)) + int(bool(free_form)) + int(yes_no is not None)
    if populated != 1:
        raise ArtifactError("QASPER answer must contain exactly one answer type")
    if extractive:
        return ", ".join(extractive), "extractive"
    if free_form:
        return free_form, "abstractive"
    return ("Yes" if yes_no else "No"), "boolean"


def _read_parquet_rows(
    source: Path,
    *,
    expected_columns: set[str],
    label: str,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise ArtifactError(
            f"{label} preprocessing requires pyarrow; install agentinputlab[benchmarks]"
        ) from exc
    try:
        table = parquet.read_table(source)
        rows = table.to_pylist()
    except Exception as exc:  # pyarrow raises version-specific Arrow exception classes
        raise ArtifactError(f"{label} source is not a readable Parquet file") from exc
    if set(table.column_names) != expected_columns or len(table.column_names) != len(
        expected_columns
    ):
        raise ArtifactError(
            f"{label} columns do not match the audited schema: {sorted(table.column_names)!r}"
        )
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ArtifactError(f"{label} contains no object rows")
    return rows


def _hotpot_struct_context(value: Any, *, label: str) -> list[list[Any]]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} context must be an object")
    titles = value.get("title")
    sentence_groups = value.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentence_groups, list):
        raise ArtifactError(f"{label} context requires title and sentences lists")
    if len(titles) != len(sentence_groups):
        raise ArtifactError(f"{label} context title/sentences lengths differ")
    return [[title, sentences] for title, sentences in zip(titles, sentence_groups, strict=True)]


def _hotpot_struct_support(value: Any, *, label: str) -> list[list[Any]]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} supporting_facts must be an object")
    titles = value.get("title")
    sentence_ids = value.get("sent_id")
    if not isinstance(titles, list) or not isinstance(sentence_ids, list):
        raise ArtifactError(f"{label} supporting_facts requires title and sent_id lists")
    if len(titles) != len(sentence_ids):
        raise ArtifactError(f"{label} supporting-fact title/sent_id lengths differ")
    return [[title, sentence_id] for title, sentence_id in zip(titles, sentence_ids, strict=True)]


def _normalize_two_wiki_aliases(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]], int]:
    aliases: dict[str, tuple[str, ...]] = {}
    audit: list[dict[str, Any]] = []
    alias_count = 0
    for row in rows:
        if set(row) != {"Q_id", "aliases", "demonyms"}:
            raise ArtifactError("2Wiki alias row schema differs from the audited release")
        entity_id = _required_text(row, "Q_id", label="2Wiki alias row")
        if entity_id in aliases:
            raise ArtifactError("2Wiki alias table contains duplicate entity IDs")
        names = _string_list(row.get("aliases"), label="2Wiki aliases", non_empty=False)
        demonyms = _string_list(row.get("demonyms"), label="2Wiki demonyms", non_empty=False)
        combined = tuple(_ordered_unique((*names, *demonyms)))
        aliases[entity_id] = combined
        alias_count += len(combined)
        audit.append(
            {
                "id": entity_id,
                "aliases": names,
                "demonyms": demonyms,
            }
        )
    audit.sort(key=lambda item: item["id"])
    return aliases, audit, alias_count


def _two_wiki_evidence_groups(
    row: Mapping[str, Any],
    *,
    aliases: Mapping[str, tuple[str, ...]],
) -> tuple[list[list[str]], list[str], list[list[str]]]:
    evidences = row.get("evidences")
    raw_ids = row.get("evidences_id")
    if not isinstance(evidences, list) or not evidences:
        raise ArtifactError("2Wiki evidences must be a non-empty list")
    if not isinstance(raw_ids, list):
        raise ArtifactError("2Wiki evidences_id must be a list")
    if raw_ids and len(raw_ids) != len(evidences):
        raise ArtifactError("2Wiki evidences and evidences_id lengths differ")
    normalized_evidences: list[list[str]] = []
    evidence_ids: list[str] = []
    evidence_options: list[list[str]] = []
    for index, evidence in enumerate(evidences):
        if (
            not isinstance(evidence, list)
            or len(evidence) != 3
            or not all(isinstance(item, str) and item.strip() for item in evidence)
        ):
            raise ArtifactError("2Wiki evidence must be a non-empty string triple")
        subject, relation, object_ = evidence
        normalized_evidences.append(list(evidence))
        literal_id = canonical_json(evidence)
        evidence_ids.append(literal_id)
        subjects = {subject}
        objects = {object_}
        if raw_ids:
            identifier = raw_ids[index]
            if (
                not isinstance(identifier, list)
                or len(identifier) != 3
                or not all(isinstance(item, str) and item.strip() for item in identifier)
            ):
                raise ArtifactError("2Wiki evidence IDs must be non-empty string triples")
            subject_id, relation_id, object_id = identifier
            if relation_id != relation:
                raise ArtifactError("2Wiki evidence relation ID differs from its relation label")
            subjects.update(aliases.get(subject_id, ()))
            objects.update(aliases.get(object_id, ()))
        options = sorted(
            canonical_json([subject_name, relation, object_name])
            for subject_name in subjects
            for object_name in objects
        )
        evidence_options.append(options)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ArtifactError("2Wiki evidences contain duplicate literal triples")
    return normalized_evidences, sorted(evidence_ids), evidence_options


def _support_identifier(title: str, sentence_index: int) -> str:
    return canonical_json([title, sentence_index])


def _string_list(
    value: Any,
    *,
    label: str,
    non_empty: bool,
    allow_empty_items: bool = False,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ArtifactError(f"{label} must be a list of strings")
    if non_empty and not value:
        raise ArtifactError(f"{label} must not be empty")
    if not allow_empty_items and any(not item.strip() for item in value):
        raise ArtifactError(f"{label} must not contain empty strings")
    return list(value)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _require_unique_ids(values: Iterable[str], *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ArtifactError(f"{label} contain duplicate IDs")
        seen.add(value)


def _normalize_space(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def _normalize_musique_leakage_audit(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ArtifactError("MuSiQue leakage audit must be a non-empty object")
    records: list[dict[str, Any]] = []
    ids: list[str] = []
    for source in sorted(value):
        if not isinstance(source, str) or not source.strip():
            raise ArtifactError("MuSiQue leakage audit source must be non-empty")
        entries = value[source]
        if not isinstance(entries, list):
            raise ArtifactError("MuSiQue leakage audit source entries must be a list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ArtifactError("MuSiQue leakage audit entries must be objects")
            source_id = entry.get("id")
            if isinstance(source_id, bool) or not isinstance(source_id, (str, int)):
                raise ArtifactError("MuSiQue leakage audit IDs must be strings or integers")
            identity = canonical_json({"source": source, "source_id": source_id})
            record_id = f"{source}:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
            ids.append(record_id)
            records.append(
                {
                    "id": record_id,
                    "question": _required_text(
                        entry,
                        "question",
                        label="MuSiQue leakage audit entry",
                    ),
                    "source": source,
                    "source_id": source_id,
                }
            )
    _require_unique_ids(ids, label="MuSiQue leakage audit")
    records.sort(key=lambda item: item["id"])
    return records


def _normalize_tau3_banking_tar(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    """Normalize the official tau2 Banking Knowledge source without extracting it.

    The upstream task files contain a user-simulator script, evaluator criteria,
    required-document labels, and initial database state.  None of those fields
    is a valid optimizer input.  Static preprocessing therefore emits only a
    task inventory; an explicit captured visible checkpoint is required later by
    :class:`Tau3BankingCheckpointAdapter`.
    """

    if asset.cell_id != _TAU3_BANKING_CELL:
        raise ArtifactError(f"unsupported tau3 Banking Knowledge cell: {asset.cell_id}")
    selected = _read_tau3_banking_members(source, source_revision=asset.source_revision)
    data_root = f"{_TAU3_BANKING_DATA_ROOT}/"

    task_members = {
        name: payload
        for name, payload in selected.items()
        if name.startswith(f"{data_root}tasks/task_") and name.endswith(".json")
    }
    document_members = {
        name: payload
        for name, payload in selected.items()
        if name.startswith(f"{data_root}documents/") and name.endswith(".json")
    }
    if not task_members or not document_members:
        raise ArtifactError("tau3 Banking Knowledge archive requires tasks and documents")

    tasks: dict[str, dict[str, Any]] = {}
    for name, payload in sorted(task_members.items()):
        task = _tau3_json_object(payload, label=name)
        if set(task) != _TAU3_TASK_FIELDS:
            raise ArtifactError("tau3 Banking Knowledge task schema differs from the lock")
        task_id = _required_text(task, "id", label="tau3 Banking Knowledge task")
        if PurePosixPath(name).stem != task_id:
            raise ArtifactError("tau3 Banking Knowledge task ID differs from its filename")
        if task_id in tasks:
            raise ArtifactError("tau3 Banking Knowledge tasks contain duplicate IDs")
        _validate_tau3_task(task)
        tasks[task_id] = task

    documents: dict[str, dict[str, Any]] = {}
    for name, payload in sorted(document_members.items()):
        document = _tau3_json_object(payload, label=name)
        if set(document) != _TAU3_DOCUMENT_FIELDS:
            raise ArtifactError("tau3 Banking Knowledge document schema differs from the lock")
        document_id = _required_text(
            document,
            "id",
            label="tau3 Banking Knowledge document",
        )
        if PurePosixPath(name).stem != document_id:
            raise ArtifactError("tau3 Banking Knowledge document ID differs from its filename")
        if document_id in documents:
            raise ArtifactError("tau3 Banking Knowledge documents contain duplicate IDs")
        _required_text(document, "title", label="tau3 Banking Knowledge document")
        _required_text(document, "content", label="tau3 Banking Knowledge document")
        documents[document_id] = document

    required_document_mentions = 0
    required_document_ids: set[str] = set()
    gold_action_count = 0
    tasks_with_initial_state = 0
    tasks_with_initialization_data = 0
    tasks_with_message_history = 0
    reward_basis_counts: dict[str, int] = defaultdict(int)
    action_names: set[str] = set()
    for task in tasks.values():
        required_documents = task["required_documents"]
        required_document_mentions += len(required_documents)
        required_document_ids.update(required_documents)
        missing_documents = set(required_documents) - set(documents)
        if missing_documents:
            raise ArtifactError("tau3 Banking Knowledge task names missing required documents")

        criteria = task["evaluation_criteria"]
        actions = criteria["actions"]
        gold_action_count += len(actions)
        for action in actions:
            action_names.add(_required_text(action, "name", label="tau3 gold action"))
        for basis in criteria["reward_basis"]:
            reward_basis_counts[basis] += 1

        initial_state = task["initial_state"]
        if initial_state is not None:
            tasks_with_initial_state += 1
            if initial_state.get("initialization_data") is not None:
                tasks_with_initialization_data += 1
            if initial_state.get("message_history") is not None:
                tasks_with_message_history += 1

    legacy_path = f"{_TAU3_BANKING_DATA_ROOT}/tasks.json"
    legacy_payload = _tau3_json_value(selected[legacy_path], label=legacy_path)
    if not isinstance(legacy_payload, list) or not all(
        isinstance(item, dict) for item in legacy_payload
    ):
        raise ArtifactError("tau3 legacy combined tasks must be an array of objects")
    legacy_tasks: dict[str, dict[str, Any]] = {}
    for task in legacy_payload:
        task_id = _required_text(task, "id", label="tau3 legacy task")
        if task_id in legacy_tasks:
            raise ArtifactError("tau3 legacy tasks contain duplicate IDs")
        legacy_tasks[task_id] = task
    if set(legacy_tasks) != set(tasks):
        raise ArtifactError("tau3 legacy and authoritative task inventories differ")
    legacy_mismatches = sorted(
        task_id
        for task_id in tasks
        if canonical_json(tasks[task_id]) != canonical_json(legacy_tasks[task_id])
    )

    database_path = f"{_TAU3_BANKING_DATA_ROOT}/db.json"
    database = _tau3_json_object(selected[database_path], label=database_path)
    pyproject = _tau3_text(selected["pyproject.toml"], label="pyproject.toml")
    version_match = _TAU3_VERSION_PATTERN.search(pyproject)
    if version_match is None:
        raise ArtifactError("tau3 source pyproject does not declare a package version")
    package_version = version_match.group("version")
    if package_version != "1.0.1":
        raise ArtifactError("tau3 Banking Knowledge source must declare package version 1.0.1")
    license_text = _tau3_text(selected["LICENSE"], label="LICENSE")
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ArtifactError("tau3 source license does not match the audited MIT grant")

    prompt_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(selected.items())
        if name.startswith(f"{_TAU3_BANKING_DATA_ROOT}/prompts/")
    }
    runtime_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(selected.items())
        if _tau3_runtime_member(name)
    }

    inputs = [
        {
            "id": task_id,
            "checkpoint_required": True,
            "visible_metadata": {
                "benchmark_cell": asset.cell_id,
                "source_split": "base",
                "source_task_id": task_id,
            },
        }
        for task_id in sorted(tasks)
    ]
    gold = [{"id": task_id, "task_spec": tasks[task_id]} for task_id in sorted(tasks)]
    corpus = [
        {
            "id": document_id,
            "text": documents[document_id]["content"],
            "title": documents[document_id]["title"],
        }
        for document_id in sorted(documents)
    ]
    audit = [
        {
            "id": "tau3-banking-knowledge-source-audit",
            "authoritative_task_layout": "individual_task_files",
            "database": database,
            "database_sha256": hashlib.sha256(selected[database_path]).hexdigest(),
            "hidden_state_notice": (
                "Evaluator-only: never place database, user_scenario, evaluation_criteria, "
                "required_documents, or initial_state into an optimizer-visible checkpoint."
            ),
            "legacy_combined_task_mismatch_ids": legacy_mismatches,
            "license_sha256": hashlib.sha256(selected["LICENSE"]).hexdigest(),
            "package_name": "tau2",
            "package_version": package_version,
            "prompt_template_sha256": prompt_hashes,
            "reward_basis_task_counts": {
                key: reward_basis_counts[key] for key in sorted(reward_basis_counts)
            },
            "runtime_source_sha256": runtime_hashes,
            "source_revision": asset.source_revision,
            "wrapped_action_names": sorted(action_names),
        }
    ]
    return {
        "inputs.jsonl": inputs,
        "gold.jsonl": gold,
        "corpus.jsonl": corpus,
        "audit.jsonl": audit,
        "counts": {
            "audit": 1,
            "corpus": len(corpus),
            "documents": len(documents),
            "gold": len(gold),
            "gold_actions": gold_action_count,
            "inputs": len(inputs),
            "legacy_combined_task_mismatches": len(legacy_mismatches),
            "required_document_labels": required_document_mentions,
            "required_document_unique": len(required_document_ids),
            "tasks": len(tasks),
            "tasks_with_initial_state": tasks_with_initial_state,
            "tasks_with_initialization_data": tasks_with_initialization_data,
            "tasks_with_message_history": tasks_with_message_history,
            "user_simulator_scripts": len(tasks),
        },
    }


def _normalize_bfcl_v3_simple_tar(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    """Normalize the official BFCL v3 Python-simple cell without vendoring it.

    Function definitions and the user request are legitimate model inputs.  The
    possible-answer file is evaluator-only and is never copied into
    ``inputs.jsonl``.  Scoring remains an external official-AST bridge: this
    processor records source identities but does not execute a model or create
    an official result.
    """

    if asset.cell_id != _BFCL_V3_CELL:
        raise ArtifactError(f"unsupported BFCL v3 cell: {asset.cell_id}")
    selected = _read_bfcl_v3_members(source)
    prompts = _bfcl_jsonl(selected[_BFCL_V3_PROMPT_PATH], label=_BFCL_V3_PROMPT_PATH)
    answers = _bfcl_jsonl(selected[_BFCL_V3_GOLD_PATH], label=_BFCL_V3_GOLD_PATH)
    if len(prompts) != len(answers):
        raise ArtifactError("BFCL v3 prompt and possible-answer counts differ")

    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    prompt_ids: list[str] = []
    answer_ids: list[str] = []
    function_count = 0
    for prompt, answer in zip(prompts, answers, strict=True):
        if set(prompt) != {"function", "id", "question"}:
            raise ArtifactError("BFCL v3 simple prompt schema differs from the audited lock")
        if set(answer) != {"ground_truth", "id"}:
            raise ArtifactError("BFCL v3 simple gold schema differs from the audited lock")
        example_id = _required_text(prompt, "id", label="BFCL v3 simple prompt")
        answer_id = _required_text(answer, "id", label="BFCL v3 simple gold")
        if example_id != answer_id:
            raise ArtifactError("BFCL v3 prompt and possible-answer IDs differ")
        if not re.fullmatch(r"simple_[0-9]+", example_id):
            raise ArtifactError("BFCL v3 simple ID differs from the official category format")

        question = prompt["question"]
        if (
            not isinstance(question, list)
            or len(question) != 1
            or not isinstance(question[0], list)
            or len(question[0]) != 1
            or not isinstance(question[0][0], Mapping)
            or set(question[0][0]) != {"content", "role"}
            or question[0][0].get("role") != "user"
        ):
            raise ArtifactError("BFCL v3 simple question must contain one user message")
        query = _required_text(
            question[0][0],
            "content",
            label="BFCL v3 simple user message",
        )
        functions = prompt["function"]
        if not isinstance(functions, list) or len(functions) != 1:
            raise ArtifactError("BFCL v3 simple prompt must contain exactly one function")
        function = functions[0]
        if not isinstance(function, Mapping) or set(function) != {
            "description",
            "name",
            "parameters",
        }:
            raise ArtifactError("BFCL v3 simple function schema differs from the audited lock")
        function_name = _required_text(
            function,
            "name",
            label="BFCL v3 simple function",
        )
        _required_text(function, "description", label="BFCL v3 simple function")
        if not isinstance(function.get("parameters"), Mapping):
            raise ArtifactError("BFCL v3 simple function parameters must be an object")

        ground_truth = answer["ground_truth"]
        if (
            not isinstance(ground_truth, list)
            or not ground_truth
            or not all(isinstance(item, Mapping) for item in ground_truth)
        ):
            raise ArtifactError("BFCL v3 simple ground truth must be a non-empty object list")
        inputs.append(
            {
                "id": example_id,
                "query": query,
                "instruction": (
                    "Return the single Python function call requested by the user, "
                    "using the provided function schema."
                ),
                "visible_metadata": {
                    "benchmark_cell": asset.cell_id,
                    "bfcl_category": "simple",
                    "function": dict(function),
                    "source_split": "test",
                },
                "visible_tool_names": [function_name],
            }
        )
        gold.append(
            {
                "id": example_id,
                "ground_truth": [dict(item) for item in ground_truth],
            }
        )
        prompt_ids.append(example_id)
        answer_ids.append(answer_id)
        function_count += len(functions)

    _require_unique_ids(prompt_ids, label="BFCL v3 simple prompt IDs")
    _require_unique_ids(answer_ids, label="BFCL v3 simple gold IDs")
    inputs.sort(key=lambda item: int(item["id"].rsplit("_", 1)[1]))
    gold.sort(key=lambda item: int(item["id"].rsplit("_", 1)[1]))

    license_text = _bfcl_text(selected["LICENSE"], label="LICENSE")
    if "Apache License" not in license_text or "Version 2.0, January 2004" not in license_text:
        raise ArtifactError("BFCL source license does not match Apache-2.0")
    pyproject_path = f"{_BFCL_V3_SOURCE_ROOT}/pyproject.toml"
    pyproject = _bfcl_text(selected[pyproject_path], label=pyproject_path)
    if (
        'name = "bfcl_eval"' not in pyproject
        or 'license = { "text" = "Apache 2.0" }' not in pyproject
    ):
        raise ArtifactError("BFCL package identity or license differs from the audited lock")

    selected_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(selected.items())
    }
    audit = [
        {
            "id": "bfcl-v3-simple-source-audit",
            "category": "simple",
            "evaluation_boundary": (
                "The official possible answers remain evaluator-only. AgentInputLab "
                "does not execute BFCL or claim an official score during preprocessing."
            ),
            "license_id": "Apache-2.0",
            "official_cli": "bfcl evaluate --test-category simple",
            "selected_source_sha256": selected_hashes,
            "source_revision": asset.source_revision,
            "source_tag": "v1.3",
        }
    ]
    return {
        "inputs.jsonl": inputs,
        "gold.jsonl": gold,
        "audit.jsonl": audit,
        "counts": {
            "audit": 1,
            "functions": function_count,
            "gold": len(gold),
            "ground_truth_candidates": sum(len(item["ground_truth"]) for item in gold),
            "inputs": len(inputs),
            "source_files": len(selected),
        },
    }


def _read_bfcl_v3_members(source: Path) -> dict[str, bytes]:
    try:
        archive = tarfile.open(source, mode="r:*")  # noqa: SIM115 - close after error remap
    except (tarfile.ReadError, tarfile.CompressionError) as exc:
        raise ArtifactError("BFCL v3 source is not a valid TAR archive") from exc
    with archive:
        members = archive.getmembers()
        if len(members) > _BFCL_V3_MAX_ARCHIVE_MEMBERS:
            raise ArtifactError("BFCL v3 archive contains too many members")
        names: set[str] = set()
        selected_members: dict[str, tarfile.TarInfo] = {}
        selected_size = 0
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ArtifactError("BFCL v3 archive contains an unsafe path")
            if member.name in names:
                raise ArtifactError("BFCL v3 archive contains duplicate member names")
            names.add(member.name)
            if not path.parts or path.parts[0] != _BFCL_V3_ARCHIVE_ROOT:
                raise ArtifactError("BFCL v3 archive root differs from the v1.3 source tag")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative not in _BFCL_V3_REQUIRED_FILES:
                continue
            if not member.isfile():
                raise ArtifactError("BFCL v3 selected archive member is not a regular file")
            if member.size > _BFCL_V3_MAX_SELECTED_MEMBER_BYTES:
                raise ArtifactError("BFCL v3 selected archive member exceeds the size limit")
            selected_size += member.size
            if selected_size > _BFCL_V3_MAX_SELECTED_BYTES:
                raise ArtifactError("BFCL v3 selected archive content exceeds the size limit")
            selected_members[relative] = member

        missing = _BFCL_V3_REQUIRED_FILES - set(selected_members)
        if missing:
            raise ArtifactError(f"BFCL v3 archive is missing required files: {sorted(missing)!r}")
        selected: dict[str, bytes] = {}
        for relative, member in sorted(selected_members.items()):
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - guarded by member.isfile()
                raise ArtifactError(f"BFCL v3 archive member cannot be read: {relative}")
            payload = handle.read(member.size + 1)
            if len(payload) != member.size:
                raise ArtifactError(f"BFCL v3 archive member is truncated: {relative}")
            selected[relative] = payload
    return selected


def _bfcl_text(payload: bytes, *, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"BFCL v3 archive member is not UTF-8: {label}") from exc


def _bfcl_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(_bfcl_text(payload, label=label).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"invalid BFCL v3 JSONL in {label} at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"BFCL v3 JSONL record in {label} must be an object")
        records.append(value)
    if not records:
        raise ArtifactError(f"BFCL v3 archive member contains no records: {label}")
    return records


def _validate_tau3_task(task: Mapping[str, Any]) -> None:
    user_scenario = task["user_scenario"]
    if not isinstance(user_scenario, Mapping):
        raise ArtifactError("tau3 task user_scenario must be an object")
    _required_text(user_scenario, "instructions", label="tau3 task user_scenario")

    required_documents = task["required_documents"]
    if not isinstance(required_documents, list) or not all(
        isinstance(item, str) and item.strip() for item in required_documents
    ):
        raise ArtifactError("tau3 task required_documents must be non-empty strings")
    if len(required_documents) != len(set(required_documents)):
        raise ArtifactError("tau3 task required_documents contain duplicates")

    user_tools = task["user_tools"]
    if not isinstance(user_tools, list) or not all(
        isinstance(item, str) and item.strip() for item in user_tools
    ):
        raise ArtifactError("tau3 task user_tools must be strings")

    initial_state = task["initial_state"]
    if initial_state is not None and not isinstance(initial_state, Mapping):
        raise ArtifactError("tau3 task initial_state must be an object or null")

    criteria = task["evaluation_criteria"]
    if not isinstance(criteria, Mapping):
        raise ArtifactError("tau3 task evaluation_criteria must be an object")
    actions = criteria.get("actions")
    if not isinstance(actions, list) or not all(isinstance(item, Mapping) for item in actions):
        raise ArtifactError("tau3 task evaluation actions must be an array of objects")
    reward_basis = criteria.get("reward_basis")
    if (
        not isinstance(reward_basis, list)
        or not reward_basis
        or not all(isinstance(item, str) and item.strip() for item in reward_basis)
    ):
        raise ArtifactError("tau3 task reward_basis must be non-empty strings")


def _read_tau3_banking_members(source: Path, *, source_revision: str) -> dict[str, bytes]:
    prefix = f"tau2-bench-{source_revision}"
    required = {
        "LICENSE",
        "pyproject.toml",
        f"{_TAU3_BANKING_DATA_ROOT}/db.json",
        f"{_TAU3_BANKING_DATA_ROOT}/tasks.json",
        f"{_TAU3_BANKING_SOURCE_ROOT}/environment.py",
        f"{_TAU3_BANKING_SOURCE_ROOT}/retrieval.py",
        f"{_TAU3_BANKING_SOURCE_ROOT}/tools.py",
        "src/tau2/data_model/tasks.py",
        "src/tau2/environment/environment.py",
        "src/tau2/evaluator/evaluator.py",
        "src/tau2/orchestrator/orchestrator.py",
        "src/tau2/runner/checkpoint.py",
        "src/tau2/user/user_simulator.py",
    }
    with _open_tau3_tar(source) as archive:
        members = archive.getmembers()
        if len(members) > _TAU3_MAX_ARCHIVE_MEMBERS:
            raise ArtifactError("tau3 archive contains too many members")
        names: set[str] = set()
        selected_members: list[tuple[str, tarfile.TarInfo]] = []
        selected_size = 0
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ArtifactError("tau3 archive contains an unsafe path")
            if member.name in names:
                raise ArtifactError("tau3 archive contains duplicate member names")
            names.add(member.name)
            if not path.parts or path.parts[0] != prefix:
                raise ArtifactError("tau3 archive root differs from the pinned source revision")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if not relative or not _tau3_selected_member(relative):
                continue
            if not member.isfile():
                raise ArtifactError("tau3 selected archive member is not a regular file")
            if member.size > _TAU3_MAX_SELECTED_MEMBER_BYTES:
                raise ArtifactError("tau3 selected archive member exceeds the size limit")
            selected_size += member.size
            if selected_size > _TAU3_MAX_SELECTED_BYTES:
                raise ArtifactError("tau3 selected archive content exceeds the size limit")
            selected_members.append((relative, member))
        if len(selected_members) > _TAU3_MAX_SELECTED_FILES:
            raise ArtifactError("tau3 archive contains too many selected files")

        selected: dict[str, bytes] = {}
        for relative, member in selected_members:
            extracted = archive.extractfile(member)
            if extracted is None:  # pragma: no cover - guarded by member.isfile()
                raise ArtifactError(f"tau3 archive member cannot be read: {relative}")
            payload = extracted.read(member.size + 1)
            if len(payload) != member.size:
                raise ArtifactError(f"tau3 archive member is truncated: {relative}")
            selected[relative] = payload
    missing = required - set(selected)
    if missing:
        raise ArtifactError(f"tau3 archive is missing required files: {sorted(missing)!r}")
    if not any(name.startswith(f"{_TAU3_BANKING_DATA_ROOT}/prompts/") for name in selected):
        raise ArtifactError("tau3 archive is missing Banking Knowledge prompt templates")
    return selected


def _open_tau3_tar(source: Path) -> tarfile.TarFile:
    try:
        return tarfile.open(source, mode="r:*")
    except (tarfile.ReadError, tarfile.CompressionError) as exc:
        raise ArtifactError("tau3 source is not a valid TAR archive") from exc


def _tau3_selected_member(name: str) -> bool:
    if name in {"LICENSE", "pyproject.toml", "src/tau2/registry.py"}:
        return True
    if name in {
        f"{_TAU3_BANKING_DATA_ROOT}/db.json",
        f"{_TAU3_BANKING_DATA_ROOT}/tasks.json",
    }:
        return True
    if name.startswith(f"{_TAU3_BANKING_DATA_ROOT}/tasks/"):
        return name.endswith(".json")
    if name.startswith(f"{_TAU3_BANKING_DATA_ROOT}/documents/"):
        return name.endswith(".json")
    if name.startswith(f"{_TAU3_BANKING_DATA_ROOT}/prompts/"):
        return name.endswith(".md")
    return _tau3_runtime_member(name)


def _tau3_runtime_member(name: str) -> bool:
    return name == "src/tau2/registry.py" or (
        name.endswith(".py")
        and any(name.startswith(prefix) for prefix in _TAU3_RUNTIME_SOURCE_PREFIXES)
    )


def _tau3_text(payload: bytes, *, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"tau3 archive member is not UTF-8: {label}") from exc


def _tau3_json_value(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(_tau3_text(payload, label=label))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"tau3 archive member is not valid JSON: {label}") from exc


def _tau3_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    value = _tau3_json_value(payload, label=label)
    if not isinstance(value, dict):
        raise ArtifactError(f"tau3 archive member must contain a JSON object: {label}")
    return value


def _read_json_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    label: str,
) -> Any:
    text = _read_text_member(archive, name)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{label} is not valid JSON") from exc


def _validate_named_zip(
    archive: zipfile.ZipFile,
    *,
    allowed_files: set[str],
    required_files: set[str],
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    label: str,
) -> None:
    infos = archive.infolist()
    if len(infos) > max_members:
        raise ArtifactError(f"{label} archive contains too many members")
    names: set[str] = set()
    files: set[str] = set()
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ArtifactError(f"{label} archive contains an unsafe path")
        if info.filename in names:
            raise ArtifactError(f"{label} archive contains duplicate member names")
        names.add(info.filename)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ArtifactError(f"{label} archive contains a symbolic link")
        if info.is_dir():
            continue
        files.add(info.filename)
        if info.filename not in allowed_files:
            raise ArtifactError(f"{label} archive contains an unexpected file: {info.filename}")
        if info.file_size > max_member_bytes:
            raise ArtifactError(f"{label} archive member exceeds the size limit")
        total += info.file_size
        if info.file_size / max(info.compress_size, 1) > _MAX_COMPRESSION_RATIO:
            raise ArtifactError(f"{label} archive member exceeds the compression-ratio limit")
    if total > max_uncompressed_bytes:
        raise ArtifactError(f"{label} archive exceeds the uncompressed size limit")
    if not files >= required_files:
        raise ArtifactError(f"{label} archive is missing required files")


def _validate_named_tar(
    archive: tarfile.TarFile,
    *,
    allowed_files: set[str],
    required_files: set[str],
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    label: str,
) -> None:
    members = archive.getmembers()
    if len(members) > max_members:
        raise ArtifactError(f"{label} archive contains too many members")
    names: set[str] = set()
    files: set[str] = set()
    total = 0
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or "\\" in member.name:
            raise ArtifactError(f"{label} archive contains an unsafe path")
        if member.name in names:
            raise ArtifactError(f"{label} archive contains duplicate member names")
        names.add(member.name)
        if member.isdir():
            continue
        if not member.isfile():
            raise ArtifactError(f"{label} archive contains a non-regular member")
        files.add(member.name)
        if member.name not in allowed_files:
            raise ArtifactError(f"{label} archive contains an unexpected file: {member.name}")
        if member.size > max_member_bytes:
            raise ArtifactError(f"{label} archive member exceeds the size limit")
        total += member.size
    if total > max_uncompressed_bytes:
        raise ArtifactError(f"{label} archive exceeds the uncompressed size limit")
    if not files >= required_files:
        raise ArtifactError(f"{label} archive is missing required files")


def _normalize_clariq_tsv(
    source: Path,
    asset: BenchmarkSmokeAsset,
) -> dict[str, Any]:
    try:
        handle = source.open(encoding="utf-8", newline="")
    except UnicodeError as exc:  # pragma: no cover - open is lazy on standard Python
        raise ArtifactError("ClariQ source is not valid UTF-8") from exc
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _CLARIQ_FIELDS:
            raise ArtifactError("ClariQ TSV schema does not match the audited dev schema")
        try:
            rows = [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            raise ArtifactError("ClariQ source is not valid UTF-8") from exc
    if not rows:
        raise ArtifactError("ClariQ TSV contains no records")

    normalized: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    topics: set[str] = set()
    for row in rows:
        for field in (
            "topic_id",
            "initial_request",
            "topic_desc",
            "clarification_need",
            "facet_id",
            "facet_desc",
            "question_id",
        ):
            _required_text(row, field, label="ClariQ row")
        reference_question = _optional_text(row, "question", label="ClariQ row")
        reference_answer = _optional_text(row, "answer", label="ClariQ row")
        topic_id = row["topic_id"]
        topics.add(topic_id)
        try:
            clarification_need = int(row["clarification_need"])
        except ValueError as exc:
            raise ArtifactError("ClariQ clarification_need must be an integer") from exc
        if clarification_need not in {1, 2, 3, 4}:
            raise ArtifactError("ClariQ clarification_need must be in [1, 4]")
        row_hash = hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()[:16]
        example_id = f"{topic_id}:{row['facet_id']}:{row['question_id']}:{row_hash}"
        visible = {
            "id": example_id,
            "query": row["initial_request"],
            "instruction": "Decide whether clarification is needed and, if so, ask one question.",
            "visible_metadata": {
                "benchmark_cell": asset.cell_id,
                "source_topic_id": topic_id,
            },
        }
        evaluator = {
            "id": example_id,
            "topic_description": row["topic_desc"],
            "clarification_need": clarification_need,
            "facet_id": row["facet_id"],
            "facet_description": row["facet_desc"],
            "question_id": row["question_id"],
            "reference_question": reference_question,
            "reference_answer": reference_answer,
        }
        normalized.append((example_id, visible, evaluator))
    normalized.sort(key=lambda item: item[0])
    ids = [item[0] for item in normalized]
    if len(set(ids)) != len(ids):
        raise ArtifactError("ClariQ normalization produced duplicate example IDs")
    return {
        "inputs.jsonl": [item[1] for item in normalized],
        "gold.jsonl": [item[2] for item in normalized],
        "counts": {
            "inputs": len(normalized),
            "gold": len(normalized),
            "topics": len(topics),
        },
    }


def _validate_zip(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > _MAX_ARCHIVE_MEMBERS:
        raise ArtifactError("benchmark archive contains too many members")
    names: set[str] = set()
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ArtifactError("benchmark archive contains an unsafe path")
        if info.filename in names:
            raise ArtifactError("benchmark archive contains duplicate member names")
        names.add(info.filename)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ArtifactError("benchmark archive contains a symbolic link")
        if info.is_dir():
            continue
        if info.filename not in _BEIR_MEMBERS:
            raise ArtifactError(f"benchmark archive contains an unexpected file: {info.filename}")
        if info.file_size > _MAX_MEMBER_BYTES:
            raise ArtifactError("benchmark archive member exceeds the size limit")
        total += info.file_size
        denominator = max(info.compress_size, 1)
        if info.file_size / denominator > _MAX_COMPRESSION_RATIO:
            raise ArtifactError("benchmark archive member exceeds the compression-ratio limit")
    if total > _MAX_UNCOMPRESSED_BYTES:
        raise ArtifactError("benchmark archive exceeds the uncompressed size limit")
    if not names >= _BEIR_MEMBERS:
        raise ArtifactError("benchmark archive is missing required BEIR files")


def _read_text_member(archive: zipfile.ZipFile, name: str) -> str:
    try:
        return archive.read(name).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"benchmark archive member is missing or not UTF-8: {name}") from exc


def _read_jsonl_member(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    text = _read_text_member(archive, name)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"invalid JSONL in {name} at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ArtifactError(f"JSONL record in {name} must be an object")
        records.append(value)
    if not records:
        raise ArtifactError(f"benchmark archive member contains no records: {name}")
    return records


def _records_by_id(
    records: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = _required_text(record, "_id", label=label)
        if record_id in indexed:
            raise ArtifactError(f"{label} contains duplicate IDs")
        indexed[record_id] = record
    return indexed


def _required_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{label} requires non-empty string field {key!r}")
    return value


def _optional_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = record.get(key, "")
    if not isinstance(value, str):
        raise ArtifactError(f"{label} field {key!r} must be a string")
    return value


def _jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return ("\n".join(canonical_json(record) for record in records) + "\n").encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _require_expected_counts(asset: BenchmarkSmokeAsset, counts: Mapping[str, int]) -> None:
    if dict(asset.expected_counts) != dict(counts):
        raise ArtifactError(
            f"prepared count mismatch for {asset.asset_id}: expected "
            f"{dict(asset.expected_counts)!r}, got {dict(counts)!r}"
        )


def _load_prepared(destination: Path, *, expected_asset: BenchmarkSmokeAsset) -> PreparedBenchmark:
    metadata_path = destination / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"prepared benchmark metadata is invalid: {destination}") from exc
    if metadata.get("schema_version") != 1:
        raise ArtifactError("prepared benchmark metadata has an unsupported schema")
    expected_identity = {
        "asset_id": expected_asset.asset_id,
        "cell_id": expected_asset.cell_id,
        "source_revision": expected_asset.source_revision,
        "source_sha256": expected_asset.sha256,
        "source_size_bytes": expected_asset.size_bytes,
        "processor": expected_asset.processor,
        "processor_revision": _PROCESSOR_REVISIONS[expected_asset.processor],
        "gold_separated": True,
    }
    if expected_asset.processor not in {"beir_zip_v1", "clariq_tsv_v1"}:
        expected_identity["split"] = _processor_split(expected_asset.processor)
    for key, expected in expected_identity.items():
        if metadata.get(key) != expected:
            raise ArtifactError(f"prepared benchmark identity mismatch for {key}")
    raw_counts = metadata.get("counts")
    raw_files = metadata.get("files")
    if not isinstance(raw_counts, dict) or not isinstance(raw_files, dict):
        raise ArtifactError("prepared benchmark metadata is missing counts or files")
    if any(
        not isinstance(key, str)
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for key, value in raw_counts.items()
    ):
        raise ArtifactError("prepared benchmark counts must be non-negative integers")
    counts = dict(raw_counts)
    files = {str(key): str(value) for key, value in raw_files.items()}
    _require_expected_counts(expected_asset, counts)
    required_files = _PROCESSOR_FILES[expected_asset.processor]
    if set(files) != required_files:
        raise ArtifactError("prepared benchmark metadata has an incomplete file set")
    if expected_asset.prepared_sha256 and files != dict(expected_asset.prepared_sha256):
        raise ArtifactError("prepared benchmark hashes do not match the audited lock")
    for name, expected_hash in files.items():
        if name not in {"inputs.jsonl", "gold.jsonl", "corpus.jsonl", "audit.jsonl"}:
            raise ArtifactError("prepared benchmark metadata names an unexpected file")
        path = destination / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ArtifactError(f"prepared benchmark cache verification failed: {name}")
    return PreparedBenchmark(
        asset_id=expected_asset.asset_id,
        processor=expected_asset.processor,
        processor_revision=_PROCESSOR_REVISIONS[expected_asset.processor],
        source_sha256=expected_asset.sha256,
        root=destination,
        counts=counts,
        file_sha256=files,
        cell_id=expected_asset.cell_id,
        source_revision=expected_asset.source_revision,
        split=_processor_split(expected_asset.processor),
    )


def _processor_split(processor: str) -> str:
    if processor == "bfcl_v3_simple_tar_v1":
        return "test"
    if processor == "tau3_banking_tar_v1":
        return "base"
    if processor in {"musique_zip_v1", "two_wiki_zip_v1", "qasper_tgz_v1"}:
        return "dev"
    if processor == "hotpotqa_parquet_v1":
        return "validation"
    return "test"
