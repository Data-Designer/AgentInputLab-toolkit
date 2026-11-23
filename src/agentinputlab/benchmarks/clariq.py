"""Small source-verified ClariQ dev adapter and native RQ1/RQ2 scoring.

No upstream rows or unlicensed scoring code are redistributed. The caller mounts
the three pinned files and explicitly reviews their local-use terms. Document
utility (RQ3) needs a different bundle and is not approximated here.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from agentinputlab.artifacts import sha256_file
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.contracts.errors import ArtifactError, ValidationError

from .base import BenchmarkExample

CLARIQ_REVISION = "46885a544581a0af8aff0681d29e4971807e2912"
CLARIQ_FILES = MappingProxyType({
    "data/dev.tsv": "68d2a5f87eab73721979b5f45f64099a9b2f080db1d0ce4b979d9daa4249906e",
    "data/question_bank.tsv": "266f501bdc31afbc8a763685ea0de82949137a2779241b325f5ee03026f42f1b",
    "src/clariq_eval_tool.py": "6b3258c4d09ee9d0aa6901a0d3210a5535bc3f75049998b1a4b1d00be84ae210",
})


@dataclass(frozen=True, slots=True)
class ClariQPrediction:
    clarification_need: int
    ranked_question_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.clarification_need) is not int or self.clarification_need not in range(1, 5):
            raise ValidationError("ClariQ clarification_need must be an integer in [1, 4]")
        if isinstance(self.ranked_question_ids, (str, bytes)):
            raise ValidationError("ClariQ ranked question IDs must be a sequence, not text")
        ids = tuple(self.ranked_question_ids)
        if not ids or any(not isinstance(value, str) or not value for value in ids):
            raise ValidationError("ClariQ requires non-empty ranked question IDs")
        if len(set(ids)) != len(ids):
            raise ValidationError("ClariQ ranked question IDs must be unique")
        object.__setattr__(self, "ranked_question_ids", ids)


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _topic_queries(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    topics: dict[str, str] = {}
    labels: dict[str, str] = {}
    for row in rows:
        topic, query, label = row["topic_id"], row["initial_request"], row["clarification_need"]
        if not topic.isdigit() or not query.strip() or label not in {"1", "2", "3", "4"}:
            raise ArtifactError("invalid ClariQ topic, initial request or clarification label")
        if topic in topics and (topics[topic] != query or labels[topic] != label):
            raise ArtifactError("conflicting rows for the same ClariQ topic")
        topics[topic], labels[topic] = query, label
    return topics


class ClariQDevBundle:
    """One initial request per topic; facet intent and reference answers stay private.

    BM25 question ranking is an explicit toolkit baseline, not an upstream model.
    The complete 3,941-question bank is small; no document corpus is loaded.
    """

    benchmark_id = "clariq.main"
    pairing_mode = "stateless_score_twice"

    def __init__(self, source_root: str | Path, *, accept_license_review: bool = False):
        if accept_license_review is not True:
            raise ArtifactError("ClariQ NOASSERTION source requires explicit local-use terms review")
        self.source_root = Path(source_root).resolve()
        self._verify()
        self._fields, self._rows = _read_tsv(self.source_root / "data/dev.tsv")
        self._topics = _topic_queries(self._rows)
        _, bank = _read_tsv(self.source_root / "data/question_bank.tsv")
        self._bank = {row["question_id"]: row["question"] for row in bank}
        if len(self._rows) != 2313 or len(self._topics) != 50 or len(self._bank) != 3941:
            raise ArtifactError("ClariQ pinned dev or question-bank counts differ")
        self._retriever = InMemoryBM25Retriever(documents=tuple(
            Document(doc_id=key, text=value, corpus_revision=CLARIQ_REVISION)
            for key, value in sorted(self._bank.items())
        ))

    def _verify(self) -> None:
        for name, expected in CLARIQ_FILES.items():
            path = self.source_root / name
            if path.is_symlink() or not path.is_file() or path.resolve().parent != path.parent:
                raise ArtifactError(f"ClariQ source is missing or linked: {name}")
            if sha256_file(path) != expected:
                raise ArtifactError(f"ClariQ source checksum differs: {name}")

    def examples(self, split: str = "dev") -> tuple[BenchmarkExample, ...]:
        if split != "dev":
            raise ValidationError("this pinned ClariQ bundle exposes only dev")
        return tuple(BenchmarkExample(
            example_id=topic,
            query=query,
            instruction=("Predict how much clarification this initial user request needs. "
                         "Return exactly one JSON object with key clarification_need: "
                         "1 means no clarification, 2/3/4 mean increasing need. "
                         "Do not answer the request or add any other keys."),
            visible_metadata={"benchmark_cell": self.benchmark_id, "source_split": "dev",
                              "source_revision": CLARIQ_REVISION},
        ) for topic, query in sorted(self._topics.items()))

    def rank_questions(self, query: str, *, top_k: int = 30) -> tuple[str, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("ClariQ question ranking requires a non-empty query")
        if type(top_k) is not int or not 1 <= top_k <= len(self._bank):
            raise ValidationError("ClariQ top_k must be a positive bank-bounded integer")
        return tuple(result.document.doc_id for result in self._retriever.retrieve(query, top_k=top_k))

    @property
    def question_retriever(self) -> InMemoryBM25Retriever:
        """Visible question-bank-only retriever; never facet descriptions or answers."""
        return self._retriever

    def fingerprint(self) -> dict:
        return {"benchmark": self.benchmark_id, "revision": CLARIQ_REVISION,
                "source_sha256": dict(CLARIQ_FILES), "question_ranker": self._retriever.fingerprint(),
                "scoring": "native_RQ1_weighted_PRF_and_RQ2_Recall5_10_20_30",
                "document_retrieval_utility_evaluated": False,
                "license_id": "NOASSERTION", "source_redistributed": False}

    def evaluate(
        self, predictions: Mapping[str, ClariQPrediction], *, topic_ids: Sequence[str],
        output: str | Path,
    ) -> dict:
        """Run unmodified native metrics on all rows of a declared topic subset.

        Missing predictions remain in the selected denominator and use native
        missing-run semantics. A partial subset is never reported as full dev.
        All run files, stdout and per-topic scores are retained under ``output``.
        This method is sequential because the upstream scorer prints to stdout.
        """
        selected = tuple(topic_ids)
        if not selected or len(set(selected)) != len(selected) or set(selected) - self._topics.keys():
            raise ValidationError("ClariQ topic_ids must be a unique non-empty known subset")
        if set(predictions) - set(selected):
            raise ValidationError("ClariQ prediction is outside the declared topic subset")
        for prediction in predictions.values():
            if not isinstance(prediction, ClariQPrediction):
                raise ValidationError("ClariQ predictions must be typed ClariQPrediction objects")
            if set(prediction.ranked_question_ids) - self._bank.keys():
                raise ValidationError("ClariQ prediction contains a question outside the source bank")
        self._verify()
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "dev.tsv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(row for row in self._rows if row["topic_id"] in selected)
        # The native scorer uses an empty-file pandas parser that raises. Sentinels
        # are deliberately outside the declared source topics, so native lookup
        # gives every missing prediction its original label 0 / recall 0 semantics.
        need_lines, question_lines = ["-1 0\n"], ["-1 Q0 Q00001 1 1 missing\n"]
        for topic, prediction in sorted(predictions.items()):
            need_lines.append(f"{topic} {prediction.clarification_need}\n")
            for rank, question in enumerate(prediction.ranked_question_ids, 1):
                question_lines.append(f"{topic} Q0 {question} {rank} {1.0 / rank:.17g} toolkit\n")
        for name, lines in (("need.run", need_lines), ("questions.run", question_lines)):
            with (directory / name).open("x", encoding="utf-8") as handle:
                handle.writelines(lines)
        scorer = self.source_root / "src/clariq_eval_tool.py"
        spec = importlib.util.spec_from_file_location("_ail_pinned_clariq_evaluator", scorer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            module.evaluate_clarification_need("dev", str(directory), str(directory / "need.run"), "", False)
            module.evaluate_question_relevance("dev", str(directory), str(directory / "questions.run"),
                                               str(directory / "question_scores.json"), False)
        metrics = {}
        names = {"Precision": "weighted_precision", "Recall": "weighted_recall", "F1": "weighted_f1"}
        for line in stdout.getvalue().splitlines():
            name, _, value = line.partition(":")
            if name.strip() in names or name.strip() in {"Recall5", "Recall10", "Recall20", "Recall30"}:
                metrics[names.get(name.strip(), name.strip())] = float(value.strip())
        if len(metrics) != 7:
            raise ArtifactError("ClariQ native evaluator did not emit the seven expected metrics")
        result = {"metrics": metrics, "native_stdout": stdout.getvalue(),
                  "topic_ids": list(selected), "predicted_topics": len(predictions),
                  "missing_predictions": sorted(set(selected) - predictions.keys()),
                  "evaluation_scope": "full_dev" if len(selected) == 50 else "fixed_dev_compatibility_subset",
                  "native_scorer_sha256": CLARIQ_FILES["src/clariq_eval_tool.py"],
                  "document_retrieval_utility_evaluated": False, "source_fingerprint": self.fingerprint()}
        with (directory / "native-evaluation.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        return result
