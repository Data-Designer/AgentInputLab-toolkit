"""LongBench v1 data adapter with separately mounted, checksum-locked scoring.

Data and official scorer code are never distributed with the toolkit. The
adapter renders official templates and exposes references only to scoring.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from collections.abc import Mapping
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

from agentinputlab.contracts import (
    ArtifactError,
    ContextBlock,
    ExternalDependencyError,
    FrozenAgentState,
    ValidationError,
)

from .base import BenchmarkExample

LONG_BENCH_SOURCE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONG_BENCH_DATA_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
LONG_BENCH_SCORER_SHA256 = "e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323"
LONG_BENCH_PROMPTS_SHA256 = "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"
_SUBSETS = {
    "longbench.qasper": ("qasper", "qa_f1_score", "answer_f1", 200),
    "longbench.multifieldqa_en": ("multifieldqa_en", "qa_f1_score", "answer_f1", 150),
    "longbench.govreport": ("gov_report", "rouge_score", "rouge", 200),
    "longbench.qmsum": ("qmsum", "rouge_score", "rouge", 200),
    "longbench.passage_retrieval_en": (
        "passage_retrieval_en",
        "retrieval_score",
        "passage_id_accuracy",
        200,
    ),
    "longbench.repobench_p": ("repobench-p", "code_sim_score", "code_metric", 500),
}


def _checked(path: Path, expected: str) -> bytes:
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise ArtifactError(f"LongBench source checksum differs: {path.name}")
    return value


class LongBenchAdapter:
    pairing_mode = "stateless_score_twice"
    split = "test"

    def __init__(
        self,
        *,
        benchmark_id: str,
        data_zip: str | Path,
        scorer_path: str | Path,
        prompts_path: str | Path,
    ):
        if benchmark_id not in _SUBSETS:
            raise ValidationError("unknown LongBench subset")
        self.benchmark_id = benchmark_id
        self.dataset, self._metric_function, self._metric_name, expected_count = _SUBSETS[
            benchmark_id
        ]
        _checked(Path(data_zip), LONG_BENCH_DATA_SHA256)
        _checked(Path(scorer_path), LONG_BENCH_SCORER_SHA256)
        prompts = json.loads(_checked(Path(prompts_path), LONG_BENCH_PROMPTS_SHA256))
        self.template = prompts[self.dataset]
        spec = importlib.util.spec_from_file_location(
            "agentinputlab_external_longbench_metrics", scorer_path,
            loader=SourceFileLoader("agentinputlab_external_longbench_metrics", str(scorer_path)),
        )
        if spec is None or spec.loader is None:
            raise ArtifactError("cannot load pinned LongBench scorer")
        self._scorer = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self._scorer)
        except ModuleNotFoundError as exc:
            raise ExternalDependencyError(
                "install agentinputlab[longbench] for the pinned native scorer"
            ) from exc
        examples = []
        with zipfile.ZipFile(data_zip) as archive:
            matches = [
                n for n in archive.namelist() if n.rsplit("/", 1)[-1] == self.dataset + ".jsonl"
            ]
            if len(matches) != 1:
                raise ArtifactError("LongBench archive subset is absent or duplicated")
            if archive.getinfo(matches[0]).file_size > 128 * 1024 * 1024:
                raise ArtifactError("LongBench subset exceeds size limit")
            rows = [
                json.loads(line) for line in archive.read(matches[0]).splitlines() if line.strip()
            ]
        if len(rows) != expected_count:
            raise ArtifactError("LongBench official subset count differs")
        for row in rows:
            if (
                row.get("dataset") != self.dataset
                or not isinstance(row.get("input"), str)
                or not isinstance(row.get("context"), str)
            ):
                raise ArtifactError("LongBench example fields differ from the official schema")
            answers = row.get("answers")
            if (
                not isinstance(answers, list)
                or not answers
                or not all(isinstance(a, str) for a in answers)
            ):
                raise ArtifactError("LongBench answer annotations are invalid")
            query = row["input"] or (
                "Complete the next line of code."
                if self.dataset == "repobench-p"
                else "Summarize the supplied report."
            )
            examples.append(
                BenchmarkExample(
                    example_id=str(row["_id"]),
                    query=query,
                    visible_context=(
                        ContextBlock(
                            text=row["context"], block_id="long_context", source=self.dataset
                        ),
                    ),
                    visible_metadata={"benchmark_id": benchmark_id, "source_split": "test"},
                    evaluator_gold={
                        "answers": tuple(answers),
                        "all_classes": row.get("all_classes") or (),
                        "native_input": row["input"],
                    },
                )
            )
        if len({e.example_id for e in examples}) != len(examples):
            raise ArtifactError("LongBench example IDs repeat")
        self._examples = tuple(examples)
        self._by_id = {e.example_id: e for e in examples}

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != "test":
            raise ValidationError("LongBench v1 exposes test only")
        return self._examples

    def example(self, split: str, example_id: str) -> BenchmarkExample:
        """Look up the exact native test ID, without substituting a split or case."""

        self.examples(split)
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValidationError("LongBench example_id must be non-empty text")
        try:
            return self._by_id[example_id]
        except KeyError as exc:
            raise ValidationError(f"unknown LongBench example ID: {example_id}") from exc

    def render(self, state: FrozenAgentState) -> str:
        # Summarization/code templates omit {input}; the short state query is
        # descriptive there and the baseline still renders the native template.
        prompt = self.template.format(
            context="\n\n".join(b.text for b in state.context), input=state.query
        )
        if state.instruction:
            prompt += "\n\n" + state.instruction
        return prompt

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if self._by_id.get(example.example_id) is not example:
            raise ValidationError("example belongs to a different LongBench adapter")
        if not isinstance(prediction, str):
            raise ValidationError("LongBench prediction must be text")
        function = getattr(self._scorer, self._metric_function)
        value = max(
            float(function(prediction, a, all_classes=example.evaluator_gold["all_classes"]))
            for a in example.evaluator_gold["answers"]
        )
        result = {self._metric_name: value}
        if self.dataset == "repobench-p":
            line = next(
                (
                    line
                    for line in prediction.lstrip("\n").split("\n")
                    if "`" not in line and "#" not in line and "//" not in line
                ),
                "",
            )
            result["exact_match"] = float(line in example.evaluator_gold["answers"])
        return result

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "data_revision": LONG_BENCH_DATA_REVISION,
            "data_sha256": LONG_BENCH_DATA_SHA256,
            "source_revision": LONG_BENCH_SOURCE_REVISION,
            "scorer_sha256": LONG_BENCH_SCORER_SHA256,
            "prompts_sha256": LONG_BENCH_PROMPTS_SHA256,
            "official_metric": self._metric_name,
            "score_scale": "0_to_1",
            "extra_code_exact_match": "first_noncomment_line_exact_equality",
        }
