"""Pinned native SmartSearch curriculum preparation, separate from inference.

These APIs do not train a model or manufacture process rewards. They consume
recorded training-only rollouts and call the original redundancy/SFT/DPO/prefix
converters. The complete SFT/DPO/GRPO training recipe remains a separate gate.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

from .smartsearch import SMARTSEARCH_UPSTREAM_COMMIT

_FILES = {
    "redundancy": "scripts/data_construction/detect_redundancy.py",
    "sft": "scripts/data_construction/transfer_sft.py",
    "dpo": "scripts/data_construction/transfer_dpo.py",
    "prefix": "scripts/data_construction/transfer_regenerate.py",
    "reward": "scripts/data_construction/process_reward.py",
    "refinement": "scripts/data_construction/query_refinement.py",
}
_PREPARATION_KEYS = ("redundancy", "sft", "dpo", "prefix")


@dataclass(frozen=True, slots=True)
class SmartSearchTrainingCase:
    case_id: str
    question: str
    split: str
    source_reference: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValidationError("SmartSearch training case requires a non-empty ID")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValidationError("SmartSearch training case requires a non-empty question")
        if self.split not in {"train", "validation", "test"}:
            raise ValidationError("SmartSearch training case split is invalid")
        try:
            object.__setattr__(self, "source_reference", freeze_mapping(self.source_reference))
        except (TypeError, ValueError) as exc:
            raise ValidationError("SmartSearch case provenance is not canonical JSON") from exc


def _question_hash(question: str) -> str:
    # This additional overlap check is not a model-input transformation.
    return content_hash(" ".join(question.split()).casefold())


def validate_smartsearch_training_splits(
    train: Sequence[SmartSearchTrainingCase],
    validation: Sequence[SmartSearchTrainingCase],
    test: Sequence[SmartSearchTrainingCase] = (),
) -> Mapping[str, Any]:
    if not train or not validation:
        raise ValidationError("SmartSearch curriculum needs non-empty training and validation collections")
    audit = {}
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for name, cases in (("train", train), ("validation", validation), ("test", test)):
        if any(not isinstance(case, SmartSearchTrainingCase) or case.split != name for case in cases):
            raise ValidationError("SmartSearch curriculum case split label mismatch")
        ids = tuple(case.case_id for case in cases)
        hashes = tuple(_question_hash(case.question) for case in cases)
        if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
            raise ValidationError("SmartSearch curriculum split contains duplicate IDs/questions")
        if set(ids) & seen_ids or set(hashes) & seen_questions:
            raise ValidationError("SmartSearch fitting, validation and test collections overlap")
        seen_ids.update(ids)
        seen_questions.update(hashes)
        audit[name] = {"case_ids": ids, "question_hashes": hashes,
                       "case_hashes": tuple(content_hash(case) for case in cases)}
    return freeze_mapping(audit)


def verify_smartsearch_preparation_source(
    checkout: str | Path, *, include_teacher: bool = False,
) -> Mapping[str, Any]:
    """Bind the exact upstream Git objects and their on-disk converter bytes."""
    root = Path(checkout)
    if root.is_symlink() or not root.is_dir():
        raise ExternalDependencyError("SmartSearch preparation requires a regular pinned Git checkout")
    root = root.resolve()

    def git(*args: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
                 "-C", str(root), *args], capture_output=True, check=True, timeout=30,
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C",
                     "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                     "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExternalDependencyError("SmartSearch preparation source Git verification failed") from exc
        return result.stdout

    if git("rev-parse", "HEAD").decode().strip() != SMARTSEARCH_UPSTREAM_COMMIT:
        raise ExternalDependencyError("SmartSearch preparation source revision changed")
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ExternalDependencyError("SmartSearch preparation checkout is not the repository root")
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ExternalDependencyError("SmartSearch preparation requires an unchanged source checkout")
    files = {}
    keys = tuple(_FILES) if include_teacher else _PREPARATION_KEYS
    for relative in (_FILES[key] for key in keys):
        path = root
        for component in Path(relative).parts:
            path = path / component
            if path.is_symlink():
                raise ExternalDependencyError("SmartSearch preparation source contains a symlink")
        if not path.is_file():
            raise ExternalDependencyError(f"SmartSearch native preparation file is missing: {relative}")
        actual = path.read_bytes()
        expected = git("show", f"{SMARTSEARCH_UPSTREAM_COMMIT}:{relative}")
        if actual != expected:
            raise ExternalDependencyError(f"SmartSearch native preparation file changed: {relative}")
        files[relative] = hashlib.sha256(actual).hexdigest()
    return freeze_mapping({"source_commit": SMARTSEARCH_UPSTREAM_COMMIT, "files": files,
                           "root": str(root), "preparation_math_modified": False})


def _native_converter(root: Path, key: str, source: Mapping[str, Any]) -> Any:
    path = root / _FILES[key]
    source_bytes = path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source["files"][_FILES[key]]:
        raise ExternalDependencyError("SmartSearch preparation source changed before import")
    name = f"_agentinputlab_smartsearch_{key}_{source['files'][_FILES[key]][:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExternalDependencyError("SmartSearch native converter could not be imported")
    module = importlib.util.module_from_spec(spec)
    try:
        # Compile the verified bytes, never an unbound pre-existing .pyc cache.
        exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    except ImportError as exc:
        raise ExternalDependencyError("Install agentinputlab[smartsearch-prepare] in the preparation worker") from exc
    class_name = {"redundancy": "Redundancy", "reward": "SearchEvaluation",
                  "refinement": "Rewrite"}.get(key, "Transfer")
    constructor = getattr(module, class_name, None)
    if not callable(constructor):
        raise ExternalDependencyError("SmartSearch native converter class signature changed")
    return constructor("controlled-provider-no-network") if key in {"reward", "refinement"} else constructor()


def _scored_rows(
    value: Sequence[Mapping[str, Any]], allowed_questions: set[str], max_rows: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, (tuple, list)) or len(value) > max_rows:
        raise ValidationError("SmartSearch recorded rollout collection exceeds its explicit bound")
    rows = []
    for row in value:
        if not isinstance(row, Mapping) or row.get("question") not in allowed_questions:
            raise ValidationError("SmartSearch curriculum rollout is outside the allowed training questions")
        if not isinstance(row.get("response"), str) or not isinstance(row.get("process_scores"), (list, tuple)):
            raise ValidationError("SmartSearch curriculum requires recorded response and process scores")
        metric = row.get("outcome_score")
        score = metric.get("f1") if isinstance(metric, Mapping) else None
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValidationError("SmartSearch curriculum requires a measured finite outcome F1")
        for process in row["process_scores"]:
            if not isinstance(process, Mapping) or not isinstance(process.get("score"), str) or process["score"] not in {"", "0", "1"}:
                raise ValidationError("SmartSearch native process-score representation is invalid")
        try:
            rows.append(json.loads(json.dumps(to_jsonable(row), allow_nan=False)))
        except (TypeError, ValueError) as exc:
            raise ValidationError("SmartSearch scored rollout is not canonical JSON") from exc
    return rows


@dataclass(frozen=True, slots=True)
class SmartSearchCurriculumArtifact:
    root: str
    artifact_id: str
    sft_count: int
    dpo_count: int
    prefix_count: int
    source_fingerprint_sha256: str
    split_audit_sha256: str
    output_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_hashes", freeze_mapping(self.output_hashes))


@dataclass(frozen=True, slots=True)
class SmartSearchCurriculumPreparation:
    max_rollout_rows: int = 1000
    max_input_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.max_rollout_rows) is not int or self.max_rollout_rows < 1 or (
            type(self.max_input_bytes) is not int or self.max_input_bytes < 1
        ):
            raise ValidationError("SmartSearch curriculum limits must be positive integers")

    def prepare(
        self, *, checkout: str | Path, output_root: str | Path,
        train: Sequence[SmartSearchTrainingCase], validation: Sequence[SmartSearchTrainingCase],
        initial_scored_rollouts: Sequence[Mapping[str, Any]],
        regenerated_scored_rollouts: Sequence[Mapping[str, Any]],
        test: Sequence[SmartSearchTrainingCase] = (),
        rollout_provenance: Mapping[str, Any],
    ) -> SmartSearchCurriculumArtifact:
        audit = validate_smartsearch_training_splits(train, validation, test)
        if not isinstance(rollout_provenance, Mapping) or not rollout_provenance:
            raise ValidationError("SmartSearch recorded rollouts require explicit producer provenance")
        allowed = {case.question for case in train}
        initial = _scored_rows(initial_scored_rollouts, allowed, self.max_rollout_rows)
        regenerated = _scored_rows(regenerated_scored_rollouts, allowed, self.max_rollout_rows)
        if not initial:
            raise ValidationError("SmartSearch curriculum needs initial recorded training rollouts")
        raw_inputs = {"initial": initial, "regenerated": regenerated,
                      "producer_provenance": to_jsonable(rollout_provenance)}
        encoded = json.dumps(raw_inputs, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > self.max_input_bytes:
            raise ValidationError("SmartSearch curriculum input exceeds its explicit byte bound")
        source = verify_smartsearch_preparation_source(checkout)
        source_root = Path(source["root"])
        root = Path(output_root)
        if root.is_symlink() or root.exists():
            raise ValidationError("SmartSearch curriculum output must be a new directory")
        root = root.resolve(strict=False)
        if root.is_relative_to(source_root) or source_root.is_relative_to(root):
            raise ValidationError("SmartSearch curriculum outputs must not overlap source checkout")
        root.mkdir(parents=True, exist_ok=False)

        def save(name: str, value: Any) -> None:
            with (root / name).open("x", encoding="utf-8") as stream:
                json.dump(to_jsonable(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)

        save("claim.json", {"source": source, "split_audit": audit, "input_hash": content_hash(raw_inputs),
                            "rollout_provenance": rollout_provenance, "model_calls_performed": 0})
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                redundancy = _native_converter(source_root, "redundancy", source)
                initial_checked = redundancy.detect_redundancy(initial)
                regenerated_checked = redundancy.detect_redundancy(regenerated)
                sft = _native_converter(source_root, "sft", source).transfer(initial_checked)
                dpo = _native_converter(source_root, "dpo", source).transfer(initial_checked, regenerated_checked)
                prefix = _native_converter(source_root, "prefix", source).transfer(initial_checked)
            for values in (sft, dpo, prefix):
                if not isinstance(values, list):
                    raise ExternalDependencyError("SmartSearch native converter returned a non-list")
            for relative, digest in source["files"].items():
                if hashlib.sha256((source_root / relative).read_bytes()).hexdigest() != digest:
                    raise ExternalDependencyError("SmartSearch preparation source changed during execution")
            outputs = {"native-initial-scored.json": initial_checked,
                       "native-regenerated-scored.json": regenerated_checked,
                       "sft.json": sft, "dpo.json": dpo, "prefix.json": prefix}
            for name, value in outputs.items():
                save(name, value)
            hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in outputs}
            manifest = {"schema": "agentinputlab.smartsearch.native-curriculum-data.v1",
                        "source": source, "split_audit": audit, "outputs": hashes,
                        "sft_count": len(sft), "dpo_count": len(dpo), "prefix_count": len(prefix),
                        "initial_rollout_count": len(initial), "regenerated_rollout_count": len(regenerated),
                        "input_hash": content_hash(raw_inputs), "rollout_provenance": rollout_provenance,
                        "model_calls_performed": 0, "neural_training_performed": False,
                        "native_selection_thresholds_changed": False, "empty_selection_is_not_training_success": True}
            artifact_id = content_hash(manifest)
            save("manifest.json", {**manifest, "artifact_id": artifact_id})
            return SmartSearchCurriculumArtifact(
                str(root), artifact_id, len(sft), len(dpo), len(prefix), content_hash(source),
                content_hash(audit), hashes,
            )
        except Exception as exc:
            save("failed.json", {"error_type": type(exc).__name__, "error": str(exc),
                                 "model_calls_performed": 0, "neural_training_performed": False})
            raise
