#!/usr/bin/env python3
"""Fail-closed Q-RAG frozen selector/reader worker.

No upstream source or model asset is distributed here.  The caller supplies a
read-only checkout, checkpoint snapshot, and embedder snapshot.  The default
compatibility manifest pins the only supported public profile.  A small
``runtime_factory`` seam exists solely so CI can exercise the protocol without
downloading the 20 GB checkpoint.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
    QRAG_INSTRUCTION_PROTOCOL,
    QRAG_TASK_READER_PROFILES,
    render_instruction_reader_prompt,
    task_reader_prompt_hash,
)

UPSTREAM_COMMIT = "e7598130b63eaab868469877e88bcacc59a44efb"
PROTOCOL = "agentinputlab.qrag.frozen.v3"
RELEASES = {
    "acquire.qrag.released_checkpoint_native",
    "acquire.qrag.reader_transfer_factorial",
}
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "answer",
        "gold",
        "gold_answer",
        "ground_truth",
        "label",
        "reward",
        "sf_idx",
        "support",
        "supporting_fact",
        "target",
    }
)


class DriverError(RuntimeError):
    def __init__(
        self, code: str, message: str, *, consumed: bool = False,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed
        self.usage = usage


@dataclass(frozen=True, slots=True)
class ReaderResult:
    answer: str
    input_tokens: int
    output_tokens: int


class Reader(Protocol):
    def __call__(
        self,
        question: str,
        passages: tuple[str, ...],
        artifacts: Mapping[str, Any],
        *,
        max_output_tokens: int,
    ) -> ReaderResult:
        """Send the effective ceiling to the actual model, without retries."""
        ...


@dataclass(frozen=True, slots=True)
class SelectorResult:
    trajectory: tuple[Mapping[str, Any], ...]
    selected_indices: tuple[int, ...]
    embedding_calls: int
    accelerator_seconds: float = 0.0
    official_candidate_builder_called: bool = True
    official_pqn_select_action_called: bool = True
    official_qa_env_step_called: bool = True
    safe_weights_only_load: bool = True
    strict_state_dict_load: bool = True
    checkpoint_eval_mode: bool = True
    seed_applied: bool = True


class QRAGFrozenDriver:
    """Validate mounts/identity before invoking one inference-only runtime."""

    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint_root: Path,
        embedder_root: Path,
        runtime_factory: Callable[[Path, Path, Path], Any],
        reader: Reader,
        manifest_path: Path = DEFAULT_MANIFEST,
        fixture_mode: bool = False,
    ) -> None:
        self.source_root = source_root.resolve()
        self.checkpoint_root = checkpoint_root.resolve()
        self.embedder_root = embedder_root.resolve()
        self.runtime_factory = runtime_factory
        self.reader = reader
        self.manifest_path = manifest_path.resolve()
        self.fixture_mode = fixture_mode
        self.manifest = _load_json(self.manifest_path)
        self.manifest_sha256 = _content_hash(self.manifest)

    def handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self._handle(request)
        except DriverError as exc:
            response = {
                "status": "error",
                "failure_code": exc.code,
                "message": exc.public_message,
                "consumed": exc.consumed,
            }
            if exc.usage is not None:
                response["usage"] = dict(exc.usage)
            return response

    def _handle(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = _validate_envelope(request)
        artifacts = _mapping(payload, "artifacts")
        acknowledgements = _mapping(payload, "acknowledgements")
        controls = _mapping(payload, "controls")
        remaining = _mapping(payload, "remaining_budget")
        contract = _mapping(payload, "inference_contract")
        _require_exact_contract(contract)
        _require_acknowledgements(acknowledgements)
        _require_profile(artifacts, self.manifest, self.manifest_sha256, fixture=self.fixture_mode)

        candidates_raw = payload.get("candidates")
        if isinstance(candidates_raw, (str, bytes)) or not isinstance(candidates_raw, Sequence):
            raise DriverError("REQUEST_INVALID", "candidates must be a sequence")
        candidates = tuple(_validate_candidate(item) for item in candidates_raw)
        if len(candidates) > _positive_int(controls, "max_candidates"):
            raise DriverError("INPUT_TOO_LARGE", "candidate count exceeds configured limit")
        if sum(len(item["text"]) for item in candidates) > _positive_int(
            controls, "max_candidate_chars"
        ):
            raise DriverError("INPUT_TOO_LARGE", "candidate text exceeds configured limit")
        if controls.get("max_select_actions") != 2:
            raise DriverError("PROFILE_DRIFT", "Q-RAG Hotpot profile controls drifted")
        configured_output = _positive_int(controls, "max_output_tokens")
        candidate_ids = tuple(item["document_id"] for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DriverError("REQUEST_INVALID", "candidate IDs must be unique")
        if payload.get("candidate_pool_sha256") != _content_hash(candidates):
            raise DriverError("IDENTITY_MISMATCH", "candidate pool hash mismatch")
        if payload.get("candidate_order_sha256") != _content_hash(candidate_ids):
            raise DriverError("IDENTITY_MISMATCH", "candidate order hash mismatch")
        if payload.get("artifact_identity_sha256") != _content_hash(artifacts):
            raise DriverError("IDENTITY_MISMATCH", "artifact identity hash mismatch")
        if _nonnegative_int(remaining, "model_calls") < 1:
            raise DriverError("MODEL_BUDGET_EXHAUSTED", "reader model budget exhausted")
        remaining_output = _nonnegative_int(remaining, "output_tokens")
        if remaining_output < 1:
            raise DriverError("OUTPUT_BUDGET_EXHAUSTED", "reader output budget exhausted")
        effective_output = min(configured_output, remaining_output)
        reader_budget = {
            "configured_max_output_tokens": configured_output,
            "effective_max_output_tokens": effective_output,
            "remaining_output_tokens": remaining_output,
            "preset": "upstream_default" if configured_output == 4000 else "custom",
        }
        if payload.get("reader_budget") != reader_budget:
            raise DriverError("IDENTITY_MISMATCH", "reader budget binding mismatch")
        instruction_adapter = artifacts.get("reader_profile") in QRAG_TASK_READER_PROFILES
        task = payload.get("reader_task")
        if instruction_adapter:
            if not isinstance(task, Mapping) or set(task) != {"instruction", "candidate_annotations"}:
                raise DriverError("REQUEST_INVALID", "reader task schema mismatch")
            annotations = task["candidate_annotations"]
            if isinstance(annotations, (str, bytes)) or not isinstance(annotations, Sequence):
                raise DriverError("REQUEST_INVALID", "reader annotations must be a sequence")
            try:
                render_instruction_reader_prompt(
                    _text(payload, "question"), tuple(c["text"] for c in candidates),
                    task_instruction=task["instruction"], passage_annotations=annotations,
                )
            except (TypeError, ValueError) as exc:
                raise DriverError("REQUEST_INVALID", "reader task annotations are invalid") from exc
            if tuple(a["document_id"] for a in annotations) != candidate_ids:
                raise DriverError("IDENTITY_MISMATCH", "reader annotations/order mismatch")
            if payload.get("reader_task_sha256") != _content_hash(task):
                raise DriverError("IDENTITY_MISMATCH", "reader task hash mismatch")
        elif task is not None or "reader_task_sha256" in payload:
            raise DriverError("PROFILE_DRIFT", "question-only reader cannot accept a reader task")
        reader_kwargs = {"max_output_tokens": effective_output}
        if instruction_adapter:
            reader_kwargs.update(task_instruction=task["instruction"], passage_annotations=())
        try:
            inspect.signature(self.reader).bind(
                _text(payload, "question"), (), artifacts, **reader_kwargs,
            )
        except (TypeError, ValueError) as exc:
            raise DriverError(
                "READER_CONFIG_INVALID",
                "reader must accept the effective max_output_tokens keyword",
            ) from exc

        source_attestation = self._verify_source(artifacts)
        self._verify_checkpoint(artifacts)
        self._verify_embedder(artifacts)
        runtime = self.runtime_factory(self.source_root, self.checkpoint_root, self.embedder_root)
        try:
            selector: SelectorResult = runtime.select(
                question=_text(payload, "question"),
                candidates=candidates,
                seed=_nonnegative_int(payload, "seed"),
                max_steps=2,
            )
        except Exception as exc:
            raise DriverError("SELECTOR_FAILED", "Q-RAG selector failed", consumed=True) from exc
        trajectory, selected = _validate_selector(selector, candidate_ids)
        passages = tuple(candidates[index]["text"] for index in selected)
        if instruction_adapter:
            reader_kwargs["passage_annotations"] = tuple(annotations[index] for index in selected)
        reader_result = self.reader(
            _text(payload, "question"), passages, artifacts, **reader_kwargs,
        )
        if not isinstance(reader_result, ReaderResult):
            raise DriverError("READER_FAILED", "reader returned an invalid answer", consumed=True)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (reader_result.input_tokens, reader_result.output_tokens)
        ):
            raise DriverError("READER_FAILED", "reader usage is invalid", consumed=True)
        reader_usage = {
            "model_calls": 1,
            "retrieval_calls": 0,
            "tool_calls": 0,
            "embedding_calls": selector.embedding_calls,
            "input_tokens": reader_result.input_tokens,
            "output_tokens": reader_result.output_tokens,
            "accelerator_seconds": selector.accelerator_seconds,
        }
        if not isinstance(reader_result.answer, str) or not reader_result.answer.strip():
            raise DriverError(
                "READER_FAILED", "reader returned an invalid answer",
                consumed=True, usage=reader_usage,
            )
        if reader_result.output_tokens > effective_output:
            raise DriverError(
                "OUTPUT_BUDGET_EXCEEDED", "reader exceeded output budget",
                consumed=True, usage=reader_usage,
            )
        context_cap = remaining.get("context_tokens")
        if context_cap is not None and reader_result.input_tokens > _nonnegative_int(
            remaining, "context_tokens"
        ):
            raise DriverError(
                "CONTEXT_BUDGET_EXCEEDED", "reader exceeded context budget",
                consumed=True, usage=reader_usage,
            )

        termination = (
            "empty_candidates"
            if not candidates
            else "candidate_exhausted"
            if len(selected) == len(candidates)
            else "fixed_step_budget"
        )
        trajectory_payload = tuple(dict(step) for step in trajectory)
        attestation = {
            "source_commit": UPSTREAM_COMMIT,
            "source_manifest_sha256": artifacts["source_manifest_sha256"],
            "compatibility_manifest_sha256": self.manifest_sha256,
            "checkpoint_repository": artifacts["checkpoint_repository"],
            "checkpoint_revision": artifacts["checkpoint_revision"],
            "checkpoint_manifest_sha256": artifacts["checkpoint_manifest_sha256"],
            "checkpoint_file_sha256": artifacts["checkpoint_file_sha256"],
            "embedder_repository": artifacts["embedder_repository"],
            "embedder_revision": artifacts["embedder_revision"],
            "official_candidate_builder_called": selector.official_candidate_builder_called,
            "official_pqn_select_action_called": selector.official_pqn_select_action_called,
            "official_qa_env_step_called": selector.official_qa_env_step_called,
            "dummy_feedback_used": True,
            "ground_truth_feedback_used": False,
            "gold_answer_used": False,
            "supporting_fact_labels_used": False,
            "reward_used": False,
            "training_called": False,
            "stop_action_used": False,
            "strict_state_dict_load": selector.strict_state_dict_load,
            "safe_weights_only_load": selector.safe_weights_only_load,
            "checkpoint_eval_mode": selector.checkpoint_eval_mode,
            "source_user_mounted": True,
            "checkpoint_user_mounted": True,
            "checkpoint_deserialization_isolated": True,
            "host_secret_mounts": False,
            "reader_profile": artifacts["reader_profile"],
            "source_attestation": source_attestation,
            "seed_applied": selector.seed_applied,
        }
        selected_ids = tuple(candidate_ids[index] for index in selected)
        response_payload = {
            "profile_id": payload["profile_id"],
            "artifacts": artifacts,
            "artifact_identity_sha256": payload["artifact_identity_sha256"],
            "candidate_pool_sha256": payload["candidate_pool_sha256"],
            "candidate_order_sha256": payload["candidate_order_sha256"],
            "trajectory": trajectory_payload,
            "trajectory_sha256": _content_hash(trajectory_payload),
            "selected_indices": selected,
            "selected_document_ids": selected_ids,
            "final_answer": reader_result.answer,
            "reader_budget": reader_budget,
            "termination": termination,
            "runtime_attestation": attestation,
        }
        if instruction_adapter:
            response_payload["reader_task_sha256"] = payload["reader_task_sha256"]
        return {
            "status": "ok",
            "request_id": request["request_id"],
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": request["protocol_version"],
            "payload": response_payload,
            "usage": reader_usage,
        }

    def _verify_source(self, artifacts: Mapping[str, Any]) -> str:
        source = _mapping(self.manifest, "source")
        required = _mapping(source, "required_files")
        for relative, expected in required.items():
            path = _safe_file(self.source_root, relative)
            if _sha256_file(path) != expected:
                raise DriverError("SOURCE_MISMATCH", f"source hash mismatch: {relative}")
        if artifacts.get("source_manifest_sha256") != source.get("content_manifest_sha256"):
            raise DriverError("SOURCE_MISMATCH", "source content manifest mismatch")
        git_dir = self.source_root / ".git"
        if git_dir.exists() and not self.fixture_mode:
            completed = subprocess.run(
                ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if completed.returncode != 0 or completed.stdout.strip() != UPSTREAM_COMMIT:
                raise DriverError("SOURCE_MISMATCH", "source checkout commit mismatch")
            return "git_checkout"
        if not self.fixture_mode:
            self._verify_source_snapshot_receipt(source)
            return "verified_archive_snapshot"
        return "verified_archive_snapshot"

    def _verify_source_snapshot_receipt(self, source: Mapping[str, Any]) -> None:
        """Accept a closed commit-addressed snapshot when Git transport is unavailable."""

        receipt = _load_json(_safe_file(self.source_root, "agentinputlab-source-receipt.json"))
        expected_identity = {
            "repository": source.get("repository"),
            "commit": source.get("commit"),
            "tree": source.get("tree"),
            "transport": "jsdelivr_commit_snapshot",
        }
        if any(receipt.get(key) != value for key, value in expected_identity.items()):
            raise DriverError("SOURCE_MISMATCH", "source snapshot identity mismatch")
        rows = receipt.get("files")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
            raise DriverError("SOURCE_MISMATCH", "source snapshot file receipt is invalid")
        declared: dict[str, tuple[str, int]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size"}:
                raise DriverError("SOURCE_MISMATCH", "source snapshot file row is invalid")
            relative = row.get("path")
            digest = row.get("sha256")
            size = row.get("size")
            if (
                not isinstance(relative, str)
                or relative == "agentinputlab-source-receipt.json"
                or relative in declared
                or not isinstance(digest, str)
                or not _HEX64.fullmatch(digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise DriverError("SOURCE_MISMATCH", "source snapshot file row is invalid")
            path = _safe_file(self.source_root, relative)
            if path.stat().st_size != size or _sha256_file(path) != digest:
                raise DriverError("SOURCE_MISMATCH", f"source snapshot hash mismatch: {relative}")
            declared[relative] = (digest, size)
        actual = {
            str(path.relative_to(self.source_root))
            for path in self.source_root.rglob("*")
            if path.is_file()
            and path.name != "agentinputlab-source-receipt.json"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        if actual != set(declared):
            raise DriverError("SOURCE_MISMATCH", "source snapshot closure mismatch")

    def _verify_checkpoint(self, artifacts: Mapping[str, Any]) -> None:
        profile = _mapping(self.manifest, "checkpoint")
        model_path = _safe_file(self.checkpoint_root, "model_best.pt")
        config_path = _safe_file(self.checkpoint_root, "config.yaml")
        model_hash = _sha256_file(model_path)
        config_hash = _sha256_file(config_path)
        expected_model = artifacts.get("checkpoint_file_sha256")
        expected_config = artifacts.get("checkpoint_config_sha256")
        if model_hash != expected_model or config_hash != expected_config:
            raise DriverError("CHECKPOINT_MISMATCH", "checkpoint file/config hash mismatch")
        if not self.fixture_mode and (
            model_hash != profile.get("model_best_sha256")
            or config_hash != profile.get("config_sha256")
        ):
            raise DriverError("CHECKPOINT_MISMATCH", "checkpoint profile mismatch")
        manifest_hash = _content_hash({"config.yaml": config_hash, "model_best.pt": model_hash})
        if artifacts.get("checkpoint_manifest_sha256") != manifest_hash:
            raise DriverError("CHECKPOINT_MISMATCH", "checkpoint manifest hash mismatch")

    def _verify_embedder(self, artifacts: Mapping[str, Any]) -> None:
        manifest_path = _safe_file(self.embedder_root, "agentinputlab-manifest.json")
        manifest = _load_json(manifest_path)
        if manifest.get("repository") != artifacts.get("embedder_repository") or manifest.get(
            "revision"
        ) != artifacts.get("embedder_revision"):
            raise DriverError("EMBEDDER_MISMATCH", "embedder repository/revision mismatch")
        if _content_hash(manifest) != artifacts.get("embedder_manifest_sha256"):
            raise DriverError("EMBEDDER_MISMATCH", "embedder manifest hash mismatch")


def _validate_envelope(request: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(request, Mapping):
        raise DriverError("REQUEST_INVALID", "request must be a mapping")
    if request.get("upstream_commit") != UPSTREAM_COMMIT:
        raise DriverError("IDENTITY_MISMATCH", "upstream commit mismatch")
    if request.get("release_id") not in RELEASES or request.get("method_id") != "qrag":
        raise DriverError("IDENTITY_MISMATCH", "method/release mismatch")
    if not isinstance(request.get("request_id"), str) or not request["request_id"]:
        raise DriverError("REQUEST_INVALID", "request_id is required")
    payload = _mapping(request, "payload")
    artifacts = _mapping(payload, "artifacts")
    adapted = artifacts.get("reader_profile") in QRAG_TASK_READER_PROFILES
    expected_protocol = QRAG_INSTRUCTION_PROTOCOL if adapted else PROTOCOL
    if request.get("protocol_version") != expected_protocol:
        raise DriverError("IDENTITY_MISMATCH", "protocol mismatch")
    if adapted and request["release_id"] != "acquire.qrag.reader_transfer_factorial":
        raise DriverError("PROFILE_DRIFT", "instruction reader requires explicit transfer profile")
    if adapted and artifacts.get("reader_prompt_sha256") != task_reader_prompt_hash(artifacts["reader_profile"]):
        raise DriverError("PROFILE_DRIFT", "instruction reader prompt identity mismatch")
    return payload


def _require_exact_contract(contract: Mapping[str, Any]) -> None:
    expected = {
        "mode": "frozen_checkpoint_inference",
        "allow_training": False,
        "allow_gold_answer": False,
        "allow_supporting_fact_labels": False,
        "allow_reward": False,
        "allow_ground_truth_feedback": False,
        "allow_stop_action": False,
    }
    if dict(contract) != expected:
        raise DriverError("INFERENCE_CONTRACT_REJECTED", "inference contract drifted")


def _require_acknowledgements(value: Mapping[str, Any]) -> None:
    required = (
        "source_noassertion_acknowledged",
        "checkpoint_cc_by_4_0_acknowledged",
        "user_mounted_source",
        "user_mounted_checkpoint",
    )
    if any(value.get(key) is not True for key in required):
        raise DriverError("LICENSE_NOT_ACKNOWLEDGED", "both Q-RAG mounts require acknowledgement")


def _require_profile(
    artifacts: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    fixture: bool,
) -> None:
    checkpoint = _mapping(manifest, "checkpoint")
    embedder = _mapping(manifest, "embedder")
    expected = {
        "profile_id": "qrag.e5_hotpotqa.frozen.v1",
        "checkpoint_repository": checkpoint["repository"],
        "checkpoint_revision": checkpoint["revision"],
        "embedder_repository": embedder["repository"],
        "embedder_revision": embedder["revision"],
        "compatibility_manifest_sha256": manifest_sha256,
    }
    if not fixture:
        expected.update(
            {
                "checkpoint_file_sha256": checkpoint["model_best_sha256"],
                "checkpoint_config_sha256": checkpoint["config_sha256"],
            }
        )
    for key, expected_value in expected.items():
        if artifacts.get(key) != expected_value:
            raise DriverError("PROFILE_DRIFT", f"artifact profile mismatch: {key}")
    for key, value in artifacts.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or not _HEX64.fullmatch(value)):
            raise DriverError("REQUEST_INVALID", f"invalid SHA-256 field: {key}")


def _validate_candidate(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"document_id", "text"}:
        raise DriverError("REQUEST_INVALID", "candidate must contain only document_id and text")
    if _FORBIDDEN_CANDIDATE_KEYS.intersection(value):
        raise DriverError("ORACLE_FIELD_REJECTED", "candidate contains oracle fields")
    document_id = value.get("document_id")
    text = value.get("text")
    if not isinstance(document_id, str) or not document_id or not isinstance(text, str) or not text:
        raise DriverError("REQUEST_INVALID", "candidate ID/text must be non-empty strings")
    return {"document_id": document_id, "text": text}


def _validate_selector(
    result: SelectorResult, candidate_ids: tuple[str, ...]
) -> tuple[tuple[Mapping[str, Any], ...], tuple[int, ...]]:
    if not isinstance(result, SelectorResult):
        raise DriverError("SELECTOR_INVALID", "selector returned an invalid result", consumed=True)
    expected_steps = min(2, len(candidate_ids))
    if len(result.trajectory) != expected_steps:
        raise DriverError("SELECTOR_INVALID", "selector horizon mismatch", consumed=True)
    selected: list[int] = []
    normalized: list[Mapping[str, Any]] = []
    for index, raw in enumerate(result.trajectory):
        if not isinstance(raw, Mapping):
            raise DriverError("SELECTOR_INVALID", "trajectory step is invalid", consumed=True)
        available = tuple(i for i in range(len(candidate_ids)) if i not in selected)
        if tuple(raw.get("available_indices", ())) != available:
            raise DriverError("SELECTOR_INVALID", "available mask drifted", consumed=True)
        q_values = tuple(raw.get("q_values", ()))
        if len(q_values) != len(available) or any(
            isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q)
            for q in q_values
        ):
            raise DriverError("SELECTOR_INVALID", "Q-values are invalid", consumed=True)
        action = raw.get("action_index")
        if isinstance(action, bool) or not isinstance(action, int) or action not in available:
            raise DriverError("SELECTOR_INVALID", "selector action is invalid", consumed=True)
        chosen = float(q_values[available.index(action)])
        if chosen < max(q_values):
            raise DriverError("SELECTOR_INVALID", "selector action is not argmax", consumed=True)
        selected.append(action)
        normalized.append(
            {
                "index": index,
                "available_indices": available,
                "q_values": tuple(float(q) for q in q_values),
                "action_index": action,
                "selected_q_value": chosen,
                "selected_indices": tuple(selected),
                "selected_document_ids": tuple(candidate_ids[i] for i in selected),
            }
        )
    if tuple(result.selected_indices) != tuple(selected):
        raise DriverError("SELECTOR_INVALID", "selected trajectory mismatch", consumed=True)
    required_true = (
        result.official_candidate_builder_called,
        result.official_pqn_select_action_called,
        result.official_qa_env_step_called,
        result.safe_weights_only_load,
        result.strict_state_dict_load,
        result.checkpoint_eval_mode,
    )
    if not all(value is True for value in required_true):
        raise DriverError("SELECTOR_INVALID", "runtime attestation failed", consumed=True)
    return tuple(normalized), tuple(selected)


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DriverError("PATH_REJECTED", "manifest path is invalid")
    path = root / relative
    if path.is_symlink():
        raise DriverError("PATH_REJECTED", "symlinks are forbidden")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise DriverError("PATH_REJECTED", "path escapes mount or is missing") from exc
    if not resolved.is_file():
        raise DriverError("PATH_REJECTED", "required path is not a file")
    return resolved


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("MANIFEST_INVALID", "manifest is unreadable") from exc
    if not isinstance(value, Mapping):
        raise DriverError("MANIFEST_INVALID", "manifest must be an object")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise DriverError("REQUEST_INVALID", f"{key} must be a mapping")
    return result


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise DriverError("REQUEST_INVALID", f"{key} must be non-empty text")
    return result


def _positive_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise DriverError("REQUEST_INVALID", f"{key} must be a positive integer")
    return result


def _nonnegative_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise DriverError("REQUEST_INVALID", f"{key} must be a non-negative integer")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    """The JSONL launcher is intentionally explicit about all user mounts."""

    print(
        "Use QRAGFrozenDriver from a dedicated non-root sidecar; see workers/qrag/README.md",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
