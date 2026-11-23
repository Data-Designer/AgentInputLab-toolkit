#!/usr/bin/env python3
"""JSONL driver for allenai/DecomP at one audited upstream revision.

This worker deliberately lives outside the core package.  It imports the
upstream checkout, preserves its controller/participant implementations, and
applies only the bounded and observable search-loop changes declared in
``compatibility_manifest.json``.

The bounded best-first compatibility loop follows the Apache-2.0-licensed
``BestFirstDecomposer.find_answer_decomp`` control flow at the pinned revision;
see ``THIRD_PARTY_NOTICES.md`` and the compatibility manifest for attribution.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import heapq
import importlib
import io
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

UPSTREAM_ENV = "AGENTINPUTLAB_UPSTREAM_ROOT"
UPSTREAM_COMMIT = "8b157f6fe9ff5676ae51bd054f1df7b015adf278"
REQUEST_SCHEMA = "agentinputlab.decomp.worker.v2"
PROFILE_ID = "plan.decomp.full"
PLANNER_SCHEMA = "agentinputlab.decomp.planner.v1"
PLANNER_PROFILE_ID = "plan.decomp.planner_component"
MANIFEST_SCHEMA = "agentinputlab.decomp.compatibility.v1"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
MAX_REQUEST_BYTES = 1024 * 1024
MAX_QUESTION_CHARS = 200_000
MAX_CONTEXT_BLOCKS = 512
MAX_STEPS = 256
_REQUEST_FIELDS = frozenset(
    {
        "case_id",
        "question",
        "paras",
        "paragraph_ids",
        "context_hash",
        "config_artifact_id",
        "backend_binding",
        "max_steps",
        "upstream_revision",
        "schema_version",
    }
)
_SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "authorization_header",
        "auth_header",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "api_secret",
        "token",
        "access_token",
        "auth_token",
        "bearer_token",
        "refresh_token",
        "session_token",
    }
)
_SECRET_KEY_SUFFIXES = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "refresh_token",
    "secret",
    "session_token",
)


class DriverError(RuntimeError):
    """A public, secret-free worker failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    case_id: str
    question: str
    paras: tuple[str, ...]
    paragraph_ids: tuple[str, ...]
    context_hash: str
    config_artifact_id: str
    backend_binding: Mapping[str, Any]
    max_steps: int


@dataclass(frozen=True, slots=True)
class Checkout:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str


class _BoundedCapture(io.TextIOBase):
    def __init__(self, limit: int = 64 * 1024) -> None:
        self.limit = limit
        self.size = 0
        self.digest = hashlib.sha256()

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("upstream output must be text")
        encoded = value.encode("utf-8", errors="replace")
        self.size += len(encoded)
        if self.size > self.limit:
            raise DriverError("UPSTREAM_OUTPUT_LIMIT", "upstream diagnostic output exceeded limit")
        self.digest.update(encoded)
        return len(value)

    @property
    def hexdigest(self) -> str:
        return self.digest.hexdigest()


class _AuditedBestFirst:
    """Control-flow-equivalent bounded form of upstream BestFirstDecomposer."""

    def __init__(self, controller: Any, search_state_class: type[Any], max_steps: int) -> None:
        self.controller = controller
        self.search_state_class = search_state_class
        self.max_steps = max_steps
        self._state_ids: dict[int, str] = {}
        self._states: list[Any] = []
        self._depths: dict[int, int] = {}
        self.edges: list[tuple[str, str]] = []
        self.participant_calls: list[dict[str, Any]] = []
        self.frontier: list[dict[str, Any]] = []
        self.execute_calls = 0

    def _register(self, state: Any, *, parent: Any | None = None) -> str:
        _validate_search_state(state)
        identity = id(state)
        if identity in self._state_ids:
            return self._state_ids[identity]
        state_id = f"s{len(self._states):04d}"
        self._state_ids[identity] = state_id
        self._states.append(state)
        depth = 0 if parent is None else self._depths[id(parent)] + 1
        self._depths[identity] = depth
        if parent is not None:
            edge = (self._state_ids[id(parent)], state_id)
            if edge not in self.edges:
                self.edges.append(edge)
        return state_id

    def run(self, example: Mapping[str, Any]) -> tuple[Any | None, str]:
        start_command = self.controller.start_state
        start_data = self.controller.init_data(dict(example))
        initial = self.search_state_class(start_data, start_command, score=0.0)
        self._register(initial)
        frontier_heap = [initial]

        while frontier_heap:
            current = heapq.heappop(frontier_heap)
            current_id = self._register(current)
            event = {
                "index": len(self.frontier),
                "state_id": current_id,
                "score": _finite_score(current.score),
                "next_command": _command_text(current.next),
                "action": "pop",
            }
            self.frontier.append(event)

            if current.next == self.controller.end_state:
                if current.data.has_tasks():
                    new_task = current.data.pop_task()
                    new_state = current.copy()
                    if new_task.task_question:
                        new_state.data.add_qgen(new_task.task_question)
                    new_state.next = new_task.task_participant
                    self._register(new_state, parent=current)
                    event["action"] = "resume_task"
                    heapq.heappush(frontier_heap, new_state)
                    continue
                event["action"] = "terminate"
                return current, "end_state"

            if self.execute_calls >= self.max_steps:
                event["action"] = "stop_max_steps"
                return current, "max_steps"

            participant = _command_text(current.next)
            outputs = self.controller.execute(current, debug=False)
            self.execute_calls += 1
            if not isinstance(outputs, list):
                raise DriverError(
                    "UPSTREAM_OUTPUT_INVALID",
                    "upstream ModelController.execute returned a non-list result",
                )
            if not outputs:
                self.participant_calls.append(
                    {
                        "index": len(self.participant_calls),
                        "participant": participant,
                        "input_state_id": current_id,
                        "output_state_id": current_id,
                        "command": participant,
                    }
                )
            for output in outputs:
                output_id = self._register(output, parent=current)
                self.participant_calls.append(
                    {
                        "index": len(self.participant_calls),
                        "participant": participant,
                        "input_state_id": current_id,
                        "output_state_id": output_id,
                        "command": participant,
                    }
                )
                heapq.heappush(frontier_heap, output)

        return None, "frontier_exhausted"

    def nodes(self, fallback_question: str) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        for state in self._states:
            data = state.data
            nodes.append(
                {
                    "node_id": self._state_ids[id(state)],
                    "question": _last_text(data, "get_last_question", fallback_question),
                    "answer": _last_text(data, "get_last_answer", ""),
                    "handler": _command_text(state.next),
                    "depth": self._depths[id(state)],
                }
            )
        return nodes


class DecompDriver:
    def __init__(
        self,
        *,
        manifest_path: Path = DEFAULT_MANIFEST,
        environment: Mapping[str, str] | None = None,
        native_completion: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.environment = dict(os.environ if environment is None else environment)
        self.native_completion = native_completion

    def smoke_upstream_core(self) -> Mapping[str, str]:
        checkout = self._checkout()
        with _upstream_import_scope(checkout.root):
            try:
                module = importlib.import_module("decomp.inference.model_search")
            except ModuleNotFoundError as exc:
                raise _dependency_error(exc) from exc
            symbol = getattr(module, "BestFirstDecomposer", None)
            if not isinstance(symbol, type):
                raise DriverError(
                    "UPSTREAM_API_MISMATCH",
                    "upstream BestFirstDecomposer is unavailable",
                )
        return {
            "upstream_revision": UPSTREAM_COMMIT,
            "symbol": "decomp.inference.model_search.BestFirstDecomposer",
            "manifest_sha256": checkout.manifest_sha256,
        }

    def run(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._run_native(value, planner_component=False)

    def run_planner(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute exactly the source controller's first planner transition."""
        if value.get("schema_version") != PLANNER_SCHEMA or value.get("max_steps") != 1:
            raise DriverError("INVALID_REQUEST", "planner requires its own schema and one transition")
        native_value = dict(value, schema_version=REQUEST_SCHEMA)
        return self._run_native(native_value, planner_component=True)

    def _run_native(self, value: Mapping[str, Any], *, planner_component: bool) -> Mapping[str, Any]:
        request = _parse_request(value)
        checkout = self._checkout()
        artifact = _resolve_artifact(checkout, request.config_artifact_id)
        config = _load_config(artifact)
        config = _validate_and_patch_config(config, checkout, request.max_steps)
        if planner_component:
            start = config.get("start_state")
            if config["models"].get(start, {}).get("name") not in {"llmqadecomp", "lmgen"}:
                raise DriverError("CONTROLLED_CONFIG_UNSUPPORTED", "component start must be a native planner")
        capture = _BoundedCapture()

        with (
            _working_directory(checkout.root),
            _upstream_import_scope(checkout.root),
            contextlib.redirect_stdout(capture),
        ):
            try:
                configurable = importlib.import_module("decomp.inference.configurable_inference")
                model_search = importlib.import_module("decomp.inference.model_search")
            except ModuleNotFoundError as exc:
                raise _dependency_error(exc) from exc
            if self.native_completion is not None:
                _bind_native_completion(checkout, config, self.native_completion)
            builder = getattr(configurable, "build_decomposer_and_models", None)
            search_state_class = getattr(model_search, "SearchState", None)
            best_first_class = getattr(model_search, "BestFirstDecomposer", None)
            if not callable(builder) or not isinstance(search_state_class, type):
                raise DriverError(
                    "UPSTREAM_API_MISMATCH",
                    "upstream configurable inference API is incompatible",
                )
            try:
                upstream_decomposer, model_map = builder(copy.deepcopy(config))
            except ModuleNotFoundError as exc:
                raise _dependency_error(exc) from exc
            except DriverError:
                raise
            except Exception as exc:
                raise DriverError(
                    "UPSTREAM_BOOT_FAILED",
                    "upstream participant construction failed",
                ) from exc
            if not isinstance(best_first_class, type) or not isinstance(
                upstream_decomposer, best_first_class
            ):
                raise DriverError(
                    "UPSTREAM_API_MISMATCH",
                    "upstream builder did not return BestFirstDecomposer",
                )
            if not isinstance(model_map, Mapping):
                raise DriverError(
                    "UPSTREAM_API_MISMATCH",
                    "upstream builder returned an invalid model map",
                )

            search = _AuditedBestFirst(
                upstream_decomposer.controller,
                search_state_class,
                request.max_steps,
            )
            example = {
                "qid": request.case_id,
                "query": request.question,
                "question": request.question,
                "paras": list(request.paras),
                "paragraph_ids": list(request.paragraph_ids),
                "context_hash": request.context_hash,
            }
            try:
                final_state, termination = search.run(example)
            except DriverError:
                raise
            except Exception as exc:
                raise DriverError(
                    "UPSTREAM_EXECUTION_FAILED",
                    "upstream DecomP execution failed",
                ) from exc
            model_calls, call_breakdown = _model_call_usage(
                model_map,
                checkout.manifest,
            )

        if model_calls < 1:
            raise DriverError(
                "USAGE_ACCOUNTING_UNAVAILABLE",
                "upstream run did not report a model call",
            )
        if planner_component:
            if model_calls != 1 or search.execute_calls != 1 or final_state is None:
                raise DriverError("UPSTREAM_OUTPUT_INVALID", "planner did not stop after one native call")
            if any(call["participant"] != config["start_state"] for call in search.participant_calls):
                raise DriverError("UPSTREAM_OUTPUT_INVALID", "component executed a non-planner participant")
            if any(_last_text(s.data, "get_last_answer", "") for s in search._states):
                raise DriverError("UPSTREAM_OUTPUT_INVALID", "planner component unexpectedly produced an answer")
            candidates = [
                {"question": _last_text(s.data, "get_last_question", ""),
                 "next_command": _command_text(s.next), "score": _finite_score(s.score)}
                for s in search._states[1:]
            ]
            selected = {"question": _last_text(final_state.data, "get_last_question", ""),
                        "next_command": _command_text(final_state.next),
                        "score": _finite_score(final_state.score)}
            if not candidates or not selected["question"].strip():
                raise DriverError("UPSTREAM_OUTPUT_INVALID", "native planner returned no next question")
            return {
                "schema_version": PLANNER_SCHEMA, "profile_id": PLANNER_PROFILE_ID,
                "case_id": request.case_id, "upstream_revision": UPSTREAM_COMMIT,
                "paragraph_ids": list(request.paragraph_ids), "context_hash": request.context_hash,
                "component_only": True, "selected": selected, "candidates": candidates,
                "participant_calls": search.participant_calls, "frontier": search.frontier,
                "native_termination": termination, "router_executed": False, "qa_executed": False,
                "usage": {"model_calls": 1, "retrieval_calls": 0,
                          "input_tokens": 0, "output_tokens": 0, "unpriced_model_calls": 1},
                "native_artifacts": {"config_artifact_id": request.config_artifact_id,
                    "config_sha256": artifact["sha256"],
                    "compatibility_manifest_sha256": checkout.manifest_sha256,
                    "backend_binding_hash": _sha256_bytes(_canonical_json(request.backend_binding).encode()),
                    "upstream_stdout_sha256": capture.hexdigest,
                    "boundary": "first_native_planner_transition_before_router_or_qa",
                    "model_call_breakdown": _canonical_json(call_breakdown)},
            }
        reasoning_chain, final_answer = _project_final_state(
            final_state,
            request.question,
        )
        backend_hash = _sha256_bytes(_canonical_json(request.backend_binding).encode("utf-8"))
        patch_ids = [
            patch["id"]
            for patch in checkout.manifest["patches"]
            if isinstance(patch, Mapping) and isinstance(patch.get("id"), str)
        ]
        return {
            "case_id": request.case_id,
            "profile_id": PROFILE_ID,
            "upstream_revision": UPSTREAM_COMMIT,
            "paragraph_ids": list(request.paragraph_ids),
            "context_hash": request.context_hash,
            "nodes": search.nodes(request.question),
            "dependency_edges": [list(edge) for edge in search.edges],
            "participant_calls": search.participant_calls,
            "frontier": search.frontier,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "termination": termination,
            "component_only": False,
            "usage": {
                "model_calls": model_calls,
                "retrieval_calls": 0,
                "unpriced_model_calls": model_calls,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "schema_version": REQUEST_SCHEMA,
            "native_artifacts": {
                "config_artifact_id": request.config_artifact_id,
                "config_sha256": artifact["sha256"],
                "compatibility_manifest_sha256": checkout.manifest_sha256,
                "compatibility_patch_ids": _canonical_json(patch_ids),
                "backend_binding_hash": backend_hash,
                "backend_binding_mode": "audit_identity_only; upstream-native model clients",
                "context_hash": request.context_hash,
                "paragraph_ids_hash": _sha256_bytes(
                    _canonical_json(request.paragraph_ids).encode("utf-8")
                ),
                "paragraph_count": str(len(request.paragraph_ids)),
                "model_call_breakdown": _canonical_json(call_breakdown),
                "token_accounting": "unavailable_in_pinned_upstream",
                "upstream_stdout_sha256": capture.hexdigest,
            },
        }

    def _checkout(self) -> Checkout:
        root_value = self.environment.get(UPSTREAM_ENV, "")
        if not root_value:
            raise DriverError(
                "UPSTREAM_ROOT_MISSING",
                f"{UPSTREAM_ENV} must point to the pinned DecomP checkout",
            )
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise DriverError("UPSTREAM_ROOT_INVALID", "upstream checkout directory is unavailable")
        manifest, manifest_sha = _load_manifest(self.manifest_path)
        _verify_checkout(root, manifest)
        return Checkout(root=root, manifest=manifest, manifest_sha256=manifest_sha)


def _parse_request(value: Mapping[str, Any]) -> WorkerRequest:
    if not isinstance(value, Mapping):
        raise DriverError("INVALID_REQUEST", "worker request must be a JSON object")
    unknown = set(value) - _REQUEST_FIELDS
    missing = _REQUEST_FIELDS - set(value)
    if unknown or missing:
        raise DriverError("INVALID_REQUEST", "worker request fields do not match schema")
    if value.get("upstream_revision") != UPSTREAM_COMMIT:
        raise DriverError("REVISION_MISMATCH", "request upstream revision is not pinned")
    if value.get("schema_version") != REQUEST_SCHEMA:
        raise DriverError("SCHEMA_MISMATCH", "request schema version is unsupported")
    case_id = _bounded_text(value.get("case_id"), "case_id", limit=512)
    question = _bounded_text(value.get("question"), "question", limit=MAX_QUESTION_CHARS)
    paras = _bounded_text_sequence(
        value.get("paras"),
        "paras",
        item_limit=MAX_QUESTION_CHARS,
    )
    paragraph_ids = _bounded_text_sequence(
        value.get("paragraph_ids"),
        "paragraph_ids",
        item_limit=512,
    )
    if len(paras) != len(paragraph_ids):
        raise DriverError("INVALID_REQUEST", "paras and paragraph_ids must have equal length")
    if any(not paragraph_id.strip() for paragraph_id in paragraph_ids):
        raise DriverError("INVALID_REQUEST", "paragraph_ids must contain non-empty IDs")
    if len(paragraph_ids) != len(set(paragraph_ids)):
        raise DriverError("INVALID_REQUEST", "paragraph_ids must be unique")
    context_hash = _bounded_hash(value.get("context_hash"), "context_hash")
    if context_hash != _context_hash(paragraph_ids, paras):
        raise DriverError("CONTEXT_IDENTITY_MISMATCH", "context_hash does not match paragraphs")
    artifact_id = _bounded_text(
        value.get("config_artifact_id"),
        "config_artifact_id",
        limit=256,
    )
    max_steps = value.get("max_steps")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= MAX_STEPS
    ):
        raise DriverError("INVALID_REQUEST", f"max_steps must be between 1 and {MAX_STEPS}")
    backend_binding = value.get("backend_binding")
    if not isinstance(backend_binding, Mapping):
        raise DriverError("INVALID_REQUEST", "backend_binding must be a JSON object")
    _reject_secret_fields(backend_binding)
    try:
        encoded_binding = _canonical_json(backend_binding)
    except (TypeError, ValueError) as exc:
        raise DriverError("INVALID_REQUEST", "backend_binding must be canonical JSON") from exc
    if len(encoded_binding.encode("utf-8")) > 64 * 1024:
        raise DriverError("INVALID_REQUEST", "backend_binding exceeds its size limit")
    return WorkerRequest(
        case_id=case_id,
        question=question,
        paras=paras,
        paragraph_ids=paragraph_ids,
        context_hash=context_hash,
        config_artifact_id=artifact_id,
        backend_binding=json.loads(encoded_binding),
        max_steps=max_steps,
    )


def _bind_native_completion(
    checkout: Checkout, config: Mapping[str, Any], completion: Callable[..., Mapping[str, Any]],
) -> None:
    """Bind only the original GPT3Generator completion seam in this isolated run."""
    for model in config["models"].values():
        if model["name"] in {"llmqa", "llmqadecomp", "lmgen"} and model.get("gen_model", "gpt3") != "gpt3":
            raise DriverError("CONTROLLED_CONFIG_UNSUPPORTED", "live binding requires native gpt3 participants")
    relative = "src/decomp/models/gpt3generator.py"
    if relative not in checkout.manifest["source_files"]:
        raise DriverError("SOURCE_MISMATCH", "live generator must be source-pinned")
    module = importlib.import_module("decomp.models.gpt3generator")
    if Path(module.__file__).resolve() != _checkout_file(checkout.root, relative):
        raise DriverError("IMPORT_STATE_UNSAFE", "native generator import escaped checkout")
    # The original generator still assembles prompts, parses choices, computes
    # sequence scores and sorts branches. Only its provider call/cache seam is
    # replaced; no participant, parser, or search answer is synthesized here.
    module.openai_call = completion
    module.openai = SimpleNamespace(Completion=SimpleNamespace(create=completion))


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("MANIFEST_INVALID", "compatibility manifest is unavailable") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise DriverError("MANIFEST_INVALID", "compatibility manifest schema is invalid")
    upstream = value.get("upstream")
    if not isinstance(upstream, Mapping) or upstream.get("commit") != UPSTREAM_COMMIT:
        raise DriverError("MANIFEST_INVALID", "compatibility manifest revision is invalid")
    for field in (
        "source_files",
        "config_artifacts",
        "allowed_model_names",
        "forbidden_model_names",
        "allowed_reader_names",
        "model_call_metric_names",
        "patches",
    ):
        if field not in value:
            raise DriverError("MANIFEST_INVALID", f"compatibility manifest lacks {field}")
    return value, _sha256_bytes(raw)


def _verify_checkout(root: Path, manifest: Mapping[str, Any]) -> None:
    upstream = manifest["upstream"]
    actual_commit = _read_git_head(root)
    if actual_commit != UPSTREAM_COMMIT:
        raise DriverError("REVISION_MISMATCH", "upstream checkout is at the wrong revision")
    license_path = root / "LICENSE"
    expected_license = upstream.get("license_sha256")
    if upstream.get("license_spdx") != "Apache-2.0" or not isinstance(expected_license, str):
        raise DriverError("MANIFEST_INVALID", "upstream license declaration is invalid")
    if _sha256_file(license_path) != expected_license:
        raise DriverError("LICENSE_MISMATCH", "upstream Apache-2.0 license does not match pin")
    source_files = manifest["source_files"]
    if not isinstance(source_files, Mapping) or not source_files:
        raise DriverError("MANIFEST_INVALID", "source-file pins are invalid")
    for relative, expected_hash in source_files.items():
        path = _checkout_file(root, relative)
        if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
            raise DriverError("SOURCE_MISMATCH", "upstream source does not match manifest")
    if upstream.get("require_clean_checkout") is True:
        _require_clean_git_checkout(root)


def _resolve_artifact(checkout: Checkout, artifact_id: str) -> Mapping[str, Any]:
    artifacts = checkout.manifest["config_artifacts"]
    if not isinstance(artifacts, Mapping):
        raise DriverError("MANIFEST_INVALID", "config artifact manifest is invalid")
    artifact = artifacts.get(artifact_id)
    if not isinstance(artifact, Mapping):
        raise DriverError("CONFIG_NOT_ALLOWED", "config artifact ID is not allowlisted")
    path = _checkout_file(checkout.root, artifact.get("path"))
    expected_hash = artifact.get("sha256")
    if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
        raise DriverError("CONFIG_MISMATCH", "config artifact does not match checksum pin")
    ext_vars = artifact.get("ext_vars", {})
    if not isinstance(ext_vars, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in ext_vars.items()
    ):
        raise DriverError("MANIFEST_INVALID", "config extVars must be pinned strings")
    return {"path": str(path), "sha256": expected_hash, "ext_vars": dict(ext_vars)}


def _load_config(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    path = Path(artifact["path"])
    try:
        if path.suffix == ".jsonnet":
            try:
                jsonnet = importlib.import_module("_jsonnet")
            except ModuleNotFoundError as exc:
                raise _dependency_error(exc) from exc
            rendered = jsonnet.evaluate_file(
                str(path),
                ext_vars=dict(artifact["ext_vars"]),
            )
            value = json.loads(rendered)
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise DriverError("CONFIG_NOT_ALLOWED", "config artifact type is unsupported")
    except DriverError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise DriverError("CONFIG_INVALID", "pinned config could not be loaded") from exc
    if not isinstance(value, Mapping):
        raise DriverError("CONFIG_INVALID", "pinned config root must be an object")
    return value


def _validate_and_patch_config(
    config: Mapping[str, Any],
    checkout: Checkout,
    max_steps: int,
) -> Mapping[str, Any]:
    patched = copy.deepcopy(dict(config))
    models = patched.get("models")
    if not isinstance(models, Mapping) or not models:
        raise DriverError("CONFIG_INVALID", "pinned config requires models")
    allowed = _string_set(checkout.manifest["allowed_model_names"], "allowed model names")
    forbidden = _string_set(checkout.manifest["forbidden_model_names"], "forbidden model names")
    for model_config in models.values():
        if not isinstance(model_config, dict):
            raise DriverError("CONFIG_INVALID", "model config must be an object")
        model_name = model_config.get("name")
        if not isinstance(model_name, str) or model_name in forbidden or model_name not in allowed:
            raise DriverError("UNSAFE_CONFIG", "config requests a forbidden participant model")
        if model_name in {"llmqadecomp", "lmgen", "randgen"}:
            declared = model_config.get("max_steps", max_steps)
            if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
                raise DriverError("CONFIG_INVALID", "participant max_steps must be positive")
            model_config["max_steps"] = min(declared, max_steps)
        for field in ("prompt_file", "operations_file", "model_questions_file"):
            if field in model_config:
                _checkout_file(checkout.root, model_config[field])
    reader = patched.get("reader")
    if reader is not None:
        if not isinstance(reader, Mapping):
            raise DriverError("CONFIG_INVALID", "reader config must be an object")
        readers = _string_set(checkout.manifest["allowed_reader_names"], "allowed readers")
        if reader.get("name") not in readers:
            raise DriverError("UNSAFE_CONFIG", "config requests a forbidden dataset reader")
    return patched


def _model_call_usage(
    model_map: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[int, Mapping[str, int]]:
    accepted = _string_set(manifest["model_call_metric_names"], "model-call metric names")
    breakdown: dict[str, int] = {}
    total = 0
    for participant_name, participant in model_map.items():
        reporter = getattr(participant, "return_model_calls", None)
        if not callable(reporter):
            raise DriverError("USAGE_ACCOUNTING_UNAVAILABLE", "participant lacks call counters")
        counters = reporter()
        if not isinstance(counters, Mapping):
            raise DriverError("USAGE_ACCOUNTING_UNAVAILABLE", "participant counters are invalid")
        for metric, amount in counters.items():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise DriverError("USAGE_ACCOUNTING_UNAVAILABLE", "participant counter is invalid")
            key = f"{participant_name}.{metric}"
            breakdown[key] = amount
            if metric in accepted:
                total += amount
    return total, breakdown


def _project_final_state(final_state: Any | None, question: str) -> tuple[str, str]:
    if final_state is None:
        return question, ""
    data = final_state.data
    printable = _last_text(data, "get_printable_reasoning_chain", "")
    reasoning = question if not printable else f"{question}\n{printable}"
    raw_answer = _last_text(data, "get_last_answer", "")
    if not raw_answer:
        return reasoning, ""
    try:
        parsed = json.loads(raw_answer)
    except (TypeError, json.JSONDecodeError):
        return reasoning, raw_answer
    if isinstance(parsed, str):
        return reasoning, parsed
    if isinstance(parsed, list):
        return reasoning, _canonical_json(parsed)
    return reasoning, raw_answer


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _upstream_import_scope(root: Path) -> Iterator[None]:
    existing = {name for name in sys.modules if name == "decomp" or name.startswith("decomp.")}
    if existing:
        raise DriverError(
            "IMPORT_STATE_UNSAFE",
            "a DecomP module was already loaded before worker execution",
        )
    source_root = str(root / "src")
    sys.path.insert(0, source_root)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(source_root)
        for name in tuple(sys.modules):
            if name == "decomp" or name.startswith("decomp."):
                del sys.modules[name]
        importlib.invalidate_caches()


def _read_git_head(root: Path) -> str:
    git_entry = root / ".git"
    try:
        if git_entry.is_file():
            marker = git_entry.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir: "):
                raise DriverError("REVISION_MISMATCH", "checkout gitdir marker is invalid")
            git_dir = (root / marker.removeprefix("gitdir: ")).resolve()
        else:
            git_dir = git_entry.resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.removeprefix("ref: ")
            ref_path = _checkout_file(git_dir, ref)
            head = ref_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise DriverError("REVISION_MISMATCH", "checkout HEAD is unavailable") from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise DriverError("REVISION_MISMATCH", "checkout HEAD is not a commit digest")
    return head


def _require_clean_git_checkout(root: Path) -> None:
    environment = {"PATH": os.environ.get("PATH", "")}
    try:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DriverError(
            "GIT_VERIFICATION_FAILED", "git cleanliness check is unavailable"
        ) from exc
    if result.returncode != 0:
        raise DriverError("GIT_VERIFICATION_FAILED", "git cleanliness check failed")
    if result.stdout.strip():
        raise DriverError("DIRTY_CHECKOUT", "upstream checkout contains uncommitted files")


def _checkout_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise DriverError("PATH_INVALID", "manifest contains an invalid relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DriverError("PATH_INVALID", "manifest path escapes the checkout") from exc
    if not candidate.is_file():
        raise DriverError("PATH_INVALID", "manifest path is not a file")
    return candidate


def _last_text(data: Any, method_name: str, fallback: str) -> str:
    method = getattr(data, method_name, None)
    if not callable(method):
        return fallback
    try:
        value = method()
    except (IndexError, KeyError, ValueError):
        return fallback
    return value if isinstance(value, str) else _canonical_json(value)


def _validate_search_state(state: Any) -> None:
    for attribute in ("data", "next", "score"):
        if not hasattr(state, attribute):
            raise DriverError("UPSTREAM_OUTPUT_INVALID", "upstream search state is incomplete")
    _finite_score(state.score)


def _finite_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "upstream search score must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "upstream search score must be finite")
    return result


def _command_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", "upstream command must be text")
    return value


def _bounded_text(value: Any, name: str, *, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise DriverError("INVALID_REQUEST", f"{name} must be bounded non-empty text")
    return value


def _bounded_text_sequence(
    value: Any,
    name: str,
    *,
    item_limit: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("INVALID_REQUEST", f"{name} must be a sequence")
    result = tuple(value)
    if len(result) > MAX_CONTEXT_BLOCKS:
        raise DriverError("INVALID_REQUEST", f"{name} exceeds the context block limit")
    if any(
        not isinstance(item, str) or not item or len(item) > item_limit or "\x00" in item
        for item in result
    ):
        raise DriverError("INVALID_REQUEST", f"{name} contains invalid text")
    if len(_canonical_json(result).encode("utf-8")) > MAX_REQUEST_BYTES:
        raise DriverError("INVALID_REQUEST", f"{name} exceeds its byte limit")
    return result  # type: ignore[return-value]


def _bounded_hash(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DriverError("INVALID_REQUEST", f"{name} must be a SHA-256 digest")
    return value


def _context_hash(paragraph_ids: tuple[str, ...], paras: tuple[str, ...]) -> str:
    return _sha256_bytes(
        _canonical_json({"paragraph_ids": paragraph_ids, "paras": paras}).encode("utf-8")
    )


def _reject_secret_fields(value: Mapping[str, Any]) -> None:
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise DriverError("INVALID_REQUEST", "backend_binding keys must be strings")
                normalized = key.lower().replace("-", "_")
                if normalized in _SECRET_KEY_NAMES or any(
                    normalized.endswith(f"_{suffix}") for suffix in _SECRET_KEY_SUFFIXES
                ):
                    raise DriverError(
                        "SECRET_REJECTED",
                        "backend_binding must contain identity metadata only",
                    )
                pending.append(item)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            pending.extend(current)


def _string_set(value: Any, label: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("MANIFEST_INVALID", f"{label} must be a string list")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise DriverError("MANIFEST_INVALID", f"{label} must contain strings")
    return frozenset(result)


def _dependency_error(error: ModuleNotFoundError) -> DriverError:
    name = error.name if isinstance(error.name, str) else "optional dependency"
    return DriverError("DEPENDENCY_MISSING", f"required upstream dependency is missing: {name}")


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DriverError("SOURCE_MISSING", "pinned upstream file is unavailable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _error_response(error: DriverError) -> Mapping[str, str]:
    return {
        "status": "error",
        "failure_code": error.code,
        "message": error.public_message,
        "schema_version": REQUEST_SCHEMA,
    }


def serve_jsonl(driver: DecompDriver, source: TextIO, sink: TextIO) -> int:
    for line in source:
        if len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
            response: Mapping[str, Any] = _error_response(
                DriverError("REQUEST_TOO_LARGE", "worker request exceeds byte limit")
            )
        else:
            try:
                value = json.loads(line)
                response = driver.run(value)
            except json.JSONDecodeError:
                response = _error_response(DriverError("INVALID_JSON", "request is not valid JSON"))
            except DriverError as exc:
                response = _error_response(exc)
            except Exception:
                response = _error_response(
                    DriverError("WORKER_FAILURE", "worker failed without a safe diagnostic")
                )
        sink.write(_canonical_json(response) + "\n")
        sink.flush()
    return 0


def main() -> int:
    return serve_jsonl(DecompDriver(), sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
