#!/usr/bin/env python3
"""Pinned JSONL driver for the original IRCoT controller loop."""

from __future__ import annotations

import contextlib
import copy
import difflib
import hashlib
import importlib
import io
import json
import math
import os
import re
import subprocess
import sys
import types
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

UPSTREAM_COMMIT = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
PROTOCOL_VERSION = "agentinputlab.ircot.v1"
MANIFEST_SCHEMA = "agentinputlab.ircot.compatibility.v1"
UPSTREAM_ENV = "AGENTINPUTLAB_UPSTREAM_ROOT"
PROFILE_ENV = "AGENTINPUTLAB_IRCOT_PROFILE"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")
DEFAULT_PROFILE = "fixture.tiny.frozen.v1"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
_REQUEST_FIELDS = frozenset(
    {
        "request_id",
        "state_hash",
        "task_id",
        "question",
        "controls",
        "generator_fingerprint",
        "retriever_fingerprint",
        "corpus_revision",
        "index_revision",
        "prompt_hash",
        "upstream_commit",
        "protocol_version",
    }
)
_CONTROL_FIELDS = frozenset(
    {
        "per_step_top_k",
        "max_steps",
        "max_total_documents",
        "max_output_tokens",
        "deduplicate",
        "reset_policy",
    }
)
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
)
_SECRET_TOKEN_KEYS = frozenset(
    {
        "token",
        "access_token",
        "auth_token",
        "bearer_token",
        "refresh_token",
        "session_token",
    }
)


class DriverError(RuntimeError):
    def __init__(self, code: str, message: str, *, pre_execution: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.pre_execution = pre_execution


@dataclass(frozen=True, slots=True)
class Controls:
    per_step_top_k: int
    max_steps: int
    max_total_documents: int
    max_output_tokens: int
    deduplicate: bool
    reset_policy: str


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    state_hash: str
    task_id: str
    question: str
    controls: Controls
    generator_fingerprint: Mapping[str, Any]
    retriever_fingerprint: Mapping[str, Any]
    corpus_revision: str
    index_revision: str
    prompt_hash: str


@dataclass(frozen=True, slots=True)
class Checkout:
    root: Path
    worker_root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    profile_id: str
    profile: Mapping[str, Any]
    config_path: Path
    prompt_path: Path
    bindings_path: Path | None


@dataclass(frozen=True, slots=True)
class Snapshot:
    question: str
    titles: tuple[str, ...]
    paragraphs: tuple[str, ...]
    generated_sentences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ControllerEvent:
    command: str
    role: str
    before: Snapshot
    after: Snapshot
    retrieval_rounds: tuple[Mapping[str, Any], ...]


class _BoundedCapture(io.TextIOBase):
    def __init__(self, limit: int = 128 * 1024) -> None:
        self.limit = limit
        self.size = 0
        self._digest = hashlib.sha256()

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8", errors="replace")
        self.size += len(encoded)
        if self.size > self.limit:
            raise DriverError(
                "UPSTREAM_OUTPUT_LIMIT",
                "upstream diagnostic output exceeded its limit",
                pre_execution=False,
            )
        self._digest.update(encoded)
        return len(value)

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class _FrozenResponse:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = copy.deepcopy(dict(payload))
        self.ok = True
        self.status_code = 200

    def json(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._payload)


class _FrozenBindings:
    def __init__(self, value: Mapping[str, Any], corpus_revision: str) -> None:
        self.corpus_revision = corpus_revision
        self.expected_question = _required_text(value, "question", "fixture bindings")
        raw_generations = _mapping_sequence(
            value.get("generator_outputs"),
            "fixture generator_outputs",
        )
        self.generations = [dict(item) for item in raw_generations]
        raw_rounds = _mapping_sequence(value.get("retrieval_rounds"), "fixture retrieval_rounds")
        self.rounds = [dict(item) for item in raw_rounds]
        self.generation_index = 0
        self.retrieval_index = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def generator_class(self) -> type[Any]:
        owner = self

        class FrozenGenerator:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = dict(kwargs)

            def generate_text_sequence(self, prompt: str) -> list[tuple[str, float]]:
                if owner.generation_index >= len(owner.generations):
                    raise DriverError(
                        "FIXTURE_EXHAUSTED",
                        "frozen generator output sequence is exhausted",
                        pre_execution=False,
                    )
                item = owner.generations[owner.generation_index]
                owner.generation_index += 1
                text = _required_text(item, "text", "fixture generation")
                owner.input_tokens += _nonnegative_integer(
                    item.get("input_tokens"),
                    "fixture input_tokens",
                )
                owner.output_tokens += _nonnegative_integer(
                    item.get("output_tokens"),
                    "fixture output_tokens",
                )
                if not prompt.strip():
                    raise DriverError(
                        "FIXTURE_PROMPT_INVALID",
                        "upstream generator received an empty prompt",
                        pre_execution=False,
                    )
                return [(text, 0.0)]

        return FrozenGenerator

    def safe_post_request(self, url: str, params: Mapping[str, Any]) -> _FrozenResponse:
        del url
        if self.retrieval_index >= len(self.rounds):
            raise DriverError(
                "FIXTURE_EXHAUSTED",
                "frozen retrieval sequence is exhausted",
                pre_execution=False,
            )
        item = self.rounds[self.retrieval_index]
        self.retrieval_index += 1
        expected_query = _required_text(item, "query", "fixture retrieval")
        actual_query = str(params.get("query_text", "")).strip()
        if actual_query != expected_query:
            raise DriverError(
                "FIXTURE_QUERY_MISMATCH",
                "upstream retrieval query differs from frozen trajectory",
                pre_execution=False,
            )
        documents = _mapping_sequence(item.get("documents"), "fixture documents")
        retrieval = []
        for document in documents:
            retrieval.append(
                {
                    **dict(document),
                    "corpus_name": self.corpus_revision,
                }
            )
        return _FrozenResponse({"retrieval": retrieval})


class _RetrievalCapture:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.rounds: list[Mapping[str, Any]] = []

    def __call__(self, url: str, params: Mapping[str, Any]) -> Any:
        response = self.delegate(url, params)
        documents: list[Mapping[str, Any]] = []
        try:
            if response.ok:
                payload = response.json()
                raw = payload.get("retrieval", []) if isinstance(payload, Mapping) else []
                documents = [dict(item) for item in raw if isinstance(item, Mapping)]
        except Exception:
            documents = []
        self.rounds.append(
            {
                "query": str(params.get("query_text", "")),
                "documents": documents,
            }
        )
        return response


class _TracingController:
    def __init__(
        self,
        controller: Any,
        role_by_command: Mapping[str, str],
        retrieval_capture: _RetrievalCapture,
    ) -> None:
        self._controller = controller
        self._roles = dict(role_by_command)
        self._capture = retrieval_capture
        self.events: list[ControllerEvent] = []

    @property
    def start_state(self) -> Any:
        return self._controller.start_state

    @property
    def end_state(self) -> Any:
        return self._controller.end_state

    def init_data(self, value: Mapping[str, Any]) -> Any:
        return self._controller.init_data(value)

    def execute(self, state: Any, debug: bool = False) -> list[Any]:
        before = _snapshot(state.data)
        capture_start = len(self._capture.rounds)
        outputs = self._controller.execute(state, debug=debug)
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise DriverError(
                "UPSTREAM_OUTPUT_INVALID",
                "IRCoT participant must return exactly one state",
                pre_execution=False,
            )
        output = outputs[0]
        command = state.next
        if not isinstance(command, str) or command not in self._roles:
            raise DriverError(
                "UPSTREAM_OUTPUT_INVALID",
                "IRCoT controller emitted an unknown command",
                pre_execution=False,
            )
        self.events.append(
            ControllerEvent(
                command=command,
                role=self._roles[command],
                before=before,
                after=_snapshot(output.data),
                retrieval_rounds=tuple(self._capture.rounds[capture_start:]),
            )
        )
        return outputs


class IRCoTDriver:
    def __init__(
        self,
        *,
        manifest_path: Path = DEFAULT_MANIFEST,
        profile_id: str | None = None,
        environment: Mapping[str, str] | None = None,
        isolated_process: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.environment = dict(os.environ if environment is None else environment)
        self.profile_id = profile_id or self.environment.get(PROFILE_ENV, DEFAULT_PROFILE)
        self.isolated_process = isolated_process

    def identity(self) -> Mapping[str, Any]:
        checkout = self._checkout()
        api_status = self.smoke_upstream_api(checkout=checkout)
        return {
            "profile_id": checkout.profile_id,
            "identity_kind": checkout.profile["identity_kind"],
            "upstream_commit": UPSTREAM_COMMIT,
            "manifest_sha256": checkout.manifest_sha256,
            "config_sha256": checkout.profile["config_sha256"],
            "prompt_sha256": checkout.profile["prompt_sha256"],
            "binding_attestation": checkout.profile["binding_attestation"],
            "full_api_status": api_status["full_api_status"],
        }

    def smoke_upstream_api(self, *, checkout: Checkout | None = None) -> Mapping[str, str]:
        checkout = checkout or self._checkout()
        with _upstream_import_scope(checkout.root, fixture_shims=False):
            try:
                model_search = importlib.import_module("commaqa.inference.model_search")
            except ModuleNotFoundError as exc:
                raise _dependency_error(exc) from exc
            if not isinstance(getattr(model_search, "BestFirstDecomposer", None), type):
                raise DriverError("UPSTREAM_API_MISMATCH", "BestFirstDecomposer is unavailable")
        full_status = "available"
        try:
            with _upstream_import_scope(checkout.root, fixture_shims=False):
                module = importlib.import_module("commaqa.inference.ircot")
                for symbol in (
                    "RetrieveAndResetParagraphsParticipant",
                    "StepByStepCOTGenParticipant",
                    "StepByStepExitControllerParticipant",
                ):
                    if not isinstance(getattr(module, symbol, None), type):
                        raise DriverError("UPSTREAM_API_MISMATCH", f"{symbol} is unavailable")
        except DriverError:
            raise
        except ModuleNotFoundError as exc:
            name = exc.name if isinstance(exc.name, str) else "optional_dependency"
            full_status = f"dependency_missing:{name}"
        return {
            "core_symbol": "commaqa.inference.model_search.BestFirstDecomposer",
            "full_api_status": full_status,
        }

    def run(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        request = _parse_request(value)
        checkout = self._checkout()
        _attest_request(request, checkout, self.environment)
        identity_kind = checkout.profile["identity_kind"]
        if identity_kind == "upstream_native" and not self.isolated_process:
            raise DriverError(
                "ISOLATION_REQUIRED",
                "upstream-native IRCoT must run through the JSONL worker process",
            )
        if identity_kind == "upstream_native":
            raise DriverError(
                "TOKEN_USAGE_OBSERVER_REQUIRED",
                "historical native clients lack observed token usage; use the startup-bound controlled runtime",
            )
        config = _load_config(checkout, self.environment)
        config = _patch_config(config, checkout, request.controls)
        fixture_bindings = (
            _load_fixture_bindings(checkout, request)
            if identity_kind == "controlled_fixture"
            else None
        )
        capture = _BoundedCapture()
        fixture_shims = fixture_bindings is not None

        with (
            _working_directory(checkout.root),
            _upstream_import_scope(checkout.root, fixture_shims=fixture_shims),
            contextlib.redirect_stdout(capture),
        ):
            try:
                configurable = importlib.import_module("commaqa.inference.configurable_inference")
                ircot_module = importlib.import_module("commaqa.inference.ircot")
            except ModuleNotFoundError as exc:
                raise _dependency_error(exc) from exc
            if fixture_bindings is not None:
                frozen_generator = fixture_bindings.generator_class()
                ircot_module.GPT3Generator = frozen_generator
                ircot_module.LLMClientGenerator = frozen_generator
                retrieval_delegate = fixture_bindings.safe_post_request
            else:
                retrieval_delegate = ircot_module.safe_post_request
            retrieval_capture = _RetrievalCapture(retrieval_delegate)
            ircot_module.safe_post_request = retrieval_capture
            builder = getattr(configurable, "build_decomposer_and_models", None)
            if not callable(builder):
                raise DriverError("UPSTREAM_API_MISMATCH", "upstream builder is unavailable")
            try:
                decomposer, model_map = builder(copy.deepcopy(config))
            except ModuleNotFoundError as exc:
                raise _dependency_error(exc) from exc
            except DriverError:
                raise
            except Exception as exc:
                raise DriverError(
                    "UPSTREAM_BOOT_FAILED",
                    "upstream IRCoT participant construction failed",
                ) from exc
            if not isinstance(model_map, Mapping) or not hasattr(decomposer, "controller"):
                raise DriverError("UPSTREAM_API_MISMATCH", "upstream builder result is invalid")
            role_by_command = {key: item["name"] for key, item in config["models"].items()}
            tracing = _TracingController(
                decomposer.controller,
                role_by_command,
                retrieval_capture,
            )
            decomposer.controller = tracing
            example = {
                "qid": request.task_id or request.request_id,
                "query": request.question,
                "question": request.question,
                "metadata": {
                    "pinned_title": None,
                    "pinned_para": None,
                    "pin_position": None,
                },
                "titles": [],
                "paras": [],
            }
            try:
                final_state, _ = decomposer.find_answer_decomp(example, debug=False)
            except DriverError:
                raise
            except SystemExit as exc:
                raise DriverError(
                    "UPSTREAM_ABORTED",
                    "upstream model client aborted execution",
                    pre_execution=False,
                ) from exc
            except Exception as exc:
                raise DriverError(
                    "UPSTREAM_EXECUTION_FAILED",
                    "upstream IRCoT execution failed",
                    pre_execution=False,
                ) from exc
            result = _project_trajectory(
                request,
                checkout,
                config,
                tracing.events,
                final_state,
                ircot_module,
                model_map,
                fixture_bindings,
            )
        result["upstream_stdout_sha256"] = capture.hexdigest
        return result

    def safe_call(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        request_id = value.get("request_id", "") if isinstance(value, Mapping) else ""
        try:
            return self.run(value)
        except DriverError as exc:
            response: dict[str, Any] = {
                "status": "error",
                "request_id": request_id if isinstance(request_id, str) else "",
                "failure_code": exc.code,
                "message": exc.public_message,
                "protocol_version": PROTOCOL_VERSION,
            }
            if exc.pre_execution:
                response["consumed"] = False
            return response

    def _checkout(self) -> Checkout:
        root_value = self.environment.get(UPSTREAM_ENV, "")
        if not root_value:
            raise DriverError("UPSTREAM_ROOT_MISSING", f"{UPSTREAM_ENV} is required")
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise DriverError("UPSTREAM_ROOT_INVALID", "IRCoT checkout is unavailable")
        manifest, manifest_sha = _load_manifest(self.manifest_path)
        _verify_checkout(root, manifest)
        profiles = manifest["profiles"]
        profile = profiles.get(self.profile_id) if isinstance(profiles, Mapping) else None
        if not isinstance(profile, Mapping):
            raise DriverError("PROFILE_NOT_ALLOWED", "IRCoT worker profile is not allowlisted")
        if profile.get("identity_kind") not in {"controlled_fixture", "upstream_native"}:
            raise DriverError("MANIFEST_INVALID", "IRCoT profile identity kind is invalid")
        worker_root = self.manifest_path.resolve().parent
        config_path = _profile_file(root, worker_root, profile, "config")
        prompt_path = _profile_file(root, worker_root, profile, "prompt")
        bindings_path = (
            _profile_file(root, worker_root, profile, "bindings")
            if "bindings_path" in profile
            else None
        )
        return Checkout(
            root=root,
            worker_root=worker_root,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            profile_id=self.profile_id,
            profile=profile,
            config_path=config_path,
            prompt_path=prompt_path,
            bindings_path=bindings_path,
        )


def _parse_request(value: Mapping[str, Any]) -> Request:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
        raise DriverError("INVALID_REQUEST", "IRCoT request fields do not match schema")
    if value.get("upstream_commit") != UPSTREAM_COMMIT:
        raise DriverError("REVISION_MISMATCH", "IRCoT request revision is not pinned")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise DriverError("PROTOCOL_MISMATCH", "IRCoT request protocol is unsupported")
    request_id = _bounded_text(value.get("request_id"), "request_id", 256)
    state_hash = _bounded_text(value.get("state_hash"), "state_hash", 128)
    task_id = value.get("task_id")
    if not isinstance(task_id, str) or len(task_id) > 512:
        raise DriverError("INVALID_REQUEST", "task_id must be bounded text")
    question = _bounded_text(value.get("question"), "question", 200_000)
    controls_value = value.get("controls")
    if not isinstance(controls_value, Mapping) or set(controls_value) != _CONTROL_FIELDS:
        raise DriverError("INVALID_REQUEST", "IRCoT controls do not match schema")
    controls = Controls(
        per_step_top_k=_positive_integer(controls_value["per_step_top_k"], "per_step_top_k", 100),
        max_steps=_positive_integer(controls_value["max_steps"], "max_steps", 128),
        max_total_documents=_positive_integer(
            controls_value["max_total_documents"],
            "max_total_documents",
            10_000,
        ),
        max_output_tokens=_positive_integer(
            controls_value["max_output_tokens"],
            "max_output_tokens",
            10_000_000,
        ),
        deduplicate=_boolean(controls_value["deduplicate"], "deduplicate"),
        reset_policy=_enum_text(
            controls_value["reset_policy"],
            "reset_policy",
            {"upstream_native", "never", "replace_each_step"},
        ),
    )
    if not controls.deduplicate:
        raise DriverError(
            "CONTROL_UNSUPPORTED",
            "pinned upstream IRCoT always de-duplicates retrieved paragraphs",
        )
    generator = _canonical_mapping(value.get("generator_fingerprint"), "generator_fingerprint")
    retriever = _canonical_mapping(value.get("retriever_fingerprint"), "retriever_fingerprint")
    return Request(
        request_id=request_id,
        state_hash=state_hash,
        task_id=task_id,
        question=question,
        controls=controls,
        generator_fingerprint=generator,
        retriever_fingerprint=retriever,
        corpus_revision=_bounded_text(value.get("corpus_revision"), "corpus_revision", 256),
        index_revision=_bounded_text(value.get("index_revision"), "index_revision", 256),
        prompt_hash=_bounded_text(value.get("prompt_hash"), "prompt_hash", 128),
    )


def _attest_request(request: Request, checkout: Checkout, environment: Mapping[str, str]) -> None:
    profile = checkout.profile
    if request.prompt_hash != profile["prompt_sha256"]:
        raise DriverError("PROMPT_MISMATCH", "request prompt hash differs from pinned prompt")
    if request.corpus_revision != profile.get("corpus_revision"):
        raise DriverError("CORPUS_MISMATCH", "request corpus revision differs from worker profile")
    expected_index = profile.get("index_revision")
    if expected_index is None:
        env_key = profile.get("index_revision_env")
        expected_index = environment.get(env_key, "") if isinstance(env_key, str) else ""
    if not expected_index or request.index_revision != expected_index:
        raise DriverError("INDEX_MISMATCH", "request index revision is not attested")
    if profile["identity_kind"] == "upstream_native":
        generator_env = profile.get("generator_fingerprint_sha256_env")
        retriever_env = profile.get("retriever_fingerprint_sha256_env")
        expected_generator = (
            environment.get(generator_env, "") if isinstance(generator_env, str) else ""
        )
        expected_retriever = (
            environment.get(retriever_env, "") if isinstance(retriever_env, str) else ""
        )
        if (
            not expected_generator
            or _canonical_hash(request.generator_fingerprint) != expected_generator
        ):
            raise DriverError("GENERATOR_IDENTITY_MISMATCH", "generator binding is not attested")
        if (
            not expected_retriever
            or _canonical_hash(request.retriever_fingerprint) != expected_retriever
        ):
            raise DriverError("RETRIEVER_IDENTITY_MISMATCH", "retriever binding is not attested")
        required = profile.get("required_runtime_env", [])
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise DriverError("MANIFEST_INVALID", "required runtime environment list is invalid")
        if any(not isinstance(key, str) or not environment.get(key, "") for key in required):
            raise DriverError("RUNTIME_BINDING_MISSING", "production model binding is unavailable")


def _load_config(checkout: Checkout, environment: Mapping[str, str]) -> Mapping[str, Any]:
    if checkout.config_path.suffix == ".json":
        try:
            value = json.loads(checkout.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DriverError("CONFIG_INVALID", "pinned JSON config cannot be loaded") from exc
    elif checkout.config_path.suffix == ".jsonnet":
        try:
            jsonnet = importlib.import_module("_jsonnet")
        except ModuleNotFoundError as exc:
            raise _dependency_error(exc) from exc
        ext_var_spec = checkout.profile.get("config_ext_vars", {})
        if not isinstance(ext_var_spec, Mapping):
            raise DriverError("MANIFEST_INVALID", "config extVar specification is invalid")
        ext_vars: dict[str, str] = {}
        for ext_name, env_name in ext_var_spec.items():
            if not isinstance(ext_name, str) or not isinstance(env_name, str):
                raise DriverError("MANIFEST_INVALID", "config extVar names must be strings")
            env_value = environment.get(env_name, "")
            if not env_value:
                raise DriverError(
                    "RUNTIME_BINDING_MISSING", "production retriever binding is unavailable"
                )
            ext_vars[ext_name] = env_value
        try:
            value = json.loads(jsonnet.evaluate_file(str(checkout.config_path), ext_vars=ext_vars))
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise DriverError(
                "CONFIG_INVALID", "pinned Jsonnet config cannot be evaluated"
            ) from exc
    else:
        raise DriverError("CONFIG_INVALID", "pinned config extension is unsupported")
    if not isinstance(value, Mapping):
        raise DriverError("CONFIG_INVALID", "pinned config root must be an object")
    return value


def _patch_config(
    config: Mapping[str, Any],
    checkout: Checkout,
    controls: Controls,
) -> Mapping[str, Any]:
    patched = copy.deepcopy(dict(config))
    models = patched.get("models")
    if not isinstance(models, Mapping):
        raise DriverError("CONFIG_INVALID", "IRCoT config requires models")
    allowed = _string_set(checkout.manifest["allowed_model_names"], "allowed model names")
    roles: set[str] = set()
    per_call_output = max(1, controls.max_output_tokens // controls.max_steps)
    for model in models.values():
        if not isinstance(model, dict):
            raise DriverError("CONFIG_INVALID", "IRCoT model config must be an object")
        name = model.get("name")
        if not isinstance(name, str) or name not in allowed:
            raise DriverError("UNSAFE_CONFIG", "IRCoT config contains a non-allowlisted model")
        roles.add(name)
        if name == "retrieve_and_reset_paragraphs":
            model["retrieval_count"] = controls.per_step_top_k
            model["global_max_num_paras"] = controls.max_total_documents
            model["cumulate_titles"] = controls.reset_policy != "replace_each_step"
        elif name == "step_by_step_cot_gen":
            model["max_num_sentences"] = controls.max_steps
            if "max_tokens" in model or model.get("gen_model") == "gpt3":
                model["max_tokens"] = min(
                    int(model.get("max_tokens", per_call_output)), per_call_output
                )
            if "max_length" in model or model.get("gen_model") == "llm_api":
                model["max_length"] = min(
                    int(model.get("max_length", per_call_output)), per_call_output
                )
            prompt_file = model.get("prompt_file")
            if checkout.profile["identity_kind"] == "controlled_fixture":
                if prompt_file != "__AGENTINPUTLAB_LOCKED_PROMPT__":
                    raise DriverError("CONFIG_INVALID", "fixture prompt marker is missing")
                model["prompt_file"] = str(checkout.prompt_path)
            else:
                resolved = _safe_file(checkout.root, prompt_file)
                if resolved != checkout.prompt_path:
                    raise DriverError("PROMPT_MISMATCH", "config uses an unpinned prompt")
        elif name == "step_by_step_exit_controller":
            model["max_num_sentences"] = controls.max_steps
            model["global_max_num_paras"] = controls.max_total_documents
    required_roles = {
        "retrieve_and_reset_paragraphs",
        "step_by_step_cot_gen",
        "step_by_step_exit_controller",
    }
    if roles != required_roles:
        raise DriverError(
            "CONFIG_INVALID", "IRCoT profile must contain the complete three-stage loop"
        )
    return patched


def _project_trajectory(
    request: Request,
    checkout: Checkout,
    config: Mapping[str, Any],
    events: Sequence[ControllerEvent],
    final_state: Any,
    ircot_module: Any,
    model_map: Mapping[str, Any],
    fixture_bindings: _FrozenBindings | None,
) -> dict[str, Any]:
    retrieval_events = [event for event in events if event.role == "retrieve_and_reset_paragraphs"]
    if not retrieval_events:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "upstream IRCoT trajectory contains no retrieval",
            pre_execution=False,
        )
    if len(retrieval_events) - 1 > request.controls.max_steps:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "upstream IRCoT trajectory exceeds max_steps",
            pre_execution=False,
        )
    document_records: dict[str, dict[str, Any]] = {}
    pair_to_id: dict[tuple[str, str], str] = {}
    seen: set[str] = set()
    steps: list[dict[str, Any]] = []
    active_ids: tuple[str, ...] = ()
    initial_ids: tuple[str, ...] = ()

    for retrieval_index, event in enumerate(retrieval_events):
        raw_hits = [
            hit
            for round_value in event.retrieval_rounds
            for hit in _mapping_sequence(round_value.get("documents", []), "retrieval documents")
        ]
        hit_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
        for hit in raw_hits:
            title = str(hit.get("title", ""))
            paragraph = str(hit.get("paragraph_text", ""))
            if title and paragraph:
                hit_by_pair[(title, paragraph)] = hit
        current_ids: list[str] = []
        for title, paragraph in zip(
            event.after.titles,
            event.after.paragraphs,
            strict=True,
        ):
            pair = (title, paragraph)
            hit = hit_by_pair.get(pair, {})
            document_id = pair_to_id.get(pair)
            if document_id is None:
                raw_id = hit.get("id") if isinstance(hit, Mapping) else None
                document_id = (
                    raw_id
                    if isinstance(raw_id, str) and raw_id
                    else str(ircot_module.get_pid_for_title_paragraph_text(title, paragraph))
                )
                pair_to_id[pair] = document_id
            score_value = hit.get("score") if isinstance(hit, Mapping) else None
            score = _optional_finite_number(score_value, "document score")
            if document_id not in document_records or hit:
                document_records[document_id] = {
                    "document_id": document_id,
                    "text": paragraph,
                    "title": title,
                    "corpus_revision": request.corpus_revision,
                    "score": score,
                    "metadata": {
                        "worker_profile": checkout.profile_id,
                        "worker_identity_kind": checkout.profile["identity_kind"],
                    },
                }
            current_ids.append(document_id)
        active_ids = tuple(dict.fromkeys(current_ids))
        raw_retrieved: list[str] = []
        for pair in hit_by_pair:
            if pair in pair_to_id and pair_to_id[pair] in active_ids:
                raw_retrieved.append(pair_to_id[pair])
        before_pairs = set(zip(event.before.titles, event.before.paragraphs, strict=True))
        new_active = [
            pair_to_id[pair]
            for pair in pair_to_id
            if pair not in before_pairs and pair_to_id[pair] in active_ids
        ]
        for document_id in new_active:
            if document_id not in raw_retrieved:
                raw_retrieved.append(document_id)
        retrieved_ids = tuple(dict.fromkeys(raw_retrieved))
        if retrieval_index == 0:
            initial_ids = active_ids
            seen.update(initial_ids)
            continue
        generated = tuple(
            str(item)
            for item in ircot_module.remove_reasoning_sentences(
                list(event.before.generated_sentences)
            )
            if str(item).strip()
        )
        latest = generated[-1].strip() if generated else request.question.strip()
        cot_step = " ".join(generated).strip() or latest
        new_ids = tuple(document_id for document_id in retrieved_ids if document_id not in seen)
        seen.update(new_ids)
        reset_applied = request.controls.reset_policy == "replace_each_step"
        steps.append(
            {
                "index": len(steps),
                "cot_step": cot_step,
                "latest_cot_sentence": latest,
                "retrieval_query": latest,
                "retrieved_document_ids": list(retrieved_ids),
                "new_document_ids": list(new_ids),
                "active_document_ids": list(active_ids),
                "reset_applied": reset_applied,
            }
        )
    final_generated = (
        _snapshot(final_state.data).generated_sentences if final_state is not None else ()
    )
    exit_model = next(
        model
        for model in config["models"].values()
        if model["name"] == "step_by_step_exit_controller"
    )
    regex = re.compile(str(exit_model.get("answer_extractor_regex", ".* answer is:? (.*)")))
    answer = ""
    if final_generated:
        match = regex.match(final_generated[-1])
        if match:
            answer = match.group(1)
            if exit_model.get("answer_extractor_remove_last_fullstop") and answer.endswith("."):
                answer = answer[:-1]
    if answer:
        exit_reason = "answer_detected"
    elif final_state is None:
        exit_reason = "upstream"
    elif len(steps) == request.controls.max_steps:
        exit_reason = "max_steps"
    else:
        exit_reason = "controller"
    model_calls = _model_calls(model_map)
    if model_calls < 1:
        raise DriverError(
            "USAGE_ACCOUNTING_UNAVAILABLE",
            "upstream IRCoT reported no model calls",
            pre_execution=False,
        )
    input_tokens = fixture_bindings.input_tokens if fixture_bindings is not None else 0
    output_tokens = fixture_bindings.output_tokens if fixture_bindings is not None else 0
    if output_tokens > request.controls.max_output_tokens:
        raise DriverError(
            "OUTPUT_BUDGET_EXCEEDED",
            "upstream IRCoT exceeded max_output_tokens",
            pre_execution=False,
        )
    return {
        "status": "ok",
        "request_id": request.request_id,
        "protocol_version": PROTOCOL_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "initial_query": request.question,
        "initial_document_ids": list(initial_ids),
        "steps": steps,
        "documents": list(document_records.values()),
        "active_document_ids": list(active_ids),
        "exit_reason": exit_reason,
        "answer": answer,
        "usage": {
            "retrieval_calls": 1 + len(steps),
            "model_calls": model_calls,
            "unpriced_model_calls": model_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "worker_profile_id": checkout.profile_id,
        "worker_identity_kind": checkout.profile["identity_kind"],
        "binding_attestation": checkout.profile["binding_attestation"],
    }


def _model_calls(model_map: Mapping[str, Any]) -> int:
    total = 0
    for participant in model_map.values():
        reporter = getattr(participant, "return_model_calls", None)
        if not callable(reporter):
            continue
        counters = reporter()
        if not isinstance(counters, Mapping):
            raise DriverError("USAGE_ACCOUNTING_UNAVAILABLE", "participant counters are invalid")
        for name, amount in counters.items():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise DriverError("USAGE_ACCOUNTING_UNAVAILABLE", "participant counter is invalid")
            if name in {"step_by_step_cot", "llm_qa"}:
                total += amount
    return total


def _snapshot(data: Any) -> Snapshot:
    try:
        titles = tuple(str(item) for item in data.get("titles", []))
        paragraphs = tuple(str(item) for item in data.get("paras", []))
        generated = tuple(str(item) for item in data.get("generated_sentences", []))
        question = str(data.get("question", ""))
    except (AttributeError, TypeError, ValueError) as exc:
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "upstream IRCoT state cannot be inspected",
            pre_execution=False,
        ) from exc
    if len(titles) != len(paragraphs):
        raise DriverError(
            "UPSTREAM_OUTPUT_INVALID",
            "upstream titles and paragraphs are unbalanced",
            pre_execution=False,
        )
    return Snapshot(question, titles, paragraphs, generated)


def _load_fixture_bindings(checkout: Checkout, request: Request) -> _FrozenBindings:
    if checkout.bindings_path is None:
        raise DriverError("MANIFEST_INVALID", "fixture profile lacks frozen bindings")
    try:
        value = json.loads(checkout.bindings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("FIXTURE_INVALID", "frozen bindings cannot be loaded") from exc
    if not isinstance(value, Mapping):
        raise DriverError("FIXTURE_INVALID", "frozen bindings root must be an object")
    bindings = _FrozenBindings(value, request.corpus_revision)
    if request.question != bindings.expected_question:
        raise DriverError("FIXTURE_QUESTION_MISMATCH", "question is outside frozen fixture")
    return bindings


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("MANIFEST_INVALID", "IRCoT manifest cannot be loaded") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise DriverError("MANIFEST_INVALID", "IRCoT manifest schema is invalid")
    upstream = value.get("upstream")
    if not isinstance(upstream, Mapping) or upstream.get("commit") != UPSTREAM_COMMIT:
        raise DriverError("MANIFEST_INVALID", "IRCoT manifest revision is invalid")
    for key in ("source_files", "allowed_model_names", "profiles", "patches"):
        if key not in value:
            raise DriverError("MANIFEST_INVALID", f"IRCoT manifest lacks {key}")
    return value, _sha256_bytes(raw)


def _verify_checkout(root: Path, manifest: Mapping[str, Any]) -> None:
    if _read_git_head(root) != UPSTREAM_COMMIT:
        raise DriverError("REVISION_MISMATCH", "IRCoT checkout is at the wrong commit")
    upstream = manifest["upstream"]
    if upstream.get("license_spdx") != "Apache-2.0":
        raise DriverError("MANIFEST_INVALID", "IRCoT license declaration is invalid")
    if _sha256_file(root / "LICENSE") != upstream.get("license_sha256"):
        raise DriverError("LICENSE_MISMATCH", "IRCoT Apache-2.0 license does not match pin")
    sources = manifest["source_files"]
    if not isinstance(sources, Mapping) or not sources:
        raise DriverError("MANIFEST_INVALID", "IRCoT source pins are invalid")
    for relative, expected in sources.items():
        if not isinstance(expected, str) or _sha256_file(_safe_file(root, relative)) != expected:
            raise DriverError("SOURCE_MISMATCH", "IRCoT source does not match manifest")
    if upstream.get("require_clean_checkout") is True:
        _require_clean_checkout(root)


def _profile_file(
    upstream_root: Path,
    worker_root: Path,
    profile: Mapping[str, Any],
    prefix: str,
) -> Path:
    scope = profile.get(f"{prefix}_scope")
    relative = profile.get(f"{prefix}_path")
    expected = profile.get(f"{prefix}_sha256")
    root = upstream_root if scope == "upstream" else worker_root if scope == "worker" else None
    if root is None or not isinstance(expected, str):
        raise DriverError("MANIFEST_INVALID", f"{prefix} artifact declaration is invalid")
    path = _safe_file(root, relative)
    if _sha256_file(path) != expected:
        raise DriverError(f"{prefix.upper()}_MISMATCH", f"pinned {prefix} checksum differs")
    return path


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _upstream_import_scope(root: Path, *, fixture_shims: bool) -> Iterator[None]:
    existing = {name for name in sys.modules if name == "commaqa" or name.startswith("commaqa.")}
    if existing:
        raise DriverError("IMPORT_STATE_UNSAFE", "commaqa modules were already loaded")
    shims = _fixture_shims() if fixture_shims else {}
    prior_modules = {name: sys.modules.get(name) for name in shims}
    for name, module in shims.items():
        sys.modules[name] = module
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for name in tuple(sys.modules):
            if name == "commaqa" or name.startswith("commaqa."):
                del sys.modules[name]
        for name, previous in prior_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        importlib.invalidate_caches()


def _fixture_shims() -> Mapping[str, types.ModuleType]:
    diskcache = types.ModuleType("diskcache")

    class Cache:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def memoize(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs

            def decorate(function: Any) -> Any:
                return function

            return decorate

    diskcache.Cache = Cache
    rapidfuzz = types.ModuleType("rapidfuzz")
    fuzz = types.ModuleType("rapidfuzz.fuzz")

    def ratio(left: str, right: str) -> float:
        return 100.0 * difflib.SequenceMatcher(None, left, right).ratio()

    fuzz.ratio = ratio
    fuzz.token_sort_ratio = lambda left, right: ratio(
        " ".join(sorted(left.split())),
        " ".join(sorted(right.split())),
    )
    rapidfuzz.fuzz = fuzz
    ftfy = types.ModuleType("ftfy")
    ftfy.fix_text = lambda value: value
    dateutil = types.ModuleType("dateutil")
    dateutil_parser = types.ModuleType("dateutil.parser")

    def unsupported_date_parse(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError("date parsing is unavailable in the frozen IRCoT fixture")

    dateutil_parser.parse = unsupported_date_parse
    dateutil.parser = dateutil_parser
    jsonnet = types.ModuleType("_jsonnet")
    jsonnet.evaluate_file = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Jsonnet is unavailable in the frozen fixture")
    )
    nltk = types.ModuleType("nltk")
    nltk_corpus = types.ModuleType("nltk.corpus")
    nltk_stem = types.ModuleType("nltk.stem")
    nltk_porter = types.ModuleType("nltk.stem.porter")

    class Stopwords:
        @staticmethod
        def words(language: str) -> list[str]:
            del language
            return []

    class PorterStemmer:
        @staticmethod
        def stem(token: str) -> str:
            return token

    nltk.word_tokenize = lambda value: value.split()
    nltk_corpus.stopwords = Stopwords()
    nltk_porter.PorterStemmer = PorterStemmer
    nltk.corpus = nltk_corpus
    nltk.stem = nltk_stem
    nltk_stem.porter = nltk_porter
    spacy = types.ModuleType("spacy")

    class Sentence:
        def __init__(self, text: str) -> None:
            self.text = text
            self.text_with_ws = text

    class Document:
        def __init__(self, text: str) -> None:
            self.sents = [Sentence(text)] if text else []

    spacy.load = lambda name: lambda text: Document(text)
    return {
        "diskcache": diskcache,
        "rapidfuzz": rapidfuzz,
        "rapidfuzz.fuzz": fuzz,
        "ftfy": ftfy,
        "dateutil": dateutil,
        "dateutil.parser": dateutil_parser,
        "_jsonnet": jsonnet,
        "nltk": nltk,
        "nltk.corpus": nltk_corpus,
        "nltk.stem": nltk_stem,
        "nltk.stem.porter": nltk_porter,
        "spacy": spacy,
    }


def _read_git_head(root: Path) -> str:
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            head = (
                _safe_file(root / ".git", head.removeprefix("ref: "))
                .read_text(encoding="utf-8")
                .strip()
            )
    except (OSError, UnicodeDecodeError) as exc:
        raise DriverError("REVISION_MISMATCH", "IRCoT checkout HEAD is unavailable") from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise DriverError("REVISION_MISMATCH", "IRCoT checkout HEAD is invalid")
    return head


def _require_clean_checkout(root: Path) -> None:
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
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DriverError("GIT_VERIFICATION_FAILED", "IRCoT git check is unavailable") from exc
    if result.returncode != 0:
        raise DriverError("GIT_VERIFICATION_FAILED", "IRCoT git check failed")
    if result.stdout.strip():
        raise DriverError("DIRTY_CHECKOUT", "IRCoT checkout contains uncommitted files")


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise DriverError("PATH_INVALID", "manifest path is invalid")
    candidate = (root.resolve() / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DriverError("PATH_INVALID", "manifest path escapes its artifact root") from exc
    if not candidate.is_file():
        raise DriverError("PATH_INVALID", "manifest path is not a file")
    return candidate


def _canonical_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DriverError("INVALID_REQUEST", f"{label} must be a mapping")
    _reject_secret_fields(value)
    try:
        encoded = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise DriverError("INVALID_REQUEST", f"{label} must be canonical JSON") from exc
    if len(encoded.encode("utf-8")) > 128 * 1024:
        raise DriverError("INVALID_REQUEST", f"{label} exceeds size limit")
    return json.loads(encoded)


def _reject_secret_fields(value: Mapping[str, Any]) -> None:
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise DriverError("INVALID_REQUEST", "fingerprint keys must be strings")
                normalized = key.lower().replace("-", "_")
                if (
                    any(part in normalized for part in _SECRET_KEY_PARTS)
                    or normalized in _SECRET_TOKEN_KEYS
                ):
                    raise DriverError("SECRET_REJECTED", "fingerprints must not contain secrets")
                pending.append(item)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            pending.extend(current)


def _mapping_sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", f"{label} must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, Mapping) for item in result):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", f"{label} must contain mappings")
    return result  # type: ignore[return-value]


def _required_text(value: Mapping[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise DriverError("FIXTURE_INVALID", f"{label}.{key} must be text")
    return item


def _bounded_text(value: Any, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise DriverError("INVALID_REQUEST", f"{label} must be bounded non-empty text")
    return value


def _positive_integer(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise DriverError("INVALID_REQUEST", f"{label} must be a bounded positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("FIXTURE_INVALID", f"{label} must be non-negative")
    return value


def _optional_finite_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DriverError("UPSTREAM_OUTPUT_INVALID", f"{label} must be finite")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise DriverError("INVALID_REQUEST", f"{label} must be boolean")
    return value


def _enum_text(value: Any, label: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise DriverError("INVALID_REQUEST", f"{label} is unsupported")
    return value


def _string_set(value: Any, label: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("MANIFEST_INVALID", f"{label} must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise DriverError("MANIFEST_INVALID", f"{label} must contain strings")
    return frozenset(result)


def _dependency_error(error: ModuleNotFoundError) -> DriverError:
    name = error.name if isinstance(error.name, str) else "optional_dependency"
    return DriverError("DEPENDENCY_MISSING", f"required upstream dependency is missing: {name}")


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DriverError("ARTIFACT_MISSING", "pinned IRCoT artifact is unavailable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def serve_jsonl(driver: IRCoTDriver, source: TextIO, sink: TextIO) -> int:
    for line in source:
        if len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
            response: Mapping[str, Any] = {
                "status": "error",
                "failure_code": "REQUEST_TOO_LARGE",
                "message": "IRCoT request exceeds byte limit",
                "protocol_version": PROTOCOL_VERSION,
                "consumed": False,
            }
        else:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                response = {
                    "status": "error",
                    "failure_code": "INVALID_JSON",
                    "message": "IRCoT request is not valid JSON",
                    "protocol_version": PROTOCOL_VERSION,
                    "consumed": False,
                }
            else:
                response = driver.safe_call(value)
        sink.write(_canonical_json(response) + "\n")
        sink.flush()
    return 0


def main() -> int:
    driver = IRCoTDriver(isolated_process=True)
    return serve_jsonl(driver, sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
