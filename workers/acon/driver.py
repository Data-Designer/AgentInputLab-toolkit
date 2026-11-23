#!/usr/bin/env python3
"""JSONL worker for the pinned microsoft/acon context optimizers.

The worker is intentionally a thin execution boundary.  It validates the
checkout and prompt artifacts, then directly instantiates and calls upstream
``HistoryOptimizer.process`` or ``ObservationOptimizer.process``.  It does not
copy or reimplement either algorithm's prompt construction or output parser.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

UPSTREAM_COMMIT = "d63f9ae18959dc7215ff62899c94c5e8c56847ae"
MANIFEST_SCHEMA = "agentinputlab.acon.compatibility.v1"
HISTORY_PROTOCOL = "agentinputlab.acon.history.v2"
OBSERVATION_PROTOCOL = "agentinputlab.acon.observation.v2"
HISTORY_RELEASE = "curate.acon.history_runtime"
OBSERVATION_RELEASE = "curate.acon.observation_runtime"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")

CHECKOUT_ENV = "AGENTINPUTLAB_ACON_CHECKOUT"
UPSTREAM_ROOT_ENV = "AGENTINPUTLAB_UPSTREAM_ROOT"
BASE_URL_ENV = "AGENTINPUTLAB_ACON_OPENAI_BASE_URL"
API_KEY_ENV = "AGENTINPUTLAB_ACON_API_KEY"
SERVING_MODEL_ENV = "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL"
OPTIMIZER_REVISION_ENV = "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL_REVISION"
TARGET_REVISION_ENV = "AGENTINPUTLAB_ACON_TARGET_MODEL_REVISION"
TOKENIZER_REVISION_ENV = "AGENTINPUTLAB_ACON_TOKENIZER_REVISION"
TEMPERATURE_ENV = "AGENTINPUTLAB_ACON_TEMPERATURE"
TIMEOUT_ENV = "AGENTINPUTLAB_ACON_TIMEOUT_SECONDS"
MAX_TOKENS_ENV = "AGENTINPUTLAB_ACON_MAX_TOKENS"
EXTRA_BODY_ENV = "AGENTINPUTLAB_ACON_EXTRA_BODY_JSON"
RESPONSE_ALIASES_ENV = "AGENTINPUTLAB_ACON_RESPONSE_MODEL_ALIASES_JSON"

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_FIELDS = frozenset(
    {
        "method_id",
        "release_id",
        "upstream_commit",
        "state_hash",
        "payload",
        "dependency_fingerprints",
        "minimum_reservation",
        "protocol_version",
        "request_id",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "optimizer_model_revision",
        "target_model_revision",
        "tokenizer_revision",
        "prompt_artifact_hash",
        "guideline_artifact_hash",
        "optimizer_deployment_fingerprint_hash",
        "target_backend_fingerprint_hash",
        "tokenizer_fingerprint_hash",
    }
)
_HISTORY_PAYLOAD_FIELDS = frozenset(
    {
        "task",
        "raw_messages",
        "history_text",
        "history_text_hash",
        "counted_material_hash",
        "previous_history_summary",
        "preserved_message_indices",
        "preserved_messages",
        "controls",
        "artifacts",
        "seed",
        "phase",
        "decision_token",
    }
)
_OBSERVATION_PAYLOAD_FIELDS = frozenset(
    {
        "task",
        "observation",
        "source_context_id",
        "history_text",
        "history_text_hash",
        "raw_messages",
        "opt_args",
        "opt_args_hash",
        "controls",
        "artifacts",
        "seed",
        "phase",
        "decision_token",
    }
)


class DriverError(RuntimeError):
    """A stable, secret-free worker failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        consumed: bool = False,
        usage: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed
        self.usage = usage


@dataclass(frozen=True, slots=True)
class Checkout:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    request_id: str
    protocol_version: str
    mode: str
    payload: Mapping[str, Any]
    state_hash: str
    dependencies: Mapping[str, Any]
    reservation: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PromptProfile:
    profile_id: str
    mode: str
    prompt_dir: Path
    system_template: str
    user_template: str
    prompt_artifact_hash: str
    guideline_artifact_hash: str


LLMFactory = Callable[..., Any]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        raise DriverError(
            "MODEL_REDIRECT_REFUSED",
            "optimizer endpoint redirects are disabled",
            consumed=True,
        )


class OpenAICompatibleLLM:
    """The small ``generate`` seam expected by pinned ACON.

    Both hosted OpenAI-compatible APIs and local vLLM servers use the same
    request shape.  Provider-specific retries, hidden defaults, and fallback
    models are deliberately excluded from this reproducibility boundary.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        seed: int,
        default_temperature: float,
        timeout_seconds: float,
        max_tokens: int | None,
        extra_body: Mapping[str, Any] | None = None,
        response_model_aliases: Sequence[str] = (),
    ) -> None:
        self.endpoint = _chat_completions_endpoint(base_url)
        if api_key and urllib.parse.urlsplit(self.endpoint).scheme != "https":
            raise DriverError(
                "MODEL_CONFIGURATION_INVALID",
                "optimizer API keys may only be sent over HTTPS",
            )
        self.api_key = api_key
        self.model = model
        self.seed = seed
        self.default_temperature = default_temperature
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        try:
            normalized_extra = json.loads(_canonical_json(dict(extra_body or {})))
        except (TypeError, ValueError) as exc:
            raise DriverError(
                "MODEL_CONFIGURATION_INVALID",
                "optimizer extra body must be canonical JSON",
            ) from exc
        reserved = set(normalized_extra) & {
            "model",
            "messages",
            "temperature",
            "seed",
            "max_tokens",
        }
        if reserved:
            raise DriverError(
                "MODEL_CONFIGURATION_INVALID",
                "optimizer extra body cannot override protocol fields",
            )
        self.extra_body = normalized_extra
        self.response_model_aliases = _response_aliases(response_model_aliases)
        self.system_message = ""
        self.total_model_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def fingerprint(self) -> Mapping[str, Any]:
        identity = {
            "transport": "openai_compatible_no_redirect_v1",
            "endpoint": self.endpoint,
            "model": self.model,
            "temperature": self.default_temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            identity["extra_body_hash"] = _content_hash(self.extra_body)
        if self.response_model_aliases:
            identity["response_model_aliases"] = list(self.response_model_aliases)
        return identity

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        if not isinstance(prompt, str):
            raise DriverError(
                "UPSTREAM_API_MISMATCH",
                "pinned ACON v1 produced a non-text optimizer prompt",
            )
        messages: list[dict[str, str]] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            "temperature": (
                self.default_temperature if temperature is None else float(temperature)
            ),
            "seed": self.seed,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        encoded = _canonical_json(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers=headers,
            method="POST",
        )
        self.total_model_calls += 1
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except DriverError:
            raise
        except urllib.error.HTTPError as exc:
            raise DriverError(
                "MODEL_REQUEST_FAILED",
                f"optimizer endpoint returned HTTP {exc.code}",
                consumed=True,
                usage=_usage(self),
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DriverError(
                "MODEL_REQUEST_FAILED",
                "optimizer endpoint request failed",
                consumed=True,
                usage=_usage(self),
            ) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise DriverError(
                "MODEL_RESPONSE_TOO_LARGE",
                "optimizer endpoint response exceeded its byte limit",
                consumed=True,
                usage=_usage(self),
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DriverError(
                "MODEL_RESPONSE_INVALID",
                "optimizer endpoint returned invalid JSON",
                consumed=True,
                usage=_usage(self),
            ) from exc
        input_tokens, output_tokens = _parse_model_usage(value)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        text, _, _ = _parse_model_response(value, expected_model=self.model,
            response_model_aliases=self.response_model_aliases)
        return text


class ACONDriver:
    """Validate one request and execute the corresponding upstream optimizer."""

    def __init__(
        self,
        *,
        manifest_path: Path = DEFAULT_MANIFEST,
        environment: Mapping[str, str] | None = None,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.environment = dict(os.environ if environment is None else environment)
        self.llm_factory = llm_factory

    def smoke_upstream_core(self) -> Mapping[str, str]:
        checkout = self._checkout()
        with (
            _upstream_import_scope(checkout.root),
            contextlib.redirect_stdout(sys.stderr),
        ):
            history_class, observation_class = _load_upstream_classes()
        return {
            "upstream_commit": UPSTREAM_COMMIT,
            "history_symbol": (f"{history_class.__module__}.{history_class.__name__}"),
            "observation_symbol": (f"{observation_class.__module__}.{observation_class.__name__}"),
            "manifest_sha256": checkout.manifest_sha256,
        }

    def run(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        request = _parse_request(value)
        checkout = self._checkout()
        artifacts = _parse_artifacts(request.payload.get("artifacts"))
        profile = _select_profile(checkout, request.mode, artifacts)
        _validate_bound_revisions(self.environment, artifacts)
        target_hash = _validate_target_dependency(request.dependencies, artifacts)
        temperature = _environment_float(
            self.environment,
            TEMPERATURE_ENV,
            default=0.0,
            minimum=0.0,
            maximum=2.0,
        )
        llm = self._make_llm(artifacts, request.payload["seed"], temperature)
        optimizer_hash = _fingerprint_hash(llm, "optimizer LLM")
        if optimizer_hash != artifacts["optimizer_deployment_fingerprint_hash"]:
            raise DriverError(
                "ARTIFACT_REVISION_MISMATCH",
                "optimizer deployment fingerprint disagrees with the request",
            )
        execute = request.payload["phase"] == "execute"

        try:
            with (
                _upstream_import_scope(checkout.root),
                contextlib.redirect_stdout(sys.stderr),
            ):
                history_class, observation_class = _load_upstream_classes()
                runner = _run_history if request.mode == "history" else _run_observation
                optimizer_class = history_class if request.mode == "history" else observation_class
                decision_payload = runner(
                    request.payload,
                    profile,
                    optimizer_class,
                    llm,
                    temperature,
                    execute=False,
                )

                tokenizer_hash = str(decision_payload.pop("_tokenizer_fingerprint_hash"))
                if tokenizer_hash != artifacts["tokenizer_fingerprint_hash"]:
                    raise DriverError(
                        "ARTIFACT_REVISION_MISMATCH",
                        "runtime tokenizer fingerprint disagrees with the request",
                    )
                attestation = {
                    "optimizer_deployment_fingerprint_hash": optimizer_hash,
                    "target_backend_fingerprint_hash": target_hash,
                    "tokenizer_fingerprint_hash": tokenizer_hash,
                }
                decision_token = _decision_token(
                    request,
                    profile=profile,
                    manifest_sha256=checkout.manifest_sha256,
                    attestation=attestation,
                    payload=decision_payload,
                )
                if execute and request.payload["decision_token"] != decision_token:
                    raise DriverError(
                        "DECISION_TOKEN_MISMATCH",
                        "ACON execute request does not match its preflight decision",
                    )
                if execute:
                    payload = runner(
                        request.payload,
                        profile,
                        optimizer_class,
                        llm,
                        temperature,
                        execute=True,
                    )
                    execute_tokenizer_hash = str(payload.pop("_tokenizer_fingerprint_hash"))
                    if execute_tokenizer_hash != tokenizer_hash or any(
                        payload[key] != decision_payload[key]
                        for key in ("raw_input_tokens", "threshold_triggered", "applied")
                    ):
                        raise DriverError(
                            "UPSTREAM_SEMANTICS_MISMATCH",
                            "ACON decision changed between preflight and execution",
                        )
                else:
                    payload = decision_payload
        except DriverError as exc:
            calls = _usage_or_none(llm)
            if calls is not None and calls[0] > 0 and exc.usage is None:
                exc.consumed = True
                exc.usage = calls
            raise

        calls, input_tokens, output_tokens = _usage(llm)
        expected_calls = 1 if execute else 0
        if calls != expected_calls:
            raise DriverError(
                "USAGE_ACCOUNTING_MISMATCH",
                "optimizer model-call accounting disagrees with the ACON branch",
            )
        payload = {
            **payload,
            "decision_token": decision_token,
            "runtime_attestation": attestation,
            "profile_id": profile.profile_id,
            "compatibility_manifest_sha256": checkout.manifest_sha256,
        }
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": request.protocol_version,
            "payload": payload,
            "usage": {
                "model_calls": calls,
                "retrieval_calls": 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    def _checkout(self) -> Checkout:
        root = _resolve_checkout(self.environment)
        manifest, manifest_hash = _load_manifest(self.manifest_path)
        _verify_checkout(root, manifest)
        return Checkout(root=root, manifest=manifest, manifest_sha256=manifest_hash)

    def _make_llm(
        self,
        artifacts: Mapping[str, str],
        seed: int,
        temperature: float,
    ) -> Any:
        model = self.environment.get(SERVING_MODEL_ENV, "").strip()
        if not model:
            model = artifacts["optimizer_model_revision"]
        if self.llm_factory is not None:
            try:
                llm = self.llm_factory(
                    model=model,
                    seed=seed,
                    environment=dict(self.environment),
                    default_temperature=temperature,
                )
            except DriverError:
                raise
            except Exception as exc:
                raise DriverError(
                    "MODEL_CONFIGURATION_INVALID",
                    "optimizer LLM factory failed",
                ) from exc
            _validate_llm_seam(llm)
            return llm
        base_url = self.environment.get(BASE_URL_ENV, "").strip()
        if not base_url:
            raise DriverError(
                "MODEL_CONFIGURATION_MISSING",
                f"{BASE_URL_ENV} is required for ACON runtime attestation",
            )
        timeout = _environment_float(
            self.environment,
            TIMEOUT_ENV,
            default=120.0,
            minimum=0.1,
            maximum=3600.0,
        )
        max_tokens = _environment_optional_positive_int(self.environment, MAX_TOKENS_ENV)
        extra_body = _environment_json_mapping(self.environment, EXTRA_BODY_ENV)
        raw_aliases = self.environment.get(RESPONSE_ALIASES_ENV, "[]")
        try:aliases = _response_aliases(json.loads(raw_aliases))
        except (ValueError, TypeError):
            raise DriverError("MODEL_CONFIGURATION_INVALID", "response model aliases must be a unique JSON string array") from None
        return OpenAICompatibleLLM(
            base_url=base_url,
            api_key=self.environment.get(API_KEY_ENV, ""),
            model=model,
            seed=seed,
            default_temperature=temperature,
            timeout_seconds=timeout,
            max_tokens=max_tokens,
            extra_body=extra_body,
            response_model_aliases=aliases,
        )


def _parse_request(value: Mapping[str, Any]) -> ParsedRequest:
    if not isinstance(value, Mapping):
        raise DriverError("INVALID_REQUEST", "worker request must be a JSON object")
    if set(value) != _TOP_FIELDS:
        raise DriverError("INVALID_REQUEST", "worker request fields do not match schema")
    if value.get("method_id") != "acon":
        raise DriverError("INVALID_REQUEST", "worker request method_id must be acon")
    if value.get("upstream_commit") != UPSTREAM_COMMIT:
        raise DriverError("REVISION_MISMATCH", "request upstream revision is not pinned")
    release = value.get("release_id")
    protocol = value.get("protocol_version")
    if release == HISTORY_RELEASE and protocol == HISTORY_PROTOCOL:
        mode = "history"
        payload_fields = _HISTORY_PAYLOAD_FIELDS
    elif release == OBSERVATION_RELEASE and protocol == OBSERVATION_PROTOCOL:
        mode = "observation"
        payload_fields = _OBSERVATION_PAYLOAD_FIELDS
    else:
        raise DriverError(
            "SCHEMA_MISMATCH",
            "request release and protocol do not identify a supported ACON v2 mode",
        )
    request_id = _sha_text(value.get("request_id"), "request_id")
    state_hash = _sha_text(value.get("state_hash"), "state_hash")
    payload = value.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != payload_fields:
        raise DriverError("INVALID_REQUEST", "ACON payload fields do not match schema")
    dependencies = value.get("dependency_fingerprints")
    reservation = value.get("minimum_reservation")
    if not isinstance(dependencies, Mapping) or not isinstance(reservation, Mapping):
        raise DriverError("INVALID_REQUEST", "worker dependency metadata must be mappings")
    if set(dependencies) != {"target_backend"} or not isinstance(
        dependencies.get("target_backend"), Mapping
    ):
        raise DriverError("INVALID_REQUEST", "ACON requires one target_backend fingerprint")
    phase = payload.get("phase")
    decision_token = payload.get("decision_token")
    if phase == "preflight":
        if decision_token != "" or dict(reservation):
            raise DriverError("INVALID_REQUEST", "ACON preflight cannot reserve or carry a token")
    elif phase == "execute":
        _sha_text(decision_token, "decision_token")
        if dict(reservation) != {"model_calls": 1}:
            raise DriverError("INVALID_REQUEST", "ACON execute must reserve one model call")
    else:
        raise DriverError("INVALID_REQUEST", "ACON phase must be preflight or execute")
    try:
        expected_id = _content_hash(
            {
                "method_id": value["method_id"],
                "release_id": release,
                "upstream_commit": value["upstream_commit"],
                "state_hash": value["state_hash"],
                "payload": payload,
                "dependency_fingerprints": dependencies,
                "minimum_reservation": reservation,
                "protocol_version": protocol,
            }
        )
    except (TypeError, ValueError) as exc:
        raise DriverError("INVALID_REQUEST", "worker request is not canonical JSON") from exc
    if request_id != expected_id:
        raise DriverError("REQUEST_ID_MISMATCH", "worker request_id is not canonical")
    normalized = json.loads(_canonical_json(payload))
    normalized_dependencies = json.loads(_canonical_json(dependencies))
    normalized_reservation = json.loads(_canonical_json(reservation))
    if mode == "history":
        _validate_history_payload(normalized)
    else:
        _validate_observation_payload(normalized)
    return ParsedRequest(
        request_id=request_id,
        protocol_version=str(protocol),
        mode=mode,
        payload=normalized,
        state_hash=state_hash,
        dependencies=normalized_dependencies,
        reservation=normalized_reservation,
    )


def _validate_history_payload(payload: Mapping[str, Any]) -> None:
    _bounded_text(payload.get("task"), "task", allow_empty=False)
    messages = _messages(payload.get("raw_messages"))
    history = _bounded_text(payload.get("history_text"), "history_text")
    previous = _bounded_text(
        payload.get("previous_history_summary"),
        "previous_history_summary",
    )
    if _content_hash(history) != _sha_text(payload.get("history_text_hash"), "history_text_hash"):
        raise DriverError("SOURCE_MAPPING_MISMATCH", "history_text_hash does not match history")
    controls = _history_controls(payload.get("controls"))
    indices = _integer_sequence(payload.get("preserved_message_indices"), "preserved indices")
    if len(indices) != len(set(indices)) or any(index >= len(messages) for index in indices):
        raise DriverError("SOURCE_MAPPING_MISMATCH", "preserved message indices are invalid")
    preserved = _message_sequence(payload.get("preserved_messages"), "preserved_messages")
    expected_preserved = [messages[index] for index in indices]
    if _canonical_json(preserved) != _canonical_json(expected_preserved):
        raise DriverError("SOURCE_MAPPING_MISMATCH", "preserved messages do not match indices")
    expected_indices = _preserved_turn_indices(messages, controls["preserve_last_k_turns"])
    if tuple(indices) != expected_indices:
        raise DriverError(
            "SOURCE_MAPPING_MISMATCH", "preserved turn selection is not upstream-faithful"
        )
    expected_history = _history_material(messages, preserved)
    if history != expected_history:
        raise DriverError(
            "SOURCE_MAPPING_MISMATCH", "history text is not derived from raw messages"
        )
    counted = f"{previous}\n{history}" if previous else history
    counted_hash = _sha_text(payload.get("counted_material_hash"), "counted_material_hash")
    if _content_hash(counted) != counted_hash:
        raise DriverError("SOURCE_MAPPING_MISMATCH", "counted material hash does not match input")
    _parse_artifacts(payload.get("artifacts"))
    _seed(payload.get("seed"))


def _validate_observation_payload(payload: Mapping[str, Any]) -> None:
    _bounded_text(payload.get("task"), "task", allow_empty=False)
    observation = _bounded_text(payload.get("observation"), "observation", allow_empty=False)
    del observation
    _bounded_text(payload.get("source_context_id"), "source_context_id", allow_empty=False)
    history = _bounded_text(payload.get("history_text"), "history_text")
    history_hash = _sha_text(payload.get("history_text_hash"), "history_text_hash")
    if _content_hash(history) != history_hash:
        raise DriverError("SOURCE_MAPPING_MISMATCH", "history_text_hash does not match history")
    messages = _messages(payload.get("raw_messages"))
    controls = _observation_controls(payload.get("controls"))
    if not controls["add_previous_history_summary"]:
        expected_history = _history_text(messages)
        if history != expected_history:
            raise DriverError(
                "SOURCE_MAPPING_MISMATCH",
                "observation history text is not derived from raw messages",
            )
    opt_args = payload.get("opt_args")
    if not isinstance(opt_args, Mapping):
        raise DriverError("INVALID_REQUEST", "opt_args must be a JSON object")
    collisions = set(opt_args) & {"task", "observation", "history"}
    if collisions:
        raise DriverError(
            "INVALID_REQUEST",
            "opt_args cannot override reserved upstream inputs: " + ", ".join(sorted(collisions)),
        )
    try:
        opt_hash = _content_hash(opt_args)
    except (TypeError, ValueError) as exc:
        raise DriverError("INVALID_REQUEST", "opt_args must be canonical JSON") from exc
    if opt_hash != _sha_text(payload.get("opt_args_hash"), "opt_args_hash"):
        raise DriverError("SOURCE_MAPPING_MISMATCH", "opt_args_hash does not match opt_args")
    _parse_artifacts(payload.get("artifacts"))
    _seed(payload.get("seed"))


def _run_history(
    payload: Mapping[str, Any],
    profile: PromptProfile,
    optimizer_class: type[Any],
    llm: Any,
    temperature: float,
    *,
    execute: bool,
) -> Mapping[str, Any]:
    controls = _history_controls(payload["controls"])
    config = {
        "history_prompt_dir": str(profile.prompt_dir),
        "prompts": {
            "prompt_system": profile.system_template,
            "prompt_history_user": profile.user_template,
        },
        "history_summarization_threshold": controls["summarization_threshold"],
        "model": payload["artifacts"]["optimizer_model_revision"],
        "temperature": temperature,
        "use_llmlingua": False,
    }
    optimizer = _construct_optimizer(optimizer_class, config, llm, "HistoryOptimizer")
    counted = (
        f"{payload['previous_history_summary']}\n{payload['history_text']}"
        if payload["previous_history_summary"]
        else payload["history_text"]
    )
    raw_tokens = _token_count(optimizer, counted)
    try:
        threshold_triggered = optimizer.check_summarization_needed(
            payload["history_text"],
            payload["previous_history_summary"],
        )
    except Exception as exc:
        raise DriverError(
            "UPSTREAM_EXECUTION_FAILED",
            "upstream HistoryOptimizer threshold check failed",
        ) from exc
    if not isinstance(threshold_triggered, bool):
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "upstream HistoryOptimizer returned a non-boolean threshold decision",
        )
    expected_threshold = (
        controls["summarization_threshold"] == -1
        or raw_tokens > controls["summarization_threshold"]
    )
    if threshold_triggered is not expected_threshold:
        raise DriverError(
            "UPSTREAM_SEMANTICS_MISMATCH",
            "upstream HistoryOptimizer threshold semantics changed",
        )
    interval_ready = (
        controls["summary_interval"] == -1
        or len(payload["raw_messages"]) // 2 >= controls["summary_interval"]
    )
    applied = bool(payload["history_text"].strip()) and threshold_triggered and interval_ready
    if execute and not applied:
        raise DriverError(
            "DECISION_NOT_EXECUTABLE", "ACON history preflight selected a skip branch"
        )
    summary = ""
    raw_response = ""
    if applied and execute:
        before = _optimizer_history_size(optimizer)
        try:
            summary = optimizer.process(
                task=payload["task"],
                history=payload["history_text"],
                prev_history_summary=payload["previous_history_summary"],
                raw_history=payload["raw_messages"],
                opt_args=None,
            )
        except DriverError:
            raise
        except ModuleNotFoundError as exc:
            raise _dependency_error(exc) from exc
        except Exception as exc:
            raise DriverError(
                "UPSTREAM_EXECUTION_FAILED",
                "upstream HistoryOptimizer.process failed",
            ) from exc
        summary = _upstream_text(summary, "history summary")
        raw_response = _raw_optimizer_response(optimizer, before)
        if not summary.strip() or not raw_response.strip():
            raise DriverError(
                "UPSTREAM_OUTPUT_INVALID",
                "upstream HistoryOptimizer returned an empty applied result",
            )
    return {
        **dict(payload["artifacts"]),
        "counted_material_hash": payload["counted_material_hash"],
        "raw_input_tokens": raw_tokens,
        "threshold_triggered": threshold_triggered,
        "applied": applied,
        "summary": summary,
        "raw_response": raw_response,
        "_tokenizer_fingerprint_hash": _tokenizer_fingerprint_hash(optimizer),
    }


def _run_observation(
    payload: Mapping[str, Any],
    profile: PromptProfile,
    optimizer_class: type[Any],
    llm: Any,
    temperature: float,
    *,
    execute: bool,
) -> Mapping[str, Any]:
    controls = _observation_controls(payload["controls"])
    config = {
        "obs_prompt_dir": str(profile.prompt_dir),
        "prompts": {
            "prompt_system": profile.system_template,
            "prompt_user": profile.user_template,
        },
        "obs_summarization_threshold": controls["summarization_threshold"],
        "model": payload["artifacts"]["optimizer_model_revision"],
        "temperature": temperature,
        "use_llmlingua": False,
    }
    optimizer = _construct_optimizer(optimizer_class, config, llm, "ObservationOptimizer")
    raw_tokens = _token_count(optimizer, payload["observation"])
    try:
        threshold_triggered = optimizer.check_summarization_needed(payload["observation"])
    except Exception as exc:
        raise DriverError(
            "UPSTREAM_EXECUTION_FAILED",
            "upstream ObservationOptimizer threshold check failed",
        ) from exc
    if not isinstance(threshold_triggered, bool):
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "upstream ObservationOptimizer returned a non-boolean threshold decision",
        )
    expected_threshold = (
        controls["summarization_threshold"] == -1
        or raw_tokens > controls["summarization_threshold"]
    )
    if threshold_triggered is not expected_threshold:
        raise DriverError(
            "UPSTREAM_SEMANTICS_MISMATCH",
            "upstream ObservationOptimizer threshold semantics changed",
        )
    optimized = payload["observation"]
    raw_response = ""
    if execute and not threshold_triggered:
        raise DriverError(
            "DECISION_NOT_EXECUTABLE",
            "ACON observation preflight selected a skip branch",
        )
    if threshold_triggered and execute:
        before = _optimizer_history_size(optimizer)
        try:
            optimized = optimizer.process(
                task=payload["task"],
                observation=payload["observation"],
                history=payload["history_text"],
                raw_history=payload["raw_messages"],
                opt_args=payload["opt_args"],
            )
        except DriverError:
            raise
        except ModuleNotFoundError as exc:
            raise _dependency_error(exc) from exc
        except Exception as exc:
            raise DriverError(
                "UPSTREAM_EXECUTION_FAILED",
                "upstream ObservationOptimizer.process failed",
            ) from exc
        optimized = _upstream_text(optimized, "optimized observation")
        raw_response = _raw_optimizer_response(optimizer, before)
        if not optimized.strip() or not raw_response.strip():
            raise DriverError(
                "UPSTREAM_OUTPUT_INVALID",
                "upstream ObservationOptimizer returned an empty applied result",
            )
    return {
        **dict(payload["artifacts"]),
        "source_context_id": payload["source_context_id"],
        "history_text_hash": payload["history_text_hash"],
        "opt_args_hash": payload["opt_args_hash"],
        "raw_input_tokens": raw_tokens,
        "threshold_triggered": threshold_triggered,
        "applied": threshold_triggered,
        "optimized_observation": optimized,
        "raw_response": raw_response,
        "_tokenizer_fingerprint_hash": _tokenizer_fingerprint_hash(optimizer),
    }


def _construct_optimizer(
    optimizer_class: type[Any],
    config: Mapping[str, Any],
    llm: Any,
    label: str,
) -> Any:
    try:
        optimizer = optimizer_class(dict(config), debug_mode=False, llm=llm)
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc) from exc
    except Exception as exc:
        raise DriverError(
            "UPSTREAM_BOOT_FAILED",
            f"upstream {label} construction failed",
        ) from exc
    if not callable(getattr(optimizer, "process", None)):
        raise DriverError("UPSTREAM_API_MISMATCH", f"upstream {label}.process is unavailable")
    if not callable(getattr(optimizer, "count_tokens", None)) or not callable(
        getattr(optimizer, "check_summarization_needed", None)
    ):
        raise DriverError("UPSTREAM_API_MISMATCH", f"upstream {label} helpers are unavailable")
    try:
        llm.system_message = optimizer.system_message
    except Exception as exc:
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "optimizer LLM seam rejected the upstream system message",
        ) from exc
    return optimizer


def _token_count(optimizer: Any, text: str) -> int:
    try:
        value = optimizer.count_tokens(text)
    except Exception as exc:
        raise DriverError(
            "UPSTREAM_EXECUTION_FAILED",
            "upstream ACON token counting failed",
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "upstream ACON returned an invalid token count",
        )
    return value


def _optimizer_history_size(optimizer: Any) -> int:
    history = getattr(optimizer, "history", None)
    if not isinstance(history, list):
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "upstream optimizer interaction history is unavailable",
        )
    return len(history)


def _raw_optimizer_response(optimizer: Any, before: int) -> str:
    history = getattr(optimizer, "history", None)
    if not isinstance(history, list) or len(history) != before + 1:
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "upstream optimizer did not record exactly one interaction",
        )
    record = history[-1]
    if (
        isinstance(record, (str, bytes))
        or not isinstance(record, Sequence)
        or len(record) < 3
        or not isinstance(record[2], str)
    ):
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "upstream optimizer interaction record is incompatible",
        )
    return record[2]


def _usage(llm: Any) -> tuple[int, int, int]:
    values = (
        getattr(llm, "total_model_calls", None),
        getattr(llm, "total_input_tokens", None),
        getattr(llm, "total_output_tokens", None),
    )
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
        raise DriverError(
            "USAGE_ACCOUNTING_UNAVAILABLE",
            "optimizer LLM seam did not expose non-negative usage counters",
        )
    return values  # type: ignore[return-value]


def _usage_or_none(llm: Any) -> tuple[int, int, int] | None:
    try:
        return _usage(llm)
    except DriverError:
        return None


def _validate_llm_seam(llm: Any) -> None:
    if not callable(getattr(llm, "generate", None)):
        raise DriverError("MODEL_CONFIGURATION_INVALID", "optimizer LLM lacks generate()")
    _usage(llm)
    _fingerprint_hash(llm, "optimizer LLM")


def _fingerprint_hash(value: Any, label: str) -> str:
    fingerprint = getattr(value, "fingerprint", None)
    if not callable(fingerprint):
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{label} lacks fingerprint()")
    identity = fingerprint()
    if not isinstance(identity, Mapping) or not identity:
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{label} fingerprint is invalid")
    try:
        return _content_hash(identity)
    except (TypeError, ValueError) as exc:
        raise DriverError(
            "MODEL_CONFIGURATION_INVALID", f"{label} fingerprint is not canonical"
        ) from exc


def _decision_token(
    request: ParsedRequest,
    *,
    profile: PromptProfile,
    manifest_sha256: str,
    attestation: Mapping[str, str],
    payload: Mapping[str, Any],
) -> str:
    return _content_hash(
        {
            "mode": request.mode,
            "state_hash": request.state_hash,
            "payload": {
                key: item
                for key, item in request.payload.items()
                if key not in {"phase", "decision_token"}
            },
            "dependencies": request.dependencies,
            "profile_id": profile.profile_id,
            "manifest_sha256": manifest_sha256,
            "runtime_attestation": attestation,
            "raw_input_tokens": payload["raw_input_tokens"],
            "threshold_triggered": payload["threshold_triggered"],
            "applied": payload["applied"],
        }
    )


def _validate_target_dependency(
    dependencies: Mapping[str, Any], artifacts: Mapping[str, str]
) -> str:
    target = dependencies.get("target_backend")
    try:
        digest = _content_hash(target)
    except (TypeError, ValueError) as exc:  # pragma: no cover - parser already canonicalizes
        raise DriverError("INVALID_REQUEST", "target backend fingerprint is not canonical") from exc
    if digest != artifacts["target_backend_fingerprint_hash"]:
        raise DriverError(
            "ARTIFACT_REVISION_MISMATCH",
            "target backend fingerprint disagrees with the request artifacts",
        )
    if artifacts["target_model_revision"] not in _text_leaves(target):
        raise DriverError(
            "ARTIFACT_REVISION_MISMATCH",
            "target model revision is absent from its backend fingerprint",
        )
    return digest


def _text_leaves(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return set().union(*(_text_leaves(item) for item in value.values()), set())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return set().union(*(_text_leaves(item) for item in value), set())
    return set()


def _tokenizer_fingerprint_hash(optimizer: Any) -> str:
    tokenizer = getattr(optimizer, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise DriverError(
            "TOKENIZER_ATTESTATION_UNAVAILABLE",
            "pinned ACON tokenizer is unavailable; approximate counting is forbidden",
        )
    custom = getattr(tokenizer, "agentinputlab_fingerprint", None)
    if callable(custom):
        identity = custom()
    else:
        name = getattr(tokenizer, "name", None)
        if not isinstance(name, str) or not name:
            raise DriverError("TOKENIZER_ATTESTATION_UNAVAILABLE", "tokenizer name is unavailable")
        try:
            version = importlib.metadata.version("tiktoken")
            probes = [
                list(tokenizer.encode(text)) for text in ("AgentInputLab", "中文", "line\nbreak")
            ]
        except Exception as exc:
            raise DriverError(
                "TOKENIZER_ATTESTATION_UNAVAILABLE",
                "tiktoken version or deterministic probe is unavailable",
            ) from exc
        identity = {
            "engine": "tiktoken",
            "version": version,
            "encoding": name,
            "probe_hash": _content_hash(probes),
        }
    if not isinstance(identity, Mapping) or not identity:
        raise DriverError("TOKENIZER_ATTESTATION_UNAVAILABLE", "tokenizer fingerprint is invalid")
    return _content_hash(identity)


def _response_aliases(value):
    if (not isinstance(value, (tuple, list)) or
        any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in value) or
        len(set(value)) != len(value)):
        raise DriverError("MODEL_CONFIGURATION_INVALID", "response model aliases must be a unique string array")
    return tuple(value)


def _parse_model_response(value: Any, *, expected_model: str,
    response_model_aliases: Sequence[str] = ()) -> tuple[str, int, int]:
    if not isinstance(value, Mapping):
        raise DriverError("MODEL_RESPONSE_INVALID", "optimizer response must be an object")
    response_model = value.get("model")
    if not isinstance(response_model, str) or response_model not in {expected_model, *_response_aliases(response_model_aliases)}:
        raise DriverError(
            "MODEL_RESPONSE_IDENTITY_MISMATCH",
            "optimizer response model does not match the configured serving alias",
        )
    choices = value.get("choices")
    if (
        isinstance(choices, (str, bytes))
        or not isinstance(choices, Sequence)
        or len(choices) != 1
        or not isinstance(choices[0], Mapping)
    ):
        raise DriverError(
            "MODEL_RESPONSE_INVALID",
            "optimizer response must contain exactly one choice",
        )
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise DriverError("MODEL_RESPONSE_INVALID", "optimizer response content is missing")
    text = message["content"]
    if len(text) > MAX_TEXT_CHARS:
        raise DriverError("MODEL_RESPONSE_TOO_LARGE", "optimizer text exceeded its size limit")
    input_tokens, output_tokens = _parse_model_usage(value)
    return text, input_tokens, output_tokens


def _parse_model_usage(value: Any) -> tuple[int, int]:
    usage = value.get("usage") if isinstance(value, Mapping) else None
    if not isinstance(usage, Mapping):
        raise DriverError(
            "USAGE_ACCOUNTING_UNAVAILABLE",
            "optimizer endpoint omitted token usage",
        )
    counters = (usage.get("prompt_tokens"), usage.get("completion_tokens"))
    if any(
        isinstance(counter, bool) or not isinstance(counter, int) or counter < 0
        for counter in counters
    ):
        raise DriverError(
            "USAGE_ACCOUNTING_UNAVAILABLE",
            "optimizer endpoint returned invalid token usage",
        )
    input_tokens, output_tokens = counters
    return input_tokens, output_tokens


def _load_upstream_classes() -> tuple[type[Any], type[Any]]:
    try:
        history_module = importlib.import_module("productive_agents.ctxopt.history_optimizer")
        observation_module = importlib.import_module("productive_agents.ctxopt.obs_optimizer")
    except ModuleNotFoundError as exc:
        raise _dependency_error(exc) from exc
    history_class = getattr(history_module, "HistoryOptimizer", None)
    observation_class = getattr(observation_module, "ObservationOptimizer", None)
    if not isinstance(history_class, type) or not isinstance(observation_class, type):
        raise DriverError(
            "UPSTREAM_API_MISMATCH",
            "pinned ACON optimizer classes are unavailable",
        )
    return history_class, observation_class


@contextlib.contextmanager
def _upstream_import_scope(root: Path) -> Iterator[None]:
    source = str(root / "src")
    previous_prefix = sys.pycache_prefix
    previous_dont_write = sys.dont_write_bytecode
    previous = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "productive_agents" or name.startswith("productive_agents.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="agentinputlab-acon-pycache-") as cache:
        sys.pycache_prefix = cache
        sys.dont_write_bytecode = True
        sys.path.insert(0, source)
        try:
            yield
        finally:
            for name in tuple(sys.modules):
                if name == "productive_agents" or name.startswith("productive_agents."):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_dont_write
            with contextlib.suppress(ValueError):
                sys.path.remove(source)


def _resolve_checkout(environment: Mapping[str, str]) -> Path:
    explicit = environment.get(CHECKOUT_ENV, "").strip()
    root_value = explicit or environment.get(UPSTREAM_ROOT_ENV, "").strip()
    if not root_value:
        raise DriverError(
            "UPSTREAM_ROOT_MISSING",
            f"{CHECKOUT_ENV} or {UPSTREAM_ROOT_ENV} must identify the pinned ACON checkout",
        )
    candidate = Path(root_value).expanduser().resolve()
    if not explicit and (candidate / "acon").is_dir():
        candidate = (candidate / "acon").resolve()
    if not candidate.is_dir():
        raise DriverError("UPSTREAM_ROOT_INVALID", "upstream checkout directory is unavailable")
    return candidate


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
    if upstream.get("license_spdx") != "MIT":
        raise DriverError("MANIFEST_INVALID", "compatibility manifest license is invalid")
    _sha_text(upstream.get("license_sha256"), "manifest license_sha256", manifest=True)
    if not isinstance(upstream.get("require_clean_checkout"), bool):
        raise DriverError("MANIFEST_INVALID", "manifest checkout policy is invalid")
    minimum_python = upstream.get("minimum_python")
    if (
        isinstance(minimum_python, (str, bytes))
        or not isinstance(minimum_python, Sequence)
        or len(minimum_python) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in minimum_python)
    ):
        raise DriverError("MANIFEST_INVALID", "manifest minimum_python is invalid")
    if tuple(sys.version_info[:2]) < tuple(minimum_python):
        raise DriverError(
            "PYTHON_VERSION_UNSUPPORTED",
            "pinned ACON requires a newer isolated Python worker environment",
        )
    source_files = value.get("source_files")
    profiles = value.get("prompt_profiles")
    if not isinstance(source_files, Mapping) or not source_files:
        raise DriverError("MANIFEST_INVALID", "manifest source_files is invalid")
    if not isinstance(profiles, Mapping) or not profiles:
        raise DriverError("MANIFEST_INVALID", "manifest prompt_profiles is invalid")
    for relative, digest in source_files.items():
        _manifest_path_hash(relative, digest, "source file")
    modes: set[str] = set()
    for profile_id, raw_profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise DriverError("MANIFEST_INVALID", "manifest profile ID is invalid")
        profile = _manifest_profile(raw_profile)
        modes.add(profile["mode"])
        expected_bundle = _sha256_bytes(_canonical_json(profile["prompt_files"]).encode("utf-8"))
        if profile["prompt_artifact_hash"] != expected_bundle:
            raise DriverError("MANIFEST_INVALID", "manifest prompt bundle hash is invalid")
        guideline_path = f"{profile['prompt_dir']}/{profile['user_template']}.jinja"
        if profile["prompt_files"].get(guideline_path) != profile["guideline_artifact_hash"]:
            raise DriverError("MANIFEST_INVALID", "manifest guideline hash is invalid")
    if modes != {"history", "observation"}:
        raise DriverError("MANIFEST_INVALID", "manifest must contain both ACON optimizer modes")
    return value, _sha256_bytes(raw)


def _verify_checkout(root: Path, manifest: Mapping[str, Any]) -> None:
    if _read_git_head(root) != UPSTREAM_COMMIT:
        raise DriverError("UPSTREAM_REVISION_MISMATCH", "upstream checkout is not pinned")
    upstream = manifest["upstream"]
    if _sha256_file(_safe_file(root, "LICENSE"), "LICENSE") != upstream["license_sha256"]:
        raise DriverError("UPSTREAM_LICENSE_MISMATCH", "upstream LICENSE hash changed")
    for relative, expected in manifest["source_files"].items():
        if _sha256_file(_safe_file(root, relative), "source") != expected:
            raise DriverError("UPSTREAM_SOURCE_MISMATCH", "pinned upstream source hash changed")
    checked: set[str] = set()
    for raw_profile in manifest["prompt_profiles"].values():
        profile = _manifest_profile(raw_profile)
        for relative, expected in profile["prompt_files"].items():
            if relative in checked:
                continue
            if _sha256_file(_safe_file(root, relative), "prompt") != expected:
                raise DriverError(
                    "UPSTREAM_PROMPT_MISMATCH",
                    "pinned upstream prompt artifact hash changed",
                )
            checked.add(relative)
    if upstream.get("require_clean_checkout") is True:
        _verify_clean_checkout(root)


def _read_git_head(root: Path) -> str:
    git_dir = root / ".git"
    if git_dir.is_file():
        try:
            marker = git_dir.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DriverError(
                "UPSTREAM_GIT_INVALID", "upstream git metadata is unavailable"
            ) from exc
        if not marker.startswith("gitdir: "):
            raise DriverError("UPSTREAM_GIT_INVALID", "upstream gitdir marker is invalid")
        git_dir = (root / marker[8:]).resolve()
    if not git_dir.is_dir():
        raise DriverError("UPSTREAM_GIT_INVALID", "upstream .git metadata is unavailable")
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DriverError("UPSTREAM_GIT_INVALID", "upstream HEAD is unavailable") from exc
    if re.fullmatch(r"[0-9a-f]{40}", head):
        return head
    if not head.startswith("ref: "):
        raise DriverError("UPSTREAM_GIT_INVALID", "upstream HEAD is invalid")
    reference = head[5:]
    if ".." in Path(reference).parts or not reference.startswith("refs/"):
        raise DriverError("UPSTREAM_GIT_INVALID", "upstream HEAD reference is invalid")
    ref_path = git_dir / reference
    try:
        if ref_path.is_file():
            resolved = ref_path.read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[0-9a-f]{40}", resolved):
                return resolved
        packed = (git_dir / "packed-refs").read_text(encoding="utf-8")
    except OSError as exc:
        raise DriverError("UPSTREAM_GIT_INVALID", "upstream HEAD reference is unavailable") from exc
    for line in packed.splitlines():
        if line.startswith(("#", "^")):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1] == reference and re.fullmatch(r"[0-9a-f]{40}", parts[0]):
            return parts[0]
    raise DriverError("UPSTREAM_GIT_INVALID", "upstream HEAD reference is unresolved")


def _verify_clean_checkout(root: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DriverError(
            "UPSTREAM_GIT_INVALID",
            "upstream cleanliness check could not run",
        ) from exc
    if result.returncode != 0:
        raise DriverError("UPSTREAM_GIT_INVALID", "upstream cleanliness check failed")
    if result.stdout.strip():
        raise DriverError("UPSTREAM_CHECKOUT_DIRTY", "upstream checkout contains local changes")


def _select_profile(
    checkout: Checkout,
    mode: str,
    artifacts: Mapping[str, str],
) -> PromptProfile:
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for profile_id, raw_profile in checkout.manifest["prompt_profiles"].items():
        profile = _manifest_profile(raw_profile)
        if (
            profile["mode"] == mode
            and profile["prompt_artifact_hash"] == artifacts["prompt_artifact_hash"]
            and profile["guideline_artifact_hash"] == artifacts["guideline_artifact_hash"]
        ):
            matches.append((profile_id, profile))
    if len(matches) != 1:
        raise DriverError(
            "PROMPT_ARTIFACT_MISMATCH",
            "request prompt and guideline hashes do not select one pinned ACON profile",
        )
    profile_id, raw = matches[0]
    prompt_dir = (checkout.root / raw["prompt_dir"]).resolve()
    try:
        prompt_dir.relative_to(checkout.root)
    except ValueError as exc:  # pragma: no cover - manifest validation invariant
        raise DriverError("MANIFEST_INVALID", "manifest prompt directory escaped checkout") from exc
    if not prompt_dir.is_dir():
        raise DriverError("UPSTREAM_PROMPT_MISMATCH", "pinned prompt directory is unavailable")
    return PromptProfile(
        profile_id=profile_id,
        mode=mode,
        prompt_dir=prompt_dir,
        system_template=raw["system_template"],
        user_template=raw["user_template"],
        prompt_artifact_hash=raw["prompt_artifact_hash"],
        guideline_artifact_hash=raw["guideline_artifact_hash"],
    )


def _manifest_profile(value: Any) -> Mapping[str, Any]:
    fields = {
        "mode",
        "prompt_dir",
        "system_template",
        "user_template",
        "prompt_files",
        "prompt_artifact_hash",
        "guideline_artifact_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DriverError("MANIFEST_INVALID", "manifest prompt profile is invalid")
    if value.get("mode") not in {"history", "observation"}:
        raise DriverError("MANIFEST_INVALID", "manifest prompt mode is invalid")
    prompt_dir = _manifest_relative_path(value.get("prompt_dir"), "prompt directory")
    system = _manifest_template_name(value.get("system_template"), "system template")
    user = _manifest_template_name(value.get("user_template"), "user template")
    files = value.get("prompt_files")
    if not isinstance(files, Mapping) or not files:
        raise DriverError("MANIFEST_INVALID", "manifest prompt_files is invalid")
    normalized_files: dict[str, str] = {}
    for relative, digest in files.items():
        relative, digest = _manifest_path_hash(relative, digest, "prompt file")
        if not relative.startswith(prompt_dir + "/"):
            raise DriverError("MANIFEST_INVALID", "manifest prompt file escaped prompt directory")
        normalized_files[relative] = digest
    required = {f"{prompt_dir}/{system}.jinja", f"{prompt_dir}/{user}.jinja"}
    if not required <= set(normalized_files):
        raise DriverError("MANIFEST_INVALID", "manifest prompt profile lacks selected templates")
    return {
        "mode": value["mode"],
        "prompt_dir": prompt_dir,
        "system_template": system,
        "user_template": user,
        "prompt_files": normalized_files,
        "prompt_artifact_hash": _sha_text(
            value.get("prompt_artifact_hash"),
            "manifest prompt_artifact_hash",
            manifest=True,
        ),
        "guideline_artifact_hash": _sha_text(
            value.get("guideline_artifact_hash"),
            "manifest guideline_artifact_hash",
            manifest=True,
        ),
    }


def _parse_artifacts(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_FIELDS:
        raise DriverError("INVALID_REQUEST", "ACON artifacts do not match schema")
    result: dict[str, str] = {}
    for field in sorted(_ARTIFACT_FIELDS):
        item = _bounded_text(value.get(field), field, allow_empty=False, limit=1024)
        if field.endswith("_hash") and not _HEX_SHA256.fullmatch(item):
            raise DriverError("INVALID_REQUEST", f"{field} must be a SHA-256 digest")
        result[field] = item
    return result


def _validate_bound_revisions(
    environment: Mapping[str, str],
    artifacts: Mapping[str, str],
) -> None:
    bindings = {
        OPTIMIZER_REVISION_ENV: "optimizer_model_revision",
        TARGET_REVISION_ENV: "target_model_revision",
        TOKENIZER_REVISION_ENV: "tokenizer_revision",
    }
    for environment_key, artifact_key in bindings.items():
        configured = environment.get(environment_key, "").strip()
        if not configured:
            raise DriverError(
                "ARTIFACT_REVISION_MISSING",
                f"{environment_key} is required for runtime attestation",
            )
        if configured != artifacts[artifact_key]:
            raise DriverError(
                "ARTIFACT_REVISION_MISMATCH",
                f"{environment_key} disagrees with the request artifact identity",
            )


def _history_controls(value: Any) -> Mapping[str, Any]:
    fields = {
        "summarization_threshold",
        "preserve_last_k_turns",
        "summary_rule",
        "summary_interval",
        "history_version",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DriverError("INVALID_REQUEST", "ACON history controls do not match schema")
    threshold = _threshold(value.get("summarization_threshold"))
    preserve = _nonnegative_int(value.get("preserve_last_k_turns"), "preserve_last_k_turns")
    rule = value.get("summary_rule")
    if rule not in {"accumulate", "reset"}:
        raise DriverError("INVALID_REQUEST", "summary_rule must be accumulate or reset")
    interval = value.get("summary_interval")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or (interval != -1 and interval < 1)
    ):
        raise DriverError("INVALID_REQUEST", "summary_interval must be -1 or positive")
    if value.get("history_version") != 1 or isinstance(value.get("history_version"), bool):
        raise DriverError("INVALID_REQUEST", "pinned ACON supports history_version 1 only")
    return {
        "summarization_threshold": threshold,
        "preserve_last_k_turns": preserve,
        "summary_rule": rule,
        "summary_interval": interval,
        "history_version": 1,
    }


def _observation_controls(value: Any) -> Mapping[str, Any]:
    fields = {
        "summarization_threshold",
        "context_index",
        "add_previous_history_summary",
        "observation_version",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DriverError("INVALID_REQUEST", "ACON observation controls do not match schema")
    threshold = _threshold(value.get("summarization_threshold"))
    index = value.get("context_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise DriverError("INVALID_REQUEST", "context_index must be an integer")
    add_previous = value.get("add_previous_history_summary")
    if not isinstance(add_previous, bool):
        raise DriverError("INVALID_REQUEST", "add_previous_history_summary must be boolean")
    version = value.get("observation_version")
    if version != 1 or isinstance(version, bool):
        raise DriverError("INVALID_REQUEST", "pinned ACON supports observation_version 1 only")
    return {
        "summarization_threshold": threshold,
        "context_index": index,
        "add_previous_history_summary": add_previous,
        "observation_version": 1,
    }


def _messages(value: Any) -> list[Mapping[str, Any]]:
    return _message_sequence(value, "raw_messages")


def _message_sequence(value: Any, label: str) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("INVALID_REQUEST", f"{label} must be a sequence")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DriverError("INVALID_REQUEST", f"{label}[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise DriverError("INVALID_REQUEST", f"{label}[{index}] role/content is invalid")
        try:
            result.append(json.loads(_canonical_json(item)))
        except (TypeError, ValueError) as exc:
            raise DriverError("INVALID_REQUEST", f"{label}[{index}] is not canonical JSON") from exc
    return result


def _history_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages):
        if index == 1 and message["role"] == "user":
            continue
        if message["role"] == "user":
            parts.append(f"USER:\n{message['content']}\n\n")
        elif message["role"] == "assistant":
            parts.append(f"ASSISTANT:\n{message['content']}\n\n")
    return "".join(parts).strip()


def _preserved_turn_indices(
    messages: Sequence[Mapping[str, Any]],
    preserve_last_k_turns: int,
) -> tuple[int, ...]:
    if not messages or preserve_last_k_turns <= 0:
        return ()
    turns: list[int] = []
    index = len(messages) - 1
    count = 0
    while index >= 0 and count < preserve_last_k_turns:
        if messages[index]["role"] != "user":
            index -= 1
            continue
        user_index = index
        assistant_index = index - 1
        while assistant_index >= 0 and messages[assistant_index]["role"] != "assistant":
            assistant_index -= 1
        if assistant_index >= 0:
            turns[0:0] = (assistant_index, user_index)
            count += 1
            index = assistant_index - 1
        else:
            turns.insert(0, user_index)
            count += 1
            index -= 1
    first_user = messages[1]["content"] if len(messages) > 1 else ""
    return tuple(
        item
        for item in turns
        if messages[item]["role"] != "system"
        and not (messages[item]["role"] == "user" and messages[item]["content"] == first_user)
    )


def _history_material(
    messages: Sequence[Mapping[str, Any]],
    preserved: Sequence[Mapping[str, Any]],
) -> str:
    start = len(messages)
    if preserved:
        first = preserved[0]
        for index, message in enumerate(messages):
            if message["role"] == first["role"] and message["content"] == first["content"]:
                start = index
                break
    summarization = messages[:start] if start > 2 else ()
    return _history_text(summarization)


def _integer_sequence(value: Any, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DriverError("INVALID_REQUEST", f"{label} must be an integer sequence")
    result = list(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        raise DriverError("INVALID_REQUEST", f"{label} must contain non-negative integers")
    return result


def _threshold(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise DriverError("INVALID_REQUEST", "summarization_threshold must be -1 or non-negative")
    return value


def _seed(value: Any) -> int:
    return _nonnegative_int(value, "seed")


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("INVALID_REQUEST", f"{label} must be a non-negative integer")
    return value


def _upstream_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_TEXT_CHARS:
        raise DriverError("UPSTREAM_OUTPUT_INVALID", f"upstream {label} is invalid")
    return value


def _bounded_text(
    value: Any,
    label: str,
    *,
    allow_empty: bool = True,
    limit: int = MAX_TEXT_CHARS,
) -> str:
    if not isinstance(value, str) or len(value) > limit or (not allow_empty and not value.strip()):
        qualifier = "bounded text" if allow_empty else "bounded non-empty text"
        raise DriverError("INVALID_REQUEST", f"{label} must be {qualifier}")
    return value


def _sha_text(value: Any, label: str, *, manifest: bool = False) -> str:
    code = "MANIFEST_INVALID" if manifest else "INVALID_REQUEST"
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise DriverError(code, f"{label} must be a SHA-256 digest")
    return value


def _manifest_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise DriverError("MANIFEST_INVALID", f"manifest {label} is invalid")
    path = Path(value)
    if ".." in path.parts or "." in path.parts:
        raise DriverError("MANIFEST_INVALID", f"manifest {label} is invalid")
    return path.as_posix()


def _manifest_template_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise DriverError("MANIFEST_INVALID", f"manifest {label} is invalid")
    return value


def _manifest_path_hash(value: Any, digest: Any, label: str) -> tuple[str, str]:
    relative = _manifest_relative_path(value, label)
    return relative, _sha_text(digest, f"manifest {label} hash", manifest=True)


def _safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DriverError("UPSTREAM_PATH_INVALID", "upstream artifact escaped checkout") from exc
    if not path.is_file():
        raise DriverError("UPSTREAM_ARTIFACT_MISSING", "pinned upstream artifact is unavailable")
    return path


def _sha256_file(path: Path, label: str) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DriverError(
            "UPSTREAM_ARTIFACT_MISSING",
            f"pinned upstream {label} is unavailable",
        ) from exc


def _environment_float(
    environment: Mapping[str, str],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = environment.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{key} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise DriverError(
            "MODEL_CONFIGURATION_INVALID",
            f"{key} must be between {minimum} and {maximum}",
        )
    return value


def _environment_optional_positive_int(
    environment: Mapping[str, str],
    key: str,
) -> int | None:
    raw = environment.get(key, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{key} must be an integer") from exc
    if value < 1:
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{key} must be positive")
    return value


def _environment_json_mapping(
    environment: Mapping[str, str],
    key: str,
) -> Mapping[str, Any]:
    raw = environment.get(key, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{key} must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{key} must contain a JSON object")
    return value


def _chat_completions_endpoint(base_url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(base_url)
    except ValueError as exc:
        raise DriverError("MODEL_CONFIGURATION_INVALID", f"{BASE_URL_ENV} is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise DriverError(
            "MODEL_CONFIGURATION_INVALID",
            f"{BASE_URL_ENV} must be an HTTP(S) URL without credentials or query",
        )
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _dependency_error(error: ModuleNotFoundError) -> DriverError:
    name = error.name if isinstance(error.name, str) else "optional dependency"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "?", name)[:200]
    return DriverError(
        "UPSTREAM_DEPENDENCY_MISSING",
        f"required pinned ACON dependency is missing: {safe_name}",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _content_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _error_response(error: DriverError, request: Any = None) -> Mapping[str, Any]:
    response: dict[str, Any] = {
        "status": "error",
        "failure_code": error.code,
        "error_message": error.public_message,
    }
    if error.usage is None:
        response["consumed"] = False
    else:
        calls, input_tokens, output_tokens = error.usage
        response["consumed"] = bool(error.consumed or calls)
        response["usage"] = {
            "model_calls": calls,
            "retrieval_calls": 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        response["usage_is_lower_bound"] = True
    if isinstance(request, Mapping):
        for field in ("request_id", "upstream_commit", "protocol_version"):
            value = request.get(field)
            if isinstance(value, str) and len(value) <= 256:
                response[field] = value
    return response


def serve_jsonl(driver: ACONDriver, source: TextIO, sink: TextIO) -> int:
    for line in source:
        request: Any = None
        if len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
            response: Mapping[str, Any] = _error_response(
                DriverError("REQUEST_TOO_LARGE", "worker request exceeds its byte limit")
            )
        else:
            try:
                request = json.loads(line)
                response = driver.run(request)
            except json.JSONDecodeError:
                response = _error_response(
                    DriverError("INVALID_JSON", "worker request is not valid JSON")
                )
            except DriverError as exc:
                response = _error_response(exc, request)
            except Exception:
                response = _error_response(
                    DriverError("WORKER_FAILURE", "worker failed without a safe diagnostic"),
                    request,
                )
        sink.write(_canonical_json(response) + "\n")
        sink.flush()
    return 0


def main() -> int:
    return serve_jsonl(ACONDriver(), sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
