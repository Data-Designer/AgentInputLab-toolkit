#!/usr/bin/env python3
"""Strict JSONL worker for pinned SmartSearch-3B frozen inference.

The worker executes the byte-verified upstream ``ReCall.run`` loop directly.
It replaces only two side-effect seams: the model POST is routed to a pinned
SGLang/OpenAI-completions deployment, and the upstream arbitrary-code executor
is replaced by a fixed Wikipedia-search call.  The upstream sandbox is never
imported, started, or sent an ``env``/``call`` string.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

UPSTREAM_COMMIT = "5e9e7ec0d6af34f45fad145d4bff442529e7e6ff"
CHECKPOINT_REPOSITORY = "vvv111222/SmartSearch-3B"
CHECKPOINT_REVISION = "82bfe8a845e3e08f0d9d00d8e1e2cfb6d50b5da8"
PROTOCOL = "agentinputlab.smartsearch.frozen.v2"
RELEASE = "recipe.smartsearch.frozen_3b_inference"
PROFILE_ID = "smartsearch.controlled_frozen.v1"
MANIFEST_SCHEMA = "agentinputlab.smartsearch.compatibility.v1"
TOOL_SCHEMA_SHA256 = "a1349ee4630e42cff2a0631252bdf931d3621f2f1a5bb005430e1c0494508731"
LOOP_CONTRACT_SHA256 = "8c64cda35d8b08f31c7dd8fba2efbb4e7adf3d57dc3a59a8a8e1c41e64d5e5b2"
PARSER_CONTRACT_SHA256 = "b2104370b1b39a1a34b809c25fd369a5e97426a70bf55300de5f5dec8bf3af6a"
DEFAULT_MANIFEST = Path(__file__).with_name("compatibility_manifest.json")

CHECKOUT_ENV = "AGENTINPUTLAB_SMARTSEARCH_CHECKOUT"
RUNTIME_MANIFEST_ENV = "AGENTINPUTLAB_SMARTSEARCH_RUNTIME_MANIFEST"
SOURCE_ARCHIVE_ENV = "AGENTINPUTLAB_SMARTSEARCH_SOURCE_ARCHIVE"
SNAPSHOT_COMMIT_ENV = "AGENTINPUTLAB_SMARTSEARCH_SNAPSHOT_COMMIT"
MODEL_API_KEY_ENV = "AGENTINPUTLAB_SMARTSEARCH_MODEL_API_KEY"
RETRIEVER_API_KEY_ENV = "AGENTINPUTLAB_SMARTSEARCH_RETRIEVER_API_KEY"
SANDBOX_ENV = "AGENTINPUTLAB_SMARTSEARCH_SANDBOX_URL"
LOCAL_RETRIEVER_TRANSPORT = "agentinputlab_prepared_corpus_v1"
LOCAL_MODEL_TRANSPORT = "openai_completions_local_snapshot"

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_TEXT_CHARS = 8 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_PROTOCOL_TAG = re.compile(r"</?(?:search|result|answer|think)>", re.IGNORECASE)
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
_PAYLOAD_FIELDS = frozenset(
    {
        "profile_id",
        "task_id",
        "question",
        "artifacts",
        "controls",
        "remaining_budget",
        "seed",
        "inference_contract",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "model_snapshot_manifest_sha256",
        "model_snapshot_tree_sha256",
        "tokenizer_revision",
        "served_model_name",
        "model_deployment_fingerprint_sha256",
        "context_admission_fingerprint_sha256",
        "retriever_revision",
        "retriever_deployment_fingerprint_sha256",
        "corpus_sha256",
        "index_sha256",
        "retrieval_config_sha256",
        "compatibility_manifest_sha256",
        "model_transport",
        "checkpoint_repository",
        "checkpoint_revision",
        "profile_id",
        "tool_schema_sha256",
        "loop_contract_sha256",
        "parser_contract_sha256",
    }
)
_CONTRACT_FIELDS = frozenset(
    {
        "profile_id",
        "tool_schema_sha256",
        "loop_contract_sha256",
        "parser_contract_sha256",
        "max_rounds",
        "top_n",
        "max_new_tokens",
        "temperature",
        "stop",
        "outer_retry_attempts",
        "search_tags_per_turn",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "ground_truth",
        "label",
        "labels",
        "reward",
        "rewards",
        "process_reward",
        "reference_answer",
        "target_answer",
        "verifier_score",
        "endpoint",
        "url",
        "command",
        "callable",
        "import",
        "api_key",
        "credential",
        "sandbox",
    }
)


class DriverError(RuntimeError):
    def __init__(self, code: str, message: str, *, consumed: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.consumed = consumed


class _UnboundRequestsModule(types.ModuleType):
    """Fail-closed stand-in for the upstream module's requests import."""

    def __init__(self) -> None:
        super().__init__("requests")

    def post(self, *_args: Any, **_kwargs: Any) -> None:
        raise DriverError(
            "UPSTREAM_TRANSPORT_UNBOUND",
            "the upstream requests seam is not bound to the controlled model transport",
        )


@dataclass(slots=True)
class Usage:
    model_calls: int = 0
    retrieval_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_mapping(self) -> Mapping[str, int]:
        return {
            "model_calls": self.model_calls,
            "retrieval_calls": self.retrieval_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class BudgetGate:
    def __init__(self, caps: Mapping[str, Any]) -> None:
        self.caps = caps
        self.usage = Usage()

    def before_model(self, admitted_input_tokens: int | None) -> None:
        if self.usage.model_calls >= self.caps["model_calls"]:
            raise DriverError("MODEL_BUDGET_EXHAUSTED", "model budget exhausted")
        remaining_output = self.caps["output_tokens"] - self.usage.output_tokens
        if remaining_output < 8192:
            raise DriverError(
                "OUTPUT_BUDGET_EXHAUSTED",
                "8192 output tokens must remain before a SmartSearch model call",
            )
        context_cap = self.caps["context_tokens"]
        if context_cap is not None:
            if admitted_input_tokens is None:
                raise DriverError(
                    "CONTEXT_ADMISSION_REQUIRED",
                    "an exact tokenizer admission result is required before model execution",
                )
            if self.usage.input_tokens + admitted_input_tokens > context_cap:
                raise DriverError(
                    "CONTEXT_BUDGET_EXHAUSTED",
                    "prompt is rejected before model execution by the context budget",
                )
        self.usage.model_calls += 1

    def after_model(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        admitted_input_tokens: int | None,
    ) -> None:
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        if admitted_input_tokens is not None and input_tokens != admitted_input_tokens:
            raise DriverError(
                "CONTEXT_ADMISSION_MISMATCH",
                "provider usage disagrees with the preflight tokenizer admission",
                consumed=True,
            )
        context_cap = self.caps["context_tokens"]
        if context_cap is not None and self.usage.input_tokens > context_cap:
            raise DriverError(
                "CONTEXT_BUDGET_EXCEEDED",
                "model-reported context token usage exceeded the supplied budget",
                consumed=True,
            )
        if self.usage.output_tokens > self.caps["output_tokens"]:
            raise DriverError(
                "OUTPUT_BUDGET_EXCEEDED",
                "model-reported output token usage exceeded the supplied budget",
                consumed=True,
            )

    def before_retrieval(self) -> None:
        if self.usage.retrieval_calls >= self.caps["retrieval_calls"]:
            raise DriverError(
                "RETRIEVAL_BUDGET_EXHAUSTED",
                "retrieval budget exhausted",
                consumed=bool(self.usage.model_calls),
            )
        self.usage.retrieval_calls += 1


@dataclass(frozen=True, slots=True)
class ModelTurn:
    text: str
    finish_reason: str
    matched: str
    input_tokens: int
    output_tokens: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        raise DriverError("REDIRECT_REFUSED", "HTTP redirects are disabled", consumed=True)


class _HTTPJSON:
    def __init__(self, *, api_key: str, timeout: float) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.opener = urllib.request.build_opener(_NoRedirect())

    def post(self, endpoint: str, body: Mapping[str, Any]) -> Any:
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DriverError("ENDPOINT_INVALID", "endpoint must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise DriverError("ENDPOINT_INVALID", "endpoint credentials/query are forbidden")
        if self.api_key and parsed.scheme != "https":
            raise DriverError("ENDPOINT_INVALID", "API keys may only be sent over HTTPS")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            endpoint,
            data=_canonical_json(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except DriverError:
            raise
        except urllib.error.HTTPError as exc:
            raise DriverError(
                "HTTP_FAILED", f"endpoint returned HTTP {exc.code}", consumed=True
            ) from exc
        except Exception as exc:
            raise DriverError("HTTP_FAILED", "endpoint request failed", consumed=True) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise DriverError(
                "RESPONSE_TOO_LARGE", "endpoint response exceeds byte limit", consumed=True
            )
        try:
            return _strict_json_loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, DriverError) as exc:
            raise DriverError(
                "RESPONSE_INVALID", "endpoint returned invalid JSON", consumed=True
            ) from exc


class HTTPModelTransport:
    def __init__(self, config: Mapping[str, Any], *, api_key: str = "") -> None:
        self.mode = _text(config, "model_transport")
        self.base_url = _http_base(_text(config, "model_base_url"))
        self.tokenizer_base_url = _http_base(_text(config, "tokenizer_base_url"))
        self.model = _text(config, "served_model_name")
        self.identity = {
            key: config[key]
            for key in (
                "checkpoint_repository",
                "checkpoint_revision",
                "model_snapshot_tree_sha256",
                "tokenizer_revision",
                "served_model_name",
                "context_admission_fingerprint_sha256",
            )
        }
        self.http = _HTTPJSON(
            api_key=api_key, timeout=_positive_float(config, "model_timeout_seconds")
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "transport": self.mode,
            "base_url": self.base_url,
            "tokenizer_base_url": self.tokenizer_base_url,
            "identity": self.identity,
        }

    def count_prompt_tokens(self, prompt: str) -> int:
        result = self.http.post(
            self.tokenizer_base_url + "/tokenize",
            {
                "model": self.model,
                "tokenizer_revision": self.identity["tokenizer_revision"],
                "text": prompt,
            },
        )
        if not isinstance(result, Mapping) or result.get("identity") != self.identity:
            raise DriverError(
                "CONTEXT_ADMISSION_IDENTITY_MISMATCH",
                "tokenizer admission identity mismatch",
                consumed=True,
            )
        return _nonnegative(result.get("input_tokens"), "admitted input_tokens")

    def generate(self, prompt: str, sampling: Mapping[str, Any]) -> ModelTurn:
        if self.mode == "sglang_generate":
            result = self.http.post(
                self.base_url + "/generate",
                {"text": prompt, "sampling_params": dict(sampling)},
            )
            if not isinstance(result, Mapping) or result.get("identity") != self.identity:
                raise DriverError(
                    "MODEL_IDENTITY_MISMATCH",
                    "SGLang immutable model identity mismatch",
                    consumed=True,
                )
            meta = result.get("meta_info")
            if not isinstance(meta, Mapping):
                raise DriverError(
                    "MODEL_RESPONSE_INVALID", "SGLang meta_info is required", consumed=True
                )
            finish = meta.get("finish_reason")
            if not isinstance(finish, Mapping):
                raise DriverError(
                    "MODEL_RESPONSE_INVALID", "SGLang finish_reason is required", consumed=True
                )
            return ModelTurn(
                text=_bounded_text(result.get("text"), "model text"),
                finish_reason=str(finish.get("type", "")),
                matched=str(finish.get("matched", "")),
                input_tokens=_nonnegative(meta.get("prompt_tokens"), "prompt_tokens"),
                output_tokens=_nonnegative(meta.get("completion_tokens"), "completion_tokens"),
            )
        if self.mode != "openai_completions":
            raise DriverError("MODEL_CONFIGURATION_INVALID", "unsupported model transport")
        result = self.http.post(
            self.base_url + "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "temperature": sampling["temperature"],
                "max_tokens": sampling["max_new_tokens"],
                "stop": sampling["stop"],
            },
        )
        if not isinstance(result, Mapping) or result.get("identity") != self.identity:
            raise DriverError(
                "MODEL_IDENTITY_MISMATCH",
                "completion immutable model identity mismatch",
                consumed=True,
            )
        choices = result.get("choices")
        usage = result.get("usage")
        if not isinstance(choices, Sequence) or not choices or not isinstance(choices[0], Mapping):
            raise DriverError(
                "MODEL_RESPONSE_INVALID", "completion choice is required", consumed=True
            )
        if not isinstance(usage, Mapping):
            raise DriverError(
                "MODEL_RESPONSE_INVALID", "completion usage is required", consumed=True
            )
        text = _bounded_text(choices[0].get("text"), "completion text")
        text, matched = _completion_stop(text, choices[0])
        return ModelTurn(
            text=text,
            finish_reason=str(choices[0].get("finish_reason", "")),
            matched=matched,
            input_tokens=_nonnegative(usage.get("prompt_tokens"), "prompt_tokens"),
            output_tokens=_nonnegative(usage.get("completion_tokens"), "completion_tokens"),
        )


class LocalSnapshotOpenAIModel:
    """Use a local vLLM endpoint after matching it to the frozen model root."""

    def __init__(self, config: Mapping[str, Any], *, api_key: str = "") -> None:
        self.mode = _text(config, "model_transport")
        if self.mode != LOCAL_MODEL_TRANSPORT:
            raise DriverError("MODEL_CONFIGURATION_INVALID", "local model transport drifted")
        self.base_url = _http_base(_text(config, "model_base_url"))
        self.tokenizer_base_url = _http_base(_text(config, "tokenizer_base_url"))
        self.model = _text(config, "served_model_name")
        self.model_root = Path(_text(config, "local_model_root")).expanduser().resolve()
        self.identity = {
            key: config[key]
            for key in (
                "checkpoint_repository",
                "checkpoint_revision",
                "model_snapshot_tree_sha256",
                "tokenizer_revision",
                "served_model_name",
                "context_admission_fingerprint_sha256",
            )
        }
        self.http = _HTTPJSON(
            api_key=api_key,
            timeout=_positive_float(config, "model_timeout_seconds"),
        )
        self._verify_served_root()

    def _verify_served_root(self) -> None:
        request = urllib.request.Request(
            self.base_url + "/v1/models",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.http.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except Exception as exc:
            raise DriverError(
                "MODEL_IDENTITY_MISMATCH", "local vLLM model inventory is unavailable"
            ) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise DriverError("MODEL_IDENTITY_MISMATCH", "model inventory is too large")
        result = _json_object(raw, "model inventory")
        rows = result.get("data")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise DriverError("MODEL_IDENTITY_MISMATCH", "model inventory is invalid")
        matches = [
            item for item in rows if isinstance(item, Mapping) and item.get("id") == self.model
        ]
        if len(matches) != 1:
            raise DriverError("MODEL_IDENTITY_MISMATCH", "served model alias is not unique")
        served_root = matches[0].get("root")
        if not isinstance(served_root, str):
            raise DriverError("MODEL_IDENTITY_MISMATCH", "served model root is absent")
        if Path(served_root).expanduser().resolve() != self.model_root:
            raise DriverError("MODEL_IDENTITY_MISMATCH", "served model root differs")

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "transport": self.mode,
            "base_url": self.base_url,
            "tokenizer_base_url": self.tokenizer_base_url,
            "identity": self.identity,
        }

    def count_prompt_tokens(self, prompt: str) -> int:
        result = self.http.post(
            self.tokenizer_base_url + "/tokenize",
            {"model": self.model, "prompt": prompt},
        )
        if not isinstance(result, Mapping):
            raise DriverError(
                "CONTEXT_ADMISSION_FAILED", "local tokenizer response is invalid", consumed=True
            )
        return _nonnegative(result.get("count"), "admitted input_tokens")

    def generate(self, prompt: str, sampling: Mapping[str, Any]) -> ModelTurn:
        result = self.http.post(
            self.base_url + "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "temperature": sampling["temperature"],
                "max_tokens": sampling["max_new_tokens"],
                "stop": sampling["stop"],
            },
        )
        choices = result.get("choices") if isinstance(result, Mapping) else None
        usage = result.get("usage") if isinstance(result, Mapping) else None
        if not isinstance(choices, Sequence) or not choices or not isinstance(choices[0], Mapping):
            raise DriverError(
                "MODEL_RESPONSE_INVALID", "completion choice is required", consumed=True
            )
        if not isinstance(usage, Mapping):
            raise DriverError(
                "MODEL_RESPONSE_INVALID", "completion usage is required", consumed=True
            )
        text = _bounded_text(choices[0].get("text"), "completion text")
        text, matched = _completion_stop(text, choices[0])
        return ModelTurn(
            text=text,
            finish_reason=str(choices[0].get("finish_reason", "")),
            matched=matched,
            input_tokens=_nonnegative(usage.get("prompt_tokens"), "prompt_tokens"),
            output_tokens=_nonnegative(usage.get("completion_tokens"), "completion_tokens"),
        )


class HTTPRetriever:
    def __init__(self, config: Mapping[str, Any], *, api_key: str = "") -> None:
        self.base_url = _http_base(_text(config, "retriever_base_url"))
        self.http = _HTTPJSON(
            api_key=api_key,
            timeout=_positive_float(config, "retriever_timeout_seconds"),
        )
        self.identity = {
            key: config[key]
            for key in (
                "retriever_revision",
                "corpus_sha256",
                "index_sha256",
                "retrieval_config_sha256",
            )
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "transport": "controlled_flashrag_search_v1",
            "base_url": self.base_url,
            **self.identity,
        }

    def search(self, query: str, *, top_n: int) -> Sequence[Mapping[str, Any]]:
        result = self.http.post(
            self.base_url + "/search",
            {"query": query, "top_n": top_n, "return_score": False},
        )
        if not isinstance(result, Mapping) or result.get("identity") != self.identity:
            raise DriverError(
                "RETRIEVER_IDENTITY_MISMATCH", "retriever identity mismatch", consumed=True
            )
        documents = result.get("documents")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise DriverError(
                "RETRIEVER_RESPONSE_INVALID", "retriever documents are required", consumed=True
            )
        return documents


class LocalPreparedRetriever:
    """Run the same checksum-locked production retriever inside the worker."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        from agentinputlab.experiments.production_builders import (
            PreparedCorpusRetrieverBuilder,
            ProductionRetrieverBuildRequest,
            retriever_product_sha256,
        )

        parameters = config.get("local_retriever_builder_parameters")
        benchmark_id = config.get("local_retriever_benchmark_id")
        model_slot = config.get("local_retriever_model_slot")
        seed = config.get("local_retriever_seed")
        if not isinstance(parameters, Mapping):
            raise DriverError(
                "RUNTIME_MANIFEST_INVALID",
                "local retriever builder parameters are required",
            )
        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise DriverError("RUNTIME_MANIFEST_INVALID", "local retriever benchmark is required")
        if not isinstance(model_slot, str) or not model_slot.strip():
            raise DriverError("RUNTIME_MANIFEST_INVALID", "local retriever model slot is required")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise DriverError("RUNTIME_MANIFEST_INVALID", "local retriever seed is invalid")
        builder = PreparedCorpusRetrieverBuilder()
        blockers = tuple(builder.preflight(parameters))
        if blockers:
            raise DriverError(
                "LOCAL_RETRIEVER_PREFLIGHT_FAILED",
                "; ".join(str(item) for item in blockers),
            )
        try:
            retriever = builder.build(
                ProductionRetrieverBuildRequest(
                    benchmark_id=benchmark_id,
                    model_slot=model_slot,
                    seed=seed,
                    builder_parameters=parameters,
                )
            )
            observed = retriever_product_sha256(retriever)
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "LOCAL_RETRIEVER_BUILD_FAILED",
                "checksum-locked local retriever could not be built",
            ) from exc
        if observed != config.get("index_sha256"):
            raise DriverError(
                "RETRIEVER_IDENTITY_MISMATCH",
                "local retriever product hash differs from the frozen index",
            )
        self.retriever = retriever
        self.identity = {
            key: config[key]
            for key in (
                "retriever_revision",
                "corpus_sha256",
                "index_sha256",
                "retrieval_config_sha256",
            )
        }
        self.builder_parameters_sha256 = _content_hash(parameters)

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "transport": LOCAL_RETRIEVER_TRANSPORT,
            "builder_parameters_sha256": self.builder_parameters_sha256,
            **self.identity,
        }

    def search(self, query: str, *, top_n: int) -> Sequence[Mapping[str, Any]]:
        try:
            rows = self.retriever.retrieve(query, top_k=top_n)
        except Exception as exc:
            raise DriverError(
                "RETRIEVER_REQUEST_FAILED", "local retriever failed", consumed=True
            ) from exc
        documents = []
        for row in rows:
            title = row.document.title.strip()
            text = row.document.text.strip()
            documents.append(
                {
                    "id": row.document.doc_id,
                    "contents": f"Title: {title}\n{text}" if title else text,
                    "score": row.score,
                }
            )
        return documents


class _UpstreamResponse:
    def __init__(self, value: Mapping[str, Any]) -> None:
        self.value = value

    def json(self) -> Mapping[str, Any]:
        return self.value


class _ModelRequestsShim:
    def __init__(self, model: Any, gate: BudgetGate) -> None:
        self.model = model
        self.gate = gate

    def post(self, url: str, json: Mapping[str, Any]) -> _UpstreamResponse:
        if url != "agentinputlab://model/generate":
            raise DriverError("UNEXPECTED_NETWORK_ROUTE", "upstream requested an unapproved URL")
        if set(json) != {"text", "sampling_params"}:
            raise DriverError("UPSTREAM_API_MISMATCH", "unexpected upstream model request")
        sampling = json["sampling_params"]
        expected = {
            "temperature": 0.0,
            "max_new_tokens": 8192,
            "stop": ["</search>", "</answer>"],
        }
        if not isinstance(sampling, Mapping) or dict(sampling) != expected:
            raise DriverError("UPSTREAM_API_MISMATCH", "upstream generation contract drifted")
        prompt = _bounded_text(json["text"], "prompt")
        admitted_input_tokens: int | None = None
        if self.gate.caps["context_tokens"] is not None:
            counter = getattr(self.model, "count_prompt_tokens", None)
            if not callable(counter):
                raise DriverError(
                    "CONTEXT_ADMISSION_REQUIRED",
                    "model adapter has no exact tokenizer admission interface",
                )
            try:
                admitted_input_tokens = _nonnegative(counter(prompt), "admitted input_tokens")
            except DriverError:
                raise
            except Exception as exc:
                raise DriverError(
                    "CONTEXT_ADMISSION_FAILED",
                    "tokenizer admission failed before model execution",
                ) from exc
        self.gate.before_model(admitted_input_tokens)
        try:
            turn = self.model.generate(prompt, sampling)
        except DriverError as exc:
            exc.consumed = True
            raise
        except Exception as exc:
            raise DriverError(
                "MODEL_REQUEST_FAILED", "model request failed", consumed=True
            ) from exc
        if not isinstance(turn, ModelTurn):
            raise DriverError(
                "MODEL_RESPONSE_INVALID", "model adapter returned invalid data", consumed=True
            )
        self.gate.after_model(
            turn.input_tokens,
            turn.output_tokens,
            admitted_input_tokens=admitted_input_tokens,
        )
        return _UpstreamResponse(
            {
                "text": turn.text,
                "meta_info": {
                    "finish_reason": {"type": turn.finish_reason, "matched": turn.matched}
                },
            }
        )


class SearchRecorder:
    def __init__(self, retriever: Any, gate: BudgetGate, artifacts: Mapping[str, Any]) -> None:
        self.retriever = retriever
        self.gate = gate
        self.artifacts = artifacts
        self.trajectory: list[Mapping[str, Any]] = []
        self.evidence: list[Mapping[str, Any]] = []
        self._documents: dict[str, tuple[str, int]] = {}

    def execute(self, tool_calls: Sequence[str]) -> list[str]:
        if len(tool_calls) != 1:
            raise DriverError(
                "SEARCH_TAG_POLICY_VIOLATION",
                "exactly one non-nested search tag is allowed per model turn",
                consumed=bool(self.gate.usage.model_calls),
            )
        query = _bounded_text(tool_calls[0], "search query").strip()
        if not query:
            raise DriverError(
                "SEARCH_QUERY_INVALID", "search query must not be empty", consumed=True
            )
        if len(self.trajectory) >= 10:
            raise DriverError(
                "SEARCH_BUDGET_EXHAUSTED", "search loop exceeded ten calls", consumed=True
            )
        if _PROTOCOL_TAG.search(query):
            raise DriverError(
                "SEARCH_TAG_POLICY_VIOLATION",
                "nested protocol tags are forbidden in a search query",
                consumed=True,
            )
        self.gate.before_retrieval()
        try:
            documents = self.retriever.search(query, top_n=5)
        except DriverError as exc:
            exc.consumed = True
            raise
        except Exception as exc:
            raise DriverError(
                "RETRIEVER_REQUEST_FAILED", "retriever request failed", consumed=True
            ) from exc
        if len(documents) > 5:
            raise DriverError(
                "RETRIEVER_RESPONSE_INVALID",
                "retriever returned more than five documents",
                consumed=True,
            )
        clean: list[dict[str, Any]] = []
        ids: list[str] = []
        for item in documents:
            if not isinstance(item, Mapping):
                raise DriverError(
                    "RETRIEVER_RESPONSE_INVALID", "document must be an object", consumed=True
                )
            unknown = set(item) - {"id", "contents", "score"}
            if unknown:
                raise DriverError(
                    "RETRIEVER_RESPONSE_INVALID", "document contains unknown fields", consumed=True
                )
            document_id = _bounded_text(item.get("id"), "document id").strip()
            contents = _bounded_text(item.get("contents"), "document contents").strip()
            if not document_id or not contents:
                raise DriverError(
                    "RETRIEVER_RESPONSE_INVALID",
                    "document id/text must not be empty",
                    consumed=True,
                )
            if _PROTOCOL_TAG.search(contents):
                raise DriverError(
                    "DOCUMENT_PROTOCOL_INJECTION", "document contains a protocol tag", consumed=True
                )
            score = item.get("score")
            if score is not None and (
                isinstance(score, bool) or not isinstance(score, (int, float))
            ):
                raise DriverError(
                    "RETRIEVER_RESPONSE_INVALID", "document score must be numeric", consumed=True
                )
            clean_item = {"id": document_id, "contents": contents, "score": score}
            clean.append(clean_item)
            ids.append(document_id)
            previous = self._documents.get(document_id)
            if previous is not None and previous[0] != contents:
                raise DriverError(
                    "DOCUMENT_ID_COLLISION", "document ID changed content", consumed=True
                )
            if previous is None:
                step = len(self.trajectory)
                self._documents[document_id] = (contents, step)
                self.evidence.append(
                    {
                        "document_id": document_id,
                        "text": contents,
                        "source": "smartsearch.wikipedia_search",
                        "score": score,
                        "first_seen_step": step,
                        "corpus_sha256": self.artifacts["corpus_sha256"],
                        "index_sha256": self.artifacts["index_sha256"],
                        "document_sha256": _content_hash(
                            {"document_id": document_id, "text": contents}
                        ),
                        "metadata": {},
                    }
                )
        if len(ids) != len(set(ids)):
            raise DriverError(
                "RETRIEVER_RESPONSE_INVALID",
                "retriever returned duplicate document IDs",
                consumed=True,
            )
        request_record = {"query": query, "top_n": 5, "return_score": False}
        self.trajectory.append(
            {
                "index": len(self.trajectory),
                "query": query,
                "raw_search_text": query,
                "raw_search_hash": _sha256(query.encode("utf-8")),
                "tag_parse_status": "ok",
                "document_ids": ids,
                "retrieval_request_sha256": _content_hash(request_record),
                "retrieval_response_sha256": _content_hash(clean),
            }
        )
        text = "\n\n".join(item["contents"] for item in clean).strip()
        return [f"result: \n{text}".strip()]


class SmartSearchDriver:
    def __init__(
        self,
        *,
        checkout: str | Path | None = None,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        runtime_config: Mapping[str, Any] | None = None,
        model_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        retriever_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.environ = dict(os.environ if environ is None else environ)
        if self.environ.get(SANDBOX_ENV):
            raise DriverError(
                "SANDBOX_FORBIDDEN", "the upstream code-execution sandbox is forbidden"
            )
        self.checkout = Path(checkout or self.environ.get(CHECKOUT_ENV, "")).expanduser()
        if not str(self.checkout):
            raise DriverError("CHECKOUT_REQUIRED", f"{CHECKOUT_ENV} is required")
        self.manifest_path = Path(manifest_path)
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = _sha256(self.manifest_bytes)
        self.manifest = _json_object(self.manifest_bytes, "compatibility manifest")
        self.runtime = dict(runtime_config or self._runtime_from_environment())
        self.source_attestation = self._verify_source()
        self._verify_runtime_manifest()
        self.recall_class, self.answer_parser = self._load_upstream_surface()
        if model_factory is not None:
            self.model_factory = model_factory
        elif self.runtime.get("model_transport") == LOCAL_MODEL_TRANSPORT:
            self.model_factory = self._local_model
        else:
            self.model_factory = self._http_model
        if retriever_factory is not None:
            self.retriever_factory = retriever_factory
        elif self.runtime.get("retriever_transport") == LOCAL_RETRIEVER_TRANSPORT:
            self.retriever_factory = self._local_retriever
        else:
            self.retriever_factory = self._http_retriever
        self.last_usage: Usage | None = None

    def _runtime_from_environment(self) -> Mapping[str, Any]:
        path = self.environ.get(RUNTIME_MANIFEST_ENV, "")
        if not path:
            raise DriverError("RUNTIME_MANIFEST_REQUIRED", f"{RUNTIME_MANIFEST_ENV} is required")
        return _json_object(Path(path).read_bytes(), "runtime manifest")

    def _verify_source(self) -> str:
        if self.manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise DriverError("MANIFEST_INVALID", "compatibility manifest schema mismatch")
        upstream = self.manifest.get("upstream")
        source_files = self.manifest.get("source_files")
        if not isinstance(upstream, Mapping) or not isinstance(source_files, Mapping):
            raise DriverError("MANIFEST_INVALID", "compatibility manifest is incomplete")
        if upstream.get("commit") != UPSTREAM_COMMIT:
            raise DriverError("MANIFEST_INVALID", "compatibility manifest commit mismatch")
        checks = {"LICENSE": upstream.get("license_sha256"), **dict(source_files)}
        for relative, expected in checks.items():
            path = self.checkout / relative
            if not path.is_file() or _sha256(path.read_bytes()) != expected:
                raise DriverError("SOURCE_HASH_MISMATCH", f"pinned source mismatch: {relative}")
        if (self.checkout / ".git").exists():
            git_environment = {
                "PATH": os.defpath,
                "LANG": "C",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            }
            try:
                head = subprocess.run(
                    [
                        "git",
                        "-c",
                        "core.fsmonitor=false",
                        "-c",
                        f"core.hooksPath={os.devnull}",
                        "rev-parse",
                        "HEAD",
                    ],
                    cwd=self.checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=git_environment,
                ).stdout.strip()
                top_level = subprocess.run(
                    [
                        "git",
                        "-c",
                        "core.fsmonitor=false",
                        "-c",
                        f"core.hooksPath={os.devnull}",
                        "rev-parse",
                        "--show-toplevel",
                    ],
                    cwd=self.checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=git_environment,
                ).stdout.strip()
                dirty = subprocess.run(
                    [
                        "git",
                        "-c",
                        "core.fsmonitor=false",
                        "-c",
                        f"core.hooksPath={os.devnull}",
                        "status",
                        "--porcelain",
                        "--untracked-files=all",
                    ],
                    cwd=self.checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=git_environment,
                ).stdout.strip()
            except Exception as exc:
                raise DriverError("GIT_ATTESTATION_FAILED", "could not attest checkout") from exc
            if (
                head != UPSTREAM_COMMIT
                or Path(top_level).resolve() != self.checkout.resolve()
                or dirty
            ):
                raise DriverError("GIT_ATTESTATION_FAILED", "checkout is not pinned and clean")
            return "git_checkout"
        archive_path = self.runtime.get("source_archive_path") or self.environ.get(
            SOURCE_ARCHIVE_ENV
        )
        snapshot_commit = self.runtime.get("snapshot_commit") or self.environ.get(
            SNAPSHOT_COMMIT_ENV
        )
        if snapshot_commit != UPSTREAM_COMMIT or not isinstance(archive_path, str):
            raise DriverError(
                "SOURCE_ATTESTATION_REQUIRED", "snapshot archive attestation is required"
            )
        archive = Path(archive_path)
        if not archive.is_file() or _sha256(archive.read_bytes()) != upstream.get("archive_sha256"):
            raise DriverError("SOURCE_ARCHIVE_MISMATCH", "source archive hash mismatch")
        return "verified_archive_snapshot"

    def _verify_runtime_manifest(self) -> None:
        artifacts = self.runtime.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise DriverError("RUNTIME_MANIFEST_INVALID", "runtime artifacts are required")
        if set(artifacts) != _ARTIFACT_FIELDS:
            raise DriverError("RUNTIME_MANIFEST_INVALID", "runtime artifact fields are not exact")
        _validate_artifact_digests(artifacts)
        if artifacts.get("compatibility_manifest_sha256") != self.manifest_sha256:
            raise DriverError("RUNTIME_MANIFEST_INVALID", "compatibility manifest hash mismatch")
        if (
            artifacts.get("checkpoint_repository") != CHECKPOINT_REPOSITORY
            or artifacts.get("checkpoint_revision") != CHECKPOINT_REVISION
        ):
            raise DriverError("RUNTIME_MANIFEST_INVALID", "released checkpoint identity mismatch")
        snapshot_path = self.runtime.get("model_snapshot_manifest_path")
        if not isinstance(snapshot_path, str) or not Path(snapshot_path).is_file():
            raise DriverError(
                "MODEL_SNAPSHOT_MANIFEST_REQUIRED", "model snapshot manifest is required"
            )
        raw = Path(snapshot_path).read_bytes()
        if _sha256(raw) != artifacts.get("model_snapshot_manifest_sha256"):
            raise DriverError("MODEL_SNAPSHOT_MISMATCH", "model snapshot manifest hash mismatch")
        snapshot = _json_object(raw, "model snapshot manifest")
        if (
            snapshot.get("repository") != CHECKPOINT_REPOSITORY
            or snapshot.get("revision") != CHECKPOINT_REVISION
        ):
            raise DriverError("MODEL_SNAPSHOT_MISMATCH", "model snapshot revision mismatch")
        if snapshot.get("tokenizer_revision") != artifacts.get("tokenizer_revision"):
            raise DriverError("MODEL_SNAPSHOT_MISMATCH", "tokenizer revision mismatch")
        files = snapshot.get("files")
        if not isinstance(files, Mapping) or not files:
            raise DriverError("MODEL_SNAPSHOT_MISMATCH", "model snapshot file manifest is empty")
        _validate_snapshot_files(files)
        snapshot_tree_sha256 = _content_hash(files)
        if (
            snapshot.get("snapshot_tree_sha256") != snapshot_tree_sha256
            or artifacts.get("model_snapshot_tree_sha256") != snapshot_tree_sha256
        ):
            raise DriverError("MODEL_SNAPSHOT_MISMATCH", "model snapshot tree mismatch")

        contracts = self.manifest.get("contracts")
        if not isinstance(contracts, Mapping) or set(contracts) != _CONTRACT_FIELDS:
            raise DriverError("MANIFEST_INVALID", "compatibility contracts are not exact")
        for key in (
            "tool_schema_sha256",
            "loop_contract_sha256",
            "parser_contract_sha256",
        ):
            if contracts.get(key) != artifacts.get(key):
                raise DriverError("CONTRACT_MISMATCH", f"{key} disagrees with artifacts")
        fixed_contracts = {
            "tool_schema_sha256": TOOL_SCHEMA_SHA256,
            "loop_contract_sha256": LOOP_CONTRACT_SHA256,
            "parser_contract_sha256": PARSER_CONTRACT_SHA256,
        }
        if any(artifacts.get(key) != expected for key, expected in fixed_contracts.items()):
            raise DriverError("CONTRACT_MISMATCH", "frozen contract digest drifted")
        if contracts.get("profile_id") != PROFILE_ID:
            raise DriverError("CONTRACT_MISMATCH", "contract profile mismatch")
        expected_loop = {
            "max_rounds": contracts.get("max_rounds"),
            "max_results_per_search": contracts.get("top_n"),
            "max_new_tokens": contracts.get("max_new_tokens"),
            "temperature": contracts.get("temperature"),
            "stop": contracts.get("stop"),
            "retry_attempts": contracts.get("outer_retry_attempts"),
            "search_tags_per_turn": contracts.get("search_tags_per_turn"),
        }
        if _content_hash(expected_loop) != artifacts.get("loop_contract_sha256"):
            raise DriverError("CONTRACT_MISMATCH", "loop contract hash mismatch")
        tool_schema = self.runtime.get("tool_schema")
        if (
            isinstance(tool_schema, (str, bytes))
            or not isinstance(tool_schema, Sequence)
            or _content_hash(tool_schema) != artifacts.get("tool_schema_sha256")
        ):
            raise DriverError("CONTRACT_MISMATCH", "runtime tool schema hash mismatch")
        for key in (
            "model_transport",
            "served_model_name",
            "model_snapshot_tree_sha256",
            "tokenizer_revision",
            "context_admission_fingerprint_sha256",
            "retriever_revision",
            "corpus_sha256",
            "index_sha256",
            "retrieval_config_sha256",
        ):
            if self.runtime.get(key) != artifacts.get(key):
                raise DriverError(
                    "RUNTIME_MANIFEST_INVALID", f"runtime/artifact identity mismatch: {key}"
                )
        if self.runtime.get("retriever_transport") == LOCAL_RETRIEVER_TRANSPORT:
            required = {
                "local_retriever_builder_parameters",
                "local_retriever_benchmark_id",
                "local_retriever_model_slot",
                "local_retriever_seed",
            }
            if any(key not in self.runtime for key in required):
                raise DriverError(
                    "RUNTIME_MANIFEST_INVALID",
                    "local retriever runtime binding is incomplete",
                )
        if self.runtime.get("model_transport") == LOCAL_MODEL_TRANSPORT:
            model_root_value = self.runtime.get("local_model_root")
            if not isinstance(model_root_value, str) or not model_root_value.strip():
                raise DriverError(
                    "RUNTIME_MANIFEST_INVALID", "local SmartSearch model root is required"
                )
            model_root = Path(model_root_value).expanduser().resolve()
            if not model_root.is_dir() or model_root.is_symlink():
                raise DriverError(
                    "MODEL_SNAPSHOT_MISMATCH", "local SmartSearch model root is invalid"
                )
            for relative, identity in files.items():
                file_path = (model_root / relative).resolve()
                try:
                    file_path.relative_to(model_root)
                except ValueError as exc:
                    raise DriverError(
                        "MODEL_SNAPSHOT_MISMATCH", "snapshot file escapes model root"
                    ) from exc
                if (
                    not file_path.is_file()
                    or file_path.is_symlink()
                    or file_path.stat().st_size != identity.get("bytes")
                    or _sha256_file(file_path) != identity.get("sha256")
                ):
                    raise DriverError(
                        "MODEL_SNAPSHOT_MISMATCH",
                        f"local model file checksum mismatch: {relative}",
                    )

    def _load_upstream_surface(self) -> tuple[type[Any], Callable[[str], str]]:
        recall_path = self.checkout / "src/re_call/inference/re_call.py"
        module = types.ModuleType(f"_agentinputlab_smartsearch_{self.manifest_sha256[:12]}")
        module.__file__ = str(recall_path)
        requests_placeholder = _UnboundRequestsModule()
        default_import = builtins.__import__

        def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name == "requests":
                if fromlist:
                    raise DriverError(
                        "UPSTREAM_API_MISMATCH",
                        "pinned SmartSearch changed its requests import contract",
                    )
                return requests_placeholder
            return default_import(name, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = {
            **vars(builtins),
            "__import__": controlled_import,
        }
        old_dont_write = sys.dont_write_bytecode
        old_prefix = sys.pycache_prefix
        try:
            sys.dont_write_bytecode = True
            sys.pycache_prefix = tempfile.mkdtemp(prefix="agentinputlab-smartsearch-pycache-")
            exec(compile(recall_path.read_bytes(), str(recall_path), "exec"), module.__dict__)
        finally:
            sys.dont_write_bytecode = old_dont_write
            sys.pycache_prefix = old_prefix
        if module.__dict__.get("requests") is not requests_placeholder:
            raise DriverError(
                "UPSTREAM_API_MISMATCH",
                "pinned SmartSearch did not bind the controlled requests import seam",
            )
        recall = module.__dict__.get("ReCall")
        if not isinstance(recall, type) or not hasattr(recall.run, "__wrapped__"):
            raise DriverError("UPSTREAM_API_MISMATCH", "pinned ReCall API is unavailable")
        active_path = self.checkout / "src/flashrag/pipeline/active_pipeline.py"
        helper_path = self.checkout / "src/verl/utils/reward_score/re_call.py"
        extractor = _load_ast_function(
            active_path, "extract_last_assistant_response", class_name="ReCallPipeline"
        )
        last_boxed = _load_ast_function(helper_path, "last_boxed_only_string")
        remove_boxed = _load_ast_function(helper_path, "remove_boxed")

        def parse_answer(transcript: str) -> str:
            part = extractor(object(), transcript)
            if not isinstance(part, str) or not part:
                return ""
            try:
                boxed = last_boxed(part)
                if boxed is None:
                    return ""
                answer = remove_boxed(boxed)
            except (AssertionError, TypeError):
                return ""
            return answer if isinstance(answer, str) else ""

        self._upstream_module = module
        return recall, parse_answer

    def _http_model(self, config: Mapping[str, Any]) -> HTTPModelTransport:
        merged = {**dict(config), **dict(self.runtime["artifacts"])}
        return HTTPModelTransport(merged, api_key=self.environ.get(MODEL_API_KEY_ENV, ""))

    def _http_retriever(self, config: Mapping[str, Any]) -> HTTPRetriever:
        merged = {**dict(config), **dict(self.runtime["artifacts"])}
        return HTTPRetriever(merged, api_key=self.environ.get(RETRIEVER_API_KEY_ENV, ""))

    def _local_model(self, config: Mapping[str, Any]) -> LocalSnapshotOpenAIModel:
        merged = {**dict(config), **dict(self.runtime["artifacts"])}
        return LocalSnapshotOpenAIModel(
            merged,
            api_key=self.environ.get(MODEL_API_KEY_ENV, ""),
        )

    def _local_retriever(self, config: Mapping[str, Any]) -> LocalPreparedRetriever:
        merged = {**dict(config), **dict(self.runtime["artifacts"])}
        return LocalPreparedRetriever(merged)

    def run(self, request: Any) -> Mapping[str, Any]:
        self.last_usage = None
        parsed = _validate_request(request)
        artifacts = parsed["payload"]["artifacts"]
        if _content_hash(artifacts) != parsed["dependency_fingerprints"].get("artifacts_sha256"):
            raise DriverError("ARTIFACT_IDENTITY_MISMATCH", "artifact dependency hash mismatch")
        if _content_hash(artifacts) != _content_hash(self.runtime["artifacts"]):
            raise DriverError("ARTIFACT_IDENTITY_MISMATCH", "request/runtime artifact mismatch")
        model = self.model_factory(self.runtime)
        retriever = self.retriever_factory(self.runtime)
        if _content_hash(model.fingerprint()) != artifacts.get(
            "model_deployment_fingerprint_sha256"
        ):
            raise DriverError("MODEL_IDENTITY_MISMATCH", "model deployment fingerprint mismatch")
        if _content_hash(retriever.fingerprint()) != artifacts.get(
            "retriever_deployment_fingerprint_sha256"
        ):
            raise DriverError(
                "RETRIEVER_IDENTITY_MISMATCH", "retriever deployment fingerprint mismatch"
            )

        gate = BudgetGate(parsed["payload"]["remaining_budget"])
        self.last_usage = gate.usage
        recorder = SearchRecorder(retriever, gate, artifacts)
        upstream_module = self._upstream_module
        upstream_module.requests = _ModelRequestsShim(model, gate)

        base = self.recall_class

        class SafeReCall(base):
            def execute_tool_calls(self, env, tool_calls):
                if env != "agentinputlab:no-exec":
                    raise DriverError("SANDBOX_FORBIDDEN", "dynamic tool environment is forbidden")
                return recorder.execute(tool_calls)

        agent = SafeReCall("agentinputlab://model", "agentinputlab://disabled-executor")
        try:
            transcript = base.run.__wrapped__(
                agent,
                "agentinputlab:no-exec",
                _canonical_json(self.runtime.get("tool_schema", [])),
                parsed["payload"]["question"],
            )
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "UPSTREAM_EXECUTION_FAILED",
                "pinned SmartSearch inference failed",
                consumed=bool(gate.usage.model_calls or gate.usage.retrieval_calls),
            ) from exc
        answer = self.answer_parser(transcript)
        if not answer.strip():
            raise DriverError(
                "NO_FINAL_ANSWER", "official parser found no boxed final answer", consumed=True
            )
        termination = "direct_answer" if not recorder.trajectory else "answer"
        if gate.usage.model_calls == 10 and len(recorder.trajectory) == 10:
            termination = "max_search_rounds"
        payload = {
            "profile_id": PROFILE_ID,
            "artifact_identity_sha256": _content_hash(artifacts),
            "artifacts": artifacts,
            "gold_answer_used": False,
            "test_time_reward_used": False,
            "external_query_refiner_used": False,
            "training_code_loaded": False,
            "trajectory": recorder.trajectory,
            "evidence": recorder.evidence,
            "final_answer": answer,
            "termination": termination,
            "raw_transcript_sha256": _sha256(transcript.encode("utf-8")),
            "runtime_attestation": {
                "source_commit": UPSTREAM_COMMIT,
                "source_attestation": self.source_attestation,
                "compatibility_manifest_sha256": self.manifest_sha256,
                "model_transport": artifacts["model_transport"],
                "model_snapshot_tree_sha256": artifacts["model_snapshot_tree_sha256"],
                "context_admission_fingerprint_sha256": artifacts[
                    "context_admission_fingerprint_sha256"
                ],
                "contract_hashes_verified": True,
                "immutable_model_identity_handshake_required": True,
                "context_admission_before_generation": True,
                "official_loop_called": True,
                "outer_retry_disabled": True,
                "dangerous_sandbox_used": False,
                "training_code_loaded": False,
                "seed_applied": False,
            },
        }
        return {
            "status": "ok",
            "request_id": parsed["request_id"],
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL,
            "payload": payload,
            "usage": gate.usage.as_mapping(),
        }


def _validate_request(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TOP_FIELDS:
        raise DriverError("REQUEST_SCHEMA_INVALID", "request envelope fields are invalid")
    _reject_forbidden(value)
    if value.get("method_id") != "recipe.smartsearch" or value.get("release_id") != RELEASE:
        raise DriverError("REQUEST_IDENTITY_MISMATCH", "method/release mismatch")
    if value.get("upstream_commit") != UPSTREAM_COMMIT or value.get("protocol_version") != PROTOCOL:
        raise DriverError("REQUEST_IDENTITY_MISMATCH", "commit/protocol mismatch")
    identity = {key: value[key] for key in _TOP_FIELDS if key != "request_id"}
    if value.get("request_id") != _content_hash(identity):
        raise DriverError("REQUEST_IDENTITY_MISMATCH", "request_id mismatch")
    if value.get("minimum_reservation") != {"model_calls": 1}:
        raise DriverError("REQUEST_SCHEMA_INVALID", "minimum reservation must be one model call")
    payload = value.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS:
        raise DriverError("REQUEST_SCHEMA_INVALID", "payload fields are invalid")
    if payload.get("profile_id") != PROFILE_ID:
        raise DriverError("REQUEST_IDENTITY_MISMATCH", "profile mismatch")
    _bounded_text(payload.get("question"), "question")
    if not isinstance(payload.get("task_id"), str):
        raise DriverError("REQUEST_SCHEMA_INVALID", "task_id must be text")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DriverError("REQUEST_SCHEMA_INVALID", "seed must be non-negative")
    if payload.get("controls") != {
        "max_search_rounds": 10,
        "max_results_per_round": 5,
        "max_evidence_documents": 50,
        "max_output_tokens_per_call": 8192,
    }:
        raise DriverError("REQUEST_SCHEMA_INVALID", "SmartSearch controls mismatch")
    expected_contract = {
        "mode": "frozen_inference",
        "allow_training": False,
        "allow_gold_answer": False,
        "allow_test_time_reward": False,
        "allow_external_query_refiner": False,
    }
    if payload.get("inference_contract") != expected_contract:
        raise DriverError("REQUEST_SCHEMA_INVALID", "frozen inference contract mismatch")
    budget = payload.get("remaining_budget")
    if not isinstance(budget, Mapping) or set(budget) != {
        "model_calls",
        "retrieval_calls",
        "output_tokens",
        "context_tokens",
    }:
        raise DriverError("REQUEST_SCHEMA_INVALID", "remaining budget fields are invalid")
    for key in ("model_calls", "retrieval_calls", "output_tokens"):
        _nonnegative(budget.get(key), key)
    if budget.get("context_tokens") is not None:
        _nonnegative(budget.get("context_tokens"), "context_tokens")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _ARTIFACT_FIELDS:
        raise DriverError("REQUEST_SCHEMA_INVALID", "artifact fields are not exact")
    _validate_artifact_digests(artifacts)
    if not isinstance(value.get("dependency_fingerprints"), Mapping):
        raise DriverError("REQUEST_SCHEMA_INVALID", "dependency fingerprints must be an object")
    return value


def _load_ast_function(
    path: Path, function_name: str, *, class_name: str = ""
) -> Callable[..., Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates: Sequence[ast.stmt] = tree.body
    if class_name:
        owner = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if owner is None:
            raise DriverError("UPSTREAM_API_MISMATCH", f"missing upstream class {class_name}")
        candidates = owner.body
    node = next(
        (
            node
            for node in candidates
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if node is None:
        raise DriverError("UPSTREAM_API_MISMATCH", f"missing upstream parser {function_name}")
    isolated = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {"re": re}
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace[function_name]


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise DriverError("FORBIDDEN_FIELD", f"forbidden request field: {key}")
            _reject_forbidden(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_forbidden(item)


def _infer_matched_stop(text: str) -> str:
    for marker in ("</search>", "</answer>"):
        if text.endswith(marker):
            return marker
    return ""


def _completion_stop(text: str, choice: Mapping[str, Any]) -> tuple[str, str]:
    """Reconstruct the SGLang stop contract without duplicating provider text.

    vLLM normally strips a matched stop string and reports it in stop_reason.
    Upstream ReCall appends that matched string itself. EOS IDs/unknown strings
    do not authorize inventing an answer/search closing tag.
    """
    if choice.get("finish_reason") != "stop":
        return text, ""
    suffix = _infer_matched_stop(text)
    reason = choice.get("stop_reason")
    matched = reason if reason in ("</search>", "</answer>") else suffix
    if matched and suffix and matched != suffix:
        raise DriverError(
            "MODEL_RESPONSE_INVALID", "conflicting completion stop identities", consumed=True
        )
    if matched and text.endswith(matched):
        text = text[: -len(matched)]
    return text, matched


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise DriverError("RUNTIME_MANIFEST_INVALID", f"runtime {key} is required")
    return result.strip()


def _bounded_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_TEXT_CHARS:
        raise DriverError("TEXT_INVALID", f"{label} must be bounded text")
    return value


def _positive_float(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, (int, float)) or result <= 0:
        raise DriverError("RUNTIME_MANIFEST_INVALID", f"runtime {key} must be positive")
    return float(result)


def _nonnegative(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriverError("VALUE_INVALID", f"{label} must be a non-negative integer")
    return value


def _validate_artifact_digests(artifacts: Mapping[str, Any]) -> None:
    for key, value in artifacts.items():
        if key.endswith("_sha256") and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in _HEX for character in value)
        ):
            raise DriverError("RUNTIME_MANIFEST_INVALID", f"invalid artifact digest: {key}")


def _validate_snapshot_files(files: Mapping[str, Any]) -> None:
    for relative, identity in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(identity, Mapping)
            or set(identity) - {"sha256", "bytes"}
        ):
            raise DriverError("MODEL_SNAPSHOT_MISMATCH", "snapshot file entry is invalid")
        digest = identity.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise DriverError("MODEL_SNAPSHOT_MISMATCH", "snapshot file digest is invalid")
        if "bytes" in identity:
            _nonnegative(identity["bytes"], "snapshot file bytes")


def _http_base(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DriverError("ENDPOINT_INVALID", "runtime endpoint must be HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DriverError("ENDPOINT_INVALID", "runtime endpoint contains forbidden components")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverError("JSON_INVALID", f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise DriverError("JSON_INVALID", f"{label} must be an object")
    return value


def _strict_json_loads(raw: str | bytes) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_no_duplicate_object,
        parse_constant=_reject_json_constant,
    )


def _reject_json_constant(value: str) -> Any:
    raise DriverError("JSON_INVALID", f"non-finite JSON number is forbidden: {value}")


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DriverError("JSON_INVALID", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _content_hash(value: Any) -> str:
    return _sha256(_canonical_json(value).encode("utf-8"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error_response(
    error: DriverError, request: Any, usage: Usage | None = None
) -> Mapping[str, Any]:
    response: dict[str, Any] = {
        "status": "error",
        "failure_code": error.code,
        "error_message": error.public_message,
    }
    if usage is None or (
        not error.consumed and not usage.model_calls and not usage.retrieval_calls
    ):
        response["consumed"] = False
    else:
        response["consumed"] = True
        response["usage"] = usage.as_mapping()
    if isinstance(request, Mapping):
        for key in ("request_id", "upstream_commit", "protocol_version"):
            if isinstance(request.get(key), str):
                response[key] = request[key]
    return response


def serve_jsonl(driver: SmartSearchDriver, source: TextIO, sink: TextIO) -> int:
    for line in source:
        request: Any = None
        if len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
            response = _error_response(DriverError("REQUEST_TOO_LARGE", "request too large"), None)
        else:
            try:
                request = _strict_json_loads(line)
                response = driver.run(request)
            except (json.JSONDecodeError, DriverError) as exc:
                error = (
                    exc
                    if isinstance(exc, DriverError)
                    else DriverError("INVALID_JSON", "invalid JSON")
                )
                response = _error_response(error, request, driver.last_usage)
            except Exception:
                response = _error_response(
                    DriverError("WORKER_FAILURE", "worker failed safely"),
                    request,
                    driver.last_usage,
                )
        sink.write(_canonical_json(response) + "\n")
        sink.flush()
    return 0


def main() -> int:
    return serve_jsonl(SmartSearchDriver(), sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
