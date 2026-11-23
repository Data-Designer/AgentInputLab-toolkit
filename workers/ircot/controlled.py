"""Startup-bound model transfer through the unchanged pinned IRCoT controller.

This is not the historical Codex/Elasticsearch deployment. The native builder,
participants, prompt selection/fitting, sentence parser and exit logic execute;
only the generator, retriever and GPT-2 tokenizer resolver are caller-bound.
Run this adapter in a dedicated worker process with its own HOME/cache root.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import os
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentinputlab.backends import GenerationRequest
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .driver import (
    DEFAULT_MANIFEST,
    PROTOCOL_VERSION,
    UPSTREAM_COMMIT,
    DriverError,
    IRCoTDriver,
    _BoundedCapture,
    _canonical_mapping,
    _FrozenResponse,
    _load_config,
    _parse_request,
    _patch_config,
    _project_trajectory,
    _RetrievalCapture,
    _TracingController,
    _upstream_import_scope,
    _working_directory,
)

PROFILE = "controlled.hotpotqa.reader_transfer.v1"
SOURCE_PROFILE = "upstream.hotpotqa.codex.v1"
_NATIVE_LOCK = threading.Lock()


class PinnedIRCoTControlledRuntime:
    """WorkerClient for one fixed, externally served reader and corpus.

    The caller supplies an actual GPT-2 tokenizer and its closed artifact
    fingerprint, not a replacement sentence splitter or fixture controller.
    Request payloads cannot select endpoints, imports, assets or startup limits.
    Per-call stops belong to the backend contract: reasoning-model callers can
    opt into FinalContentStopBackend, charging all generated tokens.
    """

    worker_id = "agentinputlab.ircot.controlled_native_loop.v1"

    def __init__(
        self,
        *,
        source_root: Path,
        backend: Any,
        retriever: Any,
        tokenizer: Any,
        tokenizer_fingerprint: Mapping[str, Any],
        dependency_fingerprint: Mapping[str, Any],
        corpus_revision: str,
        index_revision: str,
        cache_home: Path,
        seed: int = 17,
        max_model_calls: int = 8,
        max_output_tokens: int = 4096,
        per_call_output_tokens: int = 512,
        max_retrieval_calls: int = 8,
    ) -> None:
        for name, value in (
            ("seed", seed),
            ("max_model_calls", max_model_calls),
            ("max_output_tokens", max_output_tokens),
            ("per_call_output_tokens", per_call_output_tokens),
            ("max_retrieval_calls", max_retrieval_calls),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"{name} has an invalid integer value")
        if not corpus_revision or not index_revision:
            raise ValueError("explicit corpus and index revisions are required")
        if not callable(getattr(backend, "generate", None)) or not callable(
            getattr(retriever, "retrieve", None)
        ):
            raise TypeError("real backend and retriever interfaces are required")
        if not callable(getattr(tokenizer, "tokenize", None)):
            raise TypeError("the pinned GPT-2 tokenizer must expose tokenize()")
        self.cache_home = Path(cache_home).resolve(strict=True)
        if Path(os.path.expanduser("~")).resolve() != self.cache_home:
            raise ValueError("launch a dedicated worker with HOME equal to cache_home")
        if not tokenizer_fingerprint or not dependency_fingerprint:
            raise ValueError("tokenizer and native dependency fingerprints are required")
        self.source_root = Path(source_root)
        self.backend, self.retriever, self.tokenizer = backend, retriever, tokenizer
        self.generator_fingerprint = _canonical_mapping(
            to_jsonable(backend.fingerprint()), "generator"
        )
        self.retriever_fingerprint = _canonical_mapping(
            to_jsonable(retriever.fingerprint()), "retriever"
        )
        self.tokenizer_fingerprint = _canonical_mapping(
            to_jsonable(tokenizer_fingerprint), "tokenizer"
        )
        self.dependency_fingerprint = _canonical_mapping(
            to_jsonable(dependency_fingerprint), "dependencies"
        )
        self.corpus_revision, self.index_revision = corpus_revision, index_revision
        self.seed, self.max_model_calls, self.max_output_tokens = (
            seed,
            max_model_calls,
            max_output_tokens,
        )
        self.per_call_output_tokens, self.max_retrieval_calls = (
            per_call_output_tokens,
            max_retrieval_calls,
        )
        self.last_events: list[dict[str, Any]] = []
        self.last_usage: dict[str, Any] = {}

    def _checkout(self):
        if Path(os.path.expanduser("~")).resolve() != self.cache_home:
            raise DriverError("CACHE_SCOPE_CHANGED", "dedicated worker HOME changed")
        original = IRCoTDriver(
            manifest_path=DEFAULT_MANIFEST,
            profile_id=SOURCE_PROFILE,
            environment={"AGENTINPUTLAB_UPSTREAM_ROOT": str(self.source_root)},
        )._checkout()
        return replace(
            original,
            profile_id=PROFILE,
            profile={
                **original.profile,
                "identity_kind": "controlled_model_transfer",
                "corpus_revision": self.corpus_revision,
                "index_revision": self.index_revision,
                "binding_attestation": "startup_exact_backend_retriever_tokenizer_dependencies",
            },
        )

    def fingerprint(self) -> Mapping[str, Any]:
        checkout = self._checkout()
        return {
            "worker_id": self.worker_id,
            "profile_id": PROFILE,
            "identity_kind": "controlled_model_transfer",
            "upstream_commit": UPSTREAM_COMMIT,
            "source_profile": SOURCE_PROFILE,
            "manifest_sha256": checkout.manifest_sha256,
            "native_config_sha256": checkout.profile["config_sha256"],
            "prompt_sha256": checkout.profile["prompt_sha256"],
            "generator": copy.deepcopy(self.generator_fingerprint),
            "retriever": copy.deepcopy(self.retriever_fingerprint),
            "tokenizer": copy.deepcopy(self.tokenizer_fingerprint),
            "native_dependencies": copy.deepcopy(self.dependency_fingerprint),
            "corpus_revision": self.corpus_revision,
            "index_revision": self.index_revision,
            "seed": self.seed,
            "max_model_calls": self.max_model_calls,
            "max_output_tokens": self.max_output_tokens,
            "per_call_output_tokens": self.per_call_output_tokens,
            "max_retrieval_calls": self.max_retrieval_calls,
            "implicit_retry": False,
            "historical_corpus_or_Codex_parity": False,
            "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def call_controlled(self, request):
        return self.call(request.as_dict())

    def call(self, payload):
        if not _NATIVE_LOCK.acquire(blocking=False):
            raise RuntimeError(
                "IRCoT native import/cwd state requires one call per dedicated worker"
            )
        bindings = None
        self.last_events, self.last_usage = [], {}
        try:
            request = _parse_request(payload)
            checkout = self._checkout()
            if (
                to_jsonable(request.generator_fingerprint) != self.generator_fingerprint
                or to_jsonable(self.backend.fingerprint()) != self.generator_fingerprint
                or to_jsonable(request.retriever_fingerprint) != self.retriever_fingerprint
                or to_jsonable(self.retriever.fingerprint()) != self.retriever_fingerprint
                or request.corpus_revision != self.corpus_revision
                or request.index_revision != self.index_revision
                or request.prompt_hash != checkout.profile["prompt_sha256"]
            ):
                raise DriverError(
                    "BINDING_MISMATCH", "request or startup component identity changed"
                )
            if (
                request.controls.max_steps > self.max_model_calls
                or request.controls.max_output_tokens > self.max_output_tokens
            ):
                raise DriverError("CONTROL_BUDGET_EXCEEDED", "request exceeds startup model limits")
            bindings = _LiveBindings(self, request)
            config = _patch_config(
                _load_config(
                    checkout,
                    {
                        "AGENTINPUTLAB_IRCOT_RETRIEVER_HOST": "http://127.0.0.1",
                        "AGENTINPUTLAB_IRCOT_RETRIEVER_PORT": "0",
                    },
                ),
                checkout,
                request.controls,
            )
            for model in config["models"].values():
                if model["name"] == "retrieve_and_reset_paragraphs":
                    model["source_corpus_name"] = self.corpus_revision
            capture = _BoundedCapture()
            with (
                _working_directory(checkout.root),
                _upstream_import_scope(checkout.root, fixture_shims=False),
                contextlib.redirect_stdout(capture),
            ):
                configurable = importlib.import_module("commaqa.inference.configurable_inference")
                native = importlib.import_module("commaqa.inference.ircot")
                prompt_reader = importlib.import_module("commaqa.inference.prompt_reader")
                prompt_reader.get_tokenizer = bindings.get_tokenizer
                bindings.fit_prompt = prompt_reader.fit_prompt_into_given_limit
                native.GPT3Generator = bindings.generator_class()
                retrieval = _RetrievalCapture(bindings.retrieve)
                native.safe_post_request = retrieval
                decomposer, model_map = configurable.build_decomposer_and_models(
                    copy.deepcopy(config)
                )
                tracing = _TracingController(
                    decomposer.controller,
                    {k: v["name"] for k, v in config["models"].items()},
                    retrieval,
                )
                decomposer.controller = tracing
                example = {
                    "qid": request.task_id or request.request_id,
                    "query": request.question,
                    "question": request.question,
                    "metadata": {"pinned_title": None, "pinned_para": None, "pin_position": None},
                    "titles": [],
                    "paras": [],
                }
                final_state, _ = decomposer.find_answer_decomp(example, debug=False)
                result = _project_trajectory(
                    request,
                    checkout,
                    config,
                    tracing.events,
                    final_state,
                    native,
                    model_map,
                    bindings,
                )
                participant_calls = result["usage"]["model_calls"]
                result["usage"] = bindings.usage()
                result["native_controller_events"] = to_jsonable(tracing.events)
                result["native_participant_model_call_counter"] = participant_calls
                result["controlled_runtime_events"] = copy.deepcopy(self.last_events)
                result["unknown_token_usage_model_calls"] = bindings.unknown_usage_calls
                result["usage_is_complete"] = bindings.unknown_usage_calls == 0
                result["upstream_stdout_sha256"] = capture.hexdigest
                return result
        except (Exception, SystemExit) as exc:
            response = {
                "status": "error",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": payload.get("request_id", "") if isinstance(payload, Mapping) else "",
                "failure_code": exc.code
                if isinstance(exc, DriverError)
                else "CONTROLLED_NATIVE_EXECUTION_FAILED",
                "message": exc.public_message
                if isinstance(exc, DriverError)
                else "controlled native IRCoT execution failed",
                "error_type": type(exc).__name__,
            }
            if bindings is None or (not bindings.generations and not bindings.retrievals):
                response["consumed"] = False
            if bindings is not None:
                response["usage"] = bindings.usage()
                response["controlled_runtime_events"] = copy.deepcopy(self.last_events)
                response["unknown_token_usage_model_calls"] = bindings.unknown_usage_calls
                response["usage_is_complete"] = bindings.unknown_usage_calls == 0
                response["reported_tokens_are_observed_lower_bounds"] = (
                    bindings.unknown_usage_calls > 0
                )
            return response
        finally:
            self.last_usage = bindings.usage() if bindings else {}
            _NATIVE_LOCK.release()


class _LiveBindings:
    def __init__(self, owner, request):
        self.owner, self.request = owner, request
        self.generations: list[dict[str, Any]] = []
        self.retrievals = 0
        self.input_tokens = self.output_tokens = 0
        self.fit_prompt = None

    def get_tokenizer(self, name):
        if name != "gpt2":
            raise DriverError("TOKENIZER_MISMATCH", "native prompt requested a different tokenizer")
        return self.owner.tokenizer

    def generator_class(self):
        binding = self

        class Generator:
            def __init__(self, **kwargs):
                self.options = dict(kwargs)
                if kwargs.get("n", 1) != 1 or kwargs.get("best_of", 1) != 1:
                    raise DriverError(
                        "GENERATOR_CONFIGURATION_INVALID", "native branch count changed"
                    )

            def generate_text_sequence(self, prompt):
                owner = binding.owner
                remaining = binding.request.controls.max_output_tokens - binding.output_tokens
                if (
                    len(binding.generations)
                    >= min(owner.max_model_calls, binding.request.controls.max_steps)
                    or remaining < 1
                ):
                    raise DriverError(
                        "MODEL_BUDGET_EXCEEDED", "generation budget exhausted before dispatch"
                    )
                cap = min(
                    owner.per_call_output_tokens, self.options.get("max_tokens", 300), remaining
                )
                prompt = binding.fit_prompt(
                    original_prompt=prompt.rstrip(),
                    model_length_limit=8000,
                    estimated_generation_length=cap,
                    demonstration_delimiter="\n\n\n",
                    shuffle=False,
                    remove_method=self.options.get("remove_method", "first"),
                    tokenizer_model_name="gpt2",
                    last_is_test_example=True,
                )
                call = GenerationRequest(
                    prompt=prompt,
                    temperature=self.options.get("temperature", 0),
                    max_tokens=cap,
                    seed=owner.seed + len(binding.generations),
                    stop=tuple(self.options.get("stop", ["\n"])),
                )
                event = {
                    "operation": "native_generator",
                    "request_hash": call.request_hash,
                    "prompt_sha256": content_hash(prompt),
                    "max_tokens": cap,
                    "seed": call.seed,
                    "stop": list(call.stop),
                    "response_observed": False,
                }
                binding.generations.append(event)
                owner.last_events.append(event)
                response = owner.backend.generate(call)
                event.update(
                    response_observed=True,
                    response_text=response.text,
                    finish_reason=response.finish_reason,
                    usage=to_jsonable(response.usage),
                )
                binding.input_tokens += response.usage.input_tokens
                binding.output_tokens += response.usage.output_tokens
                if response.usage.output_tokens > cap:
                    raise DriverError(
                        "OUTPUT_BUDGET_EXCEEDED",
                        "provider exceeded reserved output cap",
                        pre_execution=False,
                    )
                return [(response.text, 0.0)]

        return Generator

    def retrieve(self, url, params):
        if (
            url != "http://127.0.0.1:0/retrieve"
            or params.get("corpus_name") != self.owner.corpus_revision
            or params.get("retrieval_method") != "retrieve_from_elasticsearch"
            or params.get("document_type") != "title_paragraph_text"
            or set(params)
            != {"retrieval_method", "query_text", "max_hits_count", "corpus_name", "document_type"}
        ):
            raise DriverError("RETRIEVAL_CONFIGURATION_INVALID", "native retrieval binding changed")
        if self.retrievals >= self.owner.max_retrieval_calls:
            raise DriverError(
                "RETRIEVAL_BUDGET_EXCEEDED", "retrieval budget exhausted before dispatch"
            )
        count = params["max_hits_count"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= self.request.controls.per_step_top_k
        ):
            raise DriverError("RETRIEVAL_CONFIGURATION_INVALID", "native retrieval count changed")
        self.retrievals += 1
        hits = self.owner.retriever.retrieve(params["query_text"], top_k=count)
        documents = [
            {
                "id": h.document.doc_id,
                "title": h.document.title,
                "paragraph_text": h.document.text,
                "score": h.score,
                "corpus_name": self.owner.corpus_revision,
            }
            for h in hits
        ]
        self.owner.last_events.append(
            {
                "operation": "native_retrieval",
                "query": params["query_text"],
                "top_k": count,
                "documents": copy.deepcopy(documents),
            }
        )
        return _FrozenResponse({"retrieval": documents})

    @property
    def unknown_usage_calls(self):
        return sum(not g["response_observed"] for g in self.generations)

    def usage(self):
        unknown = self.unknown_usage_calls
        result = {
            "model_calls": len(self.generations),
            "retrieval_calls": self.retrievals,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
        costs = [g["usage"]["cost_usd"] for g in self.generations if g["response_observed"]]
        if (
            self.generations
            and not unknown
            and len(costs) == len(self.generations)
            and all(c is not None for c in costs)
        ):
            result["cost_usd"] = sum(costs)
        else:
            result["unpriced_model_calls"] = len(self.generations)
        return result


__all__ = ["PinnedIRCoTControlledRuntime"]
