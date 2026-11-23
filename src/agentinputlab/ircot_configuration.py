"""File-configured isolated IRCoT; original native controller and budgets remain authoritative."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

CONTROLLED = "acquire.ircot.controlled"
FULL = "acquire.ircot.native_like_substituted"
HOTPOT_INPUT = "hotpot_question_with_external_context.v1"
HOTPOT_INSTRUCTION = "Answer the multi-hop question using the provided paragraphs and identify the supporting facts."


def validate_evaluation_input(value, config, *, benchmark_id, split):
    """Opt into the native answer-only contract, never implicitly discard policy."""
    from collections.abc import Mapping

    if (not isinstance(value, Mapping)
            or set(value) != {"projection", "acknowledge_answer_only"}
            or value.get("projection") != HOTPOT_INPUT
            or value.get("acknowledge_answer_only") is not True):
        raise ValidationError("ircot_input requires hotpot_question_with_external_context.v1 and acknowledge_answer_only=true")
    if (config.release_id != FULL or config.runtime.get("kind") != "ircot_native"
            or benchmark_id != "hotpotqa.distractor" or split != "validation"
            or config.retriever.get("kind") != "bm25"):
        raise ValidationError("ircot_input requires native-like IRCoT, Hotpot validation and its explicit BM25 paragraph index")


def prepare_evaluation_input(state, config, retriever):
    """Verify all visible paragraphs are in the already-bound external index.

    This is an explicit answer-only evaluation variant. The benchmark's support
    scoring remains unchanged; missing support predictions are not filled in.
    No gold fields or example object are accepted here.
    """
    from agentinputlab.backends import Document, InMemoryBM25Retriever
    from agentinputlab.contracts.state import FrozenAgentState

    if (state.instruction != HOTPOT_INSTRUCTION or state.messages or state.tool_names
            or dict(state.metadata) != {"benchmark_cell": "hotpotqa.distractor", "source_split": "validation"}
            or not state.context):
        raise ValidationError("ircot_input cannot discard changed instructions, conversation, tools or metadata")
    corpus = content_hash(state.context)
    documents = tuple(Document(block.block_id or f"p{i}", block.text,
        title=str(block.metadata.get("title", block.block_id or f"p{i}")),
        corpus_revision=corpus) for i, block in enumerate(state.context))
    if len({doc.doc_id for doc in documents}) != len(documents):
        raise ValidationError("ircot_input requires unique paragraph IDs")
    expected = InMemoryBM25Retriever(documents, k1=config.retriever.get("k1", 1.2),
                                   b=config.retriever.get("b", 0.75)).fingerprint()
    if (not isinstance(retriever, InMemoryBM25Retriever)
            or retriever.fingerprint() != expected
            or config.runtime.get("corpus_revision") != corpus
            or config.runtime.get("index_revision") != content_hash(expected)):
        raise ValidationError("ircot_input paragraph corpus/index differs from this benchmark example")
    projected = FrozenAgentState(query=state.query, task_id=state.task_id, budget=state.budget)
    return projected, {"projection": HOTPOT_INPUT, "acknowledge_answer_only": True,
        "original_instruction": state.instruction, "source_input_state_hash": state.state_hash,
        "execution_state_hash": projected.state_hash, "corpus_revision": corpus,
        "index_revision": content_hash(expected), "paragraphs": len(documents),
        "support_predictions_synthesized": False, "native_scorer_unchanged": True,
        "model_calls": 0, "retrieval_calls": 0}


def validate_runtime(config):
    from agentinputlab.configuration import _known

    value = config.runtime
    if config.release_id not in {CONTROLLED, FULL}:
        raise ValidationError("ircot_native supports only controlled and native_like_substituted IRCoT")
    _known(value, {"kind", "source_root", "python_executable", "tokenizer_root",
                   "tokenizer_manifest", "tokenizer_manifest_sha256", "nltk_data", "corpus_revision",
                   "index_revision", "per_call_output_tokens", "timeout_seconds"}, "runtime")
    for key in ("source_root", "python_executable", "tokenizer_root", "tokenizer_manifest",
                "nltk_data", "corpus_revision", "index_revision"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValidationError(f"runtime.{key} must be non-empty text")
    if not re.fullmatch(r"[0-9a-f]{64}", str(value.get("tokenizer_manifest_sha256", ""))):
        raise ValidationError("runtime.tokenizer_manifest_sha256 must pin the tokenizer closure")
    cap = value.get("per_call_output_tokens", 512)
    if type(cap) is not int or cap < 1:
        raise ValidationError("runtime.per_call_output_tokens must be a positive integer")
    timeout = value.get("timeout_seconds", 600)
    try:
        valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError("runtime.timeout_seconds must be positive and finite")
    if not config.backend or not config.retriever:
        raise ValidationError("isolated IRCoT needs explicit file backend and retriever bindings")
    if config.backend.get("kind", "openai_compatible") != "openai_compatible":
        raise ValidationError("isolated IRCoT needs an OpenAI-compatible chat backend")
    if config.budget.max_model_calls < 1 or config.budget.max_retrieval_calls < 1 or config.budget.max_output_tokens < 1:
        raise ValidationError("IRCoT needs positive model, retrieval and output budgets")
    if config.budget.max_context_tokens is not None or config.budget.max_cost_usd is not None:
        raise ValidationError("native IRCoT cannot pre-attest input-token or priced-cost ceilings")


class FileIRCoTRuntime:
    """A bounded native process per operation, without changing the caller HOME.

    Only file-defined backend/retriever configurations cross the boundary. The
    worker verifies their fingerprints before dispatch. No credentials enter
    JSON, no fixture shims, no source-selected commands and no implicit retries.
    """

    def __init__(self, config, backend, retriever):
        validate_runtime(config)
        self.config, self.backend, self.retriever = config, backend, retriever
        self.worker_id = ("agentinputlab.ircot.native_like_full_qa.v1" if config.release_id == FULL
                          else "agentinputlab.ircot.controlled_native_loop.v1")
        value = config.runtime
        interpreter = Path(value["python_executable"])
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValidationError("IRCoT worker Python is missing or not executable; prepare the isolated Python 3.10 environment")
        for key in ("source_root", "tokenizer_root", "nltk_data"):
            if not Path(value[key]).is_dir():
                raise ValidationError(f"IRCoT {key} directory is missing")
        self.last_model_responses = []
        self._fingerprint = None

    def _invoke(self, operation, request=None):
        from agentinputlab.integrations.external_workers.base import JsonLineSubprocessWorker

        value = self.config.runtime
        # HOME is per native invocation; shared read-only tokenizer/NLTK assets
        # stay at their explicitly bound locations. No global env mutation.
        with tempfile.TemporaryDirectory(prefix="agentinputlab-ircot-") as home:
            environment = {"HOME": home, "NLTK_DATA": value["nltk_data"],
                "HF_HOME": home, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1",
                "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            key = self.config.backend.get("api_key_env", "OPENAI_API_KEY")
            passed = (key,) if os.environ.get(key) else ()
            worker = JsonLineSubprocessWorker(
                command=(value["python_executable"], "-I", "-B", "-m", "workers.ircot.configured"),
                worker_id=self.worker_id, identity={"protocol": "agentinputlab.ircot.file.v1"},
                timeout_seconds=value.get("timeout_seconds", 600), environment=environment,
                pass_environment=passed, working_directory=home,
                max_output_bytes=16 * 1024 * 1024)
            result = worker.call({"operation": operation, "configuration": self.config.to_dict(),
                "request": to_jsonable(request), "expected_identity": self._fingerprint,
                "generator": to_jsonable(self.backend.fingerprint()),
                "retriever": to_jsonable(self.retriever.fingerprint())})
        calls = result.get("model_responses", [])
        self.last_model_responses = calls
        # evaluate() installs this exact response collector before binding us.
        from agentinputlab.benchmark_configuration import _CapturedBackend

        if isinstance(self.backend, _CapturedBackend):
            self.backend.calls.extend(calls)
        if result.get("status") != "ok":
            raise ExternalDependencyError(
                "IRCoT isolated worker prerequisite/binding failed: "
                + str(result.get("failure_code", "BINDING")) + "/" + str(result.get("error_type", "unknown")))
        return result["result"]

    def fingerprint(self):
        if self._fingerprint is None:
            self._fingerprint = self._invoke("fingerprint")
        return self._fingerprint

    def call_controlled(self, request):
        return self.call(request.as_dict())

    def call(self, payload):
        if self.config.release_id != CONTROLLED:
            raise ValidationError("controlled calls cannot invoke the full-QA profile")
        self.fingerprint()
        return self._invoke("controlled", payload)

    def run(self, request):
        from agentinputlab.algorithms.acquire.ircot_native import IRCoTNativeQARequest

        if self.config.release_id != FULL or not isinstance(request, IRCoTNativeQARequest):
            raise ValidationError("full QA requires its own typed native request")
        self.fingerprint()
        return self._invoke("full", request)


def bind_case(config, context, state):
    if context.external_runtime is not None:
        raise ValidationError("external runtime is configured twice")
    runtime = FileIRCoTRuntime(config, context.backend, context.retriever)
    return replace(context, external_runtime=runtime)


def prepare(config, output_dir, *, accept_substitutions=False):
    """Check real assets, then create a ready file and explicit substitution receipt."""
    from agentinputlab.algorithms.acquire.ircot_native import write_ircot_substitution_manifest

    if accept_substitutions is not True:
        raise ValidationError("acknowledge the explicit model/index/tokenizer substitutions with --accept-substitutions")
    validate_runtime(config)
    backend, retriever = config._backend(), config._retriever()
    runtime = FileIRCoTRuntime(config, backend, retriever)
    path = Path(output_dir).absolute()
    if path.exists():
        raise ValidationError("IRCoT preparation needs a new output directory")
    fingerprint = runtime.fingerprint()  # No model or retrieval calls.
    parameters = dict(config.parameters)
    path.mkdir(parents=True)
    if config.release_id == FULL:
        receipt = write_ircot_substitution_manifest(runtime, path / "substitutions.json", accept_substitutions=True)
        parameters.update(manifest_path=receipt["path"], expected_manifest_sha256=receipt["sha256"])
    else:
        parameters.update(corpus_revision=fingerprint["corpus_revision"],
            index_revision=fingerprint["index_revision"], prompt_hash=fingerprint["prompt_sha256"])
    ready = replace(config, parameters=parameters)
    ready.validate()
    with (path / "run.json").open("x") as stream:
        json.dump(ready.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
    proof = {"runtime": fingerprint, "runtime_sha256": content_hash(fingerprint),
             "run_path": str(path / "run.json"), "model_calls": 0, "retrieval_calls": 0,
             "run_sha256": hashlib.sha256((path / "run.json").read_bytes()).hexdigest()}
    with (path / "prepared.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
    return proof
