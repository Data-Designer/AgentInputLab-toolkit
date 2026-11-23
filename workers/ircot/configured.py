"""Isolated file-config entry for the existing pinned IRCoT native runtimes."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PrerequisiteError(Exception):
    def __init__(self, code, error_type):
        self.code, self.error_type = code, error_type
        super().__init__(code)


def _prerequisite(code, operation):
    try:
        return operation()
    except Exception as exc:
        raise PrerequisiteError(code, type(exc).__name__) from None


def _tokenizer(value):
    from transformers import AutoTokenizer

    from agentinputlab.artifacts import load_pinned_json_mapping

    closure = load_pinned_json_mapping(value["tokenizer_manifest"],
        expected_sha256=value["tokenizer_manifest_sha256"])
    root = Path(value["tokenizer_root"]).resolve(strict=True)
    files = closure.get("file_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("tokenizer closure requires file_sha256")
    for name, digest in files.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or _sha(path) != digest:
            raise ValueError("tokenizer closure file mismatch")
    # HF download locks/metadata do not participate in local from_pretrained().
    # Do not delete them, nor admit an unpinned added_tokens/config file.
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and not p.relative_to(root).is_relative_to(".cache/huggingface")}
    if actual != set(files):
        raise ValueError("tokenizer directory differs from its complete closure")
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    if not type(tokenizer).__module__.startswith("transformers.models.gpt2"):
        raise ValueError("native IRCoT requires the GPT-2 tokenizer")
    return tokenizer, closure


def _dependencies():
    from agentinputlab.contracts.serialization import content_hash

    versions = {name: importlib.metadata.version(name) for name in
        ("spacy", "en-core-web-sm", "transformers", "jsonnet", "nltk", "numpy", "openai")}
    if (versions["spacy"] != "3.4.1" or versions["en-core-web-sm"] != "3.4.1"
            or versions["transformers"] != "4.24.0"):
        raise ValueError("native IRCoT needs its pinned spaCy/parser/Transformers environment")
    parser = importlib.metadata.distribution("en-core-web-sm")
    parser_files = {str(p): _sha(parser.locate_file(p)) for p in parser.files or ()
                    if str(p).startswith("en_core_web_sm/") and "__pycache__" not in str(p)}
    if not parser_files:
        raise ValueError("native sentence parser closure is missing")
    nltk_root = Path(os.environ["NLTK_DATA"]).resolve(strict=True)
    stopwords = nltk_root / "corpora" / "stopwords" / "english"
    if not stopwords.is_file():
        raise ValueError("native NLTK English stopwords are missing")
    return {"versions": versions, "parser_closure_sha256": content_hash(parser_files),
            "nltk_english_stopwords_sha256": _sha(stopwords)}


def execute(payload, captures):
    from agentinputlab.benchmark_configuration import _CapturedBackend
    from agentinputlab.configuration import RunConfig
    from agentinputlab.contracts.serialization import content_hash, to_jsonable
    from agentinputlab.ircot_configuration import CONTROLLED, FULL, validate_runtime
    from workers.ircot.controlled import PinnedIRCoTControlledRuntime
    from workers.ircot.driver import _parse_request
    from workers.ircot.native_qa import PinnedIRCoTNativeQARuntime

    config = RunConfig.from_dict(payload["configuration"])
    validate_runtime(config)
    operation = payload["operation"]
    if operation not in {"fingerprint", "controlled", "full"}:
        raise ValueError("unknown IRCoT operation")
    if (operation == "full" and config.release_id != FULL) or (operation == "controlled" and config.release_id != CONTROLLED):
        raise ValueError("IRCoT native profile mismatch")
    backend = _CapturedBackend(config._backend())
    backend.calls = captures
    retriever = config._retriever()
    if (content_hash(backend.fingerprint()) != content_hash(payload["generator"])
            or content_hash(retriever.fingerprint()) != content_hash(payload["retriever"])):
        raise ValueError("IRCoT child backend/retriever differs from caller")
    value = config.runtime
    tokenizer, closure = _prerequisite("TOKENIZER_ASSETS", lambda: _tokenizer(value))
    dependencies = _prerequisite("NATIVE_DEPENDENCIES", _dependencies)
    runtime_type = PinnedIRCoTNativeQARuntime if config.release_id == FULL else PinnedIRCoTControlledRuntime
    runtime = runtime_type(source_root=Path(value["source_root"]), backend=backend, retriever=retriever,
        tokenizer=tokenizer, tokenizer_fingerprint=closure, dependency_fingerprint=dependencies,
        corpus_revision=value["corpus_revision"], index_revision=value["index_revision"],
        cache_home=Path(os.environ["HOME"]), seed=config.seed,
        max_model_calls=config.budget.max_model_calls, max_output_tokens=config.budget.max_output_tokens,
        max_retrieval_calls=config.budget.max_retrieval_calls,
        per_call_output_tokens=value.get("per_call_output_tokens", 512))
    identity = _prerequisite("PINNED_SOURCE", runtime.fingerprint)
    if operation == "fingerprint":
        return to_jsonable(identity)
    if content_hash(identity) != content_hash(payload["expected_identity"]):
        raise ValueError("IRCoT assets/runtime changed after preparation")
    if operation == "controlled":
        return runtime.call(payload["request"])
    from agentinputlab.algorithms.acquire.ircot_native import IRCoTNativeQARequest
    from agentinputlab.integrations.external_workers.ircot import IRCoTControls, IRCoTWorkerRequest

    request = dict(payload["request"])
    legacy = dict(request["legacy"])
    _parse_request(legacy)  # Retain the native wire-schema and bounds validation.
    legacy["controls"] = IRCoTControls(**legacy["controls"])
    # The QA runtime takes the public request and parses it itself. The driver's
    # internal Request has no as_dict() and is not this boundary's contract.
    request["legacy"] = IRCoTWorkerRequest(**legacy)
    return runtime.run(IRCoTNativeQARequest(**request))


def main():
    from agentinputlab.contracts.serialization import to_jsonable

    captures = []
    try:
        raw = sys.stdin.buffer.readline(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("IRCoT file request exceeds byte limit")
        payload = json.loads(raw)
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(payload, captures)
        response = {"status": "ok", "result": result}
    except PrerequisiteError as exc:
        response = {"status": "error", "error_type": exc.error_type, "failure_code": exc.code}
    except Exception as exc:
        # Never disclose provider error bodies or environment credentials.
        response = {"status": "error", "error_type": type(exc).__name__}
    response["model_responses"] = captures
    print(json.dumps(to_jsonable(response), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
