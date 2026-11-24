"""Installed Q-RAG native composite worker; stdin configuration, one execution."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable


class _Policy:
    def __init__(self, settings, artifacts):
        self.settings, self.artifacts = settings, artifacts

    def select(self, *, question, candidates, seed, max_steps):
        from agentinputlab.algorithms.acquire.qrag.common_support import (
            LocalQRAGFrozenPolicyRuntime,
            QRAGPolicyRequest,
            _validate_response,
        )
        from agentinputlab.backends.retrieval import Document, RetrievalResult
        from agentinputlab.qrag_configuration import load_policy_replay
        from workers.qrag.driver import SelectorResult

        value = self.settings
        request = {"question": question, "candidates": candidates, "seed": seed, "max_steps": max_steps}
        if "policy_replay_path" in value:
            response = load_policy_replay(value, self.artifacts, request)
        else:
            if not candidates:
                return SelectorResult(trajectory=(), selected_indices=(), embedding_calls=0)
            runtime = LocalQRAGFrozenPolicyRuntime(
                source_root=Path(value["source_root"]), checkpoint_root=Path(value["checkpoint_root"]),
                embedder_root=Path(value["embedder_root"]),
                compatibility_manifest_path=Path(value["compatibility_manifest_path"]),
                compatibility_manifest_sha256=self.artifacts["compatibility_manifest_sha256"],
                embedder_manifest_sha256=self.artifacts["embedder_manifest_sha256"],
                device=value.get("device", "cpu"),
                single_candidate_axis_compat=value.get("single_candidate_axis_compat", False))
            response = runtime.select(QRAGPolicyRequest(question=question, seed=seed,
                max_steps=max_steps, candidates=tuple(RetrievalResult(
                    Document(row["document_id"], row["text"]), 0.0, index + 1)
                    for index, row in enumerate(candidates))))
        ids = tuple(row["document_id"] for row in candidates)
        trajectory, selected, usage, _ = _validate_response(response, ids)
        if response.get("candidate_text_sha256") != content_hash(tuple(row["text"].strip() for row in candidates)):
            raise ValidationError("Q-RAG policy candidate text checksum mismatch")
        return SelectorResult(trajectory=trajectory, selected_indices=selected,
            embedding_calls=usage["embedding_calls"], accelerator_seconds=usage["accelerator_seconds"])


def execute(envelope, calls):
    from agentinputlab import RunConfig
    from agentinputlab.algorithms.acquire.qrag.bridge import (
        QRAG_READER_SYSTEM_PROMPT,
        QRAG_READER_USER_TEMPLATE,
    )
    from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
        QRAG_INSTRUCTION_SYSTEM_PROMPT,
        QRAG_QUERY_READER_PROFILE,
        QRAG_QUERY_SYSTEM_PROMPT,
        QRAG_TASK_READER_PROFILES,
        render_instruction_reader_prompt,
    )
    from agentinputlab.backends import GenerationRequest
    from agentinputlab.benchmark_configuration import _CapturedBackend
    from agentinputlab.qrag_configuration import validate_runtime
    from workers.qrag.driver import QRAGFrozenDriver, ReaderResult

    config = RunConfig.from_dict(envelope["configuration"])
    config.validate()
    validate_runtime(config)
    method = config.create_intervention()
    request = envelope["request"]
    payload = request["payload"]
    if request["release_id"] != config.release_id or payload["seed"] != config.seed:
        raise ValidationError("Q-RAG request release/seed differs from file configuration")
    if content_hash(payload["artifacts"]) != method.artifacts.identity_sha256:
        raise ValidationError("Q-RAG request artifacts differ from file configuration")
    backend = _CapturedBackend(config._backend())
    backend.calls = calls
    fingerprint = content_hash(backend.fingerprint())
    if (fingerprint != content_hash(envelope["reader_fingerprint"])
            or fingerprint != method.artifacts.reader_deployment_fingerprint_sha256):
        raise ValidationError("Q-RAG reader deployment changed across the worker boundary")

    def reader(question, passages, artifacts, *, max_output_tokens,
               task_instruction=None, passage_annotations=None):
        if artifacts["reader_profile"] in QRAG_TASK_READER_PROFILES:
            system = (QRAG_QUERY_SYSTEM_PROMPT
                      if artifacts["reader_profile"] == QRAG_QUERY_READER_PROFILE
                      else QRAG_INSTRUCTION_SYSTEM_PROMPT)
            prompt = render_instruction_reader_prompt(question, passages,
                task_instruction=task_instruction, passage_annotations=passage_annotations)
        else:
            system = QRAG_READER_SYSTEM_PROMPT
            prompt = QRAG_READER_USER_TEMPLATE.format(question=question, context="\n\n---\n\n".join(passages))
        response = backend.generate(GenerationRequest(system=system, prompt=prompt,
            max_tokens=max_output_tokens, seed=config.seed, temperature=0))
        return ReaderResult(response.text, response.usage.input_tokens, response.usage.output_tokens)

    value = config.runtime
    driver = QRAGFrozenDriver(source_root=Path(value["source_root"]),
        checkpoint_root=Path(value["checkpoint_root"]), embedder_root=Path(value["embedder_root"]),
        manifest_path=Path(value["compatibility_manifest_path"]),
        runtime_factory=lambda *_paths: _Policy(value, payload["artifacts"]), reader=reader)
    return driver.handle(request)


def main():
    calls = []
    try:
        raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValidationError("Q-RAG worker request exceeds 8 MiB")
        envelope = json.loads(raw)
        with contextlib.redirect_stdout(sys.stderr):
            response = execute(envelope, calls)
    except Exception as exc:
        response = {"status": "error", "failure_code": "QRAG_CONFIGURED_EXECUTION_FAILED",
                    "message": type(exc).__name__, "consumed": bool(calls)}
    print(json.dumps(to_jsonable({"response": response, "model_responses": calls}), allow_nan=False))


if __name__ == "__main__":
    main()
