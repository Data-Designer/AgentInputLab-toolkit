"""File binding for checkpoint RRR with a configured reader and visible retriever."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

RELEASES = {"rewrite.rrr.warmup_external": "warmup_checkpoint",
            "rewrite.rrr.ppo_external": "ppo_checkpoint"}


def validate_runtime(config):
    from agentinputlab.configuration import _known

    value = config.runtime
    if config.release_id not in RELEASES:
        raise ValidationError("rrr_checkpoint supports only warmup_external and ppo_external inference")
    _known(value, {"kind", "checkout", "checkpoint_root", "checkpoint_manifest_path",
                   "checkpoint_manifest_sha256", "rewrite_prompt_path", "reader_prompt_path",
                   "acknowledge_credential_quarantine", "rewriter_replay_path",
                   "rewriter_replay_sha256"}, "runtime")
    for key in ("checkout", "checkpoint_root", "checkpoint_manifest_path",
                "rewrite_prompt_path", "reader_prompt_path"):
        if not isinstance(value.get(key), str) or not value[key].strip() or "\x00" in value[key]:
            raise ValidationError(f"RRR runtime.{key} must be a local path")
    for key in ("checkpoint_manifest_sha256", "rewriter_replay_sha256"):
        if key == "rewriter_replay_sha256" and not any(
                x in value for x in ("rewriter_replay_path", key)):
            continue
        if not isinstance(value.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValidationError(f"RRR runtime.{key} must be a SHA-256 digest")
    if "rewriter_replay_sha256" in value:
        path = value.get("rewriter_replay_path")
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise ValidationError("RRR saved rewriter needs its path and SHA-256")
    if value.get("acknowledge_credential_quarantine") is not True:
        raise ValidationError("RRR requires explicit source credential-quarantine acknowledgement")
    parameters = config.parameters
    if parameters.get("source_license_acknowledged") is not True or parameters.get("checkpoint_license_acknowledged") is not True:
        raise ValidationError("RRR requires explicit source and checkpoint acknowledgements")
    controls = parameters.get("controls", {})
    if controls.get("search_mode") != "controlled_static" or controls.get("rewriter_profile") != RELEASES[config.release_id]:
        raise ValidationError("rrr_checkpoint requires controlled_static search and matching checkpoint profile")


class SavedRewriter:
    def __init__(self, path, digest, identity):
        from agentinputlab.artifacts import load_pinned_json_mapping

        saved = load_pinned_json_mapping(path, expected_sha256=digest)
        if (set(saved) != {"schema", "request", "response"}
                or saved.get("schema") != "agentinputlab.rrr.rewriter-replay.v1"):
            raise ValidationError("invalid saved RRR rewriter schema")
        self.saved, self.identity = saved, identity

    def fingerprint(self):
        return self.identity

    def generate(self, prompt, *, seed, max_tokens):
        from workers.rrr.driver import Generation

        expected = dict(prompt=prompt, seed=seed, max_tokens=max_tokens, checkpoint=self.identity)
        if content_hash(self.saved["request"]) != content_hash(expected):
            raise ValidationError("saved RRR rewrite input/seed/budget/checkpoint mismatch; live fallback disabled")
        response = self.saved["response"]
        if (not isinstance(response, dict) or set(response) != {"text", "input_tokens", "output_tokens"}
                or not isinstance(response["text"], str)
                or any(type(response[k]) is not int or response[k] < 0 for k in ("input_tokens", "output_tokens"))
                or response["output_tokens"] > max_tokens):
            raise ValidationError("invalid saved RRR generation or output budget")
        return Generation(**response)


def bind_case(config, context, state):
    from agentinputlab.backends import GenerationRequest
    from workers.rrr import driver

    validate_runtime(config)
    if context.external_runtime is not None:
        raise ValidationError("RRR external runtime is configured twice")
    if context.backend is None or context.retriever is None:
        raise ValidationError("rrr_checkpoint requires a configured reader backend and retriever")
    value = config.runtime
    method = config.create_intervention()
    # Verify all identities before loading a checkpoint or contacting a reader.
    for key, digest in (("checkpoint_manifest_path", value["checkpoint_manifest_sha256"]),
                        ("rewrite_prompt_path", method.prompt_hash),
                        ("reader_prompt_path", method.reader_prompt_hash)):
        if hashlib.sha256(Path(value[key]).read_bytes()).hexdigest() != digest:
            raise ValidationError(f"RRR {key} checksum mismatch")
    try:
        source = driver.verify_source_checkout(Path(value["checkout"]))
        checkpoint = driver.verify_checkpoint(Path(value["checkpoint_root"]), Path(value["checkpoint_manifest_path"]))
    except driver.DriverError as exc:
        raise ValidationError(f"RRR asset setup failed ({exc.code}): {exc.public_message}; checkpoint_root must contain the files listed in the pinned manifest") from exc
    if checkpoint["role"] != "rewriter" or checkpoint["profile"] != RELEASES[config.release_id] or checkpoint["revision"] != method.rewriter_checkpoint_revision:
        raise ValidationError("RRR checkpoint does not match the selected release/revision")
    identity = {key: checkpoint[key] for key in ("role", "profile", "revision", "manifest_sha256")}
    backend, bm25 = context.backend, context.retriever
    revision = backend.fingerprint().get("model_revision")
    if revision != method.reader_checkpoint_revision:
        raise ValidationError("RRR reader deployment revision differs from the frozen reader revision")

    class Search:
        def fingerprint(self):
            return dict(artifact_sha256=content_hash(bm25.fingerprint()), mode="controlled_static",
                corpus_revision=content_hash(dict(original_visible_state=state.state_hash, context=to_jsonable(state.context))),
                index_revision=content_hash(bm25.fingerprint()))

        def search(self, query, *, top_k, window_start, window_end):
            if window_start or window_end:
                raise ValidationError("visible-context RRR search cannot apply a time-window filter")
            return tuple(driver.SearchDocument(v.document.doc_id, v.document.title,
                v.document.text, v.score, {}) for v in bm25.retrieve(query, top_k=top_k))

        def retrieve(self, query, *, top_k=5):
            return bm25.retrieve(query, top_k=top_k)

    search = Search()
    fp = search.fingerprint()
    if (method.search_artifact_hash != fp["artifact_sha256"] or method.index_revision != fp["index_revision"]
            or method.corpus_revision != fp["corpus_revision"]):
        raise ValidationError("RRR state/retriever binding mismatch")
    try:
        rewriter = (SavedRewriter(value["rewriter_replay_path"], value["rewriter_replay_sha256"], identity)
                    if "rewriter_replay_path" in value else driver.LocalHFGenerator(checkpoint))
    except driver.DriverError as exc:
        raise ValidationError(f"RRR checkpoint setup failed ({exc.code}): {exc.public_message}; prepare local checkpoint inference dependencies") from exc

    class Reader:
        def fingerprint(self):
            return dict(role="reader", profile="frozen_reader", revision=revision, backend=backend.fingerprint())

        def generate(self, prompt, *, seed, max_tokens):
            result = backend.generate(GenerationRequest(prompt=prompt, seed=seed, max_tokens=max_tokens,
                temperature=0.0, stop=(), metadata=dict(release=config.release_id, native_stage="reader")))
            return driver.Generation(result.text, result.usage.input_tokens, result.usage.output_tokens)

    reader = Reader()
    native = driver.RRRRuntime(source, Path(value["rewrite_prompt_path"]), Path(value["reader_prompt_path"]),
        rewriter, search, reader, True, True, True)

    class Pair:
        model_id = backend.model_id
        capabilities = backend.capabilities

        def fingerprint(self):
            return dict(rewriter=rewriter.fingerprint(), reader=reader.fingerprint())

        def generate(self, request):
            raise ValidationError("RRR worker owns its two stages; no implicit third reader")

    class Client:
        def fingerprint(self):
            return {**native.fingerprint(), "file_runtime": "rrr_checkpoint.v1",
                "rewriter_execution": "explicit_saved_generation" if "rewriter_replay_path" in value else "local_checkpoint",
                "rewriter_replay_sha256": value.get("rewriter_replay_sha256"),
                "replay_usage_is_original_logical_cost": "rewriter_replay_path" in value}

        def call(self, request):
            try:
                return native.call(request)
            except driver.DriverError as exc:
                return driver._failure(exc, request.get("request_id", ""))

        def call_controlled(self, request):
            return self.call(request.as_dict())

    return replace(context, backend=Pair(), retriever=search, external_runtime=Client())
