"""File binding for the existing frozen Q-RAG selector-plus-reader driver."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from agentinputlab.contracts.errors import AgentInputLabError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

RELEASES = {"acquire.qrag.released_checkpoint_native", "acquire.qrag.reader_transfer_factorial"}


def load_policy_replay(settings, artifacts, request):
    """Validate saved input/trajectory before reserving any physical inference."""
    from agentinputlab.algorithms.acquire.qrag.common_support import _validate_response
    from agentinputlab.artifacts import load_pinned_json_mapping

    saved = load_pinned_json_mapping(settings["policy_replay_path"],
        expected_sha256=settings["policy_replay_sha256"])
    if saved.get("schema") != "agentinputlab.qrag.policy-replay.v1" or content_hash(
            saved.get("request")) != content_hash(request):
        raise ValidationError("saved Q-RAG policy does not match this question/candidates/seed/horizon")
    if saved.get("artifact_identity_sha256") != content_hash(artifacts):
        raise ValidationError("saved Q-RAG policy artifact identity mismatch")
    if not isinstance(saved.get("response"), Mapping):
        raise ValidationError("saved Q-RAG policy response must be an object")
    response = freeze_mapping(saved["response"])
    candidates = request["candidates"]
    _validate_response(response, tuple(row["document_id"] for row in candidates))
    if response.get("candidate_text_sha256") != content_hash(tuple(row["text"].strip() for row in candidates)):
        raise ValidationError("Q-RAG policy candidate text checksum mismatch")
    return response


def validate_runtime(config):
    from agentinputlab.configuration import _known

    value = config.runtime
    if config.release_id not in RELEASES:
        raise ValidationError("qrag_native is for the frozen Q-RAG composite profiles, not common_support")
    _known(value, {"kind", "source_root", "checkpoint_root", "embedder_root",
                   "compatibility_manifest_path", "python_executable", "device", "timeout_seconds",
                   "policy_replay_path", "policy_replay_sha256", "prepared_state_sha256"}, "runtime")
    if value.get("kind") != "qrag_native":
        raise ValidationError("runtime.kind must be qrag_native")
    if "prepared_state_sha256" in value and (
            not isinstance(value["prepared_state_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["prepared_state_sha256"])):
        raise ValidationError("Q-RAG prepared state requires a SHA-256 digest")
    for key in ("source_root", "checkpoint_root", "embedder_root",
                "compatibility_manifest_path", "python_executable"):
        if not isinstance(value.get(key), str) or not value[key].strip() or "\x00" in value[key]:
            raise ValidationError(f"runtime.{key} must be a non-empty local path")
    if not isinstance(value.get("device", "cpu"), str) or not re.fullmatch(
            r"cpu|cuda(?::[0-9]+)?", value.get("device", "cpu")):
        raise ValidationError("Q-RAG device must be cpu or cuda with an optional index")
    timeout = value.get("timeout_seconds", 600)
    try:
        valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError("Q-RAG timeout_seconds must be positive and finite")
    if "policy_replay_path" in value or "policy_replay_sha256" in value:
        path, digest = value.get("policy_replay_path"), value.get("policy_replay_sha256")
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise ValidationError("Q-RAG policy replay requires a local path")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError("Q-RAG policy replay requires its trusted file SHA-256")
    if not config.backend or config.backend.get("kind", "openai_compatible") != "openai_compatible":
        raise ValidationError("qrag_native requires an explicit OpenAI-compatible file backend")


def prepare(config, state, output_dir, *, reader_repository, reader_revision, reader_profile):
    """Bind mounted assets and one visible candidate pool; no downloads or inference."""
    from agentinputlab.algorithms.acquire.qrag.bridge import (
        QRAG_CHECKPOINT_CONFIG_SHA256,
        QRAG_CHECKPOINT_FILE_SHA256,
        QRAG_READER_PROMPT_SHA256,
        QRAG_SOURCE_MANIFEST_SHA256,
        QRAGArtifactIdentity,
    )
    from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
        QRAG_TASK_READER_PROFILES,
        task_reader_prompt_hash,
    )
    from agentinputlab.configuration import _known
    from workers.qrag.driver import DriverError, QRAGFrozenDriver, _load_json, _require_profile

    validate_runtime(config)
    _known(config.parameters, {"controls", "acknowledgements"}, "preparation parameters")
    if config.artifacts or config.retriever or config.vector_runtime:
        raise ValidationError("Q-RAG preparation uses the given visible context; do not mix other asset/retriever bindings")
    output = Path(output_dir).absolute()
    if output.exists():
        raise ValidationError("Q-RAG preparation requires a new output directory")
    state = replace(state, budget=config.budget)
    if (not state.query.strip() or state.messages
            or (state.instruction.strip() and reader_profile not in QRAG_TASK_READER_PROFILES)
            or (reader_profile in QRAG_TASK_READER_PROFILES and state.tool_names)):
        raise ValidationError("visible state is not supported by the explicit Q-RAG reader profile")
    ids = [block.block_id for block in state.context]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValidationError("Q-RAG candidates require unique non-empty IDs")
    candidates = [{"document_id": block.block_id, "text": block.text} for block in state.context]
    backend = config._backend()
    settings = config.runtime
    # Reuse the original driver verification; never load checkpoint tensors here.
    try:
        driver = QRAGFrozenDriver(source_root=Path(settings["source_root"]),
            checkpoint_root=Path(settings["checkpoint_root"]), embedder_root=Path(settings["embedder_root"]),
            manifest_path=Path(settings["compatibility_manifest_path"]),
            runtime_factory=None, reader=None)
        embedder = _load_json(Path(settings["embedder_root"]) / "agentinputlab-manifest.json")
        identity = QRAGArtifactIdentity(
            checkpoint_manifest_sha256=content_hash({"config.yaml": QRAG_CHECKPOINT_CONFIG_SHA256,
                                                    "model_best.pt": QRAG_CHECKPOINT_FILE_SHA256}),
            checkpoint_file_sha256=QRAG_CHECKPOINT_FILE_SHA256,
            embedder_manifest_sha256=content_hash(embedder), source_manifest_sha256=QRAG_SOURCE_MANIFEST_SHA256,
            compatibility_manifest_sha256=driver.manifest_sha256,
            retriever_revision="given-visible-context-native-order-no-live-search.v1",
            retriever_deployment_fingerprint_sha256=content_hash({"given_candidates": candidates}),
            corpus_sha256=content_hash(state.context), index_sha256=content_hash(ids),
            reader_repository=reader_repository, reader_revision=reader_revision,
            reader_deployment_fingerprint_sha256=content_hash(backend.fingerprint()),
            reader_profile=reader_profile, reader_prompt_sha256=(task_reader_prompt_hash(reader_profile)
                if reader_profile in QRAG_TASK_READER_PROFILES else QRAG_READER_PROMPT_SHA256))
        ready = replace(config, parameters={**config.parameters, "artifacts": to_jsonable(identity)},
                        runtime={**settings, "prepared_state_sha256": state.state_hash})
        ready.validate()
        method = ready.create_intervention()
        method.acknowledgements.require_complete()
        method._validate_reader_profile()
        if (len(candidates) > method.controls.max_candidates
                or sum(len(row["text"]) for row in candidates) > method.controls.max_candidate_chars):
            raise ValidationError("Q-RAG visible candidates exceed the configured limits")
        _require_profile(identity.as_mapping(), driver.manifest, driver.manifest_sha256, fixture=False)
        driver._verify_source(identity.as_mapping())
        driver._verify_checkpoint(identity.as_mapping())
        driver._verify_embedder(identity.as_mapping())
        if "policy_replay_path" in settings:
            load_policy_replay(settings, identity.as_mapping(), {"question": state.query,
                "candidates": candidates, "seed": config.seed, "max_steps": 2})
    except (DriverError, OSError) as exc:
        raise ValidationError(f"Q-RAG mounted assets failed verification: {type(exc).__name__}") from None
    output.mkdir(parents=True)
    for name, value in (("run.json", ready.to_dict()), ("state.json", to_jsonable(state))):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
    proof = {"release_id": ready.release_id, "artifact_identity_sha256": identity.identity_sha256,
             "state_sha256": state.state_hash, "run_path": str(output / "run.json"),
             "run_sha256": hashlib.sha256((output / "run.json").read_bytes()).hexdigest(),
             "model_calls": 0, "embedding_calls": 0, "fit_performed": False,
             "scope": "one_visible_state_given_candidates", "asset_files_verified": True}
    with (output / "prepared.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
    return proof


def _read_batch_assets(config, reader):
    """Verify shared mounted assets once, before any batch model execution."""
    from agentinputlab.algorithms.acquire.qrag.bridge import (
        QRAG_CHECKPOINT_CONFIG_SHA256,
        QRAG_CHECKPOINT_FILE_SHA256,
        QRAG_READER_PROMPT_SHA256,
        QRAG_SOURCE_MANIFEST_SHA256,
        QRAGArtifactIdentity,
    )
    from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
        QRAG_TASK_READER_PROFILES,
        task_reader_prompt_hash,
    )
    from workers.qrag.driver import DriverError, QRAGFrozenDriver, _load_json, _require_profile

    value = config.runtime
    try:
        driver = QRAGFrozenDriver(source_root=Path(value["source_root"]),
            checkpoint_root=Path(value["checkpoint_root"]), embedder_root=Path(value["embedder_root"]),
            manifest_path=Path(value["compatibility_manifest_path"]), runtime_factory=None, reader=None)
        identity = QRAGArtifactIdentity(
            checkpoint_manifest_sha256=content_hash({"config.yaml": QRAG_CHECKPOINT_CONFIG_SHA256,
                                                    "model_best.pt": QRAG_CHECKPOINT_FILE_SHA256}),
            checkpoint_file_sha256=QRAG_CHECKPOINT_FILE_SHA256,
            embedder_manifest_sha256=content_hash(_load_json(Path(value["embedder_root"]) / "agentinputlab-manifest.json")),
            source_manifest_sha256=QRAG_SOURCE_MANIFEST_SHA256, compatibility_manifest_sha256=driver.manifest_sha256,
            retriever_revision="given-visible-context-native-order-no-live-search.v1",
            retriever_deployment_fingerprint_sha256=content_hash({"given_candidates": []}),
            corpus_sha256=content_hash(()), index_sha256=content_hash([]),
            reader_repository=reader["repository"], reader_revision=reader["revision"],
            reader_deployment_fingerprint_sha256=content_hash(config._backend().fingerprint()),
            reader_profile=reader["profile"], reader_prompt_sha256=(task_reader_prompt_hash(reader["profile"])
                if reader["profile"] in QRAG_TASK_READER_PROFILES else QRAG_READER_PROMPT_SHA256))
        _require_profile(identity.as_mapping(), driver.manifest, driver.manifest_sha256, fixture=False)
        driver._verify_source(identity.as_mapping())
        driver._verify_checkpoint(identity.as_mapping())
        driver._verify_embedder(identity.as_mapping())
        return to_jsonable(identity)
    except (DriverError, OSError) as exc:
        raise ValidationError(f"Q-RAG mounted assets failed verification: {type(exc).__name__}") from None


class QRAGBatchBinder:
    """Prepare fresh per-case identities without exposing labels or fitting on test data."""

    def __init__(self, config, reader):
        from agentinputlab.algorithms.acquire.qrag.reader_adapter import QRAG_TASK_READER_PROFILES
        from agentinputlab.artifacts import load_pinned_json_mapping
        from agentinputlab.configuration import _known, _object

        validate_runtime(config)
        reader = _object(reader, "Q-RAG reader")
        _known(reader, {"repository", "revision", "profile", "policy_replays", "candidate_source",
                        "longbench_context"}, "Q-RAG reader")
        for key in ("repository", "revision", "profile"):
            if not isinstance(reader.get(key), str) or not reader[key].strip():
                raise ValidationError(f"Q-RAG reader.{key} must be explicit")
        profiles = ({"upstream_qwq32b_family_controlled"}
                    if config.release_id == "acquire.qrag.released_checkpoint_native" else
                    {"factorial_reader_transfer", *QRAG_TASK_READER_PROFILES})
        if reader["profile"] not in profiles:
            raise ValidationError("Q-RAG reader profile does not match this release")
        _known(config.parameters, {"controls", "acknowledgements"}, "batch preparation parameters")
        self.longbench_context = None
        if "longbench_context" in reader:
            projection = _object(reader["longbench_context"], "Q-RAG longbench_context")
            if projection != {"kind": "lossless_words350_bm25_top6.v1"}:
                raise ValidationError("Q-RAG longbench_context requires kind=lossless_words350_bm25_top6.v1")
            if reader["profile"] != "factorial_instruction_preserving_reader_transfer":
                raise ValidationError("LongBench context projection requires the instruction-preserving reader")
            if "candidate_source" in reader:
                raise ValidationError("choose LongBench visible context or external candidate_source, not both")
            self.longbench_context = freeze_mapping(projection)
        self.candidate_source = None
        if "candidate_source" in reader:
            source = _object(reader["candidate_source"], "Q-RAG candidate source")
            _known(source, {"kind", "top_k", "max_document_chars", "document_format"}, "Q-RAG candidate source")
            if (source.get("kind") != "retriever"
                    or any(type(source.get(k)) is not int or source[k] < 1
                           for k in ("top_k", "max_document_chars"))):
                raise ValidationError("Q-RAG candidate_source requires kind=retriever and positive integer top_k/max_document_chars")
            if source.get("document_format", "text") not in ("text", "title_text"):
                raise ValidationError("Q-RAG candidate_source.document_format must be text or title_text")
            if config.retriever.get("kind") == "visible_context_bm25":
                raise ValidationError("Q-RAG candidate retrieval requires a corpus, not the empty visible context")
            self.candidate_source = freeze_mapping(source)
        if (config.artifacts or (config.retriever and self.candidate_source is None)
                or config.vector_runtime or "prepared_state_sha256" in config.runtime):
            raise ValidationError("Q-RAG batch requires an unbound visible-context template")
        self.replays = None
        if "policy_replays" in reader:
            refs = reader["policy_replays"]
            if (not isinstance(refs, (tuple, list)) or not refs
                    or "policy_replay_path" in config.runtime):
                raise ValidationError("select one non-empty policy replay list, not a second runtime replay")
            self.replays = {}
            for reference in refs:
                ref = _object(reference, "policy replay reference")
                _known(ref, {"path", "sha256"}, "policy replay reference")
                if not isinstance(ref.get("path"), str) or not ref["path"].strip():
                    raise ValidationError("policy replay reference requires a path")
                saved = load_pinned_json_mapping(ref["path"], expected_sha256=ref.get("sha256"))
                if saved.get("schema") != "agentinputlab.qrag.policy-replay.v1" or not isinstance(saved.get("request"), Mapping):
                    raise ValidationError("invalid Q-RAG policy replay schema/request")
                key = content_hash(saved["request"])
                if key in self.replays:
                    raise ValidationError("ambiguous duplicate Q-RAG replay request")
                self.replays[key] = ref
        self.base = replace(config, parameters={**config.parameters, "artifacts": _read_batch_assets(config, reader)},
            runtime={**config.runtime, "prepared_state_sha256": content_hash({"unbound_batch_template": True})})
        self.base.validate()
        method = self.base.create_intervention()
        method.acknowledgements.require_complete()
        method._validate_reader_profile()
        if self.candidate_source and self.candidate_source["top_k"] > method.controls.max_candidates:
            raise ValidationError("Q-RAG candidate_source.top_k exceeds controls.max_candidates")
        if self.longbench_context and method.controls.max_candidates < 6:
            raise ValidationError("LongBench top6 projection requires controls.max_candidates >= 6")

    def prepare_candidates(self, state, retriever):
        """Explicit visible-query retrieval; never inspect labels or replace passages."""
        from agentinputlab.contracts import ContextBlock

        if self.candidate_source is None:
            return state, None
        if state.context:
            raise ValidationError("Q-RAG candidate_source cannot replace existing visible context")
        if retriever is None:
            raise ValidationError("Q-RAG candidate_source requires a configured corpus retriever")
        results = tuple(retriever.retrieve(state.query, top_k=self.candidate_source["top_k"]))
        limit = self.candidate_source["max_document_chars"]
        if self.candidate_source.get("document_format", "text") == "title_text":
            from agentinputlab.algorithms.acquire.qrag.common_support import _policy_candidate_text

            texts = tuple(_policy_candidate_text(r) for r in results)
        else:
            texts = tuple(r.document.text for r in results)
        blocks = tuple(ContextBlock(block_id=r.document.doc_id, text=text[:limit],
                                    source=r.document.corpus_revision, metadata=r.document.metadata)
                       for r, text in zip(results, texts, strict=True))
        if not blocks or any(not b.text.strip() for b in blocks):
            raise ValidationError("Q-RAG candidate retrieval returned no usable passages")
        prepared = replace(state, context=blocks)
        return prepared, {"mode": "retrieved_visible_candidates.v1",
                          "configuration": to_jsonable(self.candidate_source),
                          "source_state_sha256": state.state_hash,
                          "prepared_state_sha256": prepared.state_hash,
                          "retriever": to_jsonable(retriever.fingerprint()),
                          "candidate_ids": [b.block_id for b in blocks],
                          "candidate_text_sha256": content_hash([b.text for b in blocks]),
                          "clipped_documents": sum(len(text) > limit for text in texts)}

    def bind(self, state, *, budget=None):
        from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
            QRAG_TASK_READER_PROFILES,
        )

        execution_budget = self.base.budget if budget is None else budget
        state = replace(state, budget=execution_budget)
        identity = dict(self.base.parameters["artifacts"])
        instruction = identity["reader_profile"] in QRAG_TASK_READER_PROFILES
        if (not state.query.strip() or state.messages or (state.instruction.strip() and not instruction)
                or (instruction and state.tool_names)):
            raise ValidationError("visible state is not supported by the explicit Q-RAG reader profile")
        candidates = [{"document_id": b.block_id, "text": b.text} for b in state.context]
        identity.update(corpus_sha256=content_hash(state.context),
            index_sha256=content_hash([b.block_id for b in state.context]),
            retriever_deployment_fingerprint_sha256=content_hash({"given_candidates": candidates}))
        runtime = {**self.base.runtime, "prepared_state_sha256": state.state_hash}
        request = {"question": state.query, "candidates": candidates, "seed": self.base.seed, "max_steps": 2}
        if self.replays is not None:
            ref = self.replays.get(content_hash(request))
            if ref is None:
                raise ValidationError("no exact policy replay for this Q-RAG case; live fallback is disabled")
            runtime.update(policy_replay_path=ref["path"], policy_replay_sha256=ref["sha256"])
        if "policy_replay_path" in runtime:
            load_policy_replay(runtime, identity, request)
        return replace(self.base, budget=execution_budget,
                       parameters={**self.base.parameters, "artifacts": identity}, runtime=runtime)


class FileQRAGRuntime:
    """One isolated, no-retry native invocation; no arbitrary factories or commands."""

    worker_id = "agentinputlab.qrag.configured-frozen-composite.v1"

    def __init__(self, config, backend):
        validate_runtime(config)
        self.config, self.backend = config, backend
        interpreter = Path(config.runtime["python_executable"])
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValidationError("Q-RAG worker Python is missing; install agentinputlab[qrag] in that environment")
        self.last_resource_receipt = None
        self.last_model_responses = []

    def fingerprint(self):
        return {"worker_id": self.worker_id, "release_id": self.config.release_id,
                "configuration_sha256": content_hash(self.config.to_dict()),
                "reader": to_jsonable(self.backend.fingerprint()),
                "policy_execution": ("explicit_saved_trajectory" if "policy_replay_path" in self.config.runtime
                                     else "live_frozen_policy"),
                "policy_replay_sha256": self.config.runtime.get("policy_replay_sha256"),
                "replay_usage_is_original_logical_cost": "policy_replay_path" in self.config.runtime}

    def call(self, request):
        from agentinputlab.benchmark_configuration import _CapturedBackend
        from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker

        artifacts = request.get("payload", {}).get("artifacts", {})
        if artifacts.get("reader_deployment_fingerprint_sha256") != content_hash(self.backend.fingerprint()):
            raise ValidationError("Q-RAG reader deployment does not match the selected artifact")
        value = self.config.runtime
        self.last_model_responses = []
        self.last_resource_receipt = None
        if ("prepared_state_sha256" in value
                and request.get("state_hash") != value["prepared_state_sha256"]):
            return {"status": "error", "failure_code": "QRAG_PREPARED_STATE_MISMATCH",
                    "message": "prepare this visible state before execution", "consumed": False}
        if "policy_replay_path" in value:
            payload = request["payload"]
            try:
                load_policy_replay(value, artifacts, {"question": payload["question"],
                    "candidates": payload["candidates"], "seed": payload["seed"], "max_steps": 2})
            except (AgentInputLabError, OSError) as exc:
                return {"status": "error", "failure_code": "QRAG_POLICY_REPLAY_INVALID",
                        "message": type(exc).__name__, "consumed": False}
        with tempfile.TemporaryDirectory(prefix="agentinputlab-qrag-") as home:
            environment = {"HOME": home, "HF_HOME": home, "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHON_DOTENV_DISABLED": "1", "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            if value.get("device", "cpu") == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            key = self.config.backend.get("api_key_env", "OPENAI_API_KEY")
            worker = JsonLineSubprocessWorker(
                command=(value["python_executable"], "-I", "-B", "-m", "workers.qrag.configured"),
                worker_id=self.worker_id, identity=self.fingerprint(), working_directory=home,
                timeout_seconds=value.get("timeout_seconds", 600), environment=environment,
                pass_environment=(key,) if os.environ.get(key) else (),
                max_output_bytes=16 * 1024 * 1024)
            try:
                result = worker.call({"configuration": self.config.to_dict(), "request": request,
                                      "reader_fingerprint": to_jsonable(self.backend.fingerprint())})
            finally:
                self.last_resource_receipt = worker.last_resource_receipt
        self.last_model_responses = result.get("model_responses", [])
        if isinstance(self.backend, _CapturedBackend):
            self.backend.calls.extend(self.last_model_responses)
        return result["response"]
