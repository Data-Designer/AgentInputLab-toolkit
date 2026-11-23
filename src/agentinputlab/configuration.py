"""Editable, strict run configuration; runtime dependencies remain explicit.

Parsing and inspection never call a model or download assets. This is a local
Python/CLI interface, not a client-controlled HTTP execution configuration.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from agentinputlab.contracts import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable

_MAX_CONFIG_BYTES = 1024 * 1024


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise ValidationError(f"{label} must be an object with string keys")
    return dict(value)


def _known(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValidationError(f"unknown {label} fields: {', '.join(sorted(unknown))}")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate configuration field: {key}")
        result[key] = value
    return result


def _json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    value = _object(value, label)
    try:
        # Also rejects NaN/Infinity introduced through the Python API or 1e999.
        json.dumps(to_jsonable(value), allow_nan=False)
        return freeze_mapping(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValidationError(f"{label} must contain finite JSON-compatible values") from None


def _hints(factory: Any) -> dict[str, Any]:
    target = factory if is_dataclass(factory) else getattr(factory, "__init__", factory)
    try:
        return get_type_hints(target)
    except (NameError, TypeError):
        return {}


def _typed_parameter(value: Any, annotation: Any, label: str) -> Any:
    """Convert only explicitly annotated dataclasses, never arbitrary imports."""
    if get_origin(annotation) in (Union, UnionType):
        arguments = get_args(annotation)
        if len(arguments) == 2 and type(None) in arguments:
            annotation = next(arg for arg in arguments if arg is not type(None))
    origin, arguments = get_origin(annotation), get_args(annotation)
    if (origin in (list, tuple, Sequence) and arguments
            and (len(arguments) == 1 or (origin is tuple and arguments[1:] == (Ellipsis,)))
            and inspect.isclass(arguments[0]) and is_dataclass(arguments[0])):
        if not isinstance(value, (list, tuple)):
            raise ValidationError(f"{label} must be an array of objects")
        converted = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValidationError(f"{label}[{index}] must be an object")
            converted.append(_typed_parameter(item, arguments[0], f"{label}[{index}]"))
        return converted if origin is list else tuple(converted)
    if inspect.isclass(annotation) and is_dataclass(annotation) and isinstance(value, Mapping):
        allowed = {f.name for f in fields(annotation) if f.init}
        _known(value, allowed, label)
        hints = _hints(annotation)
        try:
            return annotation(**{
                key: _typed_parameter(item, hints.get(key), f"{label}.{key}")
                for key, item in value.items()
            })
        except TypeError:
            raise ValidationError(f"invalid or missing fields in {label}") from None
    return value


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One editable run specification with no implicit scientific overrides.

    ``parameters`` uses the selected method's native constructor (including
    nested ``config`` dataclasses). There is deliberately no universal
    ``max_length`` alias: model output, input window and compressor length are
    different quantities. Inject native runtimes through ``run(..., context=)``.
    """

    release_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    seed: int = 17
    backend: Mapping[str, Any] = field(default_factory=dict)
    retriever: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    vector_runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or not self.release_id.strip():
            raise ValidationError("release_id must be a non-empty string")
        if not isinstance(self.budget, ExecutionBudget):
            raise ValidationError("budget must be an ExecutionBudget")
        InterventionContext(seed=self.seed)
        for name in ("parameters", "backend", "retriever", "artifacts", "runtime", "vector_runtime"):
            object.__setattr__(self, name, _json_mapping(getattr(self, name), name))
        if {"release_id", "method_id", "family"}.intersection(self.parameters):
            raise ValidationError("parameters must not override registered method identity")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunConfig:
        data = _object(value, "configuration")
        _known(data, {"release_id", "parameters", "budget", "seed", "backend", "retriever",
                      "artifacts", "runtime", "vector_runtime"},
               "configuration")
        budget = _object(data.pop("budget", {}), "budget")
        _known(budget, {f.name for f in fields(ExecutionBudget)}, "budget")
        try:
            return cls(**data, budget=ExecutionBudget(**budget))
        except TypeError:
            raise ValidationError("configuration requires release_id and valid fields") from None

    @classmethod
    def from_file(cls, path: str | Path) -> RunConfig:
        try:
            with Path(path).open("rb") as stream:
                raw = stream.read(_MAX_CONFIG_BYTES + 1)
            if len(raw) > _MAX_CONFIG_BYTES:
                raise ValidationError("configuration exceeds the 1 MiB limit")
            value = json.loads(raw, object_pairs_hook=_unique)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValidationError("cannot read configuration: use an accessible UTF-8 JSON file") from None
        config = cls.from_dict(value)
        return config.resolve_paths(Path(path).absolute().parent)

    def resolve_paths(self, base: str | Path) -> RunConfig:
        """Resolve existing path fields relative to a containing configuration.

        This is shared with workflow files; it performs no asset loading.
        """
        config = self
        # Relative assets follow the config file, not the caller's working directory.
        base = Path(base).absolute()
        artifacts = {}
        for name, reference in config.artifacts.items():
            reference = _object(reference, f"artifacts.{name}")
            if isinstance(reference.get("path"), str) and not Path(reference["path"]).is_absolute():
                reference["path"] = str(base / reference["path"])
            artifacts[name] = reference
        runtime = dict(config.runtime)
        for key in ("source_root", "checkout", "checkpoint_root", "embedder_root",
                    "compatibility_manifest_path", "artifact_root", "base_model_root",
                    "feature_model_root", "checkpoint_manifest_path", "driver_path",
                    "python_executable", "prompt_bundle", "nltk_data", "tokenizer_root", "tokenizer_manifest",
                    "snapshot", "policy_replay_path", "response_path", "router_snapshot_path", "router_weights_path",
                    "router_tokenizer_path", "prompt_root", "deployment_path",
                    "rewrite_prompt_path", "reader_prompt_path", "rewriter_replay_path"):
            if isinstance(runtime.get(key), str) and not Path(runtime[key]).is_absolute():
                runtime[key] = str(base / runtime[key])
        retriever = dict(config.retriever)
        if retriever.get("kind") == "sqlite_fts5":
            value = retriever.get("index_path")
            if isinstance(value, str) and not Path(value).expanduser().is_absolute():
                retriever["index_path"] = str(base / value)
        if retriever.get("kind") == "beir_archive":
            for key in ("source_archive", "cache_dir", "index_path"):
                if isinstance(retriever.get(key), str) and not Path(retriever[key]).is_absolute():
                    retriever[key] = str(base / retriever[key])
        vector_runtime = dict(config.vector_runtime)
        for corpus in (retriever, vector_runtime):
            if corpus.get("kind") != "prepared_corpus":
                continue
            def relative(value):
                return str(base / value) if isinstance(value, str) and not Path(value).is_absolute() else value

            if "corpus_path" in corpus:
                corpus["corpus_path"] = relative(corpus["corpus_path"])
            if isinstance(corpus.get("options"), Mapping):
                options = dict(corpus["options"])
                if "model_path" in options:
                    options["model_path"] = relative(options["model_path"])
                corpus["options"] = options
            if isinstance(corpus.get("artifact_files"), (tuple, list)):
                corpus["artifact_files"] = [
                    {**item, "path": relative(item["path"])}
                    if isinstance(item, Mapping) and "path" in item else item
                    for item in corpus["artifact_files"]]
        parameters = dict(config.parameters)
        if config.release_id in {"acquire.ircot.native_like_substituted",
                                 "acquire.ircot.prediction_replay"}:
            manifest = parameters.get("manifest_path")
            if isinstance(manifest, str) and not Path(manifest).is_absolute():
                parameters["manifest_path"] = str(base / manifest)
        return replace(config, artifacts=artifacts, runtime=runtime, retriever=retriever,
                       vector_runtime=vector_runtime, parameters=parameters)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def _parameters(self, *, load_artifacts: bool = False) -> dict[str, Any]:
        from agentinputlab.configuration_assets import artifact_parameters
        from agentinputlab.registry import AlgorithmRegistry
        from agentinputlab.selectors import NoOpIntervention

        factory = (NoOpIntervention if self.release_id == "baseline.no_op"
                   else AlgorithmRegistry.default().resolve_entrypoint(self.release_id))
        signature = inspect.signature(factory)
        values = dict(self.parameters)
        # Even a factory accepting **kwargs cannot silently swallow typos here.
        allowed = {name for name, p in signature.parameters.items()
                   if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
        _known(values, allowed, "parameters")
        hints = _hints(factory)
        values = {k: _typed_parameter(v, hints.get(k), f"parameters.{k}")
                  for k, v in values.items()}
        overlap = values.keys() & self.artifacts.keys()
        if overlap:
            raise ValidationError(f"parameters and artifacts configure the same field: {', '.join(sorted(overlap))}")
        values.update(artifact_parameters(self.release_id, self.artifacts, load=load_artifacts))
        if (self.release_id == "acquire.adaptive_rag.selector_replay" and "records" in self.artifacts
                and values.get("artifact_sha256") != self.artifacts["records"]["sha256"]):
            raise ValidationError("Adaptive-RAG parameters.artifact_sha256 must match artifacts.records.sha256")
        if (self.release_id == "rewrite.query2doc.official_cached" and "cache" in self.artifacts
                and values.get("split") != self.artifacts["cache"]["split"]):
            raise ValidationError("Query2doc parameters.split must match artifacts.cache.split")
        bound = dict(values)
        if "release_id" in signature.parameters:
            bound["release_id"] = self.release_id
        try:
            signature.bind(**bound)
        except TypeError as exc:
            raise ValidationError(
                f"invalid parameters for {self.release_id}: {exc}; "
                f"inspect with agentinputlab show {self.release_id} --parameters"
            ) from None
        return values

    def _backend(self) -> Any:
        if not self.backend:
            return None
        from agentinputlab.backends.llm import (
            FinalContentStopBackend,
            ModelCapability,
            OpenAICompatibleBackend,
        )

        allowed = {"kind", "model_id", "base_url", "deployment_identity", "api_key_env",
                   "timeout_seconds", "extra_body", "stop_policy", "capabilities"}
        allowed.update({"response_model_aliases", "max_output_tokens_per_call", "stage_output_caps"})
        allowed.add("structured_output_transport")
        _known(self.backend, allowed, "backend (use api_key_env, not a plaintext key)")
        if not isinstance(self.backend.get("api_key_env", "OPENAI_API_KEY"), str):
            raise ValidationError("backend.api_key_env must be an environment variable name")
        if not isinstance(self.backend.get("deployment_identity"), Mapping):
            raise ValidationError("backend.deployment_identity requires an object with revision")
        if not isinstance(self.backend.get("extra_body", {}), Mapping):
            raise ValidationError("backend.extra_body must be an object")
        options = dict(self.backend)
        per_call = options.pop("max_output_tokens_per_call", None)
        stages = options.pop("stage_output_caps", {})
        if not isinstance(stages, Mapping):
            raise ValidationError("backend.stage_output_caps must be an object")
        if stages and per_call is None:
            raise ValidationError("backend.stage_output_caps requires max_output_tokens_per_call")
        if "capabilities" in options:
            raw = options["capabilities"]
            if (not isinstance(raw, (list, tuple)) or not raw
                    or any(not isinstance(value, str) for value in raw)
                    or len(raw) != len(set(raw))):
                raise ValidationError("backend.capabilities must be a non-empty unique string array")
            try:
                options["capabilities"] = frozenset(ModelCapability(value) for value in raw)
            except ValueError:
                raise ValidationError("backend.capabilities contains an unknown capability") from None
        kind = options.pop("kind", "openai_compatible")
        if not isinstance(kind, str) or kind not in {"openai_compatible", "decomp_completion", "udcs_final_completion"}:
            raise ValidationError("backend.kind must be openai_compatible, decomp_completion or udcs_final_completion")
        if per_call is not None and kind != "openai_compatible":
            raise ValidationError("native completion profiles configure per-call limits in their method parameters, not a chat-only output-cap wrapper")
        if "structured_output_transport" in options and kind != "openai_compatible":
            raise ValidationError("structured_output_transport is supported only by the OpenAI-compatible backend")
        stop_policy = options.pop("stop_policy", "provider")
        if not isinstance(stop_policy, str) or stop_policy not in {"provider", "final_content"}:
            raise ValidationError("backend.stop_policy must be provider or final_content")
        backend_type = OpenAICompatibleBackend
        if kind == "udcs_final_completion":
            from agentinputlab.backends.llm.udcs_completion import (
                RELEASES,
                UDCSFinalCompletionBackend,
            )

            if self.release_id not in RELEASES or stop_policy != "provider":
                raise ValidationError("udcs_final_completion is supported only by native UDCS profiles without stop wrappers")
            backend_type = UDCSFinalCompletionBackend
        if kind == "decomp_completion":
            from agentinputlab.backends.llm.decomp_completion import DecompCompletionBackend

            if self.release_id not in {"plan.decomp.full", "plan.decomp.planner_component"}:
                raise ValidationError("decomp_completion is supported only by native DecomP profiles")
            if stop_policy != "provider":
                raise ValidationError("DecomP requires unchanged native provider stop parameters")
            backend_type = DecompCompletionBackend
        try:
            backend = backend_type(**options)
        except TypeError:
            raise ValidationError("backend requires model_id, base_url and deployment_identity") from None
        if stop_policy == "final_content":
            backend = FinalContentStopBackend(backend)
        if per_call is not None:
            from agentinputlab.backends.llm import PerCallOutputCapBackend
            backend = PerCallOutputCapBackend(backend, per_call, stage_output_caps=stages)
        return backend

    def _retriever(self, *, load: bool = True) -> Any:
        if not self.retriever:
            return None
        if self.retriever.get("kind") == "sqlite_fts5":
            from agentinputlab.sqlite_retrieval_configuration import build

            return build(self.retriever, load=load)
        if self.retriever.get("kind") == "visible_context_bm25":
            from agentinputlab.visible_retrieval_configuration import VisibleContextBM25

            return VisibleContextBM25(self.retriever)
        if self.retriever.get("kind") == "beir_archive":
            from agentinputlab.beir_configuration import build

            return build(self.retriever, load=load)
        from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever

        if self.retriever.get("kind") == "prepared_corpus":
            from agentinputlab.experiments.production_builders import (
                PreparedCorpusRetrieverBuilder,
                ProductionRetrieverBuildRequest,
                _parse_corpus_retriever_parameters,
            )

            parameters = {key: value for key, value in self.retriever.items() if key != "kind"}
            if "schema_version" in parameters:
                raise ValidationError("retriever schema_version is managed by RunConfig")
            parameters["schema_version"] = 1
            # Parse only: validate must not read corpora or load an embedding model.
            _parse_corpus_retriever_parameters(parameters)
            if not load:
                return None
            return PreparedCorpusRetrieverBuilder().build(ProductionRetrieverBuildRequest(
                benchmark_id=parameters["benchmark_id"], model_slot="configured",
                seed=self.seed, builder_parameters=parameters))

        _known(self.retriever, {"kind", "documents", "k1", "b"}, "retriever")
        if self.retriever.get("kind") != "bm25":
            raise ValidationError(
                "file-config retriever.kind must be bm25, visible_context_bm25, prepared_corpus, sqlite_fts5 or beir_archive; inject custom retrievers "
                "through InterventionContext(retriever=...) in Python"
            )
        docs = self.retriever.get("documents")
        if not isinstance(docs, (tuple, list)) or not docs:
            raise ValidationError("retriever.documents must be a non-empty array")
        documents = []
        for i, item in enumerate(docs):
            data = _object(item, f"retriever.documents[{i}]")
            _known(data, {"doc_id", "text", "title", "corpus_revision", "metadata"},
                   f"retriever.documents[{i}]")
            for key in ("doc_id", "text"):
                if not isinstance(data.get(key), str) or not data[key].strip():
                    raise ValidationError(f"retriever.documents[{i}].{key} must be non-empty text")
            for key in ("title", "corpus_revision"):
                if key in data and not isinstance(data[key], str):
                    raise ValidationError(f"retriever.documents[{i}].{key} must be text")
            documents.append(Document(**data))
        if len({d.doc_id for d in documents}) != len(documents):
            raise ValidationError("retriever document IDs must be unique")
        # Inline documents are already fully specified here. Bind missing
        # provenance to their content rather than failing after a model call.
        # Never replace an explicitly supplied source revision.
        inline_revision = "inline-sha256:" + content_hash(documents)
        documents = [replace(d, corpus_revision=inline_revision)
                     if not d.corpus_revision.strip() else d for d in documents]
        return InMemoryBM25Retriever(documents, k1=self.retriever.get("k1", 1.2),
                                     b=self.retriever.get("b", 0.75))

    def _vector_runtime(self, *, load: bool = True) -> Any:
        if not self.vector_runtime:
            return None
        from agentinputlab.backends.retrieval.dense import retriever_profile
        from agentinputlab.backends.vector import DenseEmbedder, VectorSearcher

        if self.vector_runtime.get("kind") != "prepared_corpus":
            raise ValidationError("vector_runtime.kind must be prepared_corpus with a dense profile")
        # Use the same closed, checksum-verified corpus/model loader as retrieval.
        binding = replace(self, retriever=self.vector_runtime)
        binding._retriever(load=False)
        if retriever_profile(self.vector_runtime["profile_id"]).backend != "sentence_transformers_dense":
            raise ValidationError("vector_runtime requires a dense profile, not sparse retrieval")
        if not load:
            return None
        product = binding._retriever()
        if not isinstance(product, DenseEmbedder) or not isinstance(product, VectorSearcher):
            raise ValidationError("vector_runtime must implement DenseEmbedder and VectorSearcher")
        return product

    def validate(self) -> dict[str, Any]:
        """Validate configuration shape without constructing the method/runtime."""
        self._parameters()
        self._backend()
        self._retriever(load=False)
        self._vector_runtime(load=False)
        self._runtime(load=False)
        return {"valid": True, "scope": "configuration_only", "release_id": self.release_id,
                "model_calls": 0, "runtime_ready": None, "real_acceptance_passed": None,
                "note": "Native runtime/assets and method-specific execution constraints still apply."}

    def create_intervention(self) -> Any:
        from agentinputlab.api import create_intervention

        return create_intervention(self.release_id, **self._parameters(load_artifacts=True))

    def _runtime(self, *, load: bool, backend: Any = None) -> Any:
        if not self.runtime:
            return None
        kind = self.runtime.get("kind")
        if not isinstance(kind, str):
            raise ValidationError("runtime.kind must be a string")
        if kind == "qrag_common_support_replay":
            from agentinputlab.qrag_common_replay import runtime

            return runtime(self, load=load)
        if kind == "r2ag_common_support_replay":
            from agentinputlab.r2ag_common_replay import runtime

            return runtime(self, load=load)
        if kind == "rrr_checkpoint":
            from agentinputlab.rrr_configuration import validate_runtime

            validate_runtime(self)
            return None  # Bind the frozen rewriter, reader and visible corpus per case.
        if kind == "adaptive_rag_controlled":
            from agentinputlab.adaptive_configuration import validate_runtime

            validate_runtime(self)
            return None  # Bind to the current case's captured backend and retriever.
        if kind == "qrag_native":
            from agentinputlab.qrag_configuration import FileQRAGRuntime, validate_runtime

            validate_runtime(self)
            return FileQRAGRuntime(self, backend) if load else None
        if kind == "longllmlingua_native":
            from agentinputlab.longllmlingua_configuration import runtime

            return runtime(self, load=load)
        if kind == "recomp_abstractive_worker":
            from agentinputlab.recomp_configuration import runtime

            return runtime(self, load=load)
        if kind == "ircot_native":
            from agentinputlab.ircot_configuration import validate_runtime

            validate_runtime(self)
            return None  # Bound after the per-case response capture wrapper.
        if kind == "decomp_live":
            from agentinputlab.decomp_configuration import validate_runtime

            validate_runtime(self.release_id, self.runtime)
            if self.backend and self.backend.get("kind") != "decomp_completion":
                raise ValidationError("decomp_live requires backend.kind decomp_completion")
            return None  # Bind after the case-specific response capture proxy exists.
        if kind == "llmcompiler_live":
            from agentinputlab.llmcompiler_configuration import validate_runtime

            validate_runtime(self.release_id, self.runtime)
            # The native worker owns case-specific tools and must be bound only
            # after the visible state and response-capturing backend are known.
            return None
        supported = {"querygym_query2doc": ("rewrite.query2doc.querygym_live",
                                          "rewrite.query2doc.controlled_live"),
                     "qrag_common_support": ("acquire.qrag.common_support",),
                     "r2ag_common_support": ("acquire.r2ag.common_support",),
                     "recomp_extractive_worker": ("curate.recomp.extractive",
                                                  "curate.recomp.common_support_extractive"),
                     "ask_or_assume_worker": ("clarify.ask_or_assume.intent_monitor_component",),
                     "acon_observation_worker": ("curate.acon.common_support_observation",
                                                 "curate.acon.observation_runtime"),
                     "acon_history_worker": ("curate.acon.history_runtime",),
                     "reformer_worker": ("rewrite.reformer.repo_native_r_only",
                                         "rewrite.reformer.paper_spec_q_concat_r",
                                         "rewrite.reformer.controlled_r_only",
                                         "rewrite.reformer.controlled_q_concat_r"),
                     "infocqr_worker": ("rewrite.infocqr.common_support_rw_zsl",)}
        if kind in supported:
            if self.release_id not in supported[kind]:
                raise ValidationError(f"runtime.kind {kind} is supported only by {', '.join(supported[kind])}")
            from agentinputlab.experiments.production_builders import (
                _build_external_runtime,
                _parse_external_runtime_block,
            )

            # Both ACON protocols are implemented by the same pinned driver.
            # Reuse its existing transport validation, not a new implementation.
            transport = ({**self.runtime, "kind": "acon_observation_worker"}
                         if kind == "acon_history_worker" else self.runtime)
            parsed = _parse_external_runtime_block(transport)
            if not load:
                return None
            if kind == "querygym_query2doc" and backend is None:
                raise ValidationError("QueryGym runtime requires a configured backend or context.backend")
            if kind in {"ask_or_assume_worker", "acon_observation_worker", "acon_history_worker",
                        "reformer_worker", "infocqr_worker"} and backend is None:
                raise ValidationError(f"{kind} requires a configured backend or context.backend")
            runtime = _build_external_runtime(parsed, backend)
            if kind == "reformer_worker":
                # The same pinned driver validates all four native profiles.
                # Keep its receipt bound to the selected file profile as well.
                runtime = replace(runtime, identity={**runtime.identity, "release_id": self.release_id})
            if kind in {"acon_history_worker", "acon_observation_worker"} and self.release_id != "curate.acon.common_support_observation":
                runtime = replace(runtime,
                                  worker_id="acon-history" if kind == "acon_history_worker" else "acon-observation",
                                  identity={**runtime.identity, "release_id": self.release_id,
                                            "component_release_id": self.release_id})
            return runtime
        _known(self.runtime, {"kind", "source_root"}, "runtime")
        if (self.release_id != "plan.plan_and_budget.full"
                or self.runtime.get("kind") != "plan_and_budget_native"):
            raise ValidationError("runtime.kind plan_and_budget_native is supported only by plan.plan_and_budget.full; other runtimes use Python context injection")
        if not isinstance(self.runtime.get("source_root"), str) or not self.runtime["source_root"].strip():
            raise ValidationError("runtime.source_root must be a non-empty path to the pinned checkout")
        if not load:
            return None
        from agentinputlab.integrations.full_architecture.plan_and_budget import (
            PlanAndBudgetNativePromptRuntime,
        )
        return PlanAndBudgetNativePromptRuntime(source_root=self.runtime["source_root"])

    def _configured_context(self, context: InterventionContext | None) -> InterventionContext:
        self.validate()
        context = context or InterventionContext()
        if context.budget_ledger is not None or context.monitor is not None:
            raise ValidationError("RunConfig requires a fresh context, without a shared ledger/monitor")
        for name in ("backend", "retriever"):
            if getattr(self, name) and getattr(context, name) is not None:
                raise ValidationError(f"{name} is configured twice; choose file config or context")
        if self.runtime and context.external_runtime is not None:
            raise ValidationError("external runtime is configured twice; choose file config or context")
        if self.vector_runtime and (context.dense_embedder is not None or context.vector_searcher is not None):
            raise ValidationError("vector runtime is configured twice; choose file config or context")
        backend = self._backend() if self.backend else context.backend
        vector = self._vector_runtime() if self.vector_runtime else None
        context = replace(context, seed=self.seed,
                          backend=backend,
                          dense_embedder=vector if self.vector_runtime else context.dense_embedder,
                          vector_searcher=vector if self.vector_runtime else context.vector_searcher,
                          retriever=self._retriever() if self.retriever else context.retriever,
                          external_runtime=self._runtime(load=True, backend=backend) if self.runtime else context.external_runtime)
        return context

    def _bind_case_context(self, context: InterventionContext, state: FrozenAgentState) -> InterventionContext:
        if self.retriever.get("kind") == "visible_context_bm25":
            from agentinputlab.visible_retrieval_configuration import VisibleContextBM25

            context = replace(context, retriever=VisibleContextBM25(self.retriever).bind(state))
        if self.runtime.get("kind") == "rrr_checkpoint":
            from agentinputlab.rrr_configuration import bind_case

            return bind_case(self, context, state)
        if self.runtime.get("kind") == "adaptive_rag_controlled":
            from agentinputlab.adaptive_configuration import FileAdaptiveRuntime

            return replace(context, external_runtime=FileAdaptiveRuntime(self, context.backend, context.retriever))
        if self.runtime.get("kind") == "qrag_native":
            from agentinputlab.qrag_configuration import FileQRAGRuntime

            return replace(context, external_runtime=FileQRAGRuntime(self, context.backend))
        if self.runtime.get("kind") == "ircot_native":
            from agentinputlab.ircot_configuration import bind_case

            return bind_case(self, context, state)
        if self.runtime.get("kind") == "decomp_live":
            from agentinputlab.decomp_configuration import bind_case

            return bind_case(self, context, state)
        if self.runtime.get("kind") == "llmcompiler_live":
            from agentinputlab.llmcompiler_configuration import bind_case

            return bind_case(self, context, state)
        return context

    def evaluate(self, benchmark_id: str, **options: Any) -> dict[str, Any]:
        """Execute selected benchmark cases with native scoring and one total budget.

        See ``evaluate_config`` for dataset, predictor and explicit case selection
        options. Stateful arenas require their native execution bridge.
        """
        from agentinputlab.benchmark_configuration import evaluate_config

        return evaluate_config(self, benchmark_id, **options)

    def evaluate_arena(self, arena_config: Any, **options: Any) -> dict[str, Any]:
        """Run selected native arena cases in separate processes, without retries."""
        from agentinputlab.arena_configuration import NativeArenaConfig, evaluate_arena

        native = (arena_config if isinstance(arena_config, NativeArenaConfig)
                  else NativeArenaConfig.from_file(arena_config))
        return evaluate_arena(self, native, **options)

    def run(self, state: FrozenAgentState | str, *,
            context: InterventionContext | None = None) -> dict[str, Any]:
        """Run once, never retry; preserve patch versus composite semantics.

        This config's budget and seed are authoritative. A reused budget ledger
        is rejected. Asset-bearing profiles still need explicit dependencies.
        """
        from agentinputlab.api import run_composite_intervention, run_intervention

        context = self._configured_context(context)
        state = (FrozenAgentState(query=state, budget=self.budget) if isinstance(state, str)
                 else replace(state, budget=self.budget))
        context = self._bind_case_context(context, state)
        method = self.create_intervention()
        if callable(getattr(method, "propose", None)):
            result = run_intervention(method, state, context)
        elif callable(getattr(method, "run", None)):
            result = run_composite_intervention(method, state, context)
        else:
            raise ValidationError("this profile is a recipe/component; use its native fit/apply API")
        return {**result, "configuration": {"release_id": self.release_id,
                "seed": self.seed, "budget": to_jsonable(self.budget),
                "artifacts": to_jsonable(self.artifacts), "runtime": to_jsonable(self.runtime),
                "vector_runtime": to_jsonable(self.vector_runtime)}}


def load_run_state(path: str | Path) -> FrozenAgentState:
    """Load a complete visible state for the local CLI; never load gold implicitly."""
    from agentinputlab.api import _state

    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(_MAX_CONFIG_BYTES + 1)
        if len(raw) > _MAX_CONFIG_BYTES:
            raise ValidationError("state exceeds the 1 MiB limit")
        value = json.loads(raw, object_pairs_hook=_unique)
        value = to_jsonable(_json_mapping(value, "state"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValidationError("cannot read state: use an accessible UTF-8 JSON object") from None
    return _state(value)
