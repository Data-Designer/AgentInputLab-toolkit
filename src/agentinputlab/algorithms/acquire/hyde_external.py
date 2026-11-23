"""Thin native-source HyDE wrapper; source, clients and weights are not vendored."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.backends.vector import DenseEmbedder, VectorSearcher
from agentinputlab.contracts import (
    ArtifactError,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.integrations.full_architecture.hyde import (
    HYDE_PATCH_PROTOCOL,
    HYDE_UPSTREAM_REVISION,
    HyDEPatchedExecutionRequest,
)

from .hyde import _context_from_result, _validate_results, _validate_vectors

HYDE_EXTERNAL_RELEASE="acquire.hyde.official_external_patched"
_TASKS={"web search","scifact","arguana","trec-covid","fiqa","dbpedia-entity","trec-news","mr-tydi"}


@dataclass(frozen=True,slots=True)
class HyDENativeConfig:
    task: str="web search"
    language: str="en"
    num_hypotheses: int=8
    max_tokens: int=512
    temperature: float=0.7
    top_k: int=10
    stop: tuple[str,...]=("\n\n\n",)

    def __post_init__(self):
        if not isinstance(self.task,str) or self.task not in _TASKS or not isinstance(self.language,str) or not self.language.strip():
            raise ValidationError("HyDE native promptor task/language is invalid")
        for value in (self.num_hypotheses,self.max_tokens,self.top_k):
            if type(value) is not int or value<1:
                raise ValidationError("HyDE draw count/output cap/top_k must be positive integers")
        if isinstance(self.stop,(str,bytes)):
            raise ValidationError("HyDE stop must be a sequence of stop strings")
        object.__setattr__(self,"stop",tuple(self.stop))
        GenerationRequest(prompt="configuration",max_tokens=self.max_tokens,temperature=self.temperature,stop=self.stop)


def _identities(context):
    values={}
    for name,obj in (("generator",context.backend),("embedder",context.dense_embedder),
                     ("searcher",context.vector_searcher),("native_runtime",context.external_runtime)):
        if obj is None or not callable(getattr(obj,"fingerprint",None)):
            raise ExternalDependencyError(f"HyDE external profile requires context {name}")
        raw=obj.fingerprint()
        if not isinstance(raw,Mapping) or not raw:
            raise ArtifactError("HyDE external dependency fingerprint must be nonempty")
        try:
            values[name]=freeze_mapping(raw)
            content_hash(values[name])
        except (TypeError,ValueError) as exc:
            raise ArtifactError("HyDE external dependency fingerprint must be canonical finite data") from exc
    return values


@dataclass(frozen=True,slots=True)
class HyDEOfficialExternalPatched:
    config: HyDENativeConfig | Mapping[str,Any]=HyDENativeConfig()
    method_id: str="hyde"
    release_id: str=HYDE_EXTERNAL_RELEASE
    family: SemanticFamily=SemanticFamily.ACQUIRE

    def __post_init__(self):
        if isinstance(self.config,Mapping):
            try:
                object.__setattr__(self,"config",HyDENativeConfig(**self.config))
            except TypeError as exc:
                raise ValidationError("HyDE external configuration has unknown fields") from exc
        if not isinstance(self.config,HyDENativeConfig):
            raise ValidationError("HyDE external profile requires HyDENativeConfig")
        if self.method_id!="hyde" or self.release_id!=HYDE_EXTERNAL_RELEASE or self.family is not SemanticFamily.ACQUIRE:
            raise ValidationError("HyDE external profile identity differs from registry")

    def propose(self,state:FrozenAgentState,context:InterventionContext)->InputPatch:
        if not state.query.strip():
            raise ValidationError("HyDE requires a nonempty source query")
        if not isinstance(context.backend,LLMBackend) or not {ModelCapability.CHAT,ModelCapability.COMPLETION}.intersection(context.backend.capabilities):
            raise ExternalDependencyError("HyDE requires a chat or completion backend")
        if not isinstance(context.dense_embedder,DenseEmbedder) or not isinstance(context.vector_searcher,VectorSearcher):
            raise ExternalDependencyError("HyDE requires a dense embedder and corpus vector searcher")
        identities=_identities(context)
        native=identities["native_runtime"]
        if native.get("binding_mode")!=HYDE_PATCH_PROTOCOL or native.get("upstream_revision")!=HYDE_UPSTREAM_REVISION:
            raise ArtifactError("HyDE external runtime does not match reviewed source/patch protocol")
        runtime=context.external_runtime
        if not callable(getattr(runtime,"run",None)):
            raise ExternalDependencyError("HyDE external runtime requires run()")
        result=runtime.run(HyDEPatchedExecutionRequest(state.query,self.config,context.seed,state.budget,
            context.backend,context.dense_embedder,context.vector_searcher))
        if not isinstance(result,Mapping) or result.get("identity")!=native:
            raise ArtifactError("HyDE runtime response identity differs")
        if content_hash(_identities(context))!=content_hash(identities):
            raise ArtifactError("HyDE dependency identities changed during execution")
        hypotheses=result["hypotheses"]
        if len(hypotheses)!=self.config.num_hypotheses or any(not isinstance(v,str) for v in hypotheses):
            raise ArtifactError("HyDE runtime returned a different hypothesis inventory")
        vectors=_validate_vectors(result["encoded_vectors"],expected_count=self.config.num_hypotheses+1)
        mean=_validate_vectors((result["mean_vector"],),expected_count=1)[0]
        expected=tuple(sum(v[i] for v in vectors)/len(vectors) for i in range(len(vectors[0])))
        if len(mean)!=len(expected) or any(abs(a-b)>1e-10*max(1,abs(a),abs(b)) for a,b in zip(mean,expected,strict=True)):
            raise ArtifactError("native patched HyDE did not average query plus every hypothesis")
        hits=_validate_results(result["results"],top_k=self.config.top_k)
        if not hits:
            raise ArtifactError("native patched HyDE returned no real corpus hits")
        return InputPatch(method_id=self.method_id,release_id=self.release_id,source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.ADD_CONTEXT,tuple(_context_from_result(r) for r in hits),
                target="retrieved_evidence",rationale="real corpus evidence from patched native HyDE"),),
            trace=ExecutionTrace(method_id=self.method_id,release_id=self.release_id,
                events=(TraceEvent(TracePhase.PREPARE,"external_source_patch",native),
                    TraceEvent(TracePhase.PROPOSE,"native_hypotheses",{"documents":hypotheses,
                        "prompt":result["native_prompt"],"draws":result["draws"],"empty_indices":tuple(i for i,h in enumerate(hypotheses) if not h.strip())}),
                    TraceEvent(TracePhase.OBSERVE,"patched_native_mean",{"mean_vector":mean,"dimension":len(mean),"vector_count":len(vectors)}),
                    TraceEvent(TracePhase.OBSERVE,"native_runtime_resources",{
                        "wall_seconds":result["wall_seconds"],"cpu_seconds":result["calling_thread_cpu_seconds"],
                        "cpu_measurement_is_diagnostic_not_additional_usage":True,
                        "cpu_scope":result["usage_scope"],"process_cpu_seconds_window":result["process_cpu_seconds_window"],
                        "process_cpu_windows_not_additive":True}),
                    TraceEvent(TracePhase.FINALIZE,"real_corpus_hits",{"document_ids":tuple(r.document.doc_id for r in hits)})),
                counters={name:result[name] for name in ("model_calls","embedding_calls","retrieval_calls","input_tokens","output_tokens")}),
            implementation_identity=f"external_hyde@{HYDE_UPSTREAM_REVISION}:flatten_singleton_patch",
            profile_kind="official_native",metadata={"runtime":native,"dependency_fingerprints":identities,
                "source_algorithm_executed":True,"native_source_entrypoint":"HyDE.e2e_search",
                "hypotheses_are_evidence":False,"native_empty_hypotheses_retained":True,
                "original_generator_client_used":False,"generation_transport":"serial_observed_no_logprob_reordering",
                "original_batch_sampling_or_paper_result_parity_claimed":False,"source_redistributed":False})


__all__=["HYDE_EXTERNAL_RELEASE","HyDENativeConfig","HyDEOfficialExternalPatched"]
