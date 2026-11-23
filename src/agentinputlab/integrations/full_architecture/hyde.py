"""Execute caller-mounted HyDE source under a checksum-locked shape patch.

The package contains no upstream source, native API client, model or index.
Only the original algorithm and promptor modules are loaded from the mount.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.backends.llm import GenerationRequest
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    ExternalDependencyError,
    Message,
    ValidationError,
)

HYDE_UPSTREAM_REVISION="a2fd8734307612cb0225d71ffbf26e0d225986b8"
HYDE_PATCH_PROTOCOL="agentinputlab.hyde.external-flatten-encoder.v1"
HYDE_SOURCE_SHA256={
    "src/hyde/hyde.py":"7a395213842b80052e2d47db45ddfff4ebf416d11197fdc9760997ecb2cf7041",
    "src/hyde/promptor.py":"675a9ccc9336122095b401b6d05dc314fd6a3fbb067f78e4e660024892d78d34",
}


def _sources(source_root):
    root=Path(source_root).expanduser().resolve()
    result={}
    for name,pin in HYDE_SOURCE_SHA256.items():
        try:
            path=root/name
            if path.stat().st_size>65536:
                raise ArtifactError("HyDE source exceeds the checked module size")
            raw=path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"HyDE external source unavailable: {name}") from exc
        if hashlib.sha256(raw).hexdigest()!=pin:
            raise ArtifactError(f"HyDE external source SHA-256 differs: {name}")
        result[name]=raw
    return result


def _patched_source(sources):
    raw=sources["src/hyde/hyde.py"]
    before=b"np.array(c_emb)"
    after=b"np.asarray(c_emb).reshape(-1)"
    if raw.count(before)!=1:
        raise ArtifactError("HyDE reviewed encoder-shape patch site differs")
    return raw.replace(before,after)


def _manifest(sources):
    return dict(schema_version=1,protocol=HYDE_PATCH_PROTOCOL,upstream_revision=HYDE_UPSTREAM_REVISION,
        source_sha256=dict(HYDE_SOURCE_SHA256),
        patch=dict(target="src/hyde/hyde.py:HyDE.encode",change="flatten_each_singleton_encoder_vector_before_mean",
            patched_source_sha256=hashlib.sha256(_patched_source(sources)).hexdigest()),
        native_client_module_imported=False,source_modified_on_disk=False,
        source_review_acknowledged=True,source_redistributed_by_toolkit=False,
        substitutions=["observed_serial_generation_draws","observed_dense_embedding","observed_corpus_vector_search"])


def write_hyde_patch_manifest(source_root,output,*,accept_source_review=False):
    """Create-only operator receipt. Source review is not a redistribution grant."""
    if accept_source_review is not True:
        raise ArtifactError("review the mounted HyDE source/terms and acknowledge source review explicitly")
    value=_manifest(_sources(source_root))
    path=Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True,slots=True)
class HyDEPatchedExecutionRequest:
    """In-process callbacks are the runner's already-instrumented context objects.

    This request is not an IPC recipe. Do not serialize or store its executable
    dependencies on an intervention. Native bridge usage is charged there once.
    """
    query: str
    config: Any
    seed: int
    budget: Any
    backend: Any
    embedder: Any
    searcher: Any


class HyDEPatchedSourceRuntime:
    worker_id="agentinputlab.hyde.pinned-native-source.v1"

    def __init__(self,*,source_root,patch_manifest,expected_manifest_sha256):
        self.source_root=Path(source_root).expanduser().resolve()
        self.patch_manifest=Path(patch_manifest).expanduser().resolve()
        self.expected_manifest_sha256=expected_manifest_sha256

    def _verified(self):
        sources=_sources(self.source_root)
        receipt=load_pinned_json_mapping(self.patch_manifest,expected_sha256=self.expected_manifest_sha256,max_bytes=65536)
        if dict(receipt)!=_manifest(sources):
            raise ArtifactError("HyDE patch manifest is not the exact reviewed patch/substitution set")
        try:
            import numpy
        except ImportError as exc:
            raise ExternalDependencyError("install agentinputlab[hyde-native] for the external source runtime") from exc
        return sources,numpy

    def fingerprint(self):
        sources,np=self._verified()
        return {"runtime_id":self.worker_id,"binding_mode":HYDE_PATCH_PROTOCOL,
            "upstream_revision":HYDE_UPSTREAM_REVISION,"source_sha256":dict(HYDE_SOURCE_SHA256),
            "patched_source_sha256":hashlib.sha256(_patched_source(sources)).hexdigest(),
            "patch_manifest_sha256":self.expected_manifest_sha256,"numpy_version":np.__version__,
            "native_generator_client_used":False,"source_redistributed":False,
            "generation_transport":"observed_serial_draws_no_logprob_reordering_no_retry"}

    def run(self,request):
        from agentinputlab.algorithms.acquire.hyde import _validate_vectors
        from agentinputlab.algorithms.acquire.hyde_external import HyDENativeConfig
        if not isinstance(request,HyDEPatchedExecutionRequest) or not isinstance(request.config,HyDENativeConfig):
            raise ValidationError("HyDE patched runtime requires its typed execution request/config")
        sources,np=self._verified()
        identity=self.fingerprint()
        native_namespace={}
        prompt_namespace={}
        exec(compile(_patched_source(sources),"externally-mounted-hyde-patched","exec"),native_namespace)
        exec(compile(sources["src/hyde/promptor.py"],"externally-mounted-hyde-promptor","exec"),prompt_namespace)
        config=request.config
        if (request.budget.max_model_calls<config.num_hypotheses or request.budget.max_retrieval_calls<1
                or request.budget.max_output_tokens<config.num_hypotheses):
            raise BudgetExceededError("HyDE cannot reserve all hypothesis draws and corpus search")
        outputs=inputs=0
        draws=[]
        encoded=[]
        mean_vectors=[]

        class GeneratorBridge:
            def generate(self,prompt):
                nonlocal outputs,inputs
                texts=[]
                for index in range(config.num_hypotheses):
                    remaining=request.budget.max_output_tokens-outputs
                    reserve=config.num_hypotheses-index-1
                    if remaining<=reserve:
                        raise BudgetExceededError("HyDE output budget cannot cover remaining hypothesis draws")
                    generation=GenerationRequest(messages=(Message("user",prompt),),temperature=config.temperature,
                        max_tokens=min(config.max_tokens,remaining-reserve),seed=request.seed+index,stop=config.stop,
                        metadata={"release_id":"acquire.hyde.official_external_patched","stage":"native_hypothesis",
                            "hypothesis_index":index,"native_promptor_task":config.task})
                    response=request.backend.generate(generation)
                    if not isinstance(response.text,str):
                        raise AlgorithmOutputError("HyDE generator returned non-text")
                    inputs+=response.usage.input_tokens
                    outputs+=response.usage.output_tokens
                    if response.usage.output_tokens>generation.max_tokens:
                        raise BudgetExceededError("HyDE provider exceeded requested hypothesis output cap")
                    texts.append(response.text)
                    draws.append(dict(text=response.text,request_hash=generation.request_hash,
                        max_tokens=generation.max_tokens,seed=generation.seed,finish_reason=response.finish_reason))
                return texts

        class EncoderBridge:
            def encode(self,text):
                vectors=_validate_vectors(request.embedder.embed((text,)),expected_count=1)
                encoded.append(vectors[0])
                # Exercise the actual batched-singleton interface requiring the
                # reviewed flatten patch; native code performs the mean itself.
                return np.asarray(vectors)

        class SearchBridge:
            def search(self,vector,k):
                if vector.ndim!=2 or vector.shape[0]!=1:
                    raise AlgorithmOutputError("patched native HyDE returned a non-singleton matrix")
                values=tuple(float(v) for v in vector[0])
                mean_vectors.append(values)
                return request.searcher.search(values,top_k=k)

        promptor=prompt_namespace["Promptor"](task=config.task,language=config.language)
        native=native_namespace["HyDE"](promptor,GeneratorBridge(),EncoderBridge(),SearchBridge())
        cpu=time.thread_time()
        process_cpu=time.process_time()
        started=time.monotonic()
        hits=native.e2e_search(request.query,k=config.top_k)
        cpu=time.thread_time()-cpu
        if self.fingerprint()!=identity:
            raise ArtifactError("HyDE mounted runtime identity changed during execution")
        return dict(identity=identity,results=hits,hypotheses=tuple(d["text"] for d in draws),draws=tuple(draws),
            mean_vector=mean_vectors[0],encoded_vectors=tuple(encoded),native_prompt=promptor.build_prompt(request.query),
            model_calls=len(draws),embedding_calls=len(encoded),retrieval_calls=1,
            input_tokens=inputs,output_tokens=outputs,wall_seconds=time.monotonic()-started,
            process_cpu_seconds_window=time.process_time()-process_cpu,
            process_cpu_window_is_not_additive_across_concurrent_requests=True,
            calling_thread_cpu_seconds=cpu,
            # Delegated calls/tokens are already ledger-observed. RunMonitor
            # also records its enclosing process CPU window at finish_run;
            # adding the nested thread measurement would count it twice.
            # Keep both measurements diagnostic, not additive usage charges.
            usage=dict(model_calls=0,retrieval_calls=0,input_tokens=0,output_tokens=0,cpu_seconds=0.0),
            usage_scope="calling_thread_cpu_lower_bound_diagnostic; outer monitor owns process CPU accounting; native-worker CPU excluded from thread measure")


__all__=["HYDE_PATCH_PROTOCOL","HYDE_SOURCE_SHA256","HYDE_UPSTREAM_REVISION",
    "HyDEPatchedExecutionRequest","HyDEPatchedSourceRuntime","write_hyde_patch_manifest"]
