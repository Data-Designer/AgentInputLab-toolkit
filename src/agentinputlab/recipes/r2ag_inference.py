"""Native R²AG generation from a reader-bound fitted artifact.

The fitted native RFormer, projection, dedicated marker row and tokenizer are
restored together. Retrieval features are explicit inputs: callers can reuse
source-bound preprocessing without buying unrecorded embedding calls.
"""

from __future__ import annotations

import gc
import inspect
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .qrag_inference import _stamps
from .r2ag_training import (
    MARKER,
    PROTOCOL,
    R2AG_UPSTREAM_COMMIT,
    SOURCE_FILES,
    R2AGTrainedArtifact,
    _file,
    _module,
    _native_formatter,
    _new_root,
    _parameter_hashes,
    _quantized_reader_dependencies,
    _reader_loader,
    _safe_model,
)
from .recomp_extractive import _LOCK, _save


def _read(path):
    return json.loads(Path(path).read_text())


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ExternalDependencyError(
            "R²AG trained inference requires pip install 'agentinputlab[r2ag-fit]'"
        ) from exc
    return torch


def _tensor_digest(torch, value):
    import hashlib

    return hashlib.sha256(
        value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _generation_signature_compat(model):
    """Expose an already-forwarded kwarg to GenerationMixin's signature check.

    Gemma4 Unified's native preparation method forwards **kwargs to the base
    method, but does not name inputs_embeds. No model math, input value, cache
    policy or generation setting is changed by this instance-local shim.
    """
    result = dict(adapter="native_unmodified", applied=False)
    if getattr(model.config, "model_type", None) != "gemma4_unified":
        return result
    original = model.prepare_inputs_for_generation
    if getattr(original, "_agentinputlab_r2ag_signature_compat", False):
        return dict(adapter="gemma4_unified_explicit_inputs_embeds_signature_v1", applied=True)
    params = inspect.signature(original).parameters
    if "inputs_embeds" in params:
        return result
    if (
        not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        or "inputs_embeds" not in inspect.signature(model.forward).parameters
        or not {"input_ids", "use_cache", "is_first_iteration"} <= params.keys()
    ):
        raise ExternalDependencyError("R²AG Gemma4 generation preparation contract differs")

    def prepare(
        input_ids, use_cache=True, is_first_iteration=False, *, inputs_embeds=None, **kwargs
    ):
        return original(
            input_ids,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    prepare._agentinputlab_r2ag_signature_compat = True
    model.prepare_inputs_for_generation = prepare
    return dict(adapter="gemma4_unified_explicit_inputs_embeds_signature_v1", applied=True)


def _count_generation_reader_request(observed, kwargs):
    observed["reader_forward_attempts"] += 1
    value = kwargs.get("input_ids")
    if value is None:
        value = kwargs.get("inputs_embeds")
    if value is None:
        observed["reader_input_token_unknown_attempts"] += 1
    else:
        observed["reader_input_tokens_attempted"] += int(value.shape[0] * value.shape[1])


def _load_components(torch, model, values):
    if not isinstance(values, Mapping) or set(values) != {"r_former", "llama_proj"}:
        raise ValidationError("R²AG native checkpoint component names differ")
    for name in ("r_former", "llama_proj"):
        module = getattr(model, name)
        state, expected = values[name], module.state_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValidationError(f"R²AG native state keys differ: {name}")
        for key, tensor in state.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected[key].shape
                or tensor.dtype != expected[key].dtype
                or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
            ):
                raise ValidationError(f"R²AG native tensor shape/dtype/value differs: {name}.{key}")
        module.load_state_dict(state, strict=True)


@dataclass(frozen=True, slots=True)
class R2AGReaderDocument:
    document_id: str
    title: str
    text: str
    features: tuple[float, float, float]

    def __post_init__(self):
        for name in ("document_id", "title", "text"):
            v = getattr(self, name)
            if not isinstance(v, str) or "\x00" in v or (name != "title" and not v.strip()):
                raise ValidationError(f"R²AG reader document {name} is invalid")
        if (
            not isinstance(self.features, Sequence)
            or len(self.features) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or abs(v) > 1.0001
                for v in self.features
            )
        ):
            raise ValidationError("R²AG reader requires three finite native cosine features")
        object.__setattr__(self, "features", tuple(float(v) for v in self.features))


class R2AGTrainedReader:
    """Load the matching frozen reader and native fitted retrieval modules."""

    def __init__(
        self,
        artifact,
        *,
        checkout,
        base_model_root,
        base_model_manifest,
        base_model_manifest_sha256,
        device="cpu",
        source_and_asset_acknowledged=False,
    ):
        from .r2ag_training import validate_r2ag_training_source

        if source_and_asset_acknowledged is not True:
            raise ValidationError("R²AG inference requires mounted-source/model acknowledgement")
        if not isinstance(artifact, R2AGTrainedArtifact) or device not in {"cpu", "cuda:0"}:
            raise ValidationError("R²AG inference requires a typed artifact and one logical device")
        verified = R2AGTrainedArtifact.load(artifact.root)
        if verified.artifact_id != artifact.artifact_id:
            raise ValidationError("R²AG trained artifact identity differs")
        self.source = validate_r2ag_training_source(checkout)
        self.base, _ = _safe_model(base_model_root, base_model_manifest, base_model_manifest_sha256)
        plan = _read(verified.root / "fit-plan.json")
        if (
            plan["protocol"] != PROTOCOL
            or plan["source_commit"] != R2AG_UPSTREAM_COMMIT
            or plan["source_files"] != SOURCE_FILES
            or plan["base_model_manifest_sha256"] != base_model_manifest_sha256
        ):
            raise ValidationError("R²AG fitted source or frozen reader binding differs")
        self.artifact, self.device, self._loaded = verified, device, None
        self._generation_transport = dict(adapter="not_loaded", applied=False)
        self._paths = [
            verified.root / "artifact.json",
            _file(base_model_manifest, "model manifest"),
        ]
        self._paths += [verified.root / n for n in _read(verified.root / "artifact.json")["files"]]
        self._paths += [self.source / n for n in SOURCE_FILES]
        self._paths += sorted(p for p in self.base.rglob("*") if p.is_file())
        self._identity = _stamps(self._paths)
        self._binding = dict(
            runtime="agentinputlab.r2ag.native_trained_reader.v1",
            trained_artifact_id=verified.artifact_id,
            source_commit=R2AG_UPSTREAM_COMMIT,
            base_model_manifest_sha256=base_model_manifest_sha256,
            safe_weights_only_load=True,
            marker_rows_restored=True,
            native_forward_and_generate=True,
            generation_transport_compatibility="gemma4_unified_explicit_inputs_embeds_signature_if_needed_v1",
            embedding_features_precomputed_and_cost_separate=True,
            optimizer_loaded=False,
            training_allowed=False,
            paper_result_parity=False,
        )
        self.deployment_id = content_hash(self._binding)
        self._parameters = self._parameter_binding()

    def _parameter_binding(self):
        return (
            self.source,
            self.base,
            self.artifact,
            self.device,
            self.deployment_id,
            content_hash(self._binding),
        )

    def _verify(self):
        if self._parameter_binding() != self._parameters or _stamps(self._paths) != self._identity:
            raise ValidationError("R²AG trained deployment files or parameters changed")

    def fingerprint(self):
        self._verify()
        return freeze_mapping(dict(self._binding, deployment_id=self.deployment_id))

    def _load(self):
        self._verify()
        if self._loaded is not None:
            return self._loaded
        torch = _torch()
        from transformers import AutoTokenizer

        dependencies = _quantized_reader_dependencies(self.base)
        recorded_dependencies = self.artifact.root / "runtime-dependencies.json"
        if recorded_dependencies.exists() and dependencies != _read(recorded_dependencies):
            raise ValidationError("R²AG native runtime kernel revision differs from training")
        native = _module(
            self.source / "RRAG/models/modeling_rrag.py", "agentinputlab_r2ag_native_inference"
        )
        cfg = native.RRAGLlamaConfig.from_pretrained(
            self.artifact.root / "native", local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.artifact.root / "native/tokenizer",
            local_files_only=True,
            trust_remote_code=False,
        )
        if (
            cfg.num_k != 10
            or cfg.input_dim != 3
            or cfg.unk_token != MARKER
            or tokenizer.convert_tokens_to_ids(MARKER) != cfg.unk_token_id
            or not cfg.freeze_llm
            or cfg.load_in_8bit
        ):
            raise ValidationError("R²AG trained native configuration or marker differs")
        cfg.model_name_or_path = str(self.base)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        with _reader_loader(native, self.base, self.device, dtype):
            model = native.RRAGLlamaForCausalLM(cfg)
        self._generation_transport = _generation_signature_compat(model.llama_model)
        model.llama_model.resize_token_embeddings(len(tokenizer))
        values = torch.load(
            self.artifact.root / "native/RRAGLlama_pytorch_model.bin",
            weights_only=True,
            mmap=True,
            map_location="cpu",
        )
        _load_components(torch, model, values)
        rows = torch.load(
            self.artifact.root / "native/marker-rows.pt", weights_only=True, map_location="cpu"
        )
        if not isinstance(rows, Mapping) or set(rows) != {"marker_id", "input_row", "output_row"}:
            raise ValidationError("R²AG trained marker-row payload differs")
        if type(rows["marker_id"]) is not int or rows["marker_id"] != cfg.unk_token_id:
            raise ValidationError("R²AG trained marker-row index differs")
        with torch.no_grad():
            for name, module in (
                ("input_row", model.get_input_embeddings()),
                ("output_row", model.get_output_embeddings()),
            ):
                value, target = rows[name], module.weight[cfg.unk_token_id]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != target.shape
                    or value.dtype != target.dtype
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValidationError("R²AG trained marker row shape/dtype/value differs")
                target.copy_(value.to(target.device))
        model.eval().requires_grad_(False)
        expected = _read(self.artifact.root / "parameter-proof.json")["final"]
        if any(_parameter_hashes(getattr(model, n)) != expected[n] for n in expected):
            raise ValidationError("R²AG restored retrieval parameters differ from fitted values")
        loader = _module(
            self.source / "RRAG/dataset/load_hotpotqa.py", "agentinputlab_r2ag_inference_prompt"
        )
        loader.PROMPTS_ROOT = self.source / "RRAG/prompts"
        self._loaded = torch, model, tokenizer, loader, _native_formatter(self.source)
        return self._loaded

    def read(self, *, query, documents, output_root, max_new_tokens=512, seed=17):
        """Generate once from ten native-ranked documents; return saved raw evidence."""
        self._verify()
        if not isinstance(query, str) or not query.strip() or "\x00" in query:
            raise ValidationError("R²AG query requires non-empty text")
        if (
            not isinstance(documents, Sequence)
            or len(documents) != 10
            or any(not isinstance(d, R2AGReaderDocument) for d in documents)
            or len({d.document_id for d in documents}) != 10
            or any(a.features[0] < b.features[0] for a, b in pairwise(documents))
        ):
            raise ValidationError("R²AG requires ten unique native-ranked typed documents")
        if (
            type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= 512
            or type(seed) is not int
            or seed < 0
        ):
            raise ValidationError("R²AG generation requires seed >=0 and at most 512 output tokens")
        root = _new_root(output_root, self.source, self.base, self.artifact.root)
        observed = dict(
            generation_attempts=0,
            generation_responses=0,
            reader_forward_attempts=0,
            reader_forward_responses=0,
            reader_input_tokens_attempted=0,
            reader_input_token_unknown_attempts=0,
            conditioning_forward_responses=dict(r_former=0, llama_proj=0),
        )
        captured = {}
        _save(
            root / "claim.json",
            dict(
                deployment=dict(self.fingerprint()),
                query=query,
                documents=[
                    dict(document_id=d.document_id, title=d.title, text=d.text, features=d.features)
                    for d in documents
                ],
                seed=seed,
                max_new_tokens=max_new_tokens,
                new_embedding_calls=0,
                precomputed_embedding_cost_must_be_retained=True,
                automatic_retry=False,
            ),
        )
        start, cpu = time.monotonic(), time.process_time()
        try:
            with _LOCK:
                torch, model, tokenizer, loader, formatter = self._load()
                instruction = loader.get_qa_instruction(
                    query,
                    [dict(title=d.title, text=d.text) for d in documents],
                    True,
                    MARKER,
                    False,
                )
                prompt = formatter.format_instruction_for_response(instruction)
                tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
                    self.device
                )
                if (
                    tokens.input_ids.shape[-1] > 4096
                    or int((tokens.input_ids == model.config.unk_token_id).sum()) != 10
                ):
                    raise ValidationError("R²AG prompt exceeds 4096 tokens or changes marker count")
                features = torch.tensor(
                    [d.features for d in documents], dtype=torch.float32, device=self.device
                )
                torch.manual_seed(seed)
                if self.device.startswith("cuda"):
                    torch.cuda.manual_seed_all(seed)

                def capture(name):
                    def hook(module, args, response):
                        observed["conditioning_forward_responses"][name] += 1
                        value = response[0] if name == "r_former" else response
                        if not bool(torch.isfinite(value).all()):
                            raise ValidationError("R²AG native inference tensor is non-finite")
                        captured[name] = dict(
                            shape=list(value.shape), sha256=_tensor_digest(torch, value)
                        )

                    return hook

                hooks = [
                    getattr(model, n).register_forward_hook(capture(n))
                    for n in ("r_former", "llama_proj")
                ]

                def reader_before(module, args, kwargs):
                    _count_generation_reader_request(observed, kwargs)

                def reader_after(module, args, response):
                    observed["reader_forward_responses"] += 1

                hooks += [
                    model.llama_model.register_forward_pre_hook(reader_before, with_kwargs=True),
                    model.llama_model.register_forward_hook(reader_after),
                ]
                try:
                    observed["generation_attempts"] += 1
                    with (
                        torch.inference_mode(),
                        torch.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
                        ),
                    ):
                        outputs = model.generate(
                            inputs=dict(
                                input_ids=tokens.input_ids,
                                attention_mask=tokens.attention_mask,
                                embeds=features,
                            ),
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            repetition_penalty=1.0,
                            length_penalty=1.0,
                            temperature=1.0,
                        )
                    observed["generation_responses"] += 1
                finally:
                    for hook in hooks:
                        hook.remove()
                if (
                    outputs.ndim != 2
                    or outputs.shape[0] != 1
                    or not 1 <= outputs.shape[1] <= max_new_tokens
                ):
                    raise ValidationError("R²AG native generation token shape or length differs")
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                _save(
                    root / "response.json",
                    dict(
                        answer=answer,
                        token_ids=outputs[0].tolist(),
                        input_tokens=int(tokens.input_ids.numel()),
                        output_tokens=int(outputs.shape[1]),
                        output_cap_reached=int(outputs.shape[1]) == max_new_tokens,
                        native_tensor_captures=captured,
                        generation_transport=self._generation_transport,
                    ),
                )
                if not answer or set(captured) != {"r_former", "llama_proj"}:
                    raise ValidationError(
                        "R²AG native generation is empty or missed retrieval conditioning"
                    )
            _save(
                root / "terminal.json",
                dict(
                    status="generated_candidate",
                    **observed,
                    generation_transport=self._generation_transport,
                    native_tensor_captures=captured,
                    cpu_seconds=time.process_time() - cpu,
                    wall_seconds=time.monotonic() - start,
                    new_optimizer_steps=0,
                    automatic_retry=False,
                    benchmark_score_claimed=False,
                ),
            )
            return root
        except Exception as exc:
            _save(
                root / "terminal.json",
                dict(
                    status="failed_terminal",
                    **observed,
                    generation_transport=self._generation_transport,
                    native_tensor_captures=captured,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    cpu_seconds=time.process_time() - cpu,
                    wall_seconds=time.monotonic() - start,
                    automatic_retry=False,
                ),
            )
            raise

    def close(self):
        self._loaded = None
        gc.collect()
        torch = _torch()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


__all__ = ["R2AGReaderDocument", "R2AGTrainedReader"]
