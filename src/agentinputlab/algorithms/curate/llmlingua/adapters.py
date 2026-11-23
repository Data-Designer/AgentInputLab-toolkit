"""Thin, lazy bridges to the pinned Microsoft LLMLingua runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ContextBlock,
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
from agentinputlab.contracts.serialization import content_hash

from .budget import parse_context_budget
from .span_map import SpanMapping, build_span_map

CompressorFactory = Callable[..., Any]


def _bind_native_tokenizer_family(compressor: Any, source_name: str) -> None:
    """Fix 0.2.2 post-load dispatch for generic local snapshot directories.

    Native weights/tokenizer have already loaded from ``source_name``. Upstream
    word-boundary helpers instead inspect this field for a family substring;
    route those helpers from the actual loaded config, without loading another
    model or changing compression controls. Caller factories never use this.
    """
    if importlib.metadata.version("llmlingua") != "0.2.2":
        raise ExternalDependencyError("tokenizer family bridge requires LLMLingua 0.2.2")
    family = {"bert": "bert-base-multilingual-cased", "xlm-roberta": "xlm-roberta-large"}.get(
        compressor.model.config.model_type
    )
    if family is None or compressor.model_name not in {source_name, family}:
        raise ValidationError("LLMLingua-2 native tokenizer family or source binding differs")
    compressor.model_name = family


@lru_cache(maxsize=4)
def _cached_official_compressor(
    model_name: str, device_map: str, config_json: str, dtype_name: str,
    use_llmlingua2: bool,
) -> Any:
    """Reuse native weights without making them intervention configuration.

    The bounded hidden-dependency audit must inspect caller-supplied factories,
    not a lazy library's full tokenizer/tensor graph. Only the built-in loader
    uses this process-local cache, keyed by every model-loading option. Injected
    factories retain the original inspection path and cannot enter this cache.
    """
    import torch
    from llmlingua import PromptCompressor

    config = json.loads(config_json)
    config['torch_dtype'] = getattr(torch, dtype_name)
    compressor = PromptCompressor(
        model_name=model_name, device_map=device_map, model_config=config,
        open_api_config={}, use_llmlingua2=use_llmlingua2, llmlingua2_config={},
    )
    if use_llmlingua2:
        _bind_native_tokenizer_family(compressor, model_name)
    return compressor


@dataclass(frozen=True, slots=True)
class LongLLMLinguaConfig:
    rate: float = 0.5
    target_token: int = -1
    iterative_size: int = 200
    use_sentence_level_filter: bool = False
    use_context_level_filter: bool = True
    use_token_level_filter: bool = True
    context_budget: int | str = "+100"
    token_budget_ratio: float = 1.4
    condition_in_question: str = "none"
    reorder_context: str = "sort"
    dynamic_context_compression_ratio: float = 0.0
    condition_compare: bool = False
    add_instruction: bool = False
    concate_question: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.rate <= 1:
            raise ValidationError("LongLLMLingua rate must be in (0, 1]")
        if self.target_token == 0 or self.target_token < -1:
            raise ValidationError("LongLLMLingua target_token must be -1 or positive")
        if self.iterative_size < 1:
            raise ValidationError("LongLLMLingua iterative_size must be positive")
        if self.reorder_context not in {"original", "sort", "two_stage"}:
            raise ValidationError("unsupported LongLLMLingua reorder_context")
        parse_context_budget(self.context_budget)


@dataclass(frozen=True, slots=True)
class LLMLingua2Config:
    rate: float = 0.5
    target_token: int = -1
    use_context_level_filter: bool = False
    use_token_level_filter: bool = True
    target_context: int = -1
    context_level_rate: float = 1.0
    context_level_target_token: int = -1
    force_context_ids: tuple[int, ...] = ()
    force_tokens: tuple[str, ...] = ()
    force_reserve_digit: bool = False
    drop_consecutive: bool = False
    chunk_end_tokens: tuple[str, ...] = (".", "\n")
    return_word_label: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.rate <= 1:
            raise ValidationError("LLMLingua-2 rate must be in (0, 1]")
        if self.target_token == 0 or self.target_token < -1:
            raise ValidationError("LLMLingua-2 target_token must be -1 or positive")


@dataclass(slots=True)
class _OfficialAdapterBase:
    model_name: str
    model_revision: str
    device_map: str = "cpu"
    model_config: Mapping[str, Any] = field(default_factory=dict)
    profile_kind: str = "official_native"
    compressor_factory: CompressorFactory | None = None
    _compressor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.profile_kind not in {
            "official_native",
            "controlled_adapter",
            "factorial_adapter",
        }:
            raise ValidationError("LLMLingua profile_kind is unsupported")

    @staticmethod
    def _validate_torch_dtype(value: Any) -> None:
        if not isinstance(value, str):
            return
        label = value.removeprefix("torch.").lower()
        if label not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValidationError(
                "LLMLingua torch_dtype must be auto, float16, bfloat16, or float32"
            )

    def _runtime_model_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Bridge LLMLingua 0.2.2 legacy torch_dtype API to modern Transformers."""
        try:
            import torch
            from transformers import AutoConfig
        except (ImportError, ModuleNotFoundError) as exc:
            raise ExternalDependencyError(
                "official LLMLingua runtime requires torch and transformers"
            ) from exc

        runtime_config = dict(config)
        configured = runtime_config.get("torch_dtype", "auto")
        if isinstance(configured, str):
            label = configured.removeprefix("torch.").lower()
            if label == "auto":
                probe_config = dict(runtime_config)
                probe_config.pop("torch_dtype", None)
                try:
                    upstream_config = AutoConfig.from_pretrained(
                        self.model_name,
                        **probe_config,
                    )
                except (OSError, ValueError) as exc:
                    raise ExternalDependencyError(
                        "unable to inspect the pinned LLMLingua model dtype"
                    ) from exc
                configured = getattr(upstream_config, "dtype", None)
                if configured is None:
                    configured = torch.float32
            else:
                configured = getattr(torch, label)
        if not isinstance(configured, torch.dtype):
            raise ValidationError("LLMLingua torch_dtype must resolve to torch.dtype")
        runtime_config["torch_dtype"] = configured
        return runtime_config

    @staticmethod
    def _install_legacy_cache_bridge(compressor: Any) -> None:
        """Keep LLMLingua 0.2.2 cache slicing compatible with modern Transformers."""
        model = getattr(compressor, "model", None)
        if model is None or getattr(model, "_agentinputlab_llmlingua_legacy_cache_bridge", False):
            return
        try:
            cache_utils = importlib.import_module("transformers.cache_utils")
            dynamic_cache_type = cache_utils.DynamicCache
        except (ImportError, AttributeError):
            return
        original_forward = model.forward

        def to_dynamic_cache(legacy_cache: Any) -> Any:
            normalized = tuple(tuple(layer) for layer in legacy_cache)
            from_legacy_cache = getattr(dynamic_cache_type, "from_legacy_cache", None)
            if callable(from_legacy_cache):
                return from_legacy_cache(normalized)
            # Transformers 5 removed ``from_legacy_cache`` but retained a
            # checksum-equivalent constructor over per-layer (key, value)
            # tensors.  This is representation adaptation only: LLMLingua
            # still performs the original cache slicing before every call.
            try:
                return dynamic_cache_type(ddp_cache_data=normalized)
            except TypeError:
                return dynamic_cache_type(normalized)

        def to_legacy_cache(returned_cache: Any) -> Any:
            converter = getattr(returned_cache, "to_legacy_cache", None)
            if callable(converter):
                return converter()
            layers = getattr(returned_cache, "layers", None)
            if layers is None:
                return returned_cache
            converted = []
            for layer in layers:
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if key is None or value is None:
                    return returned_cache
                converted.append((key, value))
            return tuple(converted)

        def legacy_compatible_forward(*args: Any, **kwargs: Any) -> Any:
            legacy_cache = kwargs.get("past_key_values")
            if isinstance(legacy_cache, (list, tuple)) and legacy_cache:
                kwargs["past_key_values"] = to_dynamic_cache(legacy_cache)
            response = original_forward(*args, **kwargs)
            returned_cache = getattr(response, "past_key_values", None)
            if returned_cache is not None:
                response.past_key_values = to_legacy_cache(returned_cache)
            return response

        model.forward = legacy_compatible_forward
        model._agentinputlab_llmlingua_legacy_cache_bridge = True

    def _load(self, *, use_llmlingua2: bool) -> Any:
        if self._compressor is not None:
            return self._compressor
        config = dict(self.model_config)
        configured_revision = config.get("revision")
        if configured_revision is not None and configured_revision != self.model_revision:
            raise ValidationError(
                "LLMLingua model_config cannot override the pinned model revision"
            )
        if config.get("trust_remote_code") not in {None, False}:
            raise ValidationError("LLMLingua adapters disable trust_remote_code")
        self._validate_torch_dtype(config.get("torch_dtype", "auto"))
        config["revision"] = self.model_revision
        config["trust_remote_code"] = False
        factory = self.compressor_factory
        if factory is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise ExternalDependencyError(
                    "official LLMLingua runtime requires `pip install agentinputlab[compression]`"
                ) from exc
            factory = PromptCompressor
            config = self._runtime_model_config(config)
        try:
            if self.compressor_factory is None:
                public_config = dict(config)
                dtype_name = str(public_config.pop('torch_dtype')).removeprefix('torch.')
                compressor = _cached_official_compressor(
                    self.model_name, self.device_map,
                    json.dumps(public_config, sort_keys=True), dtype_name, use_llmlingua2,
                )
                self._install_legacy_cache_bridge(compressor)
                return compressor
            self._compressor = factory(
                model_name=self.model_name,
                device_map=self.device_map,
                model_config=config,
                open_api_config={},
                use_llmlingua2=use_llmlingua2,
                llmlingua2_config={},
            )
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise ExternalDependencyError(
                f"unable to load pinned LLMLingua asset {self.model_name}@{self.model_revision}"
            ) from exc
        self._install_legacy_cache_bridge(self._compressor)
        return self._compressor

    @staticmethod
    def _validate_context(state: FrozenAgentState) -> tuple[ContextBlock, ...]:
        if not state.context:
            raise ValidationError("LLMLingua adapters require visible context blocks")
        return state.context

    def _patch(
        self,
        *,
        state: FrozenAgentState,
        compressed_context: str,
        native_result: Mapping[str, Any],
        spans: tuple[SpanMapping, ...],
        method_id: str,
        release_id: str,
        query_consumed: bool,
        config: Mapping[str, Any],
        implementation_identity: str,
    ) -> InputPatch:
        if not compressed_context.strip():
            raise AlgorithmOutputError(
                "official LLMLingua runtime returned empty compressed context"
            )
        trace = ExecutionTrace(
            method_id=method_id,
            release_id=release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "official_runtime_config",
                    {
                        "model_name": self.model_name,
                        "model_revision": self.model_revision,
                        "device_map": self.device_map,
                        "query_consumed": query_consumed,
                        "config": dict(config),
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "official_compression_result",
                    {
                        "origin_tokens_native": native_result.get("origin_tokens"),
                        "compressed_tokens_native": native_result.get("compressed_tokens"),
                        "ratio_native": native_result.get("ratio"),
                        "rate_native": native_result.get("rate"),
                        "span_count": len(spans),
                        "native_tokenizer_semantics": "upstream_gpt_3_5_statistics",
                    },
                ),
            ),
            counters={"compressor_calls": 1},
        )
        source_refs = tuple(
            block.block_id or f"context-{i}" for i, block in enumerate(state.context)
        )
        context = ContextBlock(
            block_id=f"{release_id}:compressed",
            text=compressed_context,
            source="llmlingua_official_runtime",
            metadata={
                "source_block_ids": source_refs,
                "span_map": tuple(asdict(item) for item in spans),
                "restoration_map": tuple(
                    {
                        "block_id": block.block_id or f"context-{index}",
                        "source_state_hash": state.state_hash,
                        "text_hash": content_hash(block.text),
                    }
                    for index, block in enumerate(state.context)
                ),
                "query_consumed": query_consumed,
            },
        )
        return InputPatch(
            method_id=method_id,
            release_id=release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_CONTEXT,
                    (context,),
                    target="visible_context",
                    rationale="official LLMLingua runtime output",
                ),
            ),
            trace=trace,
            implementation_identity=implementation_identity,
            profile_kind=self.profile_kind,
            metadata={
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "query_consumed": query_consumed,
                "native_statistics": {
                    key: native_result.get(key)
                    for key in ("origin_tokens", "compressed_tokens", "ratio", "rate", "saving")
                },
            },
        )


@dataclass(slots=True)
class LongLLMLinguaOfficialAdapter(_OfficialAdapterBase):
    model_name: str = "NousResearch/Llama-2-7b-hf"
    model_revision: str = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
    config: LongLLMLinguaConfig = LongLLMLinguaConfig()
    method_id: str = "longllmlingua"
    release_id: str = "curate.longllmlingua.official"
    family: SemanticFamily = SemanticFamily.CURATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        blocks = self._validate_context(state)
        compressor = self._load(use_llmlingua2=False)
        canonical_budget = parse_context_budget(self.config.context_budget)
        kwargs = {
            "instruction": state.instruction,
            "question": state.query,
            "rate": self.config.rate,
            "target_token": self.config.target_token,
            "iterative_size": self.config.iterative_size,
            "use_sentence_level_filter": self.config.use_sentence_level_filter,
            "use_context_level_filter": self.config.use_context_level_filter,
            "use_token_level_filter": self.config.use_token_level_filter,
            "context_budget": canonical_budget,
            "token_budget_ratio": self.config.token_budget_ratio,
            "condition_in_question": self.config.condition_in_question,
            "reorder_context": self.config.reorder_context,
            "dynamic_context_compression_ratio": self.config.dynamic_context_compression_ratio,
            "condition_compare": self.config.condition_compare,
            "add_instruction": self.config.add_instruction,
            "rank_method": "longllmlingua",
            "concate_question": self.config.concate_question,
        }
        result = compressor.compress_prompt([block.text for block in blocks], **kwargs)
        native = _validate_native_result(result)
        compressed_context = _strip_native_wrapper(
            native["compressed_prompt"],
            instruction=state.instruction,
            question=state.query,
            concate_question=self.config.concate_question,
        )
        spans = build_span_map(compressed_context, blocks)
        public_config = asdict(self.config)
        public_config["context_budget"] = canonical_budget
        return self._patch(
            state=state,
            compressed_context=compressed_context,
            native_result=native,
            spans=spans,
            method_id=self.method_id,
            release_id=self.release_id,
            query_consumed=True,
            config=public_config,
            implementation_identity="thin_official_runtime_bridge",
        )


@dataclass(slots=True)
class LLMLingua2OfficialAdapter(_OfficialAdapterBase):
    model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    model_revision: str = "ebaba9b0e874dadd3003ffcff828e4397e568089"
    config: LLMLingua2Config = LLMLingua2Config()
    method_id: str = "llmlingua2"
    release_id: str = "curate.llmlingua2.xlm_roberta_official"
    family: SemanticFamily = SemanticFamily.CURATE

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        del context
        blocks = self._validate_context(state)
        compressor = self._load(use_llmlingua2=True)
        kwargs = {
            "rate": self.config.rate,
            "target_token": self.config.target_token,
            "use_context_level_filter": self.config.use_context_level_filter,
            "use_token_level_filter": self.config.use_token_level_filter,
            "target_context": self.config.target_context,
            "context_level_rate": self.config.context_level_rate,
            "context_level_target_token": self.config.context_level_target_token,
            "force_context_ids": list(self.config.force_context_ids),
            "return_word_label": self.config.return_word_label,
            "force_tokens": list(self.config.force_tokens),
            "force_reserve_digit": self.config.force_reserve_digit,
            "drop_consecutive": self.config.drop_consecutive,
            "chunk_end_tokens": list(self.config.chunk_end_tokens),
        }
        # The pinned official dispatcher intentionally ignores instruction and
        # question when use_llmlingua2=True.  Do not pre-compose them here.
        result = compressor.compress_prompt([block.text for block in blocks], **kwargs)
        native = _validate_native_result(result)
        compressed_context = native["compressed_prompt"]
        spans = build_span_map(compressed_context, blocks)
        return self._patch(
            state=state,
            compressed_context=compressed_context,
            native_result=native,
            spans=spans,
            method_id=self.method_id,
            release_id=self.release_id,
            query_consumed=False,
            config=asdict(self.config),
            implementation_identity="thin_official_runtime_bridge",
        )


def _validate_native_result(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("official LLMLingua runtime must return a mapping")
    prompt = value.get("compressed_prompt")
    if not isinstance(prompt, str):
        raise AlgorithmOutputError("official LLMLingua result lacks compressed_prompt text")
    return value


def _strip_native_wrapper(
    prompt: str,
    *,
    instruction: str,
    question: str,
    concate_question: bool,
) -> str:
    """Recover the compressed context while preserving native budget semantics."""

    context = prompt
    if instruction:
        prefix = f"{instruction}\n\n"
        if not context.startswith(prefix):
            raise AlgorithmOutputError(
                "LongLLMLingua output lost the expected instruction boundary"
            )
        context = context[len(prefix) :]
    if question and concate_question:
        suffix = f"\n\n{question}"
        if not context.endswith(suffix):
            raise AlgorithmOutputError("LongLLMLingua output lost the expected question boundary")
        context = context[: -len(suffix)]
    return context
