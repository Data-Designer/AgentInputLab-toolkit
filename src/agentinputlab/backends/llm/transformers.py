"""Optional local Transformers backend loaded without a hard runtime dependency."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.errors import CapabilityError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import freeze_mapping

from .base import GenerationRequest, GenerationResponse, LLMBackend, ModelCapability, Usage


@dataclass(slots=True)
class TransformersBackend(LLMBackend):
    model_id: str
    model_revision: str
    tokenizer_revision: str
    tokenizer: Any = None
    model: Any = None
    generation_kwargs: Mapping[str, Any] = field(default_factory=dict)
    cache_dir: str | None = None
    offline: bool = False
    capabilities: frozenset[ModelCapability] = frozenset(
        {ModelCapability.CHAT, ModelCapability.COMPLETION}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.offline, bool):
            raise ValidationError("TransformersBackend offline must be boolean")
        if self.cache_dir is not None and (
            not isinstance(self.cache_dir, str) or not self.cache_dir.strip()
        ):
            raise ValidationError("TransformersBackend cache_dir must be a non-empty path string")
        if not self.model_revision or not self.tokenizer_revision:
            raise ValidationError(
                "TransformersBackend requires explicit model and tokenizer revisions"
            )
        self.generation_kwargs = freeze_mapping(self.generation_kwargs)
        num_return_sequences = self.generation_kwargs.get("num_return_sequences", 1)
        if (
            isinstance(num_return_sequences, bool)
            or not isinstance(num_return_sequences, int)
            or num_return_sequences != 1
        ):
            raise ValidationError(
                "TransformersBackend requires num_return_sequences to be the integer 1"
            )
        self.capabilities = frozenset(self.capabilities)

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ExternalDependencyError(
                "TransformersBackend requires `pip install agentinputlab[local]`"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.tokenizer_revision,
            trust_remote_code=False,
            cache_dir=self.cache_dir,
            local_files_only=self.offline,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            trust_remote_code=False,
            cache_dir=self.cache_dir,
            local_files_only=self.offline,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.seed is not None and ModelCapability.SEED not in self.capabilities:
            raise CapabilityError(
                "TransformersBackend does not implement deterministic seed control"
            )
        if request.tools and ModelCapability.TOOL_CALLING not in self.capabilities:
            raise CapabilityError("TransformersBackend does not support structured tool calling")
        if request.response_format and not {
            ModelCapability.JSON_OBJECT,
            ModelCapability.JSON_SCHEMA,
        }.intersection(self.capabilities):
            raise CapabilityError("TransformersBackend does not support response_format")
        self._load()
        if request.messages and hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.extend(
                {"role": item.role, "content": item.content} for item in request.messages
            )
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif request.messages:
            raise CapabilityError(
                "Transformers tokenizer has no chat template for message-based generation"
            )
        else:
            prompt = f"{request.system}\n\n{request.prompt}" if request.system else request.prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature > 0,
            temperature=max(request.temperature, 1e-5),
            **dict(self.generation_kwargs),
        )
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        try:
            input_shape = inputs["input_ids"].shape
            if len(input_shape) != 2 or int(input_shape[0]) != 1:
                raise ValueError("input_ids must contain exactly one two-dimensional sequence")
            input_width = int(input_shape[1])
            if input_width < 1:
                raise ValueError("input_ids sequence must not be empty")

            sequence_shape = sequences.shape
            if len(sequence_shape) != 2 or int(sequence_shape[0]) != 1:
                raise ValueError("generate must return exactly one two-dimensional sequence")
            sequence_width = int(sequence_shape[1])
            if sequence_width < input_width:
                raise ValueError("generated sequence is shorter than its input")

            generated_sequence = sequences[0]
            new_tokens = generated_sequence[input_width:]
            output_tokens = sequence_width - input_width
        except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValidationError(
                "Transformers model.generate returned an unsupported sequence shape"
            ) from exc
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            input_tokens = input_width
        else:
            try:
                mask_shape = attention_mask.shape
                if (
                    len(mask_shape) != 2
                    or int(mask_shape[0]) != 1
                    or int(mask_shape[1]) != input_width
                ):
                    raise ValueError("attention_mask shape must match the single input sequence")
                input_tokens = int(attention_mask.sum())
                if input_tokens < 0 or input_tokens > input_width:
                    raise ValueError("attention_mask token count is outside the input width")
            except (AttributeError, OverflowError, TypeError, ValueError) as exc:
                raise ValidationError(
                    "Transformers tokenizer returned an unsupported attention_mask"
                ) from exc
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerationResponse(
            text=text,
            model=self.model_id,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "backend": "transformers",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "generation_kwargs": dict(self.generation_kwargs),
        }
