"""LongLLMLingua with immutable shared-comparison identities and settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping

from .adapters import LongLLMLinguaConfig
from .budget import parse_context_budget

_DEFAULT_CONFIG = LongLLMLinguaConfig()


@dataclass(frozen=True, slots=True)
class LongLLMLinguaCompressionRequest:
    state: FrozenAgentState
    config: LongLLMLinguaConfig
    seed: int

    def __post_init__(self):
        if (not isinstance(self.state, FrozenAgentState) or not isinstance(self.config, LongLLMLinguaConfig)
                or type(self.seed) is not int or self.seed < 0):
            raise ValidationError("LongLLMLingua requires a typed compression request")


def _identity(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError(f"LongLLMLingua {label} fingerprint must be nonempty")
    try:
        result = freeze_mapping(value)
        content_hash(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"LongLLMLingua {label} fingerprint must be canonical JSON") from exc
    return result


def _identities(context: InterventionContext) -> dict[str, Mapping[str, Any]]:
    values = {}
    for name, dependency, operation in (
        ("compressor", context.external_runtime, "run"),
        ("reader", context.backend, "generate"),
    ):
        fingerprint = getattr(dependency, "fingerprint", None)
        if not callable(fingerprint) or not callable(getattr(dependency, operation, None)):
            raise ExternalDependencyError(
                f"controlled LongLLMLingua requires {name} with {operation}() and fingerprint()"
            )
        values[name] = _identity(fingerprint(), name)
    return values


@dataclass(frozen=True, slots=True)
class LongLLMLinguaControlledBindings:
    """Operator identities, not a certificate of weights or a paired experiment.

    The compressor identity includes its helper model and tokenizer. The reader
    is checked but not called here: its benchmark driver must preserve the same
    reader, generation policy and remaining budget for all comparison arms.
    """

    compressor: Mapping[str, Any]
    reader: Mapping[str, Any]
    budget: ExecutionBudget
    config: LongLLMLinguaConfig = _DEFAULT_CONFIG
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.budget, ExecutionBudget) or not isinstance(self.config, LongLLMLinguaConfig):
            raise ValidationError("controlled LongLLMLingua requires typed budget/config")
        if type(self.seed) is not int or self.seed < 0:
            raise ValidationError("controlled LongLLMLingua seed must be nonnegative")
        object.__setattr__(self, "config", replace(self.config,
            context_budget=parse_context_budget(self.config.context_budget)))
        for name in ("compressor", "reader"):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in ("model", "tokenizer", "source"):
            _identity(self.compressor.get(name), f"compressor {name}")

    @classmethod
    def from_context(cls, context: InterventionContext, *, budget: ExecutionBudget,
                     config: LongLLMLinguaConfig = _DEFAULT_CONFIG) -> LongLLMLinguaControlledBindings:
        if not isinstance(context, InterventionContext):
            raise ValidationError("controlled LongLLMLingua requires an InterventionContext")
        return cls(**_identities(context), budget=budget, config=config, seed=context.seed)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LongLLMLinguaControlledBindings:
        required = {"compressor", "reader", "budget"}
        if not required.issubset(value) or set(value) - required - {"config", "seed"}:
            raise ValidationError("invalid controlled LongLLMLingua binding fields")
        try:
            return cls(compressor=value["compressor"], reader=value["reader"],
                budget=ExecutionBudget(**value["budget"]),
                config=LongLLMLinguaConfig(**value.get("config", {})), seed=value.get("seed", 0))
        except TypeError as exc:
            raise ValidationError("invalid controlled LongLLMLingua binding settings") from exc

    @property
    def binding_hash(self) -> str:
        return content_hash(self)

    def require_match(self, context: InterventionContext) -> None:
        if context.seed != self.seed:
            raise ValidationError("controlled LongLLMLingua seed differs from binding")
        for name, value in _identities(context).items():
            if content_hash(value) != content_hash(getattr(self, name)):
                raise ValidationError(f"controlled LongLLMLingua {name} differs from binding")


@dataclass(frozen=True, slots=True)
class LongLLMLinguaControlled:
    bindings: LongLLMLinguaControlledBindings | Mapping[str, Any]
    profile_kind: str = "controlled_adapter"
    method_id: str = "longllmlingua"
    release_id: str = "curate.longllmlingua.controlled"
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        if isinstance(self.bindings, Mapping):
            object.__setattr__(self, "bindings", LongLLMLinguaControlledBindings.from_mapping(self.bindings))
        if not isinstance(self.bindings, LongLLMLinguaControlledBindings):
            raise ValidationError("controlled LongLLMLingua requires frozen bindings")
        if self.profile_kind not in {"controlled_adapter", "factorial_adapter"}:
            raise ValidationError("controlled LongLLMLingua requires controlled or factorial mode")
        if (self.method_id != "longllmlingua" or self.release_id != "curate.longllmlingua.controlled"
                or self.family is not SemanticFamily.CURATE):
            raise ValidationError("controlled LongLLMLingua identity differs from its registry")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if state.budget != self.bindings.budget:
            raise ValidationError("controlled LongLLMLingua state budget differs from binding")
        self.bindings.require_match(context)
        result = context.external_runtime.run(LongLLMLinguaCompressionRequest(
            state, self.bindings.config, context.seed))
        native = result.get("patch") if isinstance(result, Mapping) else None
        self.bindings.require_match(context)
        if (not isinstance(native, InputPatch) or native.method_id != self.method_id
                or native.release_id != "curate.longllmlingua.official"
                or native.source_state_hash != state.state_hash):
            raise AlgorithmOutputError("LongLLMLingua runtime returned a different native patch identity")
        configs = [e.payload.get("config") for e in native.trace.events if e.name == "official_runtime_config"]
        if len(configs) != 1 or content_hash(configs[0]) != content_hash(asdict(self.bindings.config)):
            raise AlgorithmOutputError("LongLLMLingua runtime used different method settings")
        model = self.bindings.compressor["model"]
        if (native.metadata.get("model_name") != model.get("locator")
                or native.metadata.get("model_revision") != model.get("revision")):
            raise AlgorithmOutputError("LongLLMLingua native patch uses a different helper model")
        return replace(native, release_id=self.release_id, profile_kind=self.profile_kind,
            trace=replace(native.trace, release_id=self.release_id, events=(
                TraceEvent(TracePhase.PREPARE, "controlled_dependency_binding", {
                    "binding_sha256": self.bindings.binding_hash, "budget": self.bindings.budget,
                    "seed": self.bindings.seed, "identity_verification": "operator_fingerprint_equality",
                    "reader_bound_not_invoked": True}), *native.trace.events)),
            metadata={**native.metadata, "native_source_release": native.release_id,
                "controlled_binding_sha256": self.bindings.binding_hash,
                "reader_generation_performed": False, "helper_compute_accounting": "separate_from_reader_generation",
                "original_paper_result_parity_claimed": False})


__all__ = ["LongLLMLinguaCompressionRequest", "LongLLMLinguaControlled", "LongLLMLinguaControlledBindings"]
