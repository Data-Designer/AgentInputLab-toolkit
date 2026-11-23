"""Thin bridge for the Apache-2.0 QueryGym community implementation."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from agentinputlab.backends.llm import GenerationRequest
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    PatchOp,
    PatchOperation,
    ProfileKind,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.observability import external_usage_counters, normalize_external_usage

from .models import QueryGymMode, QueryGymRawResult, QueryGymRequest

QUERYGYM_REVISION = "6d2d19213c7212c7940143e7b9b5f1d2af7ff47b"


@runtime_checkable
class QueryGymRuntime(Protocol):
    """Runtime seam; implementations may be an optional import or worker client."""

    def reformulate(self, request: QueryGymRequest) -> QueryGymRawResult: ...

    def fingerprint(self) -> Mapping[str, Any]: ...


def select_demonstration_ids(
    candidate_ids: Sequence[str],
    *,
    count: int,
    seed: int,
) -> tuple[str, ...]:
    """Select demonstrations without relying on QueryGym's unseeded global RNG."""

    if count < 0:
        raise ValueError("demonstration count must be non-negative")
    unique = tuple(dict.fromkeys(candidate_ids))
    if count > len(unique):
        raise ValueError("demonstration count exceeds candidate pool")
    return tuple(random.Random(seed).sample(unique, count))


class QueryGymLiveBridge:
    method_id = "query2doc"
    release_id = "rewrite.query2doc.querygym_live"
    family = SemanticFamily.REFORMULATE

    def __init__(
        self,
        runtime: QueryGymRuntime | None = None,
        *,
        mode: str | QueryGymMode = QueryGymMode.ZERO_SHOT,
        demonstration_ids: Sequence[str] = (),
        temperature: float = 0.7,
        max_tokens: int = 256,
        profile_kind: str | ProfileKind = ProfileKind.CONTROLLED_ADAPTER,
    ) -> None:
        GenerationRequest(prompt="configuration validation", temperature=temperature,
                          max_tokens=max_tokens)
        self.runtime = runtime
        self.mode = QueryGymMode.parse(mode)
        self.demonstration_ids = tuple(demonstration_ids)
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            self.profile_kind = ProfileKind(profile_kind)
        except ValueError as exc:
            raise ValidationError(f"unsupported QueryGym profile kind: {profile_kind!r}") from exc
        if self.profile_kind not in {
            ProfileKind.OFFICIAL_NATIVE,
            ProfileKind.CONTROLLED_ADAPTER,
        }:
            raise ValidationError(
                "QueryGym supports only official_native or controlled_adapter profiles"
            )
        self.runtime_identity: Mapping[str, Any] = {}
        if runtime is not None:
            fingerprint = getattr(runtime, "fingerprint", None)
            if not callable(fingerprint):
                raise ValidationError("QueryGym runtime requires a stable fingerprint()")
            identity = fingerprint()
            if not isinstance(identity, Mapping):
                raise ValidationError("QueryGym runtime fingerprint must be a mapping")
            if identity.get("revision") != QUERYGYM_REVISION:
                raise ValidationError("QueryGym runtime revision does not match the pinned commit")
            self.runtime_identity = freeze_mapping(identity)

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        runtime = self._resolve_runtime(context)
        if runtime is None:
            raise ExternalDependencyError(
                "QueryGym runtime is unavailable; install/mount the pinned community "
                f"implementation at {QUERYGYM_REVISION}"
            )
        runtime_identity = self._validate_runtime_identity(runtime)
        backend_fingerprint = (
            dict(context.backend.fingerprint())
            if context.backend is not None and hasattr(context.backend, "fingerprint")
            else {}
        )
        request = QueryGymRequest(
            query_id=state.task_id,
            query=state.query,
            mode=self.mode,
            demonstration_ids=self.demonstration_ids,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=context.seed,
            backend_fingerprint=backend_fingerprint,
        )
        try:
            raw = runtime.reformulate(request)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ExternalDependencyError(
                "the pinned QueryGym runtime is missing an optional dependency"
            ) from exc
        if not isinstance(raw, QueryGymRawResult):
            raise AlgorithmOutputError("QueryGym runtime returned an invalid result envelope")
        usage = normalize_external_usage(raw.usage)
        if usage.model_calls != 1 or usage.retrieval_calls != 0:
            raise AlgorithmOutputError(
                "QueryGym Query2doc usage requires exactly one model call and no retrieval calls"
            )
        fallback = bool(raw.fallback or not raw.reformulated.strip())
        reformulated = state.query if fallback else raw.reformulated
        operation = (
            PatchOperation(
                PatchOp.NO_OP,
                rationale="QueryGym returned its native fallback to the original query",
            )
            if fallback
            else PatchOperation(
                PatchOp.REPLACE_QUERY,
                value=reformulated,
                target="query",
                rationale="QueryGym community Query2doc live reformulation",
            )
        )
        warnings = ("fallback_to_original_query",) if fallback else ()
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "querygym_request",
                    {
                        "mode": self.mode.value,
                        "demonstration_ids": self.demonstration_ids,
                        "seed": context.seed,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "backend_fingerprint": backend_fingerprint,
                        "upstream_revision": QUERYGYM_REVISION,
                        "runtime_identity_hash": content_hash(runtime_identity),
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "querygym_result",
                    {
                        "generated_text": raw.generated_text,
                        "reformulated": reformulated,
                        "prompt_hash": raw.prompt_hash,
                        "fallback": fallback,
                        "upstream_metadata": dict(raw.metadata),
                    },
                ),
            ),
            counters={
                **external_usage_counters(usage),
                "fallbacks": int(fallback),
            },
            warnings=warnings,
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=trace,
            implementation_identity=f"community:QueryGym@{QUERYGYM_REVISION}",
            profile_kind=self.profile_kind.value,
            metadata={
                "implementation_origin": "community",
                "upstream_profile_semantics": "community_native",
                "fallback": fallback,
                "upstream_revision": QUERYGYM_REVISION,
            },
        )

    def _resolve_runtime(self, context: InterventionContext) -> QueryGymRuntime | None:
        injected = context.external_runtime
        if injected is not None and self.runtime is not None and injected is not self.runtime:
            raise ValidationError(
                "QueryGym runtime is ambiguous; use either the constructor or "
                "InterventionContext.external_runtime"
            )
        return injected if injected is not None else self.runtime

    @staticmethod
    def _validate_runtime_identity(runtime: QueryGymRuntime) -> Mapping[str, Any]:
        fingerprint = getattr(runtime, "fingerprint", None)
        if not callable(fingerprint):
            raise ValidationError("QueryGym runtime requires a stable fingerprint()")
        identity = fingerprint()
        if not isinstance(identity, Mapping) or not identity:
            raise ValidationError("QueryGym runtime fingerprint must be a non-empty mapping")
        if identity.get("revision") != QUERYGYM_REVISION:
            raise ValidationError("QueryGym runtime revision does not match the pinned commit")
        return freeze_mapping(identity)
