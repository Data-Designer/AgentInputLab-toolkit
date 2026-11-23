"""Future-turn outcome modeling for clarification selection.

This is a controlled inference-time adapter of Zhang, Knox, and Choi (ICLR
2025).  The official method learns from double-turn preference labels.  The
adapter retains the paper's defining comparison—rank a direct answer and
clarifying questions by their simulated future outcomes—but performs that
comparison at inference time because the authors did not publish checkpoints.
It must therefore not be described as official checkpoint or result parity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.backends.llm.base import require_capabilities
from agentinputlab.contracts import (
    AlgorithmOutputError,
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

from ._structured import parse_json_object, string_list, visible_problem

FUTURE_TURNS_UPSTREAM_COMMIT = "cfda2bdf3946de19a7ab4e7a8bc764ee8526ef95"
FUTURE_TURNS_RELEASE = "clarify.future_turns.future_outcome_match"
FUTURE_TURNS_PROMPT_PROTOCOL = "agentinputlab.future-turn-clarification.v1"


@dataclass(frozen=True, slots=True)
class FutureTurnControls:
    max_interpretations: int = 5
    max_candidates: int = 3
    max_tokens: int = 768
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not 2 <= self.max_interpretations <= 10:
            raise ValidationError("future-turn max_interpretations must be in [2, 10]")
        if not 1 <= self.max_candidates <= 8:
            raise ValidationError("future-turn max_candidates must be in [1, 8]")
        if self.max_tokens < 128:
            raise ValidationError("future-turn max_tokens must be at least 128")
        if self.temperature < 0:
            raise ValidationError("future-turn temperature must be non-negative")


@dataclass(frozen=True, slots=True)
class FutureTurnOutcome:
    clarification_answer: str
    tailored_response: str
    abstain: bool


@dataclass(frozen=True, slots=True)
class FutureTurnCandidate:
    question: str
    outcomes: tuple[FutureTurnOutcome, ...]
    utility: float


@dataclass(slots=True)
class FutureTurnClarifier:
    """Select clarification only when simulated future utility beats direct answer."""

    backend: LLMBackend | None = None
    controls: FutureTurnControls = FutureTurnControls()
    method_id: str = "future_turns"
    release_id: str = FUTURE_TURNS_RELEASE
    family: SemanticFamily = SemanticFamily.CLARIFY

    def __post_init__(self) -> None:
        if not isinstance(self.controls, FutureTurnControls):
            raise ValidationError("future-turn controls must be FutureTurnControls")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        backend = self.backend or context.backend
        if backend is None:
            raise ExternalDependencyError("future-turn clarification requires an LLM backend")
        require_capabilities(
            backend,
            {ModelCapability.CHAT, ModelCapability.JSON_OBJECT, ModelCapability.SEED},
        )
        calls: list[dict[str, Any]] = []
        analysis = self._call(
            backend,
            system=(
                "Identify plausible user interpretations without using hidden benchmark labels. "
                "Return JSON only."
            ),
            prompt=(
                f"{visible_problem(state)}\n\n"
                'Return {"interpretations":[...],"direct_answer":"...",'
                '"direct_supported_ids":[0,...]}. List distinct plausible intended tasks. '
                "direct_supported_ids are interpretations safely satisfied by the direct answer."
            ),
            seed=context.seed,
            phase="interpretations_and_direct_answer",
            calls=calls,
        )
        interpretations = string_list(
            analysis.get("interpretations"),
            label="future-turn interpretations",
            minimum=1,
            maximum=self.controls.max_interpretations,
        )
        direct_answer = analysis.get("direct_answer")
        if not isinstance(direct_answer, str) or not direct_answer.strip():
            raise AlgorithmOutputError("future-turn direct_answer must be non-empty text")
        direct_ids = _integer_ids(
            analysis.get("direct_supported_ids"),
            upper=len(interpretations),
            label="direct_supported_ids",
        )
        direct_utility = len(direct_ids) / len(interpretations)

        candidates: tuple[FutureTurnCandidate, ...] = ()
        if len(interpretations) > 1:
            generated = self._call(
                backend,
                system=(
                    "Generate concise clarifying questions that distinguish the supplied user "
                    "interpretations. Return JSON only."
                ),
                prompt=(
                    f"{visible_problem(state)}\n\n"
                    f"Interpretations:\n{_numbered(interpretations)}\n\n"
                    f'Return {{"questions":[...]}} with at most '
                    f"{self.controls.max_candidates} targeted questions."
                ),
                seed=context.seed + 1,
                phase="candidate_questions",
                calls=calls,
            )
            questions = string_list(
                generated.get("questions"),
                label="future-turn candidate questions",
                minimum=1,
                maximum=self.controls.max_candidates,
            )
            candidates = tuple(
                self._simulate_candidate(
                    backend,
                    state=state,
                    interpretations=interpretations,
                    question=question,
                    seed=context.seed + 2 + index,
                    calls=calls,
                )
                for index, question in enumerate(questions)
            )

        best = max(candidates, key=lambda item: item.utility, default=None)
        should_clarify = best is not None and best.utility > direct_utility
        operation = (
            PatchOperation(
                PatchOp.REQUEST_CLARIFICATION,
                best.question,
                target="user",
                rationale="Simulated double-turn utility exceeded direct-answer utility.",
            )
            if should_clarify
            else PatchOperation(
                PatchOp.NO_OP,
                rationale=(
                    "The direct response covered at least as many modeled interpretations as "
                    "the best simulated clarification."
                ),
            )
        )
        counters = _usage_counters(calls)
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "future_turn_interpretations",
                    {
                        "count": len(interpretations),
                        "direct_supported": len(direct_ids),
                        "direct_utility": direct_utility,
                    },
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "future_outcome_preference",
                    {
                        "candidate_count": len(candidates),
                        "candidate_utilities": [item.utility for item in candidates],
                        "selected_question": best.question if should_clarify else "",
                        "clarification_selected": should_clarify,
                    },
                ),
            ),
            counters=counters,
            warnings=(
                "controlled_inference_time_adapter_not_released_trained_checkpoint",
                "future_outcomes_are_model_simulations_not_hidden_user_labels",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=trace,
            implementation_identity=(f"future_turn_outcome_match@{FUTURE_TURNS_UPSTREAM_COMMIT}"),
            profile_kind="controlled_adapter",
            metadata={
                "protocol": FUTURE_TURNS_PROMPT_PROTOCOL,
                "upstream_commit": FUTURE_TURNS_UPSTREAM_COMMIT,
                "interpretation_count": len(interpretations),
                "direct_utility": direct_utility,
                "candidate_utilities": [item.utility for item in candidates],
                "clarification_selected": should_clarify,
                "official_checkpoint_used": False,
                "paper_result_parity_claimed": False,
                "backend_fingerprint": dict(backend.fingerprint()),
            },
        )

    def _simulate_candidate(
        self,
        backend: LLMBackend,
        *,
        state: FrozenAgentState,
        interpretations: tuple[str, ...],
        question: str,
        seed: int,
        calls: list[dict[str, Any]],
    ) -> FutureTurnCandidate:
        response = self._call(
            backend,
            system=(
                "Simulate the next user turn and the tailored assistant response for each "
                "possible interpretation. A clarification answer must not reveal the final "
                "answer verbatim. Return JSON only."
            ),
            prompt=(
                f"{visible_problem(state)}\n\n"
                f"Interpretations:\n{_numbered(interpretations)}\n\n"
                f"Clarifying question: {question}\n\n"
                'Return {"outcomes":[{"interpretation_id":0,'
                '"clarification_answer":"...","tailored_response":"..."},...]}. '
                "Include every interpretation exactly once. Use empty strings when a "
                "future answer or tailored response cannot be produced. "
                "Keep each clarification_answer under 12 words and each tailored_response "
                "under 24 words so the JSON object is complete within the output budget."
            ),
            seed=seed,
            phase="simulate_future_turn",
            calls=calls,
        )
        raw = response.get("outcomes")
        if not isinstance(raw, list):
            raise AlgorithmOutputError("future-turn outcomes must be a JSON array")
        indexed: dict[int, FutureTurnOutcome] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise AlgorithmOutputError("future-turn outcome entries must be objects")
            identifier = item.get("interpretation_id")
            if isinstance(identifier, bool) or not isinstance(identifier, int):
                raise AlgorithmOutputError("future-turn interpretation_id must be an integer")
            if identifier in indexed or not 0 <= identifier < len(interpretations):
                raise AlgorithmOutputError("future-turn interpretation_id is invalid or duplicated")
            answer = item.get("clarification_answer")
            tailored = item.get("tailored_response")
            if not isinstance(answer, str) or not isinstance(tailored, str):
                raise AlgorithmOutputError("future-turn simulated answers must be strings")
            clean_answer = answer.strip()
            clean_tailored = tailored.strip()
            # The host derives whether the future turn is usable from required
            # text fields; it never trusts an unconstrained model-authored flag.
            indexed[identifier] = FutureTurnOutcome(
                clean_answer,
                clean_tailored,
                not clean_answer or not clean_tailored,
            )
        if set(indexed) != set(range(len(interpretations))):
            raise AlgorithmOutputError("future-turn outcomes must cover every interpretation")
        outcomes = tuple(indexed[index] for index in range(len(interpretations)))
        useful = [
            item
            for item in outcomes
            if not item.abstain
            and item.clarification_answer
            and item.tailored_response
            and not _leaks_response(item.clarification_answer, item.tailored_response)
        ]
        coverage = len(useful) / len(outcomes)
        diversity = len({_normalize(item.tailored_response) for item in useful}) / len(outcomes)
        return FutureTurnCandidate(
            question=question,
            outcomes=outcomes,
            utility=(coverage + diversity) / 2.0,
        )

    def _call(
        self,
        backend: LLMBackend,
        *,
        system: str,
        prompt: str,
        seed: int,
        phase: str,
        calls: list[dict[str, Any]],
    ) -> Any:
        request = GenerationRequest(
            prompt=prompt,
            system=system,
            temperature=self.controls.temperature,
            max_tokens=self.controls.max_tokens,
            seed=seed,
            response_format={"type": "json_object"},
            metadata={
                "method_id": self.method_id,
                "release_id": self.release_id,
                "phase": phase,
            },
        )
        response = backend.generate(request)
        calls.append(
            {
                "request_hash": request.request_hash,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.usage.cost_usd,
            }
        )
        return parse_json_object(response.text, label=f"future-turn {phase}")


def _integer_ids(value: Any, *, upper: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise AlgorithmOutputError(f"future-turn {label} must be a JSON array")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < upper:
            raise AlgorithmOutputError(f"future-turn {label} contains an invalid index")
        if item not in result:
            result.append(item)
    return tuple(result)


def _normalize(text: str) -> str:
    return re.sub(r"\W+", "", text.casefold())


def _leaks_response(answer: str, response: str) -> bool:
    normalized_answer = _normalize(answer)
    normalized_response = _normalize(response)
    return bool(normalized_response) and normalized_response in normalized_answer


def _numbered(items: tuple[str, ...]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items))


def _usage_counters(calls: list[dict[str, Any]]) -> dict[str, int | float]:
    priced = [item["cost_usd"] for item in calls if item["cost_usd"] is not None]
    return {
        "model_calls": len(calls),
        "input_tokens": sum(int(item["input_tokens"]) for item in calls),
        "output_tokens": sum(int(item["output_tokens"]) for item in calls),
        "cost_usd": sum(float(item) for item in priced),
        "unpriced_model_calls": sum(item["cost_usd"] is None for item in calls),
    }
