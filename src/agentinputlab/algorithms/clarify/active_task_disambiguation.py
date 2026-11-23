"""Expected-information-gain clarification from Active Task Disambiguation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentinputlab.backends.llm import GenerationRequest, LLMBackend, ModelCapability
from agentinputlab.backends.llm.base import require_capabilities
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
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
from .future_turns import _usage_counters

ACTIVE_TASK_DISAMBIGUATION_UPSTREAM_COMMIT = "4c8ecb4d4ffdbffcc611366743fc1e2461037772"
ACTIVE_TASK_DISAMBIGUATION_RELEASE = "clarify.active_task_disambiguation.eig_uniform"
ACTIVE_TASK_DISAMBIGUATION_PROTOCOL = "agentinputlab.active-task-disambiguation.v1"
ACTIVE_TASK_RESPONSE_ADAPTER = "unambiguous-text-records.v1"


@dataclass(frozen=True, slots=True)
class ActiveDisambiguationControls:
    total_hypotheses: int = 8
    total_questions: int = 4
    max_tokens: int = 768
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if not 2 <= self.total_hypotheses <= 20:
            raise ValidationError("active disambiguation hypotheses must be in [2, 20]")
        if not 1 <= self.total_questions <= 8:
            raise ValidationError("active disambiguation questions must be in [1, 8]")
        if self.max_tokens < 128:
            raise ValidationError("active disambiguation max_tokens must be at least 128")
        if self.temperature < 0:
            raise ValidationError("active disambiguation temperature must be non-negative")


@dataclass(slots=True)
class ActiveTaskDisambiguation:
    """Sample hypotheses and select the question with maximum empirical EIG."""

    backend: LLMBackend | None = None
    controls: ActiveDisambiguationControls = ActiveDisambiguationControls()
    method_id: str = "active_task_disambiguation"
    release_id: str = ACTIVE_TASK_DISAMBIGUATION_RELEASE
    family: SemanticFamily = SemanticFamily.CLARIFY

    def __post_init__(self) -> None:
        if not isinstance(self.controls, ActiveDisambiguationControls):
            raise ValidationError(
                "active disambiguation controls must be ActiveDisambiguationControls"
            )

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        backend = self.backend or context.backend
        if backend is None:
            raise ExternalDependencyError("active task disambiguation requires an LLM backend")
        require_capabilities(
            backend,
            {ModelCapability.CHAT, ModelCapability.JSON_OBJECT, ModelCapability.SEED},
        )
        per_call_output_cap = _fair_per_call_output_cap(
            state,
            requested=self.controls.max_tokens,
        )
        calls: list[dict[str, Any]] = []
        normalizations: list[dict[str, Any]] = []
        hypotheses_payload = self._call(
            backend,
            system=(
                "Sample diverse and representative solutions or user-intent hypotheses that "
                "remain compatible with the visible problem statement. Return JSON only."
            ),
            prompt=(
                f"{visible_problem(state)}\n\n"
                f'Return {{"hypotheses":[...]}} with exactly '
                f"{self.controls.total_hypotheses} concise hypotheses."
            ),
            seed=context.seed,
            phase="sample_hypotheses",
            calls=calls,
            max_tokens=per_call_output_cap,
        )
        hypotheses = _atd_string_list(
            hypotheses_payload.get("hypotheses"),
            label="active-disambiguation hypotheses",
            minimum=2,
            maximum=self.controls.total_hypotheses,
            text_fields=("text", "description"),
            normalizations=normalizations,
        )
        question_payload = self._call(
            backend,
            system=(
                "Generate questions whose possible answers distinguish the compatible "
                "hypotheses. Return JSON only."
            ),
            prompt=(
                f"{visible_problem(state)}\n\nHypotheses:\n{_numbered(hypotheses)}\n\n"
                f'Return {{"questions":[...]}} with exactly '
                f"{self.controls.total_questions} concise questions."
            ),
            seed=context.seed + 1,
            phase="sample_questions",
            calls=calls,
            max_tokens=per_call_output_cap,
        )
        questions = _atd_string_list(
            question_payload.get("questions"),
            label="active-disambiguation questions",
            minimum=1,
            maximum=self.controls.total_questions,
            text_fields=("text",),
            normalizations=normalizations,
        )
        entropies: list[float] = []
        partitions: list[tuple[str, ...]] = []
        for index, question in enumerate(questions):
            payload = self._call(
                backend,
                system=(
                    "Predict the categorical user answer to one candidate clarifying question "
                    "under every supplied hypothesis. Use the same short category string for "
                    "semantically equivalent answers. Return JSON only."
                ),
                prompt=(
                    f"Hypotheses:\n{_numbered(hypotheses)}\n\nQuestion: {question}\n\n"
                    'Return {"answers":[{"hypothesis_id":0,'
                    '"answer_category":"..."},...]}; cover every hypothesis once. '
                    "Use answer categories of at most six words."
                ),
                seed=context.seed + 2 + index,
                phase="partition_hypotheses",
                calls=calls,
                max_tokens=per_call_output_cap,
            )
            categories = _parse_partition(payload, expected=len(hypotheses))
            partitions.append(categories)
            entropies.append(empirical_entropy(categories))
        selected_index = max(range(len(questions)), key=entropies.__getitem__)
        selected_question = questions[selected_index]
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "sampled_compatible_hypotheses",
                    {"count": len(hypotheses)},
                ),
                TraceEvent(
                    TracePhase.PROPOSE,
                    "estimated_question_information_gain",
                    {
                        "question_count": len(questions),
                        "entropies": entropies,
                        "selected_index": selected_index,
                        "selected_question": selected_question,
                    },
                ),
            ),
            counters=_usage_counters(calls),
            warnings=(
                "controlled_general_task_adapter_of_released_code_generation_runtime",
                "hypothesis_answers_are_model_simulations_not_hidden_user_labels",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REQUEST_CLARIFICATION,
                    selected_question,
                    target="user",
                    rationale="Maximum uniform empirical information gain over sampled hypotheses.",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"eig_uniform_solution_partition@{ACTIVE_TASK_DISAMBIGUATION_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "protocol": ACTIVE_TASK_DISAMBIGUATION_PROTOCOL,
                "upstream_commit": ACTIVE_TASK_DISAMBIGUATION_UPSTREAM_COMMIT,
                "hypothesis_count": len(hypotheses),
                "question_count": len(questions),
                "entropies": entropies,
                "selected_question_index": selected_index,
                "logprob_weighting": False,
                "paper_result_parity_claimed": False,
                "effective_max_tokens_per_call": per_call_output_cap,
                "response_adapter_revision": ACTIVE_TASK_RESPONSE_ADAPTER,
                "response_normalizations": normalizations,
                "backend_fingerprint": dict(backend.fingerprint()),
            },
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
        max_tokens: int,
    ) -> Any:
        request = GenerationRequest(
            prompt=prompt,
            system=system,
            temperature=self.controls.temperature,
            max_tokens=max_tokens,
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
                "phase": phase,
                "finish_reason": response.finish_reason,
                "response_sha256": (
                    hashlib.sha256(response.text.encode("utf-8")).hexdigest()
                    if isinstance(response.text, str) else None
                ),
            }
        )
        try:
            return parse_json_object(response.text, label=f"active disambiguation {phase}")
        except AlgorithmOutputError as exc:
            # A paid empty/truncated response remains a failure, never a retry or
            # a license to use reasoning_content as the final answer.
            reason = response.finish_reason
            safe_reason = (
                reason
                if isinstance(reason, str) and reason in {"stop", "length", "content_filter"}
                else "other"
            )
            raise AlgorithmOutputError(
                f"{exc} [finish_reason={safe_reason}; "
                f"output_tokens={response.usage.output_tokens}; max_tokens={max_tokens}]"
            ) from exc


def _atd_string_list(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
    text_fields: tuple[str, ...],
    normalizations: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Unwrap only unambiguous text records; do not repair generated content.

    Saved O20 responses use {text: ...} questions and {id, description}
    hypotheses. IDs are display metadata: subsequent partition prompts use
    freshly enumerated positions, exactly as for the original string array.
    Unknown fields, conflicting text fields and duplicate IDs are rejected.
    """
    if not isinstance(value, list):
        return string_list(value, label=label, minimum=minimum, maximum=maximum)
    unwrapped = []
    identifiers: set[str | int] = set()
    normalized_count = 0
    for item in value:
        if isinstance(item, Mapping):
            present = [key for key in text_fields if key in item]
            if len(present) != 1 or set(item) - {present[0], "id"}:
                raise AlgorithmOutputError(f"{label} text records must be unambiguous")
            if "id" in item:
                identifier = item["id"]
                if (
                    isinstance(identifier, bool)
                    or not isinstance(identifier, (str, int))
                    or (isinstance(identifier, str) and not identifier.strip())
                    or identifier in identifiers
                ):
                    raise AlgorithmOutputError(f"{label} text-record IDs must be unique scalars")
                identifiers.add(identifier)
            item = item[present[0]]
            normalized_count += 1
        unwrapped.append(item)
    result = string_list(unwrapped, label=label, minimum=minimum, maximum=maximum)
    if normalized_count:
        normalizations.append({"label": label, "text_records_unwrapped": normalized_count})
    return result


def _fair_per_call_output_cap(state: FrozenAgentState, *, requested: int) -> int:
    """Keep every optimizer call inside the visible shared control envelope.

    The official visible state carries cumulative output and call ceilings but
    not the paper runner's separate per-call field.  Reserving an equal share
    for every permitted model call reconstructs the standard 4096 / 8 = 512
    cap and prevents this multi-call method from consuming the arena's budget
    before official execution begins.
    """

    budget = state.budget
    if budget.max_model_calls < 1 or budget.max_output_tokens < 1:
        raise BudgetExceededError("active disambiguation has no visible model-output budget")
    return min(requested, max(1, budget.max_output_tokens // budget.max_model_calls))


def _parse_partition(payload: Any, *, expected: int) -> tuple[str, ...]:
    raw = payload.get("answers")
    if not isinstance(raw, list):
        raise AlgorithmOutputError("active-disambiguation answers must be a JSON array")
    indexed: dict[int, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise AlgorithmOutputError("active-disambiguation answers must be objects")
        identifier = item.get("hypothesis_id")
        category = item.get("answer_category")
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise AlgorithmOutputError("active-disambiguation hypothesis_id must be integer")
        if identifier in indexed or not 0 <= identifier < expected:
            raise AlgorithmOutputError(
                "active-disambiguation hypothesis_id is invalid or duplicated"
            )
        if not isinstance(category, str) or not category.strip():
            raise AlgorithmOutputError("active-disambiguation answer_category must be text")
        indexed[identifier] = category.strip().casefold()
    if set(indexed) != set(range(expected)):
        raise AlgorithmOutputError(
            "active-disambiguation answer partition must cover every hypothesis"
        )
    return tuple(indexed[index] for index in range(expected))


def empirical_entropy(categories: tuple[str, ...]) -> float:
    if not categories:
        raise ValidationError("entropy requires at least one answer category")
    total = len(categories)
    return -sum((count / total) * math.log(count / total) for count in Counter(categories).values())


def _numbered(items: tuple[str, ...]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items))
