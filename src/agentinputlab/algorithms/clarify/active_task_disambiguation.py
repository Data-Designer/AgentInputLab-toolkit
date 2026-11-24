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
    CapabilityError,
    ExecutionBudget,
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
_JOINT_LAYOUT_ID = "atd.hypothesis-then-joint-question-grid.v1"
_JOINT_CAPS = {"sample_hypotheses": 512, "partition_hypotheses": 1024}
_JOINT_HYPOTHESIS_SYSTEM = (
    "Sample diverse hypotheses about unresolved user-side facts or preferences that would "
    "change a justified next action under the visible request, instructions and tool catalog. "
    "Keep all explicit observations and preferences fixed; treat a user interpretation or "
    "diagnosis as a claim, not an independently verified fact. Do not create alternatives "
    "by contradicting facts the user already supplied. Do not make the user guess database "
    "records, policy eligibility or other facts that the advertised tools can inspect. "
    "Use only the supplied visible information. Return concise JSON only."
)
_JOINT_QUESTION_SYSTEM = (
    "Using the supplied visible facts and hypotheses, write two neutral clarification "
    "questions about unresolved user-known facts or preferences needed for a justified "
    "next action. Do not repeat facts already given or ask users for database facts "
    "that available tools can inspect. Then predict a categorical answer to each "
    "question under each hypothesis. Return only the complete JSON, no explanations."
)


@dataclass(frozen=True, slots=True)
class ActiveDisambiguationControls:
    total_hypotheses: int = 8
    total_questions: int = 4
    max_tokens: int = 768
    temperature: float = 0.7
    request_layout: str = "separate"

    def __post_init__(self) -> None:
        if not 2 <= self.total_hypotheses <= 20:
            raise ValidationError("active disambiguation hypotheses must be in [2, 20]")
        if not 1 <= self.total_questions <= 8:
            raise ValidationError("active disambiguation questions must be in [1, 8]")
        if self.max_tokens < 128:
            raise ValidationError("active disambiguation max_tokens must be at least 128")
        if self.temperature < 0:
            raise ValidationError("active disambiguation temperature must be non-negative")
        if self.request_layout not in ("separate", "joint_h2q2"):
            raise ValidationError("active disambiguation request_layout must be separate or joint_h2q2")
        if self.request_layout == "joint_h2q2" and (
            self.total_hypotheses, self.total_questions, self.max_tokens
        ) != (2, 2, _JOINT_CAPS["sample_hypotheses"]):
            raise ValidationError(
                "joint_h2q2 requires total_hypotheses=2, total_questions=2, and max_tokens=512"
            )


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
        joint = self.controls.request_layout == "joint_h2q2"
        if joint:
            _require_joint_capacity(state.budget, backend, self.method_id)
        per_call_output_cap = (
            _JOINT_CAPS["sample_hypotheses"]
            if joint else _fair_per_call_output_cap(state, requested=self.controls.max_tokens)
        )
        calls: list[dict[str, Any]] = []
        normalizations: list[dict[str, Any]] = []
        hypotheses_payload = self._call(
            backend,
            system=_JOINT_HYPOTHESIS_SYSTEM if joint else (
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
            text_fields=("text", "description", "hypothesis", "statement"),
            normalizations=normalizations,
        )
        if joint:
            _check_joint_usage(state.budget, calls, per_call_output_cap)
            questions, partitions = self._joint_questions(
                backend, state, context.seed, hypotheses, calls, normalizations
            )
        else:
            questions, partitions = self._separate_questions(
                backend, state, context.seed, hypotheses, calls, normalizations, per_call_output_cap
            )
        entropies = [empirical_entropy(categories) for categories in partitions]
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
                **({
                    "request_layout_variant": _JOINT_LAYOUT_ID,
                    "maximum_optimizer_calls": 2,
                    "no_native_policy_change": True,
                } if joint else {}),
            },
        )

    def _separate_questions(
        self,
        backend: LLMBackend,
        state: FrozenAgentState,
        seed: int,
        hypotheses: tuple[str, ...],
        calls: list[dict[str, Any]],
        normalizations: list[dict[str, Any]],
        per_call_output_cap: int,
    ) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
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
            seed=seed + 1,
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
                seed=seed + 2 + index,
                phase="partition_hypotheses",
                calls=calls,
                max_tokens=per_call_output_cap,
            )
            partitions.append(_parse_partition(payload, expected=len(hypotheses)))
        return questions, partitions

    def _joint_questions(
        self,
        backend: LLMBackend,
        state: FrozenAgentState,
        seed: int,
        hypotheses: tuple[str, ...],
        calls: list[dict[str, Any]],
        normalizations: list[dict[str, Any]],
    ) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        cap = _JOINT_CAPS["partition_hypotheses"]
        payload = self._call(
            backend,
            system=_JOINT_QUESTION_SYSTEM,
            prompt=(
                f"{visible_problem(state)}\n\nHypotheses:\n{_numbered(hypotheses)}\n\n"
                'Return {"questions":["short question?","short question?"],'
                '"partitions":[["category","category"],["category","category"]]}. '
                "Exactly two questions, each at most 15 words. Each row corresponds to "
                "one question in order, and its two categories to hypotheses in order. "
                "Within a row use identical categories for equivalent answers. Cover all "
                "positions once; categories at most six words. No rationale fields."
            ),
            seed=seed + 2,
            phase="partition_hypotheses",
            calls=calls,
            max_tokens=cap,
        )
        _check_joint_usage(state.budget, calls, cap)
        questions = _atd_string_list(
            payload.get("questions"),
            label="active-disambiguation questions",
            minimum=2,
            maximum=2,
            text_fields=("text",),
            normalizations=normalizations,
        )
        return questions, _parse_joint_grid(payload)

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


def _require_joint_capacity(budget: ExecutionBudget, backend: LLMBackend, method_id: str) -> None:
    if budget.max_model_calls < 2 or budget.max_output_tokens < sum(_JOINT_CAPS.values()):
        raise BudgetExceededError("joint_h2q2 requires 2 model calls and at least 1536 output tokens")
    # Inspect declared wrappers, including observability/production bindings.
    # An unadvertised provider-side limit cannot be inferred from the model ID.
    pending = [backend.fingerprint()]
    visited: set[int] = set()
    while pending:
        entry = pending.pop()
        if not isinstance(entry, Mapping) or id(entry) in visited:
            continue
        visited.add(id(entry))
        if entry.get("backend") == "per_call_output_cap":
            overrides = entry.get("stage_output_caps", {})
            for phase, needed in _JOINT_CAPS.items():
                key = f"{method_id}:{phase}"
                cap = overrides.get(key, entry.get("max_output_tokens_per_call"))
                if type(cap) is not int or cap < needed:
                    raise CapabilityError(
                        f"joint_h2q2 requires backend output capacity of {needed} for {key}; "
                        "configure the stage cap in every limiting wrapper"
                    )
        pending.extend(entry.get(key) for key in ("inner", "backend"))


def _check_joint_usage(
    budget: ExecutionBudget, calls: list[dict[str, Any]], stage_cap: int
) -> None:
    used = _usage_counters(calls)
    if (calls[-1]["output_tokens"] > stage_cap
        or used["output_tokens"] > budget.max_output_tokens
        or (budget.max_context_tokens is not None and used["input_tokens"] > budget.max_context_tokens)
        or (budget.max_cost_usd is not None and used["cost_usd"] > budget.max_cost_usd)):
        raise BudgetExceededError("joint_h2q2 response exceeded the stage or cumulative budget")


def _parse_joint_grid(payload: Any) -> list[tuple[str, ...]]:
    rows = payload.get("partitions")
    if not isinstance(rows, list) or len(rows) != 2:
        raise AlgorithmOutputError("joint_h2q2 partitions must include one row per question")
    out = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise AlgorithmOutputError("joint_h2q2 each row must cover both hypotheses")
        out.append(_parse_partition(
            {"answers": [
                {"hypothesis_id": index, "answer_category": value}
                for index, value in enumerate(row)
            ]},
            expected=2,
        ))
    return out


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

    Questions use {text: ...}; hypothesis records also accept description,
    hypothesis, or statement as the sole text field. IDs are display metadata:
    subsequent partition prompts use
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
