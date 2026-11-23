"""Intent constraints and evaluator-safe state projection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .enums import PatchOp
from .patch import InputPatch
from .state import FrozenAgentState


@dataclass(frozen=True, slots=True)
class IntentContract:
    """Observable constraints an intervention must preserve.

    The contract intentionally contains no evaluator answer or hidden simulator
    state.  Gold data belongs to the benchmark evaluator process.
    """

    required_entities: tuple[str, ...] = ()
    required_constraints: tuple[str, ...] = ()
    forbidden_assumptions: tuple[str, ...] = ()
    may_request_clarification: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "required_entities",
            "required_constraints",
            "forbidden_assumptions",
        ):
            values = tuple(getattr(self, field_name))
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"{field_name} must contain only non-empty strings")
            if len({value.casefold() for value in values}) != len(values):
                raise ValueError(f"{field_name} must not contain duplicate clauses")
            object.__setattr__(self, field_name, values)
        if not isinstance(self.may_request_clarification, bool):
            raise TypeError("may_request_clarification must be a bool")


@dataclass(frozen=True, slots=True)
class IntentCheck:
    preserved: bool
    missing_entities: tuple[str, ...] = ()
    missing_constraints: tuple[str, ...] = ()
    introduced_assumptions: tuple[str, ...] = ()


def check_intent_contract(
    contract: IntentContract,
    state: FrozenAgentState,
    patch: InputPatch,
    output: FrozenAgentState,
) -> IntentCheck:
    """Apply conservative, observable checks without consulting benchmark gold.

    This is a hard safety gate, not a semantic judge.  It only enforces explicit
    strings supplied by the benchmark adapter and the clarification permission.
    Learned or LLM-based intent metrics belong in secondary evaluation.
    """

    del state
    visible = "\n".join(
        (
            output.query,
            output.instruction,
            *(block.text for block in output.context),
            *(message.content for message in output.messages),
        )
    )
    missing_entities = tuple(
        entity for entity in contract.required_entities if not _contains_phrase(visible, entity)
    )
    missing_constraints = tuple(
        constraint
        for constraint in contract.required_constraints
        if not _contains_phrase(visible, constraint)
    )
    introduced_assumptions = tuple(
        assumption
        for assumption in contract.forbidden_assumptions
        if _contains_phrase(visible, assumption)
    )
    requested_clarification = any(
        operation.op is PatchOp.REQUEST_CLARIFICATION for operation in patch.operations
    )
    clarification_violation = requested_clarification and not contract.may_request_clarification
    if clarification_violation:
        introduced_assumptions = (*introduced_assumptions, "<clarification-not-permitted>")
    return IntentCheck(
        preserved=not missing_entities and not missing_constraints and not introduced_assumptions,
        missing_entities=missing_entities,
        missing_constraints=missing_constraints,
        introduced_assumptions=introduced_assumptions,
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a normalized contiguous token phrase, never an arbitrary substring."""

    visible_tokens = tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
    phrase_tokens = tuple(re.findall(r"\w+", phrase.casefold(), flags=re.UNICODE))
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(
        visible_tokens[index : index + width] == phrase_tokens
        for index in range(len(visible_tokens) - width + 1)
    )
