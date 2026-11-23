"""Generic native-parity harness for pinned upstream adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentinputlab.contracts.enums import StrEnum
from agentinputlab.contracts.serialization import (
    content_hash,
    deep_freeze_jsonable,
    freeze_mapping,
)


class ParityMode(StrEnum):
    FIXTURE_EXACT = "fixture_exact"
    NUMERIC_RESULT = "numeric_result"
    DIRECTION = "direction"


@dataclass(frozen=True, slots=True)
class ParityCase:
    case_id: str
    input_payload: Mapping[str, Any]
    expected: Any
    baseline: float | None = None
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("parity case requires a non-empty ID")
        tolerances = (self.absolute_tolerance, self.relative_tolerance)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in tolerances
        ):
            raise ValueError("parity tolerances must be finite and non-negative")
        object.__setattr__(self, "input_payload", freeze_mapping(self.input_payload))
        object.__setattr__(self, "expected", deep_freeze_jsonable(self.expected))
        content_hash(self.expected)


@dataclass(frozen=True, slots=True)
class ParityObservation:
    case_id: str
    passed: bool
    expected_hash: str
    actual_hash: str
    expected_numeric: float | None = None
    actual_numeric: float | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ParityReport:
    method_id: str
    release_id: str
    mode: ParityMode
    reference_identity: str
    candidate_identity: str
    observations: tuple[ParityObservation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.observations:
            raise ValueError("parity report requires at least one observation")

    @property
    def passed(self) -> bool:
        return all(observation.passed for observation in self.observations)

    @property
    def passed_count(self) -> int:
        return sum(observation.passed for observation in self.observations)

    @property
    def report_hash(self) -> str:
        return content_hash(self)

    def require_passed(self) -> None:
        if not self.passed:
            failed = ", ".join(
                observation.case_id for observation in self.observations if not observation.passed
            )
            raise AssertionError(f"native parity failed for cases: {failed}")


class ParityRunner:
    def __init__(
        self,
        *,
        method_id: str,
        release_id: str,
        mode: ParityMode,
        reference_identity: str,
        candidate_identity: str,
    ) -> None:
        if not method_id or not release_id:
            raise ValueError("parity method and release IDs must be non-empty")
        _validate_pinned_identity(reference_identity, label="reference")
        _validate_pinned_identity(candidate_identity, label="candidate")
        if reference_identity == candidate_identity:
            raise ValueError("parity reference and candidate identities must differ")
        self.method_id = method_id
        self.release_id = release_id
        self.mode = mode
        self.reference_identity = reference_identity
        self.candidate_identity = candidate_identity

    def run(
        self,
        cases: Sequence[ParityCase],
        candidate: Callable[[Mapping[str, Any]], Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ParityReport:
        rows = tuple(cases)
        if not rows:
            raise ValueError("parity run requires cases")
        if len({case.case_id for case in rows}) != len(rows):
            raise ValueError("parity case IDs must be unique")
        observations = tuple(self._observe(case, candidate(case.input_payload)) for case in rows)
        return ParityReport(
            method_id=self.method_id,
            release_id=self.release_id,
            mode=self.mode,
            reference_identity=self.reference_identity,
            candidate_identity=self.candidate_identity,
            observations=observations,
            metadata=metadata or {},
        )

    def _observe(self, case: ParityCase, actual: Any) -> ParityObservation:
        expected_hash = content_hash(case.expected)
        actual_hash = content_hash(actual)
        if self.mode is ParityMode.FIXTURE_EXACT:
            return ParityObservation(
                case_id=case.case_id,
                passed=expected_hash == actual_hash,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                detail="canonical fixture hash comparison",
            )
        expected = _finite_float(case.expected, name="expected")
        observed = _finite_float(actual, name="actual")
        if self.mode is ParityMode.NUMERIC_RESULT:
            passed = math.isclose(
                observed,
                expected,
                rel_tol=case.relative_tolerance,
                abs_tol=case.absolute_tolerance,
            )
            detail = "numeric result within tolerance" if passed else "numeric tolerance exceeded"
        elif self.mode is ParityMode.DIRECTION:
            if case.baseline is None or not math.isfinite(case.baseline):
                raise ValueError("direction parity requires a finite baseline")
            passed = _sign(expected - case.baseline) == _sign(observed - case.baseline)
            detail = "effect direction agrees" if passed else "effect direction disagrees"
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unknown parity mode: {self.mode}")
        return ParityObservation(
            case_id=case.case_id,
            passed=passed,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            expected_numeric=expected,
            actual_numeric=observed,
            detail=detail,
        )


def _finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"parity {name} must be finite")
    return number


def _sign(value: float) -> int:
    return int(value > 0) - int(value < 0)


def _validate_pinned_identity(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"parity {label} identity must be a non-empty pinned name@revision")
    name, separator, revision = value.rpartition("@")
    if (
        not separator
        or not name
        or len(revision) < 7
        or revision.casefold() in {"latest", "main", "master"}
    ):
        raise ValueError(f"parity {label} identity must use a pinned name@revision")
