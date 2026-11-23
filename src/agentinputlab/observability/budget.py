"""Atomic resource accounting and user-supplied model price estimation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from threading import RLock
from types import MappingProxyType

from agentinputlab.contracts.errors import BudgetExceededError, ValidationError
from agentinputlab.contracts.state import ExecutionBudget

from .models import ResourceUsage


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per one million input/output tokens for a pinned deployment."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        values = (self.input_per_million, self.output_per_million)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            or not math.isfinite(value)
            for value in values
        ):
            raise ValidationError("model prices must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ComputePrice:
    """Pinned effective infrastructure rates for self-hosted experiments."""

    cpu_per_hour: float = 0.0
    accelerator_per_hour: float = 0.0

    def __post_init__(self) -> None:
        values = (self.cpu_per_hour, self.accelerator_per_hour)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            or not math.isfinite(value)
            for value in values
        ):
            raise ValidationError("compute prices must be finite and non-negative")

    def estimate(self, *, cpu_seconds: float, accelerator_seconds: float) -> float:
        return (
            cpu_seconds * self.cpu_per_hour + accelerator_seconds * self.accelerator_per_hour
        ) / 3600


@dataclass(frozen=True, slots=True)
class PricingCatalog:
    """No rolling vendor prices are baked into the library.

    Experiment manifests must provide the price attached to their pinned model
    deployment, keeping historical cost reports reproducible.
    """

    prices: Mapping[str, ModelPrice] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(value, ModelPrice) for value in self.prices.values()):
            raise ValidationError("pricing catalog values must be ModelPrice instances")
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))

    def estimate(self, model_id: str, *, input_tokens: int, output_tokens: int) -> float:
        price = self.prices.get(model_id)
        if price is None:
            return 0.0
        return (
            input_tokens * price.input_per_million + output_tokens * price.output_per_million
        ) / 1_000_000

    def has_price(self, model_id: str) -> bool:
        return model_id in self.prices


class BudgetLedger:
    """Thread-safe ledger separating reservations from irreversible consumption.

    ``reserve`` is used before a call with side effects.  A rejected reservation
    leaves usage unchanged.  ``charge_actual`` is used after a response has
    already consumed tokens or money: it always records that usage and makes any
    resulting violation sticky so an intervention cannot hide it by catching the
    immediate :class:`BudgetExceededError`.

    ``charge`` remains a backwards-compatible alias for ``reserve``.
    """

    def __init__(self, budget: ExecutionBudget) -> None:
        if not isinstance(budget, ExecutionBudget):
            raise TypeError("budget must be an ExecutionBudget")
        self.budget = budget
        self._usage = ResourceUsage()
        self._reservation_rejections: tuple[str, ...] = ()
        self._actual_violations: tuple[str, ...] = ()
        self._pending_usage = ResourceUsage()
        self._owner_run_id: str | None = None
        self._lock = RLock()

    @property
    def owner_run_id(self) -> str | None:
        """Monitored run that exclusively owns this ledger, if claimed."""

        with self._lock:
            return self._owner_run_id

    @property
    def is_pristine(self) -> bool:
        """Whether no run has claimed or charged this injectable ledger."""

        with self._lock:
            return self._is_pristine_unlocked()

    def claim_for_run(self, run_id: str) -> None:
        """Bind a pristine ledger to exactly one monitored run.

        A caller may inject a ledger to inspect accounting in a test, but a
        ledger is mutable run state rather than reusable configuration.  The
        one-way claim prevents sequential or concurrent runs from sharing
        counters and therefore turning a per-run budget into a cross-run one.
        """

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValidationError("run_id must be a non-empty string")
        with self._lock:
            if self._owner_run_id is not None:
                raise ValidationError(
                    "BudgetLedger is already owned by a monitored run; "
                    "construct a fresh BudgetLedger for every run"
                )
            if not self._is_pristine_unlocked():
                raise ValidationError(
                    "an injected BudgetLedger must be pristine; construct a fresh "
                    "BudgetLedger for every run"
                )
            self._owner_run_id = run_id

    def _is_pristine_unlocked(self) -> bool:
        return (
            self._owner_run_id is None
            and self._usage == ResourceUsage()
            and self._pending_usage == ResourceUsage()
            and not self._reservation_rejections
            and not self._actual_violations
        )

    @property
    def usage(self) -> ResourceUsage:
        with self._lock:
            return self._usage

    @property
    def actual_violations(self) -> tuple[str, ...]:
        """Budget dimensions ever exceeded by already-consumed resources."""

        with self._lock:
            return self._actual_violations

    @property
    def reservation_rejections(self) -> tuple[str, ...]:
        """Budget dimensions for calls denied before external side effects."""

        with self._lock:
            return self._reservation_rejections

    def reserve(self, delta: ResourceUsage) -> ResourceUsage:
        """Atomically reserve prospective usage, rejecting without mutation."""

        if not isinstance(delta, ResourceUsage):
            raise TypeError("delta must be a ResourceUsage")
        with self._lock:
            current = self._usage + delta
            violations = self._violations(current)
            if violations:
                self._reservation_rejections = tuple(
                    dict.fromkeys((*self._reservation_rejections, *violations))
                )
                raise BudgetExceededError(
                    "execution budget reservation rejected: " + ", ".join(violations)
                )
            self._usage = current
            return current

    def reserve_pending(self, delta: ResourceUsage) -> ResourceUsage:
        """Reserve a lower-bound call plan that may later be committed or cancelled."""

        if not isinstance(delta, ResourceUsage):
            raise TypeError("delta must be a ResourceUsage")
        with self._lock:
            current = self._usage + delta
            violations = self._violations(current)
            if violations:
                self._reservation_rejections = tuple(
                    dict.fromkeys((*self._reservation_rejections, *violations))
                )
                raise BudgetExceededError(
                    "execution budget reservation rejected: " + ", ".join(violations)
                )
            self._usage = current
            self._pending_usage = self._pending_usage + delta
            return current

    def commit_pending(self, delta: ResourceUsage) -> ResourceUsage:
        """Mark a pending lower-bound reservation as confirmed actual usage."""

        if not isinstance(delta, ResourceUsage):
            raise TypeError("delta must be a ResourceUsage")
        with self._lock:
            self._pending_usage = _subtract_usage(
                self._pending_usage,
                delta,
                label="pending reservation",
            )
            return self._usage

    def cancel_pending(self, delta: ResourceUsage) -> ResourceUsage:
        """Cancel a reservation when a typed worker failure reports no consumption."""

        if not isinstance(delta, ResourceUsage):
            raise TypeError("delta must be a ResourceUsage")
        with self._lock:
            self._pending_usage = _subtract_usage(
                self._pending_usage,
                delta,
                label="pending reservation",
            )
            self._usage = _subtract_usage(self._usage, delta, label="recorded usage")
            return self._usage

    def charge(self, delta: ResourceUsage) -> ResourceUsage:
        """Backwards-compatible alias for :meth:`reserve`."""

        return self.reserve(delta)

    def charge_actual(self, delta: ResourceUsage) -> ResourceUsage:
        """Record irreversible consumption, retaining overages before raising."""

        if not isinstance(delta, ResourceUsage):
            raise TypeError("delta must be a ResourceUsage")
        with self._lock:
            current = self._usage + delta
            self._usage = current
            violations = self._violations(current)
            if violations:
                self._actual_violations = tuple(
                    dict.fromkeys((*self._actual_violations, *violations))
                )
                raise BudgetExceededError(
                    "actual execution budget exceeded: " + ", ".join(violations)
                )
            if self._actual_violations:
                raise BudgetExceededError(
                    "actual execution budget was previously exceeded: "
                    + ", ".join(self._actual_violations)
                )
            return current

    def _violations(self, usage: ResourceUsage) -> tuple[str, ...]:
        violations: list[str] = []
        if usage.model_calls > self.budget.max_model_calls:
            violations.append("model_calls")
        if usage.retrieval_calls > self.budget.max_retrieval_calls:
            violations.append("retrieval_calls")
        if usage.tool_calls > self.budget.max_tool_calls:
            violations.append("tool_calls")
        if usage.output_tokens > self.budget.max_output_tokens:
            violations.append("output_tokens")
        if (
            self.budget.max_context_tokens is not None
            and usage.input_tokens > self.budget.max_context_tokens
        ):
            violations.append("context_tokens")
        if self.budget.max_cost_usd is not None and usage.total_cost_usd > self.budget.max_cost_usd:
            violations.append("cost_usd")
        if self.budget.max_cost_usd is not None and usage.unpriced_model_calls:
            violations.append("unknown_cost")
        return tuple(violations)

    def require_within_budget(self) -> None:
        with self._lock:
            violations = tuple(
                dict.fromkeys(
                    (
                        *self._reservation_rejections,
                        *self._actual_violations,
                        *self._violations(self._usage),
                    )
                )
            )
        if violations:
            raise BudgetExceededError("execution budget exceeded: " + ", ".join(violations))


def _subtract_usage(
    current: ResourceUsage,
    delta: ResourceUsage,
    *,
    label: str,
) -> ResourceUsage:
    values: dict[str, int | float] = {}
    for declared in fields(ResourceUsage):
        difference = getattr(current, declared.name) - getattr(delta, declared.name)
        if difference < 0:
            raise ValidationError(f"cannot release more than the {label}")
        values[declared.name] = difference
    return ResourceUsage(**values)
