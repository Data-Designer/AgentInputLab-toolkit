"""Safe normalization for the upstream ``context_budget`` expression."""

from __future__ import annotations

import math
import re

from agentinputlab.contracts.errors import ValidationError

_DELTA = re.compile(r"^([+-])(\d+)$")
_FACTOR = re.compile(r"^\*(\d+(?:\.\d+)?)$")


def parse_context_budget(
    value: int | str,
    *,
    max_abs_delta: int = 1_000_000,
    max_factor: float = 100.0,
) -> str:
    """Return a canonical expression accepted by the pinned upstream runtime.

    Upstream evaluates ``"target_token" + context_budget``.  This public seam
    accepts only an integer delta, ``+N``, ``-N``, or ``*F`` and therefore never
    forwards arbitrary Python syntax.
    """

    if isinstance(value, bool):
        raise ValidationError("context_budget does not accept booleans")
    if isinstance(value, int):
        if abs(value) > max_abs_delta:
            raise ValidationError("context_budget delta exceeds the safety bound")
        return f"{value:+d}"
    if not isinstance(value, str):
        raise ValidationError("context_budget must be an integer delta or +N/-N/*F")
    compact = value.strip()
    delta = _DELTA.fullmatch(compact)
    if delta:
        amount = int(delta.group(2))
        if amount > max_abs_delta:
            raise ValidationError("context_budget delta exceeds the safety bound")
        return f"{delta.group(1)}{amount}"
    factor = _FACTOR.fullmatch(compact)
    if factor:
        amount = float(factor.group(1))
        if not math.isfinite(amount) or not 0 < amount <= max_factor:
            raise ValidationError("context_budget factor is outside the safety bound")
        return f"*{format(amount, 'g')}"
    raise ValidationError("context_budget permits only integer deltas and +N/-N/*F")
