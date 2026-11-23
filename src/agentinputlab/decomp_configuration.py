"""File binding for the existing, source-parameter-preserving DecomP worker."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

from agentinputlab.contracts.errors import ValidationError


def validate_runtime(release_id, value):
    from agentinputlab.configuration import _known

    if release_id not in {"plan.decomp.full", "plan.decomp.planner_component"}:
        raise ValidationError("decomp_live is supported only by DecomP full/planner_component")
    _known(value, {"kind", "checkout", "python_executable", "nltk_data",
                   "per_call_output_cap", "per_call_input_token_bound", "timeout_seconds"}, "runtime")
    for key in ("checkout", "python_executable", "nltk_data"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValidationError(f"runtime.{key} must be a non-empty path")
    for key in ("per_call_output_cap", "per_call_input_token_bound"):
        number = value.get(key, 512 if key == "per_call_output_cap" else None)
        if number is None and key == "per_call_input_token_bound":
            continue
        if type(number) is not int or number < 1:
            raise ValidationError(f"runtime.{key} must be a positive integer")
    timeout = value.get("timeout_seconds", 300)
    try:
        valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError("runtime.timeout_seconds must be positive and finite")


def bind_case(config, context, state):
    from agentinputlab.integrations.external_workers.decomp_live import DecompLiveRuntime
    from agentinputlab.integrations.external_workers.decomp_planner import DecompPlannerLiveRuntime

    value = config.runtime
    validate_runtime(config.release_id, value)
    if context.external_runtime is not None:
        raise ValidationError("external runtime is configured twice")
    if not callable(getattr(context.backend, "complete_native", None)):
        raise ValidationError("decomp_live requires a native completion backend, not a chat adapter")
    interpreter = Path(value["python_executable"])
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValidationError("DecomP worker Python is missing or not executable; prepare the isolated worker environment")
    for key in ("checkout", "nltk_data"):
        if not Path(value[key]).is_dir():
            raise ValidationError(f"DecomP {key} directory is missing; prepare the pinned worker assets")
    runtime_type = DecompLiveRuntime if config.release_id == "plan.decomp.full" else DecompPlannerLiveRuntime
    runtime = runtime_type(python_executable=value["python_executable"],
        checkout=Path(value["checkout"]), nltk_data=Path(value["nltk_data"]), backend=context.backend,
        per_call_output_cap=value.get("per_call_output_cap", 512),
        per_call_input_token_bound=value.get("per_call_input_token_bound"),
        timeout_seconds=value.get("timeout_seconds", 300))
    return replace(context, external_runtime=runtime)
