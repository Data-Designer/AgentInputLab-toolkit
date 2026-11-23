"""Runtime availability is separate from source-audit status."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import StrEnum


class RunnerKind(StrEnum):
    CORE = "core"
    OPTIONAL_EXTRA = "optional_extra"
    EXTERNAL_WORKER = "external_worker"
    REPLAY_ONLY = "replay_only"


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    available: bool
    runner_kind: RunnerKind
    missing_extras: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    upstream_revision: str | None = None
    reason: str = ""
