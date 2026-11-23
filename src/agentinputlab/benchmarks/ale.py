"""Pinned external-transfer bridge for Agents' Last Exam (ALE).

ALE owns sandbox provisioning, agent deployment, hidden-reference staging, and
grading.  AgentInputLab exposes only optimizer-visible task checkpoints and a
typed result boundary.  It does not import task graders or claim atomic paired
replay.  Official runs continue to execute through ALE's own CLI.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import freeze_mapping
from agentinputlab.contracts.state import ContextBlock, Message

from .base import BenchmarkExample, opaque_checkpoint_state_id

ALE_BENCHMARK_ID = "ale.external_transfer"
ALE_PUBLIC_SPLIT = "public"
ALE_OFFICIAL_REPOSITORY = "https://github.com/rdi-berkeley/agents-last-exam"
ALE_OFFICIAL_SOURCE_REVISION = "1e615e456de7cef57706680613cb80ee13c7fc76"
ALE_SOFTWARE_LICENSE = "Apache-2.0"
ALE_DATA_LICENSE = "CC-BY-4.0"
ALE_SOURCE_HASHES = MappingProxyType(
    {
        "LICENSE": "1b69f3cee8b40f3a1b7b4db3341c8b739bab905c9575f89e557e04d7c31a169f",
        "LICENSE-DATA": "bd01b9a2a7413566f3189eb89b8f0a5af19bd9075638260d1b0b56ad48fd643c",
        "ale_run/__main__.py": ("945f001509af2bd72eda35249c4a533de54507dcb6006a44e2ab4c00a063df68"),
        "ale_run/orchestration/runner.py": (
            "25ae61390701a5cd5accefe217c9695165877b138558d278a98102eacc7e67e3"
        ),
        "pyproject.toml": "7fe312e5a4a7ba8cc2084827dea3ba0d76cdcfefb29c1168d3c612607cc5b177",
    }
)
ALE_PAIRING_MODE = "external_independent_runs"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CALLER_ASSERTED = "caller_asserted"
_EXTERNAL_BRIDGE = "external_transfer"
_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)


@dataclass(frozen=True, slots=True)
class ALEFamilyEligibility:
    """Family support derived only from the visible pre-run checkpoint."""

    checkpoint_id: str
    eligibility: Mapping[str, bool]
    reasons: Mapping[str, str]
    provenance_status: str = _CALLER_ASSERTED
    support_status: str = "external_transfer_candidate"

    def __post_init__(self) -> None:
        expected = {family.value for family in _FAMILIES}
        if set(self.eligibility) != expected or set(self.reasons) != expected:
            raise ValidationError("ALE family eligibility must cover all five family labels")
        if any(not isinstance(value, bool) for value in self.eligibility.values()):
            raise ValidationError("ALE family eligibility values must be boolean")
        if any(not isinstance(value, str) or not value.strip() for value in self.reasons.values()):
            raise ValidationError("ALE family eligibility reasons must be non-empty")
        if self.eligibility[SemanticFamily.CLARIFY.value]:
            raise ValidationError("ALE has no standardized clarification-user oracle")
        object.__setattr__(self, "eligibility", freeze_mapping(self.eligibility))
        object.__setattr__(self, "reasons", freeze_mapping(self.reasons))

    @property
    def eligible_families(self) -> tuple[SemanticFamily, ...]:
        return tuple(family for family in _FAMILIES if self.eligibility[family.value])


@dataclass(frozen=True, slots=True)
class ALECheckpoint:
    """Optimizer-visible ALE task state bound to opaque official-run identities."""

    checkpoint_id: str
    task_id: str
    domain: str
    query: str
    instruction: str
    visible_messages: tuple[Message, ...]
    visible_tool_names: tuple[str, ...]
    visible_read_tool_names: tuple[str, ...]
    run_handle: str = field(repr=False)
    task_source_sha256: str = field(repr=False)
    environment_snapshot_sha256: str = field(repr=False)
    grader_sha256: str = field(repr=False)
    agent_harness_sha256: str = field(repr=False)
    provider_identity_sha256: str = field(repr=False)
    image_identity_sha256: str = field(repr=False)
    visible_context: tuple[ContextBlock, ...] = ()
    source_revision: str = ALE_OFFICIAL_SOURCE_REVISION
    split: str = ALE_PUBLIC_SPLIT

    def __post_init__(self) -> None:
        for label in (
            "checkpoint_id",
            "task_id",
            "domain",
            "query",
            "instruction",
            "run_handle",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(f"ALE {label} must be trimmed non-empty text")
        if self.source_revision != ALE_OFFICIAL_SOURCE_REVISION:
            raise ValidationError("ALE checkpoint source revision differs from the audit lock")
        if self.split != ALE_PUBLIC_SPLIT:
            raise ValidationError("ALE bridge currently accepts only the public task split")
        for label in (
            "task_source_sha256",
            "environment_snapshot_sha256",
            "grader_sha256",
            "agent_harness_sha256",
            "provider_identity_sha256",
            "image_identity_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, label)) is None:
                raise ValidationError(f"ALE {label} must be a lowercase SHA-256")
        messages = tuple(self.visible_messages)
        context = tuple(self.visible_context)
        tools = _validated_names(self.visible_tool_names, label="visible tool names")
        read_tools = _validated_names(
            self.visible_read_tool_names,
            label="visible read-tool names",
            allow_empty=True,
        )
        if not messages or not all(isinstance(item, Message) for item in messages):
            raise ValidationError("ALE checkpoint requires captured visible messages")
        if not all(isinstance(item, ContextBlock) for item in context):
            raise ValidationError("ALE visible_context must contain ContextBlock instances")
        user_messages = [message for message in messages if message.role == "user"]
        if not user_messages or user_messages[-1].content != self.query:
            raise ValidationError("ALE query must equal the latest visible user message")
        if not set(read_tools) <= set(tools):
            raise ValidationError("ALE visible read tools must be a subset of visible tools")
        object.__setattr__(self, "visible_messages", messages)
        object.__setattr__(self, "visible_context", context)
        object.__setattr__(self, "visible_tool_names", tools)
        object.__setattr__(self, "visible_read_tool_names", read_tools)

    @property
    def family_eligibility(self) -> ALEFamilyEligibility:
        acquire = bool(self.visible_read_tool_names)
        curate = bool(self.visible_context) or len(self.visible_messages) >= 3
        eligibility = {
            SemanticFamily.REFORMULATE.value: True,
            SemanticFamily.DECOMPOSE.value: True,
            SemanticFamily.ACQUIRE.value: acquire,
            SemanticFamily.CLARIFY.value: False,
            SemanticFamily.CURATE.value: curate,
        }
        reasons = {
            SemanticFamily.REFORMULATE.value: "the official task exposes a visible instruction",
            SemanticFamily.DECOMPOSE.value: "the visible long-horizon task can be decomposed",
            SemanticFamily.ACQUIRE.value: (
                "the checkpoint exposes a caller-audited read-only tool"
                if acquire
                else "no caller-audited read-only tool is exposed"
            ),
            SemanticFamily.CLARIFY.value: (
                "ALE does not provide a standardized clarification-user oracle"
            ),
            SemanticFamily.CURATE.value: (
                "the checkpoint contains visible context or trajectory history"
                if curate
                else "initial prompt-only checkpoint has no visible context to curate"
            ),
        }
        return ALEFamilyEligibility(self.checkpoint_id, eligibility, reasons)


@dataclass(frozen=True, slots=True)
class ALEEvaluationResult:
    """Task-level outcome returned after the official hidden grader runs."""

    task_score: float
    cost_usd: float
    wall_seconds: float
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        values = {
            "task_score": self.task_score,
            "cost_usd": self.cost_usd,
            "wall_seconds": self.wall_seconds,
        }
        normalized: dict[str, float] = {}
        for label, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"ALE {label} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValidationError(f"ALE {label} must be finite")
            normalized[label] = numeric
        if not 0.0 <= normalized["task_score"] <= 1.0:
            raise ValidationError("ALE task_score must be in [0, 1]")
        if normalized["cost_usd"] < 0.0 or normalized["wall_seconds"] < 0.0:
            raise ValidationError("ALE cost and wall time must be non-negative")
        for label, value in normalized.items():
            object.__setattr__(self, label, value)

    @property
    def metrics(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "task_score": self.task_score,
                "cost_usd": self.cost_usd,
                "wall_seconds": self.wall_seconds,
            }
        )


@runtime_checkable
class ALEEvaluationBridge(Protocol):
    ale_bridge_contract: str
    benchmark_id: str
    source_revision: str
    bridge_id: str
    bridge_revision: str
    supports_fresh_environment: bool
    supports_hidden_reference_staging: bool
    provenance_status: str
    evaluation_scope: str
    official_runtime_verified: bool

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        checkpoint: ALECheckpoint,
        prediction: Any,
    ) -> ALEEvaluationResult: ...


@dataclass(frozen=True, slots=True)
class CallerAssertedALEEvaluationBridge:
    """Explicit wrapper for an operator-owned ALE orchestration callback."""

    bridge_id: str
    bridge_revision: str
    supports_fresh_environment: bool
    supports_hidden_reference_staging: bool
    callback: Callable[[Mapping[str, Any], ALECheckpoint, Any], ALEEvaluationResult] = field(
        repr=False
    )
    ale_bridge_contract: str = field(init=False, default="ale_evaluation_bridge_v1")
    benchmark_id: str = field(init=False, default=ALE_BENCHMARK_ID)
    source_revision: str = field(init=False, default=ALE_OFFICIAL_SOURCE_REVISION)
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not self.bridge_id.strip() or _SHA256.fullmatch(self.bridge_revision) is None:
            raise ValidationError("ALE bridge requires an ID and SHA-256 revision")
        if not callable(self.callback):
            raise TypeError("ALE bridge callback must be callable")
        if not self.supports_fresh_environment or not self.supports_hidden_reference_staging:
            raise ValidationError(
                "ALE bridge must preserve fresh environments and post-run hidden-reference staging"
            )

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        checkpoint: ALECheckpoint,
        prediction: Any,
    ) -> ALEEvaluationResult:
        result = self.callback(evaluator_payload, checkpoint, prediction)
        if not isinstance(result, ALEEvaluationResult):
            raise ValidationError("ALE callback must return ALEEvaluationResult")
        return result


class ALEExternalTransferAdapter:
    """Evaluator-blind adapter for official ALE independent-run checkpoints."""

    benchmark_id = ALE_BENCHMARK_ID
    adapter_metric_names = ("task_score", "cost_usd", "wall_seconds")
    pairing_mode = ALE_PAIRING_MODE
    split = ALE_PUBLIC_SPLIT

    def __init__(
        self,
        *,
        checkpoints: Iterable[ALECheckpoint],
        evaluation_bridge: ALEEvaluationBridge,
    ) -> None:
        if not isinstance(evaluation_bridge, ALEEvaluationBridge):
            raise TypeError("evaluation_bridge must implement ALEEvaluationBridge")
        _validate_bridge(evaluation_bridge)
        normalized = tuple(checkpoints)
        if not normalized or not all(isinstance(item, ALECheckpoint) for item in normalized):
            raise ValidationError("ALE requires at least one typed checkpoint")
        ids = [item.checkpoint_id for item in normalized]
        task_units = [
            (
                item.task_id,
                item.environment_snapshot_sha256,
                item.agent_harness_sha256,
                item.provider_identity_sha256,
                item.image_identity_sha256,
            )
            for item in normalized
        ]
        if len(ids) != len(set(ids)) or len(task_units) != len(set(task_units)):
            raise ValidationError("ALE checkpoints and protected run units must be unique")
        examples: list[BenchmarkExample] = []
        checkpoints_by_id: dict[str, ALECheckpoint] = {}
        for checkpoint in sorted(normalized, key=lambda item: item.checkpoint_id):
            support = checkpoint.family_eligibility
            evaluator_payload = freeze_mapping(
                {
                    "agent_harness_sha256": checkpoint.agent_harness_sha256,
                    "environment_snapshot_sha256": checkpoint.environment_snapshot_sha256,
                    "grader_sha256": checkpoint.grader_sha256,
                    "image_identity_sha256": checkpoint.image_identity_sha256,
                    "provider_identity_sha256": checkpoint.provider_identity_sha256,
                    "run_handle": checkpoint.run_handle,
                    "source_revision": checkpoint.source_revision,
                    "source_task_id": checkpoint.task_id,
                    "task_source_sha256": checkpoint.task_source_sha256,
                }
            )
            example = BenchmarkExample(
                example_id=checkpoint.checkpoint_id,
                query=checkpoint.query,
                instruction=checkpoint.instruction,
                visible_context=checkpoint.visible_context,
                visible_messages=checkpoint.visible_messages,
                visible_tool_names=checkpoint.visible_tool_names,
                visible_metadata=freeze_mapping(
                    {
                        "benchmark_cell": self.benchmark_id,
                        "domain": checkpoint.domain,
                        "family_eligibility": dict(support.eligibility),
                        "family_eligibility_reasons": dict(support.reasons),
                        "family_support_status": support.support_status,
                        "source_split": checkpoint.split,
                    }
                ),
                evaluator_gold=evaluator_payload,
                optimizer_visible_id=opaque_checkpoint_state_id(
                    self.benchmark_id,
                    checkpoint.checkpoint_id,
                    checkpoint.environment_snapshot_sha256,
                ),
            )
            examples.append(example)
            checkpoints_by_id[checkpoint.checkpoint_id] = checkpoint
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}
        self._checkpoints = MappingProxyType(checkpoints_by_id)
        self._evaluation_bridge = evaluation_bridge

    @property
    def available_splits(self) -> tuple[str, ...]:
        return (self.split,)

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self.split:
            raise ValidationError(f"ALE adapter exposes {self.split!r}, not {split!r}")
        return self._examples

    def family_eligibility(self, checkpoint_id: str) -> ALEFamilyEligibility:
        return self._checkpoint(checkpoint_id).family_eligibility

    def eligible_families(self, checkpoint_id: str) -> tuple[SemanticFamily, ...]:
        return self.family_eligibility(checkpoint_id).eligible_families

    def supports_full_block(self, checkpoint_id: str) -> bool:
        self._checkpoint(checkpoint_id)
        return False

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this ALE adapter")
        return self._evaluation_bridge.evaluate(
            example.evaluator_gold,
            self._checkpoints[example.example_id],
            prediction,
        ).metrics

    @property
    def source_provenance(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "benchmark_id": self.benchmark_id,
                "data_license": ALE_DATA_LICENSE,
                "repository_url": ALE_OFFICIAL_REPOSITORY,
                "software_license": ALE_SOFTWARE_LICENSE,
                "source_revision": ALE_OFFICIAL_SOURCE_REVISION,
            }
        )

    def _checkpoint(self, checkpoint_id: str) -> ALECheckpoint:
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise ValidationError(f"unknown ALE checkpoint {checkpoint_id!r}") from exc


def build_ale_official_command(
    *,
    checkout: str | Path,
    experiment_config: str | Path,
    dry_run: bool = False,
    uv_executable: str = "uv",
) -> tuple[str, ...]:
    """Build ALE's native command after verifying the exact upstream checkout."""

    validate_ale_checkout(checkout)
    config = Path(experiment_config).expanduser().resolve()
    if not config.is_file() or config.is_symlink():
        raise ExternalDependencyError("ALE experiment config must be a regular file")
    command = (
        uv_executable,
        "run",
        "python",
        "-m",
        "ale_run",
        "run",
        str(config),
    )
    if dry_run:
        return (*command, "--dry-run")
    return command


def validate_ale_checkout(checkout: str | Path) -> Path:
    """Verify the audited ALE commit and critical orchestration files."""

    source = Path(checkout).expanduser().resolve()
    if not source.is_dir() or source.is_symlink():
        raise ExternalDependencyError("ALE checkout must be a regular directory")
    for relative, expected_hash in ALE_SOURCE_HASHES.items():
        path = source / relative
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected_hash:
            raise ExternalDependencyError(f"ALE pinned source hash changed: {relative}")
    try:
        head = subprocess.run(
            ("git", "-C", str(source), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(source), "status", "--porcelain"),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ExternalDependencyError("ALE Git checkout verification failed") from exc
    if head != ALE_OFFICIAL_SOURCE_REVISION:
        raise ExternalDependencyError("ALE checkout is not at the pinned commit")
    if status:
        raise ExternalDependencyError("ALE pinned checkout must be clean")
    return source


def _validate_bridge(bridge: ALEEvaluationBridge) -> None:
    if (
        bridge.ale_bridge_contract != "ale_evaluation_bridge_v1"
        or bridge.benchmark_id != ALE_BENCHMARK_ID
        or bridge.source_revision != ALE_OFFICIAL_SOURCE_REVISION
    ):
        raise ValidationError("ALE evaluation bridge identity mismatch")
    if bridge.provenance_status != _CALLER_ASSERTED or bridge.evaluation_scope != _EXTERNAL_BRIDGE:
        raise ValidationError("ALE evaluation bridge provenance mismatch")
    if bridge.official_runtime_verified is not False:
        raise ValidationError("caller-asserted ALE bridge cannot claim official verification")
    if not bridge.supports_fresh_environment or not bridge.supports_hidden_reference_staging:
        raise ValidationError("ALE bridge omits a required official runtime boundary")


def _validated_names(
    values: tuple[str, ...],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValidationError(f"ALE requires {label}")
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in normalized
    ):
        raise ValidationError(f"ALE {label} must be trimmed non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValidationError(f"ALE {label} must be unique")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ALE_BENCHMARK_ID",
    "ALE_DATA_LICENSE",
    "ALE_OFFICIAL_REPOSITORY",
    "ALE_OFFICIAL_SOURCE_REVISION",
    "ALE_PAIRING_MODE",
    "ALE_PUBLIC_SPLIT",
    "ALE_SOFTWARE_LICENSE",
    "ALE_SOURCE_HASHES",
    "ALECheckpoint",
    "ALEEvaluationBridge",
    "ALEEvaluationResult",
    "ALEExternalTransferAdapter",
    "ALEFamilyEligibility",
    "CallerAssertedALEEvaluationBridge",
    "build_ale_official_command",
    "validate_ale_checkout",
]
