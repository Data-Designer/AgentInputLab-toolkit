"""Prepared-source and external checkpoint/evaluation bridge for tau3 Banking.

The official static task files are simulator/evaluator specifications, not
user-visible prompts.  This module deliberately refuses to synthesize a query
from those hidden scripts.  Callers must capture a replayable pre-action
checkpoint from the official runtime and bind it to opaque runtime identities.
This module does not yet capture, clone, continue, or evaluate that runtime by
itself; those operations remain the responsibility of the supplied bridge.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentinputlab.artifacts import sha256_file
from agentinputlab.contracts.enums import SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import canonical_json, freeze_mapping
from agentinputlab.contracts.state import ContextBlock, Message

from .base import BenchmarkExample, opaque_checkpoint_state_id
from .preprocess import PreparedBenchmark

_BENCHMARK_ID = "tau3.banking_knowledge"
_PROCESSOR = "tau3_banking_tar_v1"
_SPLIT = "base"
TAU3_OFFICIAL_SOURCE_REVISION = "cf71a8070269883e38a365ffa85f78f46844c1f4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STRUCTURAL_CAPABILITY_NAMES = frozenset(
    {
        "clarification_roundtrip_available",
        "state_restore_available",
    }
)
_CALLER_ASSERTED = "caller_asserted"
_STRUCTURAL_CANDIDATE = "structural_candidate"
_EXTERNAL_BRIDGE = "external_bridge"
_RETRIEVAL_TOOL_NAMES = frozenset(
    {"KB_search", "KB_search_bm25", "KB_search_dense", "grep", "shell"}
)
_INTERVENTION_FAMILIES = (
    SemanticFamily.REFORMULATE,
    SemanticFamily.DECOMPOSE,
    SemanticFamily.ACQUIRE,
    SemanticFamily.CLARIFY,
    SemanticFamily.CURATE,
)


@dataclass(frozen=True, slots=True)
class Tau3FamilyEligibility:
    """Visible-only family support for one captured checkpoint."""

    checkpoint_id: str
    eligibility: Mapping[str, bool]
    reasons: Mapping[str, str]
    provenance_status: str = _CALLER_ASSERTED
    support_status: str = _STRUCTURAL_CANDIDATE

    def __post_init__(self) -> None:
        expected = {family.value for family in _INTERVENTION_FAMILIES}
        if set(self.eligibility) != expected or set(self.reasons) != expected:
            raise ValidationError("tau3 family eligibility must cover the five core families")
        if any(not isinstance(value, bool) for value in self.eligibility.values()):
            raise ValidationError("tau3 family eligibility values must be boolean")
        if any(not isinstance(value, str) or not value for value in self.reasons.values()):
            raise ValidationError("tau3 family eligibility reasons must be non-empty strings")
        if self.provenance_status != _CALLER_ASSERTED:
            raise ValidationError("tau3 eligibility provenance must remain caller_asserted")
        if self.support_status != _STRUCTURAL_CANDIDATE:
            raise ValidationError("tau3 eligibility status must remain structural_candidate")
        object.__setattr__(self, "eligibility", MappingProxyType(dict(self.eligibility)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))

    @property
    def eligible_families(self) -> tuple[SemanticFamily, ...]:
        return tuple(family for family in _INTERVENTION_FAMILIES if self.eligibility[family.value])

    @property
    def supports_full_block(self) -> bool:
        return all(self.eligibility.values())


@dataclass(frozen=True, slots=True)
class Tau3BankingCheckpoint:
    """A captured optimizer-visible state with opaque official-runtime bindings.

    The four SHA-256 identities bind the checkpoint to a cloned runtime state,
    retrieval implementation, prepared corpus, and deterministic replay.  They
    intentionally carry no database rows, tool results, simulator instructions,
    evaluator criteria, or required-document labels.
    """

    checkpoint_id: str
    task_id: str
    query: str
    instruction: str
    visible_messages: tuple[Message, ...]
    tool_names: tuple[str, ...]
    runtime_snapshot_id: str
    retrieval_revision: str
    corpus_revision: str
    replay_identity: str
    retrieval_config_id: str
    visible_context: tuple[ContextBlock, ...] = ()
    structural_capabilities: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in ("checkpoint_id", "task_id", "query", "instruction", "retrieval_config_id"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(f"tau3 checkpoint {label} must be a trimmed non-empty string")
        for label in (
            "runtime_snapshot_id",
            "retrieval_revision",
            "corpus_revision",
            "replay_identity",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValidationError(f"tau3 checkpoint {label} must be a lowercase SHA-256")

        messages = tuple(self.visible_messages)
        context = tuple(self.visible_context)
        tool_names = tuple(self.tool_names)
        if not messages or not all(isinstance(message, Message) for message in messages):
            raise ValidationError("tau3 checkpoint requires captured visible Message instances")
        if not all(isinstance(block, ContextBlock) for block in context):
            raise ValidationError("tau3 checkpoint context must contain ContextBlock instances")
        if not tool_names or not all(
            isinstance(name, str) and name.strip() and name == name.strip() for name in tool_names
        ):
            raise ValidationError("tau3 checkpoint requires trimmed visible tool names")
        if len(tool_names) != len(set(tool_names)):
            raise ValidationError("tau3 checkpoint visible tool names must be unique")
        user_messages = [message for message in messages if message.role == "user"]
        if not user_messages:
            raise ValidationError("tau3 checkpoint requires at least one visible user message")
        if user_messages[-1].content != self.query:
            raise ValidationError(
                "tau3 checkpoint query must equal the latest visible user message"
            )

        capabilities = dict(self.structural_capabilities)
        if not set(capabilities) <= _STRUCTURAL_CAPABILITY_NAMES:
            raise ValidationError(
                "tau3 checkpoint contains an unknown structural capability; "
                "defect labels are not eligibility inputs"
            )
        if any(not isinstance(value, bool) for value in capabilities.values()):
            raise ValidationError("tau3 checkpoint structural capabilities must be boolean")
        object.__setattr__(self, "visible_messages", messages)
        object.__setattr__(self, "visible_context", context)
        object.__setattr__(self, "tool_names", tool_names)
        object.__setattr__(
            self,
            "structural_capabilities",
            MappingProxyType(capabilities),
        )

    @property
    def family_eligibility(self) -> Tau3FamilyEligibility:
        retrieval_available = bool(set(self.tool_names) & _RETRIEVAL_TOOL_NAMES)
        curation_material = len(self.visible_messages) >= 3 or len(self.visible_context) >= 2
        clarification = self.structural_capabilities.get(
            "clarification_roundtrip_available",
            False,
        )
        eligibility = {
            SemanticFamily.REFORMULATE.value: True,
            SemanticFamily.DECOMPOSE.value: True,
            SemanticFamily.ACQUIRE.value: retrieval_available,
            SemanticFamily.CLARIFY.value: clarification,
            SemanticFamily.CURATE.value: curation_material,
        }
        reasons = {
            SemanticFamily.REFORMULATE.value: "captured checkpoint has an explicit visible query",
            SemanticFamily.DECOMPOSE.value: (
                "captured checkpoint has an explicit visible query that can be decomposed"
            ),
            SemanticFamily.ACQUIRE.value: (
                "an official visible retrieval channel is present"
                if retrieval_available
                else "requires KB_search, KB_search_bm25, KB_search_dense, grep, or shell"
            ),
            SemanticFamily.CLARIFY.value: (
                "captured runtime exposes a replayable clarification round trip"
                if clarification
                else "requires clarification_roundtrip_available=true"
            ),
            SemanticFamily.CURATE.value: (
                "captured state contains enough visible history or context to curate"
                if curation_material
                else "requires at least three visible messages or two visible context blocks"
            ),
        }
        return Tau3FamilyEligibility(self.checkpoint_id, eligibility, reasons)


@dataclass(frozen=True, slots=True)
class Tau3EvaluationResult:
    """Typed single-arm values returned by an external evaluation bridge."""

    official_reward: float
    cost: float
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        metrics = _validated_bridge_metrics(
            {"cost": self.cost, "official_reward": self.official_reward}
        )
        object.__setattr__(self, "official_reward", metrics["official_reward"])
        object.__setattr__(self, "cost", metrics["cost"])

    @property
    def metrics(self) -> Mapping[str, float]:
        return MappingProxyType({"cost": self.cost, "official_reward": self.official_reward})


@runtime_checkable
class Tau3EvaluationBridge(Protocol):
    """Typed caller-asserted bridge; this is not an official-runtime attestation."""

    tau3_bridge_contract: str
    benchmark_id: str
    source_revision: str
    bridge_id: str
    bridge_revision: str
    cost_basis: str
    supports_state_restore: bool
    supports_user_simulator_replay: bool
    provenance_status: str
    evaluation_scope: str
    official_runtime_verified: bool

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        checkpoint: Tau3BankingCheckpoint,
        prediction: Any,
    ) -> Tau3EvaluationResult: ...


@dataclass(frozen=True, slots=True)
class CallerAssertedTau3EvaluationBridge:
    """Explicit wrapper for an unverified external tau3 evaluation callback."""

    bridge_id: str
    bridge_revision: str
    cost_basis: str
    callback: Callable[
        [Mapping[str, Any], Tau3BankingCheckpoint, Any],
        Tau3EvaluationResult,
    ] = field(repr=False)
    supports_state_restore: bool
    supports_user_simulator_replay: bool
    tau3_bridge_contract: str = field(init=False, default="tau3_evaluation_bridge_v1")
    benchmark_id: str = field(init=False, default=_BENCHMARK_ID)
    source_revision: str = field(init=False, default=TAU3_OFFICIAL_SOURCE_REVISION)
    provenance_status: str = field(init=False, default=_CALLER_ASSERTED)
    evaluation_scope: str = field(init=False, default=_EXTERNAL_BRIDGE)
    official_runtime_verified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        for label in ("bridge_id", "cost_basis"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValidationError(f"tau3 evaluation bridge {label} must be non-empty")
        if (
            not isinstance(self.bridge_revision, str)
            or _SHA256.fullmatch(self.bridge_revision) is None
        ):
            raise ValidationError("tau3 evaluation bridge revision must be a lowercase SHA-256")
        if not callable(self.callback):
            raise TypeError("tau3 evaluation bridge callback must be callable")
        for label in ("supports_state_restore", "supports_user_simulator_replay"):
            if not isinstance(getattr(self, label), bool):
                raise ValidationError(f"tau3 evaluation bridge {label} must be boolean")

    def evaluate(
        self,
        evaluator_payload: Mapping[str, Any],
        checkpoint: Tau3BankingCheckpoint,
        prediction: Any,
    ) -> Tau3EvaluationResult:
        result = self.callback(evaluator_payload, checkpoint, prediction)
        if not isinstance(result, Tau3EvaluationResult):
            raise ValidationError(
                "tau3 evaluation callback must return Tau3EvaluationResult; "
                "bare metric mappings are not accepted"
            )
        if (
            result.provenance_status != self.provenance_status
            or result.evaluation_scope != self.evaluation_scope
            or result.official_runtime_verified is not False
        ):
            raise ValidationError(
                "tau3 evaluation result provenance differs from its caller-asserted bridge"
            )
        return result


class PreparedTau3BankingBundle:
    """Verified task/corpus bundle with evaluator-only source state."""

    def __init__(self, prepared: PreparedBenchmark) -> None:
        if not isinstance(prepared, PreparedBenchmark):
            raise TypeError("prepared must be a PreparedBenchmark")
        if (
            prepared.cell_id != _BENCHMARK_ID
            or prepared.processor != _PROCESSOR
            or prepared.split != _SPLIT
        ):
            raise ArtifactError("prepared tau3 Banking Knowledge identity does not match")
        if prepared.source_revision != TAU3_OFFICIAL_SOURCE_REVISION:
            raise ArtifactError(
                "prepared tau3 Banking Knowledge source revision does not match "
                "the audited upstream commit"
            )
        if prepared.corpus_path is None or prepared.audit_path is None:
            raise ArtifactError("prepared tau3 Banking Knowledge requires corpus and audit files")
        if set(prepared.file_sha256) != {
            "audit.jsonl",
            "corpus.jsonl",
            "gold.jsonl",
            "inputs.jsonl",
        }:
            raise ArtifactError("prepared tau3 Banking Knowledge file set is incomplete")
        corpus_revision = prepared.file_sha256["corpus.jsonl"]
        if _SHA256.fullmatch(corpus_revision) is None:
            raise ArtifactError("prepared tau3 Banking Knowledge corpus hash is invalid")
        _verify_prepared_file_hashes(prepared)

        input_rows = _records_by_id(prepared.inputs_path, label="tau3 task inventory")
        gold_rows = _records_by_id(prepared.gold_path, label="tau3 evaluator gold")
        corpus_rows = _records_by_id(prepared.corpus_path, label="tau3 corpus")
        audit_rows = _read_jsonl(prepared.audit_path, label="tau3 source audit")
        if set(input_rows) != set(gold_rows):
            raise ArtifactError("tau3 task inventory and evaluator gold IDs do not match")
        if len(audit_rows) != 1:
            raise ArtifactError("tau3 source audit must contain exactly one record")

        task_specs: dict[str, Mapping[str, Any]] = {}
        for task_id in sorted(input_rows):
            visible = input_rows[task_id]
            if set(visible) != {"checkpoint_required", "id", "visible_metadata"}:
                raise ArtifactError("tau3 static input schema differs from the production lock")
            if visible["checkpoint_required"] is not True:
                raise ArtifactError("tau3 static tasks must require an explicit checkpoint")
            metadata = visible["visible_metadata"]
            if not isinstance(metadata, Mapping) or dict(metadata) != {
                "benchmark_cell": _BENCHMARK_ID,
                "source_split": _SPLIT,
                "source_task_id": task_id,
            }:
                raise ArtifactError("tau3 task inventory visible metadata differs from the lock")
            gold = gold_rows[task_id]
            if set(gold) != {"id", "task_spec"} or not isinstance(gold["task_spec"], Mapping):
                raise ArtifactError("tau3 evaluator gold schema differs from the production lock")
            task_spec = gold["task_spec"]
            if task_spec.get("id") != task_id:
                raise ArtifactError("tau3 evaluator task ID does not match its inventory ID")
            task_specs[task_id] = _frozen_mapping(task_spec, label="tau3 task specification")

        for document_id, document in corpus_rows.items():
            if set(document) != {"id", "text", "title"}:
                raise ArtifactError("tau3 corpus schema differs from the production lock")
            for key in ("text", "title"):
                value = document[key]
                if not isinstance(value, str) or not value.strip():
                    raise ArtifactError(f"tau3 corpus field {key!r} must be non-empty")
            if document["id"] != document_id:  # pragma: no cover - index invariant
                raise ArtifactError("tau3 corpus document ID mismatch")

        audit = audit_rows[0]
        if audit.get("id") != "tau3-banking-knowledge-source-audit":
            raise ArtifactError("tau3 source audit identity differs from the production lock")
        if not isinstance(audit.get("database"), Mapping):
            raise ArtifactError("tau3 source audit is missing evaluator-only database state")
        if audit.get("source_revision") != prepared.source_revision:
            raise ArtifactError("tau3 source audit revision differs from prepared metadata")

        expected_counts = {
            "inputs": len(input_rows),
            "gold": len(gold_rows),
            "corpus": len(corpus_rows),
            "tasks": len(input_rows),
            "documents": len(corpus_rows),
            "audit": 1,
        }
        for key, value in expected_counts.items():
            if prepared.counts.get(key) != value:
                raise ArtifactError(f"prepared tau3 count mismatch for {key}")

        self.benchmark_id = _BENCHMARK_ID
        self.split = _SPLIT
        self.prepared = prepared
        self.corpus_path = prepared.corpus_path
        self.corpus_revision = corpus_revision
        self.task_ids = tuple(sorted(task_specs))
        self.document_count = len(corpus_rows)
        self._task_specs = MappingProxyType(task_specs)
        self._source_audit = _frozen_mapping(audit, label="tau3 source audit")
        self._expected_file_sha256 = MappingProxyType(dict(prepared.file_sha256))
        _verify_prepared_file_hashes(prepared)

    def verify_integrity(self) -> None:
        """Re-hash every prepared file before runtime use."""

        if dict(self.prepared.file_sha256) != dict(self._expected_file_sha256):
            raise ArtifactError("prepared tau3 declared file hashes changed after bundle creation")
        _verify_prepared_file_hashes(self.prepared)

    def _evaluator_payload(self, task_id: str) -> Mapping[str, Any]:
        self.verify_integrity()
        return MappingProxyType(
            {
                "task_spec": self._task_specs[task_id],
                "source_audit": self._source_audit,
            }
        )


class Tau3BankingCheckpointAdapter:
    """Adapter over externally captured checkpoints; fails closed otherwise.

    The adapter validates prepared source data, visible checkpoint contracts,
    and evaluation-bridge outputs.  It is not an official-runtime capture,
    clone, continuation, or evaluator implementation.
    """

    benchmark_id = _BENCHMARK_ID
    adapter_metric_names = ("official_reward", "cost")
    split = _SPLIT
    pairing_mode = "unsupported_external_replay"

    def __init__(
        self,
        bundle: PreparedTau3BankingBundle,
        *,
        checkpoints: Iterable[Tau3BankingCheckpoint],
        evaluation_bridge: Tau3EvaluationBridge,
    ) -> None:
        if not isinstance(bundle, PreparedTau3BankingBundle):
            raise TypeError("bundle must be a PreparedTau3BankingBundle")
        if not isinstance(evaluation_bridge, Tau3EvaluationBridge):
            raise TypeError(
                "evaluation_bridge must implement Tau3EvaluationBridge; wrap external "
                "callbacks with CallerAssertedTau3EvaluationBridge"
            )
        _validate_evaluation_bridge(evaluation_bridge)
        bundle.verify_integrity()
        normalized = tuple(checkpoints)
        if not normalized:
            raise ValidationError(
                "tau3 Banking Knowledge requires at least one captured visible checkpoint"
            )
        if not all(isinstance(item, Tau3BankingCheckpoint) for item in normalized):
            raise TypeError("tau3 checkpoints must be Tau3BankingCheckpoint instances")
        checkpoint_ids = [item.checkpoint_id for item in normalized]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValidationError("tau3 checkpoint IDs must be unique")
        experimental_units = [
            (
                item.task_id,
                item.runtime_snapshot_id,
                item.retrieval_revision,
                item.corpus_revision,
                item.replay_identity,
                item.retrieval_config_id,
            )
            for item in normalized
        ]
        if len(experimental_units) != len(set(experimental_units)):
            raise ValidationError("tau3 protected experimental units must be unique")

        examples: list[BenchmarkExample] = []
        checkpoints_by_id: dict[str, Tau3BankingCheckpoint] = {}
        task_ids = set(bundle.task_ids)
        for checkpoint in sorted(normalized, key=lambda item: item.checkpoint_id):
            if checkpoint.task_id not in task_ids:
                raise ValidationError(
                    f"tau3 checkpoint references unknown official task {checkpoint.task_id!r}"
                )
            if checkpoint.corpus_revision != bundle.corpus_revision:
                raise ValidationError("tau3 checkpoint corpus revision differs from the bundle")
            support = checkpoint.family_eligibility
            clarification = support.eligibility[SemanticFamily.CLARIFY.value]
            state_restore = checkpoint.structural_capabilities.get(
                "state_restore_available",
                False,
            )
            if clarification and not evaluation_bridge.supports_user_simulator_replay:
                raise ValidationError(
                    "tau3 clarification checkpoint requires a bridge with "
                    "user-simulator replay support"
                )
            if state_restore and not evaluation_bridge.supports_state_restore:
                raise ValidationError(
                    "tau3 checkpoint advertises state restoration but its bridge does not"
                )
            visible_metadata = _frozen_mapping(
                {
                    "benchmark_cell": self.benchmark_id,
                    "family_eligibility": dict(support.eligibility),
                    "family_eligibility_provenance": support.provenance_status,
                    "family_eligibility_reasons": dict(support.reasons),
                    "family_support_status": support.support_status,
                    "source_split": self.split,
                },
                label="tau3 visible checkpoint metadata",
            )
            evaluator_gold = _frozen_mapping(
                {"task_spec": bundle._task_specs[checkpoint.task_id]},
                label="tau3 evaluator gold",
            )
            examples.append(
                BenchmarkExample(
                    example_id=checkpoint.checkpoint_id,
                    query=checkpoint.query,
                    instruction=checkpoint.instruction,
                    visible_context=checkpoint.visible_context,
                    visible_messages=checkpoint.visible_messages,
                    visible_tool_names=checkpoint.tool_names,
                    visible_metadata=visible_metadata,
                    evaluator_gold=evaluator_gold,
                    optimizer_visible_id=opaque_checkpoint_state_id(
                        self.benchmark_id,
                        checkpoint.checkpoint_id,
                        checkpoint.replay_identity,
                    ),
                )
            )
            checkpoints_by_id[checkpoint.checkpoint_id] = checkpoint

        self.bundle = bundle
        self._evaluation_bridge = evaluation_bridge
        self._examples = tuple(examples)
        self._examples_by_id = {example.example_id: example for example in examples}
        self._checkpoints = MappingProxyType(checkpoints_by_id)

    def examples(self, split: str) -> tuple[BenchmarkExample, ...]:
        if split != self.split:
            raise ValidationError(
                f"tau3 Banking Knowledge adapter exposes split {self.split!r}, not {split!r}"
            )
        self.bundle.verify_integrity()
        return self._examples

    def family_eligibility(self, checkpoint_id: str) -> Tau3FamilyEligibility:
        return self._checkpoint(checkpoint_id).family_eligibility

    def eligible_families(self, checkpoint_id: str) -> tuple[SemanticFamily, ...]:
        return self.family_eligibility(checkpoint_id).eligible_families

    def supports_full_block(self, checkpoint_id: str) -> bool:
        return self.family_eligibility(checkpoint_id).supports_full_block

    def score(self, example: BenchmarkExample, prediction: Any) -> Mapping[str, float]:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("example must be a BenchmarkExample")
        if self._examples_by_id.get(example.example_id) is not example:
            raise ValidationError("benchmark example does not belong to this tau3 adapter")
        checkpoint = self._checkpoints[example.example_id]
        result = self._evaluation_bridge.evaluate(
            self.bundle._evaluator_payload(checkpoint.task_id),
            checkpoint,
            prediction,
        )
        if not isinstance(result, Tau3EvaluationResult):
            raise ValidationError("tau3 evaluation bridge returned an invalid result type")
        return result.metrics

    @property
    def evaluation_provenance(self) -> Mapping[str, Any]:
        """Return explicit non-official provenance for result manifests."""

        return _frozen_mapping(
            {
                "bridge_id": self._evaluation_bridge.bridge_id,
                "bridge_revision": self._evaluation_bridge.bridge_revision,
                "benchmark_id": self._evaluation_bridge.benchmark_id,
                "cost_basis": self._evaluation_bridge.cost_basis,
                "evaluation_scope": self._evaluation_bridge.evaluation_scope,
                "official_runtime_verified": self._evaluation_bridge.official_runtime_verified,
                "provenance_status": self._evaluation_bridge.provenance_status,
                "source_revision": self._evaluation_bridge.source_revision,
                "supports_state_restore": self._evaluation_bridge.supports_state_restore,
                "supports_user_simulator_replay": (
                    self._evaluation_bridge.supports_user_simulator_replay
                ),
            },
            label="tau3 evaluation provenance",
        )

    def _checkpoint(self, checkpoint_id: str) -> Tau3BankingCheckpoint:
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise ValidationError(f"unknown tau3 checkpoint {checkpoint_id!r}") from exc


def _validated_bridge_metrics(value: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValidationError("tau3 evaluation bridge metrics must be a mapping")
    expected = {"cost", "official_reward"}
    if set(value) != expected:
        raise ValidationError("tau3 evaluation bridge must return exactly official_reward and cost")
    metrics: dict[str, float] = {}
    for key in sorted(expected):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValidationError(f"tau3 evaluation bridge metric {key!r} must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValidationError(f"tau3 evaluation bridge metric {key!r} must be finite")
        metrics[key] = numeric
    if not 0.0 <= metrics["official_reward"] <= 1.0:
        raise ValidationError("tau3 official_reward must be in [0, 1]")
    if metrics["cost"] < 0.0:
        raise ValidationError("tau3 cost must be non-negative")
    return MappingProxyType(metrics)


def _validate_evaluation_bridge(bridge: Tau3EvaluationBridge) -> None:
    if bridge.tau3_bridge_contract != "tau3_evaluation_bridge_v1":
        raise ValidationError("tau3 bridge contract identity does not match")
    if bridge.benchmark_id != _BENCHMARK_ID:
        raise ValidationError("tau3 bridge benchmark identity does not match")
    if bridge.source_revision != TAU3_OFFICIAL_SOURCE_REVISION:
        raise ValidationError("tau3 bridge source revision does not match the audited commit")
    if bridge.provenance_status != _CALLER_ASSERTED:
        raise ValidationError("tau3 bridge provenance must remain caller_asserted")
    if bridge.evaluation_scope != _EXTERNAL_BRIDGE:
        raise ValidationError("tau3 bridge evaluation scope must remain external_bridge")
    if bridge.official_runtime_verified is not False:
        raise ValidationError("current tau3 bridge cannot claim official runtime verification")
    for label in ("supports_state_restore", "supports_user_simulator_replay"):
        if not isinstance(getattr(bridge, label), bool):
            raise ValidationError(f"tau3 evaluation bridge {label} must be boolean")
    for label in ("bridge_id", "cost_basis"):
        value = getattr(bridge, label)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValidationError(f"tau3 evaluation bridge {label} must be non-empty")
    if (
        not isinstance(bridge.bridge_revision, str)
        or _SHA256.fullmatch(bridge.bridge_revision) is None
    ):
        raise ValidationError("tau3 evaluation bridge revision must be a lowercase SHA-256")


def _verify_prepared_file_hashes(prepared: PreparedBenchmark) -> None:
    for name, expected in sorted(prepared.file_sha256.items()):
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ArtifactError(f"prepared tau3 hash is invalid for {name}")
        path = prepared.root / name
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise ArtifactError(f"prepared tau3 file cannot be verified: {name}") from exc
        if actual != expected:
            raise ArtifactError(f"prepared tau3 file hash mismatch: {name}")


def _records_by_id(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, label=label)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ArtifactError(f"{label} IDs must be non-empty strings")
        if row_id in indexed:
            raise ArtifactError(f"{label} contains duplicate IDs")
        indexed[row_id] = row
    return indexed


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"{label} cannot be read") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{label} contains invalid JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ArtifactError(f"{label} rows must be objects")
        rows.append(row)
    if not rows:
        raise ArtifactError(f"{label} must not be empty")
    return rows


def _frozen_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    try:
        detached = json.loads(canonical_json(value))
        return freeze_mapping(detached)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label} must be canonically serializable") from exc


__all__ = [
    "TAU3_OFFICIAL_SOURCE_REVISION",
    "CallerAssertedTau3EvaluationBridge",
    "PreparedTau3BankingBundle",
    "Tau3BankingCheckpoint",
    "Tau3BankingCheckpointAdapter",
    "Tau3EvaluationBridge",
    "Tau3EvaluationResult",
    "Tau3FamilyEligibility",
]
