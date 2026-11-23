"""Fail-closed bridge for Q-RAG's released frozen E5/HotpotQA policy.

Q-RAG's source checkout has no repository-level license declaration.  The
source and the separately licensed checkpoint therefore stay in a dedicated,
user-mounted worker.  This module sends only the ordered candidate texts,
binds every treatment-defining artifact, and validates the official
candidate-index/Q-value trajectory.  It never exposes training, rewards,
supporting-fact labels, or a synthetic ``stop`` action.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CompositeResult,
    ExecutionTrace,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    SemanticFamily,
    TraceEvent,
    TracePhase,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    finite_number,
    mapping_sequence,
    nonnegative_integer,
    required_text,
    resolve_worker,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

from .reader_adapter import (
    QRAG_INSTRUCTION_PROTOCOL,
    QRAG_TASK_READER_PROFILES,
    task_reader_prompt_hash,
)

QRAG_UPSTREAM_COMMIT = "e7598130b63eaab868469877e88bcacc59a44efb"
QRAG_PROTOCOL = "agentinputlab.qrag.frozen.v3"
QRAG_PROFILE_ID = "qrag.e5_hotpotqa.frozen.v1"

QRAG_CHECKPOINT_REPOSITORY = "Q-RAG/qrag-ft-e5-on-hotpotqa"
QRAG_CHECKPOINT_REVISION = "e0e48389f551075ace2ce61207522b6bad55593a"
QRAG_CHECKPOINT_FILE_SHA256 = "ccd4068d3917bed2e44afcd7ee7a9d66c8487b44dd2206252e6d2985352f92b7"
QRAG_CHECKPOINT_CONFIG_SHA256 = "edbba621832e545a63723eda2b6652eccfd51b29eda7d04145ba22a7ea48c241"
QRAG_EMBEDDER_REPOSITORY = "intfloat/multilingual-e5-large"
QRAG_EMBEDDER_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
QRAG_EMBEDDER_MODEL_SHA256 = "020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6"
QRAG_SOURCE_MANIFEST_SHA256 = "86c49eee9c64329bd77f3b5a4286e937aadb7c5fcd67d1238f214421cc27ae1a"

QRAG_QA_ENV_SOURCE_SHA256 = "267c27c248f332478bf810a50f097a898a818cf6f8f4bd7b70a2893312c0eae1"
QRAG_TEXT_ENV_SOURCE_SHA256 = "2bfee69d8bea9e9fc2822dce8aa42c563344da1454a111f807fe2ac7a00fc39c"
QRAG_PQN_SOURCE_SHA256 = "5e69ee5e20f098c101ceebb0c3713a3e63ffcd0a1b614d7180c1b474465cce39"

QRAG_ENVIRONMENT_CONTRACT = freeze_mapping(
    {
        "dataset_profile": "hotpotqa",
        "max_steps": 2,
        "sort_by_index": False,
        "separator": " [SEP] ",
        "position_processor": "absolute_none",
        "feedback_model": "DummyFeedbackModel",
        "feedback_never_terminate": True,
        "gold_answer_available": False,
        "supporting_fact_labels_available": False,
    }
)
QRAG_ENVIRONMENT_CONFIG_SHA256 = content_hash(QRAG_ENVIRONMENT_CONTRACT)
QRAG_ACTION_SCHEMA = freeze_mapping(
    {
        "action": "candidate_index",
        "stop_action": False,
        "policy": "argmax_available_q_value",
        "termination": ("fixed_step_budget", "candidate_exhausted", "empty_candidates"),
    }
)
QRAG_ACTION_SCHEMA_SHA256 = content_hash(QRAG_ACTION_SCHEMA)
QRAG_CANDIDATE_BUILDER_CONTRACT = freeze_mapping(
    {
        "input_fields": ("id", "question", "chunks"),
        "candidate_fields": ("document_id", "text"),
        "preserve_input_order": True,
        "dummy_answer": "",
        "dummy_supporting_fact_indices": (),
    }
)
QRAG_CANDIDATE_BUILDER_SHA256 = content_hash(QRAG_CANDIDATE_BUILDER_CONTRACT)

QRAG_READER_SYSTEM_PROMPT = (
    "Answer the question based on the given passages.\n"
    "Only give me the short and precise answer, do not output any other words.\n"
    "Keep your reasoning very brief and concise.\n"
    'Always end your response with "Final answer: [your final answer]".\n'
)
QRAG_READER_USER_TEMPLATE = (
    "\nGIVEN PASSAGES:\n{context}\n\nQUESTION:\n{question}\n\nFinal answer: "
)
QRAG_READER_PROMPT_SHA256 = content_hash(
    {"system": QRAG_READER_SYSTEM_PROMPT, "user": QRAG_READER_USER_TEMPLATE}
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_READER_PROFILE = "upstream_qwq32b_family_controlled"
_TRANSFER_READER_PROFILE = "factorial_reader_transfer"


@dataclass(frozen=True, slots=True)
class QRAGLicenseAcknowledgement:
    """Explicitly acknowledge the two independently governed user mounts."""

    source_noassertion_acknowledged: bool = False
    checkpoint_cc_by_4_0_acknowledged: bool = False
    user_mounted_source: bool = False
    user_mounted_checkpoint: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.source_noassertion_acknowledged,
                self.checkpoint_cc_by_4_0_acknowledged,
                self.user_mounted_source,
                self.user_mounted_checkpoint,
            )
        ):
            raise ValidationError("Q-RAG license acknowledgements must be boolean")

    def require_complete(self) -> None:
        if not self.source_noassertion_acknowledged or not self.user_mounted_source:
            raise ExternalDependencyError(
                "Q-RAG source is NOASSERTION; explicitly acknowledge and user-mount "
                "the pinned source checkout before execution"
            )
        if not self.checkpoint_cc_by_4_0_acknowledged or not self.user_mounted_checkpoint:
            raise ExternalDependencyError(
                "Q-RAG checkpoint is a separate CC-BY-4.0 asset; explicitly acknowledge "
                "and user-mount the pinned checkpoint before execution"
            )

    def as_mapping(self) -> Mapping[str, bool]:
        return freeze_mapping(asdict(self))


@dataclass(frozen=True, slots=True)
class QRAGArtifactIdentity:
    """Treatment-defining identity for one frozen Q-RAG evaluation cell."""

    checkpoint_manifest_sha256: str
    checkpoint_file_sha256: str
    embedder_manifest_sha256: str
    source_manifest_sha256: str
    compatibility_manifest_sha256: str
    retriever_revision: str
    retriever_deployment_fingerprint_sha256: str
    corpus_sha256: str
    index_sha256: str
    reader_repository: str
    reader_revision: str
    reader_deployment_fingerprint_sha256: str
    reader_profile: str = _NATIVE_READER_PROFILE
    checkpoint_repository: str = QRAG_CHECKPOINT_REPOSITORY
    checkpoint_revision: str = QRAG_CHECKPOINT_REVISION
    embedder_repository: str = QRAG_EMBEDDER_REPOSITORY
    embedder_revision: str = QRAG_EMBEDDER_REVISION
    environment_config_sha256: str = QRAG_ENVIRONMENT_CONFIG_SHA256
    action_schema_sha256: str = QRAG_ACTION_SCHEMA_SHA256
    candidate_builder_sha256: str = QRAG_CANDIDATE_BUILDER_SHA256
    reader_prompt_sha256: str = QRAG_READER_PROMPT_SHA256
    checkpoint_config_sha256: str = QRAG_CHECKPOINT_CONFIG_SHA256
    profile_id: str = QRAG_PROFILE_ID

    def __post_init__(self) -> None:
        fixed = {
            "checkpoint_repository": QRAG_CHECKPOINT_REPOSITORY,
            "checkpoint_revision": QRAG_CHECKPOINT_REVISION,
            "checkpoint_file_sha256": QRAG_CHECKPOINT_FILE_SHA256,
            "checkpoint_config_sha256": QRAG_CHECKPOINT_CONFIG_SHA256,
            "embedder_repository": QRAG_EMBEDDER_REPOSITORY,
            "embedder_revision": QRAG_EMBEDDER_REVISION,
            "source_manifest_sha256": QRAG_SOURCE_MANIFEST_SHA256,
            "environment_config_sha256": QRAG_ENVIRONMENT_CONFIG_SHA256,
            "action_schema_sha256": QRAG_ACTION_SCHEMA_SHA256,
            "candidate_builder_sha256": QRAG_CANDIDATE_BUILDER_SHA256,
            "profile_id": QRAG_PROFILE_ID,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValidationError(f"Q-RAG {name} is fixed to {expected}")
        for name in (
            "checkpoint_manifest_sha256",
            "embedder_manifest_sha256",
            "compatibility_manifest_sha256",
            "retriever_deployment_fingerprint_sha256",
            "corpus_sha256",
            "index_sha256",
            "reader_deployment_fingerprint_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise ValidationError(f"Q-RAG {name} must be a lowercase SHA-256 digest")
        for name in (
            "retriever_revision",
            "reader_repository",
            "reader_revision",
            "reader_profile",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"Q-RAG {name} must not be empty")
        if self.reader_profile not in {
            _NATIVE_READER_PROFILE, _TRANSFER_READER_PROFILE, *QRAG_TASK_READER_PROFILES,
        }:
            raise ValidationError("Q-RAG reader_profile is unsupported")
        expected_prompt = (
            task_reader_prompt_hash(self.reader_profile)
            if self.reader_profile in QRAG_TASK_READER_PROFILES
            else QRAG_READER_PROMPT_SHA256
        )
        if self.reader_prompt_sha256 != expected_prompt:
            raise ValidationError("Q-RAG reader prompt does not match the explicit reader profile")

    def as_mapping(self) -> Mapping[str, Any]:
        return freeze_mapping(asdict(self))

    @property
    def identity_sha256(self) -> str:
        return content_hash(self.as_mapping())


@dataclass(frozen=True, slots=True)
class QRAGControls:
    """Fixed two-action selector; configurable reader output ceiling.

    4000 preserves the upstream reader default. The effective request ceiling
    is also bounded by the run's remaining output tokens, not a requirement
    to generate or reserve all 4000 tokens.
    """

    max_select_actions: int = 2
    max_output_tokens: int = 4000
    max_candidates: int = 128
    max_candidate_chars: int = 2_000_000

    def __post_init__(self) -> None:
        if self.max_select_actions != 2:
            raise ValidationError("Q-RAG HotpotQA frozen profile requires max_select_actions=2")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.max_output_tokens, self.max_candidates, self.max_candidate_chars)
        ):
            raise ValidationError("Q-RAG output and candidate limits must be positive integers")


@dataclass(frozen=True, slots=True)
class QRAGStep:
    index: int
    available_indices: tuple[int, ...]
    q_values: tuple[float, ...]
    action_index: int
    selected_q_value: float
    selected_indices: tuple[int, ...]
    selected_document_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> QRAGStep:
        available = _integer_tuple(
            value.get("available_indices"), label="Q-RAG available_indices", unique=True
        )
        q_raw = value.get("q_values")
        if isinstance(q_raw, (str, bytes)) or not isinstance(q_raw, Sequence):
            raise AlgorithmOutputError("Q-RAG q_values must be a sequence")
        q_values = tuple(finite_number(item, label="Q-RAG q_value") for item in q_raw)
        if len(q_values) != len(available):
            raise AlgorithmOutputError("Q-RAG q_values must align with available_indices")
        selected_ids_raw = value.get("selected_document_ids")
        if isinstance(selected_ids_raw, (str, bytes)) or not isinstance(selected_ids_raw, Sequence):
            raise AlgorithmOutputError("Q-RAG selected_document_ids must be a sequence")
        selected_ids = tuple(selected_ids_raw)
        if not all(isinstance(item, str) and item for item in selected_ids):
            raise AlgorithmOutputError("Q-RAG selected_document_ids must contain non-empty strings")
        return cls(
            index=nonnegative_integer(value.get("index"), label="Q-RAG step index"),
            available_indices=available,
            q_values=q_values,
            action_index=nonnegative_integer(value.get("action_index"), label="Q-RAG action_index"),
            selected_q_value=finite_number(
                value.get("selected_q_value"), label="Q-RAG selected_q_value"
            ),
            selected_indices=_integer_tuple(
                value.get("selected_indices"),
                label="Q-RAG selected_indices",
                unique=True,
            ),
            selected_document_ids=selected_ids,
        )


@dataclass(slots=True)
class _QRAGWorker:
    artifacts: QRAGArtifactIdentity
    acknowledgements: QRAGLicenseAcknowledgement = QRAGLicenseAcknowledgement()
    client: WorkerClient | None = None
    controls: QRAGControls = QRAGControls()
    release_id: str = "acquire.qrag.released_checkpoint_native"
    profile_kind: str = "controlled_adapter"
    method_id: str = "qrag"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, QRAGArtifactIdentity):
            raise ValidationError("Q-RAG requires QRAGArtifactIdentity")
        if not isinstance(self.acknowledgements, QRAGLicenseAcknowledgement):
            raise ValidationError("Q-RAG requires QRAGLicenseAcknowledgement")
        if not isinstance(self.controls, QRAGControls):
            raise ValidationError("Q-RAG controls must be QRAGControls")

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        self.acknowledgements.require_complete()
        self._validate_reader_profile()
        if not state.query.strip():
            raise ValidationError("Q-RAG requires a non-empty question")
        instruction_adapter = self.artifacts.reader_profile in QRAG_TASK_READER_PROFILES
        if state.messages or (state.instruction.strip() and not instruction_adapter):
            raise ValidationError(
                "the frozen Q-RAG profile accepts one question; instruction/messages "
                "require a separate factorial profile"
            )
        if instruction_adapter and state.tool_names:
            raise ValidationError("Q-RAG instruction reader does not execute tools")
        if len(state.context) > self.controls.max_candidates:
            raise ValidationError("Q-RAG candidate count exceeds max_candidates")
        if sum(len(block.text) for block in state.context) > self.controls.max_candidate_chars:
            raise ValidationError("Q-RAG candidate text exceeds max_candidate_chars")
        candidate_ids = tuple(block.block_id for block in state.context)
        if any(not value for value in candidate_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValidationError("Q-RAG candidates require unique non-empty IDs")

        candidates = tuple(
            {"document_id": block.block_id, "text": block.text} for block in state.context
        )
        candidate_pool_sha256 = content_hash(candidates)
        candidate_order_sha256 = content_hash(candidate_ids)
        remaining_budget = _remaining_budget(state, context)
        if remaining_budget["model_calls"] < 1:
            raise BudgetExceededError("Q-RAG reader requires one remaining model call")
        if remaining_budget["output_tokens"] < 1:
            raise BudgetExceededError("Q-RAG reader has no remaining output tokens")
        reader_budget = {
            "configured_max_output_tokens": self.controls.max_output_tokens,
            "effective_max_output_tokens": min(
                self.controls.max_output_tokens, remaining_budget["output_tokens"]
            ),
            "remaining_output_tokens": remaining_budget["output_tokens"],
            "preset": (
                "upstream_default" if self.controls.max_output_tokens == 4000 else "custom"
            ),
        }
        if state.budget.max_cost_usd is not None:
            raise BudgetExceededError(
                "Q-RAG external reader is unpriced; a cost budget cannot be guaranteed"
            )

        worker = resolve_worker(self.client, context, label="Q-RAG")
        worker_identity = worker_fingerprint(worker, label="Q-RAG")
        artifact_mapping = self.artifacts.as_mapping()
        reader_task = {
            "instruction": state.instruction,
            "candidate_annotations": tuple(
                {"document_id": block.block_id, "source": block.source, "metadata": block.metadata}
                for block in state.context
            ),
        } if instruction_adapter else None
        reader_task_fields = (
            {"reader_task": reader_task, "reader_task_sha256": content_hash(reader_task)}
            if instruction_adapter else {}
        )
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=QRAG_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                **reader_task_fields,
                "profile_id": self.artifacts.profile_id,
                "task_id": state.task_id,
                "question": state.query,
                "candidates": candidates,
                "candidate_pool_sha256": candidate_pool_sha256,
                "candidate_order_sha256": candidate_order_sha256,
                "artifacts": artifact_mapping,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "acknowledgements": self.acknowledgements.as_mapping(),
                "controls": asdict(self.controls),
                "remaining_budget": remaining_budget,
                "reader_budget": reader_budget,
                "seed": context.seed,
                "inference_contract": {
                    "mode": "frozen_checkpoint_inference",
                    "allow_training": False,
                    "allow_gold_answer": False,
                    "allow_supporting_fact_labels": False,
                    "allow_reward": False,
                    "allow_ground_truth_feedback": False,
                    "allow_stop_action": False,
                },
            },
            dependency_fingerprints={
                "worker": worker_identity,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "candidate_pool_sha256": candidate_pool_sha256,
                "candidate_order_sha256": candidate_order_sha256,
            },
            minimum_reservation={"model_calls": 1},
            protocol_version=QRAG_INSTRUCTION_PROTOCOL if instruction_adapter else QRAG_PROTOCOL,
        )
        response = call_worker(worker, request)
        payload = response.payload
        if instruction_adapter and payload.get("reader_task_sha256") != content_hash(reader_task):
            raise AlgorithmOutputError("Q-RAG worker task instruction binding mismatch")

        echoed_artifacts = payload.get("artifacts")
        if not isinstance(echoed_artifacts, Mapping) or content_hash(
            echoed_artifacts
        ) != content_hash(artifact_mapping):
            raise AlgorithmOutputError("Q-RAG artifact identity mismatch")
        for key, expected in (
            ("profile_id", self.artifacts.profile_id),
            ("artifact_identity_sha256", self.artifacts.identity_sha256),
            ("candidate_pool_sha256", candidate_pool_sha256),
            ("candidate_order_sha256", candidate_order_sha256),
        ):
            if required_text(payload, key, label="Q-RAG response") != expected:
                raise AlgorithmOutputError(f"Q-RAG {key} mismatch")

        attestation = _validate_runtime_attestation(payload, artifacts=self.artifacts)
        if payload.get("reader_budget") != reader_budget:
            raise AlgorithmOutputError("Q-RAG worker reader budget mismatch")
        steps = tuple(
            QRAGStep.from_mapping(row)
            for row in mapping_sequence(payload.get("trajectory"), label="Q-RAG trajectory")
        )
        selected_indices = _validate_trajectory(
            steps,
            candidate_ids=candidate_ids,
            max_select_actions=self.controls.max_select_actions,
        )
        final_selected = _integer_tuple(
            payload.get("selected_indices"),
            label="Q-RAG final selected_indices",
            unique=True,
        )
        if final_selected != selected_indices:
            raise AlgorithmOutputError("Q-RAG final selected indices disagree with trajectory")
        final_document_ids = _string_tuple(
            payload.get("selected_document_ids"),
            label="Q-RAG final selected_document_ids",
        )
        expected_document_ids = tuple(candidate_ids[index] for index in selected_indices)
        if final_document_ids != expected_document_ids:
            raise AlgorithmOutputError("Q-RAG final selected document IDs disagree with trajectory")

        termination = required_text(payload, "termination", label="Q-RAG response")
        expected_termination = _expected_termination(
            candidate_count=len(candidate_ids),
            selected_count=len(selected_indices),
            max_select_actions=self.controls.max_select_actions,
        )
        if termination != expected_termination:
            raise AlgorithmOutputError("Q-RAG termination disagrees with fixed environment horizon")
        answer = required_text(payload, "final_answer", label="Q-RAG response")
        trajectory_sha256 = required_text(payload, "trajectory_sha256", label="Q-RAG response")
        if not _HEX64.fullmatch(trajectory_sha256) or trajectory_sha256 != content_hash(
            payload.get("trajectory")
        ):
            raise AlgorithmOutputError("Q-RAG trajectory_sha256 mismatch")
        if response.usage.model_calls != 1:
            raise AlgorithmOutputError("Q-RAG frozen reader must report exactly one model call")
        if response.usage.retrieval_calls != 0 or response.usage.tool_calls != 0:
            raise AlgorithmOutputError(
                "Q-RAG receives a frozen candidate pool and cannot report retrieval/tool calls"
            )
        _validate_usage_against_sent_budget(response.usage, remaining_budget)
        if response.usage.output_tokens > reader_budget["effective_max_output_tokens"]:
            raise AlgorithmOutputError("Q-RAG worker exceeded effective reader output limit")

        selected_blocks = tuple(state.context[index] for index in selected_indices)
        trace_events: list[TraceEvent] = [
            TraceEvent(
                TracePhase.PREPARE,
                "frozen_candidate_pool",
                {
                    "profile_id": self.artifacts.profile_id,
                    "artifact_identity_sha256": self.artifacts.identity_sha256,
                    "candidate_pool_sha256": candidate_pool_sha256,
                    "candidate_order_sha256": candidate_order_sha256,
                    "checkpoint_revision": self.artifacts.checkpoint_revision,
                    "embedder_revision": self.artifacts.embedder_revision,
                    "retriever_revision": self.artifacts.retriever_revision,
                    "reader_revision": self.artifacts.reader_revision,
                    "selection_budget": self.controls.max_select_actions,
                    "reader_budget": reader_budget,
                    "paper_native_parity_claimed": False,
                },
            )
        ]
        trace_events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "candidate_index_policy_step",
                {
                    "index": step.index,
                    "available_indices": step.available_indices,
                    "q_values": step.q_values,
                    "action_index": step.action_index,
                    "selected_q_value": step.selected_q_value,
                    "selected_indices": step.selected_indices,
                    "selected_document_ids": step.selected_document_ids,
                },
            )
            for step in steps
        )
        trace_events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "selected_context_and_reader_answer",
                {
                    "selected_indices": selected_indices,
                    "selected_document_ids": expected_document_ids,
                    "termination": termination,
                    "trajectory_sha256": trajectory_sha256,
                    "answer_is_intervention_input": False,
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(trace_events),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 1,
                "policy_steps": len(steps),
                "selected_documents": len(selected_blocks),
            },
        )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=answer,
            evidence=selected_blocks,
            trace=trace,
            implementation_identity=(f"thin_external_wrapper:griver/Q-RAG@{QRAG_UPSTREAM_COMMIT}"),
            profile_kind=self.profile_kind,
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "profile_id": self.artifacts.profile_id,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "source_license_id": "NOASSERTION",
                "checkpoint_license_id": "CC-BY-4.0",
                "source_license_acknowledged": True,
                "checkpoint_license_acknowledged": True,
                "checkpoint_repository": self.artifacts.checkpoint_repository,
                "checkpoint_revision": self.artifacts.checkpoint_revision,
                "embedder_repository": self.artifacts.embedder_repository,
                "embedder_revision": self.artifacts.embedder_revision,
                "retriever_revision": self.artifacts.retriever_revision,
                "reader_repository": self.artifacts.reader_repository,
                "reader_revision": self.artifacts.reader_revision,
                "reader_profile": self.artifacts.reader_profile,
                "input_adapter": "instruction_preserving_v1" if instruction_adapter else "question_only",
                "reader_prompt_sha256": self.artifacts.reader_prompt_sha256,
                "reader_task_sha256": content_hash(reader_task) if instruction_adapter else None,
                "task_instruction_preserved": instruction_adapter,
                "reader_budget": reader_budget,
                "candidate_pool_sha256": candidate_pool_sha256,
                "candidate_order_sha256": candidate_order_sha256,
                "selected_indices": selected_indices,
                "selected_document_ids": expected_document_ids,
                "termination": termination,
                "trajectory_sha256": trajectory_sha256,
                "runtime_attestation": attestation,
                "worker_fingerprint": worker_identity,
                "treatment_scope": "native_checkpoint_embedder_and_value_policy",
                "reader_is_controlled_factor": True,
                "paper_native_parity_claimed": False,
                "gold_answer_used": False,
                "supporting_fact_labels_used": False,
                "reward_used": False,
                "training_called": False,
            },
        )

    def _validate_reader_profile(self) -> None:
        expected = (
            _TRANSFER_READER_PROFILE
            if self.profile_kind == "factorial_adapter"
            else _NATIVE_READER_PROFILE
        )
        allowed = {expected}
        if self.profile_kind == "factorial_adapter":
            allowed.update(QRAG_TASK_READER_PROFILES)
        if self.artifacts.reader_profile not in allowed:
            raise ValidationError(f"Q-RAG {self.release_id} requires reader_profile={expected}")


class QRAGNativeWorker(_QRAGWorker):
    """Native released checkpoint/retriever treatment with a controlled reader.

    The historical class name is retained for registry compatibility.  The
    result is deliberately labelled ``controlled_adapter`` and never claims
    end-to-end paper parity because upstream does not pin the reader artifact
    and the benchmark-safe environment removes ground-truth feedback.
    """


class QRAGReaderTransferWorker(_QRAGWorker):
    def __init__(
        self,
        artifacts: QRAGArtifactIdentity,
        acknowledgements: QRAGLicenseAcknowledgement | None = None,
        client: WorkerClient | None = None,
        *,
        controls: QRAGControls | None = None,
    ) -> None:
        super().__init__(
            artifacts=artifacts,
            acknowledgements=(
                acknowledgements if acknowledgements is not None else QRAGLicenseAcknowledgement()
            ),
            client=client,
            controls=controls if controls is not None else QRAGControls(),
            release_id="acquire.qrag.reader_transfer_factorial",
            profile_kind="factorial_adapter",
        )


def _validate_runtime_attestation(
    payload: Mapping[str, Any], *, artifacts: QRAGArtifactIdentity
) -> Mapping[str, Any]:
    value = payload.get("runtime_attestation")
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("Q-RAG requires runtime_attestation")
    expected: Mapping[str, Any] = {
        "source_commit": QRAG_UPSTREAM_COMMIT,
        "source_manifest_sha256": artifacts.source_manifest_sha256,
        "compatibility_manifest_sha256": artifacts.compatibility_manifest_sha256,
        "checkpoint_repository": artifacts.checkpoint_repository,
        "checkpoint_revision": artifacts.checkpoint_revision,
        "checkpoint_manifest_sha256": artifacts.checkpoint_manifest_sha256,
        "checkpoint_file_sha256": artifacts.checkpoint_file_sha256,
        "embedder_repository": artifacts.embedder_repository,
        "embedder_revision": artifacts.embedder_revision,
        "official_candidate_builder_called": True,
        "official_pqn_select_action_called": True,
        "official_qa_env_step_called": True,
        "dummy_feedback_used": True,
        "ground_truth_feedback_used": False,
        "gold_answer_used": False,
        "supporting_fact_labels_used": False,
        "reward_used": False,
        "training_called": False,
        "stop_action_used": False,
        "strict_state_dict_load": True,
        "safe_weights_only_load": True,
        "checkpoint_eval_mode": True,
        "source_user_mounted": True,
        "checkpoint_user_mounted": True,
        "checkpoint_deserialization_isolated": True,
        "host_secret_mounts": False,
        "reader_profile": artifacts.reader_profile,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AlgorithmOutputError(f"Q-RAG runtime attestation {key} mismatch")
    if value.get("source_attestation") not in {"git_checkout", "verified_archive_snapshot"}:
        raise AlgorithmOutputError("Q-RAG source attestation is unsupported")
    return freeze_mapping(value)


def _validate_trajectory(
    steps: tuple[QRAGStep, ...],
    *,
    candidate_ids: tuple[str, ...],
    max_select_actions: int,
) -> tuple[int, ...]:
    if tuple(step.index for step in steps) != tuple(range(len(steps))):
        raise AlgorithmOutputError("Q-RAG step indices must be contiguous")
    expected_steps = min(len(candidate_ids), max_select_actions)
    if len(steps) != expected_steps:
        raise AlgorithmOutputError("Q-RAG trajectory length disagrees with fixed horizon")
    selected: list[int] = []
    for step in steps:
        expected_available = tuple(
            index for index in range(len(candidate_ids)) if index not in selected
        )
        if step.available_indices != expected_available:
            raise AlgorithmOutputError("Q-RAG available-index trajectory is inconsistent")
        if step.action_index not in expected_available:
            raise AlgorithmOutputError("Q-RAG action is unavailable or out of range")
        q_position = expected_available.index(step.action_index)
        if not math.isclose(
            step.selected_q_value,
            step.q_values[q_position],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AlgorithmOutputError("Q-RAG selected_q_value disagrees with q_values")
        if step.q_values and step.selected_q_value < max(step.q_values):
            raise AlgorithmOutputError("Q-RAG action is not an argmax available Q-value")
        selected.append(step.action_index)
        if step.selected_indices != tuple(selected):
            raise AlgorithmOutputError("Q-RAG selected-index history is inconsistent")
        expected_ids = tuple(candidate_ids[index] for index in selected)
        if step.selected_document_ids != expected_ids:
            raise AlgorithmOutputError("Q-RAG selected-document history is inconsistent")
    return tuple(selected)


def _expected_termination(
    *, candidate_count: int, selected_count: int, max_select_actions: int
) -> str:
    if candidate_count == 0:
        return "empty_candidates"
    if selected_count == candidate_count:
        return "candidate_exhausted"
    if selected_count == max_select_actions:
        return "fixed_step_budget"
    raise AlgorithmOutputError("Q-RAG trajectory has no valid fixed-horizon termination")


def _integer_tuple(value: Any, *, label: str, unique: bool) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlgorithmOutputError(f"{label} must be a sequence")
    result = tuple(nonnegative_integer(item, label=label) for item in value)
    if unique and len(result) != len(set(result)):
        raise AlgorithmOutputError(f"{label} contains duplicates")
    return result


def _string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AlgorithmOutputError(f"{label} must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise AlgorithmOutputError(f"{label} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise AlgorithmOutputError(f"{label} contains duplicates")
    return result


def _remaining_budget(
    state: FrozenAgentState, context: InterventionContext
) -> Mapping[str, int | None]:
    usage = None
    ledger = context.budget_ledger
    if ledger is not None:
        ledger_budget = getattr(ledger, "budget", None)
        usage = getattr(ledger, "usage", None)
        if ledger_budget != state.budget or usage is None:
            raise ValidationError("Q-RAG received an incompatible budget ledger")

    def consumed(name: str) -> int:
        value = 0 if usage is None else getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(f"Q-RAG budget usage {name} is invalid")
        return value

    remaining_context = None
    if state.budget.max_context_tokens is not None:
        remaining_context = max(0, state.budget.max_context_tokens - consumed("input_tokens"))
    return freeze_mapping(
        {
            "model_calls": max(0, state.budget.max_model_calls - consumed("model_calls")),
            "output_tokens": max(0, state.budget.max_output_tokens - consumed("output_tokens")),
            "context_tokens": remaining_context,
        }
    )


def _validate_usage_against_sent_budget(usage: Any, budget: Mapping[str, int | None]) -> None:
    if usage.model_calls > budget["model_calls"]:
        raise AlgorithmOutputError("Q-RAG worker exceeded sent model_calls budget")
    if usage.output_tokens > budget["output_tokens"]:
        raise AlgorithmOutputError("Q-RAG worker exceeded sent output_tokens budget")
    context_cap = budget["context_tokens"]
    if context_cap is not None and usage.input_tokens > context_cap:
        raise AlgorithmOutputError("Q-RAG worker exceeded sent context_tokens budget")


__all__ = [
    "QRAG_ACTION_SCHEMA_SHA256",
    "QRAG_CANDIDATE_BUILDER_SHA256",
    "QRAG_CHECKPOINT_CONFIG_SHA256",
    "QRAG_CHECKPOINT_FILE_SHA256",
    "QRAG_CHECKPOINT_REPOSITORY",
    "QRAG_CHECKPOINT_REVISION",
    "QRAG_EMBEDDER_MODEL_SHA256",
    "QRAG_EMBEDDER_REPOSITORY",
    "QRAG_EMBEDDER_REVISION",
    "QRAG_ENVIRONMENT_CONFIG_SHA256",
    "QRAG_PROFILE_ID",
    "QRAG_PROTOCOL",
    "QRAG_SOURCE_MANIFEST_SHA256",
    "QRAG_UPSTREAM_COMMIT",
    "QRAGArtifactIdentity",
    "QRAGControls",
    "QRAGLicenseAcknowledgement",
    "QRAGNativeWorker",
    "QRAGReaderTransferWorker",
    "QRAGStep",
]
