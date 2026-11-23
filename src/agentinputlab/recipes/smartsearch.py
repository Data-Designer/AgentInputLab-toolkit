"""Pinned frozen-inference bridge for the official SmartSearch search loop.

The released policy, official ``ReCall`` loop, tag parser, and answer parser run
inside a separately pinned worker.  AgentInputLab standardizes only the
transport, identity, budget, security, trajectory, and evidence contracts.  In
particular, this frozen-inference profile does not invoke training or query
refinement. Separately metered native training interfaces live in the
smartsearch curriculum, policy, teacher and GRPO recipe modules.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CompositeResult,
    ContextBlock,
    ExecutionTrace,
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
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

SMARTSEARCH_UPSTREAM_COMMIT = "5e9e7ec0d6af34f45fad145d4bff442529e7e6ff"
SMARTSEARCH_MODEL_REPOSITORY = "vvv111222/SmartSearch-3B"
SMARTSEARCH_CHECKPOINT_REVISION = "82bfe8a845e3e08f0d9d00d8e1e2cfb6d50b5da8"
SMARTSEARCH_PROTOCOL = "agentinputlab.smartsearch.frozen.v2"
SMARTSEARCH_PROFILE_ID = "smartsearch.controlled_frozen.v1"
SMARTSEARCH_COMMON_SUPPORT_RELEASE = "rewrite.smartsearch.common_support_queries"

SMARTSEARCH_TOOL_SCHEMA: tuple[Mapping[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search for.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": ("Number of results to return. The default value is 5."),
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
)
SMARTSEARCH_TOOL_SCHEMA_SHA256 = content_hash(SMARTSEARCH_TOOL_SCHEMA)

# The first five values are upstream constants.  ``retry_attempts=1`` calls
# ReCall.run.__wrapped__ so an opaque outer retry cannot replay a whole paid
# trajectory.  One search tag per turn is the fail-closed parser hardening that
# makes the dynamic retrieval budget bounded.  These two disclosed changes are
# why this profile is a controlled adapter rather than an official-native cell.
SMARTSEARCH_LOOP_CONTRACT: Mapping[str, Any] = freeze_mapping(
    {
        "max_rounds": 10,
        "max_results_per_search": 5,
        "max_new_tokens": 8192,
        "temperature": 0.0,
        "stop": ("</search>", "</answer>"),
        "retry_attempts": 1,
        "search_tags_per_turn": 1,
    }
)
SMARTSEARCH_LOOP_CONTRACT_SHA256 = content_hash(SMARTSEARCH_LOOP_CONTRACT)
SMARTSEARCH_PARSER_CONTRACT_SHA256 = content_hash(
    {
        "active_pipeline_sha256": (
            "20394e3418b6e30b43057ca96b97a423b4e4d37a3a04901e0a0a0dbc3af221fa"
        ),
        "answer_helpers_sha256": (
            "2697ffa7479a0c9c4e6baf843164b7055b80350c627e01413584ecf4f76476bc"
        ),
        "answer_extractor": "ReCallPipeline.extract_last_assistant_response",
        "boxed_helpers": ("last_boxed_only_string", "remove_boxed"),
    }
)

_HEX_SHA256 = frozenset("0123456789abcdef")
_ALLOWED_TRANSPORTS = frozenset(
    {"sglang_generate", "openai_completions", "openai_completions_local_snapshot"}
)
_FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "expected_answer",
        "gold",
        "gold_answer",
        "ground_truth",
        "label",
        "labels",
        "process_reward",
        "reference_answer",
        "reward",
        "reward_signal",
        "rewards",
        "target_answer",
        "test_reward",
        "verifier_score",
    }
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_SHA256 for character in value.lower())
    ):
        raise ValidationError(f"{label} must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SmartSearchArtifactIdentity:
    """Content and deployment identities that define one frozen treatment.

    Upstream did not publish a pinned Wikipedia corpus, retriever checkpoint,
    or index.  Callers therefore provide content identities for those assets;
    changing any one produces a different treatment hash and must not be
    reported as paper-level native parity.
    """

    model_snapshot_manifest_sha256: str
    model_snapshot_tree_sha256: str
    tokenizer_revision: str
    served_model_name: str
    model_deployment_fingerprint_sha256: str
    context_admission_fingerprint_sha256: str
    retriever_revision: str
    retriever_deployment_fingerprint_sha256: str
    corpus_sha256: str
    index_sha256: str
    retrieval_config_sha256: str
    compatibility_manifest_sha256: str
    model_transport: str = "sglang_generate"
    checkpoint_repository: str = SMARTSEARCH_MODEL_REPOSITORY
    checkpoint_revision: str = SMARTSEARCH_CHECKPOINT_REVISION
    profile_id: str = SMARTSEARCH_PROFILE_ID

    def __post_init__(self) -> None:
        if self.checkpoint_repository != SMARTSEARCH_MODEL_REPOSITORY:
            raise ValidationError("SmartSearch requires the released 3B checkpoint repository")
        if self.checkpoint_revision != SMARTSEARCH_CHECKPOINT_REVISION:
            raise ValidationError("SmartSearch requires the released pinned checkpoint revision")
        if self.tokenizer_revision != SMARTSEARCH_CHECKPOINT_REVISION:
            raise ValidationError("SmartSearch tokenizer revision must match the released snapshot")
        if self.profile_id != SMARTSEARCH_PROFILE_ID:
            raise ValidationError("SmartSearch profile_id is not supported")
        if self.model_transport not in _ALLOWED_TRANSPORTS:
            raise ValidationError(
                "SmartSearch model_transport must be an approved frozen completion transport"
            )
        for name in (
            "served_model_name",
            "retriever_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"SmartSearch {name} must be non-empty text")
        for name in (
            "model_snapshot_manifest_sha256",
            "model_snapshot_tree_sha256",
            "model_deployment_fingerprint_sha256",
            "context_admission_fingerprint_sha256",
            "retriever_deployment_fingerprint_sha256",
            "corpus_sha256",
            "index_sha256",
            "retrieval_config_sha256",
            "compatibility_manifest_sha256",
        ):
            _validate_sha256(getattr(self, name), label=f"SmartSearch {name}")

    def as_mapping(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                **asdict(self),
                "tool_schema_sha256": SMARTSEARCH_TOOL_SCHEMA_SHA256,
                "loop_contract_sha256": SMARTSEARCH_LOOP_CONTRACT_SHA256,
                "parser_contract_sha256": SMARTSEARCH_PARSER_CONTRACT_SHA256,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return content_hash(self.as_mapping())


@dataclass(frozen=True, slots=True)
class SmartSearchControls:
    """Disclosed fixed controls for the hardened official loop."""

    max_search_rounds: int = 10
    max_results_per_round: int = 5
    max_evidence_documents: int = 50
    max_output_tokens_per_call: int = 8192

    def __post_init__(self) -> None:
        values = (
            self.max_search_rounds,
            self.max_results_per_round,
            self.max_evidence_documents,
            self.max_output_tokens_per_call,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValidationError("SmartSearch limits must be integers")
        if min(*values) < 1:
            raise ValidationError("SmartSearch limits must be positive")
        if values != (10, 5, 50, 8192):
            raise ValidationError("this frozen SmartSearch release fixes controls at 10/5/50/8192")


@dataclass(frozen=True, slots=True)
class SmartSearchStep:
    index: int
    query: str
    document_ids: tuple[str, ...]
    raw_search_hash: str
    retrieval_request_sha256: str
    retrieval_response_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SmartSearchStep:
        index = nonnegative_integer(value.get("index"), label="SmartSearch step index")
        query = required_text(value, "query", label="SmartSearch step")
        raw_search_text = required_text(value, "raw_search_text", label="SmartSearch step")
        if raw_search_text != query:
            raise AlgorithmOutputError("SmartSearch raw search text must equal the parsed query")
        raw_search_hash = required_text(value, "raw_search_hash", label="SmartSearch step")
        if raw_search_hash.lower() != _sha256_text(raw_search_text):
            raise AlgorithmOutputError("SmartSearch raw_search_hash mismatch")
        if value.get("tag_parse_status") != "ok":
            raise AlgorithmOutputError("SmartSearch requires a successfully parsed <search> tag")
        request_hash = required_text(
            value,
            "retrieval_request_sha256",
            label="SmartSearch step",
        ).lower()
        response_hash = required_text(
            value,
            "retrieval_response_sha256",
            label="SmartSearch step",
        ).lower()
        for digest, label in (
            (request_hash, "retrieval_request_sha256"),
            (response_hash, "retrieval_response_sha256"),
        ):
            if len(digest) != 64 or any(character not in _HEX_SHA256 for character in digest):
                raise AlgorithmOutputError(f"SmartSearch {label} must be a SHA-256 digest")
        return cls(
            index=index,
            query=query,
            document_ids=string_tuple(
                value.get("document_ids"),
                label="SmartSearch step document_ids",
                unique=True,
            ),
            raw_search_hash=raw_search_hash.lower(),
            retrieval_request_sha256=request_hash,
            retrieval_response_sha256=response_hash,
        )


@dataclass(frozen=True, slots=True)
class _SmartSearchEvidence:
    document_id: str
    text: str
    source: str
    score: float | None
    first_seen_step: int
    corpus_sha256: str
    index_sha256: str
    document_sha256: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        artifacts: SmartSearchArtifactIdentity,
    ) -> _SmartSearchEvidence:
        document_id = required_text(value, "document_id", label="SmartSearch evidence")
        text = required_text(value, "text", label="SmartSearch evidence")
        source = required_text(value, "source", label="SmartSearch evidence")
        if source != "smartsearch.wikipedia_search":
            raise AlgorithmOutputError("SmartSearch evidence source is not the pinned search seam")
        score_value = value.get("score")
        score = (
            None
            if score_value is None
            else finite_number(score_value, label="SmartSearch evidence score")
        )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise AlgorithmOutputError("SmartSearch evidence metadata must be a mapping")
        _reject_forbidden_inference_keys(metadata, label="SmartSearch evidence metadata")
        corpus_sha256 = required_text(value, "corpus_sha256", label="SmartSearch evidence")
        index_sha256 = required_text(value, "index_sha256", label="SmartSearch evidence")
        if corpus_sha256 != artifacts.corpus_sha256 or index_sha256 != artifacts.index_sha256:
            raise AlgorithmOutputError("SmartSearch evidence corpus/index identity mismatch")
        document_sha256 = required_text(
            value,
            "document_sha256",
            label="SmartSearch evidence",
        ).lower()
        if document_sha256 != content_hash({"document_id": document_id, "text": text}):
            raise AlgorithmOutputError("SmartSearch evidence document_sha256 mismatch")
        return cls(
            document_id=document_id,
            text=text,
            source=source,
            score=score,
            first_seen_step=nonnegative_integer(
                value.get("first_seen_step"),
                label="SmartSearch evidence first_seen_step",
            ),
            corpus_sha256=corpus_sha256,
            index_sha256=index_sha256,
            document_sha256=document_sha256,
            metadata=dict(metadata),
        )

    def as_context_block(self) -> ContextBlock:
        return ContextBlock(
            block_id=self.document_id,
            text=self.text,
            source=self.source,
            score=self.score,
            metadata={
                **dict(self.metadata),
                "smartsearch_first_seen_step": self.first_seen_step,
                "corpus_sha256": self.corpus_sha256,
                "index_sha256": self.index_sha256,
                "document_sha256": self.document_sha256,
            },
        )


@dataclass(slots=True)
class SmartSearchFrozenInference:
    """Run only the released SmartSearch-3B policy's frozen inference path."""

    artifacts: SmartSearchArtifactIdentity
    client: WorkerClient | None = None
    controls: SmartSearchControls = SmartSearchControls()
    method_id: str = "recipe.smartsearch"
    release_id: str = "recipe.smartsearch.frozen_3b_inference"
    family: SemanticFamily = SemanticFamily.COMPOSITE

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, SmartSearchArtifactIdentity):
            raise ValidationError("SmartSearch requires SmartSearchArtifactIdentity")
        if not isinstance(self.controls, SmartSearchControls):
            raise ValidationError("SmartSearch controls must be SmartSearchControls")

    def run(
        self,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> CompositeResult:
        if not state.query.strip():
            raise ValidationError("SmartSearch requires a non-empty query")
        if state.instruction.strip() or state.messages or state.context or state.tool_names:
            raise ValidationError(
                "the official SmartSearch frozen profile accepts one question only; "
                "instruction/messages/context/tools require a separate factorial profile"
            )
        remaining_budget = _remaining_budget(state, context)
        if remaining_budget["model_calls"] < 1:
            raise BudgetExceededError("SmartSearch requires one remaining model call")
        if remaining_budget["output_tokens"] < self.controls.max_output_tokens_per_call:
            raise BudgetExceededError(
                "SmartSearch requires 8192 remaining output tokens before its first model call"
            )
        if state.budget.max_cost_usd is not None:
            raise BudgetExceededError(
                "SmartSearch external calls are unpriced; a cost budget cannot be guaranteed"
            )

        worker = resolve_worker(self.client, context, label="SmartSearch")
        worker_identity = worker_fingerprint(worker, label="SmartSearch")
        artifact_mapping = self.artifacts.as_mapping()
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=SMARTSEARCH_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "profile_id": self.artifacts.profile_id,
                "task_id": state.task_id,
                "question": state.query,
                "artifacts": artifact_mapping,
                "controls": asdict(self.controls),
                "remaining_budget": remaining_budget,
                "seed": context.seed,
                "inference_contract": {
                    "mode": "frozen_inference",
                    "allow_training": False,
                    "allow_gold_answer": False,
                    "allow_test_time_reward": False,
                    "allow_external_query_refiner": False,
                },
            },
            dependency_fingerprints={
                "worker": worker_identity,
                "artifacts_sha256": self.artifacts.identity_sha256,
            },
            minimum_reservation={"model_calls": 1},
            protocol_version=SMARTSEARCH_PROTOCOL,
        )
        response = call_worker(worker, request)
        payload = response.payload
        _reject_forbidden_inference_keys(payload, label="SmartSearch response")

        echoed_artifacts = payload.get("artifacts")
        if not isinstance(echoed_artifacts, Mapping) or content_hash(
            echoed_artifacts
        ) != content_hash(artifact_mapping):
            raise AlgorithmOutputError("SmartSearch artifact identity mismatch")
        if required_text(payload, "profile_id", label="SmartSearch response") != (
            self.artifacts.profile_id
        ):
            raise AlgorithmOutputError("SmartSearch profile identity mismatch")
        if required_text(payload, "artifact_identity_sha256", label="SmartSearch response") != (
            self.artifacts.identity_sha256
        ):
            raise AlgorithmOutputError("SmartSearch artifact identity hash mismatch")
        _require_false(payload, "gold_answer_used")
        _require_false(payload, "test_time_reward_used")
        _require_false(payload, "external_query_refiner_used")
        _require_false(payload, "training_code_loaded")

        attestation = _validate_runtime_attestation(payload, artifacts=self.artifacts)
        steps = tuple(
            SmartSearchStep.from_mapping(row)
            for row in mapping_sequence(payload.get("trajectory"), label="SmartSearch trajectory")
        )
        evidence_records = tuple(
            _SmartSearchEvidence.from_mapping(row, artifacts=self.artifacts)
            for row in mapping_sequence(payload.get("evidence"), label="SmartSearch evidence")
        )
        _validate_search_lineage(steps, evidence_records, controls=self.controls)
        if response.usage.retrieval_calls != len(steps):
            raise AlgorithmOutputError(
                "SmartSearch retrieval_calls must equal parsed search-tag count"
            )
        if response.usage.model_calls < 1:
            raise AlgorithmOutputError("SmartSearch frozen inference requires a model call")
        if response.usage.model_calls not in {len(steps), len(steps) + 1}:
            raise AlgorithmOutputError("SmartSearch model/search call relationship is invalid")
        _validate_usage_against_sent_budget(response.usage, remaining_budget)

        final_answer = required_text(payload, "final_answer", label="SmartSearch response")
        termination = required_text(payload, "termination", label="SmartSearch response")
        if termination not in {"answer", "direct_answer", "max_search_rounds"}:
            raise AlgorithmOutputError("SmartSearch returned an unsupported termination reason")
        if termination == "direct_answer" and steps:
            raise AlgorithmOutputError("SmartSearch direct_answer cannot contain search steps")
        if termination == "max_search_rounds" and response.usage.model_calls != (
            self.controls.max_search_rounds
        ):
            raise AlgorithmOutputError(
                "SmartSearch max_search_rounds termination disagrees with model usage"
            )
        transcript_hash = required_text(
            payload,
            "raw_transcript_sha256",
            label="SmartSearch response",
        ).lower()
        if len(transcript_hash) != 64 or any(
            character not in _HEX_SHA256 for character in transcript_hash
        ):
            raise AlgorithmOutputError("SmartSearch raw_transcript_sha256 is invalid")

        evidence = tuple(record.as_context_block() for record in evidence_records)
        trace_events: list[TraceEvent] = [
            TraceEvent(
                TracePhase.PREPARE,
                "controlled_frozen_inference",
                {
                    "profile_id": self.artifacts.profile_id,
                    "artifact_identity_sha256": self.artifacts.identity_sha256,
                    "checkpoint_revision": self.artifacts.checkpoint_revision,
                    "corpus_sha256": self.artifacts.corpus_sha256,
                    "index_sha256": self.artifacts.index_sha256,
                    "reward_available": False,
                    "gold_available": False,
                    "external_refiner_available": False,
                    "paper_native_parity_claimed": False,
                },
            )
        ]
        trace_events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "search_step",
                {
                    "index": step.index,
                    "query": step.query,
                    "document_ids": step.document_ids,
                    "raw_search_hash": step.raw_search_hash,
                    "retrieval_request_sha256": step.retrieval_request_sha256,
                    "retrieval_response_sha256": step.retrieval_response_sha256,
                },
            )
            for step in steps
        )
        trace_events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "terminal_answer",
                {
                    "termination": termination,
                    "raw_transcript_sha256": transcript_hash,
                    "evidence_document_ids": tuple(block.block_id for block in evidence),
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
                "search_steps": len(steps),
                "evidence_documents": len(evidence),
            },
        )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=final_answer,
            evidence=evidence,
            trace=trace,
            implementation_identity=(
                f"thin_external_wrapper:RUC-NLPIR/SmartSearch@{SMARTSEARCH_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "profile_id": self.artifacts.profile_id,
                "artifact_identity_sha256": self.artifacts.identity_sha256,
                "checkpoint_repository": self.artifacts.checkpoint_repository,
                "checkpoint_revision": self.artifacts.checkpoint_revision,
                "retriever_revision": self.artifacts.retriever_revision,
                "corpus_sha256": self.artifacts.corpus_sha256,
                "index_sha256": self.artifacts.index_sha256,
                "termination": termination,
                "raw_transcript_sha256": transcript_hash,
                "worker_fingerprint": worker_identity,
                "runtime_attestation": attestation,
                "paper_native_parity_claimed": False,
                "seed_applied": False,
                "test_time_reward_used": False,
                "gold_answer_used": False,
                "external_query_refiner_used": False,
            },
        )


@dataclass(slots=True)
class SmartSearchCommonSupportQueries:
    """Expose SmartSearch search refinements to the shared downstream agent.

    The released SmartSearch checkpoint, prompt, parser, and ``ReCall`` loop run
    unchanged through :class:`SmartSearchFrozenInference`.  The component's
    terminal answer and retrieved document text are deliberately discarded.
    Unique search queries from the official trajectory are appended to the
    original request as retrieval refinements.  This is the paper-aligned
    reformulation treatment for the common-support arena, not a claim of
    end-to-end SmartSearch paper-result parity.
    """

    artifacts: SmartSearchArtifactIdentity
    client: WorkerClient | None = None
    controls: SmartSearchControls = SmartSearchControls()
    method_id: str = "smartsearch"
    release_id: str = SMARTSEARCH_COMMON_SUPPORT_RELEASE
    family: SemanticFamily = SemanticFamily.REFORMULATE

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, SmartSearchArtifactIdentity):
            raise ValidationError("SmartSearch common support requires frozen artifacts")
        if not isinstance(self.controls, SmartSearchControls):
            raise ValidationError("SmartSearch common-support controls have the wrong type")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("SmartSearch common support requires a non-empty query")
        # The official runtime accepts a question and nothing else.  Construct a
        # question-only view with the *same* execution budget; evaluator labels,
        # arena tools, prior messages, and captured context never cross the seam.
        component_state = FrozenAgentState(
            query=state.query,
            task_id=state.task_id,
            metadata={
                "common_support_source_state_hash": state.state_hash,
                "common_support_profile": SMARTSEARCH_COMMON_SUPPORT_RELEASE,
            },
            budget=state.budget,
        )
        component = SmartSearchFrozenInference(
            artifacts=self.artifacts,
            client=self.client,
            controls=self.controls,
        ).run(component_state, context)
        refinements: list[str] = []
        for event in component.trace.events:
            query = event.payload.get("query") if event.name == "search_step" else None
            if isinstance(query, str) and query.strip() and query not in refinements:
                refinements.append(query)
        evidence = tuple(component.evidence)
        if refinements:
            value = "SmartSearch retrieval refinements:\n" + "\n".join(
                f"- {query}" for query in refinements
            )
            operation = PatchOperation(
                PatchOp.APPEND_QUERY,
                value,
                target="smartsearch_search_query_refinements",
                rationale=(
                    "append unique search queries produced by the released SmartSearch "
                    "policy while retaining the original request"
                ),
            )
        else:
            operation = PatchOperation(
                PatchOp.NO_OP,
                target="smartsearch_no_search_refinement",
                rationale="the released SmartSearch policy terminated without a search query",
            )
        events = (
            TraceEvent(
                TracePhase.PREPARE,
                "smartsearch_common_support_boundary",
                {
                    "component_release_id": component.release_id,
                    "component_state_hash": component_state.state_hash,
                    "source_state_hash": state.state_hash,
                    "question_only_component": True,
                    "terminal_answer_forwarded": False,
                    "retrieved_evidence_forwarded": False,
                    "original_query_retained": True,
                    "paper_result_parity_claimed": False,
                },
            ),
            *component.trace.events,
            TraceEvent(
                TracePhase.FINALIZE,
                "smartsearch_query_refinement_patch",
                {
                    "refined_queries": tuple(refinements),
                    "refined_queries_sha256": content_hash(refinements),
                    "refinement_count": len(refinements),
                    "evidence_documents": len(evidence),
                    "component_result_sha256": component.result_hash,
                    "terminal_answer_sha256": _sha256_text(component.final_answer),
                    "terminal_answer_discarded": True,
                },
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                events=events,
                counters={
                    **dict(component.trace.counters),
                    "refinement_queries": len(refinements),
                    "internal_evidence_documents": len(evidence),
                },
                warnings=(
                    *component.trace.warnings,
                    "controlled_common_support_query_mapping_not_paper_result_parity",
                ),
            ),
            implementation_identity=(
                "query_refinement_common_support+" + component.implementation_identity
            ),
            profile_kind="controlled_adapter",
            metadata={
                **dict(component.metadata),
                "component_release_id": component.release_id,
                "component_result_sha256": component.result_hash,
                "common_support_adaptation": (
                    "discard_terminal_answer_and_evidence_append_search_queries"
                ),
                "terminal_answer_discarded": True,
                "retrieved_evidence_forwarded": False,
                "original_query_retained": True,
                "refined_queries": tuple(refinements),
                "gold_answer_used": False,
                "test_time_reward_used": False,
                "training_called": False,
                "paper_native_parity_claimed": False,
                "paper_result_parity_claimed": False,
            },
        )


def _remaining_budget(
    state: FrozenAgentState,
    context: InterventionContext,
) -> Mapping[str, int | None]:
    usage = None
    ledger = context.budget_ledger
    if ledger is not None:
        ledger_budget = getattr(ledger, "budget", None)
        usage = getattr(ledger, "usage", None)
        if ledger_budget != state.budget or usage is None:
            raise ValidationError("SmartSearch received an incompatible budget ledger")

    def consumed(name: str) -> int:
        value = 0 if usage is None else getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(f"SmartSearch budget usage {name} is invalid")
        return value

    remaining_context = None
    if state.budget.max_context_tokens is not None:
        remaining_context = max(
            0,
            state.budget.max_context_tokens - consumed("input_tokens"),
        )
    return freeze_mapping(
        {
            "model_calls": max(0, state.budget.max_model_calls - consumed("model_calls")),
            "retrieval_calls": max(
                0,
                state.budget.max_retrieval_calls - consumed("retrieval_calls"),
            ),
            "output_tokens": max(
                0,
                state.budget.max_output_tokens - consumed("output_tokens"),
            ),
            "context_tokens": remaining_context,
        }
    )


def _validate_runtime_attestation(
    payload: Mapping[str, Any],
    *,
    artifacts: SmartSearchArtifactIdentity,
) -> Mapping[str, Any]:
    value = payload.get("runtime_attestation")
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("SmartSearch requires runtime_attestation")
    expected = {
        "source_commit": SMARTSEARCH_UPSTREAM_COMMIT,
        "compatibility_manifest_sha256": artifacts.compatibility_manifest_sha256,
        "model_transport": artifacts.model_transport,
        "model_snapshot_tree_sha256": artifacts.model_snapshot_tree_sha256,
        "context_admission_fingerprint_sha256": (artifacts.context_admission_fingerprint_sha256),
        "contract_hashes_verified": True,
        "immutable_model_identity_handshake_required": True,
        "context_admission_before_generation": True,
        "official_loop_called": True,
        "outer_retry_disabled": True,
        "dangerous_sandbox_used": False,
        "training_code_loaded": False,
        "seed_applied": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AlgorithmOutputError(f"SmartSearch runtime attestation {key} mismatch")
    if value.get("source_attestation") not in {"git_checkout", "verified_archive_snapshot"}:
        raise AlgorithmOutputError("SmartSearch source attestation is unsupported")
    return freeze_mapping(value)


def _validate_usage_against_sent_budget(usage: Any, budget: Mapping[str, Any]) -> None:
    for usage_name, budget_name in (
        ("model_calls", "model_calls"),
        ("retrieval_calls", "retrieval_calls"),
        ("output_tokens", "output_tokens"),
    ):
        if getattr(usage, usage_name) > budget[budget_name]:
            raise AlgorithmOutputError(f"SmartSearch worker exceeded sent {budget_name} budget")
    context_cap = budget["context_tokens"]
    if context_cap is not None and usage.input_tokens > context_cap:
        raise AlgorithmOutputError("SmartSearch worker exceeded sent context_tokens budget")


def _validate_search_lineage(
    steps: tuple[SmartSearchStep, ...],
    evidence: tuple[_SmartSearchEvidence, ...],
    *,
    controls: SmartSearchControls,
) -> None:
    if len(steps) > controls.max_search_rounds:
        raise AlgorithmOutputError("SmartSearch exceeded max_search_rounds")
    if tuple(step.index for step in steps) != tuple(range(len(steps))):
        raise AlgorithmOutputError("SmartSearch trajectory indices must be contiguous from zero")
    for step in steps:
        if len(step.document_ids) > controls.max_results_per_round:
            raise AlgorithmOutputError("SmartSearch step exceeded max_results_per_round")
    if len(evidence) > controls.max_evidence_documents:
        raise AlgorithmOutputError("SmartSearch exceeded max_evidence_documents")

    evidence_ids = tuple(item.document_id for item in evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise AlgorithmOutputError("SmartSearch evidence contains duplicate document IDs")
    expected_order: list[str] = []
    first_seen: dict[str, int] = {}
    for step in steps:
        for document_id in step.document_ids:
            if document_id not in first_seen:
                first_seen[document_id] = step.index
                expected_order.append(document_id)
    if evidence_ids != tuple(expected_order):
        raise AlgorithmOutputError(
            "SmartSearch evidence must exactly follow first-seen trajectory order"
        )
    for item in evidence:
        if first_seen[item.document_id] != item.first_seen_step:
            raise AlgorithmOutputError("SmartSearch evidence first_seen_step mismatch")


def _require_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise AlgorithmOutputError(f"SmartSearch response requires {key}=false")


def _reject_forbidden_inference_keys(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_INFERENCE_KEYS:
                raise AlgorithmOutputError(f"{label} contains forbidden test-time field {key!r}")
            _reject_forbidden_inference_keys(item, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_forbidden_inference_keys(item, label=label)


__all__ = [
    "SMARTSEARCH_CHECKPOINT_REVISION",
    "SMARTSEARCH_LOOP_CONTRACT",
    "SMARTSEARCH_LOOP_CONTRACT_SHA256",
    "SMARTSEARCH_MODEL_REPOSITORY",
    "SMARTSEARCH_PARSER_CONTRACT_SHA256",
    "SMARTSEARCH_PROFILE_ID",
    "SMARTSEARCH_PROTOCOL",
    "SMARTSEARCH_TOOL_SCHEMA",
    "SMARTSEARCH_TOOL_SCHEMA_SHA256",
    "SMARTSEARCH_UPSTREAM_COMMIT",
    "SmartSearchArtifactIdentity",
    "SmartSearchControls",
    "SmartSearchFrozenInference",
    "SmartSearchStep",
]
