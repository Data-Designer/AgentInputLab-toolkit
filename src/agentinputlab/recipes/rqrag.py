"""RQ-RAG tree inference with controlled and official-checkpoint profiles.

The pinned upstream source and released checkpoint have no declared license.
The official-checkpoint adapter therefore remains fail-closed until explicit
license clearance and the three audited inference fixes are attested.

``RQRAGMethodTreeSearch`` is a separate, runnable controlled profile.  It keeps
the paper's four special-token actions, breadth-first query/action tree,
retrieval after every query action, and confidence/PPL/ensemble leaf selection,
while using AgentInputLab's replaceable ``LLMBackend`` and ``Retriever`` seams.
It is method-level reproduction infrastructure, not a claim of released-7B
checkpoint parity.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from agentinputlab.backends.llm.base import GenerationRequest, GenerationResponse, LLMBackend
from agentinputlab.backends.retrieval.base import RetrievalResult, Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CapabilityError,
    CompositeResult,
    ContextBlock,
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
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

RQ_RAG_UPSTREAM_COMMIT = "df5f2597eec9c229e1d54a6884bf236236870b32"
RQ_RAG_METHOD_RELEASE_ID = "recipe.rqrag.method_tree_search"
_UNCLEARED_LICENSE_IDS = frozenset({"", "NONE", "NOASSERTION", "UNKNOWN", "UNLICENSED"})
_QUERY_TYPES = frozenset({"rewrite", "decompose", "disambiguate", "direct_response"})
RQ_RAG_ACTION_TOKENS: Mapping[str, str] = freeze_mapping(
    {
        "rewrite": "[S_Rewritten_Query]",
        "decompose": "[S_Decomposed_Query]",
        "disambiguate": "[S_Disambiguated_Query]",
        "direct_response": "[A_Response]",
    }
)
_QUERY_ACTIONS = ("rewrite", "decompose", "disambiguate")
_RESERVED_PROTOCOL_TOKENS = (
    *RQ_RAG_ACTION_TOKENS.values(),
    "[EOS]",
    "[R_Evidences]",
    "[/R_Evidences]",
)


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be text")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True, slots=True)
class RQRAGLicenseClearance:
    """Auditable approval for source, checkpoint, and training-data terms."""

    source_license_id: str
    checkpoint_license_id: str
    dataset_license_id: str
    approved_by: str
    approval_reference: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_license_id",
            "checkpoint_license_id",
            "dataset_license_id",
        ):
            raw_license_id = getattr(self, field_name)
            if not isinstance(raw_license_id, str):
                raise ValidationError(f"RQ-RAG {field_name} must be text")
            license_id = raw_license_id.strip().upper()
            if license_id in _UNCLEARED_LICENSE_IDS:
                raise ValidationError(f"RQ-RAG {field_name} is not cleared")
        if (
            not isinstance(self.approved_by, str)
            or not isinstance(self.approval_reference, str)
            or not self.approved_by.strip()
            or not self.approval_reference.strip()
        ):
            raise ValidationError("RQ-RAG license clearance requires approver and reference")

    @property
    def clearance_id(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class RQRAGPatchAttestation:
    """Attest the minimal upstream fixes without redefining tree decoding."""

    patch_manifest_sha256: str
    attested_by: str
    attestation_reference: str
    tree_false_regression_passed: bool
    oracle_false_regression_passed: bool
    duplicate_stop_regression_passed: bool
    semantic_patch_only: bool
    upstream_commit: str = RQ_RAG_UPSTREAM_COMMIT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "patch_manifest_sha256",
            _require_sha256(
                self.patch_manifest_sha256,
                label="RQ-RAG patch_manifest_sha256",
            ),
        )
        if self.upstream_commit != RQ_RAG_UPSTREAM_COMMIT:
            raise ValidationError("RQ-RAG patch attestation targets the wrong upstream commit")
        if (
            not isinstance(self.attested_by, str)
            or not isinstance(self.attestation_reference, str)
            or not self.attested_by.strip()
            or not self.attestation_reference.strip()
        ):
            raise ValidationError("RQ-RAG patch attestation requires attester and reference")
        if any(
            value is not True
            for value in (
                self.tree_false_regression_passed,
                self.oracle_false_regression_passed,
                self.duplicate_stop_regression_passed,
                self.semantic_patch_only,
            )
        ):
            raise ValidationError("RQ-RAG patch attestation must pass all fail-closed gates")


@dataclass(frozen=True, slots=True)
class RQRAGControls:
    max_depth: int = 4
    max_nodes: int = 32
    max_documents: int = 100
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        values = (self.max_depth, self.max_nodes, self.max_documents, self.max_output_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValidationError("RQ-RAG limits must be integers")
        if min(*values) < 1:
            raise ValidationError("RQ-RAG limits must be positive")


@dataclass(frozen=True, slots=True)
class RQRAGTreeNode:
    node_id: str
    parent_id: str | None
    depth: int
    query_type: str
    control_token: str
    query: str
    document_ids: tuple[str, ...]
    score: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RQRAGTreeNode:
        parent = value.get("parent_id")
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            raise AlgorithmOutputError("RQ-RAG parent_id must be null or non-empty text")
        query_type = required_text(value, "query_type", label="RQ-RAG tree node")
        if query_type not in _QUERY_TYPES:
            raise AlgorithmOutputError("RQ-RAG tree node has an unsupported query_type")
        return cls(
            node_id=required_text(value, "node_id", label="RQ-RAG tree node"),
            parent_id=parent,
            depth=nonnegative_integer(value.get("depth"), label="RQ-RAG tree node depth"),
            query_type=query_type,
            control_token=required_text(
                value,
                "control_token",
                label="RQ-RAG tree node",
            ),
            query=required_text(value, "query", label="RQ-RAG tree node"),
            document_ids=string_tuple(
                value.get("document_ids", ()),
                label="RQ-RAG tree node document_ids",
                unique=True,
            ),
            score=finite_number(value.get("score"), label="RQ-RAG tree node score"),
        )


@dataclass(frozen=True, slots=True)
class _RQRAGEvidence:
    document_id: str
    text: str
    source: str
    score: float | None
    retrieved_by_node_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> _RQRAGEvidence:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise AlgorithmOutputError("RQ-RAG evidence metadata must be a mapping")
        score_value = value.get("score")
        return cls(
            document_id=required_text(value, "document_id", label="RQ-RAG evidence"),
            text=required_text(value, "text", label="RQ-RAG evidence"),
            source=required_text(value, "source", label="RQ-RAG evidence"),
            score=(
                None
                if score_value is None
                else finite_number(score_value, label="RQ-RAG evidence score")
            ),
            retrieved_by_node_ids=string_tuple(
                value.get("retrieved_by_node_ids"),
                label="RQ-RAG evidence retrieved_by_node_ids",
                allow_empty=False,
                unique=True,
            ),
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
                "rqrag_retrieved_by_node_ids": self.retrieved_by_node_ids,
            },
        )


@dataclass(frozen=True, slots=True)
class RQRAGMethodControls:
    """Controls for the backend-neutral method-level tree search.

    ``max_depth`` counts retrieval actions, not the synthetic root or terminal
    answer nodes.  A depth of one therefore evaluates a direct answer plus one
    rewrite, decomposition, and disambiguation branch, followed by an answer
    for each branch.
    """

    max_depth: int = 1
    max_nodes: int = 32
    max_documents: int = 100
    retrieval_top_k: int = 5
    max_output_tokens_per_call: int = 256
    query_temperature: float = 1.0
    answer_temperature: float = 0.0
    selection_mode: str = "confidence"
    query_actions: tuple[str, ...] = _QUERY_ACTIONS
    allow_partial_tree: bool = False

    def __post_init__(self) -> None:
        integer_values = (
            self.max_depth,
            self.max_nodes,
            self.max_documents,
            self.retrieval_top_k,
            self.max_output_tokens_per_call,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise ValidationError("RQ-RAG method controls must be integers")
        if self.max_depth < 0:
            raise ValidationError("RQ-RAG method max_depth must be non-negative")
        if self.max_nodes < 2:
            raise ValidationError("RQ-RAG method max_nodes must allow a root and answer")
        if (
            min(
                self.max_documents,
                self.retrieval_top_k,
                self.max_output_tokens_per_call,
            )
            < 1
        ):
            raise ValidationError("RQ-RAG method document/token limits must be positive")
        temperatures = (self.query_temperature, self.answer_temperature)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in temperatures
        ):
            raise ValidationError("RQ-RAG method temperatures must be finite and non-negative")
        if self.selection_mode not in {"confidence", "ppl", "ensemble"}:
            raise ValidationError("RQ-RAG selection_mode must be confidence, ppl, or ensemble")
        if not isinstance(self.allow_partial_tree, bool):
            raise ValidationError("RQ-RAG allow_partial_tree must be boolean")
        actions = tuple(self.query_actions)
        if (
            not actions
            or len(set(actions)) != len(actions)
            or any(action not in _QUERY_ACTIONS for action in actions)
        ):
            raise ValidationError(
                "RQ-RAG query_actions must be unique rewrite/decompose/disambiguate actions"
            )
        object.__setattr__(self, "query_actions", actions)


@dataclass(frozen=True, slots=True)
class RQRAGCandidateScore:
    """Model-derived scores used by the paper's post-hoc leaf selectors."""

    confidence: float | None
    perplexity: float | None
    source: str

    def __post_init__(self) -> None:
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise AlgorithmOutputError("RQ-RAG confidence must be finite in [0, 1]")
        if self.perplexity is not None and (
            isinstance(self.perplexity, bool)
            or not isinstance(self.perplexity, (int, float))
            or not math.isfinite(self.perplexity)
            or self.perplexity <= 0
        ):
            raise AlgorithmOutputError("RQ-RAG perplexity must be finite and positive")
        if not isinstance(self.source, str) or not self.source.strip():
            raise AlgorithmOutputError("RQ-RAG candidate score requires a source")


@dataclass(frozen=True, slots=True)
class _MethodPath:
    node_id: str
    retrieval_depth: int
    prompt: str


@dataclass(frozen=True, slots=True)
class _AnswerCandidate:
    node_id: str
    answer: str
    score: RQRAGCandidateScore
    ordinal: int


@dataclass(slots=True)
class _EvidenceAccumulator:
    document_id: str
    text: str
    source: str
    score: float
    metadata: Mapping[str, Any]
    node_ids: list[str]


@dataclass(slots=True)
class RQRAGMethodTreeSearch:
    """Run RQ-RAG's action tree over replaceable model/retrieval backends.

    The injected model must return a score for terminal answers, either through
    ``raw["rqrag_score"]`` (``confidence`` and/or ``perplexity``) or through
    OpenAI-compatible token log-probabilities.  Perplexity must explicitly have
    ``perplexity_scope="full_path"`` because upstream scores the whole candidate
    path; completion-only API log-probabilities are accepted for controlled
    confidence selection only.  This keeps leaf selection model-derived and
    prevents an unreported lexical heuristic from changing the treatment.
    """

    controls: RQRAGMethodControls = RQRAGMethodControls()
    method_id: str = "recipe.rqrag"
    release_id: str = RQ_RAG_METHOD_RELEASE_ID
    family: SemanticFamily = SemanticFamily.COMPOSITE

    @staticmethod
    def _require_backend(value: object) -> LLMBackend:
        if value is None:
            raise ExternalDependencyError("RQ-RAG method tree search requires an LLMBackend")
        if (
            not hasattr(value, "generate")
            or not hasattr(value, "fingerprint")
            or not hasattr(value, "model_id")
        ):
            raise CapabilityError("RQ-RAG backend does not implement LLMBackend")
        return value  # type: ignore[return-value]

    @staticmethod
    def _require_retriever(value: object) -> Retriever:
        if value is None:
            raise ExternalDependencyError("RQ-RAG method tree search requires a Retriever")
        if (
            not hasattr(value, "retrieve")
            or not hasattr(value, "fingerprint")
            or not hasattr(value, "retriever_id")
        ):
            raise CapabilityError("RQ-RAG retriever does not implement Retriever")
        return value  # type: ignore[return-value]

    def run(
        self,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> CompositeResult:
        if not state.query.strip():
            raise ValidationError("RQ-RAG requires a non-empty query")
        if state.tool_names:
            raise ValidationError(
                "RQ-RAG method tree search does not consume tool schemas; use a retrieval backend"
            )
        _reject_protocol_tokens(state.query, label="RQ-RAG input query")
        backend = self._require_backend(context.backend)
        retriever = self._require_retriever(context.retriever)
        backend_identity = _dependency_identity(backend.fingerprint(), label="RQ-RAG backend")
        retriever_identity = _dependency_identity(
            retriever.fingerprint(),
            label="RQ-RAG retriever",
        )

        initial_prompt = _method_initial_prompt(state)
        root = RQRAGTreeNode(
            node_id="rqrag-node-0000",
            parent_id=None,
            depth=0,
            query_type="root",
            control_token="[ROOT]",
            query=state.query.strip(),
            document_ids=(),
            score=0.0,
        )
        nodes: list[RQRAGTreeNode] = [root]
        frontier: deque[_MethodPath] = deque((_MethodPath(root.node_id, 0, initial_prompt),))
        candidates: list[_AnswerCandidate] = []
        evidence_accumulators: dict[str, _EvidenceAccumulator] = {}
        evidence_order: list[str] = []
        model_calls = 0
        retrieval_calls = 0
        input_tokens = 0
        output_tokens = 0
        known_cost_usd = 0.0
        unpriced_model_calls = 0
        escaped_evidence_tokens = 0
        incomplete = False
        incomplete_reasons: set[str] = set()

        def can_generate() -> bool:
            return (
                model_calls < state.budget.max_model_calls
                and output_tokens < state.budget.max_output_tokens
                and len(nodes) < self.controls.max_nodes
            )

        def generate_action(
            path: _MethodPath,
            action: str,
        ) -> tuple[str, RQRAGCandidateScore, GenerationResponse]:
            nonlocal model_calls
            nonlocal input_tokens
            nonlocal output_tokens
            nonlocal known_cost_usd
            nonlocal unpriced_model_calls
            if not can_generate():
                raise BudgetExceededError("RQ-RAG has no remaining model/tree budget")
            control_token = RQ_RAG_ACTION_TOKENS[action]
            remaining_output = state.budget.max_output_tokens - output_tokens
            request = GenerationRequest(
                prompt=f"{path.prompt}{control_token}",
                temperature=(
                    self.controls.answer_temperature
                    if action == "direct_response"
                    else self.controls.query_temperature
                ),
                max_tokens=min(
                    self.controls.max_output_tokens_per_call,
                    remaining_output,
                ),
                seed=context.seed + model_calls,
                stop=("[EOS]",),
                logprobs=(
                    action == "direct_response"
                    and self.controls.selection_mode in {"confidence", "ensemble"}
                ),
                top_logprobs=(
                    5
                    if action == "direct_response"
                    and self.controls.selection_mode in {"confidence", "ensemble"}
                    else None
                ),
                metadata={
                    "method_id": self.method_id,
                    "release_id": self.release_id,
                    "rqrag_action": action,
                    "rqrag_control_token": control_token,
                    "rqrag_parent_node_id": path.node_id,
                    "rqrag_retrieval_depth": path.retrieval_depth,
                    "paper_checkpoint_parity": False,
                },
            )
            response = backend.generate(request)
            model_calls += 1
            if not isinstance(response, GenerationResponse):
                raise AlgorithmOutputError("RQ-RAG backend must return a GenerationResponse")
            generated = _parse_action_completion(
                response.text,
                action=action,
                control_token=control_token,
            )
            score = _candidate_score(response)
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            if response.usage.output_tokens > request.max_tokens:
                raise BudgetExceededError(
                    "RQ-RAG backend reported more output tokens than requested"
                )
            if output_tokens > state.budget.max_output_tokens:
                raise BudgetExceededError("RQ-RAG exceeded max_output_tokens")
            if (
                state.budget.max_context_tokens is not None
                and input_tokens > state.budget.max_context_tokens
            ):
                raise BudgetExceededError("RQ-RAG exceeded max_context_tokens")
            call_cost = response.usage.cost_usd
            pricing = context.pricing_catalog
            if (
                call_cost is None
                and pricing is not None
                and hasattr(pricing, "has_price")
                and pricing.has_price(backend.model_id)
            ):
                call_cost = pricing.estimate(
                    backend.model_id,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            if call_cost is None:
                unpriced_model_calls += 1
                if state.budget.max_cost_usd is not None:
                    raise BudgetExceededError(
                        "RQ-RAG cannot enforce max_cost_usd for an unpriced model call"
                    )
            else:
                known_cost_usd += call_cost
                if (
                    state.budget.max_cost_usd is not None
                    and known_cost_usd > state.budget.max_cost_usd
                ):
                    raise BudgetExceededError("RQ-RAG exceeded max_cost_usd")
            return generated, score, response

        while frontier:
            path = frontier.popleft()
            if not can_generate():
                incomplete = True
                incomplete_reasons.add("model_or_tree_budget")
                break

            answer, answer_score, _ = generate_action(path, "direct_response")
            _require_selection_score(answer_score, self.controls.selection_mode)
            answer_node = RQRAGTreeNode(
                node_id=_next_node_id(len(nodes)),
                parent_id=path.node_id,
                depth=path.retrieval_depth + 1,
                query_type="direct_response",
                control_token=RQ_RAG_ACTION_TOKENS["direct_response"],
                query=answer,
                document_ids=(),
                score=_tree_score(answer_score, self.controls.selection_mode),
            )
            nodes.append(answer_node)
            candidates.append(
                _AnswerCandidate(
                    node_id=answer_node.node_id,
                    answer=answer,
                    score=answer_score,
                    ordinal=len(candidates),
                )
            )

            if path.retrieval_depth >= self.controls.max_depth:
                continue
            for action in self.controls.query_actions:
                if retrieval_calls >= state.budget.max_retrieval_calls:
                    incomplete = True
                    incomplete_reasons.add("retrieval_budget")
                    break
                if not can_generate():
                    incomplete = True
                    incomplete_reasons.add("model_or_tree_budget")
                    break
                query, query_score, _ = generate_action(path, action)
                raw_results = tuple(retriever.retrieve(query, top_k=self.controls.retrieval_top_k))
                retrieval_calls += 1
                results = _validate_retrieval_results(
                    raw_results,
                    top_k=self.controls.retrieval_top_k,
                )
                query_node_id = _next_node_id(len(nodes))
                document_ids: list[str] = []
                formatted_evidence: list[str] = []
                for result in results:
                    document = result.document
                    accumulator = evidence_accumulators.get(document.doc_id)
                    if accumulator is None:
                        if len(evidence_accumulators) >= self.controls.max_documents:
                            incomplete = True
                            incomplete_reasons.add("document_cap")
                            continue
                        accumulator = _EvidenceAccumulator(
                            document_id=document.doc_id,
                            text=document.text,
                            source=f"rqrag:{retriever.retriever_id}",
                            score=float(result.score),
                            metadata={
                                "title": document.title,
                                "corpus_revision": document.corpus_revision,
                                "rank": result.rank,
                            },
                            node_ids=[],
                        )
                        evidence_accumulators[document.doc_id] = accumulator
                        evidence_order.append(document.doc_id)
                    else:
                        _validate_repeated_document(accumulator, result)
                    if query_node_id not in accumulator.node_ids:
                        accumulator.node_ids.append(query_node_id)
                    if document.doc_id not in document_ids:
                        document_ids.append(document.doc_id)
                    title = document.title.strip() or document.doc_id
                    escaped_title, title_count = _escape_protocol_tokens(title)
                    escaped_text, text_count = _escape_protocol_tokens(document.text)
                    escaped_evidence_tokens += title_count + text_count
                    formatted_evidence.append(f"Title: {escaped_title}\nText: {escaped_text}")
                query_node = RQRAGTreeNode(
                    node_id=query_node_id,
                    parent_id=path.node_id,
                    depth=path.retrieval_depth + 1,
                    query_type=action,
                    control_token=RQ_RAG_ACTION_TOKENS[action],
                    query=query,
                    document_ids=tuple(document_ids),
                    score=_tree_score(query_score, self.controls.selection_mode),
                )
                nodes.append(query_node)
                evidence_text = "\n".join(formatted_evidence)
                next_prompt = (
                    f"{path.prompt}{RQ_RAG_ACTION_TOKENS[action]}{query}[EOS]"
                    f"[R_Evidences]{evidence_text}[/R_Evidences]\n"
                )
                frontier.append(
                    _MethodPath(
                        node_id=query_node.node_id,
                        retrieval_depth=path.retrieval_depth + 1,
                        prompt=next_prompt,
                    )
                )

        if not candidates:
            raise BudgetExceededError("RQ-RAG could not generate any terminal answer")
        if incomplete and not self.controls.allow_partial_tree:
            reasons = ", ".join(sorted(incomplete_reasons)) or "unknown"
            raise BudgetExceededError(
                "RQ-RAG tree is incomplete and non-comparable under the current budget "
                f"({reasons}); set allow_partial_tree=True only for explicitly anytime runs"
            )
        selected = _select_candidate(candidates, self.controls.selection_mode)
        node_by_id = {node.node_id: node for node in nodes}
        selected_path = _node_ancestry(selected.node_id, node_by_id)
        validation_controls = RQRAGControls(
            max_depth=self.controls.max_depth + 1,
            max_nodes=self.controls.max_nodes,
            max_documents=self.controls.max_documents,
            max_output_tokens=max(1, state.budget.max_output_tokens),
        )
        _validate_tree(
            tuple(nodes),
            selected_node_id=selected.node_id,
            selected_path=selected_path,
            controls=validation_controls,
        )
        evidence_records = tuple(
            _RQRAGEvidence(
                document_id=document_id,
                text=evidence_accumulators[document_id].text,
                source=evidence_accumulators[document_id].source,
                score=evidence_accumulators[document_id].score,
                retrieved_by_node_ids=tuple(evidence_accumulators[document_id].node_ids),
                metadata=evidence_accumulators[document_id].metadata,
            )
            for document_id in evidence_order
        )
        evidence_by_id = _validate_document_lineage(tuple(nodes), evidence_records)
        selected_document_ids = _selected_document_order(tuple(nodes), selected_path)
        evidence = tuple(
            evidence_by_id[document_id].as_context_block() for document_id in selected_document_ids
        )
        termination = "budget_exhausted" if incomplete else "selected_leaf"
        trace_events: list[TraceEvent] = [
            TraceEvent(
                TracePhase.PREPARE,
                "method_tree_configuration",
                {
                    "upstream_commit": RQ_RAG_UPSTREAM_COMMIT,
                    "action_tokens": dict(RQ_RAG_ACTION_TOKENS),
                    "controls": asdict(self.controls),
                    "backend": backend_identity,
                    "retriever": retriever_identity,
                    "input_context_blocks": len(state.context),
                    "input_messages": len(state.messages),
                    "metadata_forwarded": False,
                    "paper_checkpoint_parity": False,
                },
            )
        ]
        trace_events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "tree_node",
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "query_type": node.query_type,
                    "control_token": node.control_token,
                    "query": node.query,
                    "document_ids": node.document_ids,
                    "score": node.score,
                },
            )
            for node in nodes
        )
        trace_events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "tree_selection",
                {
                    "selected_node_id": selected.node_id,
                    "selected_path": selected_path,
                    "selected_document_ids": selected_document_ids,
                    "selection_mode": self.controls.selection_mode,
                    "allow_partial_tree": self.controls.allow_partial_tree,
                    "confidence": selected.score.confidence,
                    "perplexity": selected.score.perplexity,
                    "score_source": selected.score.source,
                    "termination": termination,
                    "incomplete_reasons": tuple(sorted(incomplete_reasons)),
                },
            )
        )
        warnings = ["controlled method-level RQ-RAG; no released-7B checkpoint parity claim"]
        if incomplete:
            warnings.append("partial anytime tree; exclude from complete-tree confirmatory cells")
        if unpriced_model_calls:
            warnings.append("one or more model calls did not report price")
        if escaped_evidence_tokens:
            warnings.append("reserved RQ-RAG tokens in retrieved text were prompt-escaped")
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(trace_events),
            counters={
                "model_calls": model_calls,
                "retrieval_calls": retrieval_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": known_cost_usd,
                "unpriced_model_calls": unpriced_model_calls,
                "tree_nodes": len(nodes),
                "tree_depth": max(node.depth for node in nodes),
                "answer_candidates": len(candidates),
                "selected_documents": len(evidence),
                "escaped_evidence_tokens": escaped_evidence_tokens,
            },
            warnings=tuple(warnings),
        )
        treatment_identity = content_hash(
            {
                "upstream_commit": RQ_RAG_UPSTREAM_COMMIT,
                "release_id": self.release_id,
                "controls": asdict(self.controls),
                "backend": backend_identity,
                "retriever": retriever_identity,
            }
        )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=selected.answer,
            evidence=evidence,
            trace=trace,
            implementation_identity=(
                f"controlled_method_tree:chanchimin/RQ-RAG@{RQ_RAG_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "implementation_origin": "controlled_clean_room_method_tree",
                "paper_checkpoint_parity": False,
                "treatment_identity_sha256": treatment_identity,
                "backend_fingerprint": backend_identity,
                "retriever_fingerprint": retriever_identity,
                "selected_node_id": selected.node_id,
                "selected_path": selected_path,
                "selected_document_ids": selected_document_ids,
                "selection_mode": self.controls.selection_mode,
                "allow_partial_tree": self.controls.allow_partial_tree,
                "comparison_eligible": not incomplete,
                "score_source": selected.score.source,
                "termination": termination,
                "incomplete_reasons": tuple(sorted(incomplete_reasons)),
                "oracle": False,
                "tree_decode": True,
            },
        )


@dataclass(slots=True)
class RQRAGReleased7BExternal:
    """Preserve released RQ-RAG tree inference behind explicit legal gates."""

    client: WorkerClient | None = None
    checkpoint_revision: str = ""
    tokenizer_revision: str = ""
    retriever_revision: str = ""
    token_protocol_sha256: str = ""
    license_clearance: RQRAGLicenseClearance | None = None
    patch_attestation: RQRAGPatchAttestation | None = None
    controls: RQRAGControls = RQRAGControls()
    method_id: str = "recipe.rqrag"
    release_id: str = "recipe.rqrag.released_7b_external"
    family: SemanticFamily = SemanticFamily.COMPOSITE

    def __post_init__(self) -> None:
        for name in ("checkpoint_revision", "tokenizer_revision", "retriever_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"RQ-RAG {name} must not be empty")
        self.token_protocol_sha256 = _require_sha256(
            self.token_protocol_sha256,
            label="RQ-RAG token_protocol_sha256",
        )

    def run(
        self,
        state: FrozenAgentState,
        context: InterventionContext,
    ) -> CompositeResult:
        if self.license_clearance is None:
            raise ExternalDependencyError(
                "RQ-RAG is license-gated; source/checkpoint/data clearance is required"
            )
        if self.patch_attestation is None:
            raise ExternalDependencyError("RQ-RAG requires the audited three-fix patch attestation")
        if not state.query.strip():
            raise ValidationError("RQ-RAG requires a non-empty query")
        worker = resolve_worker(self.client, context, label="RQ-RAG")
        identity = worker_fingerprint(worker, label="RQ-RAG")
        request = MethodWorkerRequest(
            method_id=self.method_id,
            release_id=self.release_id,
            upstream_commit=RQ_RAG_UPSTREAM_COMMIT,
            state_hash=state.state_hash,
            payload={
                "task_id": state.task_id,
                "query": state.query,
                "instruction": state.instruction,
                "messages": state.messages,
                "checkpoint_revision": self.checkpoint_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "retriever_revision": self.retriever_revision,
                "token_protocol_sha256": self.token_protocol_sha256,
                "license_clearance_id": self.license_clearance.clearance_id,
                "patch_attestation": asdict(self.patch_attestation),
                "controls": asdict(self.controls),
                "seed": context.seed,
                "oracle": False,
                "tree_decode": True,
            },
            dependency_fingerprints={"worker": identity},
            minimum_reservation={"model_calls": 1},
            protocol_version="agentinputlab.rqrag.v1",
        )
        response = call_worker(worker, request)
        payload = response.payload
        for key, expected in (
            ("checkpoint_revision", self.checkpoint_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("retriever_revision", self.retriever_revision),
            ("token_protocol_sha256", self.token_protocol_sha256),
            ("license_clearance_id", self.license_clearance.clearance_id),
            ("patch_manifest_sha256", self.patch_attestation.patch_manifest_sha256),
        ):
            if required_text(payload, key, label="RQ-RAG response") != expected:
                raise AlgorithmOutputError(f"RQ-RAG {key} mismatch")
        if payload.get("tree_decode") is not True or payload.get("oracle") is not False:
            raise AlgorithmOutputError("RQ-RAG must use non-oracle tree decoding")
        _validate_regression_echo(payload.get("regression_gates"))

        nodes = tuple(
            RQRAGTreeNode.from_mapping(row)
            for row in mapping_sequence(payload.get("nodes"), label="RQ-RAG nodes")
        )
        selected_node_id = required_text(
            payload,
            "selected_node_id",
            label="RQ-RAG response",
        )
        selected_path = string_tuple(
            payload.get("selected_path"),
            label="RQ-RAG selected_path",
            allow_empty=False,
            unique=True,
        )
        _validate_tree(
            nodes,
            selected_node_id=selected_node_id,
            selected_path=selected_path,
            controls=self.controls,
        )
        evidence_records = tuple(
            _RQRAGEvidence.from_mapping(row)
            for row in mapping_sequence(payload.get("evidence"), label="RQ-RAG evidence")
        )
        evidence_by_id = _validate_document_lineage(nodes, evidence_records)
        if len(evidence_records) > self.controls.max_documents:
            raise AlgorithmOutputError("RQ-RAG tree evidence exceeded max_documents")
        selected_document_ids = string_tuple(
            payload.get("selected_document_ids"),
            label="RQ-RAG selected_document_ids",
            unique=True,
        )
        expected_selected = _selected_document_order(nodes, selected_path)
        if selected_document_ids != expected_selected:
            raise AlgorithmOutputError(
                "RQ-RAG selected documents disagree with the selected tree path"
            )
        if len(selected_document_ids) > self.controls.max_documents:
            raise AlgorithmOutputError("RQ-RAG selected evidence exceeds max_documents")

        retrieval_node_ids = string_tuple(
            payload.get("retrieval_call_node_ids"),
            label="RQ-RAG retrieval_call_node_ids",
            unique=True,
        )
        expected_retrieval_nodes = tuple(
            node.node_id for node in nodes if node.query_type != "direct_response"
        )
        if retrieval_node_ids != expected_retrieval_nodes:
            raise AlgorithmOutputError(
                "RQ-RAG retrieval-call lineage disagrees with non-terminal query nodes"
            )
        if response.usage.retrieval_calls != len(retrieval_node_ids):
            raise AlgorithmOutputError("RQ-RAG retrieval usage disagrees with tree trajectory")
        if response.usage.model_calls < 1:
            raise AlgorithmOutputError("RQ-RAG inference requires a model call")

        selection_mode = required_text(payload, "selection_mode", label="RQ-RAG response")
        if selection_mode not in {"ppl", "confidence", "ensemble"}:
            raise AlgorithmOutputError("RQ-RAG selection_mode is unsupported")
        final_answer = required_text(payload, "final_answer", label="RQ-RAG response")
        termination = required_text(payload, "termination", label="RQ-RAG response")
        if termination not in {"selected_leaf", "budget_exhausted", "model_stop"}:
            raise AlgorithmOutputError("RQ-RAG termination is unsupported")
        if termination == "selected_leaf" and any(
            node.parent_id == selected_node_id for node in nodes
        ):
            raise AlgorithmOutputError("RQ-RAG selected_leaf termination selected a non-leaf")

        evidence = tuple(evidence_by_id[item].as_context_block() for item in selected_document_ids)
        trace_events: list[TraceEvent] = [
            TraceEvent(
                TracePhase.PREPARE,
                "gated_tree_inference",
                {
                    "license_clearance_id": self.license_clearance.clearance_id,
                    "patch_manifest_sha256": self.patch_attestation.patch_manifest_sha256,
                    "token_protocol_sha256": self.token_protocol_sha256,
                },
            )
        ]
        trace_events.extend(
            TraceEvent(
                TracePhase.OBSERVE,
                "tree_node",
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "depth": node.depth,
                    "query_type": node.query_type,
                    "control_token": node.control_token,
                    "query": node.query,
                    "document_ids": node.document_ids,
                    "score": node.score,
                },
            )
            for node in nodes
        )
        trace_events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "tree_selection",
                {
                    "selected_node_id": selected_node_id,
                    "selected_path": selected_path,
                    "selected_document_ids": selected_document_ids,
                    "selection_mode": selection_mode,
                    "termination": termination,
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
                "tree_nodes": len(nodes),
                "tree_depth": max(node.depth for node in nodes),
                "selected_documents": len(evidence),
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
                f"thin_external_wrapper:chanchimin/RQ-RAG@{RQ_RAG_UPSTREAM_COMMIT}"
            ),
            profile_kind="official_native",
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "license_clearance_id": self.license_clearance.clearance_id,
                "patch_manifest_sha256": self.patch_attestation.patch_manifest_sha256,
                "checkpoint_revision": self.checkpoint_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "retriever_revision": self.retriever_revision,
                "token_protocol_sha256": self.token_protocol_sha256,
                "selected_node_id": selected_node_id,
                "selected_path": selected_path,
                "selection_mode": selection_mode,
                "termination": termination,
                "worker_fingerprint": identity,
                "license_gate": "explicitly_cleared_external_only",
            },
        )


def _dependency_identity(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise CapabilityError(f"{label} fingerprint must be a non-empty mapping")
    try:
        frozen = freeze_mapping(value)
        content_hash(frozen)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} fingerprint is not canonical: {exc}") from exc
    return frozen


def _reject_protocol_tokens(value: str, *, label: str) -> None:
    found = tuple(token for token in _RESERVED_PROTOCOL_TOKENS if token in value)
    if found:
        raise ValidationError(f"{label} contains reserved RQ-RAG protocol tokens: {found}")


def _method_initial_prompt(state: FrozenAgentState) -> str:
    _reject_protocol_tokens(state.instruction, label="RQ-RAG instruction")
    for index, message in enumerate(state.messages):
        _reject_protocol_tokens(
            message.content,
            label=f"RQ-RAG message {index} content",
        )
    for index, block in enumerate(state.context):
        _reject_protocol_tokens(
            block.text,
            label=f"RQ-RAG input context {index}",
        )
    system = state.instruction.strip() or (
        "Answer the question concisely. Refine or decompose the query and use "
        "retrieved evidence when it improves the answer."
    )
    parts = [f"<s><|system|>\n{system}</s>\n"]
    if state.messages:
        parts.append("<|history|>\n")
        parts.extend(f"{message.role}: {message.content}\n" for message in state.messages)
        parts.append("</|history|>\n")
    if state.context:
        parts.append("<|provided_context|>\n")
        parts.extend(
            f"Context {index + 1}: {block.text}\n" for index, block in enumerate(state.context)
        )
        parts.append("</|provided_context|>\n")
    parts.append(f"<|user|>\n{state.query.strip()}</s>\n<|assistant|>\n")
    return "".join(parts)


def _parse_action_completion(
    value: object,
    *,
    action: str,
    control_token: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlgorithmOutputError(f"RQ-RAG {action} generation returned empty text")
    text = value.strip()
    if control_token in text:
        text = text.rsplit(control_token, 1)[1].lstrip()
    if "[EOS]" in text:
        text, trailing = text.split("[EOS]", 1)
        if trailing.strip():
            raise AlgorithmOutputError("RQ-RAG completion has content after [EOS]")
    text = text.strip()
    if not text:
        raise AlgorithmOutputError(f"RQ-RAG {action} generation parsed as empty")
    unexpected = tuple(token for token in _RESERVED_PROTOCOL_TOKENS if token in text)
    if unexpected:
        raise AlgorithmOutputError(
            f"RQ-RAG {action} completion contains nested protocol tokens: {unexpected}"
        )
    return text


def _optional_finite_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AlgorithmOutputError(f"{label} must be a finite number")
    return float(value)


def _candidate_score(response: GenerationResponse) -> RQRAGCandidateScore:
    raw = response.raw
    explicit: object = raw.get("rqrag_score")
    source = "response.raw.rqrag_score"
    if explicit is None:
        choices = raw.get("choices")
        if (
            isinstance(choices, Sequence)
            and not isinstance(choices, (str, bytes))
            and choices
            and isinstance(choices[0], Mapping)
        ):
            explicit = choices[0].get("rqrag_score")
            source = "response.raw.choices[0].rqrag_score"
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise AlgorithmOutputError("RQ-RAG rqrag_score must be a mapping")
        confidence = _optional_finite_number(
            explicit.get("confidence"),
            label="RQ-RAG confidence",
        )
        perplexity = _optional_finite_number(
            explicit.get("perplexity"),
            label="RQ-RAG perplexity",
        )
        if perplexity is not None and explicit.get("perplexity_scope") != "full_path":
            raise AlgorithmOutputError(
                "RQ-RAG perplexity must declare perplexity_scope='full_path'"
            )
        return RQRAGCandidateScore(confidence, perplexity, source)

    token_confidences = _openai_content_confidences(raw)
    if token_confidences:
        return RQRAGCandidateScore(
            confidence=sum(token_confidences) / len(token_confidences),
            perplexity=None,
            source="openai_compatible_completion_logprobs",
        )
    return RQRAGCandidateScore(None, None, "unreported")


def _openai_content_confidences(raw: Mapping[str, Any]) -> tuple[float, ...]:
    choices = raw.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        return ()
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, Mapping):
        return ()
    content = logprobs.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)) or not content:
        return ()
    values: list[float] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise AlgorithmOutputError("RQ-RAG token logprobs must be mappings")
        top_logprobs = item.get("top_logprobs")
        if (
            not isinstance(top_logprobs, Sequence)
            or isinstance(top_logprobs, (str, bytes))
            or not top_logprobs
        ):
            raise AlgorithmOutputError(
                "RQ-RAG confidence scoring requires non-empty top_logprobs "
                "for every generated token"
            )
        candidate_logprobs: list[float] = []
        for top_item in top_logprobs:
            if not isinstance(top_item, Mapping):
                raise AlgorithmOutputError("RQ-RAG top token logprobs must be mappings")
            top_value = _optional_finite_number(
                top_item.get("logprob"),
                label="RQ-RAG top token logprob",
            )
            if top_value is not None:
                candidate_logprobs.append(top_value)
        logprob = max(candidate_logprobs) if candidate_logprobs else None
        if logprob is None:
            raise AlgorithmOutputError("RQ-RAG token logprob is missing")
        if logprob > 0:
            raise AlgorithmOutputError("RQ-RAG token logprob must not be positive")
        values.append(math.exp(logprob))
    return tuple(values)


def _require_selection_score(score: RQRAGCandidateScore, selection_mode: str) -> None:
    if selection_mode == "ppl" and score.perplexity is None:
        raise AlgorithmOutputError("RQ-RAG ppl selection requires model-derived perplexity")
    if selection_mode in {"confidence", "ensemble"} and score.confidence is None:
        raise AlgorithmOutputError(
            f"RQ-RAG {selection_mode} selection requires model-derived confidence"
        )


def _tree_score(score: RQRAGCandidateScore, selection_mode: str) -> float:
    if selection_mode == "ppl":
        if score.perplexity is not None:
            return -score.perplexity
        if score.confidence is not None:
            return score.confidence
        return 0.0
    if score.confidence is not None:
        return score.confidence
    if score.perplexity is not None:
        return 1.0 / score.perplexity
    return 0.0


def _select_candidate(
    candidates: Sequence[_AnswerCandidate],
    selection_mode: str,
) -> _AnswerCandidate:
    if not candidates:
        raise AlgorithmOutputError("RQ-RAG selection requires answer candidates")
    if selection_mode == "ppl":
        return min(
            candidates,
            key=lambda item: (
                float(item.score.perplexity),
                item.ordinal,
            ),
        )
    if selection_mode == "confidence":
        return max(
            candidates,
            key=lambda item: (
                float(item.score.confidence),
                -item.ordinal,
            ),
        )

    totals: dict[str, float] = {}
    first_ordinal: dict[str, int] = {}
    for candidate in candidates:
        totals.setdefault(candidate.answer, 0.0)
        totals[candidate.answer] += float(candidate.score.confidence)
        first_ordinal.setdefault(candidate.answer, candidate.ordinal)
    selected_answer = max(
        totals,
        key=lambda answer: (totals[answer], -first_ordinal[answer]),
    )
    return max(
        (candidate for candidate in candidates if candidate.answer == selected_answer),
        key=lambda item: (float(item.score.confidence), -item.ordinal),
    )


def _next_node_id(index: int) -> str:
    return f"rqrag-node-{index:04d}"


def _node_ancestry(
    node_id: str,
    node_by_id: Mapping[str, RQRAGTreeNode],
) -> tuple[str, ...]:
    path: list[str] = []
    seen: set[str] = set()
    cursor: str | None = node_id
    while cursor is not None:
        if cursor in seen or cursor not in node_by_id:
            raise AlgorithmOutputError("RQ-RAG tree ancestry is cyclic or incomplete")
        seen.add(cursor)
        path.append(cursor)
        cursor = node_by_id[cursor].parent_id
    return tuple(reversed(path))


def _validate_retrieval_results(
    results: tuple[RetrievalResult, ...],
    *,
    top_k: int,
) -> tuple[RetrievalResult, ...]:
    if len(results) > top_k:
        raise AlgorithmOutputError("RQ-RAG retriever returned more than top_k results")
    seen: set[str] = set()
    for expected_rank, result in enumerate(results, start=1):
        if not isinstance(result, RetrievalResult):
            raise AlgorithmOutputError("RQ-RAG retriever must return RetrievalResult instances")
        if result.rank != expected_rank:
            raise AlgorithmOutputError("RQ-RAG retrieval ranks must be contiguous")
        if (
            isinstance(result.score, bool)
            or not isinstance(result.score, (int, float))
            or not math.isfinite(result.score)
        ):
            raise AlgorithmOutputError("RQ-RAG retrieval scores must be finite")
        if not result.document.text.strip():
            raise AlgorithmOutputError("RQ-RAG retrieved documents must contain text")
        if result.document.doc_id in seen:
            raise AlgorithmOutputError("RQ-RAG retrieval results contain duplicate IDs")
        seen.add(result.document.doc_id)
    return results


def _validate_repeated_document(
    accumulator: _EvidenceAccumulator,
    result: RetrievalResult,
) -> None:
    document = result.document
    expected = {
        "text": accumulator.text,
        "title": accumulator.metadata.get("title", ""),
        "corpus_revision": accumulator.metadata.get("corpus_revision", ""),
    }
    observed = {
        "text": document.text,
        "title": document.title,
        "corpus_revision": document.corpus_revision,
    }
    if content_hash(expected) != content_hash(observed):
        raise AlgorithmOutputError("RQ-RAG retriever reused a document ID with different content")


def _escape_protocol_tokens(value: str) -> tuple[str, int]:
    escaped = value
    replacements = 0
    for token in _RESERVED_PROTOCOL_TOKENS:
        count = escaped.count(token)
        if count:
            escaped = escaped.replace(
                token,
                token.replace("[", "\uff3b").replace("]", "\uff3d"),
            )
            replacements += count
    return escaped, replacements


def _validate_regression_echo(value: object) -> None:
    if not isinstance(value, Mapping):
        raise AlgorithmOutputError("RQ-RAG response requires regression_gates")
    required = {
        "tree_false": True,
        "oracle_false": True,
        "duplicate_stop": True,
        "semantic_patch_only": True,
    }
    if set(value) != set(required) or any(
        value[key] is not expected for key, expected in required.items()
    ):
        raise AlgorithmOutputError("RQ-RAG worker did not attest every inference regression gate")


def _validate_tree(
    nodes: tuple[RQRAGTreeNode, ...],
    *,
    selected_node_id: str,
    selected_path: tuple[str, ...],
    controls: RQRAGControls,
) -> None:
    if not nodes:
        raise AlgorithmOutputError("RQ-RAG tree must contain at least one node")
    if len(nodes) > controls.max_nodes:
        raise AlgorithmOutputError("RQ-RAG tree exceeded max_nodes")
    node_by_id: dict[str, RQRAGTreeNode] = {}
    token_by_type: dict[str, str] = {}
    type_by_token: dict[str, str] = {}
    roots: list[str] = []
    for node in nodes:
        if node.node_id in node_by_id:
            raise AlgorithmOutputError("RQ-RAG tree contains duplicate node IDs")
        if node.query_type == "direct_response" and node.document_ids:
            raise AlgorithmOutputError("RQ-RAG direct_response nodes cannot retrieve documents")
        if node.depth > controls.max_depth:
            raise AlgorithmOutputError("RQ-RAG tree exceeded max_depth")
        if node.parent_id is None:
            if node.depth != 0:
                raise AlgorithmOutputError("RQ-RAG root depth must be zero")
            roots.append(node.node_id)
        else:
            parent = node_by_id.get(node.parent_id)
            if parent is None:
                raise AlgorithmOutputError("RQ-RAG parents must precede children")
            if node.depth != parent.depth + 1:
                raise AlgorithmOutputError("RQ-RAG child depth is inconsistent with parent")
        prior_token = token_by_type.setdefault(node.query_type, node.control_token)
        if prior_token != node.control_token:
            raise AlgorithmOutputError("RQ-RAG query type changed control tokens")
        prior_type = type_by_token.setdefault(node.control_token, node.query_type)
        if prior_type != node.query_type:
            raise AlgorithmOutputError("RQ-RAG control token maps to multiple query types")
        node_by_id[node.node_id] = node
    if len(roots) != 1:
        raise AlgorithmOutputError("RQ-RAG tree must contain exactly one root")
    if selected_node_id not in node_by_id:
        raise AlgorithmOutputError("RQ-RAG selected_node_id is absent from tree")
    derived: list[str] = []
    cursor: str | None = selected_node_id
    while cursor is not None:
        derived.append(cursor)
        cursor = node_by_id[cursor].parent_id
    expected_path = tuple(reversed(derived))
    if selected_path != expected_path or selected_path[0] != roots[0]:
        raise AlgorithmOutputError("RQ-RAG selected_path is not the selected node ancestry")


def _validate_document_lineage(
    nodes: tuple[RQRAGTreeNode, ...],
    evidence: tuple[_RQRAGEvidence, ...],
) -> Mapping[str, _RQRAGEvidence]:
    evidence_by_id: dict[str, _RQRAGEvidence] = {}
    expected_nodes: dict[str, list[str]] = {}
    expected_order: list[str] = []
    for node in nodes:
        for document_id in node.document_ids:
            if document_id not in expected_nodes:
                expected_nodes[document_id] = []
                expected_order.append(document_id)
            expected_nodes[document_id].append(node.node_id)
    for item in evidence:
        if item.document_id in evidence_by_id:
            raise AlgorithmOutputError("RQ-RAG evidence contains duplicate document IDs")
        evidence_by_id[item.document_id] = item
    if tuple(evidence_by_id) != tuple(expected_order):
        raise AlgorithmOutputError("RQ-RAG evidence must follow first-seen tree order")
    for document_id, node_ids in expected_nodes.items():
        if evidence_by_id[document_id].retrieved_by_node_ids != tuple(node_ids):
            raise AlgorithmOutputError("RQ-RAG evidence node lineage mismatch")
    return MappingProxyType(evidence_by_id)


def _selected_document_order(
    nodes: tuple[RQRAGTreeNode, ...],
    selected_path: tuple[str, ...],
) -> tuple[str, ...]:
    node_by_id = {node.node_id: node for node in nodes}
    ordered: list[str] = []
    seen: set[str] = set()
    for node_id in selected_path:
        for document_id in node_by_id[node_id].document_ids:
            if document_id not in seen:
                seen.add(document_id)
                ordered.append(document_id)
    return tuple(ordered)


__all__ = [
    "RQ_RAG_ACTION_TOKENS",
    "RQ_RAG_METHOD_RELEASE_ID",
    "RQ_RAG_UPSTREAM_COMMIT",
    "RQRAGCandidateScore",
    "RQRAGControls",
    "RQRAGLicenseClearance",
    "RQRAGMethodControls",
    "RQRAGMethodTreeSearch",
    "RQRAGPatchAttestation",
    "RQRAGReleased7BExternal",
    "RQRAGTreeNode",
]
