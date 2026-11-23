"""Controlled Adaptive-RAG runtime for the three shared paper arenas.

The native worker remains the strict upstream-router path.  This module is a
separate common-support profile: it preserves Adaptive-RAG's trained
question-only A/B/C router, first-decoder-step logit rule, and selected-arm
execution while replacing the unavailable released T5-large checkpoint with a
source-declared T5-family retraining artifact.  The substitution is explicit
in every trace and never counts as upstream router or paper-result parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentinputlab.backends.retrieval import RetrievalResult, Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    CapabilityError,
    ContextBlock,
    ExecutionBudget,
    ExecutionTrace,
    ExternalDependencyError,
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
from agentinputlab.contracts.serialization import freeze_mapping

from .adaptive_rag.runtime import ADAPTIVE_RAG_UPSTREAM_COMMIT
from .ircot_common_support import IRCoTCommonSupport, IRCoTCommonSupportConfig

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABELS = ("A", "B", "C")
_STRATEGIES = {"A": "no_retrieval", "B": "single_retrieval", "C": "ircot"}
_MANIFEST_NAME = "artifact-manifest.json"
_MANIFEST_SCHEMA = "agentinputlab.adaptive-rag-router-artifact.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AdaptiveRAGRoute:
    label: str
    logits: Mapping[str, float]
    input_tokens: int
    output_tokens: int = 1

    def __post_init__(self) -> None:
        if self.label not in _LABELS:
            raise ValidationError("Adaptive-RAG route label must be A, B, or C")
        logits = dict(self.logits)
        if set(logits) != set(_LABELS) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in logits.values()
        ):
            raise ValidationError("Adaptive-RAG route logits must contain finite A/B/C values")
        expected = max(_LABELS, key=lambda name: logits[name])
        if self.label != expected:
            raise ValidationError("Adaptive-RAG route violates A-first argmax")
        if (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or self.input_tokens < 1
        ):
            raise ValidationError("Adaptive-RAG router input_tokens must be positive")
        if isinstance(self.output_tokens, bool) or not isinstance(self.output_tokens, int):
            raise ValidationError("Adaptive-RAG router output_tokens must be an integer")
        if self.output_tokens != 1:
            raise ValidationError("Adaptive-RAG first-step router consumes one output token")
        object.__setattr__(self, "logits", freeze_mapping(logits))


@runtime_checkable
class AdaptiveRAGRouter(Protocol):
    def route(self, question: str) -> AdaptiveRAGRoute:
        """Return the first-step A/B/C decision for one visible question."""

    def fingerprint(self) -> Mapping[str, Any]:
        """Return a canonical router and training-artifact identity."""


@dataclass(slots=True)
class T5AdaptiveRAGRouter:
    """Lazy local-only T5 router with a closed artifact manifest."""

    artifact_root: Path
    manifest_sha256: str
    device: str = "cuda"
    _manifest: Mapping[str, Any] = field(init=False, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _tokenizer: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        unresolved_root = self.artifact_root.expanduser()
        if unresolved_root.is_symlink():
            raise ValidationError("Adaptive-RAG router artifact_root must be a real directory")
        root = unresolved_root.resolve()
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValidationError("Adaptive-RAG router artifact_root must be a real directory")
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise ValidationError("Adaptive-RAG router manifest_sha256 must be lowercase SHA-256")
        manifest_path = root / _MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValidationError("Adaptive-RAG router manifest is missing or unsafe")
        if _sha256_file(manifest_path) != self.manifest_sha256:
            raise ValidationError("Adaptive-RAG router manifest checksum mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError("Adaptive-RAG router manifest is invalid") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != _MANIFEST_SCHEMA:
            raise ValidationError("Adaptive-RAG router manifest schema mismatch")
        if manifest.get("upstream_commit") != ADAPTIVE_RAG_UPSTREAM_COMMIT:
            raise ValidationError("Adaptive-RAG router artifact targets another upstream commit")
        if manifest.get("route_rule") != "first_decoder_step_logit_argmax_A_first":
            raise ValidationError("Adaptive-RAG router route rule drifted")
        if manifest.get("strategy_map") != _STRATEGIES:
            raise ValidationError("Adaptive-RAG router strategy map drifted")
        if manifest.get("upstream_router_checkpoint_parity_claimed") is not False:
            raise ValidationError("Adaptive-RAG router artifact must not claim upstream parity")
        if manifest.get("paper_result_parity_claimed") is not False:
            raise ValidationError("Adaptive-RAG router artifact must not claim paper parity")
        files = manifest.get("artifact_files")
        if not isinstance(files, Mapping) or not files:
            raise ValidationError("Adaptive-RAG router artifact file closure is empty")
        normalized: dict[str, str] = {}
        for raw_name, raw_digest in files.items():
            if not isinstance(raw_name, str) or not raw_name or Path(raw_name).is_absolute():
                raise ValidationError("Adaptive-RAG router artifact has an unsafe relative path")
            relative = Path(raw_name)
            if ".." in relative.parts or _SHA256.fullmatch(str(raw_digest)) is None:
                raise ValidationError("Adaptive-RAG router artifact manifest entry is invalid")
            target = root / relative
            if not target.is_file() or target.is_symlink() or _sha256_file(target) != raw_digest:
                raise ValidationError(f"Adaptive-RAG router artifact closure mismatch: {raw_name}")
            normalized[raw_name] = str(raw_digest)
        required = {"config.json", "tokenizer_config.json"}
        if not required.issubset(normalized):
            raise ValidationError(
                "Adaptive-RAG router artifact lacks model/tokenizer configuration"
            )
        if not any(name.endswith((".safetensors", ".bin")) for name in normalized):
            raise ValidationError("Adaptive-RAG router artifact lacks model weights")
        actual_files = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValidationError("Adaptive-RAG router artifact has an unlisted symlink")
            if candidate.is_file() and candidate != manifest_path:
                actual_files.add(candidate.relative_to(root).as_posix())
        if actual_files != set(normalized):
            raise ValidationError("Adaptive-RAG router artifact has unlisted or missing files")
        object.__setattr__(self, "artifact_root", root)
        object.__setattr__(self, "_manifest", freeze_mapping(dict(manifest)))

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise ExternalDependencyError(
                    "Adaptive-RAG local router requires torch and transformers"
                ) from exc
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise ExternalDependencyError("Adaptive-RAG router requested unavailable CUDA")
            tokenizer = AutoTokenizer.from_pretrained(
                self.artifact_root,
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.artifact_root,
                local_files_only=True,
                trust_remote_code=False,
            ).to(self.device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
        return self._model, self._tokenizer

    def route(self, question: str) -> AdaptiveRAGRoute:
        if not question.strip():
            raise ValidationError("Adaptive-RAG router requires a non-empty question")
        try:
            import torch
        except ImportError as exc:
            raise ExternalDependencyError("Adaptive-RAG router requires torch") from exc
        model, tokenizer = self._load()
        maximum = int(self._manifest.get("max_seq_length", 384))
        encoded = tokenizer(
            question.strip(),
            truncation=True,
            max_length=maximum,
            return_tensors="pt",
        ).to(self.device)
        start_id = model.config.decoder_start_token_id
        if not isinstance(start_id, int):
            raise AlgorithmOutputError("Adaptive-RAG router has no decoder start token")
        decoder = torch.tensor([[start_id]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            scores = model(**encoded, decoder_input_ids=decoder).logits[0, 0]
        label_ids = {label: tokenizer(label).input_ids[0] for label in _LABELS}
        if len(set(label_ids.values())) != 3:
            raise AlgorithmOutputError("Adaptive-RAG A/B/C tokenizer IDs must be distinct")
        logits = {
            label: float(scores[token_id].detach().cpu()) for label, token_id in label_ids.items()
        }
        selected = max(_LABELS, key=lambda label: logits[label])
        input_tokens = int(encoded["attention_mask"].sum().detach().cpu())
        return AdaptiveRAGRoute(label=selected, logits=logits, input_tokens=input_tokens)

    def fingerprint(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "router_kind": "t5_seq2seq_first_step",
                "artifact_root": str(self.artifact_root),
                "manifest_sha256": self.manifest_sha256,
                "artifact_closure_sha256": self._manifest.get("artifact_closure_sha256"),
                "training_data_sha256": self._manifest.get("training_data_sha256"),
                "base_model_tree_sha256": self._manifest.get("base_model_tree_sha256"),
                "architecture": self._manifest.get("architecture"),
                "architecture_substitution": self._manifest.get("architecture_substitution"),
                "upstream_commit": ADAPTIVE_RAG_UPSTREAM_COMMIT,
                "route_rule": "first_decoder_step_logit_argmax_A_first",
            }
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRAGCommonSupportConfig:
    ircot: IRCoTCommonSupportConfig
    single_retrieval_top_k: int = 6

    def __post_init__(self) -> None:
        if not isinstance(self.ircot, IRCoTCommonSupportConfig):
            raise ValidationError("Adaptive-RAG common-support requires an IRCoT config")
        if (
            isinstance(self.single_retrieval_top_k, bool)
            or not isinstance(self.single_retrieval_top_k, int)
            or self.single_retrieval_top_k < 1
        ):
            raise ValidationError("Adaptive-RAG single_retrieval_top_k must be positive")


@dataclass(slots=True)
class AdaptiveRAGCommonSupport:
    """Run one trained complexity route over common locked dependencies."""

    router: AdaptiveRAGRouter
    config: AdaptiveRAGCommonSupportConfig
    method_id: str = "adaptive_rag"
    release_id: str = "acquire.adaptive_rag.common_support"
    family: SemanticFamily = SemanticFamily.ACQUIRE

    @staticmethod
    def _retriever(value: object) -> Retriever:
        if value is None:
            raise ExternalDependencyError("Adaptive-RAG common-support requires a Retriever")
        if not hasattr(value, "retrieve") or not hasattr(value, "fingerprint"):
            raise CapabilityError("Adaptive-RAG retriever does not implement Retriever")
        return value  # type: ignore[return-value]

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        if not state.query.strip():
            raise ValidationError("Adaptive-RAG requires a non-empty question")
        if not isinstance(self.router, AdaptiveRAGRouter):
            raise CapabilityError("Adaptive-RAG router does not implement the router protocol")
        if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
            raise BudgetExceededError("Adaptive-RAG has no budget for its classifier")
        route = self.router.route(state.query)
        if (
            state.budget.max_context_tokens is not None
            and route.input_tokens > state.budget.max_context_tokens
        ):
            raise BudgetExceededError("Adaptive-RAG classifier exceeded max_context_tokens")
        strategy = _STRATEGIES[route.label]
        retriever = self._retriever(context.retriever)
        active: tuple[RetrievalResult, ...] = ()
        ircot_result = None
        retrieval_calls = 0
        generator_model_calls = 0
        generator_input_tokens = 0
        generator_output_tokens = 0
        events: list[TraceEvent] = [
            TraceEvent(
                TracePhase.PREPARE,
                "adaptive_rag_common_support_configuration",
                {
                    "upstream_commit": ADAPTIVE_RAG_UPSTREAM_COMMIT,
                    "strategy_map": dict(_STRATEGIES),
                    "route_rule": "first_decoder_step_logit_argmax_A_first",
                    "router": dict(self.router.fingerprint()),
                    "retriever": dict(retriever.fingerprint()),
                    "router_parity_claimed": False,
                    "upstream_arm_parity_claimed": False,
                },
            ),
            TraceEvent(
                TracePhase.PROPOSE,
                "adaptive_rag_route",
                {
                    "label": route.label,
                    "strategy": strategy,
                    "logits": dict(route.logits),
                    "input_tokens": route.input_tokens,
                    "output_tokens": route.output_tokens,
                },
            ),
        ]

        if strategy == "single_retrieval":
            if state.budget.max_retrieval_calls < 1:
                raise BudgetExceededError("Adaptive-RAG single retrieval exceeds its budget")
            active = _validated_results(
                retriever.retrieve(state.query, top_k=self.config.single_retrieval_top_k),
                top_k=self.config.single_retrieval_top_k,
            )
            if not active:
                raise AlgorithmOutputError("Adaptive-RAG single retrieval returned no evidence")
            retrieval_calls = 1
            events.append(
                TraceEvent(
                    TracePhase.OBSERVE,
                    "adaptive_rag_single_retrieval",
                    {
                        "query": state.query,
                        "document_ids": tuple(item.document.doc_id for item in active),
                    },
                )
            )
        elif strategy == "ircot":
            if state.budget.max_model_calls < 2:
                raise BudgetExceededError("Adaptive-RAG IRCoT route has no generator-call budget")
            if state.budget.max_output_tokens < 2:
                raise BudgetExceededError("Adaptive-RAG IRCoT route has no generator-token budget")
            remaining_context = (
                None
                if state.budget.max_context_tokens is None
                else state.budget.max_context_tokens - route.input_tokens
            )
            if remaining_context is not None and remaining_context < 0:
                raise BudgetExceededError("Adaptive-RAG classifier exhausted context budget")
            branch_state = replace(
                state,
                budget=ExecutionBudget(
                    max_model_calls=state.budget.max_model_calls - 1,
                    max_retrieval_calls=state.budget.max_retrieval_calls,
                    max_tool_calls=state.budget.max_tool_calls,
                    max_output_tokens=state.budget.max_output_tokens - route.output_tokens,
                    max_context_tokens=remaining_context,
                    max_cost_usd=state.budget.max_cost_usd,
                ),
            )
            ircot_result = IRCoTCommonSupport(
                config=self.config.ircot,
                method_id=self.method_id,
                release_id=self.release_id,
            ).run(branch_state, context)
            active = ircot_result.active_results
            retrieval_calls = ircot_result.retrieval_calls
            generator_model_calls = ircot_result.model_calls
            generator_input_tokens = ircot_result.input_tokens
            generator_output_tokens = ircot_result.output_tokens
            events.extend(
                TraceEvent(
                    TracePhase.OBSERVE,
                    "adaptive_rag_ircot_step",
                    {
                        "index": step.index,
                        "generated_sentence": step.generated_sentence,
                        "retrieval_query": step.retrieval_query,
                        "retrieved_document_ids": step.retrieved_document_ids,
                        "new_document_ids": step.new_document_ids,
                        "active_document_ids": step.active_document_ids,
                    },
                )
                for step in ircot_result.steps
            )

        model_calls = 1 + generator_model_calls
        input_tokens = route.input_tokens + generator_input_tokens
        output_tokens = route.output_tokens + generator_output_tokens
        if model_calls > state.budget.max_model_calls:
            raise BudgetExceededError("Adaptive-RAG exceeded max_model_calls")
        if retrieval_calls > state.budget.max_retrieval_calls:
            raise BudgetExceededError("Adaptive-RAG exceeded max_retrieval_calls")
        if output_tokens > state.budget.max_output_tokens:
            raise BudgetExceededError("Adaptive-RAG exceeded max_output_tokens")
        if (
            state.budget.max_context_tokens is not None
            and input_tokens > state.budget.max_context_tokens
        ):
            raise BudgetExceededError("Adaptive-RAG exceeded max_context_tokens")

        events.append(
            TraceEvent(
                TracePhase.FINALIZE,
                "adaptive_rag_selected_arm_exit",
                {
                    "label": route.label,
                    "strategy": strategy,
                    "document_ids": tuple(item.document.doc_id for item in active),
                    "ircot_exit_reason": ircot_result.exit_reason if ircot_result else None,
                    "generated_answer_visible_to_downstream": False,
                },
            )
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=tuple(events),
            counters={
                "model_calls": model_calls,
                "router_calls": 1,
                "generator_calls": generator_model_calls,
                "retrieval_calls": retrieval_calls,
                "active_documents": len(active),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        if active:
            operation = PatchOperation(
                PatchOp.ADD_CONTEXT,
                tuple(_context_from_result(item) for item in active),
                target="retrieved_evidence",
                rationale=f"real corpus evidence selected by Adaptive-RAG route {route.label}",
            )
        else:
            operation = PatchOperation(
                PatchOp.NO_OP,
                rationale="Adaptive-RAG route A selected the no-retrieval strategy",
            )
        fingerprint = dict(self.router.fingerprint())
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(operation,),
            trace=trace,
            implementation_identity=(
                f"controlled_t5_family_retraining:starsuzi/Adaptive-RAG@"
                f"{ADAPTIVE_RAG_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                "selected_route": route.label,
                "selected_strategy": strategy,
                "router_manifest_sha256": fingerprint.get("manifest_sha256"),
                "router_architecture": fingerprint.get("architecture"),
                "router_architecture_substitution": fingerprint.get("architecture_substitution"),
                "controlled_upstream_router_parity": False,
                "upstream_arm_execution_parity": False,
                "generated_answer_visible_to_downstream": False,
                "generated_reasoning_is_evidence": False,
            },
        )


def _validated_results(
    values: Sequence[RetrievalResult], *, top_k: int
) -> tuple[RetrievalResult, ...]:
    try:
        results = tuple(values)
    except TypeError as exc:
        raise AlgorithmOutputError("Adaptive-RAG retriever returned a non-sequence") from exc
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise AlgorithmOutputError("Adaptive-RAG retriever returned an invalid result")
    if len(results) > top_k:
        raise AlgorithmOutputError("Adaptive-RAG retriever exceeded top_k")
    ids = tuple(item.document.doc_id for item in results)
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise AlgorithmOutputError("Adaptive-RAG retrieval IDs must be non-empty and unique")
    if tuple(item.rank for item in results) != tuple(range(1, len(results) + 1)):
        raise AlgorithmOutputError("Adaptive-RAG retrieval ranks must be contiguous and one-based")
    return results


def _context_from_result(result: RetrievalResult) -> ContextBlock:
    document = result.document
    if not document.corpus_revision:
        raise AlgorithmOutputError("Adaptive-RAG evidence requires a corpus revision")
    reserved = {
        "corpus_revision",
        "document_id",
        "evidence_kind",
        "rank",
        "retrieval_features",
        "title",
    }
    overlap = reserved & set(document.metadata)
    if overlap:
        raise AlgorithmOutputError(
            "Adaptive-RAG document metadata uses reserved provenance keys: "
            + ", ".join(sorted(overlap))
        )
    return ContextBlock(
        block_id=document.doc_id,
        text=document.text,
        source=document.corpus_revision,
        score=result.score,
        metadata={
            **dict(document.metadata),
            "document_id": document.doc_id,
            "title": document.title,
            "corpus_revision": document.corpus_revision,
            "rank": result.rank,
            "retrieval_features": dict(result.features),
            "evidence_kind": "real_corpus_document",
        },
    )


__all__ = [
    "AdaptiveRAGCommonSupport",
    "AdaptiveRAGCommonSupportConfig",
    "AdaptiveRAGRoute",
    "AdaptiveRAGRouter",
    "T5AdaptiveRAGRouter",
]
