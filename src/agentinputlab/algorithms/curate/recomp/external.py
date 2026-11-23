"""License-gated external bridges for the two released RECOMP compressors.

The pinned repository exposes an extractive sentence scorer and a Hugging Face
seq2seq prediction script, rather than one online RECOMP runtime.  These
adapters therefore keep model execution in a user-mounted worker and make the
small amount of AgentInputLab assembly explicit and auditable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ContextBlock,
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
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers import WorkerClient
from agentinputlab.integrations.external_workers.method import (
    MethodWorkerRequest,
    call_worker,
    dependency_fingerprint,
    finite_number,
    mapping_sequence,
    nonnegative_integer,
    optional_text,
    required_text,
    resolve_worker,
    string_tuple,
    worker_fingerprint,
)
from agentinputlab.observability import external_usage_counters

RECOMP_UPSTREAM_COMMIT = "51d4432151efb3275257a9407dc71d1e5ec6634d"
RECOMP_EXTRACTIVE_PROTOCOL = "agentinputlab.recomp.extractive.v3"
RECOMP_ABSTRACTIVE_PROTOCOL = "agentinputlab.recomp.abstractive.v3"
RECOMP_CANDIDATE_UNIT_KIND = "presegmented_sentence"
RECOMP_COMMON_SUPPORT_RELEASE = "curate.recomp.common_support_extractive"
RECOMP_COMMON_SUPPORT_SPLITTER = "agentinputlab.visible_context_sentence_split.v1"
RECOMP_ABSTRACTIVE_INPUT_TEMPLATE = "Question: {question}\n Document: {passages}\n Summary: "
RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256 = hashlib.sha256(
    RECOMP_ABSTRACTIVE_INPUT_TEMPLATE.encode("utf-8")
).hexdigest()
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RECOMPExtractiveControls:
    """AgentInputLab's explicit assembly policy for upstream sentence scores."""

    max_sentences: int = 5
    minimum_score: float | None = None
    output_order: str = "source_order"
    assembly_profile: str = "controlled"
    dataset: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_sentences, bool)
            or not isinstance(self.max_sentences, int)
            or (self.max_sentences != -1 and self.max_sentences < 1)
        ):
            raise ValidationError("RECOMP max_sentences must be -1 or positive")
        if self.minimum_score is not None:
            finite_number(self.minimum_score, label="RECOMP minimum_score")
        if self.output_order not in {"source_order", "score_descending"}:
            raise ValidationError("RECOMP output_order must be source_order or score_descending")
        if self.assembly_profile not in {"controlled", "native_paper_qa"}:
            raise ValidationError("RECOMP assembly_profile is unsupported")
        if self.assembly_profile == "controlled":
            if self.dataset:
                raise ValidationError("controlled RECOMP assembly must not claim a paper dataset")
            return
        expected = {"nq": 1, "tqa": 1, "hotpotqa": 2}.get(self.dataset)
        if expected is None:
            raise ValidationError("paper-native RECOMP dataset must be nq, tqa, or hotpotqa")
        if (
            self.max_sentences != expected
            or self.minimum_score is not None
            or self.output_order != "score_descending"
        ):
            raise ValidationError(
                "paper-native RECOMP requires top-1 NQ/TQA or top-2 HotpotQA by score"
            )

    @classmethod
    def paper_native(cls, dataset: str) -> RECOMPExtractiveControls:
        normalized = dataset.lower()
        expected = {"nq": 1, "tqa": 1, "hotpotqa": 2}.get(normalized)
        if expected is None:
            raise ValidationError("paper-native RECOMP dataset must be nq, tqa, or hotpotqa")
        return cls(
            max_sentences=expected,
            output_order="score_descending",
            assembly_profile="native_paper_qa",
            dataset=normalized,
        )


@dataclass(frozen=True, slots=True)
class RECOMPAbstractiveControls:
    max_source_tokens: int = 1024
    max_target_tokens: int = 512
    num_beams: int = 1
    empty_output_policy: str = "clear_context"
    input_assembly: str = "controlled_concatenate_no_separator"

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.max_source_tokens, self.max_target_tokens, self.num_beams)
        ):
            raise ValidationError("RECOMP generation limits must be positive")
        if self.empty_output_policy != "clear_context":
            raise ValidationError(
                "RECOMP empty output is selective augmentation and must clear context"
            )
        if self.input_assembly not in {
            "controlled_concatenate_no_separator",
            "prejoined_official_artifact",
        }:
            raise ValidationError("RECOMP abstractive input assembly is unsupported")


@dataclass(frozen=True, slots=True)
class RECOMPSentenceScore:
    sentence_id: str
    document_id: str
    document_index: int
    sentence_index: int
    char_start: int
    char_end: int
    text: str
    score: float

    @property
    def source_order(self) -> tuple[int, int, int]:
        return (self.document_index, self.char_start, self.sentence_index)


@dataclass(frozen=True, slots=True)
class _SourceSentence:
    block: ContextBlock
    source_block_id: str
    source_block_index: int
    char_start: int
    char_end: int


def _common_support_sentences(
    context: tuple[ContextBlock, ...],
    *,
    max_candidates: int | None = None,
) -> tuple[_SourceSentence, ...]:
    """Split visible blocks into immutable RECOMP candidate units.

    The upstream inference function consumes already segmented sentences.  Its
    repository does not publish the dataset-specific splitter, so the common-
    support profile freezes a minimal punctuation/newline splitter and records
    exact offsets back to the captured arena block.  No hidden benchmark state
    or answer label participates in this preprocessing step.
    """

    candidates: list[_SourceSentence] = []
    boundary = re.compile(r"(?<=[.!?])\s+|\n+")
    for block_index, source in enumerate(context):
        source_id = source.block_id or f"context-{block_index}"
        cursor = 0
        sentence_index = 0
        for match in (*boundary.finditer(source.text), None):
            stop = len(source.text) if match is None else match.start()
            raw = source.text[cursor:stop]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            start = cursor + leading
            end = cursor + trailing
            if end > start:
                sentence_id = f"{source_id}:sentence:{sentence_index}"
                candidates.append(
                    _SourceSentence(
                        block=ContextBlock(
                            block_id=sentence_id,
                            text=source.text[start:end],
                            source="recomp_common_support_preprocessor",
                            metadata={
                                "source_context_block_id": source_id,
                                "source_context_block_index": block_index,
                                "source_char_start": start,
                                "source_char_end": end,
                                "splitter_revision": RECOMP_COMMON_SUPPORT_SPLITTER,
                            },
                        ),
                        source_block_id=source_id,
                        source_block_index=block_index,
                        char_start=start,
                        char_end=end,
                    )
                )
                if max_candidates is not None and len(candidates) > max_candidates:
                    raise ValidationError("RECOMP sentence candidate limit exceeded; no truncation applied")
                sentence_index += 1
            if match is None:
                break
            cursor = match.end()
    if not candidates:
        raise ValidationError("RECOMP common-support requires non-empty visible context")
    return tuple(candidates)


def prepare_recomp_sentence_state(
    state: FrozenAgentState, *, max_candidates: int = 4096,
) -> FrozenAgentState:
    """Prepare bounded, source-offset-bound inputs for the extractive interface.

    This is the same disclosed punctuation/newline splitter as the controlled
    common-support adapter, not a claimed upstream dataset-specific splitter.
    No model, retrieval, reference labels or network are used. Apply it to the
    common input of both NO_OP and compressor arms, not just to the treatment.
    """

    if not isinstance(state, FrozenAgentState):
        raise ValidationError("RECOMP sentence preparation requires a FrozenAgentState")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValidationError("RECOMP max_candidates must be a positive integer")
    if "recomp_sentence_preparation" in state.metadata:
        raise ValidationError("state already has RECOMP sentence preparation; use it directly")
    sentences = _common_support_sentences(state.context, max_candidates=max_candidates)
    ids = [item.block.block_id for item in sentences]
    if len(ids) != len(set(ids)):
        raise ValidationError("RECOMP sentence preparation has ambiguous source block IDs")
    blocks = tuple(replace(item.block, metadata={
        **dict(item.block.metadata), "source_captured_state_hash": state.state_hash,
    }) for item in sentences)
    return replace(state, context=blocks, metadata={
        **dict(state.metadata),
        "recomp_sentence_preparation": {
            "splitter_revision": RECOMP_COMMON_SUPPORT_SPLITTER,
            "source_state_hash": state.state_hash,
            "source_context_blocks": len(state.context),
            "candidate_sentences": len(blocks),
            "max_candidates": max_candidates,
            "reference_or_network_access": False,
            "paper_native_preprocessing_claimed": False,
        },
    })


@dataclass(slots=True)
class RECOMPExtractiveCommonSupport:
    """Source-faithful scorer plus a frozen visible-context sentence adapter."""

    checkpoint_revision: str = ""
    tokenizer_revision: str = ""
    asset_manifest_hash: str = ""
    asset_license_acknowledged: bool = False
    controls: RECOMPExtractiveControls = RECOMPExtractiveControls()
    method_id: str = "recomp"
    release_id: str = RECOMP_COMMON_SUPPORT_RELEASE
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        _validate_asset_identity(
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            asset_manifest_hash=self.asset_manifest_hash,
        )

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        sentences = _common_support_sentences(state.context)
        derived = replace(state, context=tuple(item.block for item in sentences))
        component = RECOMPExtractiveExternal(
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            sentence_splitter_revision=RECOMP_COMMON_SUPPORT_SPLITTER,
            asset_manifest_hash=self.asset_manifest_hash,
            asset_license_acknowledged=self.asset_license_acknowledged,
            controls=self.controls,
        ).propose(derived, context)
        source_by_sentence = {item.block.block_id: item for item in sentences}
        operations: list[PatchOperation] = []
        for operation in component.operations:
            if operation.op is not PatchOp.REPLACE_CONTEXT:
                raise AlgorithmOutputError(
                    "RECOMP common-support component emitted an unexpected patch operation"
                )
            selected: list[ContextBlock] = []
            for block in operation.value:
                sentence_id = str(block.metadata.get("source_document_id", ""))
                source = source_by_sentence.get(sentence_id)
                if source is None or block.text != source.block.text:
                    raise AlgorithmOutputError(
                        "RECOMP common-support output lost its source sentence binding"
                    )
                selected.append(
                    ContextBlock(
                        block_id=block.block_id,
                        text=block.text,
                        source=block.source,
                        score=block.score,
                        metadata={
                            **dict(block.metadata),
                            "source_context_block_id": source.source_block_id,
                            "source_context_block_index": source.source_block_index,
                            "source_char_start": source.char_start,
                            "source_char_end": source.char_end,
                            "source_captured_state_hash": state.state_hash,
                            "splitter_revision": RECOMP_COMMON_SUPPORT_SPLITTER,
                        },
                    )
                )
            operations.append(
                PatchOperation(
                    PatchOp.REPLACE_CONTEXT,
                    tuple(selected),
                    target=operation.target,
                    rationale=(
                        "official RECOMP sentence scores after frozen visible-context "
                        "presegmentation"
                    ),
                )
            )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "recomp_common_support_presegmentation",
                    {
                        "splitter_revision": RECOMP_COMMON_SUPPORT_SPLITTER,
                        "source_state_hash": state.state_hash,
                        "derived_state_hash": derived.state_hash,
                        "source_block_count": len(state.context),
                        "candidate_sentence_count": len(sentences),
                        "candidate_manifest_hash": content_hash(
                            tuple(
                                {
                                    "sentence_id": item.block.block_id,
                                    "text": item.block.text,
                                    "source_block_id": item.source_block_id,
                                    "source_block_index": item.source_block_index,
                                    "char_start": item.char_start,
                                    "char_end": item.char_end,
                                }
                                for item in sentences
                            )
                        ),
                    },
                ),
                *component.trace.events,
            ),
            counters={
                **dict(component.trace.counters),
                "source_context_blocks": len(state.context),
                "candidate_sentences": len(sentences),
            },
            warnings=(
                *component.trace.warnings,
                "controlled_common_support_sentence_splitter_not_paper_result_parity",
            ),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=tuple(operations),
            trace=trace,
            implementation_identity=(
                "common_support_preprocessor+"
                f"thin_external_wrapper:carriex/recomp@{RECOMP_UPSTREAM_COMMIT}"
            ),
            profile_kind="controlled_adapter",
            metadata={
                **dict(component.metadata),
                "component_release_id": component.release_id,
                "common_support_splitter_revision": RECOMP_COMMON_SUPPORT_SPLITTER,
                "source_state_hash": state.state_hash,
                "derived_candidate_state_hash": derived.state_hash,
                "paper_result_parity_claimed": False,
            },
        )


@dataclass(slots=True)
class RECOMPExtractiveExternal:
    """Run the official scorer externally, then apply a frozen assembly rule."""

    client: WorkerClient | None = None
    checkpoint_revision: str = ""
    tokenizer_revision: str = ""
    sentence_splitter_revision: str = ""
    asset_manifest_hash: str = ""
    asset_license_acknowledged: bool = False
    controls: RECOMPExtractiveControls = RECOMPExtractiveControls()
    method_id: str = "recomp"
    release_id: str = "curate.recomp.extractive"
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        _validate_asset_identity(
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            asset_manifest_hash=self.asset_manifest_hash,
        )
        if not self.sentence_splitter_revision.strip():
            raise ValidationError("RECOMP sentence_splitter_revision must not be empty")

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        document_ids = _document_ids(state)
        _require_asset_acknowledgement(self.asset_license_acknowledged)
        worker = resolve_worker(self.client, context, label="RECOMP extractive")
        worker_identity = worker_fingerprint(worker, label="RECOMP extractive")
        retriever_fp = dependency_fingerprint(
            context.retriever, label="RECOMP upstream retriever", required=True
        )
        retriever_hash = content_hash(retriever_fp)
        input_hash = content_hash(
            {
                "query": state.query,
                "documents": state.context,
                "document_ids": document_ids,
                "candidate_unit_kind": RECOMP_CANDIDATE_UNIT_KIND,
            }
        )
        remaining_budget = _remaining_execution_budget(state, context)
        _require_local_compute_budget(
            remaining_budget,
            output_tokens_required=0,
        )
        base_payload = {
            "task_id": state.task_id,
            "query": state.query,
            "documents": state.context,
            "document_ids": document_ids,
            "checkpoint_revision": self.checkpoint_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "sentence_splitter_revision": self.sentence_splitter_revision,
            "asset_manifest_hash": self.asset_manifest_hash,
            "retriever_artifact_hash": retriever_hash,
            "input_artifact_hash": input_hash,
            "candidate_unit_kind": RECOMP_CANDIDATE_UNIT_KIND,
            "controls": asdict(self.controls),
            "seed": context.seed,
        }
        preflight_payload, response = _preflight_then_execute(
            worker=worker,
            worker_identity=worker_identity,
            method_id=self.method_id,
            release_id=self.release_id,
            protocol=RECOMP_EXTRACTIVE_PROTOCOL,
            state_hash=state.state_hash,
            base_payload=base_payload,
            dependencies={"retriever": retriever_fp},
            remaining_budget=remaining_budget,
        )
        _validate_response_identity(
            response.payload,
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            asset_manifest_hash=self.asset_manifest_hash,
            document_ids=document_ids,
            retriever_artifact_hash=retriever_hash,
            input_artifact_hash=input_hash,
        )
        _validate_worker_attestation(response.payload, worker_identity)
        if response.payload.get("candidate_unit_kind") != RECOMP_CANDIDATE_UNIT_KIND:
            raise AlgorithmOutputError("RECOMP worker changed fixed candidate units")
        if response.payload.get("scoring_api") != (
            "run_extractive_compressor.get_contriever_scores(top_k=-1)"
        ):
            raise AlgorithmOutputError("RECOMP worker did not use the pinned scoring API")
        if response.usage.model_calls != 1 or response.usage.retrieval_calls != 0:
            raise AlgorithmOutputError("RECOMP scorer usage is inconsistent")
        returned_splitter = required_text(
            response.payload,
            "sentence_splitter_revision",
            label="RECOMP extractive response",
        )
        if returned_splitter != self.sentence_splitter_revision:
            raise AlgorithmOutputError("RECOMP sentence splitter revision mismatch")
        scores = _parse_sentence_scores(response.payload.get("sentence_scores"), state)
        selected = _select_sentences(scores, self.controls)
        blocks = tuple(
            ContextBlock(
                block_id=f"recomp:{record.sentence_id}",
                text=record.text,
                source="recomp_extractive",
                score=record.score,
                metadata={
                    "source_document_id": record.document_id,
                    "source_document_index": record.document_index,
                    "source_sentence_index": record.sentence_index,
                    "source_char_start": record.char_start,
                    "source_char_end": record.char_end,
                    "source_state_hash": state.state_hash,
                },
            )
            for record in selected
        )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "recomp_extractive_assets",
                    {
                        "checkpoint_revision": self.checkpoint_revision,
                        "tokenizer_revision": self.tokenizer_revision,
                        "sentence_splitter_revision": self.sentence_splitter_revision,
                        "document_ids": document_ids,
                    },
                ),
                TraceEvent(
                    TracePhase.OBSERVE,
                    "official_sentence_scores",
                    {
                        "sentence_ids": tuple(item.sentence_id for item in scores),
                        "scores": tuple(item.score for item in scores),
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "agentinputlab_assembly",
                    {
                        "controls": asdict(self.controls),
                        "selected_sentence_ids": tuple(item.sentence_id for item in selected),
                    },
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 2,
                "scored_sentences": len(scores),
                "selected_sentences": len(selected),
            },
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_CONTEXT,
                    blocks,
                    target="compressed_retrieved_context",
                    rationale="official RECOMP scores with a frozen AgentInputLab assembly policy",
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"thin_external_wrapper:carriex/recomp@{RECOMP_UPSTREAM_COMMIT}"
            ),
            profile_kind=(
                "paper_native_adapter"
                if self.controls.assembly_profile == "native_paper_qa"
                else "controlled_adapter"
            ),
            metadata={
                "implementation_origin": "official_scorer_plus_controlled_assembly",
                "known_deviation": "upstream repository does not provide online assembly",
                "asset_license": "NOASSERTION_acknowledged_by_user",
                "asset_manifest_hash": self.asset_manifest_hash,
                "assembly_policy_hash": content_hash(asdict(self.controls)),
                "assembly_profile": self.controls.assembly_profile,
                "retriever_artifact_hash": retriever_hash,
                "input_artifact_hash": input_hash,
                "compatibility_manifest_sha256": response.payload["compatibility_manifest_sha256"],
                "execution_evidence": response.payload["execution_evidence"],
                "worker_fingerprint": worker_identity,
                "runtime_fingerprint": response.payload["runtime_fingerprint"],
                "runtime_fingerprint_hash": response.payload["runtime_fingerprint_hash"],
                "checkpoint_tree_sha256": response.payload["checkpoint_tree_sha256"],
                "preflight_token": preflight_payload["preflight_token"],
            },
        )


@dataclass(slots=True)
class RECOMPAbstractiveExternal:
    """Bridge the official seq2seq prediction path, preserving empty summaries."""

    client: WorkerClient | None = None
    checkpoint_revision: str = ""
    tokenizer_revision: str = ""
    input_template_hash: str = ""
    asset_manifest_hash: str = ""
    asset_license_acknowledged: bool = False
    controls: RECOMPAbstractiveControls = RECOMPAbstractiveControls()
    method_id: str = "recomp"
    release_id: str = "curate.recomp.abstractive"
    family: SemanticFamily = SemanticFamily.CURATE

    def __post_init__(self) -> None:
        _validate_asset_identity(
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            asset_manifest_hash=self.asset_manifest_hash,
        )
        if self.input_template_hash != RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256:
            raise ValidationError(
                "RECOMP input_template_hash must match the pinned official prompt"
            )

    def propose(self, state: FrozenAgentState, context: InterventionContext) -> InputPatch:
        document_ids = _document_ids(state)
        _require_asset_acknowledgement(self.asset_license_acknowledged)
        worker = resolve_worker(self.client, context, label="RECOMP abstractive")
        worker_identity = worker_fingerprint(worker, label="RECOMP abstractive")
        retriever_fp = dependency_fingerprint(
            context.retriever, label="RECOMP upstream retriever", required=True
        )
        retriever_hash = content_hash(retriever_fp)
        if self.controls.input_assembly == "prejoined_official_artifact":
            if len(state.context) != 1:
                raise ValidationError("prejoined RECOMP input requires exactly one context block")
            passages = state.context[0].text
        else:
            passages = "".join(block.text for block in state.context)
        passages_hash = content_hash(passages)
        input_hash = content_hash(
            {
                "query": state.query,
                "documents": state.context,
                "document_ids": document_ids,
                "passages": passages,
                "input_template_hash": self.input_template_hash,
            }
        )
        remaining_budget = _remaining_execution_budget(state, context)
        _require_local_compute_budget(
            remaining_budget,
            output_tokens_required=self.controls.max_target_tokens,
        )
        base_payload = {
            "task_id": state.task_id,
            "query": state.query,
            "documents": state.context,
            "document_ids": document_ids,
            "passages": passages,
            "passages_hash": passages_hash,
            "checkpoint_revision": self.checkpoint_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "input_template_hash": self.input_template_hash,
            "asset_manifest_hash": self.asset_manifest_hash,
            "retriever_artifact_hash": retriever_hash,
            "input_artifact_hash": input_hash,
            "controls": asdict(self.controls),
            "seed": context.seed,
        }
        preflight_payload, response = _preflight_then_execute(
            worker=worker,
            worker_identity=worker_identity,
            method_id=self.method_id,
            release_id=self.release_id,
            protocol=RECOMP_ABSTRACTIVE_PROTOCOL,
            state_hash=state.state_hash,
            base_payload=base_payload,
            dependencies={"retriever": retriever_fp},
            remaining_budget=remaining_budget,
        )
        payload = response.payload
        _validate_response_identity(
            payload,
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            asset_manifest_hash=self.asset_manifest_hash,
            document_ids=document_ids,
            retriever_artifact_hash=retriever_hash,
            input_artifact_hash=input_hash,
        )
        _validate_worker_attestation(payload, worker_identity)
        if (
            required_text(payload, "input_template_hash", label="RECOMP abstractive response")
            != self.input_template_hash
        ):
            raise AlgorithmOutputError("RECOMP input template hash mismatch")
        if payload.get("passages_hash") != passages_hash:
            raise AlgorithmOutputError("RECOMP passages hash mismatch")
        expected_prompt = RECOMP_ABSTRACTIVE_INPUT_TEMPLATE.format(
            question=state.query, passages=passages
        )
        if payload.get("prompt_hash") != hashlib.sha256(expected_prompt.encode()).hexdigest():
            raise AlgorithmOutputError("RECOMP official prompt hash mismatch")
        if payload.get("generation_api") != ("AutoModelForSeq2SeqLM.generate+batch_decode(strip)"):
            raise AlgorithmOutputError("RECOMP worker did not use the pinned generation path")
        if payload.get("applied") is not True:
            raise AlgorithmOutputError(
                "successful RECOMP generation must report applied=true, including empty output"
            )
        summary = optional_text(payload, "summary", label="RECOMP abstractive response")
        raw_generation = optional_text(
            payload, "raw_generation", label="RECOMP abstractive response"
        )
        if summary != summary.strip():
            raise AlgorithmOutputError("RECOMP summary must be normalized without edge whitespace")
        selective_empty = not summary
        if payload.get("selective_augmentation") is not selective_empty:
            raise AlgorithmOutputError("RECOMP selective-augmentation flag disagrees with output")
        parse_status = required_text(payload, "parse_status", label="RECOMP abstractive response")
        expected_status = "generated_empty" if selective_empty else "generated_text"
        if parse_status != expected_status:
            raise AlgorithmOutputError("RECOMP generation parse status is inconsistent")
        if response.usage.model_calls < 1:
            raise AlgorithmOutputError("RECOMP abstractive generation must report a model call")

        blocks: tuple[ContextBlock, ...]
        if selective_empty:
            blocks = ()
        else:
            blocks = (
                ContextBlock(
                    block_id=f"{self.release_id}:summary",
                    text=summary,
                    source="recomp_abstractive",
                    metadata={
                        "source_document_ids": document_ids,
                        "source_state_hash": state.state_hash,
                        "raw_generation_hash": content_hash(raw_generation),
                    },
                ),
            )
        trace = ExecutionTrace(
            method_id=self.method_id,
            release_id=self.release_id,
            events=(
                TraceEvent(
                    TracePhase.PREPARE,
                    "recomp_abstractive_assets",
                    {
                        "checkpoint_revision": self.checkpoint_revision,
                        "tokenizer_revision": self.tokenizer_revision,
                        "input_template_hash": self.input_template_hash,
                        "document_ids": document_ids,
                    },
                ),
                TraceEvent(
                    TracePhase.FINALIZE,
                    "selective_generation",
                    {
                        "parse_status": parse_status,
                        "selective_augmentation": selective_empty,
                        "raw_generation": raw_generation,
                        "selected_context_count": len(blocks),
                    },
                ),
            ),
            counters={
                **external_usage_counters(response.usage),
                "worker_calls": 2,
                "selected_context_blocks": len(blocks),
            },
            warnings=("selective_augmentation_cleared_context",) if selective_empty else (),
        )
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(
                    PatchOp.REPLACE_CONTEXT,
                    blocks,
                    target="compressed_retrieved_context",
                    rationale=(
                        "successful empty RECOMP output disables augmentation"
                        if selective_empty
                        else "official RECOMP abstractive compressor output"
                    ),
                ),
            ),
            trace=trace,
            implementation_identity=(
                f"thin_external_wrapper:carriex/recomp@{RECOMP_UPSTREAM_COMMIT}"
            ),
            profile_kind=(
                "official_prompt_adapter"
                if self.controls.input_assembly == "prejoined_official_artifact"
                else "controlled_adapter"
            ),
            metadata={
                "implementation_origin": "thin_external_wrapper",
                "asset_license": "NOASSERTION_acknowledged_by_user",
                "asset_manifest_hash": self.asset_manifest_hash,
                "selective_augmentation": selective_empty,
                "applied": True,
                "raw_generation": raw_generation,
                "input_assembly": self.controls.input_assembly,
                "retriever_artifact_hash": retriever_hash,
                "input_artifact_hash": input_hash,
                "compatibility_manifest_sha256": payload["compatibility_manifest_sha256"],
                "execution_evidence": payload["execution_evidence"],
                "worker_fingerprint": worker_identity,
                "runtime_fingerprint": payload["runtime_fingerprint"],
                "runtime_fingerprint_hash": payload["runtime_fingerprint_hash"],
                "checkpoint_tree_sha256": payload["checkpoint_tree_sha256"],
                "preflight_token": preflight_payload["preflight_token"],
            },
        )


def _remaining_execution_budget(
    state: FrozenAgentState, context: InterventionContext
) -> Mapping[str, int | float | None]:
    usage = None
    ledger = context.budget_ledger
    if ledger is not None:
        ledger_budget = getattr(ledger, "budget", None)
        usage = getattr(ledger, "usage", None)
        if ledger_budget != state.budget or usage is None:
            raise ValidationError("RECOMP received an incompatible budget ledger")

    def consumed_integer(name: str) -> int:
        value = 0 if usage is None else getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(f"RECOMP budget usage {name} is invalid")
        return value

    remaining_context = None
    if state.budget.max_context_tokens is not None:
        remaining_context = max(
            0,
            state.budget.max_context_tokens - consumed_integer("input_tokens"),
        )
    remaining_cost: float | None = None
    if state.budget.max_cost_usd is not None:
        consumed_cost = 0.0 if usage is None else getattr(usage, "total_cost_usd", None)
        if (
            isinstance(consumed_cost, bool)
            or not isinstance(consumed_cost, (int, float))
            or consumed_cost < 0
        ):
            raise ValidationError("RECOMP budget usage total_cost_usd is invalid")
        remaining_cost = max(0.0, float(state.budget.max_cost_usd) - float(consumed_cost))
    return {
        "model_calls": max(0, state.budget.max_model_calls - consumed_integer("model_calls")),
        "context_tokens": remaining_context,
        "output_tokens": max(0, state.budget.max_output_tokens - consumed_integer("output_tokens")),
        "cost_usd": remaining_cost,
    }


def _require_local_compute_budget(
    budget: Mapping[str, int | float | None], *, output_tokens_required: int
) -> None:
    if budget["model_calls"] < 1:  # type: ignore[operator]
        raise BudgetExceededError("RECOMP requires one remaining model call")
    if budget["output_tokens"] < output_tokens_required:  # type: ignore[operator]
        raise BudgetExceededError(
            "RECOMP remaining output-token budget is below the generation maximum"
        )
    if budget["cost_usd"] is not None:
        raise BudgetExceededError(
            "RECOMP local inference is unpriced; a configured cost cap cannot be guaranteed"
        )


def _preflight_then_execute(
    *,
    worker: WorkerClient,
    worker_identity: Mapping[str, Any],
    method_id: str,
    release_id: str,
    protocol: str,
    state_hash: str,
    base_payload: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    remaining_budget: Mapping[str, int | float | None],
) -> tuple[Mapping[str, Any], Any]:
    def request(*, phase: str, token: str) -> MethodWorkerRequest:
        return MethodWorkerRequest(
            method_id=method_id,
            release_id=release_id,
            upstream_commit=RECOMP_UPSTREAM_COMMIT,
            state_hash=state_hash,
            payload={
                **base_payload,
                "phase": phase,
                "preflight_token": token,
                "checkpoint_license_acknowledged": True,
                "remaining_budget": remaining_budget,
            },
            dependency_fingerprints=dependencies,
            minimum_reservation={} if phase == "preflight" else {"model_calls": 1},
            protocol_version=protocol,
        )

    preflight = call_worker(worker, request(phase="preflight", token=""))
    if (
        preflight.usage.model_calls
        or preflight.usage.retrieval_calls
        or preflight.usage.input_tokens
        or preflight.usage.output_tokens
        or preflight.usage.unpriced_model_calls
    ):
        raise AlgorithmOutputError("RECOMP preflight must report zero inference usage")
    payload = preflight.payload
    _validate_worker_attestation(payload, worker_identity)
    _validate_preflight_payload(payload, remaining_budget=remaining_budget)
    token = required_text(payload, "preflight_token", label="RECOMP preflight")
    execute = call_worker(worker, request(phase="execute", token=token))
    _validate_worker_attestation(execute.payload, worker_identity)
    _validate_preflight_payload(execute.payload, remaining_budget=remaining_budget)
    for key in (
        "preflight_token",
        "runtime_fingerprint_hash",
        "checkpoint_tree_sha256",
        "remaining_budget_hash",
        "input_tokens_required",
        "output_tokens_required",
    ):
        if execute.payload.get(key) != payload.get(key):
            raise AlgorithmOutputError(f"RECOMP {key} changed after preflight")
    return payload, execute


def _validate_preflight_payload(
    payload: Mapping[str, Any],
    *,
    remaining_budget: Mapping[str, int | float | None],
) -> None:
    if (
        payload.get("checkpoint_license") != "NOASSERTION"
        or payload.get("checkpoint_license_acknowledged") is not True
    ):
        raise AlgorithmOutputError("RECOMP checkpoint license acknowledgement was not attested")
    if payload.get("remaining_budget_hash") != content_hash(remaining_budget):
        raise AlgorithmOutputError("RECOMP worker budget binding mismatch")
    input_required = nonnegative_integer(
        payload.get("input_tokens_required"), label="RECOMP input_tokens_required"
    )
    output_required = nonnegative_integer(
        payload.get("output_tokens_required"), label="RECOMP output_tokens_required"
    )
    context_cap = remaining_budget["context_tokens"]
    if context_cap is not None and input_required > context_cap:
        raise AlgorithmOutputError("RECOMP preflight exceeded the sent context-token budget")
    if output_required > remaining_budget["output_tokens"]:  # type: ignore[operator]
        raise AlgorithmOutputError("RECOMP preflight exceeded the sent output-token budget")
    runtime = payload.get("runtime_fingerprint")
    if not isinstance(runtime, Mapping) or not runtime:
        raise AlgorithmOutputError("RECOMP response omits runtime fingerprint")
    runtime_hash = required_text(payload, "runtime_fingerprint_hash", label="RECOMP response")
    if runtime_hash != content_hash(runtime):
        raise AlgorithmOutputError("RECOMP runtime fingerprint hash mismatch")
    required_keys = {
        "device",
        "python_version",
        "python_implementation",
        "dependency_versions",
        "container_digest",
        "custom_model_loader",
    }
    if set(runtime) != required_keys or not isinstance(runtime.get("dependency_versions"), Mapping):
        raise AlgorithmOutputError("RECOMP runtime fingerprint schema mismatch")
    for key in ("checkpoint_tree_sha256", "preflight_token"):
        if not _HEX64.fullmatch(required_text(payload, key, label="RECOMP response")):
            raise AlgorithmOutputError(f"RECOMP {key} must be a SHA-256 digest")
    nonnegative_integer(payload.get("checkpoint_file_count"), label="RECOMP checkpoint_file_count")


def _validate_asset_identity(
    *, checkpoint_revision: str, tokenizer_revision: str, asset_manifest_hash: str
) -> None:
    for label, value in (
        ("checkpoint_revision", checkpoint_revision),
        ("tokenizer_revision", tokenizer_revision),
        ("asset_manifest_hash", asset_manifest_hash),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"RECOMP {label} must not be empty")
    if not _HEX64.fullmatch(asset_manifest_hash):
        raise ValidationError("RECOMP asset_manifest_hash must be a SHA-256 digest")


def _require_asset_acknowledgement(acknowledged: bool) -> None:
    if not acknowledged:
        raise ExternalDependencyError(
            "RECOMP model artifacts are NOASSERTION; explicitly acknowledge the "
            "user-mounted asset license before execution"
        )


def _validate_worker_attestation(
    payload: Mapping[str, Any], worker_identity: Mapping[str, Any]
) -> None:
    for key in ("compatibility_manifest_sha256", "execution_evidence"):
        expected = worker_identity.get(key)
        if not isinstance(expected, str) or not expected:
            raise ValidationError(f"RECOMP worker fingerprint omits {key}")
        if payload.get(key) != expected:
            raise AlgorithmOutputError(f"RECOMP worker {key} attestation mismatch")
    expected_runtime = worker_identity.get("runtime_fingerprint_hash")
    if expected_runtime is not None:
        if not isinstance(expected_runtime, str) or not _HEX64.fullmatch(expected_runtime):
            raise ValidationError("RECOMP worker fingerprint runtime hash is invalid")
        if payload.get("runtime_fingerprint_hash") != expected_runtime:
            raise AlgorithmOutputError("RECOMP worker runtime fingerprint attestation mismatch")


def _document_ids(state: FrozenAgentState) -> tuple[str, ...]:
    if not state.context:
        raise ValidationError("RECOMP requires already-retrieved context")
    values = tuple(block.block_id for block in state.context)
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValidationError("RECOMP documents require unique non-empty block IDs")
    return values


def _validate_response_identity(
    payload: Mapping[str, Any],
    *,
    checkpoint_revision: str,
    tokenizer_revision: str,
    asset_manifest_hash: str,
    document_ids: tuple[str, ...],
    retriever_artifact_hash: str,
    input_artifact_hash: str,
) -> None:
    for key, expected in (
        ("checkpoint_revision", checkpoint_revision),
        ("tokenizer_revision", tokenizer_revision),
        ("asset_manifest_hash", asset_manifest_hash),
    ):
        if required_text(payload, key, label="RECOMP response") != expected:
            raise AlgorithmOutputError(f"RECOMP {key} mismatch")
    returned_ids = string_tuple(
        payload.get("source_document_ids"),
        label="RECOMP source_document_ids",
        allow_empty=False,
        unique=True,
    )
    if returned_ids != document_ids:
        raise AlgorithmOutputError("RECOMP worker changed source document order")
    for key, expected in (
        ("retriever_artifact_hash", retriever_artifact_hash),
        ("input_artifact_hash", input_artifact_hash),
    ):
        if required_text(payload, key, label="RECOMP response") != expected:
            raise AlgorithmOutputError(f"RECOMP {key} mismatch")
    for key in ("compatibility_manifest_sha256",):
        if not _HEX64.fullmatch(required_text(payload, key, label="RECOMP response")):
            raise AlgorithmOutputError(f"RECOMP {key} must be a SHA-256 digest")
    if required_text(payload, "execution_evidence", label="RECOMP response") not in {
        "pinned_upstream_source",
        "fake_checkout_plumbing",
    }:
        raise AlgorithmOutputError("RECOMP execution evidence is invalid")


def _parse_sentence_scores(raw: Any, state: FrozenAgentState) -> tuple[RECOMPSentenceScore, ...]:
    rows = mapping_sequence(raw, label="RECOMP sentence_scores")
    if len(rows) != len(state.context):
        raise AlgorithmOutputError("RECOMP score count must match fixed candidate sentences")
    result: list[RECOMPSentenceScore] = []
    for row_index, (row, source) in enumerate(zip(rows, state.context, strict=True)):
        label = f"RECOMP sentence score {row_index}"
        sentence_id = required_text(row, "sentence_id", label=label)
        document_id = required_text(row, "document_id", label=label)
        index = nonnegative_integer(row.get("document_index"), label=f"{label} document_index")
        if sentence_id != source.block_id or document_id != source.block_id or index != row_index:
            raise AlgorithmOutputError("RECOMP fixed candidate ID/order mismatch")
        sentence_index = nonnegative_integer(
            row.get("sentence_index"), label=f"{label} sentence_index"
        )
        char_start = nonnegative_integer(row.get("char_start"), label=f"{label} char_start")
        char_end = nonnegative_integer(row.get("char_end"), label=f"{label} char_end")
        text = required_text(row, "text", label=label)
        if sentence_index != 0 or char_start != 0 or char_end != len(source.text):
            raise AlgorithmOutputError("RECOMP fixed candidate offsets changed")
        if text != source.text:
            raise AlgorithmOutputError("RECOMP fixed candidate text changed")
        result.append(
            RECOMPSentenceScore(
                sentence_id=sentence_id,
                document_id=document_id,
                document_index=index,
                sentence_index=sentence_index,
                char_start=char_start,
                char_end=char_end,
                text=text,
                score=finite_number(row.get("score"), label=f"{label} score"),
            )
        )
    return tuple(result)


def _select_sentences(
    scores: tuple[RECOMPSentenceScore, ...], controls: RECOMPExtractiveControls
) -> tuple[RECOMPSentenceScore, ...]:
    eligible = tuple(
        item
        for item in scores
        if controls.minimum_score is None or item.score >= controls.minimum_score
    )
    ranked = sorted(eligible, key=lambda item: (-item.score, item.source_order))
    if controls.max_sentences != -1:
        ranked = ranked[: controls.max_sentences]
    if controls.output_order == "source_order":
        ranked.sort(key=lambda item: item.source_order)
    return tuple(ranked)


__all__ = [
    "RECOMP_ABSTRACTIVE_INPUT_TEMPLATE",
    "RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256",
    "RECOMP_CANDIDATE_UNIT_KIND",
    "RECOMP_UPSTREAM_COMMIT",
    "RECOMPAbstractiveControls",
    "RECOMPAbstractiveExternal",
    "RECOMPExtractiveControls",
    "RECOMPExtractiveExternal",
    "RECOMPSentenceScore",
]
