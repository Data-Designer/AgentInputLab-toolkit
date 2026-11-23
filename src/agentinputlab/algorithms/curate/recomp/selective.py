"""Learned compression, selective augmentation and a shared-budget QA reader.

The compressor is the pinned RECOMP worker. The reader/assembly is an explicit
AgentInputLab controlled runtime, not a reconstruction of unpublished prompts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from agentinputlab.backends import GenerationRequest
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
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.observability import BudgetLedger, external_usage_counters

from .external import (
    RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
    RECOMPAbstractiveControls,
    RECOMPAbstractiveExternal,
    _remaining_execution_budget,
)

RECOMP_SELECTIVE_RELEASE = "curate.recomp.selective"
RECOMP_SELECTIVE_ASSEMBLY = "agentinputlab.recomp.learned_empty_gate_reader.v1"
RECOMP_SELECTIVE_IR_ASSEMBLY = "agentinputlab.recomp.learned_empty_gate_ir_query_reader.v1"
RECOMP_QUERY_DECODER_ID = "agentinputlab.recomp.complete_query_envelope.v1"
RECOMP_QUERY_DECODER_SHA256 = content_hash(
    {
        "id": RECOMP_QUERY_DECODER_ID,
        "schema": {"query": "one nonempty string, preserved verbatim"},
        "envelope": "complete JSON or complete lowercase json fence only",
        "duplicate_keys": "reject",
        "additional_fields": "reject",
        "prose_or_reasoning_extraction": "reject",
        "model_calls": 0,
    }
)
RECOMP_SELECTIVE_READER_TEMPLATE = (
    "Answer the question. Use the compressed evidence when it is provided. "
    "Return only the answer, without an explanation.\n\n"
    "{instruction}{evidence}Question: {question}\nAnswer:"
)
RECOMP_SELECTIVE_IR_READER_TEMPLATE = (
    "Generate one useful search query for the indexed corpus. "
    'Return exactly a JSON object with one nonempty string field "query". '
    "Do not answer the question, invent document IDs, execute tools, "
    "or add text outside the JSON object.\n\n"
    "{instruction}{evidence}Original query: {question}\nSearch query JSON:"
)


def parse_recomp_query_reader_answer(
    text: str, *, envelope_policy: str = "strict_json"
) -> Mapping[str, str]:
    """Decode one complete query object; never salvage prose or invent a query."""
    if not isinstance(envelope_policy, str) or envelope_policy not in {
        "strict_json",
        "complete_json_or_fence",
    }:
        raise ValidationError("RECOMP query envelope policy is unsupported")
    if not isinstance(text, str):
        raise AlgorithmOutputError("RECOMP IR reader must return query JSON text")
    raw = text.strip()
    if (
        envelope_policy == "complete_json_or_fence"
        and raw.startswith("```json\n")
        and raw.endswith("\n```")
    ):
        raw = raw[8:-4]

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AlgorithmOutputError("RECOMP IR reader returned duplicate query keys")
            result[key] = value
        return result

    try:
        query = json.loads(raw, object_pairs_hook=unique_object)
    except (TypeError, ValueError) as exc:
        raise AlgorithmOutputError("RECOMP IR reader returned invalid query JSON") from exc
    if (
        not isinstance(query, dict)
        or set(query) != {"query"}
        or not isinstance(query["query"], str)
        or not query["query"].strip()
    ):
        raise AlgorithmOutputError("RECOMP IR reader requires one nonempty query string")
    return query


@dataclass(slots=True)
class RECOMPSelectiveRuntime:
    """Run the actual abstractive compressor and then one bounded QA reader.

    Supply retrieved context in the state, its retriever identity on the context,
    and the pinned compressor worker through ``context.external_runtime``. An
    empty successful summary disables augmentation; a worker error propagates.
    All helper and reader calls use the same observed budget ledger.
    """

    checkpoint_revision: str = ""
    tokenizer_revision: str = ""
    asset_manifest_hash: str = ""
    asset_license_acknowledged: bool = False
    compressor_controls: RECOMPAbstractiveControls = field(
        default_factory=RECOMPAbstractiveControls
    )
    reader_max_tokens: int = 512
    assembly_policy: str = RECOMP_SELECTIVE_ASSEMBLY
    method_id: str = "recomp"
    release_id: str = RECOMP_SELECTIVE_RELEASE
    family: SemanticFamily = SemanticFamily.CURATE
    output_format: str = "answer_text"
    query_envelope: str = "strict_json"

    def __post_init__(self) -> None:
        if isinstance(self.compressor_controls, Mapping):
            self.compressor_controls = RECOMPAbstractiveControls(**self.compressor_controls)
        if not isinstance(self.compressor_controls, RECOMPAbstractiveControls):
            raise ValidationError("RECOMP selective requires typed compressor controls")
        if (
            type(self.reader_max_tokens) is not int
            or not 1 <= self.reader_max_tokens <= 512
            or self.compressor_controls.max_target_tokens > 512
        ):
            raise ValidationError("RECOMP selective helper and reader caps must be within 1..512")
        if not isinstance(self.output_format, str) or self.output_format not in {
            "answer_text",
            "ir_query_json",
        }:
            raise ValidationError("RECOMP selective output_format is unsupported")
        if not isinstance(self.query_envelope, str) or self.query_envelope not in {
            "strict_json",
            "complete_json_or_fence",
        }:
            raise ValidationError("RECOMP query envelope policy is unsupported")
        if self.output_format == "answer_text" and self.query_envelope != "strict_json":
            raise ValidationError("RECOMP query envelope requires the explicit IR output mode")
        if (
            self.output_format == "ir_query_json"
            and self.assembly_policy == RECOMP_SELECTIVE_ASSEMBLY
        ):
            self.assembly_policy = RECOMP_SELECTIVE_IR_ASSEMBLY
        expected = (
            RECOMP_SELECTIVE_ASSEMBLY
            if self.output_format == "answer_text"
            else RECOMP_SELECTIVE_IR_ASSEMBLY
        )
        if self.assembly_policy != expected:
            raise ValidationError("RECOMP selective assembly policy must be the frozen v1 policy")
        self._compressor()  # Validate immutable checkpoint/tokenizer/prompt identities now.

    def _compressor(self) -> RECOMPAbstractiveExternal:
        return RECOMPAbstractiveExternal(
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            asset_manifest_hash=self.asset_manifest_hash,
            asset_license_acknowledged=self.asset_license_acknowledged,
            input_template_hash=RECOMP_ABSTRACTIVE_INPUT_TEMPLATE_SHA256,
            controls=self.compressor_controls,
        )

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        if not state.query.strip() or not state.context:
            raise ValidationError(
                "RECOMP selective needs a question and captured retrieved context"
            )
        if state.messages or state.tool_names:
            raise ValidationError("RECOMP selective QA does not silently discard chat/tool state")
        if context.backend is None:
            raise ExternalDependencyError("RECOMP selective requires a reader backend")
        if not isinstance(context.budget_ledger, BudgetLedger):
            raise ValidationError("use run_composite_intervention for shared RECOMP accounting")
        remaining = _remaining_execution_budget(state, context)
        reservation = self.compressor_controls.max_target_tokens + self.reader_max_tokens
        if remaining["model_calls"] < 2 or remaining["output_tokens"] < reservation:
            raise BudgetExceededError(
                "RECOMP selective must reserve compressor plus reader before either call"
            )

        component = self._compressor().propose(state, context)
        compressed = component.apply(state)
        empty = bool(component.metadata["selective_augmentation"])
        if empty != (not compressed.context):
            raise AlgorithmOutputError("RECOMP learned gate and compressed context disagree")
        summary = "\n\n".join(block.text for block in compressed.context)
        instruction = f"Instructions: {state.instruction}\n\n" if state.instruction else ""
        evidence = f"Compressed evidence:\n{summary}\n\n" if not empty else ""
        template = (
            RECOMP_SELECTIVE_READER_TEMPLATE
            if self.output_format == "answer_text"
            else RECOMP_SELECTIVE_IR_READER_TEMPLATE
        )
        prompt = template.format(instruction=instruction, evidence=evidence, question=state.query)
        remaining_after = _remaining_execution_budget(state, context)
        if (
            remaining_after["model_calls"] < 1
            or remaining_after["output_tokens"] < self.reader_max_tokens
        ):
            raise BudgetExceededError(
                "RECOMP actual compressor usage leaves no reserved reader budget"
            )
        response = context.backend.generate(
            GenerationRequest(
                prompt=prompt,
                max_tokens=self.reader_max_tokens,
                seed=context.seed,
                temperature=0,
                metadata={
                    "release_id": self.release_id,
                    "stage": "selective_reader",
                    "assembly_policy": self.assembly_policy,
                    "learned_empty_summary_gate": empty,
                },
            )
        )
        if not isinstance(response.text, str) or not response.text.strip():
            raise AlgorithmOutputError(
                "RECOMP reader returned an empty answer; no fallback or retry"
            )
        if self.output_format == "ir_query_json":
            parse_recomp_query_reader_answer(response.text, envelope_policy=self.query_envelope)
        policy = {
            "revision": self.assembly_policy,
            "reader_template": template,
            "reader_max_tokens": self.reader_max_tokens,
            "compressor_controls": asdict(self.compressor_controls),
            "empty_summary": "omit_all_retrieved_evidence",
            "compressor_failure": "propagate_no_reader_or_original_context_fallback",
        }
        if self.output_format != "answer_text":
            policy["output_format"] = self.output_format
            if self.query_envelope != "strict_json":
                policy.update(
                    query_envelope=self.query_envelope,
                    query_decoder_id=RECOMP_QUERY_DECODER_ID,
                    query_decoder_sha256=RECOMP_QUERY_DECODER_SHA256,
                )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=response.text,
            evidence=compressed.context,
            implementation_identity="pinned_RECOMP_abstractive+explicit_controlled_reader_v1",
            profile_kind="controlled_adapter",
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                events=(
                    *component.trace.events,
                    TraceEvent(
                        TracePhase.OBSERVE,
                        "learned_selective_augmentation",
                        {
                            "empty_summary": empty,
                            "original_context_in_reader": False,
                            "compressed_state_hash": compressed.state_hash,
                        },
                    ),
                    TraceEvent(
                        TracePhase.FINALIZE,
                        "selective_QA_reader",
                        {
                            "reader_prompt_hash": content_hash(prompt),
                            "reader_backend": dict(context.backend.fingerprint()),
                            "assembly_policy_hash": content_hash(policy),
                        },
                    ),
                ),
                counters={
                    **external_usage_counters(context.budget_ledger.usage),
                    "compressor_calls": component.trace.counters["model_calls"],
                    "reader_calls": 1,
                    "selected_context_blocks": len(compressed.context),
                },
                warnings=component.trace.warnings,
            ),
            metadata={
                "assembly_policy": policy,
                "assembly_policy_hash": content_hash(policy),
                "compressor_patch_hash": component.patch_hash,
                "compressor_metadata": dict(component.metadata),
                "selective_augmentation": empty,
                "all_calls_share_budget": True,
                "upstream_FLAN_UL2_full_prompt_or_published_score_parity": False,
            },
        )
