"""Full native IRCoT-QA, with source-locked and explicit dependency substitutions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
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
)
from agentinputlab.contracts.serialization import content_hash, freeze_mapping, to_jsonable
from agentinputlab.integrations.external_workers.ircot import (
    IRCOT_UPSTREAM_COMMIT,
    IRCoTControls,
    IRCoTWorkerRequest,
    IRCoTWorkerResponse,
)

IRCOT_NATIVE_QA_RELEASE = "acquire.ircot.native_like_substituted"
IRCOT_NATIVE_QA_BINDING_MODE = "agentinputlab.ircot.native_like_qa.v1"
IRCOT_NATIVE_QA_MANIFEST_SCHEMA = "agentinputlab.ircot.native_like_substitutions.v1"
_ROLES = {
    "retrieve_and_reset_paragraphs",
    "step_by_step_cot_gen",
    "step_by_step_exit_controller",
    "copy_question",
    "llmqa",
    "answer_extractor",
}


def _manifest(fingerprint):
    if (
        not isinstance(fingerprint, Mapping)
        or fingerprint.get("binding_mode") != IRCOT_NATIVE_QA_BINDING_MODE
        or fingerprint.get("upstream_commit") != IRCOT_UPSTREAM_COMMIT
        or fingerprint.get("full_six_participant_QA_graph") is not True
        or fingerprint.get("native_final_prediction_extraction") is not True
    ):
        raise ArtifactError("IRCoT native-like profile requires the complete pinned QA runtime")
    for key in ("generator", "retriever", "tokenizer", "native_dependencies"):
        if not isinstance(fingerprint.get(key), Mapping) or not fingerprint[key]:
            raise ArtifactError("IRCoT native-like runtime dependency fingerprints are incomplete")
    for key in ("corpus_revision", "index_revision", "prompt_sha256"):
        if not isinstance(fingerprint.get(key), str) or not fingerprint[key].strip():
            raise ArtifactError("IRCoT native-like source identities are incomplete")
    for key in (
        "max_model_calls",
        "max_output_tokens",
        "max_retrieval_calls",
        "per_call_output_tokens",
    ):
        if type(fingerprint.get(key)) is not int or fingerprint[key] < 1:
            raise ArtifactError("IRCoT native-like runtime budgets must be positive integers")
    if type(fingerprint.get("seed")) is not int or fingerprint["seed"] < 0:
        raise ArtifactError("IRCoT native-like runtime seed must be a nonnegative integer")
    return dict(
        schema=IRCOT_NATIVE_QA_MANIFEST_SCHEMA,
        release_id=IRCOT_NATIVE_QA_RELEASE,
        runtime_fingerprint=to_jsonable(fingerprint),
        runtime_fingerprint_sha256=content_hash(fingerprint),
        substitutions=[
            "generation_backend",
            "retrieval_backend_and_index",
            "gpt2_tokenizer_resolver",
        ],
        original_controller_and_six_participants=True,
        native_answer_extraction=True,
        original_native_generator_default_cap=300,
        per_call_limit_is_a_ceiling_not_an_increase=True,
        shared_final_call_and_output_budget=True,
        original_cache_or_retry=False,
        historical_Codex_full_Elasticsearch_or_published_score_parity=False,
    )


def write_ircot_substitution_manifest(runtime, output, *, accept_substitutions=False):
    """Create-only operator acceptance of the exact native-like dependency set."""
    if accept_substitutions is not True:
        raise ArtifactError(
            "review the IRCoT model/index substitutions and acknowledge them explicitly"
        )
    value = _manifest(runtime.fingerprint())
    path = Path(output).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@dataclass(frozen=True, slots=True)
class IRCoTNativeQARequest:
    legacy: IRCoTWorkerRequest
    max_model_calls: int
    per_call_output_tokens: int
    seed: int
    expected_runtime_fingerprint_sha256: str

    def external_reservation(self):
        return {"model_calls": 1, "retrieval_calls": 1}

    def raise_for_external_status(self, result):
        if not isinstance(result, Mapping) or result.get("status") != "ok":
            code = (
                result.get("failure_code", "INVALID_RESPONSE")
                if isinstance(result, Mapping)
                else "INVALID_RESPONSE"
            )
            if code == "NATIVE_QA_EMPTY_OR_NON_TEXT":
                raise AlgorithmOutputError(
                    "native IRCoT-QA preserved an empty/non-text final prediction"
                )
            raise ExternalDependencyError(f"native IRCoT-QA failed with {code}")


@dataclass(frozen=True, slots=True)
class IRCoTNativeLikeSubstituted:
    manifest_path: str | Path
    expected_manifest_sha256: str
    controls: IRCoTControls | Mapping = field(
        default_factory=lambda: IRCoTControls(
            per_step_top_k=6, max_steps=7, max_total_documents=15, max_output_tokens=4096
        )
    )
    method_id: str = "ircot"
    release_id: str = IRCOT_NATIVE_QA_RELEASE
    family: SemanticFamily = SemanticFamily.ACQUIRE
    _binding: Mapping = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.controls, Mapping):
            try:
                object.__setattr__(self, "controls", IRCoTControls(**self.controls))
            except (TypeError, ValueError) as exc:
                raise ArtifactError("IRCoT full QA control fields are invalid") from exc
        if not isinstance(self.controls, IRCoTControls):
            raise ArtifactError("IRCoT full QA requires explicit controls")
        for key in ("per_step_top_k", "max_steps", "max_total_documents", "max_output_tokens"):
            if type(getattr(self.controls, key)) is not int or getattr(self.controls, key) < 1:
                raise ArtifactError("IRCoT full QA controls must be positive integers")
        if self.controls.deduplicate is not True:
            raise ArtifactError("the pinned native IRCoT source always deduplicates")
        if (
            self.method_id != "ircot"
            or self.release_id != IRCOT_NATIVE_QA_RELEASE
            or self.family is not SemanticFamily.ACQUIRE
        ):
            raise ArtifactError("IRCoT full QA registered profile identity differs")
        value = load_pinned_json_mapping(
            self.manifest_path, expected_sha256=self.expected_manifest_sha256
        )
        if value != _manifest(value.get("runtime_fingerprint")):
            raise ArtifactError(
                "IRCoT full QA substitution manifest differs from the exact supported protocol"
            )
        object.__setattr__(self, "_binding", freeze_mapping(value))

    def run(self, state: FrozenAgentState, context: InterventionContext) -> CompositeResult:
        if not state.query.strip() or not state.task_id:
            raise CapabilityError("IRCoT full QA requires question text and an explicit task ID")
        if state.context or state.messages or state.tool_names or state.instruction:
            raise CapabilityError(
                "IRCoT full QA accepts question-only state plus an explicitly bound external corpus"
            )
        binding = load_pinned_json_mapping(
            self.manifest_path, expected_sha256=self.expected_manifest_sha256
        )
        if content_hash(binding) != content_hash(self._binding):
            raise ArtifactError("IRCoT full QA substitution manifest changed")
        runtime = context.external_runtime
        if (
            runtime is None
            or not callable(getattr(runtime, "run", None))
            or not callable(getattr(runtime, "fingerprint", None))
        ):
            raise ExternalDependencyError(
                "IRCoT full QA requires the startup-bound native QA runtime"
            )
        fingerprint = runtime.fingerprint()
        if content_hash(fingerprint) != binding["runtime_fingerprint_sha256"]:
            raise ArtifactError("IRCoT full QA exact runtime/model/index fingerprint mismatch")
        for component, key in ((context.backend, "generator"), (context.retriever, "retriever")):
            if component is None or not callable(getattr(component, "fingerprint", None)):
                raise ExternalDependencyError(
                    "IRCoT full QA requires backend and retriever fingerprints"
                )
            if content_hash(component.fingerprint()) != content_hash(fingerprint[key]):
                raise ArtifactError(
                    "IRCoT full QA context is not the acknowledged model/index substitution"
                )
        controls = self.controls
        if (
            context.seed != fingerprint["seed"]
            or controls.max_steps + 1 > state.budget.max_model_calls
            or controls.max_steps + 1 > fingerprint["max_model_calls"]
            or controls.max_output_tokens
            > min(state.budget.max_output_tokens, fingerprint["max_output_tokens"])
            or controls.max_steps + 1
            > min(state.budget.max_retrieval_calls, fingerprint["max_retrieval_calls"])
        ):
            raise BudgetExceededError(
                "IRCoT full QA cannot reserve its declared reasoning/final workflow under the bound budget/seed"
            )
        base = dict(
            state_hash=state.state_hash,
            task_id=state.task_id,
            question=state.query,
            controls=asdict(controls),
            generator_fingerprint=fingerprint["generator"],
            retriever_fingerprint=fingerprint["retriever"],
            corpus_revision=fingerprint["corpus_revision"],
            index_revision=fingerprint["index_revision"],
            prompt_hash=fingerprint["prompt_sha256"],
        )
        legacy = IRCoTWorkerRequest(
            request_id=content_hash(
                dict(
                    base,
                    release_id=self.release_id,
                    manifest_sha256=self.expected_manifest_sha256,
                    seed=context.seed,
                )
            ),
            **{**base, "controls": controls},
        )
        request = IRCoTNativeQARequest(
            legacy,
            min(state.budget.max_model_calls, fingerprint["max_model_calls"]),
            fingerprint["per_call_output_tokens"],
            context.seed,
            content_hash(fingerprint),
        )
        raw = runtime.run(request)
        request.raise_for_external_status(raw)
        if raw.get("identity") != fingerprint or content_hash(
            runtime.fingerprint()
        ) != content_hash(fingerprint):
            raise ArtifactError("IRCoT full QA response/runtime identity changed")
        response = IRCoTWorkerResponse.from_mapping(raw, request=legacy)
        roles = [e["role"] for e in raw["native_controller_events"]]
        # The original CopyQuestion participant schedules a task. After answer
        # extraction the source controller visits it again to drain that task;
        # its eoq_after_n_calls=1 guard ends without another model generation.
        tail = ["copy_question", "llmqa", "answer_extractor", "copy_question"]
        prefix = roles[:-4]
        loop = [
            "retrieve_and_reset_paragraphs",
            "step_by_step_cot_gen",
            "step_by_step_exit_controller",
        ]
        if (
            not set(roles) >= _ROLES
            or roles[-4:] != tail
            or not prefix
            or len(prefix) % 3
            or any(prefix[i : i + 3] != loop for i in range(0, len(prefix), 3))
        ):
            raise AlgorithmOutputError(
                "IRCoT full QA did not execute the complete native final-answer graph"
            )
        if (
            not response.answer.strip()
            or response.usage["model_calls"] > request.max_model_calls
            or response.usage["model_calls"] != roles.count("step_by_step_cot_gen") + 1
        ):
            raise AlgorithmOutputError(
                "IRCoT full QA answer/model accounting differs from its native graph"
            )
        by_id = {d.document_id: d for d in response.documents}
        evidence = tuple(
            ContextBlock(
                text=by_id[i].text,
                block_id=i,
                source=response.documents[0].corpus_revision,
                score=by_id[i].score,
                metadata={
                    "title": by_id[i].title,
                    "corpus_revision": by_id[i].corpus_revision,
                    "evidence_kind": "real_corpus_document",
                },
            )
            for i in response.active_document_ids
        )
        return CompositeResult(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            final_answer=response.answer,
            evidence=evidence,
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                events=(
                    TraceEvent(TracePhase.PREPARE, "native_qa_substitution_binding", binding),
                    TraceEvent(
                        TracePhase.OBSERVE,
                        "native_qa_controller",
                        {
                            "events": raw["native_controller_events"],
                            "model_classes": raw["native_model_classes"],
                            "runtime_events": raw["controlled_runtime_events"],
                        },
                    ),
                    TraceEvent(
                        TracePhase.FINALIZE,
                        "native_qa_prediction",
                        {
                            "answer": response.answer,
                            "exit_reason": response.exit_reason,
                            "active_document_ids": response.active_document_ids,
                            "native_chain": raw["native_chain"],
                        },
                    ),
                ),
                counters=dict(response.usage),
            ),
            implementation_identity=IRCOT_NATIVE_QA_BINDING_MODE,
            profile_kind="controlled_adapter",
            metadata={
                "substitution_manifest_sha256": self.expected_manifest_sha256,
                "runtime_fingerprint": fingerprint,
                "native_full_QA_graph_executed": True,
                "native_final_prediction_extraction": True,
                "independent_downstream_answer_call_added": False,
                "native_generator_default_output_cap": 300,
                "current_provider_usage_is_measured": raw["usage_is_complete"],
                "original_historical_backbone_index_or_published_score_parity": False,
                "automatic_retry": False,
                "official_scoring": "caller_owned_independent_scorer",
            },
        )


__all__ = [
    "IRCOT_NATIVE_QA_RELEASE",
    "IRCoTNativeLikeSubstituted",
    "IRCoTNativeQARequest",
    "write_ircot_substitution_manifest",
]
