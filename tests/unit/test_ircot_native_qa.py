"""Synthetic boundary/cost tests; actual source/model acceptance is separate."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentinputlab import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.acquire.ircot_native import (
    IRCOT_NATIVE_QA_BINDING_MODE,
    IRCOT_NATIVE_QA_RELEASE,
    IRCoTNativeQARequest,
    write_ircot_substitution_manifest,
)
from agentinputlab.api import create_intervention, run_composite_intervention
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, Usage
from agentinputlab.contracts import ArtifactError, BudgetExceededError, CapabilityError
from agentinputlab.integrations.external_workers.ircot import (
    IRCOT_UPSTREAM_COMMIT,
    IRCoTControls,
    IRCoTWorkerRequest,
)
from workers.ircot.driver import DriverError, _parse_request
from workers.ircot.native_qa import QA_ROLES, _QABindings, patch_native_qa_config


class RuntimeFixture:
    def __init__(self, backend, retriever):
        self.calls = []
        self.roles = [
            "retrieve_and_reset_paragraphs",
            "step_by_step_cot_gen",
            "step_by_step_exit_controller",
            "copy_question",
            "llmqa",
            "answer_extractor",
            "copy_question",
        ]
        self.identity = dict(
            binding_mode=IRCOT_NATIVE_QA_BINDING_MODE,
            upstream_commit=IRCOT_UPSTREAM_COMMIT,
            full_six_participant_QA_graph=True,
            native_final_prediction_extraction=True,
            generator=backend.fingerprint(),
            retriever=retriever.fingerprint(),
            seed=17,
            max_model_calls=8,
            max_output_tokens=4096,
            max_retrieval_calls=8,
            per_call_output_tokens=512,
            tokenizer={"id": "synthetic-tokenizer"},
            native_dependencies={"id": "synthetic-native-dependencies"},
            corpus_revision="synthetic-corpus",
            index_revision="synthetic-index",
            prompt_sha256="a" * 64,
        )

    def fingerprint(self):
        return copy.deepcopy(self.identity)

    def run(self, request):
        self.calls.append(request)
        return dict(
            status="ok",
            identity=self.fingerprint(),
            request_id=request.legacy.request_id,
            upstream_commit=IRCOT_UPSTREAM_COMMIT,
            protocol_version="agentinputlab.ircot.v1",
            initial_query=request.legacy.question,
            initial_document_ids=["doc"],
            steps=[],
            documents=[
                dict(
                    document_id="doc",
                    text="SYNTHETIC SOURCE DOCUMENT",
                    title="Fixture",
                    corpus_revision="synthetic-corpus",
                )
            ],
            active_document_ids=["doc"],
            answer="SYNTHETIC NATIVE ANSWER",
            exit_reason="answer_detected",
            usage=dict(model_calls=2, retrieval_calls=1, input_tokens=20, output_tokens=10),
            native_controller_events=[{"role": r} for r in self.roles],
            native_model_classes={"fixture": "synthetic"},
            controlled_runtime_events=[],
            native_chain="SYNTHETIC CHAIN",
            usage_is_complete=True,
        )


@pytest.fixture
def bound(tmp_path):
    backend = CallableBackend(
        lambda request: GenerationResponse("UNEXPECTED PROVIDER CALL"),
        revision="fixture",
        callable_id="fixture",
    )
    retriever = InMemoryBM25Retriever(
        [Document("doc", "SYNTHETIC SOURCE DOCUMENT", corpus_revision="synthetic-corpus")]
    )
    runtime = RuntimeFixture(backend, retriever)
    receipt = write_ircot_substitution_manifest(
        runtime, tmp_path / "binding.json", accept_substitutions=True
    )
    method = create_intervention(
        IRCOT_NATIVE_QA_RELEASE,
        manifest_path=receipt["path"],
        expected_manifest_sha256=receipt["sha256"],
    )
    context = InterventionContext(
        backend=backend, retriever=retriever, external_runtime=runtime, seed=17
    )
    return method, context, runtime


def test_full_composite_factory_and_external_usage_charged_once(bound):
    method, context, runtime = bound
    result = run_composite_intervention(
        method, FrozenAgentState(query="Synthetic question?", task_id="fixture"), context
    )
    assert result["final_answer"] == "SYNTHETIC NATIVE ANSWER" and not result["scored"]
    assert (
        result["record"]["usage"]["model_calls"] == 2
        and result["record"]["usage"]["retrieval_calls"] == 1
    )
    assert (
        result["record"]["usage"]["input_tokens"] == 20
        and result["record"]["usage"]["output_tokens"] == 10
    )
    assert len(runtime.calls) == 1 and not context.backend.calls
    assert (
        result["record"]["result"]["metadata"]["independent_downstream_answer_call_added"] is False
    )
    assert result["evidence"][0]["block_id"] == "doc"


@pytest.mark.parametrize(
    "changes", [{"max_model_calls": 7}, {"max_output_tokens": 512}, {"max_retrieval_calls": 0}]
)
def test_declared_native_workflow_rejects_insufficient_shared_budget(bound, changes):
    method, context, runtime = bound
    with pytest.raises(BudgetExceededError):
        method.run(
            FrozenAgentState(query="Q", task_id="fixture", budget=ExecutionBudget(**changes)),
            context,
        )
    assert not runtime.calls


@pytest.mark.parametrize(
    "field,value", [("instruction", "Changed policy"), ("tool_names", ("tool",)), ("task_id", "")]
)
def test_unsupported_state_is_not_silently_ignored(bound, field, value):
    method, context, runtime = bound
    state = replace(FrozenAgentState(query="Q", task_id="fixture"), **{field: value})
    with pytest.raises(CapabilityError):
        method.run(state, context)
    assert not runtime.calls


def test_wrong_model_and_wrong_runtime_fail_before_execution(bound):
    method, context, runtime = bound
    runtime.identity["generator"] = {"model": "another-model"}
    with pytest.raises(ArtifactError, match="fingerprint mismatch"):
        method.run(FrozenAgentState(query="Q", task_id="fixture"), context)
    assert not runtime.calls


def test_retrieval_only_response_cannot_be_presented_as_full_qa(bound):
    method, context, runtime = bound
    runtime.roles = [
        "retrieve_and_reset_paragraphs",
        "step_by_step_cot_gen",
        "step_by_step_exit_controller",
    ]
    with pytest.raises(Exception, match="complete native final-answer graph"):
        method.run(FrozenAgentState(query="Q", task_id="fixture"), context)


def test_native_task_stack_drain_is_required_not_an_extra_answer(bound):
    method, context, runtime = bound
    runtime.roles = runtime.roles[:-1]
    with pytest.raises(Exception, match="complete native final-answer graph"):
        method.run(FrozenAgentState(query="Q", task_id="fixture"), context)


def test_substitution_acknowledgement_and_create_only_receipt(bound, tmp_path):
    _, _, runtime = bound
    with pytest.raises(ArtifactError, match="acknowledge"):
        write_ircot_substitution_manifest(runtime, tmp_path / "absent.json")
    with pytest.raises(FileExistsError):
        write_ircot_substitution_manifest(
            runtime, tmp_path / "binding.json", accept_substitutions=True
        )


def bindings(max_output=4096):
    controls = IRCoTControls(
        per_step_top_k=6, max_steps=7, max_total_documents=15, max_output_tokens=max_output
    )
    legacy = IRCoTWorkerRequest(
        request_id="fixture",
        state_hash="state",
        task_id="fixture",
        question="Q",
        controls=controls,
        generator_fingerprint={"id": "fixture"},
        retriever_fingerprint={"id": "fixture"},
        corpus_revision="fixture",
        index_revision="fixture",
        prompt_hash="a" * 64,
    )
    requests = []
    backend = CallableBackend(
        lambda r: GenerationResponse("TEXT", usage=Usage(3, r.max_tokens)),
        revision="fixture",
        callable_id="fixture",
    )
    owner = SimpleNamespace(per_call_output_tokens=512, seed=17, backend=backend, last_events=[])
    request = IRCoTNativeQARequest(legacy, 8, 512, 17, "b" * 64)
    value = _QABindings(owner, _parse_request(legacy.as_dict()), request)
    value.fit_prompt = lambda **kwargs: requests.append(kwargs) or kwargs["original_prompt"]
    return value, backend, requests


def test_reasoning_and_native_final_share_eight_calls_and_keep_native_300():
    value, backend, _ = bindings()
    reasoning = value.generator_class("reasoning")()
    final = value.generator_class("final_answer")()
    for _ in range(7):
        reasoning.generate_text_sequence("SOURCE PROMPT")
    with pytest.raises(DriverError, match="shared budget"):
        reasoning.generate_text_sequence("Q")
    final.generate_text_sequence("NATIVE FINAL PROMPT")
    assert len(backend.calls) == 8 and [r.max_tokens for r in backend.calls] == [300] * 8
    assert [r.seed for r in backend.calls] == list(range(17, 25))
    assert value.usage()["output_tokens"] == 2400
    with pytest.raises(DriverError, match="shared budget"):
        final.generate_text_sequence("Q")


def test_final_output_reservation_not_spent_by_another_reasoning_call():
    value, backend, _ = bindings(max_output=600)
    value.generator_class("reasoning")().generate_text_sequence("Q")
    with pytest.raises(DriverError, match="shared budget"):
        value.generator_class("reasoning")().generate_text_sequence("Q")
    value.generator_class("final_answer")().generate_text_sequence("FINAL")
    assert value.usage()["output_tokens"] == 600 and len(backend.calls) == 2


def test_unsupported_native_sampling_options_reject_before_provider():
    value, backend, _ = bindings()
    with pytest.raises(DriverError, match="sampling options"):
        value.generator_class("reasoning")(n=2).generate_text_sequence("Q")
    assert not backend.calls


def test_budget_patch_preserves_six_participant_transitions_and_parser(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("SYNTHETIC SOURCE PROMPT")
    models = {r: dict(name=r, next_model="native-next-" + r) for r in QA_ROLES}
    for r in ("step_by_step_cot_gen", "llmqa"):
        models[r].update(prompt_file="prompt.txt", gen_model="gpt3")
    models["answer_extractor"]["regex"] = "NATIVE REGEX"
    config = dict(
        models=models, prediction_type="answer", start_state="native-start", end_state="native-end"
    )
    patched = patch_native_qa_config(
        config,
        SimpleNamespace(root=tmp_path, prompt_path=prompt),
        IRCoTControls(per_step_top_k=6, max_steps=7, max_total_documents=15),
    )
    assert {k: v["next_model"] for k, v in patched["models"].items()} == {
        k: v["next_model"] for k, v in models.items()
    }
    assert patched["models"]["answer_extractor"]["regex"] == "NATIVE REGEX"
    assert patched["models"]["step_by_step_cot_gen"]["max_num_sentences"] == 7
    assert "max_tokens" not in patched["models"]["llmqa"]
    del config["models"]["llmqa"]
    with pytest.raises(DriverError, match="six original participants"):
        patch_native_qa_config(
            config, SimpleNamespace(root=tmp_path, prompt_path=prompt), IRCoTControls()
        )
