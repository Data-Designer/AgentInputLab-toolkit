"""Public Q-RAG composite file entry; synthetic fixtures never claim model acceptance."""

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from agentinputlab import RunConfig
from agentinputlab.algorithms.acquire.qrag import (
    QRAG_CHECKPOINT_FILE_SHA256,
    QRAG_SOURCE_MANIFEST_SHA256,
    QRAGArtifactIdentity,
)
from agentinputlab.contracts.errors import ValidationError


def configuration(release="acquire.qrag.reader_transfer_factorial"):
    digest = "a" * 64
    identity = QRAGArtifactIdentity(
        checkpoint_manifest_sha256=digest, checkpoint_file_sha256=QRAG_CHECKPOINT_FILE_SHA256,
        embedder_manifest_sha256=digest, source_manifest_sha256=QRAG_SOURCE_MANIFEST_SHA256,
        compatibility_manifest_sha256=digest, retriever_revision="fixture",
        retriever_deployment_fingerprint_sha256=digest, corpus_sha256=digest,
        index_sha256=digest, reader_repository="fixture", reader_revision="fixture",
        reader_deployment_fingerprint_sha256=digest,
        reader_profile=("factorial_reader_transfer" if "transfer" in release
                        else "upstream_qwq32b_family_controlled"))
    return {
        "release_id": release,
        "parameters": {"artifacts": asdict(identity), "acknowledgements": {
            "source_noassertion_acknowledged": True, "checkpoint_cc_by_4_0_acknowledged": True,
            "user_mounted_source": True, "user_mounted_checkpoint": True}},
        "backend": {"kind": "openai_compatible", "model_id": "fixture",
                    "base_url": "http://127.0.0.1:9999/v1",
                    "deployment_identity": {"revision": "fixture"}},
        "runtime": {"kind": "qrag_native", "source_root": "source",
                    "checkpoint_root": "checkpoint", "embedder_root": "embedder",
                    "compatibility_manifest_path": "manifest.json",
                    "python_executable": sys.executable, "device": "cpu"},
    }


@pytest.mark.parametrize("release", ["acquire.qrag.reader_transfer_factorial",
                                     "acquire.qrag.released_checkpoint_native"])
def test_composite_file_is_valid_without_loading_assets(tmp_path, monkeypatch, release):
    value = configuration(release)
    value["runtime"].update(policy_replay_path="saved.json", policy_replay_sha256="b" * 64)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value))
    monkeypatch.chdir(tmp_path.parent)
    config = RunConfig.from_file(path)
    assert config.validate()["valid"] is True
    assert config.runtime["policy_replay_path"] == str(tmp_path / "saved.json")
    assert config.runtime["source_root"] == str(tmp_path / "source")
    assert config.create_intervention().release_id == release


@pytest.mark.parametrize("change", [
    {"device": "cuda; command"}, {"timeout_seconds": 0}, {"timeout_seconds": True},
    {"python_executable": ""}, {"policy_replay_path": "saved.json"},
    {"policy_replay_sha256": "b" * 64}, {"shell": "arbitrary"}, {"fixture_mode": True},
    {"single_candidate_axis_compat": "true"}, {"single_candidate_axis_compat": 1},
    {"single_candidate_axis_compat": None},
    {"single_candidate_axis_compat": True, "policy_replay_path": "saved.json",
     "policy_replay_sha256": "b" * 64},
])
def test_invalid_native_runtime_rejected(change):
    value = configuration()
    value["runtime"].update(change)
    with pytest.raises(ValidationError):
        RunConfig.from_dict(value).validate()


def test_common_support_cannot_be_silently_replaced():
    value = configuration()
    value["release_id"] = "acquire.qrag.common_support"
    value["parameters"] = {}
    with pytest.raises(ValidationError):
        RunConfig.from_dict(value).validate()


def policy_response(candidates):
    from agentinputlab.contracts.serialization import content_hash

    ids = tuple(row["document_id"] for row in candidates)
    steps = []
    for index in range(min(2, len(ids))):
        selected = tuple(range(index + 1))
        steps.append({"index": index, "available_indices": tuple(range(index, len(ids))),
            "q_values": tuple(1.0 for _ in range(index, len(ids))), "action_index": index,
            "selected_q_value": 1.0, "selected_indices": selected,
            "selected_document_ids": tuple(ids[i] for i in selected)})
    return {"status": "ok", "trajectory": steps, "trajectory_sha256": content_hash(steps),
        "selected_indices": selected, "selected_document_ids": tuple(ids[i] for i in selected),
        "candidate_order_sha256": content_hash(ids),
        "candidate_text_sha256": content_hash(tuple(row["text"].strip() for row in candidates)),
        "policy_steps": len(steps), "usage": {"model_calls": 0, "retrieval_calls": 0,
            "input_tokens": 0, "output_tokens": 0, "embedding_calls": 2 + 2 * len(steps),
            "accelerator_seconds": 0.0},
        "attestation": {**{key: True for key in ("official_candidate_builder_called",
            "official_pqn_select_action_called", "official_qa_env_step_called", "safe_weights_only_load",
            "strict_state_dict_load", "checkpoint_eval_mode", "dummy_feedback_used", "seed_applied")},
            **{key: False for key in ("gold_answer_used", "supporting_fact_labels_used",
                                     "reward_used", "training_called", "stop_action_used")}}}


@pytest.fixture
def wired(monkeypatch):
    from agentinputlab.algorithms.acquire.qrag import common_support
    from agentinputlab.contracts.serialization import content_hash, freeze_mapping
    from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker
    from tests.unit.test_benchmark_configuration import Backend
    from workers.qrag import configured, driver

    backend = Backend([("Final answer: fixture", 3)])
    value = configuration()
    value["seed"] = 23
    value["budget"] = {"max_output_tokens": 200, "max_model_calls": 1}
    value["parameters"]["artifacts"]["reader_deployment_fingerprint_sha256"] = content_hash(backend.fingerprint())
    value["parameters"]["artifacts"]["compatibility_manifest_sha256"] = content_hash(
        json.loads(driver.DEFAULT_MANIFEST.read_text()))
    value["runtime"]["compatibility_manifest_path"] = str(driver.DEFAULT_MANIFEST)
    monkeypatch.setattr(RunConfig, "_backend", lambda self: backend)
    # Only mounted assets and physical inference are synthetic in this unit fixture.
    # Native envelopes, profile/trajectory checks, prompts, reader and budgets run unchanged.
    monkeypatch.setattr(driver.QRAGFrozenDriver, "_verify_source", lambda *_: "git_checkout")
    monkeypatch.setattr(driver.QRAGFrozenDriver, "_verify_checkpoint", lambda *_: None)
    monkeypatch.setattr(driver.QRAGFrozenDriver, "_verify_embedder", lambda *_: None)
    selections, invocations = [], []

    class Policy:
        def __init__(self, **options):
            assert options["device"] == "cpu"
            assert options["single_candidate_axis_compat"] is value["runtime"].get(
                "single_candidate_axis_compat", False)

        def select(self, request):
            selections.append(request)
            return freeze_mapping(policy_response(tuple({"document_id": r.document.doc_id,
                "text": r.document.text} for r in request.candidates)))

    monkeypatch.setattr(common_support, "LocalQRAGFrozenPolicyRuntime", Policy)

    def invoke(worker, envelope):
        invocations.append((worker, envelope))
        calls = []
        return {"response": configured.execute(envelope, calls), "model_responses": calls}

    monkeypatch.setattr(JsonLineSubprocessWorker, "call", invoke)
    return value, backend, selections, invocations


def state(instruction=""):
    from agentinputlab import ContextBlock, FrozenAgentState

    return FrozenAgentState(query="Which?", instruction=instruction,
        context=(ContextBlock(block_id="c0", text="first"), ContextBlock(block_id="c1", text="second")))


def test_file_live_composite_preserves_native_prompt_seed_budget_and_cost(wired):
    from agentinputlab.algorithms.acquire.qrag.bridge import (
        QRAG_READER_SYSTEM_PROMPT,
        QRAG_READER_USER_TEMPLATE,
    )

    value, backend, selections, invocations = wired
    result = RunConfig.from_dict(value).run(state())
    assert result["record"]["result"]["final_answer"] == "Final answer: fixture"
    assert result["record"]["usage"]["model_calls"] == 1
    assert result["record"]["usage"]["embedding_calls"] == 6
    assert result["record"]["usage"]["output_tokens"] == 3
    assert len(selections) == len(backend.requests) == len(invocations) == 1
    assert selections[0].seed == 23 and selections[0].max_steps == 2
    request = backend.requests[0]
    assert request.max_tokens == 200 and request.seed == 23 and request.temperature == 0
    assert request.system == QRAG_READER_SYSTEM_PROMPT
    assert request.prompt == QRAG_READER_USER_TEMPLATE.format(question="Which?", context="first\n\n---\n\nsecond")
    worker = invocations[0][0]
    assert worker.command == (sys.executable, "-I", "-B", "-m", "workers.qrag.configured")
    assert worker.environment["CUDA_VISIBLE_DEVICES"] == ""
    assert not Path(worker.working_directory).exists()


def test_file_live_native_axis_opt_in_reaches_worker_on_single_candidate(wired):
    from agentinputlab import ContextBlock, FrozenAgentState
    from agentinputlab.algorithms.acquire.qrag.bridge import QRAG_READER_USER_TEMPLATE

    value, backend, selections, invocations = wired
    value["release_id"] = "acquire.qrag.released_checkpoint_native"
    value["parameters"]["artifacts"]["reader_profile"] = "upstream_qwq32b_family_controlled"
    value["runtime"]["single_candidate_axis_compat"] = True
    visible = FrozenAgentState(query="Which?", context=(ContextBlock(block_id="c0", text="only"),))
    result = RunConfig.from_dict(value).run(visible)
    assert len(invocations) == len(selections) == len(backend.requests) == 1
    assert invocations[0][1]["configuration"]["runtime"]["single_candidate_axis_compat"] is True
    assert len(selections[0].candidates) == 1 and selections[0].max_steps == 2
    request = backend.requests[0]
    assert request.prompt == QRAG_READER_USER_TEMPLATE.format(question="Which?", context="only")
    assert request.max_tokens == 200 and request.seed == 23
    assert result["record"]["usage"]["model_calls"] == 1
    assert result["record"]["usage"]["embedding_calls"] == 4


def test_file_instruction_profile_preserves_instruction_and_selected_annotations(wired):
    from agentinputlab.algorithms.acquire.qrag.reader_adapter import (
        QRAG_INSTRUCTION_PROMPT_SHA256,
        QRAG_INSTRUCTION_READER_PROFILE,
        QRAG_INSTRUCTION_SYSTEM_PROMPT,
    )

    value, backend, _, _ = wired
    value["parameters"]["artifacts"].update(reader_profile=QRAG_INSTRUCTION_READER_PROFILE,
                                           reader_prompt_sha256=QRAG_INSTRUCTION_PROMPT_SHA256)
    RunConfig.from_dict(value).run(state("Return evidence verbatim."))
    request = backend.requests[0]
    assert request.system == QRAG_INSTRUCTION_SYSTEM_PROMPT
    assert json.loads(request.prompt)["task_instruction"] == "Return evidence verbatim."
    assert [p["document_id"] for p in json.loads(request.prompt)["selected_passages"]] == ["c0", "c1"]


@pytest.mark.parametrize(("instruction_profile", "answer", "scored"), [
    (True, "", True), (False, "", False), (True, None, False),
])
def test_qrag_instruction_evaluate_scores_empty_text_without_a_second_call(
    wired, instruction_profile, answer, scored,
):
    from agentinputlab.algorithms.acquire.qrag import (
        QRAG_INSTRUCTION_PROMPT_SHA256, QRAG_INSTRUCTION_READER_PROFILE,
    )
    from agentinputlab.benchmarks import BenchmarkExample
    from tests.unit.test_benchmark_configuration import Adapter

    class VisibleQA(Adapter):
        def examples(self, split):
            return (BenchmarkExample(example_id="a", query="Which?",
                instruction="Answer using the evidence." if instruction_profile else "",
                visible_context=state().context),)

        def score(self, example, prediction):
            assert prediction.answer == ""
            return super().score(example, prediction)

    value, backend, selections, invocations = wired
    if instruction_profile:
        value["parameters"]["artifacts"].update(reader_profile=QRAG_INSTRUCTION_READER_PROFILE,
                                               reader_prompt_sha256=QRAG_INSTRUCTION_PROMPT_SHA256)
    backend.responses = [(json.dumps(dict(answer=answer, support_ids=[], evidence_ids=[], evidence_texts=[])), 7)]
    adapter = VisibleQA()
    result = RunConfig.from_dict(value).evaluate("2wiki.multihop", split="dev", limit=1, adapter=adapter,
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "structured_json"}})
    row, = result["records"]
    assert row["scored"] is scored
    assert row["usage"]["model_calls"] == len(selections) == len(invocations) == len(backend.requests) == 1
    if scored:
        assert result["metrics_over_scored_cases"] == {"answer_em": 0.0}
        assert row["downstream_metadata"]["empty_answer_policy"] == "qrag_instruction_native_string.v1"
    else:
        assert row["failure"]["type"] == "ValidationError" and not adapter.scored


def test_reader_identity_mismatch_stops_before_policy_or_model(wired):
    value, backend, selections, invocations = wired
    value["parameters"]["artifacts"]["reader_deployment_fingerprint_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="reader deployment"):
        RunConfig.from_dict(value).run(state())
    assert not selections and not backend.requests and not invocations


def test_explicit_query_reader_does_not_mix_qa_and_query_instructions(wired):
    from agentinputlab.algorithms.acquire.qrag import (
        QRAG_QUERY_PROMPT_SHA256,
        QRAG_QUERY_READER_PROFILE,
        QRAG_QUERY_SYSTEM_PROMPT,
        parse_native_query_reader_answer,
    )

    value, backend, selections, invocations = wired
    value["parameters"]["artifacts"].update(reader_profile=QRAG_QUERY_READER_PROFILE,
                                           reader_prompt_sha256=QRAG_QUERY_PROMPT_SHA256)
    backend.responses = [('{"query":"visible request"}', 7)]
    result = RunConfig.from_dict(value).run(state('Return exactly {"query": "..."}.'))
    request = backend.requests[0]
    assert request.system == QRAG_QUERY_SYSTEM_PROMPT
    assert "support_ids" not in request.system and "answer (string)" not in request.system
    assert json.loads(request.prompt)["task_instruction"] == 'Return exactly {"query": "..."}.'
    assert request.max_tokens == 200 and request.seed == 23
    assert parse_native_query_reader_answer(result["record"]["result"]["final_answer"]) == {
        "query": "visible request"}
    assert len(selections) == len(invocations) == len(backend.requests) == 1
    assert result["record"]["usage"]["output_tokens"] == 7


def test_query_profile_rejects_old_qa_prompt_identity_before_execution(wired):
    from agentinputlab.algorithms.acquire.qrag import (
        QRAG_INSTRUCTION_PROMPT_SHA256,
        QRAG_QUERY_READER_PROFILE,
    )

    value, backend, selections, invocations = wired
    value["parameters"]["artifacts"].update(reader_profile=QRAG_QUERY_READER_PROFILE,
                                           reader_prompt_sha256=QRAG_INSTRUCTION_PROMPT_SHA256)
    with pytest.raises(ValidationError, match="reader prompt"):
        RunConfig.from_dict(value).run(state())
    assert not selections and not backend.requests and not invocations


def test_query_profile_rejects_qa_evaluation_before_inference(wired):
    from agentinputlab.algorithms.acquire.qrag import (
        QRAG_QUERY_PROMPT_SHA256,
        QRAG_QUERY_READER_PROFILE,
    )
    from tests.unit.test_benchmark_configuration import Adapter

    value, backend, selections, invocations = wired
    value["parameters"]["artifacts"].update(reader_profile=QRAG_QUERY_READER_PROFILE,
                                           reader_prompt_sha256=QRAG_QUERY_PROMPT_SHA256)
    with pytest.raises(ValidationError, match="requires composite_retrieval"):
        RunConfig.from_dict(value).evaluate("2wiki.multihop", split="dev", limit=1, adapter=Adapter())
    assert not selections and not backend.requests and not invocations


@pytest.mark.parametrize("query_profile", [False, True])
def test_ir_evaluation_requires_consistent_qrag_profile(wired, query_profile):
    from agentinputlab import ContextBlock, InterventionContext
    from agentinputlab.algorithms.acquire.qrag import (
        QRAG_INSTRUCTION_PROMPT_SHA256,
        QRAG_INSTRUCTION_READER_PROFILE,
        QRAG_QUERY_PROMPT_SHA256,
        QRAG_QUERY_READER_PROFILE,
    )
    from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
    from agentinputlab.benchmarks import BenchmarkExample
    from tests.unit.test_composite_ir_evaluate_configuration import IRAdapter, options

    class VisibleIR(IRAdapter):
        def examples(self, split):
            return (BenchmarkExample(example_id="a", query="savings",
                instruction='Return exactly {"query":"..."}.',
                visible_context=(ContextBlock(block_id="d", text="savings account"),
                                 ContextBlock(block_id="e", text="account rates")),
                evaluator_gold={"secret": "NEVER_SEND_GOLD"}),)

    value, backend, selections, invocations = wired
    value["parameters"]["artifacts"].update(
        reader_profile=QRAG_QUERY_READER_PROFILE if query_profile else QRAG_INSTRUCTION_READER_PROFILE,
        reader_prompt_sha256=QRAG_QUERY_PROMPT_SHA256 if query_profile else QRAG_INSTRUCTION_PROMPT_SHA256)
    backend.responses = [('{"query":"savings"}', 7)]
    config = RunConfig.from_dict(value)
    context = InterventionContext(retriever=InMemoryBM25Retriever([Document(doc_id="d", text="savings account")]))
    kwargs = dict(split="test", limit=1, adapter=VisibleIR(), context=context, predictor_parameters=options())
    if not query_profile:
        with pytest.raises(ValidationError, match="QA reader conflicts"):
            config.evaluate("beir.arguana", **kwargs)
        assert not selections and not backend.requests and not invocations
    else:
        result = config.evaluate("beir.arguana", **kwargs)
        assert result["scored_count"] == 1 and result["failed_count"] == 0
        row = result["records"][0]
        assert row["usage"]["model_calls"] == row["usage"]["retrieval_calls"] == 1
        assert len(selections) == len(invocations) == len(backend.requests) == 1
        assert "NEVER_SEND_GOLD" not in str(backend.requests)


@pytest.mark.parametrize("answer,tokens,failed", [
    ("answer", 3, False), ("", 3, True), ("answer", 201, True),
])
def test_evaluate_captures_child_response_and_charges_once(wired, answer, tokens, failed):
    from agentinputlab.benchmarks import BenchmarkExample
    from tests.unit.test_benchmark_configuration import Adapter

    class VisibleAdapter(Adapter):
        def examples(self, split):
            original = state()
            return (BenchmarkExample(example_id="a", query=original.query,
                                     visible_context=original.context, evaluator_gold={"answer": "answer"}),)

    value, backend, selections, invocations = wired
    backend.responses = [(answer, tokens)]
    result = RunConfig.from_dict(value).evaluate("2wiki.multihop", split="dev",
        example_ids=["a"], adapter=VisibleAdapter(("a",)),
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}})
    assert result["failed_count"] == int(failed)
    assert result["scored_count"] == int(not failed)
    row = result["records"][0]
    assert len(row["model_responses"]) == 1
    assert row["model_responses"][0]["response"]["text"] == answer
    assert row["usage"]["model_calls"] == 1
    assert row["usage"]["output_tokens"] == tokens
    assert row["usage"]["embedding_calls"] == 6
    assert len(backend.requests) == len(selections) == len(invocations) == 1
    if failed:
        assert "evaluation" not in row


@pytest.mark.parametrize("tamper", ["seed", "release", "artifacts", "fingerprint"])
def test_worker_rechecks_parent_identity_before_inference(wired, tamper):
    from workers.qrag import configured

    value, backend, selections, invocations = wired
    # First obtain the actual public envelope; then reset fixture observations.
    RunConfig.from_dict(value).run(state())
    envelope = json.loads(json.dumps(invocations[0][1]))
    backend.requests.clear()
    selections.clear()
    if tamper == "seed":
        envelope["request"]["payload"]["seed"] += 1
    elif tamper == "release":
        envelope["request"]["release_id"] = "baseline.no_op"
    elif tamper == "artifacts":
        envelope["request"]["payload"]["artifacts"]["corpus_sha256"] = "b" * 64
    else:
        envelope["reader_fingerprint"] = {"model_id": "different"}
    calls = []
    with pytest.raises(ValidationError):
        configured.execute(envelope, calls)
    assert not calls and not backend.requests and not selections


@pytest.mark.parametrize("tamper", [None, "query", "hash", "artifact", "trajectory"])
def test_explicit_policy_replay_never_falls_back_to_new_selection(wired, tmp_path, tamper):
    from agentinputlab.contracts.errors import AgentInputLabError
    from agentinputlab.contracts.serialization import to_jsonable

    value, backend, selections, invocations = wired
    candidates = ({"document_id": "c0", "text": "first"}, {"document_id": "c1", "text": "second"})
    record = {"schema": "agentinputlab.qrag.policy-replay.v1",
        "request": {"question": "Which?", "candidates": candidates, "seed": 23, "max_steps": 2},
        "artifact_identity_sha256": RunConfig.from_dict(value).create_intervention().artifacts.identity_sha256,
        "response": policy_response(candidates)}
    if tamper == "query":
        record["request"]["question"] = "Other question"
    if tamper == "artifact":
        record["artifact_identity_sha256"] = "b" * 64
    if tamper == "trajectory":
        record["response"]["selected_indices"] = [1, 1]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(to_jsonable(record)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    value["runtime"].update(policy_replay_path=str(path), policy_replay_sha256=digest)
    if tamper == "hash":
        path.write_text("{}")
    if tamper is None:
        result = RunConfig.from_dict(value).run(state())
        assert result["record"]["usage"]["embedding_calls"] == 6
        assert len(backend.requests) == 1
        assert invocations[0][0].identity["policy_execution"] == "explicit_saved_trajectory"
    else:
        with pytest.raises(AgentInputLabError):
            RunConfig.from_dict(value).run(state())
        assert not backend.requests and not invocations
    assert not selections


def test_invalid_saved_policy_is_not_a_consumed_model_attempt(wired, tmp_path):
    from agentinputlab.benchmarks import BenchmarkExample
    from tests.unit.test_benchmark_configuration import Adapter

    class VisibleAdapter(Adapter):
        def examples(self, split):
            return (BenchmarkExample(example_id="a", query=state().query,
                                     visible_context=state().context),)

    value, backend, selections, invocations = wired
    path = tmp_path / "invalid-policy.json"
    path.write_text('{}')
    value["runtime"].update(policy_replay_path=str(path),
        policy_replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = RunConfig.from_dict(value).evaluate("2wiki.multihop", split="dev",
        example_ids=["a"], adapter=VisibleAdapter(("a",)),
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"}})
    assert result["failed_count"] == 1
    assert not backend.requests and not selections
    assert not invocations
    assert result["records"][0]["usage"]["model_calls"] == 0
