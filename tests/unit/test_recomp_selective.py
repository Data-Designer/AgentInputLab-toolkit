"""Synthetic composition boundaries, not checkpoint/model acceptance evidence."""

from dataclasses import replace

import pytest

from agentinputlab import ContextBlock, ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.curate.recomp import (
    RECOMPAbstractiveExternal,
    parse_recomp_query_reader_answer,
)
from agentinputlab.api import create_intervention, run_composite_intervention
from agentinputlab.backends import GenerationResponse, Usage
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ExecutionTrace,
    InputPatch,
    Message,
    PatchOp,
    PatchOperation,
    ValidationError,
)
from agentinputlab.observability import RunMonitor


class Backend:
    model_id = "synthetic-reader"
    capabilities = frozenset()

    def __init__(self, answer="answer"):
        self.answer, self.requests = answer, []

    def fingerprint(self):
        return {"model": self.model_id}

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(self.answer, model=self.model_id, usage=Usage(3, 4))


class Worker:
    worker_id = "synthetic-compressor"

    def __init__(self, summary="compressed fact", fail=False):
        self.summary, self.calls, self.fail = summary, 0, fail

    def fingerprint(self):
        return {"runtime": self.worker_id, "evidence": "synthetic_fixture_only"}

    def call(self, _request):
        self.calls += 1
        return {
            "summary": self.summary,
            "fail": self.fail,
            "usage": {
                "model_calls": 1,
                "retrieval_calls": 0,
                "input_tokens": 7,
                "output_tokens": 5,
            },
        }


@pytest.fixture(autouse=True)
def synthetic_compressor(monkeypatch):
    def propose(self, state, context):
        response = context.external_runtime.call({"fixture": True})
        if response["fail"]:
            raise AlgorithmOutputError("synthetic consumed compressor failure")
        summary = response["summary"]
        blocks = (ContextBlock(text=summary, block_id="synthetic-summary"),) if summary else ()
        return InputPatch(
            method_id=self.method_id,
            release_id=self.release_id,
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.REPLACE_CONTEXT, blocks),),
            trace=ExecutionTrace(
                method_id=self.method_id,
                release_id=self.release_id,
                counters={"model_calls": 1, "input_tokens": 7, "output_tokens": 5},
            ),
            implementation_identity="synthetic_fixture_only",
            profile_kind="controlled_adapter",
            metadata={"selective_augmentation": not summary},
        )

    monkeypatch.setattr(RECOMPAbstractiveExternal, "propose", propose)


def method(**kwargs):
    return create_intervention(
        "curate.recomp.selective",
        checkpoint_revision="fixture",
        tokenizer_revision="fixture",
        asset_manifest_hash="a" * 64,
        asset_license_acknowledged=True,
        **kwargs,
    )


def state():
    return FrozenAgentState(
        query="question?",
        instruction="be precise",
        context=(ContextBlock(block_id="raw", text="original uncompressed secret marker"),),
    )


@pytest.mark.parametrize("summary", ["compressed fact", ""])
def test_real_composition_preserves_gate_and_charges_both_stages_once(summary):
    worker, backend = Worker(summary), Backend()
    result = run_composite_intervention(
        method(), state(), InterventionContext(backend=backend, external_runtime=worker, seed=17)
    )
    assert result["final_answer"] == "answer" and result["scored"] is False
    record = result["record"]
    assert record["usage"]["model_calls"] == 2
    assert record["usage"]["input_tokens"] == 10 and record["usage"]["output_tokens"] == 9
    assert worker.calls == len(backend.requests) == 1
    request = backend.requests[0]
    assert request.seed == 17 and request.max_tokens == 512
    assert "original uncompressed secret marker" not in request.prompt
    assert ("Compressed evidence:" in request.prompt) == bool(summary)
    assert "Instructions: be precise" in request.prompt and "Question: question?" in request.prompt
    assert record["result"]["metadata"]["selective_augmentation"] is (not summary)
    assert len(result["evidence"]) == bool(summary)
    assert record["result"]["trace"]["counters"]["model_calls"] == 2


@pytest.mark.parametrize(
    "budget", [ExecutionBudget(max_model_calls=1), ExecutionBudget(max_output_tokens=1023)]
)
def test_reserve_two_calls_and_both_maxima_before_consuming_helper(budget):
    worker, backend = Worker(), Backend()
    with pytest.raises(BudgetExceededError, match="reserve"):
        run_composite_intervention(
            method(),
            replace(state(), budget=budget),
            InterventionContext(backend=backend, external_runtime=worker),
        )
    assert worker.calls == len(backend.requests) == 0


def test_compressor_error_is_not_empty_summary_or_original_context_fallback():
    worker, backend, monitor = Worker(fail=True), Backend(), RunMonitor()
    with pytest.raises(AlgorithmOutputError, match="compressor failure"):
        run_composite_intervention(
            method(),
            state(),
            InterventionContext(backend=backend, external_runtime=worker, monitor=monitor),
        )
    assert worker.calls == 1 and not backend.requests
    assert monitor.list_snapshots()[0].usage.model_calls == 1


def test_empty_reader_stays_consumed_failure_without_retry():
    worker, backend, monitor = Worker(""), Backend(""), RunMonitor()
    with pytest.raises(AlgorithmOutputError, match="empty answer"):
        run_composite_intervention(
            method(),
            state(),
            InterventionContext(backend=backend, external_runtime=worker, monitor=monitor),
        )
    assert worker.calls == len(backend.requests) == 1
    assert monitor.list_snapshots()[0].usage.model_calls == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"query": " "},
        {"context": ()},
        {"messages": (Message(role="user", content="prior"),)},
        {"tool_names": ("tool",)},
    ],
)
def test_invalid_or_silently_ignored_state_rejected_before_calls(changes):
    worker, backend = Worker(), Backend()
    with pytest.raises(ValidationError):
        run_composite_intervention(
            method(),
            replace(state(), **changes),
            InterventionContext(backend=backend, external_runtime=worker),
        )
    assert worker.calls == len(backend.requests) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"reader_max_tokens": 513},
        {"reader_max_tokens": True},
        {"assembly_policy": "tuned_after_test"},
        {"compressor_controls": {"max_target_tokens": 513}},
    ],
)
def test_invalid_policy_or_budget_caps_fail_at_construction(changes):
    with pytest.raises(ValidationError):
        method(**changes)


def test_data_only_compressor_config_supported_without_executable_constructor_args():
    instance = method(compressor_controls={"max_target_tokens": 200}, reader_max_tokens=400)
    assert instance.compressor_controls.max_target_tokens == 200
    assert instance.reader_max_tokens == 400


def test_default_reader_prompt_remains_byte_exact():
    backend = Backend()
    run_composite_intervention(
        method(), state(), InterventionContext(backend=backend, external_runtime=Worker(), seed=17)
    )
    assert backend.requests[0].prompt == (
        "Answer the question. Use the compressed evidence when it is provided. "
        "Return only the answer, without an explanation.\n\n"
        "Instructions: be precise\n\nCompressed evidence:\ncompressed fact\n\n"
        "Question: question?\nAnswer:"
    )
    assert backend.requests[0].metadata["assembly_policy"] == (
        "agentinputlab.recomp.learned_empty_gate_reader.v1"
    )


@pytest.mark.parametrize("summary", ["compressed fact", ""])
def test_explicit_ir_query_mode_preserves_instructions_and_original_shared_caps(summary):
    backend, worker = Backend('{"query":"useful search query"}'), Worker(summary)
    original = state()
    result = run_composite_intervention(
        method(output_format="ir_query_json"),
        original,
        InterventionContext(backend=backend, external_runtime=worker, seed=17),
    )
    assert result["final_answer"] == '{"query":"useful search query"}'
    assert original == state() and result["record"]["usage"]["model_calls"] == 2
    assert worker.calls == len(backend.requests) == 1
    request = backend.requests[0]
    assert request.seed == 17 and request.max_tokens == 512
    assert 'one nonempty string field "query"' in request.prompt
    assert (
        "Instructions: be precise" in request.prompt
        and "Original query: question?" in request.prompt
    )
    assert "original uncompressed secret marker" not in request.prompt
    assert ("Compressed evidence:" in request.prompt) is bool(summary)
    metadata = result["record"]["result"]["metadata"]
    assert metadata["assembly_policy"]["output_format"] == "ir_query_json"
    assert metadata["assembly_policy"]["revision"] == (
        "agentinputlab.recomp.learned_empty_gate_ir_query_reader.v1"
    )


@pytest.mark.parametrize(
    "answer",
    ["plain answer", "{}", '{"query":""}', '{"query":1}', '{"query":"x","answer":"y"}', '["x"]'],
)
def test_ir_wrong_output_is_consumed_failure_without_fallback_or_retry(answer):
    backend, worker, monitor = Backend(answer), Worker(), RunMonitor()
    with pytest.raises(AlgorithmOutputError, match="IR reader"):
        run_composite_intervention(
            method(output_format="ir_query_json"),
            state(),
            InterventionContext(backend=backend, external_runtime=worker, monitor=monitor),
        )
    assert worker.calls == len(backend.requests) == 1
    assert monitor.list_snapshots()[0].usage.model_calls == 2


@pytest.mark.parametrize(
    "budget", [ExecutionBudget(max_model_calls=1), ExecutionBudget(max_output_tokens=1023)]
)
def test_ir_mode_uses_original_preconsumption_budget_reservation(budget):
    worker, backend = Worker(), Backend('{"query":"x"}')
    with pytest.raises(BudgetExceededError):
        run_composite_intervention(
            method(output_format="ir_query_json"),
            replace(state(), budget=budget),
            InterventionContext(backend=backend, external_runtime=worker),
        )
    assert worker.calls == len(backend.requests) == 0


def test_unknown_output_mode_rejected_at_construction():
    with pytest.raises(ValidationError, match="output_format"):
        method(output_format="guess_from_test_results")


def test_complete_fence_decoder_preserves_query_verbatim_and_strict_failure():
    text = '```json\n{"query":"  visible query  "}\n```'
    with pytest.raises(AlgorithmOutputError):
        parse_recomp_query_reader_answer(text)
    assert parse_recomp_query_reader_answer(text, envelope_policy="complete_json_or_fence") == {
        "query": "  visible query  "
    }


@pytest.mark.parametrize(
    "text",
    [
        'prose ```json\n{"query":"x"}\n```',
        '```json\n{"query":"x"}\n``` explanation',
        '{"query":"x","query":"y"}',
        '{"query":"x","answer":"y"}',
        '```json\n{"answer":"x"}\n```',
        '```json\n{"query":"x"}\n```\n```json\n{"query":"y"}\n```',
        '```json\n{"query":""}\n```',
    ],
)
def test_query_decoder_rejects_prose_duplicates_QA_fields_and_multiple_objects(text):
    with pytest.raises(AlgorithmOutputError):
        parse_recomp_query_reader_answer(text, envelope_policy="complete_json_or_fence")


def test_explicit_fence_mode_changes_only_decoding_not_request_or_helper():
    text = '```json\n{"query":"x"}\n```'
    strict_backend, compatible_backend = Backend(text), Backend(text)
    with pytest.raises(AlgorithmOutputError):
        run_composite_intervention(
            method(output_format="ir_query_json"),
            state(),
            InterventionContext(backend=strict_backend, external_runtime=Worker()),
        )
    result = run_composite_intervention(
        method(output_format="ir_query_json", query_envelope="complete_json_or_fence"),
        state(),
        InterventionContext(backend=compatible_backend, external_runtime=Worker()),
    )
    assert strict_backend.requests == compatible_backend.requests
    assert result["final_answer"] == text
    assert result["record"]["usage"]["model_calls"] == 2
    assert result["record"]["result"]["metadata"]["assembly_policy"]["query_envelope"] == (
        "complete_json_or_fence"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_format": []},
        {"query_envelope": []},
        {"query_envelope": "complete_json_or_fence"},
        {"output_format": "ir_query_json", "query_envelope": "extract_arbitrary_text"},
    ],
)
def test_bad_envelope_or_mode_rejected_without_model_calls(kwargs):
    with pytest.raises(ValidationError):
        method(**kwargs)
