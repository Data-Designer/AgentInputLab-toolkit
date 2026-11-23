"""Public LongBench binding contracts; synthetic fixtures are not model acceptance."""
import json

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.benchmarks.longbench import _SUBSETS
from agentinputlab.contracts import ContextBlock
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from tests.unit.test_benchmark_configuration import Backend


class Adapter:
    pairing_mode = "stateless_score_twice"

    def __init__(self, cell):
        self.benchmark_id = cell
        self.predictions = []

    def examples(self, split):
        assert split == "test"
        return (BenchmarkExample(example_id="case", query="Question",
            visible_context=(ContextBlock(block_id="long_context", text="Visible context"),),
            evaluator_gold={"answers": ("NEVER_SEND_GOLD",)}),)

    def render(self, state):
        return "NATIVE TEMPLATE\n" + state.query + "\n" + "\n".join(b.text for b in state.context)

    def score(self, example, prediction):
        assert isinstance(prediction, str)
        self.predictions.append(prediction)
        return {"native_score": float(prediction == "answer")}


def run(backend, cell="longbench.qasper", **kwargs):
    config = kwargs.pop("config", RunConfig("baseline.no_op"))
    return config.evaluate(cell, split="test", example_ids=["case"],
        adapter=kwargs.pop("adapter", Adapter(cell)), context=InterventionContext(backend=backend), **kwargs)


@pytest.mark.parametrize("cell", _SUBSETS)
def test_all_six_native_text_cells_without_qa_json_or_gold(cell):
    backend = Backend([("answer", 3)])
    result = run(backend, cell)
    assert result["scored_count"] == 1
    assert result["metrics_over_scored_cases"] == {"native_score": 1.0}
    request = backend.requests[0]
    assert request.prompt == "NATIVE TEMPLATE\nQuestion\nVisible context"
    assert request.max_tokens == 512 and request.seed == 17
    assert "NEVER_SEND_GOLD" not in repr(request)
    assert result["records"][0]["usage"]["model_calls"] == 1


def test_caps_temperature_and_native_zero_are_preserved():
    backend = Backend([("wrong", 3)])
    result = run(backend, config=RunConfig("baseline.no_op", budget=ExecutionBudget(max_output_tokens=40)),
        predictor_parameters={"schema_version": 1, "native_text": {"max_output_tokens": 80, "temperature": .3}})
    assert backend.requests[0].max_tokens == 40 and backend.requests[0].temperature == .3
    assert result["scored_count"] == 1 and result["metrics_over_scored_cases"] == {"native_score": 0.0}


def test_failure_keeps_response_and_denominator_no_retry():
    backend = Backend([("answer", 9)])
    result = run(backend, config=RunConfig("baseline.no_op", budget=ExecutionBudget(max_output_tokens=4)))
    assert result["failed_count"] == result["selected_count"] == 1
    assert len(backend.requests) == 1
    assert result["records"][0]["model_responses"][0]["response"]["text"] == "answer"
    assert result["records"][0]["usage"]["output_tokens"] == 9


def test_no_budget_does_not_generate():
    backend = Backend([])
    result = run(backend, config=RunConfig("baseline.no_op", budget=ExecutionBudget(max_model_calls=0)))
    assert result["failed_count"] == 1 and not backend.requests


def test_missing_public_source_is_actionable_without_model_call():
    backend = Backend([])
    with pytest.raises(ArtifactError, match="accept_license_review"):
        RunConfig("baseline.no_op").evaluate("longbench.qasper", split="test", limit=1,
            context=InterventionContext(backend=backend))
    assert not backend.requests


def test_cache_evaluation_uses_public_loader(tmp_path, monkeypatch):
    import agentinputlab.benchmarks as benchmarks

    seen = []
    adapter = Adapter("longbench.qasper")
    def load(cell, **kwargs):
        seen.append((cell, kwargs))
        return adapter
    monkeypatch.setattr(benchmarks, "load_benchmark", load)
    backend = Backend([("answer", 3)])
    result = RunConfig("baseline.no_op").evaluate("longbench.qasper", split="test", limit=1,
        cache_dir=tmp_path, offline=True, accept_license_review=True,
        context=InterventionContext(backend=backend))
    assert seen == [("longbench.qasper", {"cache_dir": tmp_path, "offline": True,
                                        "accept_license_review": True})]
    assert result["scored_count"] == 1 and len(backend.requests) == 1


def test_source_file_relative_paths_and_unknown_fields(tmp_path):
    from agentinputlab.longbench_configuration import read_sources

    path = tmp_path / "source.json"
    path.write_text(json.dumps({"data_zip": "data.zip", "scorer_path": "metrics.py", "prompts_path": "prompts.json"}))
    assert read_sources(path)["data_zip"] == str(tmp_path / "data.zip")
    path.write_text('{"data_zip":"a","data_zip":"b"}')
    with pytest.raises(ValidationError):
        read_sources(path)


def test_other_benchmark_rejects_longbench_sources_before_generation():
    backend = Backend([])
    with pytest.raises(ValidationError, match="only.*LongBench"):
        RunConfig("baseline.no_op").evaluate("2wiki.multihop", split="dev", limit=1,
            longbench_sources={}, context=InterventionContext(backend=backend))
    assert not backend.requests


def test_rewritten_search_keeps_original_task_and_native_reader_budget(monkeypatch):
    from dataclasses import dataclass

    from agentinputlab.contracts import ExecutionTrace, InputPatch, PatchOp, PatchOperation, SemanticFamily

    @dataclass
    class Rewrite:
        method_id: str = "query2doc"
        release_id: str = "rewrite.query2doc.querygym_live"
        family: SemanticFamily = SemanticFamily.REFORMULATE

        def propose(self, state, context):
            return InputPatch(method_id=self.method_id, release_id=self.release_id,
                source_state_hash=state.state_hash, implementation_identity="test_only",
                trace=ExecutionTrace(self.method_id, self.release_id),
                profile_kind="synthetic", operations=(PatchOperation(PatchOp.REPLACE_QUERY, "context"),))

    monkeypatch.setattr(RunConfig, "create_intervention", lambda self: Rewrite())
    cfg = RunConfig(Rewrite.release_id, retriever={"kind": "visible_context_bm25", "chunk_words": 1},
                    budget=ExecutionBudget(max_retrieval_calls=1, max_model_calls=1, max_output_tokens=12))
    backend = Backend([("answer", 3)])
    result = run(backend, "longbench.qmsum", config=cfg,
        predictor_parameters={"schema_version": 1, "native_text": {
            "preserve_original_question": True, "rewrite_retrieval_top_k": 1}})
    row, = result["records"]
    assert row["status"] == "scored", row.get("failure")
    assert row["longbench_question_projection"]["added_retrieval_calls"] == 1
    assert backend.requests[0].prompt == "NATIVE TEMPLATE\nQuestion\ncontext"
    assert backend.requests[0].max_tokens == 12
    assert row["usage"]["retrieval_calls"] == row["usage"]["model_calls"] == 1


def test_question_projection_rejects_noncase_evidence_and_preserves_unchanged_input():
    from dataclasses import replace

    from agentinputlab import FrozenAgentState
    from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
    from agentinputlab.longbench_configuration import project_native_question

    state = FrozenAgentState(query="Original", context=(ContextBlock("Original body", block_id="src"),))
    other = InMemoryBM25Retriever((Document("outside", "Not in the case"),))
    assert project_native_question(state, state, retriever=other, top_k=1)[0] is state
    with pytest.raises(ValidationError, match="outside the original case"):
        project_native_question(state, replace(state, query="rewrite"), retriever=other, top_k=1)


def test_reformer_prefill_and_rewrite_use_same_full_case_index(monkeypatch):
    from dataclasses import dataclass

    from agentinputlab.contracts import ExecutionTrace, InputPatch, PatchOp, PatchOperation, SemanticFamily

    seen = []

    @dataclass
    class Reformer:
        method_id: str = "reformer"
        release_id: str = "rewrite.reformer.repo_native_r_only"
        family: SemanticFamily = SemanticFamily.REFORMULATE
        top_document_count: int = 3

        def propose(self, state, context):
            seen.extend(b.text.strip() for b in state.context)
            assert state.budget.max_retrieval_calls == 1
            return InputPatch(method_id=self.method_id, release_id=self.release_id,
                source_state_hash=state.state_hash, implementation_identity="test_only",
                trace=ExecutionTrace(self.method_id, self.release_id),
                profile_kind="synthetic", operations=(PatchOperation(PatchOp.REPLACE_QUERY, "z"),))

    class DocumentAdapter(Adapter):
        def examples(self, split):
            return (BenchmarkExample(example_id="case", query="a",
                visible_context=(ContextBlock(block_id="original", text="a b c d z"),)),)

    monkeypatch.setattr(RunConfig, "create_intervention", lambda self: Reformer())
    cfg = RunConfig(Reformer.release_id, retriever={"kind": "visible_context_bm25", "chunk_words": 1},
                    budget=ExecutionBudget(max_retrieval_calls=2, max_model_calls=1, max_output_tokens=12))
    backend = Backend([("answer", 3)])
    result = run(backend, "longbench.qmsum", config=cfg, adapter=DocumentAdapter("longbench.qmsum"),
        predictor_parameters={"schema_version": 1, "native_text": {
            "preserve_original_question": True, "rewrite_retrieval_top_k": 1, "reformer_context_top_k": 3}})
    row, = result["records"]
    assert row["status"] == "scored", row.get("failure")
    assert seen == ["a", "b", "c"]
    assert backend.requests[0].prompt == "NATIVE TEMPLATE\na\nz"
    assert row["usage"]["retrieval_calls"] == 2
    assert row["usage"]["model_calls"] == 1


def test_cli_mounts_native_files_relative_to_source_config(tmp_path, monkeypatch, capsys):
    from agentinputlab.cli.main import main
    from tests.unit.test_longbench_adapter import fixture_adapter

    fixture_adapter(tmp_path, monkeypatch)  # hash-locked synthetic native source
    source = tmp_path / "sources.json"
    source.write_text(json.dumps({"data_zip": "synthetic-data.zip", "scorer_path": "synthetic_metrics.py",
        "prompts_path": "synthetic_prompts.json"}))
    runfile = tmp_path / "run.json"
    runfile.write_text('{"release_id":"baseline.no_op"}')
    backend = Backend([("wrong", 3)])
    monkeypatch.setattr(RunConfig, "_configured_context", lambda *args: InterventionContext(backend=backend))
    output = tmp_path / "results.jsonl"
    assert main(["config", "evaluate", str(runfile), "--benchmark", "longbench.qasper", "--split", "test",
        "--limit", "1", "--longbench-config", str(source), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["scored_count"] == 1
    assert "Public synthetic document" in backend.requests[0].prompt
    assert "gold-only-answer" not in repr(backend.requests)
    # A second invocation may not overwrite results or trigger another call.
    assert main(["config", "evaluate", str(runfile), "--benchmark", "longbench.qasper", "--split", "test",
        "--limit", "1", "--longbench-config", str(source), "--output", str(output)]) == 2
    assert len(backend.requests) == 1


@pytest.mark.parametrize("backend_present", [False, True])
def test_composite_final_text_scores_without_a_second_reader(monkeypatch, backend_present):
    from dataclasses import dataclass

    from agentinputlab.contracts import CompositeResult, ExecutionTrace, SemanticFamily

    @dataclass
    class Method:
        method_id: str = "fixture"
        release_id: str = "fixture"
        family: SemanticFamily = SemanticFamily.DECOMPOSE

        def run(self, state, context):
            return CompositeResult(method_id=self.method_id, release_id=self.release_id,
                source_state_hash=state.state_hash, final_answer="answer", evidence=(),
                trace=ExecutionTrace(method_id=self.method_id, release_id=self.release_id),
                implementation_identity="synthetic", profile_kind="synthetic")

    monkeypatch.setattr(RunConfig, "create_intervention", lambda self: Method())
    result = run(Backend([]) if backend_present else None)
    assert result["scored_count"] == 1
    assert result["records"][0]["model_responses"] == []
    assert result["records"][0]["downstream_metadata"]["native_template_used"] is False


class CompositeAdapter(Adapter):
    def examples(self, split):
        from dataclasses import replace

        return tuple(replace(e, example_id="a", query="Visible question a")
                     for e in super().examples(split))


@pytest.mark.parametrize("cell", _SUBSETS)
@pytest.mark.parametrize("fenced", [False, True])
def test_structured_composite_scores_answer_only_without_generation(tmp_path, cell, fenced):
    from tests.unit.test_composite_ir_evaluate_configuration import setup

    text = json.dumps(dict(answer="answer", support_ids=["x", "x"],
                           evidence_ids=["e", "e"], evidence_texts=[]))
    if fenced:
        text = "```json\n" + text + "\n```"
    config, backend, context = setup(tmp_path, text=text)
    adapter = CompositeAdapter(cell)
    result = config.evaluate(cell, split="test", example_ids=["a"], adapter=adapter,
        context=context, predictor_parameters={"schema_version": 1, "native_text": {
            "composite_output_format": "structured_qa_json", "allow_json_fence": fenced}})
    assert result["scored_count"] == 1
    assert adapter.predictions == ["answer"]
    assert len(backend.requests) == 1
    row = result["records"][0]
    assert row["intervention"]["result"]["final_answer"] == text
    assert row["usage"]["model_calls"] == 1
    assert row["downstream_metadata"]["scoring_fields"] == ["answer"]
    assert row["downstream_metadata"]["support_and_evidence_scored"] is False


@pytest.mark.parametrize("text", [
    "not json", '{"answer":"answer"}',
    '{"answer":"answer","answer":"other","support_ids":[],"evidence_ids":[],"evidence_texts":[]}',
    '```json\n{"answer":"answer","support_ids":[],"evidence_ids":[],"evidence_texts":[]}\n```',
])
def test_structured_composite_rejects_invalid_or_unapproved_envelope(tmp_path, text):
    from tests.unit.test_composite_ir_evaluate_configuration import setup

    config, backend, context = setup(tmp_path, text=text)
    adapter = CompositeAdapter("longbench.qmsum")
    result = config.evaluate("longbench.qmsum", split="test", example_ids=["a"],
        adapter=adapter, context=context, predictor_parameters={"schema_version": 1,
            "native_text": {"composite_output_format": "structured_qa_json"}})
    assert result["failed_count"] == 1 and not adapter.predictions
    assert len(backend.requests) == 1
    assert result["records"][0]["intervention"]["result"]["final_answer"] == text


def test_default_composite_keeps_text_unmodified(tmp_path):
    from tests.unit.test_composite_ir_evaluate_configuration import setup

    text = '{"answer":"answer","support_ids":[],"evidence_ids":[],"evidence_texts":[]}'
    config, backend, context = setup(tmp_path, text=text)
    adapter = CompositeAdapter("longbench.qmsum")
    result = config.evaluate("longbench.qmsum", split="test", example_ids=["a"],
        adapter=adapter, context=context)
    assert result["scored_count"] == 1 and adapter.predictions == [text]
    assert result["metrics_over_scored_cases"]["native_score"] == 0.0


@pytest.mark.parametrize("settings", [
    {"composite_output_format": "unknown"}, {"allow_json_fence": True},
    {"composite_output_format": "structured_qa_json", "allow_json_fence": "true"},
    {"composite_output_format": "structured_qa_json", "allow_json_fence": 1},
])
def test_invalid_decoder_configuration_rejected_before_calls(settings):
    backend = Backend([])
    with pytest.raises(ValidationError):
        run(backend, predictor_parameters={"schema_version": 1, "native_text": settings})
    assert not backend.requests


def test_structured_composite_option_rejects_patch_before_calls():
    backend = Backend([])
    with pytest.raises(ValidationError, match="answer-producing"):
        run(backend, predictor_parameters={"schema_version": 1, "native_text": {
            "composite_output_format": "structured_qa_json"}})
    assert not backend.requests
