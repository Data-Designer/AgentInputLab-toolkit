"""Public composite-to-IR binding; synthetic fixtures, not live-model evidence."""

import hashlib
import json

import pytest

from agentinputlab import ExecutionBudget, InterventionContext, RunConfig
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.contracts.errors import ValidationError
from tests.algorithms.unit.test_plan_and_budget_full import PromptFixture
from tests.unit.test_benchmark_configuration import Backend


class IRAdapter:
    benchmark_id = "beir.arguana"
    pairing_mode = "stateless_score_twice"

    def examples(self, split):
        return (
            BenchmarkExample(
                example_id="a",
                query="Visible question a",
                evaluator_gold={"secret": "NEVER_SEND_GOLD"},
            ),
        )

    def score(self, example, prediction):
        assert set(prediction.scores) == {"d"}
        return {"ndcg_at_10": 1.0}


def setup(tmp_path, text='{"query":"savings"}', budget=None):
    path = tmp_path / "cache.json"
    raw = json.dumps(
        {
            "last_question_total": 1,
            "stats": [
                {
                    "id": "a",
                    "query": "Visible question a",
                    "steps": ["Find"],
                    "credits": [100],
                    "level": 3,
                }
            ],
        }
    ).encode()
    path.write_bytes(raw)
    config = RunConfig(
        "plan.plan_and_budget.full",
        parameters={"output_format": "JSON query"},
        artifacts={"records": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}},
        **({"budget": budget} if budget else {}),
    )
    backend = Backend([(text, 4)])
    retriever = InMemoryBM25Retriever([Document(doc_id="d", text="savings account")])
    context = InterventionContext(
        backend=backend, retriever=retriever, external_runtime=PromptFixture()
    )
    return config, backend, context


def evaluate(config, context, options):
    return config.evaluate(
        "beir.arguana",
        split="test",
        example_ids=["a"],
        adapter=IRAdapter(),
        context=context,
        predictor_parameters=options,
    )


def options(**binding):
    return {
        "schema_version": 1,
        "retrieval": {"top_k": 1},
        "composite_retrieval": {"output_format": "json_query", **binding},
    }


def test_explicit_binding_scores_own_query_no_extra_reader(tmp_path):
    config, backend, context = setup(tmp_path)
    result = evaluate(config, context, options())
    row = result["records"][0]
    assert result["scored_count"] == 1 and len(backend.requests) == 1
    assert row["usage"]["model_calls"] == row["usage"]["retrieval_calls"] == 1
    assert row["downstream_metadata"]["decoder_policy"] == "strict_json_query.v1"
    assert "NEVER_SEND_GOLD" not in str(backend.requests)


def test_default_still_rejects_without_model_calls(tmp_path):
    config, backend, context = setup(tmp_path)
    with pytest.raises(ValidationError, match="QA prediction/scorer"):
        evaluate(config, context, {"schema_version": 1})
    assert backend.requests == []


@pytest.mark.parametrize("compat,expected", [(False, 0), (True, 1)])
def test_fence_is_opt_in_and_retains_response_and_usage(tmp_path, compat, expected):
    config, backend, context = setup(tmp_path, '```json\n{"query":"savings"}\n```')
    result = evaluate(config, context, options(allow_json_fence=compat))
    row = result["records"][0]
    assert result["scored_count"] == expected and len(backend.requests) == 1
    assert row["model_responses"][0]["response"]["text"].startswith("```")
    assert row["usage"]["output_tokens"] == 4
    assert row["usage"]["retrieval_calls"] == expected


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"savings"}',
        '{"query":"a","query":"b"}',
        'prefix {"query":"savings"}',
        '{"query":""}',
    ],
)
def test_invalid_output_never_falls_back_or_retries(tmp_path, text):
    config, backend, context = setup(tmp_path, text)
    result = evaluate(config, context, options(allow_json_fence=True))
    row = result["records"][0]
    assert result["scored_count"] == 0 and len(backend.requests) == 1
    assert row["usage"]["retrieval_calls"] == 0 and row["usage"]["output_tokens"] == 4


def test_retrieval_budget_still_enforced(tmp_path):
    config, backend, context = setup(tmp_path, budget=ExecutionBudget(max_retrieval_calls=0))
    result = evaluate(config, context, options())
    assert result["scored_count"] == 0 and len(backend.requests) == 1
    assert result["records"][0]["usage"]["retrieval_calls"] == 0


@pytest.mark.parametrize(
    "binding",
    [
        None,
        {},
        {"output_format": "answer"},
        {"output_format": "json_query", "allow_json_fence": 1},
        {"output_format": "json_query", "surprise": True},
    ],
)
def test_bad_bindings_fail_before_model(tmp_path, binding):
    config, backend, context = setup(tmp_path)
    with pytest.raises(ValidationError, match="composite_retrieval"):
        evaluate(config, context, {"schema_version": 1, "composite_retrieval": binding})
    assert backend.requests == []


@pytest.mark.parametrize(
    "retrieval", [{"query_mode": "instruction_and_query"}, {"source_mode": "context_or_retriever"}]
)
def test_ambiguous_query_source_rejected(tmp_path, retrieval):
    config, backend, context = setup(tmp_path)
    cfg = options()
    cfg["retrieval"] = retrieval
    with pytest.raises(ValidationError, match="composite_retrieval"):
        evaluate(config, context, cfg)
    assert backend.requests == []


def test_patch_profiles_cannot_silently_ignore_binding(tmp_path):
    _, backend, context = setup(tmp_path)
    with pytest.raises(ValidationError, match="composite_retrieval"):
        evaluate(RunConfig("baseline.no_op"), context, options())
    assert backend.requests == []


def test_cli_reads_explicit_binding_and_scores(tmp_path, monkeypatch, capsys):
    import agentinputlab.benchmarks
    from agentinputlab.cli.main import main

    config, backend, context = setup(tmp_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config.to_dict()))
    predictor_path = tmp_path / "ir-query.json"
    predictor_path.write_text(json.dumps(options()))
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(RunConfig, "_configured_context", lambda self, supplied: context)
    monkeypatch.setattr(
        agentinputlab.benchmarks, "load_production_benchmark_adapter", lambda *a: IRAdapter()
    )
    assert (
        main(
            [
                "config",
                "evaluate",
                str(config_path),
                "--benchmark",
                "beir.arguana",
                "--split",
                "test",
                "--case-id",
                "a",
                "--prepared-root",
                str(tmp_path),
                "--predictor-config",
                str(predictor_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    scored = [row for row in rows if row.get("status") == "scored"]
    assert len(scored) == 1 and len(backend.requests) == 1
    assert scored[0]["downstream_metadata"]["decoder_policy"] == "strict_json_query.v1"
