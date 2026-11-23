import json
from types import SimpleNamespace

import pytest

from agentinputlab.benchmarks.base import BenchmarkExample
from agentinputlab.contracts import ValidationError
from agentinputlab.recipes import ACEBFCLFeedback, ACEOnlineBFCLFeedback
from agentinputlab.recipes.ace_bfcl import parse_bfcl_ace_answer


class Adapter:
    benchmark_id = "bfcl.v3.simple_python"

    def __init__(self):
        self.evaluation_provenance = {"scorer": "fixture_AST", "revision": "one"}
        self.prepared = SimpleNamespace(file_sha256={"fixture": "digest"})
        self.cases = tuple(
            BenchmarkExample(
                f"case-{i}",
                f"Find item {i}.",
                instruction="Use typed arguments.",
                visible_tool_names=("lookup",),
                visible_metadata={"function": {"name": "lookup", "parameters": {"x": "integer"}}},
                evaluator_gold={"hidden": "secret-reference"},
            )
            for i in range(3)
        )
        self.calls = []

    def verify_integrity(self):
        pass

    def function_tools(self, example):
        function = dict(example.visible_metadata["function"])
        function["name"] = function["name"].replace(".", "_")
        return ({"type": "function", "function": function},)

    def score(self, example, prediction):
        self.calls.append((example.example_id, prediction))
        assert all(isinstance(arguments, str) for call in prediction for arguments in call.values())
        decoded = [
            {name: json.loads(arguments)} for call in prediction for name, arguments in call.items()
        ]
        return {"ast_accuracy": float(decoded == [{"lookup": {"x": 3}}])}


def make():
    a = Adapter()
    return ACEBFCLFeedback(adapter=a, train=a.cases[:1], validation=a.cases[1:2]), a


def test_visible_inputs_preserved_gold_blind_and_frozen():
    p, a = make()
    rows = p.fit_examples("train")
    text = json.dumps(rows[0].as_dict()["input"])
    assert all(s in text for s in ("Find item 0", "Use typed arguments", "lookup", "integer"))
    assert "secret-reference" not in text and rows[0].expected_output == "case-0"
    p.validate_examples([e.as_dict() for e in rows], [p.fit_examples("validation")[0].as_dict()])
    changed = rows[0].as_dict()
    changed["input"]["question"] += "changed"
    with pytest.raises(ValidationError, match="frozen"):
        p.validate_examples([changed], [p.fit_examples("validation")[0].as_dict()])
    assert not a.calls


def test_native_scorer_not_exact_match_and_mean():
    p, a = make()
    assert p.answer_is_correct('[{"lookup": {"x": 3}}]', "case-0")
    assert (
        p.evaluate_accuracy(['[{"lookup":{"x":3}}]', '[{"lookup":{"x":4}}]'], ["case-0", "case-1"])
        == 0.5
    )
    assert a.calls[-1] == ("case-1", [{"lookup": '{"x":4}'}])
    assert p.evaluate_accuracy([], []) == 0
    with pytest.raises(ValidationError, match="length"):
        p.evaluate_accuracy(["[]"], [])


@pytest.mark.parametrize(
    "text",
    [
        "x",
        "[]",
        "{}",
        '[{"a":{},"b":{}}]',
        '[{"a":[]}]',
        '[{"a":{},"a":{}}]',
        '[{"a":{"x":NaN}}]',
        '```json\n[{"a":{}}]\n```',
    ],
)
def test_invalid_output_not_repaired_or_scored(text):
    p, a = make()
    assert not p.answer_is_correct(text, "case-0")
    assert not a.calls
    with pytest.raises(ValueError):
        parse_bfcl_ace_answer(text)


def test_unbound_target_drift_and_scorer_errors_are_not_zero():
    p, a = make()
    with pytest.raises(ValidationError, match="outside"):
        p.score_answer('[{"lookup":{}}]', "case-2")
    a.prepared.file_sha256["fixture"] = "changed"
    with pytest.raises(ValidationError, match="changed"):
        p.score_answer('[{"lookup":{}}]', "case-0")
    p, a = make()
    a.score = lambda *args: (_ for _ in ()).throw(OSError("native scorer missing"))
    with pytest.raises(OSError):
        p.score_answer('[{"lookup":{}}]', "case-0")


def test_disjoint_groups_and_not_standard_leaderboard_claim():
    p, a = make()
    with pytest.raises(ValidationError, match="disjoint"):
        ACEBFCLFeedback(adapter=a, train=a.cases[:1], validation=a.cases[:1])
    with pytest.raises(ValidationError, match="only"):
        p.fit_examples("test")
    assert not p.fingerprint()["native_BFCL_handler_prompt_parity_claimed"]


def test_visible_catalog_uses_native_names_without_repairing_model_output():
    a = Adapter()
    a.cases = tuple(
        BenchmarkExample(
            str(i),
            "query" + str(i),
            visible_tool_names=("math.factorial",),
            visible_metadata={"function": {"name": "math.factorial"}},
        )
        for i in range(2)
    )
    p = ACEBFCLFeedback(adapter=a, train=a.cases[:1], validation=a.cases[1:])
    text = p.fit_examples("train")[0].input["question"]
    assert "math.factorial" in text and "math_factorial" in text and "native_function_tools" in text
    p.score_answer('[{"math.factorial":{"x":3}}]', "0")
    assert a.calls[-1][1] == [{"math.factorial": '{"x":3}'}]


@pytest.mark.parametrize("count", [1, 2])
def test_native_online_stream_is_ordered_transductive_feedback_not_offline(count):
    a = Adapter()
    p = ACEOnlineBFCLFeedback(adapter=a, stream=a.cases[:count])
    rows = tuple(row.as_dict() for row in p.stream_examples())
    assert [r["example_id"] for r in rows] == [e.example_id for e in a.cases[:count]]
    assert all(r["split"] == "test" for r in rows)
    assert "secret-reference" not in json.dumps([r["input"] for r in rows])
    p.validate_online_examples(rows)
    assert p.fingerprint()["transductive_feedback_not_offline_or_heldout"]
    assert p.fingerprint()["inputs"].keys() == {"stream"}
    assert p.score_answer('[{"lookup":{"x":3}}]', "case-0") == 1
    with pytest.raises(ValidationError, match="offline"):
        p.fit_examples("train")
    with pytest.raises(ValidationError, match="offline"):
        p.validate_examples([], [])
    changed = [dict(r) for r in rows]
    changed[0]["expected_output"] = "unknown"
    with pytest.raises(ValidationError, match="frozen"):
        p.validate_online_examples(changed)


def test_online_stream_reordering_and_unknown_feedback_target_are_rejected():
    a = Adapter()
    p = ACEOnlineBFCLFeedback(adapter=a, stream=a.cases[:2])
    with pytest.raises(ValidationError, match="frozen"):
        p.validate_online_examples(tuple(reversed(p.stream_examples())))
    with pytest.raises(ValidationError, match="outside"):
        p.score_answer('[{"lookup":{"x":3}}]', "case-2")
    with pytest.raises(ValidationError, match="nonempty"):
        ACEOnlineBFCLFeedback(adapter=a, stream=[])
