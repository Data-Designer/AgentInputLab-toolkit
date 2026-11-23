"""Opt-in final JSON envelope handling; these are synthetic decoder fixtures."""
import json

import pytest

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.contracts.state import FrozenAgentState
from agentinputlab.experiments.production_builders import (
    _parse_predictor_parameters,
    _parse_visible_qa_prediction,
)

RAW = json.dumps(dict(answer="unchanged", support_ids=[], evidence_ids=[], evidence_texts=[]))


def parse(text, **options):
    return _parse_visible_qa_prediction(
        text, benchmark_id="2wiki.multihop", state=FrozenAgentState(query="visible"),
        output_format="structured_json", **options,
    )


@pytest.mark.parametrize("prefix", ["```json\n", "```\n", "```json \r\n"])
def test_explicit_single_fence_only(prefix):
    text = prefix + RAW + "\n```"
    with pytest.raises(ValidationError):
        parse(text)
    prediction, audit = parse(text, allow_json_fence=True)
    strict, strict_audit = parse(RAW)
    assert prediction == strict
    assert audit["json_fence_removed"] is True
    assert audit["json_envelope_policy"] == "single_json_fence.v1"
    assert audit["original_response_text_sha256"] == content_hash(text)
    assert "json_envelope_policy" not in strict_audit


@pytest.mark.parametrize("text", [
    "Explanation\n```json\n" + RAW + "\n```",
    "```json\n" + RAW + "\n```\nextra",
    "```python\n" + RAW + "\n```",
    "```json\n" + RAW,
    "```json\n" + RAW + "\n```\n```json\n" + RAW + "\n```",
    "```json\n```json\n" + RAW + "\n```\n```",
    '```json\n{"answer":"a","answer":"b","support_ids":[],"evidence_ids":[],"evidence_texts":[]}\n```',
    '```json\n{"answer":"a","support_ids":[],"evidence_ids":[],"evidence_texts":[],"x":NaN}\n```',
    '```json\n{"answer":""}\n```',
])
def test_no_reasoning_salvage_schema_relaxation_or_duplicate_keys(text):
    with pytest.raises(ValidationError):
        parse(text, allow_json_fence=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_flag_is_boolean(value):
    with pytest.raises(ValidationError, match="boolean"):
        parse(RAW, allow_json_fence=value)
    with pytest.raises(ValidationError, match="boolean"):
        _parse_predictor_parameters({"schema_version": 1, "qa": {"allow_json_fence": value}})


def test_default_fingerprint_unchanged_and_opt_in_recorded():
    default = _parse_predictor_parameters({"schema_version": 1})
    disabled = _parse_predictor_parameters({"schema_version": 1, "qa": {"allow_json_fence": False}})
    enabled = _parse_predictor_parameters({"schema_version": 1, "qa": {"allow_json_fence": True}})
    assert default == disabled
    assert "allow_json_fence" not in default["qa"]
    assert enabled["qa"]["allow_json_fence"] is True
    assert content_hash(default) != content_hash(enabled)
    prediction, audit = parse(RAW, allow_json_fence=True)
    assert prediction.answer == "unchanged" and audit["json_fence_removed"] is False
    with pytest.raises(ValidationError, match="structured_json"):
        _parse_predictor_parameters({"schema_version": 1, "qa": {
            "output_format": "answer_only", "allow_json_fence": True}})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_public_evaluate_binds_flag_without_extra_calls(tmp_path, enabled, composite):
    from agentinputlab import InterventionContext, RunConfig
    from tests.unit.test_benchmark_configuration import Adapter, Backend
    from tests.unit.test_composite_ir_evaluate_configuration import setup

    text = "```json\n" + RAW + "\n```"
    if composite:
        config, backend, context = setup(tmp_path, text=text)
    else:
        config = RunConfig("baseline.no_op")
        backend = Backend([(text, 4)])
        context = InterventionContext(backend=backend)
    result = config.evaluate("2wiki.multihop", split="dev", example_ids=["a"],
        adapter=Adapter(), context=context, predictor_parameters={
            "schema_version": 1, "qa": {"allow_json_fence": enabled}})
    assert len(backend.requests) == 1
    assert result["scored_count"] == int(enabled)
    row = result["records"][0]
    assert row["usage"]["model_calls"] == 1 and row["usage"]["output_tokens"] == 4
    assert row["status"] == ("scored" if enabled else "failed")
    if enabled and composite:
        assert row["downstream_metadata"]["identifier_audit"]["json_fence_removed"] is True
