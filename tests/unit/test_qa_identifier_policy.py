"""Evaluator-blind native QA identifier normalization and legacy replay."""

import json
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.backends.llm import GenerationResponse, ReplayBackend
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import canonical_json, content_hash, to_jsonable
from agentinputlab.contracts.state import ContextBlock, FrozenAgentState
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import (
    ProductionPredictorBuilder,
    _parse_visible_qa_prediction,
)


def _parse(supports=(), evidence=(), *, benchmark="hotpotqa.distractor", policy="native_sets.v1"):
    state = FrozenAgentState(
        query="visible question",
        context=tuple(
            ContextBlock(
                text="visible text",
                block_id=f"block-{index}",
                metadata={"official_support_id": canonical_json([title, sentence])},
            )
            for index, (title, sentence) in enumerate((("Title", 0), ("Other", 2)))
        ),
        metadata={"gold": "never read"},
    )
    return _parse_visible_qa_prediction(
        json.dumps(dict(answer="unchanged", support_ids=supports, evidence_ids=evidence, evidence_texts=[])),
        benchmark_id=benchmark,
        state=state,
        output_format="structured_json",
        identifier_policy=policy,
    )


@pytest.mark.parametrize(
    "supports",
    [
        ["Title", 0, "Other", 2],
        [["Title", 0], ["Other", 2]],
        [json.dumps(["Title", 0]), '["Other", 2]'],
        [canonical_json(["Title", 0]), "Other", 2],
    ],
)
def test_native_pair_representations_produce_same_visible_ids(supports):
    prediction, audit = _parse(supports)
    assert prediction.answer == "unchanged"
    assert prediction.support_ids == (canonical_json(["Title", 0]), canonical_json(["Other", 2]))
    assert audit["identifier_policy"] == "native_sets.v1"
    assert audit["valid_support_id_count"] == 2
    assert audit["invalid_support_id_count"] == 0
    assert audit["support_raw_identifier_array_sha256"] == content_hash(supports)


def test_pair_deduplication_occurs_after_canonicalization():
    prediction, audit = _parse(["Title", 0, ["Title", 0], '["Title", 0]'])
    assert prediction.support_ids == (canonical_json(["Title", 0]),)
    assert audit["support_normalized_id_count"] == 3
    assert audit["support_duplicate_id_count"] == 2
    assert audit["support_duplicate_ids_sha256"] == content_hash([canonical_json(["Title", 0])] * 2)


def test_native_triples_accept_nested_or_json_strings_but_never_infer_relations():
    triple = ["Subject", "relation", "Object"]
    prediction, audit = _parse(
        ["Title", 0, "unknown-chunk"],
        [json.dumps(triple), triple, "Subject", "relation", "Object", ["Title", 0]],
        benchmark="2wiki.multihop",
    )
    assert prediction.support_ids == (canonical_json(["Title", 0]),)
    assert prediction.evidence_ids == (canonical_json(triple),)
    assert audit["evidence_duplicate_id_count"] == 1
    assert audit["invalid_support_id_count"] == 1
    assert audit["invalid_evidence_id_count"] == 4


@pytest.mark.parametrize("bad_index", [True, False, -1, 0.0, "0", None, {}, []])
def test_never_coerce_invalid_native_indices(bad_index):
    prediction, audit = _parse([["Title", bad_index]])
    assert prediction.answer == "unchanged"
    assert prediction.support_ids == ()
    assert audit["invalid_support_id_count"] == 1
    assert audit["invalid_support_ids_sha256"] == content_hash((["Title", bad_index],))


@pytest.mark.parametrize("bad_index", [True, False, -1, 0.0, "0", None])
def test_ambiguous_flat_pairs_are_not_completed_from_visible_titles(bad_index):
    prediction, audit = _parse(["Title", bad_index])
    assert prediction.support_ids == ()
    assert audit["invalid_support_id_count"] == 2


def test_visible_filter_does_not_find_hidden_or_title_only_support():
    prediction, audit = _parse(["Title", "Hidden", 0, ["Title", 9], ["Other", 2]])
    assert prediction.support_ids == (canonical_json(["Other", 2]),)
    assert audit["invalid_support_id_count"] == 3


def test_non_native_ids_are_opaque_not_reinterpreted_as_support_pairs():
    prediction, audit = _parse([["Title", 0], "Title", 0], benchmark="musique.answerable")
    assert prediction.support_ids == ()
    assert audit["support_normalized_id_count"] == 0
    assert audit["invalid_support_id_count"] == 3


def test_duplicate_opaque_evidence_ids_are_audited_not_an_answer_failure():
    prediction, audit = _parse(evidence=["Title", "Title", "Other"])
    assert prediction.evidence_ids == ("Title", "Other")
    assert audit["evidence_duplicate_id_count"] == 1
    assert audit["evidence_duplicate_ids_sha256"] == content_hash(["Title"])


@pytest.mark.parametrize(
    "supports,evidence,message",
    [(["Title", 0], [], "non-empty strings"), ([], ["x", "x"], "duplicates")],
)
def test_explicit_legacy_policy_retains_original_type_and_duplicate_errors(supports, evidence, message):
    with pytest.raises(ValidationError, match=message):
        _parse(supports, evidence, policy="strict_strings.v1")


def test_legacy_policy_retains_noncanonical_string_filtering():
    prediction, audit = _parse(
        ['["Title", 0]'], ['["Subject", "relation", "Object"]'],
        benchmark="2wiki.multihop", policy="strict_strings.v1",
    )
    assert prediction.support_ids == prediction.evidence_ids == ()
    assert audit["invalid_support_id_count"] == audit["invalid_evidence_id_count"] == 1


@pytest.mark.parametrize("text", ['{"answer":', '{"answer":"a","answer":"b"}', '{"answer":"a"}'])
def test_normalization_never_repairs_json_or_missing_response_fields(text):
    with pytest.raises(ValidationError):
        _parse_visible_qa_prediction(
            text, benchmark_id="hotpotqa.distractor", state=FrozenAgentState(query="question"),
            output_format="structured_json",
        )


def test_parser_policy_changes_fingerprint_not_neural_request(monkeypatch):
    captured = []

    def generate(self, request):
        captured.append(to_jsonable(request))
        return GenerationResponse(text=json.dumps(dict(answer="answer", support_ids=[], evidence_ids=[], evidence_texts=[])))

    monkeypatch.setattr(ReplayBackend, "generate", generate)
    runtime = StaticRuntimeComponents(backend=ReplayBackend(["unused"]))
    builder = ProductionPredictorBuilder()
    fingerprints = []
    for policy in ("native_sets.v1", "strict_strings.v1"):
        predictor = builder.build(StaticPredictorBuildRequest(
            benchmark_id="hotpotqa.distractor", split="validation", model_slot="fixture", seed=17,
            builder_parameters={"schema_version": 1, "qa": {"identifier_policy": policy}}, runtime=runtime,
        ))
        result = predictor.predict(FrozenAgentState(query="question"))
        assert result.metadata["identifier_policy"] == policy
        fingerprints.append(predictor.fingerprint())
    assert captured[0] == captured[1]
    assert fingerprints[0] != fingerprints[1]
    with pytest.raises(ValidationError, match="identifier_policy"):
        builder.build(StaticPredictorBuildRequest(
            benchmark_id="hotpotqa.distractor", split="validation", model_slot="fixture", seed=17,
            builder_parameters={"schema_version": 1, "qa": {"identifier_policy": "guess"}}, runtime=runtime,
        ))


@pytest.mark.parametrize("invalid", [None, True, 0, [], {}, "guess"])
def test_invalid_identifier_policy_is_a_schema_error(invalid):
    with pytest.raises(ValidationError, match="identifier policy"):
        _parse_visible_qa_prediction(
            "unused", benchmark_id="hotpotqa.distractor", state=FrozenAgentState(query="q"),
            output_format="structured_json", identifier_policy=invalid,
        )


@pytest.mark.parametrize("invalid", [None, "Title", 0, {}])
def test_identifier_field_itself_must_remain_an_array(invalid):
    with pytest.raises(ValidationError, match="must be an array"):
        _parse(invalid)
