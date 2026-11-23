"""Synthetic protocol fixtures, not real routing/model/benchmark evidence."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from prepare_adaptive_trained_acceptance import choose_cases, query_key
from run_adaptive_trained_acceptance import StrictCapturedRouter, capture_identity

from agentinputlab.contracts.serialization import content_hash
from workers.adaptive_rag.driver import RuntimeIdentity


def example(identifier, question):
    return SimpleNamespace(
        example_id=identifier, visible_state=lambda: SimpleNamespace(query=question)
    )


def test_selection_is_input_based_source_sorted_and_excludes_warm_start():
    examples = [
        example("c", "New c"),
        example("a", " TRAIN \u00a0QUERY "),
        example("b", "New b"),
        example("d", "New d"),
    ]
    chosen, skipped = choose_cases(examples, {query_key("train query")}, start_index=0)
    assert [(i, e.example_id) for i, e in chosen] == [(1, "b"), (2, "c")]
    assert skipped == [dict(index=0, case_id="a", reason="training_query_overlap")]
    assert query_key("\uff21 B") == query_key("a b")


def test_selection_does_not_silently_shrink_if_fewer_disjoint_cases():
    with pytest.raises(ValueError, match="insufficient"):
        choose_cases([example("a", "question")], set(), start_index=0)


def fixture():
    identity = RuntimeIdentity(
        **{
            k: "a" * 64 if k.endswith("sha256") else "fixture-fixed-v1"
            for k in RuntimeIdentity.__dataclass_fields__
            if k != "fixture_runtime"
        }
    )
    bindings = {
        k: getattr(identity, k)
        for k in (
            "router_revision",
            "router_snapshot_sha256",
            "router_weights_sha256",
            "router_tokenizer_sha256",
        )
    }
    row = dict(
        case=dict(question="Original", question_sha256=content_hash("Original")),
        seed=17,
        trained_artifact_id="fixture",
        status="native_router_returned",
        router_bindings=bindings,
        result=dict(
            logits=dict(A=0.5, B=1.5, C=-1),
            input_tokens=7,
            output_tokens=1,
            accelerator_seconds=0.0,
            cost_usd=None,
        ),
    )
    return identity, dict(row, capture_id=content_hash(row))


def test_exact_capture_reuse_is_once_only_and_keeps_original_usage():
    identity, row = fixture()
    router = StrictCapturedRouter(row, identity)
    assert router.estimate_input_tokens("Original") == 7
    result = router.first_decoder_step("Original", seed=17)
    assert result.logits == row["result"]["logits"] and result.cost_usd is None
    assert result.output_tokens == 1 and len(router.logical_calls) == 1
    assert router.logical_calls[0]["new_physical_router_calls"] == 0
    with pytest.raises(ValueError, match="once-only"):
        router.first_decoder_step("Original", seed=17)


def test_mismatched_snapshot_query_seed_and_mutated_capture_rejected():
    identity, row = fixture()
    with pytest.raises(ValueError, match="identity differs"):
        StrictCapturedRouter(row, replace(identity, router_weights_sha256="b" * 64))
    router = StrictCapturedRouter(row, identity)
    with pytest.raises(ValueError, match="query differs"):
        router.estimate_input_tokens("changed")
    with pytest.raises(ValueError, match="seed"):
        router.first_decoder_step("Original", seed=18)
    changed = {**row, "result": {**row["result"], "output_tokens": 2}}
    with pytest.raises(ValueError, match="identity differs"):
        capture_identity(changed)


def test_failed_real_capture_is_preserved_not_a_successful_placeholder():
    identity, row = fixture()
    row.pop("capture_id")
    row.pop("result")
    row.update(status="failed_native_router", error="preserved failure")
    row["capture_id"] = content_hash(row)
    router = StrictCapturedRouter(row, identity)
    with pytest.raises(RuntimeError, match="preserved failed"):
        router.first_decoder_step("Original", seed=17)
    assert len(router.logical_calls) == 1
