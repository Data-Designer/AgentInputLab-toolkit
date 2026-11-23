from __future__ import annotations

from copy import deepcopy

from agentinputlab.experiments import (
    automatic_input_label,
    automatic_proximal_repair_quality,
    automatic_trajectory_validity,
    deterministic_stratified_sample,
    infer_visible_contract,
    visible_contract_retention,
)


def _checkpoint() -> dict[str, object]:
    return {
        "benchmark_id": "tau3.banking_knowledge",
        "checkpoint_id": "checkpoint-1",
        "group_id": "group-1",
        "visible_state": {
            "task_id": "task-1",
            "query": "Transfer exactly $250 to account AB12CD34 before Friday.",
            "instruction": "Do not change the recipient.",
            "context": [],
            "messages": [],
            "tool_names": ["banking_api"],
            "metadata": {},
            "budget": {},
        },
    }


def test_automatic_input_label_is_outcome_blind_and_five_family_eligible() -> None:
    checkpoint = _checkpoint()
    first = automatic_input_label(checkpoint)
    with_outcomes = deepcopy(checkpoint)
    with_outcomes["gold"] = "secret"
    with_outcomes["reward"] = 1.0
    second = automatic_input_label(with_outcomes)

    assert first == second
    assert all(first["eligibility_labels"].values())
    assert first["evidence"]["outcome_fields_accessed"] is False
    assert first["evidence"]["gold_fields_accessed"] is False


def test_automatic_trajectory_checks_entity_numeric_and_constraint_preservation() -> None:
    checkpoint = _checkpoint()
    pre_state = checkpoint["visible_state"]
    contract = infer_visible_contract(pre_state)
    base = {
        "slot_id": "slot-1",
        "group_id": "group-1",
        "benchmark_id": "tau3.banking_knowledge",
        "checkpoint_id": "checkpoint-1",
        "candidate_family": "clarify",
        "pre_state": pre_state,
        "protected_contract": contract,
    }
    valid = automatic_trajectory_validity(
        {
            **base,
            "post_state": {
                "query": (
                    "Transfer exactly $250 to account AB12CD34 before Friday. "
                    "Do not change the recipient."
                ),
                "instruction": "",
                "context": [],
                "messages": [],
            },
        }
    )
    invalid = automatic_trajectory_validity(
        {
            **base,
            "post_state": {
                "query": "Transfer $500 to a different account.",
                "instruction": "",
                "context": [],
                "messages": [],
            },
        }
    )

    assert valid["auto_contract_valid"] is True
    assert invalid["auto_contract_valid"] is False
    diff = invalid["checkpoint_results"][0]["diff"]
    assert "$250" in diff["missing_numbers"]
    assert diff["missing_entities"]
    assert diff["missing_constraints"]


def test_contract_extraction_excludes_structural_tool_catalogs() -> None:
    state = deepcopy(_checkpoint()["visible_state"])
    assert isinstance(state, dict)
    state["query"] = "Cancel order ORD-7512, but do not refund more than $50."
    state["context"] = [
        {
            "block_id": "official_tool_catalog",
            "source": "state_bench.domain.tool_schemas",
            "text": '{"name":"cancel_order","required":["order_id"],"maximum":999}',
        },
        {
            "block_id": "official_policy",
            "source": "state_bench.domain.agent_system_prompt",
            "text": "The agent must verify the customer before cancellation.",
        },
    ]
    contract = infer_visible_contract(state)
    serialized = " ".join(
        contract["required_entities"]
        + contract["required_numbers"]
        + contract["required_constraints"]
    )
    assert "order_id" not in serialized
    assert "999" not in serialized
    assert "ORD-7512" in contract["required_entities"]
    assert "$50" in contract["required_numbers"]
    assert any("must verify" in value for value in contract["required_constraints"])

    retained = visible_contract_retention(pre_state=state, post_state=state)
    assert retained["entity_retention"] == 1.0
    assert retained["numeric_retention"] == 1.0
    assert retained["constraint_retention"] == 1.0


def test_deterministic_stratified_sample_is_exact_and_reproducible() -> None:
    rows = [
        {"group_id": f"group-{index}", "benchmark_id": f"bench-{index % 3}"} for index in range(50)
    ]
    first = deterministic_stratified_sample(
        rows,
        count=17,
        seed=2027,
        strata=("benchmark_id",),
    )
    second = deterministic_stratified_sample(
        rows,
        count=17,
        seed=2027,
        strata=("benchmark_id",),
    )

    assert first == second
    assert len(first) == 17
    assert {row["benchmark_id"] for row in first} == {"bench-0", "bench-1", "bench-2"}


def test_proximal_repair_quality_requires_target_need_and_contract_retention() -> None:
    pre = {
        "query": "Move it?",
        "instruction": "Ask which account before acting.",
        "context": [],
        "messages": [],
        "tool_names": ["banking_api"],
        "metadata": {},
        "budget": {},
    }
    post = deepcopy(pre)
    post["metadata"] = {"clarification_request": "Which account should receive it?"}
    preserved = automatic_proximal_repair_quality(
        pre_state=pre,
        post_state=post,
        candidate_family="clarify",
        contract_retention={
            "instruction_exactly_preserved": True,
            "entity_retention": 1.0,
            "numeric_retention": 1.0,
            "constraint_retention": 1.0,
        },
        action_evidence=1.0,
    )
    destructive = automatic_proximal_repair_quality(
        pre_state=pre,
        post_state=post,
        candidate_family="clarify",
        contract_retention={
            "instruction_exactly_preserved": False,
            "entity_retention": 0.0,
            "numeric_retention": 1.0,
            "constraint_retention": 0.0,
        },
        action_evidence=1.0,
    )

    assert preserved["repair_quality"] > 0
    assert preserved["repair_success"] is True
    assert destructive["repair_quality"] < 0
    assert destructive["repair_success"] is False
    assert preserved["outcome_fields_accessed"] is False
