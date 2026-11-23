from __future__ import annotations

from prepare_adaptive_boundary_cases import select_boundaries


def record(case, a, b, c):
    return {
        "case_id": case,
        "status": "router_response",
        "response": {"logits": {"A": a, "B": b, "C": c}},
    }


def test_boundary_selection_first_source_case_not_best_reader_quality():
    records = [
        record("b", 0, 2, 1),
        record("first-c", 0, 1, 2),
        record("first-a", 2, 1, 0),
        record("later-a", 100, 0, 0),
    ]
    selected = select_boundaries(records)
    assert [r["case_id"] for r in selected] == ["first-a", "first-c"]
    assert [r["route"] for r in selected] == ["A", "C"]


def test_missing_routes_are_not_fabricated_and_failed_calls_not_retried():
    assert (
        select_boundaries(
            [record("only-b", 0, 2, 1), {"case_id": "bad", "status": "router_failed"}]
        )
        == []
    )
    assert [r["route"] for r in select_boundaries([record("tie-a", 1, 1, 1)])] == ["A"]
