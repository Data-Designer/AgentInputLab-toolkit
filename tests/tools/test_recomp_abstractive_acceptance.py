from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from run_recomp_abstractive_acceptance import select_cases


def test_case_selection_is_fixed_source_order_and_does_not_inspect_outcomes():
    values = [SimpleNamespace(example_id=f"q-{index:03d}") for index in reversed(range(70))]
    adapter = MagicMock()
    adapter.examples.return_value = values
    selected = select_cases(adapter, "dev")
    assert [row.example_id for row in selected] == ["q-064", "q-065"]
    adapter.examples.assert_called_once_with("dev")


def test_missing_source_cases_do_not_silently_substitute_different_cases():
    adapter = MagicMock()
    adapter.examples.return_value = [SimpleNamespace(example_id=str(i)) for i in range(65)]
    with pytest.raises(ValueError, match="frozen source cases"):
        select_cases(adapter, "dev")
