from types import SimpleNamespace

import pytest
from run_recomp_sentence_acceptance import RELEASE, SOURCE_OFFSET, select_cases


def test_sentence_profile_selection_is_fixed_before_results():
    rows = [SimpleNamespace(example_id=f"case-{i:03}") for i in range(102)]
    adapter = SimpleNamespace(examples=lambda split: list(reversed(rows)))
    assert SOURCE_OFFSET == 98
    assert RELEASE == "curate.recomp.extractive"
    assert [row.example_id for row in select_cases(adapter, "fixed")] == ["case-098", "case-099"]


def test_missing_fixed_cases_does_not_choose_an_easier_fallback():
    adapter = SimpleNamespace(examples=lambda split: [SimpleNamespace(example_id="short")])
    with pytest.raises(ValueError, match="98:100"):
        select_cases(adapter, "fixed")
