"""Validate fixed cohort/policy selection before contacting any model service."""
import pytest
from run_self_ask_component_acceptance import selected_final_stop_models, selected_qa_cells
from run_static_acceptance import QA_CELLS


def test_legacy_operator_defaults_preserve_all_qa_and_no_new_stop_policy():
    assert selected_qa_cells({}) == tuple(QA_CELLS)
    assert selected_final_stop_models({'models': [{'id': 'q27'}]}) == frozenset()


def test_explicit_subset_and_stop_model_binding():
    cells = selected_qa_cells({'qa_cells': ['musique.answerable', '2wiki.multihop']})
    assert {v[0] for v in cells} == {'musique.answerable', '2wiki.multihop'}
    assert selected_final_stop_models({'models': [{'id': 'q27'}, {'id': 'q4'}],
        'final_content_stop_model_ids': ['q27']}) == {'q27'}


@pytest.mark.parametrize('selection', [[], ['unknown'], ['musique.answerable'] * 2, None, True])
def test_invalid_qa_selection_rejected(selection):
    with pytest.raises(ValueError):
        selected_qa_cells({'qa_cells': selection})


@pytest.mark.parametrize('selection', [['unknown'], ['q27', 'q27'], None, True])
def test_implicit_or_unknown_stop_policy_rejected(selection):
    with pytest.raises(ValueError):
        selected_final_stop_models({'models': [{'id': 'q27'}], 'final_content_stop_model_ids': selection})
