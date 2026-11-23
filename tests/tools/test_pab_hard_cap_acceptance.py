"""Source-visible context-window fixtures, not real tokenizer/model acceptance."""
from types import SimpleNamespace

from run_pab_hard_cap_acceptance import bounded_state

from agentinputlab import ContextBlock
from agentinputlab.benchmarks.base import BenchmarkExample


class CharacterFixtureTokenizer:
    def encode(self,text,add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=list(map(ord,text)))

    def decode(self,tokens,skip_special_tokens):
        assert skip_special_tokens is True
        return ''.join(map(chr,tokens))


def test_short_visible_context_is_not_modified_or_gold_exposed():
    example=BenchmarkExample(example_id='fixture',query='Question',visible_context=(ContextBlock('Text'),),
        evaluator_gold={'answer':'NOT_VISIBLE'})
    original,state,window=bounded_state(example,CharacterFixtureTokenizer(),100)
    assert original is state and not window['middle_truncation']
    assert 'NOT_VISIBLE' not in repr(state)


def test_overlength_visible_context_keeps_declared_first_and_last_tokens():
    example=BenchmarkExample(example_id='fixture',query='Question',visible_context=(ContextBlock('0123456789'),))
    original,state,window=bounded_state(example,CharacterFixtureTokenizer(),6)
    assert original.state_hash!=state.state_hash and state.context[0].text=='012789'
    assert state.query==original.query and state.budget==original.budget
    assert window['middle_truncation'] and window['context_tokens_before']==10
