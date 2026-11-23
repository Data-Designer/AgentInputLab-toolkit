"""Pinned native CPU prefix/parser fixtures; no neural calls or training."""
import os
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.recipes import (
    SmartSearchNativeGRPOPrefixBuilder,
    SmartSearchTeacherCompletion,
    SmartSearchTeacherControls,
)

SOURCE = os.environ.get("AGENTINPUTLAB_SMARTSEARCH_TRAINING_CHECKOUT")
pytestmark = pytest.mark.skipif(not SOURCE, reason="mount pinned SmartSearch source for CPU prefix fixtures")


def prefix_inputs():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids, **kwargs):
            return ''.join(chr(int(i)) for i in ids if int(i))

    tok = Tokenizer()
    prompt = '<|im_start|>user\nFixture train<|im_end|>\n'
    before = '<think>Need evidence</think><search>fixture query</search>'
    retrieved = '<result>result: Fixture evidence</result>'
    after = r'<think>Done</think><answer>\boxed{gold}</answer>'
    response = before + retrieved + after
    ids = [0] * (1024 - len(prompt)) + tok.encode(prompt)
    mask = [1] * len(before) + [0] * len(retrieved) + [1] * len(after) + [0] * 3
    return SimpleNamespace(batch=dict(
        prompts=torch.tensor([ids]), input_ids=torch.tensor([ids + tok.encode(response) + [0] * 3]),
        responses=torch.tensor([tok.encode(response) + [0] * 3]), loss_mask=torch.tensor([mask]))), tok


def builder(tmp_path, provider, *, calls=8, tokens=16384):
    return SmartSearchNativeGRPOPrefixBuilder(provider, checkout=SOURCE,
        references={' Fixture train ': ['gold']}, model_fingerprint_sha256='a' * 64,
        controls=SmartSearchTeacherControls(max_model_calls=calls, max_total_output_tokens=tokens),
        output_root=tmp_path / 'prefix')


@pytest.mark.parametrize('replacement,expected_prefixes', [('<search>new query</search>', 1),
                                                         ('<search>missing close', 0)])
def test_native_refinement_preserves_parser_tensor_masks_and_no_stop(tmp_path, replacement, expected_prefixes):
    inputs, tok = prefix_inputs()
    calls = []

    def provider(prompt, **kwargs):
        calls.append((prompt, kwargs))
        text = '<answer>0</answer><explanation>fixture issue</explanation>' if len(calls) == 1 else replacement
        return SmartSearchTeacherCompletion(text, 100, 20, 'a' * 64)

    instance = builder(tmp_path, provider)
    ids, masks = instance(inputs, tok, ['gold'])
    assert len(ids) == len(masks) == expected_prefixes
    assert len(calls) == 2
    assert all(kw == dict(temperature=.6, max_new_tokens=2048, stop=(), seed=17) for _, kw in calls)
    if expected_prefixes:
        assert ids[0].shape == masks[0].shape == (1024,)
        assert '<search>new query</search>' in tok.decode(ids[0])
        assert '<result>' not in tok.decode(ids[0])
    usage = instance.assert_healthy()
    assert usage['output_tokens'] == 40 and usage['prefixes_produced'] == expected_prefixes


def test_native_correct_query_has_one_score_call_and_no_refinement(tmp_path):
    inputs, tok = prefix_inputs()
    calls = []

    def provider(prompt, **kwargs):
        calls.append(prompt)
        return SmartSearchTeacherCompletion('<answer>1</answer>', 100, 10, 'a' * 64)

    instance = builder(tmp_path, provider)
    assert instance(inputs, tok, ['gold']) == ([], [])
    assert len(calls) == 1 and instance.assert_healthy()['model_responses'] == 1


def test_native_helper_failure_keeps_unknown_reservation_and_cannot_repeat(tmp_path):
    inputs, tok = prefix_inputs()
    calls = []

    def provider(*args, **kwargs):
        calls.append(1)
        raise ConnectionError('explicit fixture transport failure')

    instance = builder(tmp_path, provider)
    with pytest.raises(ConnectionError):
        instance(inputs, tok, ['gold'])
    with pytest.raises(ExternalDependencyError, match='no automatic helper retry'):
        instance(inputs, tok, ['gold'])
    assert calls == [1] and instance.usage['unknown_output_token_reservations'] == 2048
    with pytest.raises(ExternalDependencyError):
        instance.assert_healthy()


def test_native_helper_cannot_shrink_2048_cap_to_fit_remaining_budget(tmp_path):
    inputs, tok = prefix_inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError('no model call fits this explicit quota')

    instance = builder(tmp_path, forbidden, tokens=2047)
    with pytest.raises(BudgetExceededError):
        instance(inputs, tok, ['gold'])
    assert instance.usage['model_attempts'] == 0


def test_native_prefix_rejects_unbound_reference_before_helper(tmp_path):
    inputs, tok = prefix_inputs()
    instance = builder(tmp_path, lambda *a, **kw: pytest.fail('reference leak'))
    with pytest.raises(ValidationError, match='bound TRAIN reference'):
        instance(inputs, tok, ['unbound'])
    assert instance.usage['model_attempts'] == 0
