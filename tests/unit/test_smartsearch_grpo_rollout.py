"""Non-neural batch fixtures; no generated text or trained policy is fabricated."""
import ast
from types import SimpleNamespace

import pytest

from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.recipes.smartsearch_grpo_rollout import (
    _batch_contract_ast,
    _compile_batch_contract,
    install_native_rollout_batch_contracts,
)

# An explicit structural fixture, not a copied native implementation or
# acceptance model. Mirrors only the buggy assembly inventory's variable names.
FIXTURE = '''
def generate_sequences(self, prompts, **kwargs):
    if not prompts.meta_info.get('do_sample', True):
        return self.generate_sequence(prompts, **kwargs)
    ori_input_ids = prompts.batch['input_ids']
    ori_attention_mask = prompts.batch['attention_mask']
    ori_position_ids = prompts.batch['position_ids']
    active_indices = list(range(ori_input_ids.size(0)))
    results = [[] for _ in active_indices]
    while active_indices:
        outputs = self.generate_sequence(prompts.select_idxs(active_indices), **kwargs)
        prefix_input_ids_list, prefix_loss_mask_list = [], []
        input_ids_list, attention_mask_list, position_ids_list, index = [], [], [], []
        for i, idx in enumerate(active_indices):
            results[idx].append(outputs.select_idxs([i]))
            prefix_ids, masks = generate_new_prompts(outputs.select_idxs([i]), None, idx)
            for prefix_input_ids, prefix_loss_mask in zip(prefix_ids, masks):
                prefix_input_ids_list.append(prefix_input_ids)
                prefix_loss_mask_list.append(prefix_loss_mask)
                input_ids_list.append(ori_input_ids[idx])
                attention_mask_list.append(ori_attention_mask[idx])
                position_ids_list.append(ori_position_ids[idx])
                index.append(idx)
            prefix_input_ids = torch.stack(prefix_input_ids_list, dim=0)
            prefix_loss_mask = torch.stack(prefix_loss_mask_list, dim=0)
            input_ids = torch.stack(input_ids_list, dim=0)
            attention_mask = torch.stack(attention_mask_list, dim=0)
            position_ids = torch.stack(position_ids_list, dim=0)
            batch = TensorDict(dict(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, prefix_input_ids=prefix_input_ids, prefix_loss_mask=prefix_loss_mask), batch_size=input_ids.size(0))
            new_prompts = DataProto(batch=batch)
            new_prompts.non_tensor_batch = prompts.non_tensor_batch
            new_prompts.meta_info = prompts.meta_info
            new_outputs = self.generate_sequence(new_prompts, **kwargs)
            for i, idx in enumerate(index):
                results[idx].append(new_outputs.select_idxs([i]))
        for i, result in enumerate(results):
            if len(result) >= self.sampling_params.n:
                results[i] = result[:self.sampling_params.n]
                if i in active_indices:
                    active_indices.remove(i)
    return results
'''


class Rows(list):
    def size(self, dim):
        assert dim == 0
        return len(self)


class Data:
    def __init__(self, batch=None, labels=(), items=(), meta=None):
        self.batch = batch or {}
        self.non_tensor_batch = {"id": list(labels)}
        self.items = list(items)
        self.meta_info = dict(meta or {})

    def select_idxs(self, indices):
        return Data({name: Rows(value[i] for i in indices) for name, value in self.batch.items()},
                    [self.non_tensor_batch['id'][i] for i in indices],
                    [self.items[i] for i in indices] if self.items else (), self.meta_info)


def exercise(prefix_counts, *, sample=True):
    method = ast.parse(FIXTURE).body[0]
    original = ast.dump(method)
    stacks, generated = [], []

    def stack(values, dim):
        assert values, 'empty native stack must never be invoked'
        stacks.append(list(values))
        return Rows(values)

    def prefix_builder(output, tokenizer, original_index):
        # Refine the first generated batch only in this explicit fixture.
        count = prefix_counts[original_index] if output.items[0].endswith(':0') else 0
        return list(range(count)), [1] * count

    namespace = dict(torch=SimpleNamespace(stack=stack), DataProto=Data,
                     TensorDict=lambda fields, **kw: fields)
    function, _ = _compile_batch_contract(method, namespace, prefix_builder, 'explicit-fixture.py')
    assert ast.dump(method) == original  # no in-place source AST edits
    owner = SimpleNamespace(sampling_params=SimpleNamespace(n=2))

    def generate(prompts, **kwargs):
        labels = prompts.non_tensor_batch['id']
        assert len(labels) == len(prompts.batch['input_ids']), 'metadata/physical rows must match'
        prefix = 'prefix_input_ids' in prompts.batch
        number = len(generated)
        generated.append(dict(labels=list(labels), is_prefix=prefix))
        return Data(labels=labels, items=[label + ':' + str(number) for label in labels])

    owner.generate_sequence = generate
    prompts = Data({name: Rows(['A', 'B']) for name in ('input_ids', 'attention_mask', 'position_ids')},
                   ['A', 'B'], meta=dict(do_sample=sample))
    return function(owner, prompts), generated, stacks


def test_no_refined_prefixes_never_stack_empty_and_fill_original_group_size():
    result, calls, stacks = exercise([0, 0])
    assert [len(group) for group in result] == [2, 2]
    assert len(calls) == 2 and not any(c['is_prefix'] for c in calls) and not stacks


def test_mixed_prefix_rows_select_original_metadata_and_generate_once():
    result, calls, _ = exercise([0, 1])
    assert [len(group) for group in result] == [2, 2]
    assert calls == [dict(labels=['A', 'B'], is_prefix=False), dict(labels=['B'], is_prefix=True),
                     dict(labels=['A'], is_prefix=False)]


def test_multiple_prefixes_do_not_repeat_the_growing_batch_or_hide_discarded_cost():
    result, calls, _ = exercise([2, 1])
    assert [len(group) for group in result] == [2, 2]
    assert calls == [dict(labels=['A', 'B'], is_prefix=False), dict(labels=['A', 'A', 'B'], is_prefix=True)]
    # Five fixture outputs generated, four retained by unchanged native cut.
    assert sum(len(c['labels']) for c in calls) == 5
    assert sum(len(group) for group in result) == 4


def test_validation_non_sampling_branch_remains_one_native_call():
    result, calls, stacks = exercise([2, 1], sample=False)
    assert isinstance(result, Data) and len(calls) == 1 and not stacks


@pytest.mark.parametrize('changed', [
    FIXTURE.replace('new_prompts.non_tensor_batch = prompts.non_tensor_batch', 'new_prompts.non_tensor_batch = different'),
    FIXTURE.replace('prefix_loss_mask = torch.stack(prefix_loss_mask_list, dim=0)', 'pass'),
    FIXTURE.replace('while active_indices:', 'while different:'),
])
def test_unexpected_source_layout_refused(changed):
    with pytest.raises(ExternalDependencyError):
        _batch_contract_ast(ast.parse(changed).body[0])


def test_install_requires_explicit_native_class_and_provider_without_any_import():
    with pytest.raises(ValidationError):
        install_native_rollout_batch_contracts(checkout='absent', rollout_class=object,
            prefix_builder=None, prefix_builder_binding={})
    original = type('vLLMRolloutWithTool', (), {'_agentinputlab_batch_contract_installed': True})
    with pytest.raises(ExternalDependencyError, match='fresh native worker'):
        install_native_rollout_batch_contracts(checkout='absent', rollout_class=original,
            prefix_builder=lambda *a: None, prefix_builder_binding={'fixture_only': True})
