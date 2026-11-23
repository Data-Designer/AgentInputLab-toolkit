"""CPU tensor fixtures executing the pinned native GRPO batch method.

No vLLM engine, model output, optimizer, or real acceptance is simulated here.
The fixture generator exposes batch alignment and native final assembly only.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinputlab.recipes.smartsearch_grpo_rollout import install_native_rollout_batch_contracts

SOURCE = os.environ.get("AGENTINPUTLAB_SMARTSEARCH_TRAINING_CHECKOUT")
pytestmark = pytest.mark.skipif(not SOURCE, reason="mount pinned SmartSearch source for CPU batch fixtures")


@pytest.mark.parametrize("counts", [(0, 0), (0, 1), (2, 1)])
def test_pinned_native_method_keeps_tensor_and_metadata_alignment(counts):
    torch = pytest.importorskip("torch")
    path = Path(SOURCE).resolve() / "src/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    original_source = path.read_bytes()
    tree = ast.parse(original_source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMRolloutWithTool")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "generate_sequences")

    class Data:
        def __init__(self, batch=None, non_tensor_batch=None):
            self.batch = batch or {}
            self.non_tensor_batch = non_tensor_batch or {}
            self.meta_info = {}

        def select_idxs(self, indices):
            result = Data({k: v[indices] for k, v in self.batch.items()},
                          {k: [v[i] for i in indices] for k, v in self.non_tensor_batch.items()})
            result.meta_info = self.meta_info
            return result

    def tensor_dict(fields, *, batch_size):
        assert all(v.shape[0] == batch_size for v in fields.values())
        return fields

    def forbidden_default(*args):
        raise AssertionError("hardcoded native network helper must never be reached")

    namespace = dict(torch=torch, DataProto=Data, TensorDict=tensor_dict,
                     generate_new_prompts=forbidden_default)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    native_class = type("vLLMRolloutWithTool", (), {"generate_sequences": namespace["generate_sequences"]})
    calls = []

    def prefix_builder(output, tokenizer, ground_truth):
        # Expose the native original-row ground truth, not a model completion.
        assert ground_truth in ("A", "B")
        if int(output.batch['responses'][0, -1]) != 0:
            return [], []
        count = counts[ord(ground_truth) - ord('A')]
        return [torch.tensor([7, 8, 9])] * count, [torch.ones(3, dtype=torch.int64)] * count

    record = install_native_rollout_batch_contracts(checkout=SOURCE, rollout_class=native_class,
        prefix_builder=prefix_builder, prefix_builder_binding={"fixture_only": True, "real_model_calls": 0})
    runner = native_class()
    runner.sampling_params = SimpleNamespace(n=2)
    runner.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "explicit tensor fixture")

    def generate(prompts, **kwargs):
        labels = prompts.non_tensor_batch['ground_truth']
        assert len(labels) == prompts.batch['input_ids'].shape[0]
        index = len(calls)
        calls.append((labels, 'prefix_input_ids' in prompts.batch))
        values = torch.tensor([[ord(v), 1, index] for v in labels])
        return Data(dict(prompts=values, responses=values, input_ids=torch.cat((values, values), dim=1),
                         attention_mask=torch.ones(len(labels), 6, dtype=torch.int64),
                         loss_mask=torch.ones(len(labels), 3, dtype=torch.int64),
                         position_ids=torch.arange(6).repeat(len(labels), 1)))

    runner.generate_sequence = generate
    prompts = Data({k: torch.tensor([[1, 2, 3], [4, 5, 6]]) for k in
                    ('input_ids', 'attention_mask', 'position_ids')}, {'ground_truth': ['A', 'B']})
    output = runner.generate_sequences(prompts)
    assert output.batch['responses'].shape == (4, 3)
    assert output.batch['responses'][:, 0].tolist() == [ord('A'), ord('A'), ord('B'), ord('B')]
    prefix_calls = [labels for labels, is_prefix in calls if is_prefix]
    assert prefix_calls == ([] if not any(counts) else [['A'] * counts[0] + ['B'] * counts[1]])
    if counts == (2, 1):
        assert sum(len(labels) for labels, _ in calls) == 5  # one native-discarded output still generated
    assert record['full_grpo_training_certified'] is False
    assert record['native_source_files_modified'] is False
    assert path.read_bytes() == original_source
