"""Shape-only regression tests; no pretrained encoder, model or benchmark draw."""
import ast
import unittest
from types import SimpleNamespace

from agentinputlab.algorithms.acquire.qrag.native_axis import _axis_stable_forward
from agentinputlab.contracts import ExternalDependencyError

try:
    import torch
except ImportError:
    torch = None

SOURCE = '''
class NativePolicy:
    def forward(self, s, a_embeds, alpha, return_arg_max=False):
        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed.unsqueeze(1)
        logits = (s_embed * a_embeds).sum(-1).squeeze(-1)
        logits[s.available_mask == False] = logits.min() - 1
        top_k_actions = min(logits.size(1), self.top_k_actions)
        top_ids = torch.topk(logits, top_k_actions, dim=1).indices
        top_mask = torch.zeros_like(logits > 0).scatter_(1, top_ids, True)
        if return_arg_max:
            return torch.argmax(logits, -1), logits
        probs = ((logits - logits.max(-1, keepdim=True).values) / alpha).softmax(-1)
        probs[(s.available_mask & top_mask) == False] = 0
        probs = probs / probs.sum(-1, keepdim=True)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample(), logits
'''


class AxisSourceTests(unittest.TestCase):
    def test_rejects_missing_or_ambiguous_replacement(self):
        for source in (SOURCE.replace('.squeeze(-1)', ''), SOURCE + SOURCE.replace('NativePolicy', 'Second')):
            with self.subTest(source=source), self.assertRaises(ExternalDependencyError):
                _axis_stable_forward(source, {}, '<fixture>')

    def test_fixed_code_only_removes_the_final_squeeze(self):
        fixed = _axis_stable_forward(SOURCE, {}, '<fixture>')
        expected = SOURCE.replace('.sum(-1).squeeze(-1)', '.sum(-1)')
        scope = {}
        exec(compile(ast.parse(expected), '<fixture>', 'exec'), scope)
        self.assertEqual(fixed.__code__.co_code, scope['NativePolicy'].forward.__code__.co_code)


@unittest.skipIf(torch is None, 'optional torch is required for tensor arithmetic oracle')
class AxisTensorTests(unittest.TestCase):
    def test_matches_multicandidate_original_and_repairs_singleton(self):
        namespace = {'torch': torch}
        exec(SOURCE, namespace)
        fixed = _axis_stable_forward(SOURCE, namespace, '<fixture>')
        for batch in (1, 2):
            for candidates in (1, 2, 7):
                for argmax in (False, True):
                    with self.subTest(batch=batch, candidates=candidates, argmax=argmax):
                        vector = torch.arange(1, batch * 4 + 1, dtype=torch.float32).reshape(batch, 4)
                        actions = torch.arange(1, batch * candidates * 4 + 1, dtype=torch.float32).reshape(batch, candidates, 4)
                        state = SimpleNamespace(input_ids=None, attention_mask=None,
                            available_mask=torch.ones((batch, candidates), dtype=torch.bool))
                        if candidates > 1:
                            state.available_mask[:, 0] = False
                        obj = namespace['NativePolicy']()
                        obj.top_k_actions = 2
                        obj.state_embed = lambda vector=vector, **kwargs: vector
                        torch.manual_seed(17)
                        actual = fixed(obj, state, actions, 0.5, argmax)
                        self.assertEqual(tuple(actual[1].shape), (batch, candidates))
                        if candidates == 1:
                            self.assertTrue(torch.equal(actual[0], torch.zeros(batch, dtype=torch.int64)))
                            with self.assertRaises(IndexError):
                                obj.forward(state, actions, 0.5, argmax)
                        else:
                            torch.manual_seed(17)
                            expected = obj.forward(state, actions, 0.5, argmax)
                            self.assertTrue(torch.equal(actual[0], expected[0]))
                            self.assertTrue(torch.equal(actual[1], expected[1]))


if __name__ == '__main__':
    unittest.main()
