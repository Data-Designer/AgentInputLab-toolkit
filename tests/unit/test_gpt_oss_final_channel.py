from __future__ import annotations

import copy
import unittest

from agentinputlab.backends.llm.base import GenerationRequest
from agentinputlab.backends.llm.gpt_oss_final_channel import (
    GPTOSSFinalChannelStopBackend,
    POLICY_ID,
)
from agentinputlab.backends.llm.openai_compatible import OpenAICompatibleBackend
from agentinputlab.contracts.errors import ValidationError


class FinalChannelStopTests(unittest.TestCase):
    def backend(self, *, content, reasoning="analysis\n** <END_OF_PLAN>", reason="stop", tokens=157):
        self.calls = []
        self.native = {
            "model": "gpt-oss-20b",
            "choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 3279, "completion_tokens": tokens},
        }
        def transport(url, headers, payload, timeout):
            self.calls.append(copy.deepcopy(payload))
            return self.native
        inner = OpenAICompatibleBackend(
            model_id="gpt-oss-20b", base_url="http://127.0.0.1:8560/v1",
            deployment_identity={"revision": "fixture", "server": "vllm"},
            api_key="", transport=transport,
        )
        return GPTOSSFinalChannelStopBackend(inner)

    def test_stops_apply_to_final_only_and_preserve_request_controls(self):
        backend = self.backend(content="one valid query\nsecond line")
        request = GenerationRequest(prompt="fixture", system="fixture system", stop=("\n",),
                                    max_tokens=160, seed=17, temperature=0.0, metadata={"stage": "decomposer"})
        response = backend.generate(request)
        self.assertEqual(response.text, "one valid query")
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("stop", self.calls[0])
        self.assertEqual(self.calls[0]["max_tokens"], 160)
        self.assertEqual(self.calls[0]["seed"], 17)
        self.assertEqual(response.usage.output_tokens, 157)
        self.assertEqual(request.stop, ("\n",))
        self.assertEqual(response.raw["choices"][0]["message"]["content"], "one valid query\nsecond line")

    def test_no_reasoning_to_final_fallback_on_token_exhaustion(self):
        backend = self.backend(content=None, reasoning="apparently useful but unfinished plan", reason="length", tokens=512)
        response = backend.generate(GenerationRequest(prompt="fixture", max_tokens=512, stop=("<END_OF_PLAN>",)))
        self.assertEqual(response.text, "")
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.usage.output_tokens, 512)
        self.assertEqual(len(self.calls), 1)

    def test_logical_stop_does_not_discard_server_usage(self):
        backend = self.backend(content="query**unused tail", reason="length", tokens=100)
        response = backend.generate(GenerationRequest(prompt="fixture", max_tokens=100, stop=("**",)))
        self.assertEqual(response.text, "query")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.output_tokens, 100)
        self.assertEqual(response.raw["agentinputlab_final_channel_stop"]["provider_finish_reason"], "length")

    def test_first_final_stop_wins(self):
        backend = self.backend(content="answer\nrest**tail")
        response = backend.generate(GenerationRequest(prompt="fixture", stop=("**", "\n")))
        self.assertEqual(response.text, "answer")

    def test_no_stop_is_unchanged(self):
        backend = self.backend(content="answer\n**keep")
        response = backend.generate(GenerationRequest(prompt="fixture"))
        self.assertEqual(response.text, "answer\n**keep")
        self.assertNotIn("agentinputlab_final_channel_stop", response.raw)
        self.assertEqual(len(self.calls), 1)

    def test_opt_in_fingerprint_differs_from_legacy(self):
        backend = self.backend(content="answer")
        self.assertEqual(backend.fingerprint()["policy_id"], POLICY_ID)
        self.assertNotEqual(backend.fingerprint(), backend.inner.fingerprint())

    def test_legacy_backend_retains_original_stop_semantics(self):
        backend = self.backend(content="answer")
        backend.inner.generate(GenerationRequest(prompt="fixture", stop=("\n",)))
        self.assertEqual(self.calls[0]["stop"], ["\n"])

    def test_reject_other_models_and_empty_stops(self):
        backend = self.backend(content="answer")
        with self.assertRaises(ValidationError):
            backend.generate(GenerationRequest(prompt="fixture", stop=("",)))
        self.assertEqual(self.calls, [])
        backend.inner.model_id = "qwen3-4b"
        with self.assertRaises(ValidationError):
            GPTOSSFinalChannelStopBackend(backend.inner)


if __name__ == "__main__":
    unittest.main()
