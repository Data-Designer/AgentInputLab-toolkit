from __future__ import annotations

import http.client
import io
import json
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest import mock

from agentinputlab.backends.llm import (
    CallableBackend,
    GenerationRequest,
    ModelCapability,
    OpenAICompatibleBackend,
    ReplayBackend,
    TransformersBackend,
    Usage,
    openai_compatible,
)
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import ExecutionBudget
from agentinputlab.contracts.errors import (
    BudgetExceededError,
    CapabilityError,
    ValidationError,
)
from agentinputlab.observability import BudgetLedger, ObservedLLMBackend, RunMonitor


class FakeTensor:
    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        if self.values and isinstance(self.values[0], list):
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    def __getitem__(self, index):
        value = self.values[index]
        return FakeTensor(value) if isinstance(value, list) else value

    def sum(self):
        if self.values and isinstance(self.values[0], list):
            return sum(sum(row) for row in self.values)
        return sum(self.values)


class FakeTokenizer:
    def __init__(
        self,
        *,
        include_attention_mask: bool = True,
        input_ids=None,
        attention_mask=None,
    ):
        self.include_attention_mask = include_attention_mask
        self.input_ids = [[101, 11, 12, 0]] if input_ids is None else input_ids
        self.attention_mask = [[1, 1, 1, 0]] if attention_mask is None else attention_mask
        self.decoded = None

    def __call__(self, prompt, *, return_tensors):
        del prompt, return_tensors
        inputs = {"input_ids": FakeTensor(self.input_ids)}
        if self.include_attention_mask:
            inputs["attention_mask"] = FakeTensor(self.attention_mask)
        return inputs

    def decode(self, tokens, *, skip_special_tokens):
        self.decoded = (tokens.values, skip_special_tokens)
        return "generated"


class FakeTransformersModel:
    def __init__(self, *, return_dict: bool = True, sequences=None):
        self.return_dict = return_dict
        self.sequences = [[101, 11, 12, 0, 21, 22]] if sequences is None else sequences
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        sequences = FakeTensor(self.sequences)
        return SimpleNamespace(sequences=sequences) if self.return_dict else sequences


class BackendTests(unittest.TestCase):
    def _transformers_backend(
        self,
        *,
        include_attention_mask: bool = True,
        return_dict: bool = True,
        input_ids=None,
        attention_mask=None,
        sequences=None,
        generation_kwargs=None,
    ) -> TransformersBackend:
        return TransformersBackend(
            model_id="fixture",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            tokenizer=FakeTokenizer(
                include_attention_mask=include_attention_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ),
            model=FakeTransformersModel(return_dict=return_dict, sequences=sequences),
            generation_kwargs={} if generation_kwargs is None else generation_kwargs,
        )

    def test_callable_backend_records_normalized_request(self) -> None:
        backend = CallableBackend(
            lambda request: request.prompt.upper(),
            model_id="fake",
            revision="fixture-v1",
            callable_id="tests.uppercase.v1",
        )
        response = backend.generate(GenerationRequest(prompt="hello", seed=1))
        self.assertEqual(response.text, "HELLO")
        self.assertEqual(len(backend.calls), 1)
        self.assertNotIn("api_key", backend.fingerprint())

    def test_replay_backend_fails_when_exhausted(self) -> None:
        backend = ReplayBackend(["one"])
        self.assertEqual(backend.generate(GenerationRequest(prompt="q")).text, "one")
        with self.assertRaises(RuntimeError):
            backend.generate(GenerationRequest(prompt="q2"))

    def test_openai_compatible_payload_and_response(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return {
                "model": "served",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }

        backend = OpenAICompatibleBackend(
            model_id="model",
            base_url="http://127.0.0.1:8000/v1",
            deployment_identity={"revision": "weights-abc"},
            api_key="secret",
            extra_body={"top_p": 0.9},
            transport=transport,
        )
        response = backend.generate(GenerationRequest(prompt="question", seed=7))
        self.assertEqual(response.text, "answer")
        self.assertEqual(captured["payload"]["model"], "model")
        self.assertEqual(captured["payload"]["seed"], 7)
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertNotIn("secret", str(backend.fingerprint()))

    def test_openai_backend_materializes_nested_frozen_payloads(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            # This is the operation performed by the default HTTP transport.
            json.dumps(payload)
            return {
                "model": "served",
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }

        backend = OpenAICompatibleBackend(
            model_id="model",
            base_url="http://127.0.0.1:8000/v1",
            deployment_identity={"revision": "weights-abc"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            capabilities=frozenset({ModelCapability.CHAT, ModelCapability.JSON_SCHEMA}),
            transport=transport,
        )
        response = backend.generate(
            GenerationRequest(
                prompt="question",
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                },
            )
        )

        self.assertEqual(response.text, "OK")
        self.assertEqual(
            captured["payload"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(
            captured["payload"]["response_format"]["json_schema"]["schema"],
            {"type": "object"},
        )

    def test_openai_backend_repr_hides_direct_credentials(self) -> None:
        secret = "opaqueValue987654321"
        backend = OpenAICompatibleBackend(
            model_id="model",
            base_url="https://example.test/v1",
            deployment_identity={"revision": "weights-abc"},
            api_key=secret,
            extra_headers={"X-Route": secret},
            extra_body={"route": secret},
        )
        self.assertNotIn(secret, repr(backend))

    def test_openai_backend_rejects_sensitive_extra_headers(self) -> None:
        with self.assertRaisesRegex(ValidationError, "sensitive extra header"):
            OpenAICompatibleBackend(
                model_id="model",
                base_url="https://example.test/v1",
                deployment_identity={"revision": "weights-abc"},
                extra_headers={"aUtHoRiZaTiOn": "opaqueValue987654321"},
            )

    def test_openai_response_rejects_exact_configured_credential(self) -> None:
        secret = "opaqueValue987654321"

        def transport(url, headers, payload, timeout):
            del url, headers, payload, timeout
            return {
                "model": "served",
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": f"reflected-{secret}",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }

        backend = OpenAICompatibleBackend(
            model_id="model",
            base_url="https://example.test/v1",
            deployment_identity={"revision": "weights-abc"},
            api_key=secret,
            transport=transport,
        )
        with self.assertRaisesRegex(RuntimeError, "configured credential") as caught:
            backend.generate(GenerationRequest(prompt="question"))
        self.assertNotIn(secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_openai_response_schema_error_does_not_echo_remote_value(self) -> None:
        remote_value = "opaqueFieldValue987654321"

        def transport(url, headers, payload, timeout):
            del url, headers, payload, timeout
            return {
                "model": "served",
                "choices": [{"message": {"content": "answer"}}],
                "usage": {
                    "prompt_tokens": remote_value,
                    "completion_tokens": 1,
                },
            }

        backend = OpenAICompatibleBackend(
            model_id="model",
            base_url="https://example.test/v1",
            deployment_identity={"revision": "weights-abc"},
            transport=transport,
        )
        with self.assertRaisesRegex(RuntimeError, "schema is invalid") as caught:
            backend.generate(GenerationRequest(prompt="question"))
        self.assertNotIn(remote_value, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_openai_custom_transport_error_does_not_echo_credentials(self) -> None:
        secret = "opaqueValue987654321"

        def transport(url, headers, payload, timeout):
            del url, headers, payload, timeout
            raise RuntimeError(f"provider reflected {secret}")

        backend = OpenAICompatibleBackend(
            model_id="model",
            base_url="https://example.test/v1",
            deployment_identity={"revision": "weights-abc"},
            api_key=secret,
            transport=transport,
        )
        with self.assertRaisesRegex(RuntimeError, "transport failed") as caught:
            backend.generate(GenerationRequest(prompt="question"))
        self.assertNotIn(secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_http_500_never_hides_a_second_transport_attempt(self) -> None:
        routes = (
            ("ircot", "acquire.ircot.common_support", "", {}),
            ("adaptive_rag", "acquire.adaptive_rag.common_support", "", {"step_index": 0}),
            ("udcs", "clarify.udcs.repo_native_prompt_route", "UDCS system contract", {}),
            ("hyde", "acquire.hyde", "", {}),
        )
        for method, release, system, metadata in routes:
            for failure in ("sanitized_http", "custom_http", "timeout"):
                with self.subTest(method=method, failure=failure):
                    calls = []

                    def transport(url, headers, payload, timeout, *, calls=calls, failure=failure):
                        del headers, timeout
                        calls.append((url, payload))
                        if failure == "sanitized_http":
                            raise openai_compatible._OpenAICompatibleHTTPStatusError(500)
                        if failure == "custom_http":
                            error = urllib.error.HTTPError(url, 500, "private body", None, None)
                            error.close()
                            raise error
                        raise TimeoutError("private provider diagnostic")

                    backend = OpenAICompatibleBackend(
                        model_id="gpt-oss-20b",
                        base_url="http://127.0.0.1:8202/v1",
                        deployment_identity={"revision": "weights-abc", "server": "vllm"},
                        transport=transport,
                    )
                    with self.assertRaisesRegex(RuntimeError, "transport failed") as caught:
                        backend.generate(GenerationRequest(
                            system=system,
                            prompt="question",
                            seed=17,
                            max_tokens=512,
                            metadata={"method_id": method, "release_id": release, **metadata},
                        ))
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0][0], "http://127.0.0.1:8202/v1/chat/completions")
                    self.assertEqual(calls[0][1]["seed"], 17)
                    self.assertEqual(calls[0][1]["max_tokens"], 512)
                    self.assertIsNone(caught.exception.__cause__)
                    self.assertIsNone(caught.exception.__context__)
                    self.assertNotIn("private", str(caught.exception))
                    self.assertEqual(backend.fingerprint()["transport_attempt_policy"],
                                     "single_transport_no_implicit_retry.v1")

    def test_default_transport_http_500_does_not_request_a_completion(self) -> None:
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "https://example.test/v1/chat/completions", 500,
            "private diagnostic", None, io.BytesIO(b"private body"),
        )
        backend = OpenAICompatibleBackend(
            model_id="gpt-oss-20b",
            base_url="https://example.test/v1",
            deployment_identity={"revision": "weights-abc", "server": "vllm"},
        )
        with (
            mock.patch.object(openai_compatible.urllib.request, "build_opener", return_value=opener),
            self.assertRaisesRegex(RuntimeError, "transport failed"),
        ):
            backend.generate(GenerationRequest(
                prompt="question",
                metadata={"method_id": "ircot", "release_id": "acquire.ircot.common_support"},
            ))
        opener.open.assert_called_once()
        self.assertEqual(opener.open.call_args.args[0].full_url,
                         "https://example.test/v1/chat/completions")

    def test_loopback_http_500_preserves_one_actual_post(self) -> None:
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, body))
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"untrusted provider body")

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = OpenAICompatibleBackend(
                model_id="gpt-oss-20b",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                deployment_identity={"revision": "fixture", "server": "vllm"},
                timeout_seconds=2,
            )
            with (
                mock.patch.object(openai_compatible.urllib.request, "getproxies", return_value={}),
                self.assertRaisesRegex(RuntimeError, r"transport failed \(HTTP 500\)") as caught,
            ):
                backend.generate(GenerationRequest(
                    system="system contract", prompt="question", seed=17, max_tokens=512,
                    stop=("logical stop",),
                    metadata={"method_id": "udcs", "release_id": "clarify.udcs.repo_native_prompt_route"},
                ))
            self.assertIsNone(caught.exception.__context__)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][0], "/v1/chat/completions")
            self.assertEqual(received[0][1]["seed"], 17)
            self.assertEqual(received[0][1]["max_tokens"], 512)
            self.assertEqual(received[0][1]["stop"], ["logical stop"])
            self.assertEqual(received[0][1]["messages"][0]["role"], "system")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_openai_default_transport_does_not_echo_remote_error_content(self) -> None:
        secret = "opaqueValue987654321"
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            url="https://example.test/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(f"reflected Authorization: Bearer {secret}".encode()),
        )
        with (
            mock.patch.object(
                openai_compatible.urllib.request,
                "build_opener",
                return_value=opener,
            ),
            self.assertRaisesRegex(RuntimeError, r"HTTP 401") as caught,
        ):
            openai_compatible._default_transport(
                "https://example.test/v1/chat/completions",
                {"Authorization": f"Bearer {secret}"},
                {"model": "fixture"},
                1.0,
            )
        self.assertNotIn(secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_openai_default_transport_does_not_echo_url_error_reason(self) -> None:
        secret = "opaqueValue987654321"
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.URLError(f"https://user:{secret}@example.test")
        with (
            mock.patch.object(
                openai_compatible.urllib.request,
                "build_opener",
                return_value=opener,
            ),
            self.assertRaisesRegex(RuntimeError, "endpoint is unavailable") as caught,
        ):
            openai_compatible._default_transport(
                "https://example.test/v1/chat/completions",
                {"Authorization": f"Bearer {secret}"},
                {"model": "fixture"},
                1.0,
            )
        self.assertNotIn(secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_openai_default_transport_does_not_echo_invalid_status_line(self) -> None:
        secret = "opaqueValue987654321"
        opener = mock.Mock()
        opener.open.side_effect = http.client.BadStatusLine(secret)
        with (
            mock.patch.object(
                openai_compatible.urllib.request,
                "build_opener",
                return_value=opener,
            ),
            self.assertRaisesRegex(RuntimeError, "invalid HTTP response") as caught,
        ):
            openai_compatible._default_transport(
                "https://example.test/v1/chat/completions",
                {"Authorization": f"Bearer {secret}"},
                {"model": "fixture"},
                1.0,
            )
        self.assertNotIn(secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_openai_compatible_requests_logprobs_only_with_capability(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return {
                "model": "served",
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"token": "answer", "logprob": -0.1}]},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }

        backend = OpenAICompatibleBackend(
            model_id="model",
            base_url="http://127.0.0.1:8000/v1",
            deployment_identity={"revision": "weights-abc"},
            capabilities=frozenset(
                {
                    ModelCapability.CHAT,
                    ModelCapability.LOGPROBS,
                }
            ),
            transport=transport,
        )
        response = backend.generate(
            GenerationRequest(prompt="question", logprobs=True, top_logprobs=5)
        )

        self.assertTrue(captured["payload"]["logprobs"])
        self.assertEqual(captured["payload"]["top_logprobs"], 5)
        self.assertIn("logprobs", response.raw["choices"][0])

        unsupported = OpenAICompatibleBackend(
            model_id="model",
            base_url="http://127.0.0.1:8000/v1",
            deployment_identity={"revision": "weights-abc"},
            capabilities=frozenset({ModelCapability.CHAT}),
            transport=transport,
        )
        with self.assertRaisesRegex(CapabilityError, "logprobs"):
            unsupported.generate(GenerationRequest(prompt="question", logprobs=True))

    def test_capability_mismatch_is_explicit(self) -> None:
        backend = OpenAICompatibleBackend(
            model_id="m",
            base_url="https://example.test/v1",
            deployment_identity={"revision": "served-v1"},
            capabilities=frozenset({ModelCapability.CHAT}),
        )
        with self.assertRaises(CapabilityError):
            backend.generate(GenerationRequest(prompt="q", seed=1))

    def test_openai_fingerprint_includes_full_base_path(self) -> None:
        first = OpenAICompatibleBackend(
            model_id="m",
            base_url="https://example.test/v1",
            deployment_identity={"revision": "served-v1"},
        )
        second = OpenAICompatibleBackend(
            model_id="m",
            base_url="https://example.test/deployments/a/v1",
            deployment_identity={"revision": "served-v1"},
        )
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_openai_base_url_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(ValidationError):
            OpenAICompatibleBackend(
                model_id="m",
                base_url="https://secret@example.test/v1",
                deployment_identity={"revision": "served-v1"},
            )

    def test_openai_fingerprint_detects_routing_header_and_deployment_changes(self) -> None:
        common = {
            "model_id": "m",
            "base_url": "https://example.test/v1",
        }
        first = OpenAICompatibleBackend(
            **common,
            deployment_identity={"revision": "weights-a"},
            extra_headers={"X-Model-Route": "blue"},
        )
        second = OpenAICompatibleBackend(
            **common,
            deployment_identity={"revision": "weights-a"},
            extra_headers={"X-Model-Route": "green"},
        )
        third = OpenAICompatibleBackend(
            **common,
            deployment_identity={"revision": "weights-b"},
            extra_headers={"X-Model-Route": "blue"},
        )
        self.assertNotEqual(first.fingerprint(), second.fingerprint())
        self.assertNotEqual(first.fingerprint(), third.fingerprint())

    def test_callable_backend_rejects_implicit_identity(self) -> None:
        with self.assertRaisesRegex(ValidationError, "stable revision"):
            CallableBackend(lambda request: request.prompt)

    def test_inmemory_bm25_is_deterministic(self) -> None:
        retriever = InMemoryBM25Retriever(
            [
                Document("a", "Paris is the capital of France", corpus_revision="tiny-v1"),
                Document("b", "Berlin is the capital of Germany", corpus_revision="tiny-v1"),
            ]
        )
        first = retriever.retrieve("France capital", top_k=2)
        second = retriever.retrieve("France capital", top_k=2)
        self.assertEqual(tuple(item.document.doc_id for item in first), ("a", "b"))
        self.assertEqual(first, second)
        self.assertEqual(tuple(item.rank for item in first), (1, 2))

    def test_custom_bm25_tokenizer_requires_identity(self) -> None:
        with self.assertRaises(ValidationError):
            InMemoryBM25Retriever(
                [Document("a", "text", corpus_revision="tiny-v1")],
                tokenizer=lambda text: tuple(text.split()),
            )

    def test_transformers_seed_fails_closed_before_model_loading(self) -> None:
        backend = TransformersBackend(
            model_id="fixture",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
        )
        with self.assertRaisesRegex(CapabilityError, "seed"):
            backend.generate(GenerationRequest(prompt="q", seed=1))

    def test_http_backend_exposes_only_public_declared_revision(self) -> None:
        backend = OpenAICompatibleBackend(
            model_id="served-model",
            base_url="http://127.0.0.1:9000/v1",
            deployment_identity={
                "revision": "immutable-weights-sha",
                "private_operator_note": "not-published",
            },
        )
        fingerprint = backend.fingerprint()
        self.assertEqual(fingerprint["model_revision"], "immutable-weights-sha")
        self.assertNotIn("private_operator_note", fingerprint)
        self.assertNotIn("not-published", repr(fingerprint))

    def test_generation_request_and_usage_reject_invalid_numeric_values(self) -> None:
        invalid_requests = (
            {"max_tokens": True},
            {"max_tokens": 1.5},
            {"max_tokens": 0},
            {"temperature": True},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"temperature": 10**1000},
            {"temperature": -0.1},
            {"seed": True},
            {"seed": 1.5},
            {"seed": -1},
            {"logprobs": "yes"},
            {"top_logprobs": True, "logprobs": True},
            {"top_logprobs": 0, "logprobs": True},
            {"top_logprobs": 21, "logprobs": True},
            {"top_logprobs": 5},
        )
        for values in invalid_requests:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                GenerationRequest(prompt="q", **values)

        invalid_usage = (
            {"input_tokens": True},
            {"input_tokens": 1.5},
            {"input_tokens": -1},
            {"output_tokens": True},
            {"output_tokens": -1},
            {"cost_usd": True},
            {"cost_usd": float("nan")},
            {"cost_usd": float("inf")},
            {"cost_usd": 10**1000},
            {"cost_usd": -0.01},
        )
        for values in invalid_usage:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Usage(**values)

    def test_transformers_usage_counts_non_padding_input_and_generated_tokens(self) -> None:
        backend = self._transformers_backend()
        response = backend.generate(GenerationRequest(prompt="q", max_tokens=8))

        self.assertEqual(response.text, "generated")
        self.assertEqual(response.usage, Usage(input_tokens=3, output_tokens=2))
        self.assertEqual(backend.tokenizer.decoded, ([21, 22], True))

    def test_transformers_usage_falls_back_to_input_width_for_plain_outputs(self) -> None:
        backend = self._transformers_backend(
            include_attention_mask=False,
            return_dict=False,
        )
        response = backend.generate(GenerationRequest(prompt="q"))
        self.assertEqual(response.usage, Usage(input_tokens=4, output_tokens=2))

    def test_transformers_usage_triggers_observed_output_budget(self) -> None:
        backend = self._transformers_backend()
        monitor = RunMonitor()
        run_id = monitor.start_run(release_id="fixture-v1", state_hash="state")
        observed = ObservedLLMBackend(
            backend,
            monitor,
            BudgetLedger(ExecutionBudget(max_output_tokens=1)),
            run_id,
        )

        with self.assertRaisesRegex(BudgetExceededError, "output_tokens"):
            observed.generate(GenerationRequest(prompt="q"))
        self.assertEqual(monitor.snapshot(run_id).usage.output_tokens, 2)

    def test_transformers_requires_exactly_one_return_sequence(self) -> None:
        for value in (True, 0, 2, 1.0, "1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValidationError, "num_return_sequences"),
            ):
                self._transformers_backend(generation_kwargs={"num_return_sequences": value})

        backend = self._transformers_backend(generation_kwargs={"num_return_sequences": 1})
        self.assertEqual(backend.generate(GenerationRequest(prompt="q")).usage.output_tokens, 2)

    def test_transformers_rejects_malformed_input_and_output_shapes(self) -> None:
        invalid_cases = (
            {"input_ids": [101, 11, 12, 0]},
            {"input_ids": [[101, 11], [101, 12]]},
            {"sequences": [101, 11, 12, 0, 21]},
            {"sequences": [[101, 11, 12, 0, 21], [101, 11, 12, 0, 22]]},
            {"sequences": [[101, 11, 12]]},
        )
        for values in invalid_cases:
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValidationError, "unsupported sequence shape"),
            ):
                self._transformers_backend(**values).generate(GenerationRequest(prompt="q"))

    def test_transformers_rejects_malformed_attention_masks(self) -> None:
        invalid_masks = (
            [1, 1, 1, 0],
            [[1, 1, 1, 0], [1, 1, 1, 0]],
            [[1, 1, 1]],
            [[1, 1, 1, 2]],
            [[-1, 0, 0, 0]],
        )
        for attention_mask in invalid_masks:
            with (
                self.subTest(attention_mask=attention_mask),
                self.assertRaisesRegex(ValidationError, "unsupported attention_mask"),
            ):
                self._transformers_backend(attention_mask=attention_mask).generate(
                    GenerationRequest(prompt="q")
                )


if __name__ == "__main__":
    unittest.main()
