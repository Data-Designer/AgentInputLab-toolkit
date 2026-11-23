from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentinputlab.integrations.external_workers.method import MethodWorkerRequest
from workers.reformer.driver import (
    CORE_PATH,
    DEFAULT_PATTERN_PATH,
    INDUCTION_PATH,
    METHOD_ID,
    ADAPTER_REVISION,
    PROMPT_MANAGER_PATH,
    PROMPT_PATH,
    PROTOCOL_VERSION,
    UPSTREAM_COMMIT,
    BackendConfig,
    CompatibilityPolicy,
    CompletionResult,
    DriverError,
    OpenAICompatibleBackend,
    ReFormeRRuntime,
    audit_source_symbols,
    load_compatibility_policy,
    serve,
    verify_checkout,
    _require_library_pattern,
)

_README = "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]\n"
_PROMPTS = {
    "system_prompts": {
        "pattern_selection": "select system",
        "pattern_application": "apply system",
        "pattern_extraction": "extract system",
    },
    "user_prompts": {
        "pattern_selection": {
            "template": "QUERY={query}\nDOCS={documents_text}\nPATTERNS={patterns_json}",
            "placeholders": ["query", "documents_text", "patterns_json"],
        },
        "pattern_application": {
            "template": "QUERY={query}\nRULE={transformation_rule}\nEXAMPLES={examples_json}",
            "placeholders": ["query", "transformation_rule", "examples_json"],
        },
    },
    "models": {},
    "parameters": {},
}
_PATTERNS = [
    {
        "pattern_name": "Clarify Intent",
        "description": "Clarify the intended information need.",
        "transformation_rule": "Make the intended target explicit.",
        "examples": [["Dune author?", "who wrote the novel Dune"]],
    },
    {
        "pattern_name": "Contextual Expansion",
        "description": "Add relevant document context.",
        "transformation_rule": "Add context without changing the target.",
        "examples": [["Mercury year", "length of a year on planet Mercury"]],
    },
]
_PROMPT_MANAGER = """
import json

class PromptManager:
    def __init__(self, prompts_file=None):
        with open(prompts_file, 'r') as handle:
            self.prompts = json.load(handle)

    def create_messages(self, prompt_type, system_prompt_type=None, **kwargs):
        messages = []
        if system_prompt_type:
            messages.append({
                'role': 'system',
                'content': self.prompts['system_prompts'][system_prompt_type],
            })
        template = self.prompts['user_prompts'][prompt_type]['template']
        messages.append({'role': 'user', 'content': template.format(**kwargs)})
        return messages
"""
_CORE = """
import json
import logging
from vllm import LLM, SamplingParams
from ..prompt_manager import PromptManager

class Reformulator:
    def select_best_pattern(self, query, documents):
        if not documents:
            return self.patterns[0] if self.patterns else None
        documents_text = '\\n\\n'.join(documents[:3])
        try:
            messages = self.prompt_manager.create_messages(
                prompt_type='pattern_selection',
                system_prompt_type='pattern_selection',
                query=query,
                documents_text=documents_text,
                patterns_json=json.dumps(self.patterns, indent=2),
            )
            chat_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = self.llm.generate(
                prompts=[chat_prompt], sampling_params=self.sampling_params
            )
            return json.loads(outputs[0].outputs[0].text.strip())
        except Exception as exc:
            logging.error('selection failed: %s', exc)
            return self.patterns[0] if self.patterns else None

    def apply_pattern(self, query, pattern):
        if not pattern:
            return query
        try:
            messages = self.prompt_manager.create_messages(
                prompt_type='pattern_application',
                system_prompt_type='pattern_application',
                query=query,
                transformation_rule=pattern.get('transformation_rule', ''),
                examples_json=json.dumps(pattern.get('examples', []), indent=2),
            )
            chat_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = self.llm.generate(
                prompts=[chat_prompt], sampling_params=self.sampling_params
            )
            return outputs[0].outputs[0].text.strip()
        except Exception as exc:
            logging.error('application failed: %s', exc)
            return query

    def reformulate_query(self, query_id, query):
        selected = self.select_best_pattern(query, self.get_top_documents(query_id))
        return {'reformulated_query': self.apply_pattern(query, selected)}
"""
_INDUCTION = """
class IterativePatternExtractor:
    def extract_patterns_from_batch(self, query_pairs, batch_number):
        return []

    def run_iterative_extraction(self):
        return None
"""
_MODEL_FP = {"provider": "fixture", "model": "fixture-model", "revision": "v1"}
_RETRIEVER_FP = {"provider": "fixture", "index": "fixture-index", "revision": "v1"}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


class FakeCheckout:
    def __init__(self, root: Path) -> None:
        self.root = root
        _write(root / ".git" / "HEAD", UPSTREAM_COMMIT + "\n")
        files = {
            "README.md": _README,
            "requirements.txt": "fixture-dependency\n",
            "reformer/__init__.py": "",
            "reformer/core/__init__.py": "",
            CORE_PATH: _CORE,
            PROMPT_MANAGER_PATH: _PROMPT_MANAGER,
            PROMPT_PATH: json.dumps(_PROMPTS, indent=2),
            DEFAULT_PATTERN_PATH: json.dumps(_PATTERNS, indent=2),
            "reformer/patterns/query_reformulation_prompts.py": "class QueryPair: pass\n",
            INDUCTION_PATH: _INDUCTION,
        }
        for relative, content in files.items():
            _write(root / relative, content)
        source_hashes = {
            relative: _sha_bytes((root / relative).read_bytes())
            for relative in files
            if relative not in {"reformer/__init__.py", "reformer/core/__init__.py"}
        }
        self.pattern_hash = source_hashes[DEFAULT_PATTERN_PATH]
        self.prompt_hash = source_hashes[PROMPT_PATH]
        self.policy = CompatibilityPolicy(
            commit=UPSTREAM_COMMIT,
            license_status="NOASSERTION",
            license_file_present=False,
            license_evidence_path="README.md",
            license_evidence_sha256=source_hashes["README.md"],
            require_clean_checkout=False,
            source_hashes=source_hashes,
            profiles={
                "rewrite.reformer.repo_native_r_only": {
                    "semantics": "r_only",
                    "pattern_mode": "automatic",
                },
                "rewrite.reformer.paper_spec_q_concat_r": {
                    "semantics": "q_concat_r",
                    "pattern_mode": "automatic",
                },
                "rewrite.reformer.controlled_r_only": {
                    "semantics": "r_only",
                    "pattern_mode": "manual",
                },
                "rewrite.reformer.controlled_q_concat_r": {
                    "semantics": "q_concat_r",
                    "pattern_mode": "manual",
                },
            },
        )


class FixtureBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def fingerprint(self):
        return dict(_MODEL_FP)

    def complete(self, *, messages, seed, temperature, max_output_tokens):
        self.calls.append(
            {
                "messages": messages,
                "seed": seed,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        if not self.outputs:
            raise AssertionError("fixture output sequence exhausted")
        return CompletionResult(self.outputs.pop(0), input_tokens=12, output_tokens=4)


class FingerprintedDependency:
    def __init__(self, value):
        self.value = value

    def fingerprint(self):
        return dict(self.value)


class RuntimeClient:
    worker_id = "reformer-fixture"

    def __init__(self, runtime: ReFormeRRuntime) -> None:
        self.runtime = runtime

    def call(self, payload):
        output = io.StringIO()
        serve(self.runtime, io.StringIO(_canonical(payload) + "\n"), output)
        return json.loads(output.getvalue())

    def fingerprint(self):
        return {"worker_id": self.worker_id, "upstream_commit": UPSTREAM_COMMIT}


def backend_config() -> BackendConfig:
    return BackendConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="FIXTURE_KEY",
        model="fixture-model",
        fingerprint=_MODEL_FP,
        extra_body={},
        timeout_seconds=5.0,
        max_input_tokens_per_call=512,
        max_output_tokens=128,
        temperature=0.1,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )


def request_payload(
    fixture: FakeCheckout,
    *,
    release_id: str = "rewrite.reformer.repo_native_r_only",
    semantics: str = "r_only",
    pattern_mode: str = "automatic",
    manual_pattern_id: str = "",
    controls: dict[str, object] | None = None,
) -> dict[str, object]:
    request = MethodWorkerRequest(
        method_id=METHOD_ID,
        release_id=release_id,
        upstream_commit=UPSTREAM_COMMIT,
        state_hash="a" * 64,
        payload={
            "task_id": "fixture-task",
            "query": "Dune author?",
            "context": [
                {
                    "text": "Frank Herbert wrote Dune.",
                    "block_id": "d1",
                    "source": "fixture",
                    "score": 1.0,
                    "metadata": {},
                }
            ],
            "top_document_count": 3,
            "pattern_artifact_hash": fixture.pattern_hash,
            "prompt_artifact_hash": fixture.prompt_hash,
            "semantics": semantics,
            "pattern_mode": pattern_mode,
            "manual_pattern_id": manual_pattern_id,
            "source_license_acknowledged": True,
            "controls": controls or {"max_model_calls": 2, "max_tokens": 1000, "max_cost_usd": 1.0},
            "seed": 11,
        },
        dependency_fingerprints={
            "backend": _MODEL_FP,
            "retriever": _RETRIEVER_FP if pattern_mode == "automatic" else {},
        },
        minimum_reservation={"model_calls": 2 if pattern_mode == "automatic" else 1},
        protocol_version=PROTOCOL_VERSION,
    )
    return request.as_dict()


def rehash(payload: dict[str, object]) -> None:
    payload["request_id"] = _sha_bytes(
        _canonical({key: value for key, value in payload.items() if key != "request_id"}).encode()
    )


class ReFormeRWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.checkout = FakeCheckout(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runtime(self, outputs: list[str]):
        backend = FixtureBackend(outputs)
        runtime = ReFormeRRuntime(
            self.checkout.root,
            backend_config(),
            backend,
            source_license_acknowledged=True,
            policy=self.checkout.policy,
        )
        return runtime, backend

    def test_backend_config_accepts_agentinputlab_nested_model_identity(self) -> None:
        fingerprint = {
            "implementation": "agentinputlab.bound_llm_backend.v1",
            "backend": {"backend": "openai_compatible", "model_id": "served-model"},
        }
        config = BackendConfig.from_json(
            json.dumps(
                {
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key_env": "FIXTURE_API_KEY",
                    "model": "served-model",
                    "fingerprint": fingerprint,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    "timeout_seconds": 10.0,
                    "max_input_tokens_per_call": 512,
                    "max_output_tokens": 128,
                    "temperature": 0.0,
                    "input_cost_per_million": 0.0,
                    "output_cost_per_million": 0.0,
                }
            )
        )
        self.assertFalse(config.extra_body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(config.fingerprint, fingerprint)

    def test_fake_checkout_automatic_e2e_calls_official_selection_and_application(self) -> None:
        runtime, backend = self.runtime([json.dumps(_PATTERNS[1]), "expanded Dune author query"])
        response = runtime.run(request_payload(self.checkout))

        self.assertFalse(any(self.checkout.root.rglob("__pycache__")))
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["usage"]["model_calls"], 2)
        self.assertEqual(response["payload"]["selected_pattern_id"], "Contextual Expansion")
        self.assertEqual(response["payload"]["reformulation"], "expanded Dune author query")
        self.assertEqual(
            response["payload"]["official_calls"],
            ["Reformulator.select_best_pattern", "Reformulator.apply_pattern"],
        )
        self.assertIn("DOCS=Frank Herbert wrote Dune.", backend.calls[0]["messages"][1]["content"])
        self.assertIn("EXAMPLES=", backend.calls[1]["messages"][1]["content"])
        self.assertFalse(response["payload"]["vendored_upstream_source"])
        self.assertFalse(response["payload"]["paper_result_parity_claimed"])

    def test_four_profiles_diverge_only_on_pattern_mode_and_output_semantics(self) -> None:
        profiles = (
            (
                "rewrite.reformer.repo_native_r_only",
                "r_only",
                "automatic",
                "",
                "expanded query",
            ),
            (
                "rewrite.reformer.paper_spec_q_concat_r",
                "q_concat_r",
                "automatic",
                "",
                "Dune author? expanded query",
            ),
            (
                "rewrite.reformer.controlled_r_only",
                "r_only",
                "manual",
                "Clarify Intent",
                "expanded query",
            ),
            (
                "rewrite.reformer.controlled_q_concat_r",
                "q_concat_r",
                "manual",
                "Clarify Intent",
                "Dune author? expanded query",
            ),
        )
        for release_id, semantics, mode, manual_id, expected in profiles:
            with self.subTest(profile=release_id):
                outputs = (
                    [json.dumps(_PATTERNS[0]), "expanded query"]
                    if mode == "automatic"
                    else ["expanded query"]
                )
                runtime, backend = self.runtime(outputs)
                response = runtime.run(
                    request_payload(
                        self.checkout,
                        release_id=release_id,
                        semantics=semantics,
                        pattern_mode=mode,
                        manual_pattern_id=manual_id,
                        controls={
                            "max_model_calls": 2 if mode == "automatic" else 1,
                            "max_tokens": 1000,
                            "max_cost_usd": 1.0,
                        },
                    )
                )
                materialized = (
                    response["payload"]["reformulation"]
                    if semantics == "r_only"
                    else f"Dune author? {response['payload']['reformulation']}"
                )
                self.assertEqual(materialized, expected)
                self.assertEqual(response["payload"]["pattern_mode"], mode)
                self.assertEqual(response["payload"]["semantics"], semantics)
                self.assertEqual(len(backend.calls), 2 if mode == "automatic" else 1)

    def test_description_only_difference_keeps_exact_application_prompt_and_budget(self) -> None:
        original, original_backend = self.runtime([json.dumps(_PATTERNS[1]), "expanded query"])
        expected = original.run(request_payload(self.checkout))
        for description in ("Different non-operative annotation", None):
            selected = dict(_PATTERNS[1])
            if description is None:
                selected.pop("description")
            else:
                selected["description"] = description
            runtime, backend = self.runtime([json.dumps(selected), "expanded query"])
            actual = runtime.run(request_payload(self.checkout))
            self.assertEqual(backend.calls, original_backend.calls)
            self.assertEqual(actual["usage"], expected["usage"])
            self.assertEqual(actual["payload"]["selected_pattern_hash"],
                             expected["payload"]["selected_pattern_hash"])
            self.assertEqual(actual["payload"]["selection_library_match"], "application_fields")
            self.assertEqual(actual["payload"]["adapter_revision"], ADAPTER_REVISION)
            self.assertNotEqual(actual["payload"]["raw_selected_pattern_hash"],
                                actual["payload"]["selected_pattern_hash"])

    def test_unique_name_reference_uses_pinned_application_without_extra_call(self) -> None:
        for pattern in _PATTERNS:
            with self.subTest(pattern=pattern["pattern_name"]):
                original, original_backend = self.runtime([json.dumps(pattern), "expanded query"])
                expected = original.run(request_payload(self.checkout))
                runtime, backend = self.runtime([
                    json.dumps({"pattern_name": pattern["pattern_name"]}), "expanded query"
                ])
                actual = runtime.run(request_payload(self.checkout))
                self.assertEqual(backend.calls, original_backend.calls)
                self.assertEqual(actual["usage"], expected["usage"])
                self.assertEqual(actual["payload"]["selected_pattern_hash"],
                                 expected["payload"]["selected_pattern_hash"])
                self.assertEqual(actual["payload"]["selection_library_match"], "unique_name_reference")
                self.assertEqual(actual["payload"]["adapter_revision"], ADAPTER_REVISION)
                self.assertNotEqual(actual["payload"]["raw_selected_pattern_hash"],
                                    actual["payload"]["selected_pattern_hash"])
                self.assertEqual(len(backend.calls), 2)

    def test_ambiguous_name_reference_is_rejected(self) -> None:
        with self.assertRaises(DriverError) as error:
            _require_library_pattern({"pattern_name": _PATTERNS[0]["pattern_name"]},
                                     [_PATTERNS[0], dict(_PATTERNS[0])])
        self.assertEqual(error.exception.code, "PATTERN_SELECTION_OUTSIDE_LIBRARY")

    def test_unknown_name_or_changed_application_fails_and_records_paid_selection(self) -> None:
        variants = (
            {"pattern_name": "Unknown"},
            {"pattern_name": _PATTERNS[0]["pattern_name"] + " "},
            {"pattern_name": _PATTERNS[0]["pattern_name"].lower()},
            {"pattern_name": _PATTERNS[0]["pattern_name"], "transformation_rule": "Invent a different task"},
            {"pattern_name": _PATTERNS[0]["pattern_name"], "examples": []},
            {"pattern_name": _PATTERNS[0]["pattern_name"], "unexpected": True},
            {**_PATTERNS[0], "transformation_rule": "Invent a different task"},
            {**_PATTERNS[0], "examples": [["a", "b"]]},
            {**_PATTERNS[0], "pattern_name": "Unknown"},
        )
        for selected in variants:
            with self.subTest(selected=selected):
                runtime, backend = self.runtime([json.dumps(selected)])
                response = RuntimeClient(runtime).call(request_payload(self.checkout))
                self.assertEqual(response["failure_code"], "PATTERN_SELECTION_OUTSIDE_LIBRARY")
                self.assertTrue(response["consumed"])
                self.assertEqual(response["usage"]["model_calls"], 1)
                self.assertEqual(response["usage"]["input_tokens"], 12)
                self.assertEqual(response["usage"]["output_tokens"], 4)
                self.assertEqual(len(backend.calls), 1)

    def test_null_final_keeps_usage_and_does_not_use_reasoning_or_upstream_fallback(self) -> None:
        raw = json.dumps({
            "model": "fixture-model",
            "choices": [{"finish_reason": "length", "message": {
                "content": None, "reasoning_content": "PRIVATE NOT A FINAL ANSWER"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 128},
        }).encode()
        runtime = ReFormeRRuntime(
            self.checkout.root, backend_config(), OpenAICompatibleBackend(backend_config()),
            source_license_acknowledged=True, policy=self.checkout.policy,
        )
        with patch.dict(os.environ, {"FIXTURE_KEY": "PRIVATE CREDENTIAL"}), patch(
            "workers.reformer.driver.urllib.request.urlopen", return_value=io.BytesIO(raw)
        ) as transport:
            response = RuntimeClient(runtime).call(request_payload(self.checkout))
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(response["failure_code"], "BACKEND_OUTPUT_INVALID")
        self.assertTrue(response["consumed"])
        self.assertEqual(response["usage"]["model_calls"], 1)
        self.assertEqual(response["usage"]["input_tokens"], 50)
        self.assertEqual(response["usage"]["output_tokens"], 128)
        self.assertAlmostEqual(response["usage"]["cost_usd"], 306 / 1_000_000)
        self.assertEqual(response["diagnostics"]["failure_kind"], "empty_final_at_output_limit")
        self.assertEqual(response["diagnostics"]["response_sha256"], _sha_bytes(raw))
        self.assertNotIn("PRIVATE", json.dumps(response))

    def test_application_failure_adds_second_call_cost_to_successful_selection(self) -> None:
        responses = [
            {"choices": [{"message": {"content": json.dumps(_PATTERNS[0])}}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 4}},
            {"choices": [{"message": {"content": None}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 128}},
        ]
        runtime = ReFormeRRuntime(
            self.checkout.root, backend_config(), OpenAICompatibleBackend(backend_config()),
            source_license_acknowledged=True, policy=self.checkout.policy,
        )
        with patch.dict(os.environ, {"FIXTURE_KEY": "fixture"}), patch(
            "workers.reformer.driver.urllib.request.urlopen",
            side_effect=[io.BytesIO(json.dumps(row).encode()) for row in responses],
        ) as transport:
            response = RuntimeClient(runtime).call(request_payload(self.checkout))
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(response["usage"]["model_calls"], 2)
        self.assertEqual(response["usage"]["input_tokens"], 32)
        self.assertEqual(response["usage"]["output_tokens"], 132)
        self.assertEqual(len(response["diagnostics"]["model_trace"]), 2)

    def test_empty_text_fixture_must_not_activate_first_pattern_fallback(self) -> None:
        runtime, backend = self.runtime([""])
        response = RuntimeClient(runtime).call(request_payload(self.checkout))
        self.assertEqual(response["failure_code"], "BACKEND_OUTPUT_INVALID")
        self.assertEqual(response["usage"]["output_tokens"], 4)
        self.assertEqual(len(backend.calls), 1)

    def test_license_is_dual_gated_and_license_file_state_is_fail_closed(self) -> None:
        with self.assertRaises(DriverError) as startup:
            ReFormeRRuntime(
                self.checkout.root,
                backend_config(),
                FixtureBackend([]),
                source_license_acknowledged=False,
                policy=self.checkout.policy,
            )
        self.assertEqual(startup.exception.code, "SOURCE_LICENSE_NOT_ACKNOWLEDGED")

        runtime, backend = self.runtime([])
        payload = request_payload(self.checkout)
        payload["payload"]["source_license_acknowledged"] = False
        rehash(payload)
        with self.assertRaises(DriverError) as request_error:
            runtime.run(payload)
        self.assertEqual(request_error.exception.code, "SOURCE_LICENSE_NOT_ACKNOWLEDGED")
        self.assertEqual(backend.calls, [])

        _write(self.checkout.root / "LICENSE", "not part of the pinned revision")
        with self.assertRaises(DriverError) as state_error:
            verify_checkout(self.checkout.root, self.checkout.policy)
        self.assertEqual(state_error.exception.code, "LICENSE_STATE_MISMATCH")

    def test_budget_identity_and_request_injection_fail_closed(self) -> None:
        mutations = (
            (
                "pattern",
                lambda payload: payload["payload"].update({"pattern_artifact_hash": "0" * 64}),
                "PATTERN_IDENTITY_MISMATCH",
            ),
            (
                "model",
                lambda payload: payload["dependency_fingerprints"]["backend"].update(
                    {"model": "attacker"}
                ),
                "BACKEND_IDENTITY_MISMATCH",
            ),
            (
                "endpoint",
                lambda payload: payload["payload"].update(
                    {"base_url": "https://attacker.invalid/v1"}
                ),
                "INVALID_SCHEMA",
            ),
        )
        for label, mutate, code in mutations:
            with self.subTest(label=label):
                runtime, backend = self.runtime([])
                payload = request_payload(self.checkout)
                mutate(payload)
                rehash(payload)
                with self.assertRaises(DriverError) as raised:
                    runtime.run(payload)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(backend.calls, [])

        budget_cases = (
            (
                {"max_model_calls": 1, "max_tokens": 1000, "max_cost_usd": 1.0},
                "MODEL_CALL_BUDGET_EXCEEDED",
            ),
            (
                {"max_model_calls": 2, "max_tokens": 15, "max_cost_usd": 1.0},
                "TOKEN_BUDGET_EXCEEDED",
            ),
            (
                {"max_model_calls": 2, "max_tokens": 1000, "max_cost_usd": 0.000001},
                "COST_BUDGET_EXCEEDED",
            ),
        )
        for controls, code in budget_cases:
            with self.subTest(code=code):
                runtime, _ = self.runtime([json.dumps(_PATTERNS[0]), "rewrite"])
                with self.assertRaises(DriverError) as raised:
                    runtime.run(request_payload(self.checkout, controls=controls))
                self.assertEqual(raised.exception.code, code)

    def test_source_symbol_audit_includes_inference_and_nonexecuted_induction(self) -> None:
        symbols = audit_source_symbols(self.checkout.root, self.checkout.policy)
        self.assertEqual(
            symbols["Reformulator"],
            ("select_best_pattern", "apply_pattern", "reformulate_query"),
        )
        self.assertEqual(
            symbols["IterativePatternExtractor"],
            ("extract_patterns_from_batch", "run_iterative_extraction"),
        )
        default = load_compatibility_policy()
        self.assertEqual(default.license_status, "NOASSERTION")
        self.assertFalse(default.license_file_present)

    def test_jsonl_duplicate_keys_emit_structured_error(self) -> None:
        runtime, _ = self.runtime([])
        output = io.StringIO()
        serve(runtime, io.StringIO('{"request_id":"a","request_id":"b"}\n'), output)
        response = json.loads(output.getvalue())
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["failure_code"], "INVALID_JSON")

    @unittest.skipUnless(
        os.environ.get("REFORMER_UPSTREAM_CHECKOUT"),
        "set REFORMER_UPSTREAM_CHECKOUT to a clean pinned Git checkout",
    )
    def test_real_checkout_source_hash_and_symbol_smoke(self) -> None:
        root = verify_checkout(
            Path(os.environ["REFORMER_UPSTREAM_CHECKOUT"]), load_compatibility_policy()
        )
        symbols = audit_source_symbols(root, load_compatibility_policy())
        self.assertIn("select_best_pattern", symbols["Reformulator"])


if __name__ == "__main__":
    unittest.main()
