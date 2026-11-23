from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentinputlab.algorithms.clarify.ask_or_assume import ASK_OR_ASSUME_INTENT_PROMPT_SHA256
from agentinputlab.integrations.external_workers.method import MethodWorkerRequest
from workers.ask_or_assume.driver import (
    CLARIFY_TOOL_PATH,
    INTENT_PROMPT_PATH,
    INTENT_TOOL_PATH,
    METHOD_ID,
    PROTOCOL_VERSION,
    RELEASE_ID,
    TOOL_NAMES_PATH,
    UPSTREAM_COMMIT,
    AskOrAssumeRuntime,
    BackendConfig,
    CompatibilityPolicy,
    CompletionResult,
    DriverError,
    RoleConfig,
    load_compatibility_policy,
    load_official_assets,
    serve,
    verify_checkout,
)

_LICENSE = """Portions of this software are licensed as follows:
* All content that resides under the enterprise/ directory uses enterprise/LICENSE.
* Content outside that directory is available under the MIT license.

The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy.
"""
_PROMPT = """You are Intent Agent. Assess ambiguity and call `clarify_decision`.
Keep reasoning short and use a boolean needs_clarification verdict.
"""
_TOOL_NAMES = """
CLARIFY_TOOL_NAME = 'clarify'
CLARIFY_DECISION_TOOL_NAME = 'clarify_decision'
"""
_INTENT_TOOL = """
_DESCRIPTION = 'Decide whether clarification is required.'
ClarifyDecisionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CLARIFY_DECISION_TOOL_NAME,
        description=_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['needs_clarification'],
            'properties': {
                'needs_clarification': {'type': 'boolean'},
                'message': {'type': 'string'},
                'reasons': {'type': 'string'},
            },
            'additionalProperties': False,
        },
    ),
)
"""
_CLARIFY_TOOL = """
_DESCRIPTION = 'Ask the user targeted clarification questions.'
ClarifyTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CLARIFY_TOOL_NAME,
        description=_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['questions'],
            'properties': {
                'message': {'type': 'string'},
                'questions': {
                    'type': 'array',
                    'minItems': 1,
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'text': {'type': 'string'},
                            'options': {'type': 'array', 'items': {'type': 'string'}},
                            'default': {'type': 'string'},
                        },
                        'required': ['id', 'text'],
                    },
                },
                'wait_for_response': {'type': 'boolean', 'default': True},
            },
            'additionalProperties': False,
        },
    ),
)
"""

_INTENT_FP = {"provider": "fixture", "model": "intent-model", "revision": "v1"}
_CODING_FP = {"provider": "fixture", "model": "coding-model", "revision": "v1"}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_fake_checkout(root: Path) -> CompatibilityPolicy:
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(f"{UPSTREAM_COMMIT}\n", encoding="utf-8")
    files = {
        "LICENSE": _LICENSE,
        INTENT_PROMPT_PATH: _PROMPT,
        TOOL_NAMES_PATH: _TOOL_NAMES,
        INTENT_TOOL_PATH: _INTENT_TOOL,
        CLARIFY_TOOL_PATH: _CLARIFY_TOOL,
        "enterprise/LICENSE": "PolyForm Free Trial License 1.0.0\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return CompatibilityPolicy(
        commit=UPSTREAM_COMMIT,
        license_path="LICENSE",
        license_spdx="MIT",
        license_sha256=_sha(_LICENSE),
        require_clean_checkout=False,
        excluded_prefixes=("enterprise/",),
        source_hashes={
            relative: _sha(content)
            for relative, content in files.items()
            if relative != "LICENSE" and not relative.startswith("enterprise/")
        },
    )


def backend_config() -> BackendConfig:
    return BackendConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="FIXTURE_API_KEY",
        timeout_seconds=10.0,
        roles={
            "intent": RoleConfig(
                model="intent-model",
                fingerprint=_INTENT_FP,
                max_output_tokens=128,
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            ),
            "coding": RoleConfig(
                model="coding-model",
                fingerprint=_CODING_FP,
                max_output_tokens=256,
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            ),
        },
    )


class FixtureBackend:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def fingerprint(self, role: str):
        return _INTENT_FP if role == "intent" else _CODING_FP

    def complete(self, *, role, messages, tool, seed):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "tool": tool,
                "seed": seed,
            }
        )
        if not self.responses:
            raise AssertionError("fixture backend response sequence is exhausted")
        return CompletionResult(
            arguments=self.responses.pop(0),
            input_tokens=10,
            output_tokens=4,
        )


class RuntimeClient:
    worker_id = "ask-or-assume-fixture"

    def __init__(self, runtime: AskOrAssumeRuntime) -> None:
        self.runtime = runtime
        self.last_payload = None
        self.last_response = None

    def call(self, payload):
        self.last_payload = payload
        output = io.StringIO()
        serve(self.runtime, io.StringIO(json.dumps(payload) + "\n"), output)
        lines = output.getvalue().splitlines()
        if len(lines) != 1:
            raise AssertionError(f"worker emitted {len(lines)} JSON lines")
        self.last_response = json.loads(lines[0])
        return self.last_response

    def fingerprint(self):
        return {
            "worker_id": self.worker_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "execution_mode": "controlled_component_harness",
        }


def request_payload(
    *,
    prompt_hash: str,
    turn_index: int = 1,
    user_turns: int = 0,
    controls: dict[str, object] | None = None,
) -> dict[str, object]:
    configured_controls = controls or {
        "skip_first_turn": True,
        "max_coding_turns": 100,
        "max_user_turns": 3,
        "max_model_calls": 2,
        "max_tokens": 1000,
        "max_cost_usd": 1.0,
    }
    request = MethodWorkerRequest(
        method_id=METHOD_ID,
        release_id=RELEASE_ID,
        upstream_commit=UPSTREAM_COMMIT,
        state_hash="a" * 64,
        payload={
            "task_id": "fixture-task",
            "task_view": {
                "query": "Fix it",
                "instruction": "",
                "context": [],
                "tool_names": ["read_file"],
            },
            "history": [
                {"role": "user", "content": "Fix it", "name": None},
                {"role": "assistant", "content": "I inspected the repo.", "name": None},
            ],
            "turn_index": turn_index,
            "clarification_turns": user_turns,
            "controls": configured_controls,
            "intent_prompt_hash": prompt_hash,
            "seed": 7,
        },
        dependency_fingerprints={
            "intent_model": _INTENT_FP,
            "coding_model": _CODING_FP,
        },
        minimum_reservation={"model_calls": 1},
        protocol_version=PROTOCOL_VERSION,
    )
    return request.as_dict()


def rehash(payload: dict[str, object]) -> None:
    payload["request_id"] = _canonical_hash(
        {key: value for key, value in payload.items() if key != "request_id"}
    )


class AskOrAssumeWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.checkout = Path(self.temp.name)
        self.policy = write_fake_checkout(self.checkout)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runtime(self, responses: list[dict[str, object]]):
        backend = FixtureBackend(responses)
        runtime = AskOrAssumeRuntime(
            self.checkout,
            backend_config(),
            backend,
            policy=self.policy,
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
                    "timeout_seconds": 10.0,
                    "roles": {
                        role: {
                            "model": "served-model",
                            "fingerprint": fingerprint,
                            "max_output_tokens": 128,
                            "input_cost_per_million": 0.0,
                            "output_cost_per_million": 0.0,
                        }
                        for role in ("intent", "coding")
                    },
                }
            )
        )
        self.assertEqual(config.roles["coding"].fingerprint, fingerprint)

    def test_fake_checkout_e2e_preserves_decision_reasons_messages_and_tools(self) -> None:
        runtime, backend = self.runtime(
            [
                {
                    "needs_clarification": True,
                    "reasons": "The target repository is ambiguous.",
                    "message": "Repository choice is missing.",
                },
                {
                    "message": "Before I edit files, choose a repository.",
                    "questions": [
                        {
                            "id": "repo",
                            "text": "Which repository should I edit?",
                            "options": ["frontend", "backend"],
                            "default": "backend",
                        }
                    ],
                    "wait_for_response": True,
                },
            ]
        )
        client = RuntimeClient(runtime)

        # Build the transport request directly because the production adapter
        # correctly refuses the synthetic prompt identity.
        response = client.call(request_payload(prompt_hash=_sha(_PROMPT)))

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["usage"]["model_calls"], 2)
        self.assertEqual([call["role"] for call in backend.calls], ["intent", "coding"])
        self.assertEqual(
            [call["tool"]["function"]["name"] for call in backend.calls],
            ["clarify_decision", "clarify"],
        )
        result = response["payload"]
        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["reasons"], ["The target repository is ambiguous."])
        self.assertEqual(result["message"], "Repository choice is missing.")
        self.assertIn("Q1. Which repository should I edit?", result["clarification_question"])
        self.assertEqual(
            [(item["path"], item["message"]) for item in result["decision_trace"]],
            [
                ("intent_decision", "Repository choice is missing."),
                ("forced_clarify", "Before I edit files, choose a repository."),
            ],
        )
        self.assertTrue(result["decision_trace"][1]["wait_for_response"])
        self.assertTrue(result["component_only"])
        self.assertFalse(result["openhands_full_system"])
        self.assertFalse(result["ua_multi_parity_claimed"])

    def test_verdict_only_path_uses_one_model_call(self) -> None:
        runtime, backend = self.runtime(
            [
                {
                    "needs_clarification": False,
                    "reasons": "",
                    "message": "The task is sufficiently specified.",
                }
            ]
        )
        response = runtime.run(request_payload(prompt_hash=_sha(_PROMPT)))
        self.assertFalse(response["payload"]["needs_clarification"])
        self.assertEqual(response["payload"]["transition"], "resume")
        self.assertEqual(response["usage"]["model_calls"], 1)
        self.assertEqual([call["role"] for call in backend.calls], ["intent"])

    def test_effective_turn_and_provider_budgets_are_hard_limits(self) -> None:
        cases = (
            (
                "coding",
                request_payload(
                    prompt_hash=_sha(_PROMPT),
                    turn_index=2,
                    controls={
                        "skip_first_turn": True,
                        "max_coding_turns": 2,
                        "max_user_turns": 3,
                        "max_model_calls": 2,
                        "max_tokens": 1000,
                        "max_cost_usd": 1.0,
                    },
                ),
                "CODING_TURN_BUDGET_EXCEEDED",
                0,
            ),
            (
                "user",
                request_payload(prompt_hash=_sha(_PROMPT), user_turns=3),
                "USER_TURN_BUDGET_EXCEEDED",
                1,
            ),
            (
                "model",
                request_payload(
                    prompt_hash=_sha(_PROMPT),
                    controls={
                        "skip_first_turn": True,
                        "max_coding_turns": 100,
                        "max_user_turns": 3,
                        "max_model_calls": 1,
                        "max_tokens": 1000,
                        "max_cost_usd": 1.0,
                    },
                ),
                "MODEL_CALL_BUDGET_EXCEEDED",
                1,
            ),
        )
        for label, payload, code, expected_calls in cases:
            with self.subTest(label=label):
                runtime, backend = self.runtime(
                    [
                        {
                            "needs_clarification": True,
                            "reasons": "Missing repository.",
                            "message": "Need repository.",
                        },
                        {
                            "questions": [{"id": "repo", "text": "Which repository?"}],
                        },
                    ]
                )
                with self.assertRaises(DriverError) as raised:
                    runtime.run(payload)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(len(backend.calls), expected_calls)

    def test_provider_usage_enforces_token_and_cost_budgets(self) -> None:
        cases = (
            ("tokens", "max_tokens", 13, "TOKEN_BUDGET_EXCEEDED"),
            ("cost", "max_cost_usd", 0.000001, "COST_BUDGET_EXCEEDED"),
        )
        for label, control, limit, code in cases:
            with self.subTest(label=label):
                configured = {
                    "skip_first_turn": True,
                    "max_coding_turns": 100,
                    "max_user_turns": 3,
                    "max_model_calls": 2,
                    "max_tokens": 1000,
                    "max_cost_usd": 1.0,
                }
                configured[control] = limit
                runtime, backend = self.runtime(
                    [
                        {
                            "needs_clarification": False,
                            "reasons": "",
                            "message": "Task is clear.",
                        }
                    ]
                )
                with self.assertRaises(DriverError) as raised:
                    runtime.run(
                        request_payload(
                            prompt_hash=_sha(_PROMPT),
                            controls=configured,
                        )
                    )
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(len(backend.calls), 1)

    def test_request_binds_prompt_models_and_rejects_runtime_injection(self) -> None:
        mutations = (
            (
                "request id",
                lambda payload: payload.update({"request_id": "0" * 64}),
                False,
                "REQUEST_ID_MISMATCH",
            ),
            (
                "prompt",
                lambda payload: payload["payload"].update({"intent_prompt_hash": "0" * 64}),
                True,
                "PROMPT_IDENTITY_MISMATCH",
            ),
            (
                "model",
                lambda payload: payload["dependency_fingerprints"]["intent_model"].update(
                    {"model": "attacker-model"}
                ),
                True,
                "BACKEND_IDENTITY_MISMATCH",
            ),
            (
                "endpoint",
                lambda payload: payload["payload"].update(
                    {"base_url": "https://attacker.invalid/v1"}
                ),
                True,
                "INVALID_SCHEMA",
            ),
        )
        for label, mutate, update_hash, code in mutations:
            with self.subTest(label=label):
                runtime, backend = self.runtime([])
                payload = request_payload(prompt_hash=_sha(_PROMPT))
                mutate(payload)
                if update_hash:
                    rehash(payload)
                with self.assertRaises(DriverError) as raised:
                    runtime.run(payload)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(backend.calls, [])

    def test_license_boundary_excludes_enterprise_and_clean_gate_is_default(self) -> None:
        default = load_compatibility_policy()
        self.assertEqual(default.license_path, "LICENSE")
        self.assertEqual(default.excluded_prefixes, ("enterprise/",))
        self.assertTrue(default.require_clean_checkout)
        self.assertFalse(any(path.startswith("enterprise/") for path in default.source_hashes))

        enterprise_policy = replace(
            self.policy,
            source_hashes={"enterprise/LICENSE": _sha("PolyForm Free Trial License 1.0.0\n")},
        )
        with self.assertRaises(DriverError) as excluded:
            verify_checkout(self.checkout, enterprise_policy)
        self.assertEqual(excluded.exception.code, "ENTERPRISE_EXCLUDED")

        dirty = subprocess.CompletedProcess(
            args=("git", "status"), returncode=0, stdout="?? enterprise-copy.py\n", stderr=""
        )
        with (
            patch("workers.ask_or_assume.driver.subprocess.run", return_value=dirty),
            self.assertRaises(DriverError) as raised,
        ):
            verify_checkout(self.checkout, replace(self.policy, require_clean_checkout=True))
        self.assertEqual(raised.exception.code, "DIRTY_CHECKOUT")

    def test_root_license_and_backend_outputs_fail_closed(self) -> None:
        license_path = self.checkout / "LICENSE"
        license_path.write_text(_LICENSE + "tampered\n", encoding="utf-8")
        with self.assertRaises(DriverError) as license_error:
            verify_checkout(self.checkout, self.policy)
        self.assertEqual(license_error.exception.code, "UPSTREAM_LICENSE_MISMATCH")

        # Restore the fixture and ensure truthy integers cannot become decisions.
        license_path.write_text(_LICENSE, encoding="utf-8")
        runtime, backend = self.runtime(
            [{"needs_clarification": 1, "reasons": "ambiguous", "message": "need input"}]
        )
        with self.assertRaises(DriverError) as output_error:
            runtime.run(request_payload(prompt_hash=_sha(_PROMPT)))
        self.assertEqual(output_error.exception.code, "BACKEND_OUTPUT_INVALID")
        self.assertEqual(len(backend.calls), 1)

    def test_jsonl_boundary_emits_one_structured_error(self) -> None:
        runtime, _ = self.runtime([])
        output = io.StringIO()
        serve(runtime, io.StringIO('{"request_id":"a","request_id":"b"}\n'), output)
        response = json.loads(output.getvalue())
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["failure_code"], "INVALID_JSON")

    @unittest.skipUnless(
        os.environ.get("ASK_OR_ASSUME_UPSTREAM_CHECKOUT"),
        "set ASK_OR_ASSUME_UPSTREAM_CHECKOUT for the pinned real-checkout smoke",
    )
    def test_real_checkout_prompt_schema_and_license_smoke(self) -> None:
        checkout = Path(os.environ["ASK_OR_ASSUME_UPSTREAM_CHECKOUT"])
        root = verify_checkout(checkout, load_compatibility_policy())
        assets = load_official_assets(root)
        self.assertEqual(assets.intent_prompt_sha256, ASK_OR_ASSUME_INTENT_PROMPT_SHA256)
        self.assertEqual(assets.intent_tool["function"]["name"], "clarify_decision")
        self.assertEqual(assets.clarify_tool["function"]["name"], "clarify")


if __name__ == "__main__":
    unittest.main()
