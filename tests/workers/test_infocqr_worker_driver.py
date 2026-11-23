from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from agentinputlab.algorithms.reformulate.infocqr import (
    INFOCQR_PROMPT_SOURCE_SHA256,
    InfoCQRExternalControls,
    InfoCQRLiveTeacherExternal,
    InfoCQRMountedCacheReplay,
)
from agentinputlab.contracts import FrozenAgentState, InterventionContext, Message
from agentinputlab.integrations.external_workers.method import MethodWorkerRequest
from workers.infocqr.driver import (
    FUSE_PATH,
    METHOD_ID,
    POSTPROCESS_PATH,
    PROTOCOL_VERSION,
    T5_TEST_PATH,
    T5_TRAIN_PATH,
    UPSTREAM_COMMIT,
    BackendConfig,
    CompatibilityPolicy,
    CompletionResult,
    DriverError,
    InfoCQRRuntime,
    MountedCache,
    load_compatibility_policy,
    serve,
    verify_checkout,
)

_MODEL_FP = {"provider": "fixture", "model": "fixture-model", "revision": "v1"}
_RETRIEVER_FP = {"provider": "fixture", "corpus": "qrecc-passages", "revision": "v1"}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


_SOURCE = """
import openai
import jsonlines
from tqdm import tqdm
import time

def generate_rewrite(line):
    history = []
    for index, text in enumerate(line['NewContext']):
        history.append(('Q: ' if index % 2 == 0 else 'A: ') + text)
    candidate = line.get('GPT_rewrite', line.get('T5_rewrite', ''))
    prompt = 'Context: [' + ' '.join(history) + ']\\nQuestion: ' + line['Question']
    if candidate:
        prompt += '\\nRewrite: ' + candidate + '\\nEdit: '
    else:
        prompt += '\\nRewrite: '
    return openai.ChatCompletion.create(
        model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': prompt}],
        temperature=0, max_tokens=2560, n=1,
    )
"""


class FakeCheckout:
    def __init__(self, root: Path) -> None:
        self.root = root
        files = {
            "README.md": "# InfoCQR fixture\n",
            "requirements.txt": "openai\n",
            "rewrite_chatgpt_zsl.py": _SOURCE,
            "rewrite_chatgpt_inctx.py": _SOURCE,
            "rewrite_chatgpt_editor.py": _SOURCE,
            "rewrite_chatgpt_self_editor.py": _SOURCE,
            POSTPROCESS_PATH: "def main(args): return None\n",
            FUSE_PATH: "def main(args): return None\n",
            "dataset_cleaning.py": "def main(args): return None\n",
            T5_TEST_PATH: "def main(): return None\n",
            T5_TRAIN_PATH: "def main(): return None\n",
        }
        for relative, content in files.items():
            _write(root / relative, content)
        hashes = {relative: _sha((root / relative).read_bytes()) for relative in files}
        self.policy = CompatibilityPolicy(
            commit=UPSTREAM_COMMIT,
            license_status="NOASSERTION",
            license_file_present=False,
            license_evidence_path="README.md",
            license_evidence_sha256=hashes["README.md"],
            require_clean_checkout=False,
            source_hashes=hashes,
            prompt_profiles={
                profile: {
                    "source_path": path,
                    "source_sha256": INFOCQR_PROMPT_SOURCE_SHA256[profile],
                    "official_function": "generate_rewrite",
                }
                for profile, path in {
                    "rw_zsl": "rewrite_chatgpt_zsl.py",
                    "rw_fsl": "rewrite_chatgpt_inctx.py",
                    "t5_edit": "rewrite_chatgpt_editor.py",
                    "self_edit": "rewrite_chatgpt_self_editor.py",
                }.items()
            },
            known_caches={},
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
        return CompletionResult(self.outputs.pop(0), input_tokens=10, output_tokens=3)


class Dependency:
    def __init__(self, value):
        self.value = value

    def fingerprint(self):
        return dict(self.value)


class RuntimeClient:
    worker_id = "infocqr-fixture"

    def __init__(self, runtime: InfoCQRRuntime) -> None:
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
        timeout_seconds=5,
        max_input_tokens_per_call=128,
        max_output_tokens=256,
        temperature=0,
        input_cost_per_million=1,
        output_cost_per_million=2,
    )


def request(
    fixture: FakeCheckout,
    *,
    profile: str = "rw_zsl",
    operation: str = "live_teacher",
    release_id: str = "rewrite.infocqr.live_teacher_external",
    cache: MountedCache | None = None,
    controls: dict[str, object] | None = None,
):
    calls = (
        2
        if operation == "live_teacher" and profile == "self_edit"
        else (1 if operation == "live_teacher" else 0)
    )
    item = MethodWorkerRequest(
        method_id=METHOD_ID,
        release_id=release_id,
        upstream_commit=UPSTREAM_COMMIT,
        state_hash="a" * 64,
        payload={
            "operation": operation,
            "task_id": "qrecc-1-2",
            "dataset_id": "qrecc",
            "dataset_revision": "qrecc-v1",
            "split": "test",
            "conversation_source": "trec",
            "conversation_no": 1,
            "turn_no": 2,
            "query": "What does it cost?",
            "new_context": ["What is a PA?", "A physician assistant."],
            "prompt_profile": profile,
            "prompt_source_sha256": fixture.policy.prompt_profiles[profile]["source_sha256"],
            "cache_artifact_id": cache.artifact_id if cache else "",
            "cache_artifact_sha256": cache.sha256 if cache else "",
            "allow_first_turn_truth_rewrite": False,
            "source_license_acknowledged": True,
            "retrieval_corpus_id": "qrecc-passages-v1",
            "profile_kind": "artifact_replay" if cache else "controlled_adapter",
            "initial_rewrite": "",
            "initial_rewrite_source_sha256": "",
            "controls": controls
            or {"max_model_calls": calls, "max_tokens": 1000, "max_cost_usd": 1.0},
            "seed": 7,
        },
        dependency_fingerprints={
            "backend": _MODEL_FP if operation == "live_teacher" else {},
            "retriever": _RETRIEVER_FP,
        },
        minimum_reservation={"model_calls": calls},
        protocol_version=PROTOCOL_VERSION,
    )
    return item.as_dict()


class InfoCQRWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.checkout = FakeCheckout(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runtime(self, outputs: list[str], *, cache=None):
        backend = FixtureBackend(outputs)
        runtime = InfoCQRRuntime(
            self.checkout.root,
            backend_config(),
            backend,
            cache=cache,
            source_license_acknowledged=True,
            policy=self.checkout.policy,
        )
        return runtime, backend

    def test_fake_checkout_live_and_host_adapter_call_official_prompt(self) -> None:
        runtime, backend = self.runtime(["What is the cost of becoming a PA?"])
        state = FrozenAgentState(
            query="What does it cost?",
            task_id="qrecc-1-2",
            metadata={
                "conversation_source": "trec",
                "conversation_no": 1,
                "turn_no": 2,
                "new_context": ["What is a PA?", "A physician assistant."],
            },
        )
        adapter = InfoCQRLiveTeacherExternal(
            prompt_profile="rw_zsl",
            source_license_acknowledged=True,
            dataset_revision="qrecc-v1",
            split="test",
            retrieval_corpus_id="qrecc-passages-v1",
            controls=InfoCQRExternalControls(max_model_calls=1),
        )
        patch = adapter.propose(
            state,
            InterventionContext(
                backend=Dependency(_MODEL_FP),
                retriever=Dependency(_RETRIEVER_FP),
                external_runtime=RuntimeClient(runtime),
                seed=7,
            ),
        )
        self.assertEqual(patch.apply(state).query, "What is the cost of becoming a PA?")
        self.assertIn(
            "Context: [Q: What is a PA? A: A physician assistant.]",
            backend.calls[0]["messages"][0]["content"],
        )
        self.assertEqual(patch.trace.counters["model_calls"], 1)

    def test_common_support_serializes_visible_dialogue_and_preserves_official_prompt(self) -> None:
        runtime, backend = self.runtime(["What does a physician assistant cost?"])
        state = FrozenAgentState(
            query="What does it cost?",
            task_id="state-task-7",
            messages=(
                Message("user", "What is a PA?"),
                Message("assistant", "A physician assistant."),
                Message("user", "What does it cost?"),
            ),
            metadata={"benchmark_id": "state_bench.common_support"},
        )
        adapter = InfoCQRLiveTeacherExternal(
            prompt_profile="rw_zsl",
            source_license_acknowledged=True,
            dataset_id="state_bench.common_support",
            dataset_revision="state-fixture-v1",
            split="test",
            retrieval_corpus_id="state-visible-context-v1",
            profile_kind="controlled_adapter",
            release_id="rewrite.infocqr.common_support_rw_zsl",
            controls=InfoCQRExternalControls(max_model_calls=1),
        )
        patch = adapter.propose(
            state,
            InterventionContext(
                backend=Dependency(_MODEL_FP),
                retriever=Dependency(_RETRIEVER_FP),
                external_runtime=RuntimeClient(runtime),
                seed=11,
            ),
        )
        assert patch.apply(state).query == "What does a physician assistant cost?"
        assert patch.release_id == "rewrite.infocqr.common_support_rw_zsl"
        assert patch.profile_kind == "controlled_adapter"
        assert (
            "Context: [Q: What is a PA? A: A physician assistant.]"
            in (backend.calls[0]["messages"][0]["content"])
        )
        event = next(item for item in patch.trace.events if item.name == "dataset_identity")
        assert event.payload["dataset_id"] == "state_bench.common_support"
        assert event.payload["turn_no"] == 2

    def test_backend_config_accepts_bound_fingerprint_and_rejects_reserved_extra_body(self) -> None:
        value = {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key_env": "FIXTURE_KEY",
            "model": "fixture-model",
            "fingerprint": {
                "implementation": "agentinputlab.bound_llm_backend.v1",
                "backend": {"backend": "openai_compatible", "model_id": "fixture-model"},
                "binding": {"kind": "openai_compatible"},
            },
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout_seconds": 5,
            "max_input_tokens_per_call": 128,
            "max_output_tokens": 256,
            "temperature": 0,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        config = BackendConfig.from_json(_canonical(value))
        assert config.model == "fixture-model"
        assert config.extra_body["chat_template_kwargs"]["enable_thinking"] is False

        value["extra_body"] = {"model": "attacker-model"}
        with self.assertRaises(DriverError) as caught:
            BackendConfig.from_json(_canonical(value))
        assert caught.exception.code == "INVALID_BACKEND_CONFIG"

    def test_self_edit_uses_two_seeded_calls_and_pinned_fallback(self) -> None:
        runtime, backend = self.runtime(["initial rewrite?", "No edit needed"])
        response = runtime.run(request(self.checkout, profile="self_edit"))
        self.assertEqual(response["payload"]["rewrite"], "initial rewrite?")
        self.assertEqual([call["seed"] for call in backend.calls], [7, 8])
        self.assertEqual(response["usage"]["model_calls"], 2)
        self.assertIn("edit:non_question_fallback", response["payload"]["postprocess_status"])

    def test_cache_checksum_schema_license_and_gold_gates(self) -> None:
        cache_path = Path(self.temp.name) / "cache.jsonl"
        row = {
            "Conversation_source": "trec",
            "Conversation_no": 1,
            "Turn_no": 2,
            "Question": "What does it cost?",
            "NewContext": ["What is a PA?", "A physician assistant."],
            "GPT_rewrite": {"choices": [{"message": {"content": "What does a PA cost?"}}]},
        }
        _write(cache_path, _canonical(row) + "\n")
        digest = _sha(cache_path.read_bytes())
        cache = MountedCache.load(
            cache_path,
            artifact_id="fixture-cache",
            expected_sha256=digest,
            profile="rw_zsl",
            split="test",
        )
        runtime, backend = self.runtime([], cache=cache)
        response = runtime.run(
            request(
                self.checkout,
                profile="rw_zsl",
                operation="cached_replay",
                release_id="rewrite.infocqr.cached_rw_zsl",
                cache=cache,
            )
        )
        self.assertEqual(response["payload"]["rewrite"], "What does a PA cost?")
        self.assertEqual(response["payload"]["cache_artifact_format"], "jsonl")
        self.assertEqual(response["payload"]["execution_label"], "compatibility_replay")
        self.assertEqual(response["payload"]["official_calls"], [])
        self.assertTrue(response["payload"]["reconstructed_calls"])
        self.assertEqual(response["usage"]["model_calls"], 0)
        self.assertEqual(backend.calls, [])
        state = FrozenAgentState(
            query="What does it cost?",
            task_id="qrecc-1-2",
            metadata={
                "conversation_source": "trec",
                "conversation_no": 1,
                "turn_no": 2,
                "new_context": ["What is a PA?", "A physician assistant."],
            },
        )
        patch = InfoCQRMountedCacheReplay(
            prompt_profile="rw_zsl",
            cache_artifact_id="fixture-cache",
            cache_artifact_sha256=digest,
            source_license_acknowledged=True,
            dataset_revision="qrecc-v1",
            split="test",
            retrieval_corpus_id="qrecc-passages-v1",
        ).propose(
            state,
            InterventionContext(
                retriever=Dependency(_RETRIEVER_FP),
                external_runtime=RuntimeClient(runtime),
            ),
        )
        self.assertEqual(patch.apply(state).query, "What does a PA cost?")
        self.assertEqual(patch.profile_kind, "controlled_adapter")
        self.assertEqual(patch.metadata["implementation_origin"], "compatibility_replay")
        with self.assertRaises(DriverError) as mismatch:
            MountedCache.load(
                cache_path,
                artifact_id="fixture-cache",
                expected_sha256="0" * 64,
                profile="rw_zsl",
                split="test",
            )
        self.assertEqual(mismatch.exception.code, "CACHE_CHECKSUM_MISMATCH")
        with self.assertRaises(DriverError) as startup:
            InfoCQRRuntime(
                self.checkout.root,
                backend_config(),
                FixtureBackend([]),
                source_license_acknowledged=False,
                policy=self.checkout.policy,
            )
        self.assertEqual(startup.exception.code, "SOURCE_LICENSE_NOT_ACKNOWLEDGED")

    def test_identity_injection_and_budget_fail_closed(self) -> None:
        runtime, backend = self.runtime([])
        payload = request(self.checkout)
        payload["payload"]["base_url"] = "https://attacker.invalid/v1"
        payload["request_id"] = _sha(
            _canonical({k: v for k, v in payload.items() if k != "request_id"}).encode()
        )
        with self.assertRaises(DriverError) as injected:
            runtime.run(payload)
        self.assertEqual(injected.exception.code, "INVALID_SCHEMA")
        self.assertEqual(backend.calls, [])

        runtime, backend = self.runtime(["rewrite?"])
        with self.assertRaises(DriverError) as budget:
            runtime.run(
                request(
                    self.checkout,
                    controls={"max_model_calls": 1, "max_tokens": 5, "max_cost_usd": 1.0},
                )
            )
        self.assertEqual(budget.exception.code, "TOKEN_BUDGET_EXCEEDED")
        self.assertEqual(backend.calls, [])

    def test_json_array_cache_and_consumed_failure_usage(self) -> None:
        cache_path = Path(self.temp.name) / "misleading.jsonl"
        row = {
            "Conversation_source": "trec",
            "Conversation_no": 1,
            "Turn_no": 2,
            "Question": "What does it cost?",
            "NewContext": ["What is a PA?", "A physician assistant."],
            "GPT_rewrite": {"choices": [{"message": {"content": "Array rewrite?"}}]},
        }
        _write(cache_path, json.dumps([row], ensure_ascii=False, indent=2))
        cache = MountedCache.load(
            cache_path,
            artifact_id="array-cache",
            expected_sha256=_sha(cache_path.read_bytes()),
            profile="rw_zsl",
            split="test",
        )
        self.assertEqual(cache.artifact_format, "json_array")

        class ConsumedFailureBackend(FixtureBackend):
            def complete(self, **kwargs):
                self.calls.append(dict(kwargs))
                raise DriverError("BACKEND_CALL_FAILED", "provider failed", consumed=True)

        backend = ConsumedFailureBackend([])
        runtime = InfoCQRRuntime(
            self.checkout.root,
            backend_config(),
            backend,
            source_license_acknowledged=True,
            policy=self.checkout.policy,
        )
        output = io.StringIO()
        serve(runtime, io.StringIO(_canonical(request(self.checkout)) + "\n"), output)
        response = json.loads(output.getvalue())
        self.assertEqual(response["failure_code"], "BACKEND_CALL_FAILED")
        self.assertTrue(response["consumed"])
        self.assertEqual(response["usage"]["model_calls"], 1)

    def test_real_checkout_source_and_prompt_smoke_is_opt_in(self) -> None:
        checkout = os.environ.get("AGENTINPUTLAB_INFOCQR_CHECKOUT")
        if not checkout:
            self.skipTest("set AGENTINPUTLAB_INFOCQR_CHECKOUT for real-source smoke")
        policy = load_compatibility_policy()
        root, identity = verify_checkout(Path(checkout), policy)
        self.assertIn(identity, {"source_hash_set", "git_commit_and_source_hash_set"})
        backend = FixtureBackend(["What does a physician assistant program cost?"])
        runtime = InfoCQRRuntime(
            root,
            backend_config(),
            backend,
            source_license_acknowledged=True,
            policy=policy,
        )
        payload = request(self.checkout)
        payload["payload"]["prompt_source_sha256"] = INFOCQR_PROMPT_SOURCE_SHA256["rw_zsl"]
        payload["request_id"] = _sha(
            _canonical({k: v for k, v in payload.items() if k != "request_id"}).encode()
        )
        response = runtime.run(payload)
        self.assertEqual(response["status"], "ok")
        self.assertIn(
            "Given a question and its context", backend.calls[0]["messages"][0]["content"]
        )


if __name__ == "__main__":
    unittest.main()
