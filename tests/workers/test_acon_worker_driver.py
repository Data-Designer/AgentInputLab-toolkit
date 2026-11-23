from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest import mock

from agentinputlab.algorithms.curate.acon import (
    ACON_COMMON_SUPPORT_CONTEXT_RULE,
    ACON_COMMON_SUPPORT_RELEASE,
    ACONHistoryRuntimeExternal,
    ACONObservationCommonSupport,
    ACONObservationRuntimeExternal,
)
from agentinputlab.algorithms.curate.acon.common import (
    ACONArtifactIdentity,
    ACONHistoryControls,
    ACONObservationControls,
    history_text,
    preserved_turn_indices,
)
from agentinputlab.contracts import (
    BudgetExceededError,
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
    Message,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers.method import MethodWorkerRequest
from agentinputlab.observability import BudgetLedger, ObservedExternalRuntime, RunMonitor
from workers.acon.driver import (
    HISTORY_PROTOCOL,
    HISTORY_RELEASE,
    MANIFEST_SCHEMA,
    OBSERVATION_PROTOCOL,
    OBSERVATION_RELEASE,
    UPSTREAM_COMMIT,
    ACONDriver,
    DriverError,
    OpenAICompatibleLLM,
    _NoRedirectHandler,
    serve_jsonl,
)

_BASE = """
class FixtureTokenizer:
    def encode(self, text):
        return text.split()

    def agentinputlab_fingerprint(self):
        return {"engine": "fixture", "revision": "fixture-tokenizer@revision"}

class BaseContextOptimizer:
    def __init_tokenizer(self):
        self.tokenizer = FixtureTokenizer()

    def count_tokens(self, text):
        if not hasattr(self, "tokenizer"):
            self.__init_tokenizer()
        return len(self.tokenizer.encode(text))

    def parse_output(self, response, keyword):
        if keyword in response:
            return response[response.index(keyword) + len(keyword):].strip()
        return response
"""


def _is_git_checkout(root: Path) -> bool:
    if not root.is_dir():
        return False
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "--verify", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


_HISTORY = """
print("fixture upstream import diagnostic")
from .base import BaseContextOptimizer

class HistoryOptimizer(BaseContextOptimizer):
    def __init__(self, config, prompt_dir=None, debug_mode=True, llm=None):
        assert prompt_dir is None
        assert debug_mode is False
        assert config["use_llmlingua"] is False
        assert config["prompts"]["prompt_history_user"] == "history_prompt"
        self.config = config
        self.llm = llm
        self.system_message = "fixture history system"
        self.history = []

    def check_summarization_needed(self, history_text, prev_history_summary=None):
        if self.config["history_summarization_threshold"] == -1:
            return True
        material = history_text
        if prev_history_summary:
            material = prev_history_summary + "\\n" + history_text
        return self.count_tokens(material) > self.config["history_summarization_threshold"]

    def process(self, task, history, prev_history_summary=None, raw_history=None,
                opt_args=None, **kwargs):
        assert kwargs == {}
        marker = (
            "HISTORY_PROCESS|task=" + task
            + "|history=" + history
            + "|previous=" + (prev_history_summary or "")
            + "|raw_count=" + str(len(raw_history))
            + "|opt_args=" + repr(opt_args)
        )
        raw_response = self.llm.generate(
            marker, temperature=self.config["temperature"]
        ).strip()
        self.history.append((self.system_message, marker, raw_response, {"task": task}))
        return self.parse_output(raw_response, "# History Summary")
"""

_OBSERVATION = """
from .base import BaseContextOptimizer

class ObservationOptimizer(BaseContextOptimizer):
    def __init__(self, config, prompt_dir=None, debug_mode=False, llm=None):
        assert prompt_dir is None
        assert debug_mode is False
        assert config["use_llmlingua"] is False
        assert config["prompts"]["prompt_user"] == "observation_prompt"
        self.config = config
        self.llm = llm
        self.system_message = "fixture observation system"
        self.history = []

    def check_summarization_needed(self, observation):
        if self.config["obs_summarization_threshold"] == -1:
            return True
        return self.count_tokens(observation) > self.config["obs_summarization_threshold"]

    def process(self, task, observation, history, raw_history, opt_args, **kwargs):
        assert kwargs == {}
        marker = (
            "OBSERVATION_PROCESS|task=" + task
            + "|observation=" + observation
            + "|history=" + history
            + "|raw_count=" + str(len(raw_history))
            + "|opt_args=" + repr(opt_args)
        )
        raw_response = self.llm.generate(marker).strip()
        self.history.append((self.system_message, marker, raw_response, opt_args))
        return self.parse_output(raw_response, "# Refined Observation")
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
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


class FixtureLLM:
    def __init__(self, model: str, *, fail: bool = False) -> None:
        self.model = model
        self.fail = fail
        self.system_message = ""
        self.total_model_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.prompts: list[str] = []

    def fingerprint(self) -> dict[str, str]:
        return {"transport": "fixture", "model": self.model}

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        self.prompts.append(prompt)
        self.total_model_calls += 1
        self.total_input_tokens += len(prompt.split())
        if self.fail:
            raise RuntimeError("fixture optimizer failure")
        if prompt.startswith("HISTORY_PROCESS|"):
            self.total_output_tokens += 2
            self.history_temperature = temperature
            return "# History Summary\nfixture-summary"
        if prompt.startswith("OBSERVATION_PROCESS|"):
            self.total_output_tokens += 2
            return "# Refined Observation\nfixture-observation"
        raise AssertionError("unexpected fixture prompt")


class FakeCheckout:
    def __init__(self, parent: Path, *, missing_dependency: bool = False) -> None:
        self.root = parent / "acon"
        self.root.mkdir()
        _write(self.root / ".git" / "HEAD", UPSTREAM_COMMIT + "\n")
        _write(self.root / "LICENSE", "MIT fixture license\n")
        _write(self.root / "src" / "productive_agents" / "__init__.py", "")
        _write(self.root / "src" / "productive_agents" / "ctxopt" / "__init__.py", "")
        _write(self.root / "src" / "productive_agents" / "ctxopt" / "base.py", _BASE)
        history_source = _HISTORY
        if missing_dependency:
            history_source = "import fixture_missing_acon_dependency\n" + history_source
        _write(
            self.root / "src" / "productive_agents" / "ctxopt" / "history_optimizer.py",
            history_source,
        )
        _write(
            self.root / "src" / "productive_agents" / "ctxopt" / "obs_optimizer.py",
            _OBSERVATION,
        )
        prompt_root = self.root / "prompts"
        _write(prompt_root / "system_prompt.jinja", "fixture system {{ task }}\n")
        _write(prompt_root / "history_prompt.jinja", "fixture history {{ history }}\n")
        _write(
            prompt_root / "observation_prompt.jinja",
            "fixture observation {{ observation }}\n",
        )
        history_files = self._prompt_files("system_prompt.jinja", "history_prompt.jinja")
        observation_files = self._prompt_files("system_prompt.jinja", "observation_prompt.jinja")
        self.history_prompt_hash = _sha256(_canonical_json(history_files).encode("utf-8"))
        self.history_guideline_hash = history_files["prompts/history_prompt.jinja"]
        self.observation_prompt_hash = _sha256(_canonical_json(observation_files).encode("utf-8"))
        self.observation_guideline_hash = observation_files["prompts/observation_prompt.jinja"]
        source_files = {
            "src/productive_agents/ctxopt/base.py": _sha256(
                (self.root / "src/productive_agents/ctxopt/base.py").read_bytes()
            ),
            "src/productive_agents/ctxopt/history_optimizer.py": _sha256(
                (self.root / "src/productive_agents/ctxopt/history_optimizer.py").read_bytes()
            ),
            "src/productive_agents/ctxopt/obs_optimizer.py": _sha256(
                (self.root / "src/productive_agents/ctxopt/obs_optimizer.py").read_bytes()
            ),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "upstream": {
                "repository": "https://example.invalid/acon-fixture",
                "commit": UPSTREAM_COMMIT,
                "license_spdx": "MIT",
                "license_sha256": _sha256((self.root / "LICENSE").read_bytes()),
                "minimum_python": [3, 10],
                "require_clean_checkout": False,
            },
            "source_files": source_files,
            "prompt_profiles": {
                "fixture.history.v1": {
                    "mode": "history",
                    "prompt_dir": "prompts",
                    "system_template": "system_prompt",
                    "user_template": "history_prompt",
                    "prompt_files": history_files,
                    "prompt_artifact_hash": self.history_prompt_hash,
                    "guideline_artifact_hash": self.history_guideline_hash,
                },
                "fixture.observation.v1": {
                    "mode": "observation",
                    "prompt_dir": "prompts",
                    "system_template": "system_prompt",
                    "user_template": "observation_prompt",
                    "prompt_files": observation_files,
                    "prompt_artifact_hash": self.observation_prompt_hash,
                    "guideline_artifact_hash": self.observation_guideline_hash,
                },
            },
        }
        self.manifest_path = parent / "acon-manifest.json"
        _write(self.manifest_path, json.dumps(manifest))
        self.llms: list[FixtureLLM] = []

    def _prompt_files(self, *names: str) -> dict[str, str]:
        return {
            f"prompts/{name}": _sha256((self.root / "prompts" / name).read_bytes())
            for name in names
        }

    def artifacts(self, mode: str) -> ACONArtifactIdentity:
        if mode == "history":
            prompt_hash = self.history_prompt_hash
            guideline_hash = self.history_guideline_hash
        else:
            prompt_hash = self.observation_prompt_hash
            guideline_hash = self.observation_guideline_hash
        return ACONArtifactIdentity(
            optimizer_model_revision="fixture-optimizer@revision",
            target_model_revision="fixture-target@revision",
            tokenizer_revision="fixture-tokenizer@revision",
            prompt_artifact_hash=prompt_hash,
            guideline_artifact_hash=guideline_hash,
            optimizer_deployment_fingerprint_hash=content_hash(
                {"transport": "fixture", "model": "fixture-optimizer@revision"}
            ),
            target_backend_fingerprint_hash=content_hash(
                {"backend": "fixture", "model_revision": "fixture-target@revision"}
            ),
            tokenizer_fingerprint_hash=content_hash(
                {"engine": "fixture", "revision": "fixture-tokenizer@revision"}
            ),
        )

    def driver(self, *, with_llm: bool = True, fail_model: bool = False) -> ACONDriver:
        self.fail_model = fail_model
        factory = self._llm_factory if with_llm else None
        return ACONDriver(
            manifest_path=self.manifest_path,
            environment={
                "AGENTINPUTLAB_UPSTREAM_ROOT": str(self.root),
                "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL_REVISION": ("fixture-optimizer@revision"),
                "AGENTINPUTLAB_ACON_TARGET_MODEL_REVISION": "fixture-target@revision",
                "AGENTINPUTLAB_ACON_TOKENIZER_REVISION": "fixture-tokenizer@revision",
            },
            llm_factory=factory,
        )

    def _llm_factory(self, **kwargs: Any) -> FixtureLLM:
        self.factory_kwargs = kwargs
        llm = FixtureLLM(kwargs["model"], fail=getattr(self, "fail_model", False))
        self.llms.append(llm)
        return llm


def _messages() -> tuple[Message, ...]:
    return (
        Message(role="system", content="system"),
        Message(role="user", content="initial task"),
        Message(role="assistant", content="old action"),
        Message(role="user", content="old observation"),
        Message(role="assistant", content="recent action"),
        Message(role="user", content="recent observation"),
    )


def _history_request(
    checkout: FakeCheckout,
    *,
    threshold: int = -1,
    interval: int = -1,
    previous: str = "previous summary",
    artifacts: ACONArtifactIdentity | None = None,
    phase: str = "preflight",
    decision_token: str = "",
) -> dict[str, Any]:
    messages = _messages()
    controls = ACONHistoryControls(
        summarization_threshold=threshold,
        preserve_last_k_turns=1,
        summary_interval=interval,
    )
    indices = preserved_turn_indices(messages, controls.preserve_last_k_turns)
    preserved = tuple(messages[index] for index in indices)
    start = len(messages)
    if preserved:
        first = preserved[0]
        for index, message in enumerate(messages):
            if message.role == first.role and message.content == first.content:
                start = index
                break
    material = history_text(messages[:start] if start > 2 else ())
    counted = f"{previous}\n{material}" if previous else material
    identity = artifacts or checkout.artifacts("history")
    request = MethodWorkerRequest(
        method_id="acon",
        release_id=HISTORY_RELEASE,
        upstream_commit=UPSTREAM_COMMIT,
        state_hash=content_hash("fixture-history-state"),
        protocol_version=HISTORY_PROTOCOL,
        payload={
            "task": "fixture task",
            "raw_messages": messages,
            "history_text": material,
            "history_text_hash": content_hash(material),
            "counted_material_hash": content_hash(counted),
            "previous_history_summary": previous,
            "preserved_message_indices": indices,
            "preserved_messages": preserved,
            "controls": asdict(controls),
            "artifacts": identity.as_payload(),
            "seed": 17,
            "phase": phase,
            "decision_token": decision_token,
        },
        dependency_fingerprints={
            "target_backend": {
                "backend": "fixture",
                "model_revision": "fixture-target@revision",
            }
        },
        minimum_reservation={"model_calls": 1} if phase == "execute" else {},
    )
    return request.as_dict()


def _observation_request(
    checkout: FakeCheckout,
    *,
    threshold: int = -1,
    phase: str = "preflight",
    decision_token: str = "",
) -> dict[str, Any]:
    messages = _messages()
    controls = ACONObservationControls(summarization_threshold=threshold, context_index=0)
    current_history = history_text(messages)
    opt_args = {"current_app": "mail", "required_reference": "row-4"}
    request = MethodWorkerRequest(
        method_id="acon",
        release_id=OBSERVATION_RELEASE,
        upstream_commit=UPSTREAM_COMMIT,
        state_hash=content_hash("fixture-observation-state"),
        protocol_version=OBSERVATION_PROTOCOL,
        payload={
            "task": "fixture task",
            "observation": "large fixture observation with row four",
            "source_context_id": "tool-observation-1",
            "history_text": current_history,
            "history_text_hash": content_hash(current_history),
            "raw_messages": messages,
            "opt_args": opt_args,
            "opt_args_hash": content_hash(opt_args),
            "controls": asdict(controls),
            "artifacts": checkout.artifacts("observation").as_payload(),
            "seed": 23,
            "phase": phase,
            "decision_token": decision_token,
        },
        dependency_fingerprints={
            "target_backend": {
                "backend": "fixture",
                "model_revision": "fixture-target@revision",
            }
        },
        minimum_reservation={"model_calls": 1} if phase == "execute" else {},
    )
    return request.as_dict()


def _run_two_phase(driver: ACONDriver, request: dict[str, Any]) -> dict[str, Any]:
    preflight = driver.run(request)
    if not preflight["payload"]["applied"]:
        return preflight
    payload = dict(request["payload"])
    payload.update(
        phase="execute",
        decision_token=preflight["payload"]["decision_token"],
    )
    execute = MethodWorkerRequest(
        method_id=request["method_id"],
        release_id=request["release_id"],
        upstream_commit=request["upstream_commit"],
        state_hash=request["state_hash"],
        payload=payload,
        dependency_fingerprints=request["dependency_fingerprints"],
        minimum_reservation={"model_calls": 1},
        protocol_version=request["protocol_version"],
    )
    return driver.run(execute.as_dict())


class FixtureTargetBackend:
    def fingerprint(self) -> dict[str, str]:
        return {"backend": "fixture", "model_revision": "fixture-target@revision"}


class InProcessWorker:
    worker_id = "fixture.acon.in-process"

    def __init__(self, driver: ACONDriver) -> None:
        self.driver = driver
        self.calls = 0

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        source = io.StringIO(_canonical_json(payload) + "\n")
        sink = io.StringIO()
        serve_jsonl(self.driver, source, sink)
        lines = sink.getvalue().splitlines()
        if len(lines) != 1:  # pragma: no cover - protocol assertion
            raise AssertionError("fixture JSONL worker emitted the wrong line count")
        value = json.loads(lines[0])
        if not isinstance(value, dict):  # pragma: no cover - protocol assertion
            raise AssertionError("fixture JSONL worker emitted a non-object")
        return value

    def fingerprint(self) -> dict[str, str]:
        return {
            "worker_id": self.worker_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol": "agentinputlab.jsonl.v1",
        }


class ACONWorkerDriverTests(unittest.TestCase):
    def test_history_calls_upstream_process_and_preserves_raw_parser_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            response = _run_two_phase(checkout.driver(), _history_request(checkout))

        payload = response["payload"]
        self.assertEqual(response["status"], "ok")
        self.assertTrue(payload["applied"])
        self.assertEqual(payload["profile_id"], "fixture.history.v1")
        self.assertEqual(payload["summary"], "fixture-summary")
        self.assertEqual(payload["raw_response"], "# History Summary\nfixture-summary")
        self.assertEqual(response["usage"]["model_calls"], 1)
        self.assertGreater(response["usage"]["input_tokens"], 0)
        self.assertIn("HISTORY_PROCESS|task=fixture task", checkout.llms[-1].prompts[0])
        self.assertIn("|previous=previous summary", checkout.llms[-1].prompts[0])
        self.assertIn("|raw_count=6|opt_args=None", checkout.llms[-1].prompts[0])
        self.assertEqual(checkout.llms[-1].system_message, "fixture history system")
        self.assertEqual(checkout.factory_kwargs["model"], "fixture-optimizer@revision")
        self.assertEqual(checkout.factory_kwargs["seed"], 17)

    def test_observation_calls_upstream_process_with_exact_source_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            response = _run_two_phase(checkout.driver(), _observation_request(checkout))

        payload = response["payload"]
        self.assertTrue(payload["applied"])
        self.assertEqual(payload["profile_id"], "fixture.observation.v1")
        self.assertEqual(payload["optimized_observation"], "fixture-observation")
        self.assertEqual(
            payload["raw_response"],
            "# Refined Observation\nfixture-observation",
        )
        prompt = checkout.llms[-1].prompts[0]
        self.assertIn("OBSERVATION_PROCESS|task=fixture task", prompt)
        self.assertIn("|observation=large fixture observation with row four", prompt)
        self.assertIn("'required_reference': 'row-4'", prompt)
        self.assertEqual(checkout.llms[-1].system_message, "fixture observation system")

    def test_strict_threshold_equality_skips_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            history_request = _history_request(checkout)
            history_payload = history_request["payload"]
            counted = (
                history_payload["previous_history_summary"] + "\n" + history_payload["history_text"]
            )
            history_response = _run_two_phase(
                checkout.driver(), _history_request(checkout, threshold=len(counted.split()))
            )
            observation_request = _observation_request(checkout)
            observation_tokens = len(observation_request["payload"]["observation"].split())
            observation_response = _run_two_phase(
                checkout.driver(), _observation_request(checkout, threshold=observation_tokens)
            )

        self.assertFalse(history_response["payload"]["threshold_triggered"])
        self.assertFalse(history_response["payload"]["applied"])
        self.assertEqual(history_response["payload"]["summary"], "")
        self.assertEqual(history_response["usage"]["model_calls"], 0)
        self.assertFalse(observation_response["payload"]["threshold_triggered"])
        self.assertEqual(
            observation_response["payload"]["optimized_observation"],
            observation_request["payload"]["observation"],
        )
        self.assertEqual(observation_response["usage"]["model_calls"], 0)
        self.assertEqual(sum(len(llm.prompts) for llm in checkout.llms), 0)

    def test_both_bridges_propose_end_to_end_through_upstream_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            worker = InProcessWorker(checkout.driver())
            messages = _messages()
            history_state = FrozenAgentState(
                query="fixture task",
                task_id="history-case",
                messages=messages,
                metadata={"acon_previous_history_summary": "previous summary"},
            )
            history_patch = ACONHistoryRuntimeExternal(
                client=worker,
                artifacts=checkout.artifacts("history"),
            ).propose(
                history_state,
                InterventionContext(backend=FixtureTargetBackend(), seed=17),
            )
            history_output = history_patch.apply(history_state)

            observation_state = FrozenAgentState(
                query="fixture task",
                task_id="observation-case",
                messages=messages,
                context=(
                    ContextBlock(
                        block_id="tool-observation-1",
                        text="large fixture observation with row four",
                        source="fixture-tool",
                    ),
                ),
            )
            observation_patch = ACONObservationRuntimeExternal(
                client=worker,
                artifacts=checkout.artifacts("observation"),
                controls=ACONObservationControls(context_index=0),
            ).propose(
                observation_state,
                InterventionContext(
                    backend=FixtureTargetBackend(),
                    seed=23,
                    run_config={
                        "acon_opt_args": {
                            "current_app": "mail",
                            "required_reference": "row-4",
                        }
                    },
                ),
            )
            observation_output = observation_patch.apply(observation_state)

        self.assertIn("<HISTORY_SUMMARY>\nfixture-summary", history_output.messages[1].content)
        self.assertEqual(history_output.messages[-2:], messages[-2:])
        self.assertEqual(observation_output.context[0].block_id, "tool-observation-1")
        self.assertEqual(observation_output.context[0].source, "fixture-tool")
        self.assertEqual(observation_output.context[0].text, "fixture-observation")
        self.assertTrue(observation_output.context[0].metadata["acon_applied"])
        self.assertEqual(history_patch.trace.counters["model_calls"], 1)
        self.assertEqual(observation_patch.trace.counters["model_calls"], 1)

    def test_common_support_freezes_block_zero_and_preserves_other_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            worker = InProcessWorker(checkout.driver())
            state = FrozenAgentState(
                query="fixture task",
                task_id="common-support-case",
                messages=_messages(),
                context=(
                    ContextBlock(
                        block_id="policy",
                        text="large fixture observation with row four",
                        source="fixture-policy",
                    ),
                    ContextBlock(
                        block_id="tool-catalog",
                        text="unchanged tool definitions",
                        source="fixture-tools",
                    ),
                ),
            )
            patch = ACONObservationCommonSupport(
                client=worker,
                artifacts=checkout.artifacts("observation"),
            ).propose(
                state,
                InterventionContext(backend=FixtureTargetBackend(), seed=23),
            )
            output = patch.apply(state)

        self.assertEqual(patch.release_id, ACON_COMMON_SUPPORT_RELEASE)
        self.assertEqual(output.context[0].text, "fixture-observation")
        self.assertEqual(output.context[1], state.context[1])
        self.assertEqual(patch.trace.counters["selected_context_blocks"], 1)
        self.assertEqual(
            patch.metadata["common_support_context_rule"],
            ACON_COMMON_SUPPORT_CONTEXT_RULE,
        )
        self.assertFalse(patch.metadata["paper_result_parity_claimed"])

    def test_history_bridge_persists_summary_and_original_prompt_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            worker = InProcessWorker(checkout.driver())
            adapter = ACONHistoryRuntimeExternal(
                client=worker,
                artifacts=checkout.artifacts("history"),
            )
            state = FrozenAgentState(query="fixture task", messages=_messages())
            context = InterventionContext(backend=FixtureTargetBackend(), seed=17)

            after_first = adapter.propose(state, context).apply(state)
            second_state = replace(
                after_first,
                messages=(
                    *after_first.messages,
                    Message("assistant", "second action"),
                    Message("user", "second observation"),
                ),
            )
            after_second = adapter.propose(second_state, context).apply(second_state)

        self.assertEqual(after_first.metadata["acon_original_first_user_prompt"], "initial task")
        self.assertEqual(after_first.metadata["acon_previous_history_summary"], "fixture-summary")
        self.assertEqual(after_second.metadata["acon_original_first_user_prompt"], "initial task")
        self.assertIn("|previous=fixture-summary", checkout.llms[-1].prompts[0])

    def test_execute_reservation_rejects_before_optimizer_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            inner = InProcessWorker(checkout.driver())
            monitor = RunMonitor()
            ledger = BudgetLedger(ExecutionBudget(max_model_calls=0))
            state = FrozenAgentState(query="fixture task", messages=_messages())
            with monitor.run_scope(
                release_id=HISTORY_RELEASE,
                state_hash=state.state_hash,
            ) as run_id:
                observed = ObservedExternalRuntime(
                    inner=inner,
                    monitor=monitor,
                    ledger=ledger,
                    run_id=run_id,
                )
                adapter = ACONHistoryRuntimeExternal(
                    client=observed,
                    artifacts=checkout.artifacts("history"),
                )
                with self.assertRaises(BudgetExceededError):
                    adapter.propose(
                        state,
                        InterventionContext(backend=FixtureTargetBackend(), seed=17),
                    )

        self.assertEqual(inner.calls, 1)
        self.assertEqual(sum(len(llm.prompts) for llm in checkout.llms), 0)
        self.assertEqual(ledger.usage.model_calls, 0)

    def test_reserved_opt_args_are_rejected_before_worker_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            worker = InProcessWorker(checkout.driver())
            state = FrozenAgentState(
                query="fixture task",
                messages=_messages(),
                context=(ContextBlock("observation", block_id="obs"),),
            )
            adapter = ACONObservationRuntimeExternal(
                client=worker,
                artifacts=checkout.artifacts("observation"),
            )
            with self.assertRaisesRegex(ValidationError, "reserved upstream inputs"):
                adapter.propose(
                    state,
                    InterventionContext(
                        backend=FixtureTargetBackend(),
                        run_config={"acon_opt_args": {"task": "collision"}},
                    ),
                )
        self.assertEqual(worker.calls, 0)

    def test_optimizer_transport_refuses_key_over_http_and_all_redirects(self) -> None:
        with self.assertRaises(DriverError) as insecure:
            OpenAICompatibleLLM(
                base_url="http://127.0.0.1:8000/v1",
                api_key="secret",
                model="fixture-model",
                seed=0,
                default_temperature=0.0,
                timeout_seconds=1.0,
                max_tokens=None,
            )
        self.assertEqual(insecure.exception.code, "MODEL_CONFIGURATION_INVALID")
        with self.assertRaises(DriverError) as redirect:
            _NoRedirectHandler().redirect_request(None, None, 307, "", {}, "https://other")
        self.assertEqual(redirect.exception.code, "MODEL_REDIRECT_REFUSED")

    def test_optimizer_transport_sends_and_fingerprints_public_extra_body(self) -> None:
        captured: dict[str, Any] = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(
                    {
                        "model": "fixture-model",
                        "choices": [{"message": {"content": "# Refined Observation\nshort"}}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                    }
                ).encode()

        class Opener:
            def open(self, request, timeout):
                captured["body"] = json.loads(request.data)
                captured["timeout"] = timeout
                return Response()

        llm = OpenAICompatibleLLM(
            base_url="http://127.0.0.1:8000/v1",
            api_key="",
            model="fixture-model",
            seed=17,
            default_temperature=0.0,
            timeout_seconds=3.0,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        llm._opener = Opener()

        self.assertEqual(llm.generate("fixture prompt"), "# Refined Observation\nshort")
        self.assertEqual(captured["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(captured["body"]["seed"], 17)
        self.assertEqual(captured["timeout"], 3.0)
        self.assertEqual(
            llm.fingerprint()["extra_body_hash"],
            content_hash({"chat_template_kwargs": {"enable_thinking": False}}),
        )

    def test_invalid_final_content_retains_provider_token_usage(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return json.dumps(
                    {
                        "model": "fixture-model",
                        "choices": [{"message": {"content": None}}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 512},
                    }
                ).encode()

        llm = OpenAICompatibleLLM(
            base_url="http://127.0.0.1:8000/v1",
            api_key="",
            model="fixture-model",
            seed=17,
            default_temperature=0,
            timeout_seconds=1,
            max_tokens=512,
        )
        llm._opener = mock.Mock(open=mock.Mock(return_value=Response()))
        with self.assertRaises(DriverError):
            llm.generate("actual failed-output fixture")
        self.assertEqual(
            (llm.total_model_calls, llm.total_input_tokens, llm.total_output_tokens), (1, 7, 512)
        )
        with self.assertRaises(DriverError) as reserved:
            OpenAICompatibleLLM(
                base_url="http://127.0.0.1:8000/v1",
                api_key="",
                model="fixture-model",
                seed=17,
                default_temperature=0.0,
                timeout_seconds=3.0,
                max_tokens=64,
                extra_body={"model": "shadow-model"},
            )
        self.assertEqual(reserved.exception.code, "MODEL_CONFIGURATION_INVALID")

    def test_worker_rejects_reserved_collision_and_stale_decision_before_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            driver = checkout.driver()

            observation = _observation_request(checkout)
            collided_payload = dict(observation["payload"])
            collided_payload["opt_args"] = {"task": "override"}
            collided_payload["opt_args_hash"] = content_hash(collided_payload["opt_args"])
            collided = MethodWorkerRequest(
                method_id="acon",
                release_id=OBSERVATION_RELEASE,
                upstream_commit=UPSTREAM_COMMIT,
                state_hash=observation["state_hash"],
                payload=collided_payload,
                dependency_fingerprints=observation["dependency_fingerprints"],
                protocol_version=OBSERVATION_PROTOCOL,
            )
            with self.assertRaises(DriverError) as reserved:
                driver.run(collided.as_dict())
            self.assertEqual(reserved.exception.code, "INVALID_REQUEST")

            history = _history_request(checkout)
            preflight = driver.run(history)
            stale_payload = dict(history["payload"])
            stale_payload.update(
                task="changed after preflight",
                phase="execute",
                decision_token=preflight["payload"]["decision_token"],
            )
            stale = MethodWorkerRequest(
                method_id="acon",
                release_id=HISTORY_RELEASE,
                upstream_commit=UPSTREAM_COMMIT,
                state_hash=history["state_hash"],
                payload=stale_payload,
                dependency_fingerprints=history["dependency_fingerprints"],
                minimum_reservation={"model_calls": 1},
                protocol_version=HISTORY_PROTOCOL,
            )
            with self.assertRaises(DriverError) as stale_error:
                driver.run(stale.as_dict())
            self.assertEqual(stale_error.exception.code, "DECISION_TOKEN_MISMATCH")

        self.assertEqual(sum(len(llm.prompts) for llm in checkout.llms), 0)

    def test_failed_optimizer_call_is_reported_as_consumed_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            driver = checkout.driver(fail_model=True)
            request = _history_request(checkout)
            preflight = driver.run(request)
            execute_payload = dict(request["payload"])
            execute_payload.update(
                phase="execute",
                decision_token=preflight["payload"]["decision_token"],
            )
            execute = MethodWorkerRequest(
                method_id="acon",
                release_id=HISTORY_RELEASE,
                upstream_commit=UPSTREAM_COMMIT,
                state_hash=request["state_hash"],
                payload=execute_payload,
                dependency_fingerprints=request["dependency_fingerprints"],
                minimum_reservation={"model_calls": 1},
                protocol_version=HISTORY_PROTOCOL,
            )
            source = io.StringIO(_canonical_json(execute.as_dict()) + "\n")
            sink = io.StringIO()
            serve_jsonl(driver, source, sink)
            failure = json.loads(sink.getvalue())

        self.assertEqual(failure["status"], "error")
        self.assertTrue(failure["consumed"])
        self.assertEqual(failure["usage"]["model_calls"], 1)
        self.assertGreater(failure["usage"]["input_tokens"], 0)

    def test_checkout_commit_license_source_and_prompt_tampering_fail_closed(self) -> None:
        cases = (
            (".git/HEAD", "0" * 40 + "\n", "UPSTREAM_REVISION_MISMATCH"),
            ("LICENSE", "tampered license\n", "UPSTREAM_LICENSE_MISMATCH"),
            (
                "src/productive_agents/ctxopt/history_optimizer.py",
                "tampered source\n",
                "UPSTREAM_SOURCE_MISMATCH",
            ),
            ("prompts/history_prompt.jinja", "tampered prompt\n", "UPSTREAM_PROMPT_MISMATCH"),
        )
        for relative, replacement, code in cases:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                checkout = FakeCheckout(Path(directory))
                _write(checkout.root / relative, replacement)
                with self.assertRaises(DriverError) as raised:
                    checkout.driver().run(_history_request(checkout))
                self.assertEqual(raised.exception.code, code)

    def test_prompt_identity_and_source_mapping_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            wrong_artifacts = checkout.artifacts("observation")
            with self.assertRaises(DriverError) as raised:
                checkout.driver().run(_history_request(checkout, artifacts=wrong_artifacts))
            self.assertEqual(raised.exception.code, "PROMPT_ARTIFACT_MISMATCH")

            request = _history_request(checkout)
            request["payload"]["history_text"] += " tampered"
            identity = {key: value for key, value in request.items() if key != "request_id"}
            request["request_id"] = content_hash(identity)
            with self.assertRaises(DriverError) as raised:
                checkout.driver().run(request)
            self.assertEqual(raised.exception.code, "SOURCE_MAPPING_MISMATCH")

    def test_jsonl_is_one_line_per_request_and_upstream_stdout_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            source = io.StringIO(_canonical_json(_history_request(checkout)) + "\nnot-json\n")
            sink = io.StringIO()
            diagnostics = io.StringIO()
            with mock.patch("sys.stderr", diagnostics):
                self.assertEqual(serve_jsonl(checkout.driver(), source, sink), 0)

        lines = sink.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["status"], "ok")
        self.assertEqual(json.loads(lines[1])["failure_code"], "INVALID_JSON")
        self.assertNotIn("fixture upstream import diagnostic", lines[0])
        self.assertIn("fixture upstream import diagnostic", diagnostics.getvalue())

    def test_missing_upstream_dependency_has_specific_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory), missing_dependency=True)
            with self.assertRaises(DriverError) as raised:
                checkout.driver().run(_history_request(checkout))
        self.assertEqual(raised.exception.code, "UPSTREAM_DEPENDENCY_MISSING")
        self.assertIn("fixture_missing_acon_dependency", raised.exception.public_message)

    def test_triggered_branch_requires_explicit_model_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            with self.assertRaises(DriverError) as raised:
                checkout.driver(with_llm=False).run(_history_request(checkout))
        self.assertEqual(raised.exception.code, "MODEL_CONFIGURATION_MISSING")

    def test_request_identity_and_runtime_revision_bindings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            request = _history_request(checkout)
            request["request_id"] = "0" * 64
            with self.assertRaises(DriverError) as raised:
                checkout.driver().run(request)
            self.assertEqual(raised.exception.code, "REQUEST_ID_MISMATCH")

            driver = ACONDriver(
                manifest_path=checkout.manifest_path,
                environment={
                    "AGENTINPUTLAB_ACON_CHECKOUT": str(checkout.root),
                    "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL_REVISION": ("fixture-optimizer@revision"),
                    "AGENTINPUTLAB_ACON_TARGET_MODEL_REVISION": "different-target",
                    "AGENTINPUTLAB_ACON_TOKENIZER_REVISION": "fixture-tokenizer@revision",
                },
                llm_factory=checkout._llm_factory,
            )
            with self.assertRaises(DriverError) as raised:
                driver.run(_history_request(checkout))
            self.assertEqual(raised.exception.code, "ARTIFACT_REVISION_MISMATCH")

    def test_real_pinned_checkout_imports_core_or_reports_optional_dependency(self) -> None:
        upstream = Path(os.environ.get("AGENTINPUTLAB_ACON_CHECKOUT", "/private/tmp/agentinputlab-upstreams/acon"))
        if not _is_git_checkout(upstream):
            self.skipTest("pinned ACON checkout is not present")
        driver = ACONDriver(environment={"AGENTINPUTLAB_ACON_CHECKOUT": str(upstream)})
        try:
            result = driver.smoke_upstream_core()
        except DriverError as exc:
            if exc.code == "UPSTREAM_DEPENDENCY_MISSING":
                self.skipTest(exc.public_message)
            raise
        self.assertEqual(result["upstream_commit"], UPSTREAM_COMMIT)
        self.assertEqual(
            result["history_symbol"],
            "productive_agents.ctxopt.history_optimizer.HistoryOptimizer",
        )
        self.assertEqual(
            result["observation_symbol"],
            "productive_agents.ctxopt.obs_optimizer.ObservationOptimizer",
        )


if __name__ == "__main__":
    unittest.main()
