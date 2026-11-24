from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agentinputlab.algorithms.decompose.decomp import DecompFullBridge
from agentinputlab.algorithms.decompose.decomp.worker_bridge import (
    DECOMP_REVISION,
    DECOMP_WORKER_SCHEMA,
    DecompFrontierEvent,
    DecompGraphNode,
    DecompParticipantCall,
    DecompWorkerResponse,
    decomp_context_hash,
    validate_decomp_full_response,
)
from agentinputlab.contracts import ContextBlock, FrozenAgentState, InterventionContext
from agentinputlab.runner import CompositeInterventionRunner
from workers.decomp.driver import (
    MANIFEST_SCHEMA,
    PROFILE_ID,
    REQUEST_SCHEMA,
    UPSTREAM_COMMIT,
    DecompDriver,
    DriverError,
    _validate_routed_instruction,
    serve_jsonl,
)
from workers.decomp.runtime import PinnedDecompRuntime

_MODEL_SEARCH = """
import copy

class SearchState:
    def __init__(self, json_data, command, score=0.0):
        self._data = json_data
        self._next = command
        self._score = score
    def copy(self):
        return SearchState(copy.deepcopy(self._data), self._next, self._score)
    def __lt__(self, other):
        return self._score < other._score
    @property
    def data(self):
        return self._data
    @property
    def score(self):
        return self._score
    @property
    def next(self):
        return self._next
    @next.setter
    def next(self, value):
        self._next = value

class BestFirstDecomposer:
    def __init__(self, controller):
        self.controller = controller
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


_CONFIGURABLE = """
import json
from decomp.inference.model_search import BestFirstDecomposer

class Data:
    def __init__(self, value):
        self.value = dict(value)
        self.questions = []
        self.answers = []
    def has_tasks(self):
        return False
    def get_last_question(self):
        if not self.questions:
            raise ValueError("no generated question")
        return self.questions[-1]
    def get_last_answer(self):
        if not self.answers:
            raise ValueError("no answer")
        return self.answers[-1]
    def get_printable_reasoning_chain(self):
        return "Q: " + self.questions[-1] + "\\nA: " + self.answers[-1]

class Model:
    def __init__(self):
        self.calls = 0
    def return_model_calls(self):
        return {"fake_llm": self.calls}

class Controller:
    start_state = "gen"
    end_state = "[EOQ]"
    def __init__(self, model):
        self.model = model
    def init_data(self, value):
        return Data(value)
    def execute(self, state, debug=False):
        assert debug is False
        assert state.data.value["paras"] == ["Fixture title||Pinned evidence paragraph."]
        assert state.data.value["paragraph_ids"] == ["fixture-p1"]
        self.model.calls += 1
        output = state.copy()
        output.data.questions.append("safe decomposition")
        output.data.answers.append(json.dumps("fixture-answer"))
        output._score += 0.25
        output.next = self.end_state
        return [output]

def build_decomposer_and_models(config):
    assert config["models"]["gen"]["max_steps"] == 3
    print("fake upstream boot diagnostic")
    model = Model()
    return BestFirstDecomposer(Controller(model)), {"gen": model}
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _request(**overrides: Any) -> dict[str, Any]:
    paras = ("Fixture title||Pinned evidence paragraph.",)
    paragraph_ids = ("fixture-p1",)
    value: dict[str, Any] = {
        "case_id": "case-1",
        "question": "What is the fixture answer?",
        "paras": list(paras),
        "paragraph_ids": list(paragraph_ids),
        "context_hash": decomp_context_hash(paragraph_ids, paras),
        "config_artifact_id": "fixture.safe",
        "backend_binding": {"backend": "fixture", "model_id": "fixture-model"},
        "max_steps": 3,
        "upstream_revision": UPSTREAM_COMMIT,
        "schema_version": REQUEST_SCHEMA,
    }
    value.update(overrides)
    return value


class FakeCheckout:
    def __init__(self, parent: Path, *, model_name: str = "llmqadecomp") -> None:
        self.root = parent / "DecomP"
        self.root.mkdir()
        _write(self.root / ".git" / "HEAD", UPSTREAM_COMMIT + "\n")
        _write(self.root / "LICENSE", "Apache License 2 fixture\n")
        _write(self.root / "src" / "decomp" / "__init__.py", "")
        _write(self.root / "src" / "decomp" / "inference" / "__init__.py", "")
        _write(
            self.root / "src" / "decomp" / "inference" / "model_search.py",
            _MODEL_SEARCH,
        )
        _write(
            self.root / "src" / "decomp" / "inference" / "configurable_inference.py",
            _CONFIGURABLE,
        )
        config = {
            "start_state": "gen",
            "end_state": "[EOQ]",
            "models": {"gen": {"name": model_name, "max_steps": 99}},
        }
        config_path = self.root / "configs" / "fixture.json"
        _write(config_path, json.dumps(config))
        self.manifest_path = parent / "manifest.json"
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "upstream": {
                "commit": UPSTREAM_COMMIT,
                "license_spdx": "Apache-2.0",
                "license_sha256": _sha256((self.root / "LICENSE").read_bytes()),
                "require_clean_checkout": False,
            },
            "source_files": {
                "src/decomp/inference/configurable_inference.py": _sha256(
                    (self.root / "src/decomp/inference/configurable_inference.py").read_bytes()
                ),
                "src/decomp/inference/model_search.py": _sha256(
                    (self.root / "src/decomp/inference/model_search.py").read_bytes()
                ),
            },
            "config_artifacts": {
                "fixture.safe": {
                    "path": "configs/fixture.json",
                    "sha256": _sha256(config_path.read_bytes()),
                    "ext_vars": {},
                }
            },
            "allowed_model_names": ["llmqadecomp"],
            "forbidden_model_names": ["expr_eval"],
            "allowed_reader_names": ["drop"],
            "model_call_metric_names": ["fake_llm"],
            "patches": [{"id": "fixture-patch"}],
        }
        _write(self.manifest_path, json.dumps(manifest))

    def driver(self) -> DecompDriver:
        return DecompDriver(
            manifest_path=self.manifest_path,
            environment={"AGENTINPUTLAB_UPSTREAM_ROOT": str(self.root)},
        )


def _typed_response(value: dict[str, Any]) -> DecompWorkerResponse:
    return DecompWorkerResponse(
        case_id=value["case_id"],
        profile_id=value["profile_id"],
        upstream_revision=value["upstream_revision"],
        paragraph_ids=tuple(value["paragraph_ids"]),
        context_hash=value["context_hash"],
        nodes=tuple(DecompGraphNode(**item) for item in value["nodes"]),
        dependency_edges=tuple(tuple(edge) for edge in value["dependency_edges"]),
        participant_calls=tuple(
            DecompParticipantCall(**item) for item in value["participant_calls"]
        ),
        frontier=tuple(DecompFrontierEvent(**item) for item in value["frontier"]),
        reasoning_chain=value["reasoning_chain"],
        final_answer=value["final_answer"],
        termination=value["termination"],
        component_only=value["component_only"],
        usage=value["usage"],
        schema_version=value["schema_version"],
        native_artifacts=value["native_artifacts"],
    )


class DecompWorkerDriverTests(unittest.TestCase):
    def test_unmatched_native_route_fails_without_echo_or_dispatch(self) -> None:
        class RoutedExecutionParticipant:
            model_question_regex = re.compile(r"\[([^\]]+)\] (.*)")
            operation_model_question_regex = re.compile(r"\((.+)\) \[([^\]]+)\] (.*)")
            operation_only_regex = re.compile(r"\((.+)\)(.*)")

            def query(self, state, debug=False):
                raise AssertionError("the guard must not dispatch native work")

        router = RoutedExecutionParticipant()
        module = SimpleNamespace(RoutedExecutionParticipant=RoutedExecutionParticipant)
        controller = SimpleNamespace(model_list={"router": router.query})

        def state(text):
            return SimpleNamespace(_data=SimpleNamespace(get_last_question=lambda: text))

        with patch.dict(sys.modules, {"decomp.inference.participant_execution_routed": module}):
            for instruction in ("[qa] question", "(select) [qa] question", "(merge) #1 #2"):
                with self.subTest(instruction=instruction):
                    _validate_routed_instruction(controller, state(instruction), "router")
            for malformed in ("", "unmatched private transcript"):
                with self.subTest(instruction=malformed), self.assertRaises(DriverError) as caught:
                    _validate_routed_instruction(controller, state(malformed), "router")
                self.assertEqual(caught.exception.code, "UPSTREAM_OUTPUT_INVALID")
                self.assertEqual(
                    caught.exception.public_message, "native DecomP router received an invalid instruction"
                )
            _validate_routed_instruction(controller, object(), "different_participant")

    def test_distributed_upstream_license_matches_pinned_apache_text(self) -> None:
        license_path = (
            Path(__file__).parents[2] / "workers" / "decomp" / ("UPSTREAM_LICENSE.Apache-2.0.txt")
        )
        self.assertEqual(
            _sha256(license_path.read_bytes()),
            "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        )

    def test_fake_checkout_exercises_upstream_plumbing_and_wire_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            response = dict(checkout.driver().run(_request()))

        self.assertEqual(response["profile_id"], PROFILE_ID)
        self.assertEqual(response["upstream_revision"], DECOMP_REVISION)
        self.assertEqual(response["schema_version"], DECOMP_WORKER_SCHEMA)
        self.assertEqual(response["paragraph_ids"], ["fixture-p1"])
        self.assertEqual(response["context_hash"], _request()["context_hash"])
        self.assertEqual(response["final_answer"], "fixture-answer")
        self.assertEqual(response["termination"], "end_state")
        self.assertEqual(response["usage"]["model_calls"], 1)
        self.assertEqual(len(response["nodes"]), 2)
        self.assertEqual(response["dependency_edges"], [["s0000", "s0001"]])
        self.assertEqual(response["participant_calls"][0]["participant"], "gen")
        self.assertEqual(response["frontier"][-1]["action"], "terminate")
        self.assertIn("safe decomposition", response["reasoning_chain"])
        self.assertEqual(
            response["native_artifacts"]["backend_binding_mode"],
            "audit_identity_only; upstream-native model clients",
        )
        typed = _typed_response(response)
        validate_decomp_full_response(typed)

    def test_jsonl_keeps_upstream_stdout_out_of_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            source = io.StringIO(json.dumps(_request()) + "\nnot-json\n")
            sink = io.StringIO()
            self.assertEqual(serve_jsonl(checkout.driver(), source, sink), 0)
        lines = sink.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        success = json.loads(lines[0])
        failure = json.loads(lines[1])
        self.assertNotIn("fake upstream boot diagnostic", lines[0])
        self.assertEqual(success["case_id"], "case-1")
        self.assertEqual(failure["failure_code"], "INVALID_JSON")

    def test_runtime_adapter_executes_bridge_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            runtime = PinnedDecompRuntime(
                manifest_path=checkout.manifest_path,
                environment={"AGENTINPUTLAB_UPSTREAM_ROOT": str(checkout.root)},
            )
            state = FrozenAgentState(
                query="What is the fixture answer?",
                task_id="case-1",
                context=(
                    ContextBlock(
                        text="Fixture title||Pinned evidence paragraph.",
                        block_id="fixture-p1",
                    ),
                ),
            )
            record = CompositeInterventionRunner().run(
                DecompFullBridge(
                    None,
                    config_artifact_id="fixture.safe",
                    max_steps=3,
                ),
                state,
                InterventionContext(external_runtime=runtime),
            )
            result = record.result
            fingerprint = runtime.fingerprint()

        self.assertEqual(result.source_state_hash, state.state_hash)
        self.assertEqual(result.release_id, PROFILE_ID)
        self.assertEqual(result.final_answer, "fixture-answer")
        self.assertEqual(result.evidence, state.context)
        self.assertIn("safe decomposition", result.metadata["reasoning_chain"])
        self.assertEqual(result.metadata["paragraph_ids"], ("fixture-p1",))
        self.assertEqual(result.trace.counters["model_calls"], 1)
        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual(fingerprint["upstream_revision"], UPSTREAM_COMMIT)
        self.assertEqual(fingerprint["concurrency_mode"], "process_global_serial_rlock")
        self.assertEqual(
            fingerprint["worker_id"],
            "agentinputlab.decomp.pinned.in_process.v1",
        )

    def test_driver_rejects_mismatched_context_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeCheckout(Path(directory)).driver()
            with self.assertRaises(DriverError) as raised:
                driver.run(_request(context_hash="0" * 64))
        self.assertEqual(raised.exception.code, "CONTEXT_IDENTITY_MISMATCH")

    def test_driver_accepts_revision_named_fingerprint_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response = (
                FakeCheckout(Path(directory))
                .driver()
                .run(
                    _request(
                        backend_binding={
                            "model_revision": "model-v1",
                            "tokenizer_revision": "tokenizer-v2",
                            "max_output_tokens": 128,
                        }
                    )
                )
            )
        self.assertEqual(response["final_answer"], "fixture-answer")

    def test_request_rejects_revision_unknown_config_and_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeCheckout(Path(directory)).driver()
            cases = (
                (_request(upstream_revision="0" * 40), "REVISION_MISMATCH"),
                (_request(config_artifact_id="../../arbitrary.py"), "CONFIG_NOT_ALLOWED"),
                (_request(backend_binding={"api_key": "secret"}), "SECRET_REJECTED"),
                (_request(max_steps=0), "INVALID_REQUEST"),
            )
            for request, code in cases:
                with self.subTest(code=code), self.assertRaises(DriverError) as raised:
                    driver.run(request)
                self.assertEqual(raised.exception.code, code)

    def test_dangerous_upstream_eval_participant_is_blocked_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory), model_name="expr_eval")
            with self.assertRaises(DriverError) as raised:
                checkout.driver().run(_request())
        self.assertEqual(raised.exception.code, "UNSAFE_CONFIG")

    def test_license_source_and_config_tampering_fail_closed(self) -> None:
        targets = (
            ("LICENSE", "LICENSE_MISMATCH"),
            ("src/decomp/inference/model_search.py", "SOURCE_MISMATCH"),
            ("configs/fixture.json", "CONFIG_MISMATCH"),
        )
        for relative, code in targets:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                checkout = FakeCheckout(Path(directory))
                path = checkout.root / relative
                path.write_text(path.read_text(encoding="utf-8") + "\ntampered", encoding="utf-8")
                with self.assertRaises(DriverError) as raised:
                    checkout.driver().run(_request())
                self.assertEqual(raised.exception.code, code)

    def test_driver_has_no_dynamic_python_or_shell_execution_path(self) -> None:
        driver_path = Path(__file__).parents[2] / "workers" / "decomp" / "driver.py"
        tree = ast.parse(driver_path.read_text(encoding="utf-8"))
        forbidden_names = {"eval", "exec", "compile"}
        forbidden_attributes = {("os", "system"), ("subprocess", "Popen")}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, forbidden_names)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                self.assertNotIn((node.func.value.id, node.func.attr), forbidden_attributes)
            for keyword in node.keywords:
                if keyword.arg == "shell":
                    self.assertIsInstance(keyword.value, ast.Constant)
                    self.assertFalse(keyword.value.value)

    def test_real_pinned_checkout_imports_best_first_core_when_available(self) -> None:
        upstream = Path(os.environ.get("AGENTINPUTLAB_DECOMP_CHECKOUT", "/private/tmp/agentinputlab-upstreams/DecomP"))
        if not _is_git_checkout(upstream):
            self.skipTest("pinned DecomP checkout is not present")
        driver = DecompDriver(
            environment={"AGENTINPUTLAB_UPSTREAM_ROOT": str(upstream)},
        )
        result = driver.smoke_upstream_core()
        self.assertEqual(result["upstream_revision"], UPSTREAM_COMMIT)
        self.assertEqual(
            result["symbol"],
            "decomp.inference.model_search.BestFirstDecomposer",
        )


if __name__ == "__main__":
    unittest.main()
