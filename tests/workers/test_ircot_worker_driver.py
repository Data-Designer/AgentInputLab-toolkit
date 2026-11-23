from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agentinputlab.algorithms.acquire.ircot import (
    IRCoTControlledWorker,
    IRCoTControls,
)
from agentinputlab.backends.llm import CallableBackend
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import FrozenAgentState, InterventionContext
from workers.ircot.driver import (
    MANIFEST_SCHEMA,
    PROTOCOL_VERSION,
    UPSTREAM_COMMIT,
    IRCoTDriver,
    serve_jsonl,
)
from workers.ircot.runtime import PinnedIRCoTWorker

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_WORKER_ROOT = _REPOSITORY_ROOT / "workers" / "ircot"
_REAL_CHECKOUT = Path(os.environ.get("AGENTINPUTLAB_IRCOT_CHECKOUT", "/private/tmp/agentinputlab-upstreams/ircot"))


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


_MODEL_SEARCH = """
import copy

class SearchState:
    def __init__(self, data, command, score=0.0):
        self._data = data
        self._next = command
        self._score = score
    def copy(self):
        return SearchState(copy.deepcopy(self._data), self._next, self._score)
    @property
    def data(self):
        return self._data
    @property
    def next(self):
        return self._next
    @next.setter
    def next(self, value):
        self._next = value

class BestFirstDecomposer:
    def __init__(self, controller):
        self.controller = controller
    def find_answer_decomp(self, example, debug=False):
        state = SearchState(
            self.controller.init_data(example),
            self.controller.start_state,
        )
        iterations = 0
        while state.next != self.controller.end_state:
            outputs = self.controller.execute(state, debug=debug)
            if len(outputs) != 1:
                raise ValueError("fixture controller branch count changed")
            state = outputs[0]
            iterations += 1
            if iterations > 32:
                raise ValueError("fixture controller failed to terminate")
        return state, []
"""

_IRCOT = """
import hashlib

class RetrieveAndResetParagraphsParticipant:
    pass
class StepByStepCOTGenParticipant:
    pass
class StepByStepExitControllerParticipant:
    pass
class GPT3Generator:
    pass
class LLMClientGenerator:
    pass

def safe_post_request(url, params):
    raise RuntimeError("external retrieval must be replaced by frozen bindings")

def remove_reasoning_sentences(sentences):
    return list(sentences)

def get_pid_for_title_paragraph_text(title, paragraph):
    value = (title + "\\0" + paragraph).encode("utf-8")
    return hashlib.sha256(value).hexdigest()
"""

_CONFIGURABLE = """
from commaqa.inference import ircot
from commaqa.inference.model_search import BestFirstDecomposer

class Data(dict):
    pass

class Participant:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.calls = 0
        self.generator = (
            ircot.GPT3Generator(**config)
            if name == "step_by_step_cot_gen"
            else None
        )
    def return_model_calls(self):
        key = "step_by_step_cot" if self.name == "step_by_step_cot_gen" else self.name
        return {key: self.calls}

class Controller:
    def __init__(self, config, participants):
        self.config = config
        self.participants = participants
        self.start_state = config["start_state"]
        self.end_state = config["end_state"]
    def init_data(self, value):
        return Data(
            question=value.get("question") or value["query"],
            titles=list(value.get("titles", [])),
            paras=list(value.get("paras", [])),
            generated_sentences=[],
        )
    def execute(self, state, debug=False):
        if debug:
            raise ValueError("debug must remain disabled")
        output = state.copy()
        model_config = self.config["models"][state.next]
        participant = self.participants[state.next]
        if participant.name == "retrieve_and_reset_paragraphs":
            generated = output.data["generated_sentences"]
            query = generated[-1] if generated else output.data["question"]
            response = ircot.safe_post_request("http://fixture.invalid/retrieve", {"query_text": query})
            documents = response.json()["retrieval"][: model_config["retrieval_count"]]
            if not model_config["cumulate_titles"]:
                output.data["titles"] = []
                output.data["paras"] = []
            known = set(zip(output.data["titles"], output.data["paras"]))
            for document in documents:
                pair = (document["title"], document["paragraph_text"])
                if pair not in known:
                    output.data["titles"].append(pair[0])
                    output.data["paras"].append(pair[1])
                    known.add(pair)
        elif participant.name == "step_by_step_cot_gen":
            prompt = "Question: " + output.data["question"]
            text = participant.generator.generate_text_sequence(prompt)[0][0]
            participant.calls += 1
            output.data["generated_sentences"].append(text)
        elif participant.name == "step_by_step_exit_controller":
            latest = output.data["generated_sentences"][-1]
            if "answer is" in latest:
                output.next = self.end_state
                return [output]
        else:
            raise ValueError("unexpected participant")
        output.next = model_config["next_model"]
        return [output]

def build_decomposer_and_models(config):
    participants = {
        key: Participant(value["name"], value)
        for key, value in config["models"].items()
    }
    controller = Controller(config, participants)
    return BestFirstDecomposer(controller), participants
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


class FakeCheckout:
    def __init__(self, parent: Path) -> None:
        self.root = parent / "ircot-upstream"
        self.root.mkdir()
        _write(self.root / ".git" / "HEAD", UPSTREAM_COMMIT + "\n")
        _write(self.root / "LICENSE", "Apache License 2.0 fixture\n")
        _write(self.root / "commaqa" / "__init__.py", "")
        _write(self.root / "commaqa" / "inference" / "__init__.py", "")
        _write(self.root / "commaqa" / "inference" / "model_search.py", _MODEL_SEARCH)
        _write(self.root / "commaqa" / "inference" / "ircot.py", _IRCOT)
        _write(
            self.root / "commaqa" / "inference" / "configurable_inference.py",
            _CONFIGURABLE,
        )

        self.manifest_path = parent / "compatibility_manifest.json"
        for name in ("fixture_config.json", "fixture_prompt.txt", "fixture_bindings.json"):
            _write(parent / name, (_WORKER_ROOT / name).read_text(encoding="utf-8"))
        config = parent / "fixture_config.json"
        prompt = parent / "fixture_prompt.txt"
        bindings = parent / "fixture_bindings.json"
        source_files = {
            "commaqa/inference/model_search.py": _sha256(
                (self.root / "commaqa" / "inference" / "model_search.py").read_bytes()
            ),
            "commaqa/inference/ircot.py": _sha256(
                (self.root / "commaqa" / "inference" / "ircot.py").read_bytes()
            ),
            "commaqa/inference/configurable_inference.py": _sha256(
                (self.root / "commaqa" / "inference" / "configurable_inference.py").read_bytes()
            ),
        }
        fixture_profile = {
            "identity_kind": "controlled_fixture",
            "config_scope": "worker",
            "config_path": config.name,
            "config_sha256": _sha256(config.read_bytes()),
            "prompt_scope": "worker",
            "prompt_path": prompt.name,
            "prompt_sha256": _sha256(prompt.read_bytes()),
            "bindings_scope": "worker",
            "bindings_path": bindings.name,
            "bindings_sha256": _sha256(bindings.read_bytes()),
            "corpus_revision": "fixture-corpus-v1",
            "index_revision": "fixture-index-v1",
            "binding_attestation": "fixture_identity_only",
            "config_ext_vars": {},
        }
        native_profile = {
            key: value
            for key, value in fixture_profile.items()
            if not key.startswith("bindings_") and key != "index_revision"
        }
        native_profile.update(
            {
                "identity_kind": "upstream_native",
                "index_revision_env": "IRCOT_TEST_INDEX_REVISION",
                "binding_attestation": "exact_fingerprint_hashes_from_environment",
                "generator_fingerprint_sha256_env": "IRCOT_TEST_GENERATOR_SHA256",
                "retriever_fingerprint_sha256_env": "IRCOT_TEST_RETRIEVER_SHA256",
                "required_runtime_env": ["IRCOT_TEST_MODEL_READY"],
            }
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "upstream": {
                "repository": "https://example.invalid/pinned-ircot-fixture",
                "commit": UPSTREAM_COMMIT,
                "license_spdx": "Apache-2.0",
                "license_sha256": _sha256((self.root / "LICENSE").read_bytes()),
                "require_clean_checkout": False,
            },
            "source_files": source_files,
            "allowed_model_names": [
                "retrieve_and_reset_paragraphs",
                "step_by_step_cot_gen",
                "step_by_step_exit_controller",
            ],
            "profiles": {
                "fixture.tiny.frozen.v1": fixture_profile,
                "upstream.fixture.gated.v1": native_profile,
            },
            "patches": [{"id": "test-manifest-lock"}],
        }
        _write(self.manifest_path, json.dumps(manifest, sort_keys=True))
        self.prompt_hash = fixture_profile["prompt_sha256"]

    def environment(self) -> dict[str, str]:
        return {"AGENTINPUTLAB_UPSTREAM_ROOT": str(self.root)}

    def driver(self, *, isolated_process: bool = False) -> IRCoTDriver:
        return IRCoTDriver(
            manifest_path=self.manifest_path,
            environment=self.environment(),
            isolated_process=isolated_process,
        )

    def runtime(self, *, profile_id: str = "fixture.tiny.frozen.v1") -> PinnedIRCoTWorker:
        return PinnedIRCoTWorker(
            manifest_path=self.manifest_path,
            profile_id=profile_id,
            environment=self.environment(),
        )


def _request(prompt_hash: str, **overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "request_id": "request-1",
        "state_hash": "state-1",
        "task_id": "case-1",
        "question": "Fixture entity relation?",
        "controls": {
            "per_step_top_k": 2,
            "max_steps": 2,
            "max_total_documents": 8,
            "max_output_tokens": 64,
            "deduplicate": True,
            "reset_policy": "upstream_native",
        },
        "generator_fingerprint": {"backend": "fixture"},
        "retriever_fingerprint": {
            "retriever": "fixture",
            "tokenizer_id": "safe-identity-not-a-secret",
        },
        "corpus_revision": "fixture-corpus-v1",
        "index_revision": "fixture-index-v1",
        "prompt_hash": prompt_hash,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol_version": PROTOCOL_VERSION,
    }
    value.update(overrides)
    return value


def _context() -> InterventionContext:
    return InterventionContext(
        backend=CallableBackend(
            lambda _: "unused",
            model_id="fixture-generator",
            revision="fixture-v1",
            callable_id="tests.ircot.fixture-generator.v1",
        ),
        retriever=InMemoryBM25Retriever(
            [
                Document(
                    doc_id="local-binding",
                    text="Only its fingerprint is used by the external worker.",
                    corpus_revision="fixture-corpus-v1",
                )
            ]
        ),
    )


def _algorithm(client: PinnedIRCoTWorker, prompt_hash: str) -> IRCoTControlledWorker:
    return IRCoTControlledWorker(
        client=client,
        corpus_revision="fixture-corpus-v1",
        index_revision="fixture-index-v1",
        prompt_hash=prompt_hash,
        controls=IRCoTControls(
            per_step_top_k=2,
            max_steps=2,
            max_total_documents=8,
            max_output_tokens=64,
        ),
    )


class IRCoTWorkerDriverTests(unittest.TestCase):
    def test_fake_pinned_checkout_runs_bridge_propose_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            state = FrozenAgentState(
                query="Fixture entity relation?",
                task_id="case-1",
            )

            patch = _algorithm(checkout.runtime(), checkout.prompt_hash).propose(
                state,
                _context(),
            )
            materialized = patch.apply(state)

            self.assertEqual(
                [block.block_id for block in materialized.context],
                ["fixture-d1", "fixture-d2"],
            )
            self.assertEqual(patch.trace.counters["retrieval_calls"], 2)
            self.assertEqual(patch.trace.counters["model_calls"], 2)
            self.assertEqual(patch.trace.counters["reasoning_steps"], 1)
            self.assertEqual(
                patch.metadata["worker_fingerprint"]["identity_kind"],
                "controlled_fixture",
            )
            self.assertEqual(
                patch.trace.events[-1].payload["exit_reason"],
                "answer_detected",
            )

    def test_jsonl_fixture_has_one_clean_structured_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            source = io.StringIO(json.dumps(_request(checkout.prompt_hash)) + "\n")
            sink = io.StringIO()

            self.assertEqual(serve_jsonl(checkout.driver(isolated_process=True), source, sink), 0)
            lines = sink.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            response = json.loads(lines[0])
            self.assertEqual(response["status"], "ok")
            self.assertEqual(response["answer"], "fixture-answer")
            self.assertEqual(response["worker_identity_kind"], "controlled_fixture")

    def test_request_cannot_select_paths_imports_commands_or_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            for injected in (
                {"path": "/tmp/untrusted"},
                {"import": "untrusted.module"},
                {"command": ["sh", "-c", "payload"]},
            ):
                with self.subTest(field=next(iter(injected))):
                    response = checkout.driver().safe_call(
                        {**_request(checkout.prompt_hash), **injected}
                    )
                    self.assertEqual(response["failure_code"], "INVALID_REQUEST")
                    self.assertFalse(response["consumed"])
            secret = _request(
                checkout.prompt_hash,
                generator_fingerprint={"openai_api_key": "must-not-cross-worker-boundary"},
            )
            response = checkout.driver().safe_call(secret)
            self.assertEqual(response["failure_code"], "SECRET_REJECTED")
            self.assertFalse(response["consumed"])

    def test_artifact_commit_license_and_source_checks_fail_closed(self) -> None:
        cases = (
            ("prompt", "fixture_prompt.txt", "PROMPT_MISMATCH"),
            ("config", "fixture_config.json", "CONFIG_MISMATCH"),
            ("license", None, "LICENSE_MISMATCH"),
            ("source", None, "SOURCE_MISMATCH"),
        )
        for label, artifact, code in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                checkout = FakeCheckout(Path(directory))
                if artifact is not None:
                    target = checkout.manifest_path.parent / artifact
                elif label == "license":
                    target = checkout.root / "LICENSE"
                else:
                    target = checkout.root / "commaqa" / "inference" / "ircot.py"
                target.write_text(target.read_text(encoding="utf-8") + "\ntampered\n")
                response = checkout.driver().safe_call(_request(checkout.prompt_hash))
                self.assertEqual(response["failure_code"], code)
                self.assertFalse(response["consumed"])

        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            _write(checkout.root / ".git" / "HEAD", "0" * 40 + "\n")
            response = checkout.driver().safe_call(_request(checkout.prompt_hash))
            self.assertEqual(response["failure_code"], "REVISION_MISMATCH")
            self.assertFalse(response["consumed"])

    def test_upstream_native_identity_is_distinct_and_requires_jsonl_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = FakeCheckout(Path(directory))
            generator = {"backend": "production-model", "revision": "pinned"}
            retriever = {"backend": "production-es", "index": "pinned"}
            environment = {
                **checkout.environment(),
                "IRCOT_TEST_INDEX_REVISION": "production-index-v1",
                "IRCOT_TEST_GENERATOR_SHA256": _canonical_hash(generator),
                "IRCOT_TEST_RETRIEVER_SHA256": _canonical_hash(retriever),
                "IRCOT_TEST_MODEL_READY": "1",
            }
            runtime = PinnedIRCoTWorker(
                manifest_path=checkout.manifest_path,
                profile_id="upstream.fixture.gated.v1",
                environment=environment,
            )
            fingerprint = runtime.fingerprint()
            self.assertEqual(fingerprint["identity_kind"], "upstream_native")
            self.assertNotEqual(fingerprint["identity_kind"], "controlled_fixture")
            request = _request(
                checkout.prompt_hash,
                generator_fingerprint=generator,
                retriever_fingerprint=retriever,
                index_revision="production-index-v1",
            )
            response = runtime.call(request)
            self.assertEqual(response["failure_code"], "ISOLATION_REQUIRED")
            self.assertFalse(response["consumed"])

            isolated = IRCoTDriver(
                manifest_path=checkout.manifest_path,
                profile_id="upstream.fixture.gated.v1",
                environment=environment,
                isolated_process=True,
            ).safe_call(request)
            self.assertEqual(isolated["failure_code"], "TOKEN_USAGE_OBSERVER_REQUIRED")
            self.assertFalse(isolated["consumed"])
            self.assertNotIn("usage", isolated)

    @unittest.skipUnless(
        _is_git_checkout(_REAL_CHECKOUT),
        "real pinned IRCoT checkout is unavailable",
    )
    def test_real_pinned_checkout_import_smoke_and_fixture_loop(self) -> None:
        environment = {"AGENTINPUTLAB_UPSTREAM_ROOT": str(_REAL_CHECKOUT)}
        runtime = PinnedIRCoTWorker(environment=environment)
        fingerprint = runtime.fingerprint()
        self.assertEqual(fingerprint["upstream_revision"], UPSTREAM_COMMIT)
        self.assertEqual(fingerprint["identity_kind"], "controlled_fixture")
        full_api_status = fingerprint["full_api_status"]
        self.assertIn(
            full_api_status.split(":", maxsplit=1)[0], {"available", "dependency_missing"}
        )
        if full_api_status.startswith("dependency_missing:"):
            self.skipTest(f"required pinned IRCoT dependency is missing: {full_api_status}")
        prompt_hash = _sha256((_WORKER_ROOT / "fixture_prompt.txt").read_bytes())
        state = FrozenAgentState(query="Fixture entity relation?", task_id="real-smoke")
        patch = _algorithm(runtime, prompt_hash).propose(state, _context())
        self.assertEqual(
            [block.block_id for block in patch.apply(state).context],
            ["fixture-d1", "fixture-d2"],
        )


if __name__ == "__main__":
    unittest.main()
