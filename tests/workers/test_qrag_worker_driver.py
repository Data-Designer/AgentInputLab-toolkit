from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from typing import Any

from agentinputlab.algorithms.acquire.qrag import (
    QRAG_INSTRUCTION_PROMPT_SHA256,
    QRAG_INSTRUCTION_PROTOCOL,
    QRAG_INSTRUCTION_READER_PROFILE,
    QRAG_INSTRUCTION_SYSTEM_PROMPT,
    parse_instruction_reader_answer,
    render_instruction_reader_prompt,
)
from workers.qrag.driver import (
    PROTOCOL,
    UPSTREAM_COMMIT,
    QRAGFrozenDriver,
    ReaderResult,
    SelectorResult,
)


def content_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeOfficialRuntime:
    def __init__(self, source_root: Path) -> None:
        self.builder = _load_module(source_root / "candidate_builder.py", "qrag_fake_builder")
        self.policy = _load_module(source_root / "pqn.py", "qrag_fake_pqn")
        self.environment = _load_module(source_root / "qa_env.py", "qrag_fake_env")

    def select(self, *, question, candidates, seed, max_steps):
        built = self.builder.build_candidates(question, candidates)
        agent = self.policy.PQN()
        agent.load_weights_only(strict=True)
        agent.eval()
        env = self.environment.QAEnv(built, max_steps=max_steps)
        trajectory = []
        selected = []
        for index in range(min(max_steps, len(built))):
            available = tuple(i for i in range(len(built)) if i not in selected)
            action, q_values = agent.select_action(available, evaluate=True, random=False)
            env.step(action)
            selected.append(action)
            trajectory.append(
                {
                    "index": index,
                    "available_indices": available,
                    "q_values": q_values,
                    "action_index": action,
                }
            )
        return SelectorResult(
            trajectory=tuple(trajectory),
            selected_indices=tuple(selected),
            embedding_calls=2,
        )


class RecordingReader:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, question, passages, artifacts, *, max_output_tokens):
        self.calls.append((question, passages, artifacts["reader_revision"], max_output_tokens))
        return ReaderResult("fixture answer", input_tokens=17, output_tokens=3)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QRAGWorkerDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = (self.root / "source").resolve()
        self.checkpoint = (self.root / "checkpoint").resolve()
        self.embedder = (self.root / "embedder").resolve()
        for path in (self.source, self.checkpoint, self.embedder):
            path.mkdir()
        (self.source / "candidate_builder.py").write_text(
            "def build_candidates(question, candidates):\n"
            "    assert question and all(set(c) == {'document_id', 'text'} for c in candidates)\n"
            "    return tuple(candidates)\n"
        )
        (self.source / "pqn.py").write_text(
            "class PQN:\n"
            "    def load_weights_only(self, strict): assert strict\n"
            "    def eval(self): return self\n"
            "    def select_action(self, available, evaluate, random):\n"
            "        assert evaluate and not random\n"
            "        table = {3: (2, (0.2, 0.1, 0.9)), 2: (0, (0.8, 0.5))}\n"
            "        action_pos, q = table[len(available)]\n"
            "        return available[action_pos], q\n"
        )
        (self.source / "qa_env.py").write_text(
            "class QAEnv:\n"
            "    def __init__(self, candidates, max_steps): self.available=set(range(len(candidates)))\n"
            "    def step(self, action): assert action in self.available; self.available.remove(action)\n"
        )
        (self.checkpoint / "model_best.pt").write_bytes(b"tensor-only-fixture")
        (self.checkpoint / "config.yaml").write_text("fixture: true\n")
        embedder_manifest = {
            "repository": "intfloat/multilingual-e5-large",
            "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
            "files": {"fixture": "0" * 64},
        }
        (self.embedder / "agentinputlab-manifest.json").write_text(json.dumps(embedder_manifest))
        source_hashes = {
            name: file_hash(self.source / name)
            for name in ("candidate_builder.py", "pqn.py", "qa_env.py")
        }
        self.checkpoint_model_hash = file_hash(self.checkpoint / "model_best.pt")
        self.checkpoint_config_hash = file_hash(self.checkpoint / "config.yaml")
        self.checkpoint_manifest_hash = content_hash(
            {
                "config.yaml": self.checkpoint_config_hash,
                "model_best.pt": self.checkpoint_model_hash,
            }
        )
        self.source_manifest_hash = "a" * 64
        manifest = {
            "source": {
                "repository": "griver/Q-RAG",
                "commit": UPSTREAM_COMMIT,
                "tree": "fixture-tree",
                "content_manifest_sha256": self.source_manifest_hash,
                "required_files": source_hashes,
            },
            "checkpoint": {
                "repository": "Q-RAG/qrag-ft-e5-on-hotpotqa",
                "revision": "e0e48389f551075ace2ce61207522b6bad55593a",
                "model_best_sha256": "b" * 64,
                "config_sha256": "c" * 64,
            },
            "embedder": embedder_manifest,
        }
        source_receipt = {
            "repository": "griver/Q-RAG",
            "commit": UPSTREAM_COMMIT,
            "tree": "fixture-tree",
            "transport": "jsdelivr_commit_snapshot",
            "files": [
                {
                    "path": name,
                    "sha256": digest,
                    "size": (self.source / name).stat().st_size,
                }
                for name, digest in sorted(source_hashes.items())
            ],
        }
        (self.source / "agentinputlab-source-receipt.json").write_text(json.dumps(source_receipt))
        self.manifest = self.root / "compatibility.json"
        self.manifest.write_text(json.dumps(manifest))
        self.manifest_hash = content_hash(manifest)
        self.embedder_manifest_hash = content_hash(embedder_manifest)
        self.reader = RecordingReader()
        self.runtime_calls = 0

        def runtime_factory(source, checkpoint, embedder):
            self.runtime_calls += 1
            self.assertEqual((checkpoint, embedder), (self.checkpoint, self.embedder))
            return FakeOfficialRuntime(source)

        self.driver = QRAGFrozenDriver(
            source_root=self.source,
            checkpoint_root=self.checkpoint,
            embedder_root=self.embedder,
            runtime_factory=runtime_factory,
            reader=self.reader,
            manifest_path=self.manifest,
            fixture_mode=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def request(self) -> dict[str, Any]:
        candidates = (
            {"document_id": "c0", "text": "alpha"},
            {"document_id": "c1", "text": "beta"},
            {"document_id": "c2", "text": "gamma"},
        )
        artifacts = {
            "profile_id": "qrag.e5_hotpotqa.frozen.v1",
            "checkpoint_repository": "Q-RAG/qrag-ft-e5-on-hotpotqa",
            "checkpoint_revision": "e0e48389f551075ace2ce61207522b6bad55593a",
            "checkpoint_file_sha256": self.checkpoint_model_hash,
            "checkpoint_config_sha256": self.checkpoint_config_hash,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_hash,
            "embedder_repository": "intfloat/multilingual-e5-large",
            "embedder_revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
            "embedder_manifest_sha256": self.embedder_manifest_hash,
            "source_manifest_sha256": self.source_manifest_hash,
            "compatibility_manifest_sha256": self.manifest_hash,
            "reader_profile": "upstream_qwq32b_family_controlled",
            "reader_revision": "976055f8c83f394f35dbd3ab09a285a984907bd0",
            "reader_repository": "Qwen/QwQ-32B",
            "reader_deployment_fingerprint_sha256": "d" * 64,
        }
        payload = {
            "profile_id": artifacts["profile_id"],
            "question": "Which chunks?",
            "candidates": candidates,
            "candidate_pool_sha256": content_hash(candidates),
            "candidate_order_sha256": content_hash(("c0", "c1", "c2")),
            "artifacts": artifacts,
            "artifact_identity_sha256": content_hash(artifacts),
            "acknowledgements": {
                "source_noassertion_acknowledged": True,
                "checkpoint_cc_by_4_0_acknowledged": True,
                "user_mounted_source": True,
                "user_mounted_checkpoint": True,
            },
            "controls": {
                "max_select_actions": 2,
                "max_output_tokens": 4000,
                "max_candidates": 128,
                "max_candidate_chars": 2_000_000,
            },
            "remaining_budget": {
                "model_calls": 1,
                "output_tokens": 4000,
                "context_tokens": None,
            },
            "reader_budget": {
                "configured_max_output_tokens": 4000,
                "effective_max_output_tokens": 4000,
                "remaining_output_tokens": 4000,
                "preset": "upstream_default",
            },
            "seed": 7,
            "inference_contract": {
                "mode": "frozen_checkpoint_inference",
                "allow_training": False,
                "allow_gold_answer": False,
                "allow_supporting_fact_labels": False,
                "allow_reward": False,
                "allow_ground_truth_feedback": False,
                "allow_stop_action": False,
            },
        }
        return {
            "method_id": "qrag",
            "release_id": "acquire.qrag.released_checkpoint_native",
            "upstream_commit": UPSTREAM_COMMIT,
            "protocol_version": PROTOCOL,
            "request_id": "fixture-request",
            "payload": payload,
        }

    def test_fake_checkout_e2e_preserves_actions_q_values_and_reader_order(self) -> None:
        response = self.driver.handle(self.request())

        self.assertEqual(response["status"], "ok")
        payload = response["payload"]
        self.assertEqual(payload["selected_indices"], (2, 0))
        self.assertEqual(payload["selected_document_ids"], ("c2", "c0"))
        self.assertEqual(payload["termination"], "fixed_step_budget")
        self.assertEqual(payload["trajectory"][0]["q_values"], (0.2, 0.1, 0.9))
        self.assertEqual(payload["trajectory"][1]["available_indices"], (0, 1))
        self.assertNotIn("action", payload["trajectory"][0])
        self.assertEqual(self.reader.calls[0][1], ("gamma", "alpha"))
        attestation = payload["runtime_attestation"]
        self.assertTrue(attestation["safe_weights_only_load"])
        self.assertFalse(attestation["training_called"])
        self.assertFalse(attestation["gold_answer_used"])
        self.assertEqual(self.reader.calls[0][3], 4000)

    def instruction_request(self):
        request = self.request()
        request.update(release_id="acquire.qrag.reader_transfer_factorial",
                       protocol_version=QRAG_INSTRUCTION_PROTOCOL)
        payload = request["payload"]
        payload["artifacts"].update(reader_profile=QRAG_INSTRUCTION_READER_PROFILE,
                                    reader_prompt_sha256=QRAG_INSTRUCTION_PROMPT_SHA256)
        payload["artifact_identity_sha256"] = content_hash(payload["artifacts"])
        task = {"instruction": "Answer and identify the supporting facts.\nDo not drop this.",
                "candidate_annotations": tuple(
                    dict(document_id=f"c{i}", source="visible source",
                         metadata={"official_support_id": json.dumps([f"title {i}", i])})
                    for i in range(3))}
        payload.update(reader_task=task, reader_task_sha256=content_hash(task))
        return request

    def test_instruction_variant_reaches_actual_serialized_reader_request(self) -> None:
        from agentinputlab.backends import GenerationRequest, OpenAICompatibleBackend

        sent = []
        answer = dict(answer="fixture", support_ids=['["title 2", 2]'],
                      evidence_ids=[], evidence_texts=["gamma"])

        def transport(url, headers, body, timeout):
            sent.append(body)
            return dict(choices=[dict(message=dict(content=json.dumps(answer)), finish_reason="stop")],
                        usage=dict(prompt_tokens=33, completion_tokens=19))

        backend = OpenAICompatibleBackend(model_id="fixture", base_url="http://127.0.0.1:1/v1",
            deployment_identity={"revision": "fixture-no-network"}, transport=transport)

        def reader(question, passages, artifacts, *, max_output_tokens,
                   task_instruction, passage_annotations):
            prompt = render_instruction_reader_prompt(question, passages,
                task_instruction=task_instruction, passage_annotations=passage_annotations)
            result = backend.generate(GenerationRequest(system=QRAG_INSTRUCTION_SYSTEM_PROMPT,
                prompt=prompt, max_tokens=max_output_tokens, seed=7))
            return ReaderResult(result.text, result.usage.input_tokens, result.usage.output_tokens)

        self.driver.reader = reader
        request = self.instruction_request()
        response = self.driver.handle(request)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["protocol_version"], QRAG_INSTRUCTION_PROTOCOL)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["max_tokens"], 4000)
        data = json.loads(sent[0]["messages"][-1]["content"])
        self.assertEqual(data["task_instruction"], request["payload"]["reader_task"]["instruction"])
        self.assertEqual([p["document_id"] for p in data["selected_passages"]], ["c2", "c0"])
        self.assertEqual([p["text"] for p in data["selected_passages"]], ["gamma", "alpha"])
        self.assertEqual(data["selected_passages"][0]["metadata"]["official_support_id"], '["title 2", 2]')
        self.assertEqual(parse_instruction_reader_answer(response["payload"]["final_answer"]), answer)
        self.assertEqual(response["payload"]["reader_task_sha256"], request["payload"]["reader_task_sha256"])

    def test_instruction_variant_rejects_lost_task_binding_and_old_callbacks_before_policy(self):
        for drift in ("hash", "order", "profile", "protocol", "callback"):
            with self.subTest(drift=drift):
                request = self.instruction_request()
                if drift == "hash":
                    request["payload"]["reader_task"]["instruction"] = "silently replaced"
                elif drift == "order":
                    request["payload"]["reader_task"]["candidate_annotations"] = ()
                elif drift == "profile":
                    request["payload"]["artifacts"]["reader_prompt_sha256"] = "0" * 64
                elif drift == "protocol":
                    request["protocol_version"] = PROTOCOL
                response = self.driver.handle(request)
                self.assertEqual(response["status"], "error")
        self.assertEqual(self.runtime_calls, 0)
        self.assertEqual(self.reader.calls, [])

    def test_question_only_profile_cannot_silently_adopt_instruction_adapter(self):
        request = self.request()
        request["payload"]["reader_task"] = self.instruction_request()["payload"]["reader_task"]
        self.assertEqual(self.driver.handle(request)["failure_code"], "PROFILE_DRIFT")
        self.assertEqual(self.runtime_calls, 0)

    def test_instruction_json_parser_does_not_manufacture_support_or_salvage_reasoning(self):
        from agentinputlab.contracts import AlgorithmOutputError

        prediction = dict(answer="unknown", support_ids=[], evidence_ids=[], evidence_texts=[])
        self.assertEqual(parse_instruction_reader_answer(json.dumps(prediction)), prediction)
        for raw in ("reasoning then " + json.dumps(prediction), json.dumps({"answer": "incomplete"}),
                    json.dumps(dict(prediction, support_ids=["s", "s"])),
                    json.dumps(dict(prediction, evidence_texts="not an array"))):
            with self.subTest(raw=raw), self.assertRaises(AlgorithmOutputError):
                parse_instruction_reader_answer(raw)

    def test_configured_output_and_remaining_budget_reach_reader(self) -> None:
        for configured, remaining in ((512, 4096), (4000, 96), (6000, 4096)):
            with self.subTest(configured=configured, remaining=remaining):
                request = self.request()
                payload = request["payload"]
                payload["controls"]["max_output_tokens"] = configured
                payload["remaining_budget"]["output_tokens"] = remaining
                payload["reader_budget"] = {
                    "configured_max_output_tokens": configured,
                    "effective_max_output_tokens": min(configured, remaining),
                    "remaining_output_tokens": remaining,
                    "preset": "upstream_default" if configured == 4000 else "custom",
                }
                response = self.driver.handle(request)
                self.assertEqual(response["status"], "ok")
                self.assertEqual(self.reader.calls[-1][3], min(configured, remaining))
                self.assertEqual(response["payload"]["reader_budget"], payload["reader_budget"])

    def test_output_exhaustion_or_invalid_limit_precedes_runtime(self) -> None:
        for configured in (0, -1, True, 1.5, "512"):
            with self.subTest(configured=configured):
                request = self.request()
                request["payload"]["controls"]["max_output_tokens"] = configured
                self.assertEqual(self.driver.handle(request)["status"], "error")
        request = self.request()
        request["payload"]["remaining_budget"]["output_tokens"] = 0
        self.assertEqual(self.driver.handle(request)["failure_code"], "OUTPUT_BUDGET_EXHAUSTED")
        self.assertEqual(self.runtime_calls, 0)
        self.assertEqual(self.reader.calls, [])

    def test_unbound_or_legacy_reader_fails_before_runtime(self) -> None:
        request = self.request()
        request["payload"]["reader_budget"]["effective_max_output_tokens"] = 512
        self.assertEqual(self.driver.handle(request)["failure_code"], "IDENTITY_MISMATCH")
        self.driver.reader = lambda question, passages, artifacts: ReaderResult("old", 1, 1)
        self.assertEqual(self.driver.handle(self.request())["failure_code"], "READER_CONFIG_INVALID")
        self.assertEqual(self.runtime_calls, 0)

    def test_reader_usage_cannot_exceed_effective_limit(self) -> None:
        self.driver.reader = lambda q, p, a, *, max_output_tokens: ReaderResult(
            "overspent", 1, max_output_tokens + 1
        )
        for configured, remaining in ((4000, 4000), (512, 4096), (4000, 96)):
            request = self.request()
            request["payload"]["controls"]["max_output_tokens"] = configured
            request["payload"]["remaining_budget"]["output_tokens"] = remaining
            request["payload"]["reader_budget"] = {
                "configured_max_output_tokens": configured,
                "effective_max_output_tokens": min(configured, remaining),
                "remaining_output_tokens": remaining,
                "preset": "upstream_default" if configured == 4000 else "custom",
            }
            response = self.driver.handle(request)
            self.assertEqual(response["failure_code"], "OUTPUT_BUDGET_EXCEEDED")
            self.assertTrue(response["consumed"])
            self.assertEqual(response["usage"]["model_calls"], 1)
            self.assertEqual(response["usage"]["output_tokens"], min(configured, remaining) + 1)

    def test_invalid_answer_retains_known_usage_without_retry(self) -> None:
        calls = []

        def empty_reader(question, passages, artifacts, *, max_output_tokens):
            calls.append(max_output_tokens)
            return ReaderResult("", 17, 512)

        self.driver.reader = empty_reader
        response = self.driver.handle(self.request())
        self.assertEqual(response["failure_code"], "READER_FAILED")
        self.assertTrue(response["consumed"])
        self.assertEqual(response["usage"]["output_tokens"], 512)
        self.assertEqual(calls, [4000])

    def test_v2_request_cannot_silently_use_the_old_budget_contract(self) -> None:
        request = self.request()
        request["protocol_version"] = "agentinputlab.qrag.frozen.v2"
        self.assertEqual(self.driver.handle(request)["status"], "error")
        self.assertEqual(self.runtime_calls, 0)

    def test_effective_ceiling_reaches_serialized_model_request(self) -> None:
        from agentinputlab.backends.llm import GenerationRequest, OpenAICompatibleBackend

        requests = []

        def transport(url, headers, payload, timeout):
            requests.append(payload)
            return {
                "choices": [{"message": {"content": "fixture answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 3},
            }

        backend = OpenAICompatibleBackend(
            model_id="fixture-reader", base_url="http://127.0.0.1:1/v1",
            deployment_identity={"revision": "fixture-no-network"}, transport=transport,
        )

        def reader(question, passages, artifacts, *, max_output_tokens):
            response = backend.generate(GenerationRequest(
                prompt=question + "\n" + "\n".join(passages),
                max_tokens=max_output_tokens, seed=7,
            ))
            return ReaderResult(response.text, response.usage.input_tokens, response.usage.output_tokens)

        self.driver.reader = reader
        for configured, remaining in ((512, 4096), (4000, 96), (6000, 4096)):
            request = self.request()
            request["payload"]["controls"]["max_output_tokens"] = configured
            request["payload"]["remaining_budget"]["output_tokens"] = remaining
            request["payload"]["reader_budget"] = {
                "configured_max_output_tokens": configured,
                "effective_max_output_tokens": min(configured, remaining),
                "remaining_output_tokens": remaining,
                "preset": "upstream_default" if configured == 4000 else "custom",
            }
            response = self.driver.handle(request)
            self.assertEqual(response["status"], "ok")
            self.assertEqual(requests[-1]["max_tokens"], min(configured, remaining))
            self.assertEqual(requests[-1]["seed"], 7)
        self.assertEqual(len(requests), 3)  # One transport per call, no retries.

    def test_license_and_budget_fail_before_runtime_construction(self) -> None:
        request = self.request()
        request["payload"]["acknowledgements"]["checkpoint_cc_by_4_0_acknowledged"] = False
        self.assertEqual(self.driver.handle(request)["failure_code"], "LICENSE_NOT_ACKNOWLEDGED")

        request = self.request()
        request["payload"]["remaining_budget"]["model_calls"] = 0
        self.assertEqual(self.driver.handle(request)["failure_code"], "MODEL_BUDGET_EXHAUSTED")
        self.assertEqual(self.runtime_calls, 0)

    def test_candidate_order_identity_and_oracle_fields_fail_closed(self) -> None:
        request = self.request()
        request["payload"]["candidate_order_sha256"] = "0" * 64
        self.assertEqual(self.driver.handle(request)["failure_code"], "IDENTITY_MISMATCH")

        request = self.request()
        request["payload"]["candidates"][0]["sf_idx"] = [0]
        self.assertEqual(self.driver.handle(request)["failure_code"], "REQUEST_INVALID")

    def test_checkpoint_hash_and_symlink_mount_fail_closed(self) -> None:
        request = self.request()
        request["payload"]["artifacts"]["checkpoint_file_sha256"] = "f" * 64
        request["payload"]["artifact_identity_sha256"] = content_hash(
            request["payload"]["artifacts"]
        )
        self.assertEqual(self.driver.handle(request)["failure_code"], "CHECKPOINT_MISMATCH")

        manifest_path = self.embedder / "agentinputlab-manifest.json"
        outside = self.root / "outside.json"
        outside.write_text("{}")
        manifest_path.unlink()
        manifest_path.symlink_to(outside)
        self.assertEqual(self.driver.handle(self.request())["failure_code"], "PATH_REJECTED")

    def test_non_argmax_action_is_rejected(self) -> None:
        class BadRuntime:
            def select(self, **_kwargs):
                return SelectorResult(
                    trajectory=(
                        {
                            "available_indices": (0, 1, 2),
                            "q_values": (0.9, 0.2, 0.1),
                            "action_index": 2,
                        },
                        {
                            "available_indices": (0, 1),
                            "q_values": (0.8, 0.1),
                            "action_index": 0,
                        },
                    ),
                    selected_indices=(2, 0),
                    embedding_calls=2,
                )

        driver = QRAGFrozenDriver(
            source_root=self.source,
            checkpoint_root=self.checkpoint,
            embedder_root=self.embedder,
            runtime_factory=lambda *_args: BadRuntime(),
            reader=self.reader,
            manifest_path=self.manifest,
            fixture_mode=True,
        )
        response = driver.handle(self.request())
        self.assertEqual(response["failure_code"], "SELECTOR_INVALID")
        self.assertTrue(response["consumed"])

    def test_commit_addressed_source_snapshot_is_accepted_and_closed(self) -> None:
        driver = QRAGFrozenDriver(
            source_root=self.source,
            checkpoint_root=self.checkpoint,
            embedder_root=self.embedder,
            runtime_factory=lambda *_args: None,
            reader=self.reader,
            manifest_path=self.manifest,
            fixture_mode=False,
        )
        artifacts = self.request()["payload"]["artifacts"]
        self.assertEqual(driver._verify_source(artifacts), "verified_archive_snapshot")

        (self.source / "unreviewed.py").write_text("raise RuntimeError('must not load')\n")
        with self.assertRaisesRegex(Exception, "closure mismatch"):
            driver._verify_source(artifacts)

    @unittest.skipUnless(
        os.environ.get("AGENTINPUTLAB_RUN_REAL_QRAG_IMPORT_SMOKE") == "1",
        "set AGENTINPUTLAB_RUN_REAL_QRAG_IMPORT_SMOKE=1 for user-mounted checkout smoke",
    )
    def test_real_checkout_opt_in_imports_required_symbols(self) -> None:
        checkout = Path(os.environ["AGENTINPUTLAB_QRAG_CHECKOUT"]).resolve()
        compatibility = json.loads(
            resources.files("workers.qrag").joinpath("compatibility_manifest.json").read_text()
        )
        for relative, expected in compatibility["source"]["required_files"].items():
            self.assertEqual(file_hash(checkout / relative), expected, relative)
        code = "\n".join(
            (
                "import sys",
                f"sys.path.insert(0, {str(checkout)!r})",
                "from agentinputlab.algorithms.acquire.qrag.common_support import _unused_vllm_import_compatibility",
                "with _unused_vllm_import_compatibility():",
                "    from rl.agents.pqn import PQN",
                "    from envs.qa_env import QAEnv",
                "    from rl.feedback.feedback import AFeedbackModel, DummyFeedbackModel",
                "class RunnableDummyFeedbackModel(DummyFeedbackModel):",
                "    def reset(self, obs, info): AFeedbackModel.reset(self, obs, info)",
                "assert callable(PQN.select_action) and callable(QAEnv.step)",
                "feedback = RunnableDummyFeedbackModel()",
                "feedback.reset({}, {})",
                "assert feedback.never_terminate and feedback.reward({}, {}) == 0.0",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
