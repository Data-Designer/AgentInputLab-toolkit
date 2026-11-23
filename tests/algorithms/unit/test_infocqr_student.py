"""Offline student contracts. Fixtures are not InfoCQR/model acceptance results."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentinputlab.algorithms.reformulate.infocqr.student import (
    INFOCQR_STUDENT_MODE,
    INFOCQR_STUDENT_SOURCE_HASHES,
    InfoCQRStudentExternal,
    InfoCQRStudentRequest,
)
from agentinputlab.api import ToolkitService, create_intervention
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.integrations.external_workers.infocqr_student import (
    STUDENT_ARTIFACT_SCHEMA,
    InfoCQRStudentRuntime,
)
from agentinputlab.runner import InterventionRunner

RELEASE = "rewrite.infocqr.t5_student_external"
REVISION = "2953c461aa0c6817b1b7656dbd5168050227b31d"


def state(**kwargs):
    return FrozenAgentState(query="Who founded it?", task_id="fixture-2", metadata={
        "conversation_source": "fixture", "conversation_no": 1, "turn_no": 2,
        "new_context": ["What is Example?", "Example is a test organization."],
        "evaluator_gold": "not visible to student",
    }, **kwargs)


class StudentFixture:
    def __init__(self, *, text="Who founded Example?", load_error=False):
        self.requests, self.text, self.load_error = [], text, load_error

    def fingerprint(self):
        return {"mode": INFOCQR_STUDENT_MODE, "upstream_revision": REVISION,
                "checkpoint_manifest_sha256": "a"*64, "dataset_revision": "fixture-v1",
                "provenance_kind": "source_recipe_finetuned"}

    def run(self, request):
        self.requests.append(request)
        if request.phase == "discard":
            return {"status": "discarded", "request_id": request.request_id,
                    "identity": self.fingerprint(), "usage": {"model_calls": 0,
                    "retrieval_calls": 0, "input_tokens": 0, "output_tokens": 0}}
        if request.phase == "prepare":
            if self.load_error:
                raise ExternalDependencyError("fixture execution dependency missing before generation")
            return {"request_id": request.request_id, "identity": self.fingerprint(),
                    "status": "ready", "input_tokens": 25,
                    "usage": {"model_calls": 0, "retrieval_calls": 0,
                              "input_tokens": 0, "output_tokens": 0}}
        return {"request_id": request.request_id, "identity": self.fingerprint(),
                "status": "ok" if self.text else "empty_output", "rewrite": self.text,
                "source_text_hash": content_hash("offline fixture"),
                "usage": {"model_calls": 1, "retrieval_calls": 0,
                          "input_tokens": 25, "output_tokens": 9}}


def method(**kwargs):
    return create_intervention(RELEASE, checkpoint_manifest_sha256="a"*64,
        dataset_revision="fixture-v1", source_license_acknowledged=True, **kwargs)


class InfoCQRStudentTests(unittest.TestCase):
    def test_registered_student_preserves_non_query_state_and_accounts_auxiliary(self):
        runtime = StudentFixture()
        original = state()
        record = InterventionRunner().run(method(), original,
            InterventionContext(external_runtime=runtime, seed=17))
        updated = record.patch.apply(original)
        self.assertEqual(updated.query, "Who founded Example?")
        self.assertEqual(updated.context, original.context)
        self.assertEqual(updated.messages, original.messages)
        self.assertEqual(updated.instruction, original.instruction)
        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual((record.usage.input_tokens, record.usage.output_tokens), (25, 9))
        self.assertEqual([r.phase for r in runtime.requests], ["prepare", "generate"])
        self.assertEqual([r.seed for r in runtime.requests], [17, 17])
        self.assertNotIn("not visible to student", repr(runtime.requests))
        self.assertTrue(record.patch.metadata["fixed_auxiliary_not_downstream_backbone"])

    def test_prepare_failure_is_not_a_consumed_model_attempt(self):
        runtime = StudentFixture(load_error=True)
        runner = InterventionRunner()
        with self.assertRaisesRegex(ExternalDependencyError, "before generation"):
            runner.run(method(), state(), InterventionContext(external_runtime=runtime))
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(runner.monitor.list_snapshots()[0].usage.model_calls, 0)

    def test_empty_student_is_one_real_failed_generation_no_teacher_or_cache_retry(self):
        runtime = StudentFixture(text="")
        runner = InterventionRunner()
        with self.assertRaisesRegex(AlgorithmOutputError, "empty rewrite"):
            runner.run(method(), state(), InterventionContext(external_runtime=runtime))
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 1)
        self.assertEqual(snapshot.usage.output_tokens, 9)
        self.assertEqual(len(runtime.requests), 2)

    def test_wrong_artifact_dataset_mode_or_author_label_rejected_before_prepare(self):
        for field, value in (("checkpoint_manifest_sha256", "b"*64),
                             ("dataset_revision", "other"), ("mode", "teacher")):
            with self.subTest(field=field):
                runtime = StudentFixture()
                identity = dict(runtime.fingerprint(), **{field: value})
                runtime.fingerprint = lambda identity=identity: identity
                with self.assertRaises(ArtifactError):
                    method().propose(state(), InterventionContext(external_runtime=runtime))
                self.assertEqual(runtime.requests, [])
        runtime = StudentFixture()
        with self.assertRaisesRegex(ArtifactError, "author"):
            method(profile_kind="official_native").propose(state(), InterventionContext(external_runtime=runtime))

    def test_native_budget_guard_and_input_cap_are_pre_generation(self):
        for budget in (ExecutionBudget(max_model_calls=0), ExecutionBudget(max_output_tokens=62)):
            runtime = StudentFixture()
            with self.assertRaises(BudgetExceededError):
                method().propose(state(budget=budget), InterventionContext(external_runtime=runtime))
            self.assertEqual(runtime.requests, [])
        runtime = StudentFixture()
        runner = InterventionRunner()
        with self.assertRaises(BudgetExceededError):
            runner.run(method(), state(budget=ExecutionBudget(max_context_tokens=24)),
                       InterventionContext(external_runtime=runtime))
        self.assertEqual(runner.monitor.list_snapshots()[0].usage.model_calls, 0)

    def test_cannot_relabel_arbitrary_QA_or_incomplete_history_as_QReCC(self):
        original = state()
        for changes in ({"benchmark_id": "hotpotqa.distractor"}, {"new_context": ["unpaired"]}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                method().propose(replace(original, metadata={**original.metadata, **changes}),
                                 InterventionContext(external_runtime=StudentFixture()))

    def test_python_and_operator_owned_sdk_surface(self):
        self.assertIsInstance(method(), InfoCQRStudentExternal)
        runtime = StudentFixture()
        service = ToolkitService(factories={RELEASE: method},
            context_factory=lambda seed: InterventionContext(external_runtime=runtime, seed=seed))
        response = service.execute({"release_id": RELEASE, "seed": 17,
            "state": {"query": state().query, "task_id": "fixture", "metadata": dict(state().metadata)}})
        self.assertFalse(response["scored"])
        self.assertEqual(response["state"]["query"], "Who founded Example?")

    def test_artifact_and_request_invalid_inputs(self):
        with self.assertRaises(ValidationError):
            InfoCQRStudentExternal(checkpoint_manifest_sha256="a"*64, dataset_revision="fixture-v1")
        runtime = StudentFixture()
        with self.assertRaises(ExternalDependencyError):
            method().propose(state(), InterventionContext())
        request = InfoCQRStudentRequest(state().state_hash, "id", "query", "fixture", 1, 1,
            (), "fixture-v1", "a"*64, 17)
        self.assertEqual(request.external_reservation(), {"model_calls": 0})
        self.assertEqual(replace(request, phase="generate").external_reservation(), {"model_calls": 1})
        with self.assertRaises(ValidationError):
            replace(request, phase="teacher_fallback")
        self.assertEqual(runtime.requests, [])


class StudentArtifactTests(unittest.TestCase):
    """Synthetic checkpoint bytes test the mount boundary, never model loading."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/"source"
        self.checkpoint = self.root/"checkpoint"
        self.source.mkdir()
        self.checkpoint.mkdir()
        self.hashes = {}
        for relative in INFOCQR_STUDENT_SOURCE_HASHES:
            path = self.source/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = b"offline source fixture; not executable\n"
            path.write_bytes(raw)
            self.hashes[relative] = hashlib.sha256(raw).hexdigest()
        self.files = {"config.json": json.dumps({"model_type": "t5", "is_encoder_decoder": True}).encode(),
                      "model.safetensors": b"not real model weights; mount fixture only"}
        for name, raw in self.files.items():
            (self.checkpoint/name).write_bytes(raw)
        self.artifact = {
            "schema_version": STUDENT_ARTIFACT_SCHEMA, "upstream_revision": REVISION,
            "dataset_id": "qrecc", "dataset_revision": "fixture-v1",
            "provenance_kind": "source_recipe_finetuned", "training_receipt_sha256": "b"*64,
            "training_source_sha256": self.hashes["T5QR/T5train.py"],
            "files": {n: hashlib.sha256(b).hexdigest() for n,b in self.files.items()},
        }

    def runtime(self):
        path = self.root/"manifest.json"
        path.write_text(json.dumps(self.artifact))
        with patch("agentinputlab.integrations.external_workers.infocqr_student.INFOCQR_STUDENT_SOURCE_HASHES", self.hashes):
            return InfoCQRStudentRuntime(checkout=self.source, checkpoint=self.checkpoint,
                manifest=path, expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_license_acknowledged=True)

    def test_closed_file_tree_changed_bytes_and_added_configuration_are_rejected(self):
        runtime = self.runtime()
        (self.checkpoint/"generation_config.json").write_text("{}")
        with self.assertRaisesRegex(ArtifactError, "file set changed"):
            runtime.fingerprint()
        (self.checkpoint/"generation_config.json").unlink()
        (self.checkpoint/"model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ArtifactError, "changed since"):
            runtime.fingerprint()

    def test_generic_base_model_receipt_and_pickle_weights_are_rejected(self):
        self.artifact["provenance_kind"] = "generic_t5_base"
        with self.assertRaises(ArtifactError):
            self.runtime()
        self.artifact["provenance_kind"] = "source_recipe_finetuned"
        self.artifact["files"]["pytorch_model.bin"] = "a"*64
        with self.assertRaises(ArtifactError):
            self.runtime()

    def test_bad_hash_unlisted_file_and_symbolic_checkpoint_are_rejected(self):
        (self.checkpoint/"model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ArtifactError, "checksum"):
            self.runtime()
        (self.checkpoint/"model.safetensors").write_bytes(self.files["model.safetensors"])
        (self.checkpoint/"unlisted").write_text("extra")
        with self.assertRaisesRegex(ArtifactError, "unlisted"):
            self.runtime()
        (self.checkpoint/"unlisted").unlink()
        linked = self.root/"linked"
        linked.symlink_to(self.checkpoint, target_is_directory=True)
        self.checkpoint = linked
        with self.assertRaisesRegex(ArtifactError, "not mounted"):
            self.runtime()

    def test_generation_removes_labels_counts_decoder_tokens_and_preserves_failure_usage_unknown(self):
        runtime = self.runtime()
        model_calls, seeds = [], []

        class Batch(dict):
            def to(self, device):
                self.device = device
                return self

        def collate(rows):
            self.assertEqual(rows, [{"source": "offline serialized fixture", "target": ""}])
            return Batch(input_ids="fixture-input", attention_mask=SimpleNamespace(sum=lambda: 25),
                         decoder_input_ids="unused", labels="unused")

        def generate(**kwargs):
            model_calls.append(kwargs)
            self.assertNotIn("labels", kwargs)
            self.assertNotIn("decoder_input_ids", kwargs)
            self.assertEqual(kwargs["max_length"], 64)
            return SimpleNamespace(shape=(1, 10))

        runtime._model = SimpleNamespace(generate=generate)
        runtime._tokenizer = SimpleNamespace(batch_decode=lambda outputs, **kwargs: ["offline rewrite fixture"])
        runtime._collator = collate
        torch = SimpleNamespace(random=SimpleNamespace(
            fork_rng=lambda **kwargs: nullcontext(),
            default_generator=SimpleNamespace(manual_seed=seeds.append)),
            inference_mode=nullcontext)
        request = InfoCQRStudentRequest(state().state_hash, "fixture", "query", "fixture", 1, 1, (),
            "fixture-v1", runtime.manifest_sha256, 17)
        def serializer():
            return lambda lines, rewrite_type: {"source": ["offline serialized fixture"]}
        with patch.object(runtime, "_serializer", serializer), patch(
                "agentinputlab.integrations.external_workers.infocqr_student.importlib.import_module",
                return_value=torch):
            preflight = runtime.run(request)
            self.assertEqual(preflight["status"], "ready")
            self.assertEqual(preflight["usage"]["model_calls"], 0)
            self.assertEqual(model_calls, [])
            generated = runtime.run(replace(request, phase="generate"))
            self.assertEqual(generated["usage"], {"model_calls": 1, "retrieval_calls": 0,
                                                  "input_tokens": 25, "output_tokens": 9})
            self.assertEqual(seeds, [17])
            self.assertEqual(runtime.last_attempt["status"], "finished")
            with self.assertRaisesRegex(ValidationError, "unconsumed"):
                runtime.run(replace(request, phase="generate"))
            runtime._tokenizer.batch_decode = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("decode fixture"))
            runtime.run(request)
            decoded = runtime.run(replace(request, phase="generate"))
            self.assertEqual(decoded["status"], "decode_failed")
            self.assertEqual(decoded["usage"]["output_tokens"], 9)
            with self.assertRaisesRegex(AlgorithmOutputError, "usage retained"):
                request.raise_for_external_status(decoded)
            runtime._model.generate = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline failure fixture"))
            runtime.run(request)
            with self.assertRaises(RuntimeError):
                runtime.run(replace(request, phase="generate"))
            self.assertIsNone(runtime.last_attempt["output_tokens"])
            self.assertEqual(runtime.last_attempt["status"], "failed_output_usage_unknown")


if __name__ == "__main__":
    unittest.main()
