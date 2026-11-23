from __future__ import annotations

import unittest
from dataclasses import replace

from agentinputlab.algorithms.acquire.ircot import (
    IRCOT_UPSTREAM_COMMIT,
    IRCoTControlledWorker,
    IRCoTControls,
)
from agentinputlab.backends.llm import CallableBackend
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.contracts import (
    BudgetExceededError,
    DependencyBinding,
    DependencyRole,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ProfileKind,
    ValidationError,
)
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import ComparisonArm, ComparisonProfile, InterventionRunner
from agentinputlab.selectors import NoOpIntervention
from agentinputlab.studies import PairedArm, PairedStudyRunner


class FakeIRCoTWorker:
    worker_id = "ircot-fixture"

    def __init__(self, mutate=None):
        self.payload = None
        self.mutate = mutate

    def call(self, payload):
        self.payload = payload
        response = {
            "status": "ok",
            "request_id": payload["request_id"],
            "protocol_version": "agentinputlab.ircot.v1",
            "upstream_commit": IRCOT_UPSTREAM_COMMIT,
            "initial_query": payload["question"],
            "initial_document_ids": ["d1"],
            "steps": [
                {
                    "index": 0,
                    "cot_step": "We need a second supporting fact.",
                    "latest_cot_sentence": "We need a second supporting fact.",
                    "retrieval_query": "We need a second supporting fact.",
                    "retrieved_document_ids": ["d2"],
                    "new_document_ids": ["d2"],
                    "active_document_ids": ["d1", "d2"],
                    "reset_applied": False,
                }
            ],
            "documents": [
                {
                    "document_id": "d1",
                    "text": "First fact.",
                    "title": "One",
                    "corpus_revision": "corpus-v1",
                    "score": 0.9,
                },
                {
                    "document_id": "d2",
                    "text": "Second fact.",
                    "title": "Two",
                    "corpus_revision": "corpus-v1",
                    "score": 0.8,
                },
            ],
            "active_document_ids": ["d1", "d2"],
            "exit_reason": "controller",
            "answer": "combined",
            "usage": {
                "retrieval_calls": 2,
                "model_calls": 2,
                "input_tokens": 64,
                "output_tokens": 128,
            },
        }
        if self.mutate:
            self.mutate(response)
        return response

    def fingerprint(self):
        return {"worker_id": self.worker_id, "image_digest": "sha256:fixture"}


def context():
    return InterventionContext(
        backend=CallableBackend(
            lambda _: "unused",
            model_id="generator",
            revision="fixture-v1",
            callable_id="tests.ircot.generator.v1",
        ),
        retriever=InMemoryBM25Retriever(
            [Document(doc_id="local", text="fixture", corpus_revision="corpus-v1")]
        ),
    )


class IRCoTWorkerTests(unittest.TestCase):
    def algorithm(self, worker=None):
        return IRCoTControlledWorker(
            client=worker or FakeIRCoTWorker(),
            corpus_revision="corpus-v1",
            index_revision="index-v1",
            prompt_hash="prompt-v1",
            controls=IRCoTControls(max_steps=2),
        )

    def test_complete_loop_is_delegated_and_validated(self):
        worker = FakeIRCoTWorker()
        state = FrozenAgentState(query="multi-hop question", task_id="case-1")
        patch = self.algorithm(worker).propose(state, context())

        self.assertEqual(worker.payload["question"], state.query)
        self.assertEqual(worker.payload["upstream_commit"], IRCOT_UPSTREAM_COMMIT)
        materialized = patch.apply(state)
        self.assertEqual([item.block_id for item in materialized.context], ["d1", "d2"])
        self.assertEqual(patch.trace.counters["reasoning_steps"], 1)
        self.assertEqual(
            patch.metadata["worker_fingerprint"]["image_digest"],
            "sha256:fixture",
        )

    def test_followup_query_must_be_latest_cot_sentence(self):
        worker = FakeIRCoTWorker(
            lambda response: response["steps"][0].update(
                {"retrieval_query": "rewritten behind the adapter"}
            )
        )
        with self.assertRaisesRegex(ValidationError, "latest CoT sentence"):
            self.algorithm(worker).propose(FrozenAgentState(query="q"), context())

    def test_dedup_violation_is_rejected(self):
        def mutate(response):
            response["steps"][0]["retrieved_document_ids"] = ["d1"]
            response["steps"][0]["new_document_ids"] = ["d1"]
            response["steps"][0]["active_document_ids"] = ["d1"]
            response["active_document_ids"] = ["d1"]

        with self.assertRaisesRegex(ValidationError, "deduplication"):
            self.algorithm(FakeIRCoTWorker(mutate)).propose(
                FrozenAgentState(query="q"),
                context(),
            )

    def test_unsolicited_document_record_is_rejected(self):
        def mutate(response):
            response["documents"].append(
                {
                    "document_id": "not-retrieved",
                    "text": "Worker-injected text.",
                    "title": "Injected",
                    "corpus_revision": "corpus-v1",
                }
            )

        with self.assertRaisesRegex(ValidationError, "exactly match"):
            self.algorithm(FakeIRCoTWorker(mutate)).propose(
                FrozenAgentState(query="q"),
                context(),
            )

    def test_reset_policy_is_enforced(self):
        def mutate(response):
            response["steps"][0]["reset_applied"] = True
            response["steps"][0]["active_document_ids"] = ["d2"]
            response["active_document_ids"] = ["d2"]

        algorithm = IRCoTControlledWorker(
            client=FakeIRCoTWorker(mutate),
            corpus_revision="corpus-v1",
            index_revision="index-v1",
            prompt_hash="prompt-v1",
            controls=IRCoTControls(max_steps=2, reset_policy="never"),
        )
        with self.assertRaisesRegex(ValidationError, "reset_policy"):
            algorithm.propose(FrozenAgentState(query="q"), context())

    def test_worker_failure_is_typed(self):
        class Failed(FakeIRCoTWorker):
            def call(self, payload):
                return {
                    "status": "error",
                    "request_id": payload["request_id"],
                    "failure_code": "ENVIRONMENT_BOOT_FAILURE",
                    "consumed": False,
                }

        with self.assertRaises(ExternalDependencyError):
            self.algorithm(Failed()).propose(FrozenAgentState(query="q"), context())

    def test_worker_document_metadata_cannot_spoof_provenance(self):
        def mutate(response):
            response["documents"][0]["metadata"] = {"corpus_revision": "spoofed-corpus"}

        with self.assertRaisesRegex(ValidationError, "reserved provenance"):
            self.algorithm(FakeIRCoTWorker(mutate)).propose(
                FrozenAgentState(query="q"),
                context(),
            )

    def test_controlled_runner_rejects_hidden_worker_client(self):
        worker = FakeIRCoTWorker()
        with self.assertRaisesRegex(ValidationError, "runtime dependencies"):
            InterventionRunner().run(
                self.algorithm(worker),
                FrozenAgentState(query="q"),
                context(),
            )
        self.assertIsNone(worker.payload)

    def test_context_worker_usage_maps_retrieval_rounds_to_retrieval_calls(self):
        worker = FakeIRCoTWorker()
        algorithm = IRCoTControlledWorker(
            client=None,
            corpus_revision="corpus-v1",
            index_revision="index-v1",
            prompt_hash="prompt-v1",
            controls=IRCoTControls(max_steps=2),
        )
        record = InterventionRunner().run(
            algorithm,
            FrozenAgentState(query="multi-hop question", task_id="case-1"),
            replace(context(), external_runtime=worker),
        )
        self.assertEqual(record.usage.model_calls, 2)
        self.assertEqual(record.usage.retrieval_calls, 2)
        self.assertEqual(record.usage.input_tokens, 64)
        self.assertEqual(record.usage.output_tokens, 128)
        self.assertEqual(record.usage.unpriced_model_calls, 2)
        self.assertEqual(record.patch.trace.counters["retrieval_calls"], 2)
        self.assertNotIn("retrieval_rounds", record.patch.trace.counters)

    def test_model_budget_rejects_before_ircot_worker_side_effect(self):
        worker = FakeIRCoTWorker()
        algorithm = IRCoTControlledWorker(
            client=None,
            corpus_revision="corpus-v1",
            index_revision="index-v1",
            prompt_hash="prompt-v1",
            controls=IRCoTControls(max_steps=2),
        )
        with self.assertRaisesRegex(BudgetExceededError, "model_calls"):
            InterventionRunner().run(
                algorithm,
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_model_calls=0),
                ),
                replace(context(), external_runtime=worker),
            )
        self.assertIsNone(worker.payload)

    def test_retrieval_budget_rejects_before_ircot_worker_side_effect(self):
        worker = FakeIRCoTWorker()
        algorithm = IRCoTControlledWorker(
            client=None,
            corpus_revision="corpus-v1",
            index_revision="index-v1",
            prompt_hash="prompt-v1",
            controls=IRCoTControls(max_steps=2),
        )
        with self.assertRaisesRegex(BudgetExceededError, "retrieval_calls"):
            InterventionRunner().run(
                algorithm,
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_retrieval_calls=0),
                ),
                replace(context(), external_runtime=worker),
            )
        self.assertIsNone(worker.payload)

    def test_structured_failure_with_usage_is_accounted_before_typed_error(self):
        class FailedWithUsage(FakeIRCoTWorker):
            def call(self, payload):
                return {
                    "status": "error",
                    "request_id": payload["request_id"],
                    "failure_code": "INDEX_SERVICE_FAILURE",
                    "usage": {
                        "model_calls": 2,
                        "retrieval_calls": 1,
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "cost_usd": 0.2,
                    },
                }

        worker = FailedWithUsage()
        algorithm = IRCoTControlledWorker(
            client=None,
            corpus_revision="corpus-v1",
            index_revision="index-v1",
            prompt_hash="prompt-v1",
        )
        runner = InterventionRunner()
        with self.assertRaisesRegex(ExternalDependencyError, "INDEX_SERVICE_FAILURE"):
            runner.run(
                algorithm,
                FrozenAgentState(query="q"),
                replace(context(), external_runtime=worker),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 2)
        self.assertEqual(snapshot.usage.retrieval_calls, 1)
        self.assertEqual(snapshot.usage.input_tokens, 20)
        self.assertEqual(snapshot.usage.output_tokens, 5)
        self.assertEqual(snapshot.usage.cost_usd, 0.2)

    def test_boot_failure_bypasses_usage_parsing_and_invalidates_paired_cell(self):
        class BootFailure(FakeIRCoTWorker):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def call(self, payload):
                self.calls += 1
                return {
                    "status": "error",
                    "request_id": payload["request_id"],
                    "failure_code": "ENVIRONMENT_BOOT_FAILURE",
                    "consumed": False,
                }

        failed_worker = BootFailure()
        monitor = RunMonitor()
        bindings = tuple(
            DependencyBinding(slot, DependencyRole.NUISANCE, f"fixture-{slot}", "v1")
            for slot in ("backend", "retriever", "external_runtime")
        )
        failed_algorithm = IRCoTControlledWorker(
            client=None,
            corpus_revision="corpus-v1",
            index_revision="index-v1",
            prompt_hash="prompt-v1",
            controls=IRCoTControls(max_steps=2),
        )
        failed_context = replace(
            context(),
            external_runtime=failed_worker,
            monitor=monitor,
        )
        healthy_context = replace(
            context(),
            external_runtime=FakeIRCoTWorker(),
        )

        with self.assertRaisesRegex(ExternalDependencyError, "ENVIRONMENT_BOOT_FAILURE"):
            PairedStudyRunner().run(
                FrozenAgentState(query="q", task_id="case"),
                ComparisonProfile(
                    "ircot-boot-failure",
                    ProfileKind.CONTROLLED_ADAPTER,
                    required_slots=("backend", "retriever", "external_runtime"),
                ),
                (
                    PairedArm(
                        ComparisonArm("failed", failed_algorithm.release_id, bindings),
                        failed_algorithm,
                        failed_context,
                    ),
                    PairedArm(
                        ComparisonArm("healthy", "baseline.no_op", bindings),
                        NoOpIntervention(),
                        healthy_context,
                    ),
                ),
            )
        self.assertEqual(failed_worker.calls, 1)
        snapshot = monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 0)
        self.assertEqual(snapshot.usage.retrieval_calls, 0)
        self.assertEqual(snapshot.failure.exception_type, "ExternalDependencyError")


if __name__ == "__main__":
    unittest.main()
