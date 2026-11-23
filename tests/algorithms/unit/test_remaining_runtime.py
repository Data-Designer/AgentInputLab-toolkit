from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from agentinputlab.algorithms.acquire.adaptive_rag import (
    AdaptiveRAGEvaluationLabelAccess,
    AdaptiveRAGPredictionSource,
    AdaptiveRAGPredictionType,
    AdaptiveRAGRecord,
    AdaptiveRAGRouterPrediction,
    AdaptiveRAGSelectorReplay,
    AdaptiveRAGStrategy,
)
from agentinputlab.algorithms.acquire.qrag import (
    QRAG_CHECKPOINT_FILE_SHA256,
    QRAG_SOURCE_MANIFEST_SHA256,
    QRAGArtifactIdentity,
    QRAGControls,
    QRAGLicenseAcknowledgement,
    QRAGNativeWorker,
)
from agentinputlab.algorithms.clarify.ask_or_assume import AskOrAssumeIntentMonitor
from agentinputlab.algorithms.decompose.llm_compiler import (
    LLMCompilerFullBridge,
    LLMCompilerPlannerBridge,
)
from agentinputlab.algorithms.decompose.llm_compiler import bridge as llmcompiler_bridge
from agentinputlab.connectors.tools import InMemoryToolConnector, ToolSpec
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    ContextBlock,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    PatchOp,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.observability import BudgetLedger, ResourceUsage, RunMonitor
from agentinputlab.runner import CompositeInterventionRunner


class FakeMethodWorker:
    worker_id = "remaining-runtime-fixture"

    def __init__(self, payload: Callable[[Any], Mapping[str, Any]]) -> None:
        self._payload = payload
        self.calls = 0

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError("tests use the typed call_controlled seam")

    def call_controlled(self, request: Any) -> Mapping[str, Any]:
        self.calls += 1
        response_payload = dict(self._payload(request))
        return {
            "status": "ok",
            "request_id": request.request_id,
            "upstream_commit": request.upstream_commit,
            "protocol_version": request.protocol_version,
            "payload": response_payload,
            "usage": {
                "model_calls": 2 if response_payload.get("needs_clarification") else 1,
                "retrieval_calls": 0,
                "input_tokens": 24,
                "output_tokens": 8,
            },
        }

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.worker_id,
            "image_digest": "sha256:remaining-runtime-fixture",
        }


class FingerprintedDependency:
    def __init__(self, identity: str) -> None:
        self.identity = identity

    def fingerprint(self) -> Mapping[str, Any]:
        return {"dependency": self.identity, "revision": "fixture-v1"}


def compiler_context(worker: FakeMethodWorker) -> InterventionContext:
    tools = InMemoryToolConnector(
        (ToolSpec("echo", "Return the provided text"),),
        {"echo": lambda text="": text},
    )
    return InterventionContext(
        backend=FingerprintedDependency("compiler-backend"),
        tools=tools,
        external_runtime=worker,
    )


def qrag_artifacts() -> QRAGArtifactIdentity:
    return QRAGArtifactIdentity(
        checkpoint_manifest_sha256="1" * 64,
        checkpoint_file_sha256=QRAG_CHECKPOINT_FILE_SHA256,
        embedder_manifest_sha256="3" * 64,
        source_manifest_sha256=QRAG_SOURCE_MANIFEST_SHA256,
        compatibility_manifest_sha256="5" * 64,
        retriever_revision="retriever-v1",
        retriever_deployment_fingerprint_sha256="6" * 64,
        corpus_sha256="7" * 64,
        index_sha256="8" * 64,
        reader_repository="Qwen/QwQ-32B",
        reader_revision="reader-v1",
        reader_deployment_fingerprint_sha256="9" * 64,
    )


def qrag_acknowledgements(
    *, source: bool = True, checkpoint: bool = True
) -> QRAGLicenseAcknowledgement:
    return QRAGLicenseAcknowledgement(
        source_noassertion_acknowledged=source,
        checkpoint_cc_by_4_0_acknowledged=checkpoint,
        user_mounted_source=source,
        user_mounted_checkpoint=checkpoint,
    )


def qrag_payload(request: Any, *, inconsistent_history: bool = False) -> Mapping[str, Any]:
    artifacts = dict(request.payload["artifacts"])
    candidates = request.payload["candidates"]
    trajectory = [
        {
            "index": 0,
            "available_indices": [0, 1],
            "q_values": [0.2, 0.8],
            "action_index": 1,
            "selected_q_value": 0.8,
            "selected_indices": [1],
            "selected_document_ids": ["d2"],
        },
        {
            "index": 1,
            "available_indices": [0],
            "q_values": [0.7],
            "action_index": 0,
            "selected_q_value": 0.7,
            "selected_indices": [] if inconsistent_history else [1, 0],
            "selected_document_ids": ["d2", "d1"],
        },
    ]
    return {
        "profile_id": request.payload["profile_id"],
        "artifacts": artifacts,
        "artifact_identity_sha256": request.payload["artifact_identity_sha256"],
        "candidate_pool_sha256": request.payload["candidate_pool_sha256"],
        "candidate_order_sha256": request.payload["candidate_order_sha256"],
        "reader_budget": request.payload["reader_budget"],
        "trajectory": trajectory,
        "trajectory_sha256": content_hash(trajectory),
        "selected_indices": [1, 0],
        "selected_document_ids": [candidates[1]["document_id"], candidates[0]["document_id"]],
        "final_answer": "selected answer",
        "termination": "candidate_exhausted",
        "runtime_attestation": {
            "source_commit": request.upstream_commit,
            "source_manifest_sha256": artifacts["source_manifest_sha256"],
            "compatibility_manifest_sha256": artifacts["compatibility_manifest_sha256"],
            "checkpoint_repository": artifacts["checkpoint_repository"],
            "checkpoint_revision": artifacts["checkpoint_revision"],
            "checkpoint_manifest_sha256": artifacts["checkpoint_manifest_sha256"],
            "checkpoint_file_sha256": artifacts["checkpoint_file_sha256"],
            "embedder_repository": artifacts["embedder_repository"],
            "embedder_revision": artifacts["embedder_revision"],
            "official_candidate_builder_called": True,
            "official_pqn_select_action_called": True,
            "official_qa_env_step_called": True,
            "dummy_feedback_used": True,
            "ground_truth_feedback_used": False,
            "gold_answer_used": False,
            "supporting_fact_labels_used": False,
            "reward_used": False,
            "training_called": False,
            "stop_action_used": False,
            "strict_state_dict_load": True,
            "safe_weights_only_load": True,
            "checkpoint_eval_mode": True,
            "source_user_mounted": True,
            "checkpoint_user_mounted": True,
            "checkpoint_deserialization_isolated": True,
            "host_secret_mounts": False,
            "reader_profile": artifacts["reader_profile"],
            "source_attestation": "git_checkout",
        },
    }


def adaptive_prediction(
    strategy: AdaptiveRAGStrategy,
    *,
    label_access: AdaptiveRAGEvaluationLabelAccess = (AdaptiveRAGEvaluationLabelAccess.NONE),
) -> AdaptiveRAGRouterPrediction:
    return AdaptiveRAGRouterPrediction(
        predicted_strategy=strategy,
        router_checkpoint="adaptive-rag-router",
        router_revision="router-commit-0c88670a",
        prediction_source=AdaptiveRAGPredictionSource.UPSTREAM_ROUTER,
        prediction_type=AdaptiveRAGPredictionType.GENERATIVE_CLASSIFICATION,
        prediction_output_hash="a" * 64,
        logits_hash="b" * 64,
        evaluation_label_access=label_access,
    )


class RemainingRuntimeTests(unittest.TestCase):
    def test_llmcompiler_happy_path_materializes_valid_task_dag(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "tasks": [
                    {
                        "task_id": 1,
                        "tool_name": "echo",
                        "arguments": {"text": "hello"},
                        "dependencies": [],
                    }
                ],
                "scheduler_events": [],
                "replans": 0,
                "streaming": True,
                "raw_plan_hash": "plan-v1",
                "termination": "planner_complete",
                "final_answer": "",
            }
        )
        state = FrozenAgentState(query="make a plan", task_id="compiler-1")

        patch = LLMCompilerPlannerBridge(prompt_hash="prompt-v1").propose(
            state,
            compiler_context(worker),
        )
        updated = patch.apply(state)

        self.assertEqual(patch.release_id, "plan.llm_compiler.planner_component")
        self.assertEqual(updated.context[-1].source, "llm_compiler_task_dag")
        self.assertEqual(updated.metadata["llm_compiler_termination"], "planner_complete")

    def test_llmcompiler_rejects_plan_using_undeclared_tool(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "tasks": [
                    {
                        "task_id": 1,
                        "tool_name": "shell",
                        "arguments": {},
                        "dependencies": [],
                    }
                ],
                "scheduler_events": [],
                "replans": 0,
                "streaming": True,
                "raw_plan_hash": "plan-v1",
                "termination": "planner_complete",
                "final_answer": "",
            }
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "unknown tool"):
            LLMCompilerPlannerBridge(prompt_hash="prompt-v1").propose(
                FrozenAgentState(query="make a plan"),
                compiler_context(worker),
            )

    def test_llmcompiler_full_profile_returns_terminal_composite_result(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "tasks": [
                    {
                        "task_id": 1,
                        "tool_name": "echo",
                        "arguments": {"text": "hello"},
                        "dependencies": [],
                    }
                ],
                "scheduler_events": [
                    {"index": 0, "task_id": 1, "state": "planned"},
                    {"index": 1, "task_id": 1, "state": "ready"},
                    {"index": 2, "task_id": 1, "state": "running"},
                    {"index": 3, "task_id": 1, "state": "completed"},
                ],
                "replans": 0,
                "streaming": True,
                "raw_plan_hash": "plan-v1",
                "termination": "finish",
                "final_answer": "terminal answer",
            }
        )

        result = LLMCompilerFullBridge(prompt_hash="prompt-v1").run(
            FrozenAgentState(query="execute a plan", task_id="compiler-full"),
            compiler_context(worker),
        )

        self.assertEqual(result.release_id, "plan.llm_compiler.full")
        self.assertEqual(result.final_answer, "terminal answer")
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.profile_kind, "controlled_adapter")
        self.assertEqual(result.metadata["profile_attestation"], "controlled_default")
        self.assertIn("internal_validation_patch_hash", result.metadata)

    def test_llmcompiler_official_label_requires_allowlist_and_complete_attestation(self) -> None:
        attestation = dict(llmcompiler_bridge._OFFICIAL_NATIVE_ATTESTATION)
        complete = {
            "worker_id": "agentinputlab.llmcompiler.official-native.pinned.v1",
            "upstream_commit": llmcompiler_bridge.LLMCOMPILER_UPSTREAM_COMMIT,
            "official_native_attestation": attestation,
        }

        self.assertEqual(
            llmcompiler_bridge._profile_identity(complete, full_system=True)[0],
            "official_native",
        )
        missing_origin = {
            **complete,
            "official_native_attestation": {
                key: value for key, value in attestation.items() if key != "import_origins_verified"
            },
        }
        self.assertEqual(
            llmcompiler_bridge._profile_identity(missing_origin, full_system=True)[0],
            "controlled_adapter",
        )
        spoofed = {**complete, "worker_id": "untrusted-worker"}
        self.assertEqual(
            llmcompiler_bridge._profile_identity(spoofed, full_system=True)[0],
            "controlled_adapter",
        )

    def test_llmcompiler_full_profile_rejects_incomplete_scheduler_history(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "tasks": [
                    {
                        "task_id": 1,
                        "tool_name": "echo",
                        "arguments": {"text": "first"},
                        "dependencies": [],
                    },
                    {
                        "task_id": 2,
                        "tool_name": "echo",
                        "arguments": {"text": "second"},
                        "dependencies": [1],
                    },
                ],
                "scheduler_events": [
                    {"index": 0, "task_id": 1, "state": "running"},
                    {"index": 1, "task_id": 1, "state": "completed"},
                ],
                "replans": 0,
                "streaming": True,
                "raw_plan_hash": "plan-v1",
                "termination": "finish",
                "final_answer": "unsupported terminal answer",
            }
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "omitted scheduler histories"):
            LLMCompilerFullBridge(prompt_hash="prompt-v1").run(
                FrozenAgentState(query="execute every task"),
                compiler_context(worker),
            )

    def test_adaptive_rag_happy_path_replays_selected_retrieval_arm(self) -> None:
        dependency_hash = "c" * 64
        record = AdaptiveRAGRecord(
            record_id="adaptive-1",
            split="dev",
            query="question",
            strategy=AdaptiveRAGStrategy.SINGLE_RETRIEVAL,
            router_prediction=adaptive_prediction(AdaptiveRAGStrategy.SINGLE_RETRIEVAL),
            dependency_hash=dependency_hash,
            strategy_output_hash="d" * 64,
            contexts=(ContextBlock("retrieved fact", block_id="d1"),),
            selected_answer="answer",
        )
        replay = AdaptiveRAGSelectorReplay(
            (record,),
            split="dev",
            dependency_hash=dependency_hash,
            artifact_id="adaptive-dev-v1",
            artifact_sha256="e" * 64,
        )
        state = FrozenAgentState(query="question", task_id="adaptive-1")

        patch = replay.propose(
            state,
            InterventionContext(run_config={"adaptive_rag_dependency_hash": dependency_hash}),
        )

        self.assertEqual(patch.apply(state).context[-1].block_id, "d1")
        self.assertEqual(patch.metadata["strategy"], "single_retrieval")
        self.assertEqual(patch.metadata["prediction_source"], "upstream_router")
        self.assertEqual(
            patch.metadata["prediction_type"],
            "generative_classification",
        )
        self.assertEqual(patch.metadata["evaluation_label_access"], "none")
        self.assertEqual(patch.metadata["prediction_output_hash"], "a" * 64)
        self.assertEqual(patch.metadata["logits_hash"], "b" * 64)
        self.assertNotIn("silver_label", patch.metadata)

    def test_adaptive_rag_rejects_label_derived_routes_before_replay(self) -> None:
        forbidden_access = (
            AdaptiveRAGEvaluationLabelAccess.GOLD,
            AdaptiveRAGEvaluationLabelAccess.SILVER,
            AdaptiveRAGEvaluationLabelAccess.ORACLE,
        )
        for label_access in forbidden_access:
            with (
                self.subTest(label_access=label_access.value),
                self.assertRaisesRegex(
                    ArtifactError,
                    "cannot use gold, silver, or oracle labels",
                ),
            ):
                adaptive_prediction(
                    AdaptiveRAGStrategy.NO_RETRIEVAL,
                    label_access=label_access,
                )

    def test_adaptive_rag_fails_closed_if_prediction_provenance_is_tampered(
        self,
    ) -> None:
        dependency_hash = "c" * 64
        prediction = adaptive_prediction(AdaptiveRAGStrategy.NO_RETRIEVAL)
        record = AdaptiveRAGRecord(
            record_id="adaptive-1",
            split="dev",
            query="question",
            strategy=AdaptiveRAGStrategy.NO_RETRIEVAL,
            router_prediction=prediction,
            dependency_hash=dependency_hash,
            strategy_output_hash="d" * 64,
        )
        replay = AdaptiveRAGSelectorReplay(
            (record,),
            split="dev",
            dependency_hash=dependency_hash,
            artifact_id="adaptive-dev-v1",
            artifact_sha256="e" * 64,
        )
        object.__setattr__(
            prediction,
            "evaluation_label_access",
            AdaptiveRAGEvaluationLabelAccess.ORACLE,
        )

        with self.assertRaisesRegex(
            ArtifactError,
            "cannot use gold, silver, or oracle labels",
        ):
            replay.propose(
                FrozenAgentState(query="question", task_id="adaptive-1"),
                InterventionContext(),
            )

    def test_adaptive_rag_rejects_route_not_emitted_by_router(self) -> None:
        with self.assertRaisesRegex(ArtifactError, "does not match"):
            AdaptiveRAGRecord(
                record_id="adaptive-1",
                split="dev",
                query="question",
                strategy=AdaptiveRAGStrategy.NO_RETRIEVAL,
                router_prediction=adaptive_prediction(AdaptiveRAGStrategy.SINGLE_RETRIEVAL),
                dependency_hash="c" * 64,
                strategy_output_hash="d" * 64,
            )

    def test_adaptive_rag_rejects_runtime_dependency_bundle_drift(self) -> None:
        dependency_hash = "c" * 64
        replay = AdaptiveRAGSelectorReplay(
            (
                AdaptiveRAGRecord(
                    record_id="adaptive-1",
                    split="dev",
                    query="question",
                    strategy=AdaptiveRAGStrategy.NO_RETRIEVAL,
                    router_prediction=adaptive_prediction(AdaptiveRAGStrategy.NO_RETRIEVAL),
                    dependency_hash=dependency_hash,
                    strategy_output_hash="d" * 64,
                ),
            ),
            split="dev",
            dependency_hash=dependency_hash,
            artifact_id="adaptive-dev-v1",
            artifact_sha256="e" * 64,
        )

        with self.assertRaisesRegex(ArtifactError, "dependency bundle"):
            replay.propose(
                FrozenAgentState(query="question", task_id="adaptive-1"),
                InterventionContext(run_config={"adaptive_rag_dependency_hash": "e" * 64}),
            )

    def test_qrag_happy_path_selects_ordered_candidate_subset(self) -> None:
        state = FrozenAgentState(
            query="question",
            context=(
                ContextBlock("first", block_id="d1"),
                ContextBlock("second", block_id="d2"),
            ),
        )
        worker = FakeMethodWorker(qrag_payload)
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(),
            acknowledgements=qrag_acknowledgements(),
        )

        result = adapter.run(state, InterventionContext(external_runtime=worker))

        self.assertEqual(result.final_answer, "selected answer")
        self.assertEqual(tuple(block.block_id for block in result.evidence), ("d2", "d1"))
        self.assertEqual(result.metadata["selected_indices"], (1, 0))
        self.assertFalse(result.metadata["paper_native_parity_claimed"])

    def test_qrag_instruction_variant_is_explicit_and_source_state_is_preserved(self):
        from agentinputlab.algorithms.acquire.qrag import (
            QRAG_INSTRUCTION_PROMPT_SHA256,
            QRAG_INSTRUCTION_PROTOCOL,
            QRAG_INSTRUCTION_READER_PROFILE,
        )
        from agentinputlab.api import create_intervention
        from agentinputlab.contracts import Message

        artifacts = replace(qrag_artifacts(), reader_profile=QRAG_INSTRUCTION_READER_PROFILE,
                            reader_prompt_sha256=QRAG_INSTRUCTION_PROMPT_SHA256)
        with self.assertRaises(ValidationError):
            replace(qrag_artifacts(), reader_profile=QRAG_INSTRUCTION_READER_PROFILE)
        state = FrozenAgentState(query="Which?", instruction="Retain this exact instruction.\nCite facts.",
            context=(ContextBlock("alpha", block_id="d1", metadata={"official_support_id": "visible1"}),
                     ContextBlock("beta", block_id="d2")),
            budget=ExecutionBudget(max_output_tokens=4096, max_model_calls=8))
        original_hash = state.state_hash
        requests = []

        def reply(request):
            requests.append(request)
            value = dict(qrag_payload(request))
            value["reader_task_sha256"] = request.payload["reader_task_sha256"]
            return value

        worker = FakeMethodWorker(reply)
        method = create_intervention("acquire.qrag.reader_transfer_factorial", artifacts=artifacts,
            acknowledgements=qrag_acknowledgements(), controls=QRAGControls(max_output_tokens=4000))
        result = method.run(state, InterventionContext(external_runtime=worker, seed=17))
        self.assertEqual(state.state_hash, original_hash)
        self.assertEqual(requests[0].protocol_version, QRAG_INSTRUCTION_PROTOCOL)
        self.assertEqual(requests[0].payload["reader_task"]["instruction"], state.instruction)
        self.assertEqual(requests[0].payload["reader_task"]["candidate_annotations"][0]["metadata"],
                         state.context[0].metadata)
        self.assertEqual(result.metadata["input_adapter"], "instruction_preserving_v1")
        self.assertTrue(result.metadata["task_instruction_preserved"])
        with self.assertRaises(ValidationError):
            method.run(replace(state, messages=(Message("user", "history"),)), InterventionContext(external_runtime=worker))
        with self.assertRaises(ValidationError):
            QRAGNativeWorker(artifacts=artifacts, acknowledgements=qrag_acknowledgements()).run(
                state, InterventionContext(external_runtime=worker))
        legacy = create_intervention("acquire.qrag.reader_transfer_factorial", artifacts=replace(
            qrag_artifacts(), reader_profile="factorial_reader_transfer"), acknowledgements=qrag_acknowledgements())
        with self.assertRaises(ValidationError):
            legacy.run(state, InterventionContext(external_runtime=worker))
        self.assertEqual(worker.calls, 1)

    def test_qrag_rejects_inconsistent_selected_index_history(self) -> None:
        worker = FakeMethodWorker(lambda request: qrag_payload(request, inconsistent_history=True))
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(),
            acknowledgements=qrag_acknowledgements(),
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "history is inconsistent"):
            adapter.run(
                FrozenAgentState(
                    query="question",
                    context=(
                        ContextBlock("first", block_id="d1"),
                        ContextBlock("second", block_id="d2"),
                    ),
                ),
                InterventionContext(external_runtime=worker),
            )

    def test_qrag_configured_budget_through_both_registered_factories(self) -> None:
        from agentinputlab.api import create_intervention

        for release, profile in (
            ("acquire.qrag.released_checkpoint_native", "upstream_qwq32b_family_controlled"),
            ("acquire.qrag.reader_transfer_factorial", "factorial_reader_transfer"),
        ):
            for configured, total, consumed in ((512, 4096, 0), (4000, 4096, 4000), (6000, 4096, 0)):
                with self.subTest(release=release, configured=configured, consumed=consumed):
                    sent = []

                    def payload(request, *, sent=sent):
                        sent.append(request)
                        return qrag_payload(request)

                    worker = FakeMethodWorker(payload)
                    adapter = create_intervention(
                        release,
                        artifacts=replace(qrag_artifacts(), reader_profile=profile),
                        acknowledgements=qrag_acknowledgements(),
                        controls=QRAGControls(max_output_tokens=configured),
                    )
                    budget = ExecutionBudget(max_output_tokens=total, max_model_calls=8)
                    ledger = BudgetLedger(budget)
                    ledger.charge_actual(ResourceUsage(output_tokens=consumed))
                    result = adapter.run(
                        FrozenAgentState(query="question", budget=budget, context=(
                            ContextBlock("first", block_id="d1"),
                            ContextBlock("second", block_id="d2"),
                        )),
                        InterventionContext(external_runtime=worker, budget_ledger=ledger),
                    )
                    expected = {
                        "configured_max_output_tokens": configured,
                        "effective_max_output_tokens": min(configured, total - consumed),
                        "remaining_output_tokens": total - consumed,
                        "preset": "upstream_default" if configured == 4000 else "custom",
                    }
                    self.assertEqual(sent[0].payload["reader_budget"], expected)
                    self.assertEqual(result.metadata["reader_budget"], expected)
                    self.assertEqual(worker.calls, 1)
                    self.assertFalse(result.metadata["paper_native_parity_claimed"])

    def test_qrag_budget_zero_prevents_worker_and_invalid_limits_are_rejected(self) -> None:
        for invalid in (0, -1, True, 512.0, "512", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                QRAGControls(max_output_tokens=invalid)
        worker = FakeMethodWorker(qrag_payload)
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(), acknowledgements=qrag_acknowledgements(),
        )
        for budget in (ExecutionBudget(max_output_tokens=0), ExecutionBudget(max_model_calls=0)):
            with self.subTest(budget=budget), self.assertRaises(BudgetExceededError):
                adapter.run(
                    FrozenAgentState(query="question", budget=budget),
                    InterventionContext(external_runtime=worker),
                )
        self.assertEqual(worker.calls, 0)

    def test_qrag_rejects_worker_ignoring_effective_limit_or_binding(self) -> None:
        state = FrozenAgentState(query="question", context=(
            ContextBlock("first", block_id="d1"), ContextBlock("second", block_id="d2"),
        ))
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(), acknowledgements=qrag_acknowledgements(),
            controls=QRAGControls(max_output_tokens=4),
        )
        worker = FakeMethodWorker(qrag_payload)  # Reports eight actual output tokens.
        with self.assertRaisesRegex(AlgorithmOutputError, "effective reader output limit"):
            adapter.run(state, InterventionContext(external_runtime=worker))
        self.assertEqual(worker.calls, 1)

    def test_qrag_monitored_over_cap_reply_preserves_consumed_cost(self) -> None:
        from agentinputlab.integrations.external_workers.method import MethodWorkerRequest

        class MappingWorker(FakeMethodWorker):
            def call(self, payload):
                request = MethodWorkerRequest(**{
                    key: value for key, value in payload.items() if key != "request_id"
                })
                self_test.assertEqual(request.request_id, payload["request_id"])
                return self.call_controlled(request)

        self_test = self
        state = FrozenAgentState(query="question", context=(
            ContextBlock("first", block_id="d1"), ContextBlock("second", block_id="d2"),
        ))
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(), acknowledgements=qrag_acknowledgements(),
            controls=QRAGControls(max_output_tokens=4),
        )
        monitor = RunMonitor()
        worker = MappingWorker(qrag_payload)
        runner = CompositeInterventionRunner(monitor=monitor)
        with self.assertRaisesRegex(AlgorithmOutputError, "effective reader output limit"):
            runner.run(adapter, state, InterventionContext(external_runtime=worker))
        snapshots = monitor.list_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].usage.model_calls, 1)
        self.assertEqual(snapshots[0].usage.output_tokens, 8)
        self.assertEqual(snapshots[0].status.value, "failed")
        self.assertEqual(worker.calls, 1)

    def test_qrag_public_composite_api_monitors_context_bound_worker(self) -> None:
        from agentinputlab.api import create_intervention, run_composite_intervention
        from agentinputlab.integrations.external_workers.method import MethodWorkerRequest

        class MappingWorker(FakeMethodWorker):
            def call(self, payload):
                request = MethodWorkerRequest(**{
                    key: value for key, value in payload.items() if key != "request_id"
                })
                if request.request_id != payload["request_id"] or request.payload["seed"] != 17:
                    raise AssertionError("public request identity or seed drifted")
                return self.call_controlled(request)

        state = FrozenAgentState(query="question", context=(
            ContextBlock("first", block_id="d1"), ContextBlock("second", block_id="d2"),
        ), budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096))
        for release, profile in (
            ("acquire.qrag.released_checkpoint_native", "upstream_qwq32b_family_controlled"),
            ("acquire.qrag.reader_transfer_factorial", "factorial_reader_transfer"),
        ):
            with self.subTest(release=release):
                worker = MappingWorker(qrag_payload)
                method = create_intervention(
                    release, artifacts=replace(qrag_artifacts(), reader_profile=profile),
                    acknowledgements=qrag_acknowledgements(),
                    controls=QRAGControls(max_output_tokens=512),
                )
                result = run_composite_intervention(
                    method, state, InterventionContext(external_runtime=worker, seed=17),
                )
                self.assertEqual(result["record"]["usage"]["model_calls"], 1)
                self.assertEqual(result["record"]["usage"]["output_tokens"], 8)
                self.assertEqual(result["record"]["result"]["metadata"]["reader_budget"]
                    ["effective_max_output_tokens"], 512)
                self.assertEqual(worker.calls, 1)
                self.assertFalse(result["scored"])

    def test_qrag_rejects_worker_changing_the_budget_binding(self) -> None:
        state = FrozenAgentState(query="question", context=(
            ContextBlock("first", block_id="d1"), ContextBlock("second", block_id="d2"),
        ))
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(), acknowledgements=qrag_acknowledgements(),
            controls=QRAGControls(max_output_tokens=512),
        )

        def wrong_budget(request):
            payload = dict(qrag_payload(request))
            payload["reader_budget"] = {"effective_max_output_tokens": 4000}
            return payload

        worker = FakeMethodWorker(wrong_budget)
        with self.assertRaisesRegex(AlgorithmOutputError, "reader budget mismatch"):
            adapter.run(state, InterventionContext(external_runtime=worker))
        self.assertEqual(worker.calls, 1)

    def test_qrag_requires_license_acknowledgement_before_worker_call(self) -> None:
        worker = FakeMethodWorker(qrag_payload)
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(),
        )

        with self.assertRaisesRegex(ExternalDependencyError, "NOASSERTION"):
            adapter.run(
                FrozenAgentState(query="question"),
                InterventionContext(external_runtime=worker),
            )
        self.assertEqual(worker.calls, 0)

    def test_qrag_requires_checkpoint_acknowledgement_before_worker_call(self) -> None:
        worker = FakeMethodWorker(qrag_payload)
        adapter = QRAGNativeWorker(
            artifacts=qrag_artifacts(),
            acknowledgements=qrag_acknowledgements(checkpoint=False),
        )

        with self.assertRaisesRegex(ExternalDependencyError, "CC-BY-4.0"):
            adapter.run(
                FrozenAgentState(query="question"),
                InterventionContext(external_runtime=worker),
            )
        self.assertEqual(worker.calls, 0)

    def test_ask_or_assume_happy_path_requests_exact_worker_question(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "needs_clarification": True,
                "reasons": ["repository is ambiguous"],
                "clarification_question": "Which repository should I edit?",
                "intent_summary": "repository is ambiguous",
                "transition": "await",
                "delegated": True,
                "forced": True,
                "first_turn_skipped": False,
            }
        )
        state = FrozenAgentState(
            query="fix it",
            metadata={"turn_index": 1},
        )
        context = InterventionContext(
            backend=FingerprintedDependency("intent-model"),
            external_runtime=worker,
        )

        patch = AskOrAssumeIntentMonitor().propose(
            state,
            context,
        )

        self.assertIs(patch.operations[0].op, PatchOp.REQUEST_CLARIFICATION)
        self.assertEqual(
            patch.apply(state).metadata["clarification_request"],
            "Which repository should I edit?",
        )
        self.assertEqual(
            patch.apply(state).metadata["ask_or_assume_clarification_turns"],
            1,
        )

    def test_ask_or_assume_rejects_clarification_beyond_turn_limit(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "needs_clarification": True,
                "reasons": ["repository is ambiguous"],
                "clarification_question": "Which repository should I edit?",
                "intent_summary": "repository is ambiguous",
                "transition": "await",
                "delegated": True,
                "forced": True,
                "first_turn_skipped": False,
            }
        )
        state = FrozenAgentState(
            query="fix it",
            metadata={
                "turn_index": 4,
                "ask_or_assume_clarification_turns": 3,
            },
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "max_user_turns"):
            AskOrAssumeIntentMonitor().propose(
                state,
                InterventionContext(
                    backend=FingerprintedDependency("intent-model"),
                    external_runtime=worker,
                ),
            )

    def test_ask_or_assume_first_turn_skip_does_not_call_worker(self) -> None:
        worker = FakeMethodWorker(
            lambda _request: {
                "needs_clarification": False,
                "reasons": [],
                "clarification_question": "",
                "intent_summary": "first turn bypass",
                "transition": "skip_first_turn",
                "delegated": False,
                "forced": False,
                "first_turn_skipped": True,
            }
        )
        state = FrozenAgentState(query="fix it", metadata={"turn_index": 0})

        patch = AskOrAssumeIntentMonitor().propose(
            state,
            InterventionContext(
                backend=FingerprintedDependency("intent-model"),
                external_runtime=worker,
            ),
        )

        self.assertIs(patch.operations[0].op, PatchOp.NO_OP)
        self.assertEqual(worker.calls, 0, "first-turn skip must happen before external inference")

    def test_ask_or_assume_rejects_non_boolean_flags_and_invalid_transition(self) -> None:
        base = {
            "needs_clarification": False,
            "reasons": ["task is sufficiently specified"],
            "clarification_question": "",
            "intent_summary": "clear enough",
            "transition": "resume",
            "delegated": True,
            "forced": False,
            "first_turn_skipped": False,
        }
        mutations = (
            (
                {**base, "needs_clarification": 1},
                "needs_clarification must be a strict boolean",
            ),
            ({**base, "delegated": 1}, "delegated must be boolean"),
            ({**base, "transition": "invented"}, "invalid state transition"),
        )
        for payload, message in mutations:
            with self.subTest(message=message):
                worker = FakeMethodWorker(lambda _request, value=payload: value)
                with self.assertRaisesRegex(AlgorithmOutputError, message):
                    AskOrAssumeIntentMonitor().propose(
                        FrozenAgentState(query="fix it", metadata={"turn_index": 1}),
                        InterventionContext(
                            backend=FingerprintedDependency("intent-model"),
                            external_runtime=worker,
                        ),
                    )

    def test_ask_or_assume_requires_worker_after_first_turn(self) -> None:
        with self.assertRaises(ExternalDependencyError):
            AskOrAssumeIntentMonitor().propose(
                FrozenAgentState(query="fix it", metadata={"turn_index": 1}),
                InterventionContext(backend=FingerprintedDependency("intent-model")),
            )


if __name__ == "__main__":
    unittest.main()
