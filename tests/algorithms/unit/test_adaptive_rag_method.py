from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping
from typing import Any

from agentinputlab.algorithms.acquire.adaptive_rag import (
    AdaptiveRAGArtifactIdentity,
    AdaptiveRAGControlledWorker,
    AdaptiveRAGControls,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
)
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.runner import CompositeInterventionRunner

_SOURCE_TREE = "8dadc754562d61d5d6e6e3ee2fc3aa4481763320"
_MANIFEST_HASH = "85299456b7f858734c34de84fe4625e0a014651242001d7242ca33b260be7382"
_SOURCE_CLOSURE_HASH = "43677c9862dd16a69aafadab36a7ac9ca9cdfb037c8854321e64a2a3df14597c"
_EVENT_SCHEMA = "agentinputlab.adaptive-rag.supervisor-event.v1"


def artifacts() -> AdaptiveRAGArtifactIdentity:
    return AdaptiveRAGArtifactIdentity(
        router_checkpoint="starsuzi/adaptive-rag-router",
        worker_fingerprint_sha256=content_hash(
            {
                "worker_id": "adaptive-rag-method-fixture",
                "image_digest": "sha256:adaptive-rag-fixture",
                "source_revision": "fixture-v1",
            }
        ),
        router_revision="router-checkpoint-v1",
        router_snapshot_sha256="0" * 64,
        router_weights_sha256="1" * 64,
        router_tokenizer_sha256="5" * 64,
        base_model_revision="flan-t5-xl-v1",
        generator_artifact_sha256="6" * 64,
        generator_config_sha256="7" * 64,
        retriever_revision="bm25-v1",
        retriever_implementation_sha256="8" * 64,
        retriever_config_sha256="9" * 64,
        corpus_revision="wiki-2018-v1",
        corpus_manifest_sha256="a" * 64,
        index_revision="elastic-7.10.2-v1",
        index_manifest_sha256="b" * 64,
        no_retrieval_prompt_sha256="2" * 64,
        single_retrieval_prompt_sha256="3" * 64,
        ircot_prompt_sha256="4" * 64,
    )


def _request_payload(request: Any) -> Mapping[str, Any]:
    return request["payload"] if isinstance(request, Mapping) else request.payload


def _request_value(request: Any, name: str) -> str:
    return str(request[name] if isinstance(request, Mapping) else getattr(request, name))


def _event_usage(
    *,
    model_calls: int = 0,
    retrieval_calls: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, int | float]:
    return {
        "model_calls": model_calls,
        "retrieval_calls": retrieval_calls,
        "tool_calls": 0,
        "embedding_calls": 0,
        "unpriced_model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cpu_seconds": 0.0,
        "accelerator_seconds": 0.0,
        "cost_usd": 0.0,
        "compute_cost_usd": 0.0,
    }


def _execution_evidence(
    request: Any,
    *,
    strategy: str,
    route_logits: Mapping[str, float],
    answer: str,
    initial_retrieval: Mapping[str, Any] | None,
    ircot_steps: list[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _request_payload(request)
    identity = payload["artifacts"]
    operations = ["router.first_decoder_step"]
    results: list[Mapping[str, Any]] = [
        {
            "question_sha256": content_hash(payload["question"]),
            "route_logits": dict(route_logits),
        }
    ]
    if strategy == "no_retrieval":
        operations.append("answer.no_retrieval")
        results.append(
            {"text_sha256": content_hash(answer), "document_ids": (), "reset_retrieval": False}
        )
    elif strategy == "single_retrieval":
        operations.extend(("retrieve.single.initial", "answer.single_retrieval"))
        assert initial_retrieval is not None
        results.extend(
            (
                {
                    "query": initial_retrieval["query"],
                    "query_sha256": content_hash(initial_retrieval["query"]),
                    "top_k": 15,
                    "document_ids": tuple(initial_retrieval["document_ids"]),
                    "documents_sha256": "0" * 64,
                },
                {
                    "text_sha256": content_hash(answer),
                    "document_ids": tuple(initial_retrieval["document_ids"]),
                    "reset_retrieval": False,
                },
            )
        )
    else:
        assert initial_retrieval is not None
        operations.append("retrieve.ircot.initial")
        results.append(
            {
                "query": initial_retrieval["query"],
                "query_sha256": content_hash(initial_retrieval["query"]),
                "top_k": 6,
                "document_ids": tuple(initial_retrieval["document_ids"]),
                "documents_sha256": "0" * 64,
            }
        )
        for step in ircot_steps:
            operations.extend(
                (f"ircot.reason.{step['index']}", f"retrieve.ircot.step.{step['index']}")
            )
            results.extend(
                (
                    {
                        "text_sha256": content_hash(step["cot_step"]),
                        "document_ids": tuple(step["active_document_ids"]),
                        "reset_retrieval": False,
                    },
                    {
                        "query": step["retrieval_query"],
                        "query_sha256": content_hash(step["retrieval_query"]),
                        "top_k": 6,
                        "document_ids": tuple(step["retrieved_document_ids"]),
                        "documents_sha256": "0" * 64,
                    },
                )
            )
        operations.append("answer.ircot")
        results.append(
            {
                "text_sha256": content_hash(answer),
                "document_ids": tuple(ircot_steps[-1]["active_document_ids"]),
                "reset_retrieval": False,
            }
        )
    model_indices = [
        index for index, operation in enumerate(operations) if not operation.startswith("retrieve.")
    ]
    model_inputs = [0] * len(model_indices)
    model_outputs = [0] * len(model_indices)
    model_inputs[0] = 32
    model_outputs[-1] = 8
    before = {
        "model_calls": payload["remaining_budget"]["model_calls"],
        "retrieval_calls": payload["remaining_budget"]["retrieval_calls"],
        "input_tokens": payload["remaining_budget"]["context_tokens"],
        "output_tokens": payload["remaining_budget"]["output_tokens"],
    }
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    model_position = 0
    for index, (operation, event_result) in enumerate(zip(operations, results, strict=True)):
        is_model = index in model_indices
        input_tokens = model_inputs[model_position] if is_model else 0
        output_tokens = model_outputs[model_position] if is_model else 0
        if is_model:
            model_position += 1
        usage = _event_usage(
            model_calls=1 if is_model else 0,
            retrieval_calls=0 if is_model else 1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        requested_output_cap = (
            1
            if operation == "router.first_decoder_step"
            else payload["controls"]["max_output_tokens"]
        )
        unsigned = {
            "schema": _EVENT_SCHEMA,
            "sequence": index,
            "previous_event_sha256": previous,
            "call_kind": "model" if is_model else "retrieval",
            "operation": operation,
            "pre_call_remaining": dict(before),
            "reservation": {
                "model_calls": 1 if is_model else 0,
                "retrieval_calls": 0 if is_model else 1,
                "estimated_input_tokens": input_tokens,
                "max_output_tokens": (
                    min(requested_output_cap, before["output_tokens"]) if is_model else 0
                ),
            },
            "usage": usage,
            "result": dict(event_result),
        }
        event = {**unsigned, "event_sha256": content_hash(unsigned)}
        events.append(event)
        previous = event["event_sha256"]
        before["model_calls"] -= usage["model_calls"]
        before["retrieval_calls"] -= usage["retrieval_calls"]
        if before["input_tokens"] is not None:
            before["input_tokens"] -= input_tokens
        before["output_tokens"] -= output_tokens
    return {
        "schema": "agentinputlab.adaptive-rag.execution-evidence.v1",
        "evidence_kind": "controlled_component_closure",
        "fixture_runtime": False,
        "controlled_method_execution": True,
        "controlled_upstream_router_parity": True,
        "upstream_arm_parity_claimed": False,
        "official_parity_claimed": False,
        "source_commit": _request_value(request, "upstream_commit"),
        "source_tree": _SOURCE_TREE,
        "source_state": "clean_git_checkout",
        "source_closure_sha256": _SOURCE_CLOSURE_HASH,
        "compatibility_manifest_sha256": _MANIFEST_HASH,
        "router_revision": identity["router_revision"],
        "router_snapshot_sha256": identity["router_snapshot_sha256"],
        "router_weights_sha256": identity["router_weights_sha256"],
        "router_tokenizer_sha256": identity["router_tokenizer_sha256"],
        "base_model_revision": identity["base_model_revision"],
        "generator_artifact_sha256": identity["generator_artifact_sha256"],
        "generator_config_sha256": identity["generator_config_sha256"],
        "retriever_revision": identity["retriever_revision"],
        "retriever_implementation_sha256": identity["retriever_implementation_sha256"],
        "retriever_config_sha256": identity["retriever_config_sha256"],
        "corpus_revision": identity["corpus_revision"],
        "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
        "index_revision": identity["index_revision"],
        "index_manifest_sha256": identity["index_manifest_sha256"],
        "prompt_sha256": {
            "no_retrieval": identity["no_retrieval_prompt_sha256"],
            "single_retrieval": identity["single_retrieval_prompt_sha256"],
            "ircot": identity["ircot_prompt_sha256"],
        },
        "selected_strategy": strategy,
        "event_log_sha256": content_hash(events),
        "final_event_sha256": events[-1]["event_sha256"],
        "events": events,
    }


def response_payload(request: Any, *, route: str = "B") -> dict[str, Any]:
    request_payload = _request_payload(request)
    strategy = {
        "A": "no_retrieval",
        "B": "single_retrieval",
        "C": "ircot",
    }[route]
    route_logits = {
        "A": {"A": 1.0, "B": 1.0, "C": 1.0},
        "B": {"A": -1.0, "B": 3.0, "C": 0.5},
        "C": {"A": -2.0, "B": 0.1, "C": 4.0},
    }[route]
    documents: list[dict[str, Any]] = []
    initial_retrieval: dict[str, Any] | None = None
    ircot_steps: list[dict[str, Any]] = []
    active_document_ids: list[str] = []
    if route in {"B", "C"}:
        documents.append(
            {
                "document_id": "d1",
                "text": "The first retrieved fact.",
                "title": "First",
                "corpus_revision": "wiki-2018-v1",
                "score": 4.0,
                "metadata": {"source_uri": "fixture://d1"},
            }
        )
        initial_retrieval = {
            "query": request_payload["question"],
            "document_ids": ["d1"],
        }
        active_document_ids = ["d1"]
    if route == "C":
        documents.append(
            {
                "document_id": "d2",
                "text": "The second retrieved fact.",
                "title": "Second",
                "corpus_revision": "wiki-2018-v1",
                "score": 3.0,
                "metadata": {},
            }
        )
        ircot_steps = [
            {
                "index": 0,
                "cot_step": "We need the second fact.",
                "latest_cot_sentence": "We need the second fact.",
                "retrieval_query": "We need the second fact.",
                "retrieved_document_ids": ["d2"],
                "new_document_ids": ["d2"],
                "active_document_ids": ["d1", "d2"],
                "reset_applied": False,
            }
        ]
        active_document_ids = ["d1", "d2"]
    answer = f"{strategy} answer"
    result = {
        "question": request_payload["question"],
        "artifacts": dict(request_payload["artifacts"]),
        "artifact_identity_sha256": request_payload["artifact_identity_sha256"],
        "route_logits": route_logits,
        "route_label": route,
        "selected_strategy": strategy,
        "answer": answer,
        "documents": documents,
        "initial_retrieval": initial_retrieval,
        "ircot_steps": ircot_steps,
        "active_document_ids": active_document_ids,
    }
    result["execution_evidence"] = _execution_evidence(
        request,
        strategy=strategy,
        route_logits=route_logits,
        answer=answer,
        initial_retrieval=initial_retrieval,
        ircot_steps=ircot_steps,
    )
    return result


class FakeAdaptiveRAGWorker:
    worker_id = "adaptive-rag-method-fixture"

    def __init__(
        self,
        builder: Callable[[Any], Mapping[str, Any]],
        *,
        usage: Mapping[str, int | float],
    ) -> None:
        self.builder = builder
        self.usage = dict(usage)
        self.calls = 0
        self.last_request: Any = None

    def _respond(self, request: Any) -> Mapping[str, Any]:
        self.calls += 1
        self.last_request = request
        return {
            "status": "ok",
            "request_id": _request_value(request, "request_id"),
            "upstream_commit": _request_value(request, "upstream_commit"),
            "protocol_version": _request_value(request, "protocol_version"),
            "payload": dict(self.builder(request)),
            "usage": dict(self.usage),
        }

    def call(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._respond(request)

    def call_controlled(self, request: Any) -> Mapping[str, Any]:
        return self._respond(request)

    def fingerprint(self) -> Mapping[str, Any]:
        return {
            "worker_id": self.worker_id,
            "image_digest": "sha256:adaptive-rag-fixture",
            "source_revision": "fixture-v1",
        }


def usage(*, model_calls: int = 2, retrieval_calls: int = 1) -> Mapping[str, int]:
    return {
        "model_calls": model_calls,
        "retrieval_calls": retrieval_calls,
        "input_tokens": 32,
        "output_tokens": 8,
    }


class AdaptiveRAGMethodTests(unittest.TestCase):
    def test_single_retrieval_executes_through_observed_composite_runner(self) -> None:
        worker = FakeAdaptiveRAGWorker(
            lambda request: response_payload(request, route="B"),
            usage=usage(),
        )
        method = AdaptiveRAGControlledWorker(artifacts())
        state = FrozenAgentState(query="Who wrote the work?", task_id="adaptive-b")

        record = CompositeInterventionRunner().run(
            method,
            state,
            InterventionContext(external_runtime=worker, seed=7),
        )

        result = record.result
        self.assertEqual(result.final_answer, "single_retrieval answer")
        self.assertEqual(result.metadata["route_label"], "B")
        self.assertEqual(result.metadata["strategy"], "single_retrieval")
        self.assertTrue(result.metadata["controlled_method_execution"])
        self.assertTrue(result.metadata["controlled_upstream_router_parity"])
        self.assertFalse(result.metadata["upstream_arm_execution_parity"])
        self.assertTrue(result.metadata["controlled_online_selected_arm"])
        self.assertFalse(result.metadata["official_execution_parity"])
        self.assertEqual(result.evidence[0].block_id, "d1")
        self.assertEqual(
            result.evidence[0].metadata["adaptive_rag_strategy"],
            "single_retrieval",
        )
        self.assertEqual(record.usage.model_calls, 2)
        self.assertEqual(record.usage.retrieval_calls, 1)
        sent = _request_payload(worker.last_request)
        self.assertEqual(
            sent["execution_contract"]["route_rule"],
            "first_decoder_step_logit_argmax_A_first",
        )
        self.assertEqual(sent["execution_contract"]["evaluation_label_access"], "none")
        self.assertTrue(sent["execution_contract"]["selected_arm_only"])
        self.assertFalse(sent["execution_contract"]["fallback_allowed"])

    def test_no_retrieval_uses_upstream_tie_break_order_and_has_no_evidence(self) -> None:
        result = AdaptiveRAGControlledWorker(artifacts()).run(
            FrozenAgentState(query="A simple question"),
            InterventionContext(
                external_runtime=FakeAdaptiveRAGWorker(
                    lambda request: response_payload(request, route="A"),
                    usage=usage(retrieval_calls=0),
                )
            ),
        )

        self.assertEqual(result.metadata["route_label"], "A")
        self.assertEqual(result.evidence, ())
        self.assertEqual(result.trace.counters["retrieval_calls"], 0)

    def test_ircot_route_validates_iterative_retrieval_trajectory(self) -> None:
        result = AdaptiveRAGControlledWorker(artifacts()).run(
            FrozenAgentState(query="A multi-hop question"),
            InterventionContext(
                external_runtime=FakeAdaptiveRAGWorker(
                    lambda request: response_payload(request, route="C"),
                    usage=usage(model_calls=3, retrieval_calls=2),
                )
            ),
        )

        self.assertEqual(result.metadata["strategy"], "ircot")
        self.assertEqual(
            tuple(block.block_id for block in result.evidence),
            ("d1", "d2"),
        )
        self.assertEqual(result.trace.counters["ircot_steps"], 1)
        self.assertEqual(
            tuple(event.name for event in result.trace.events),
            (
                "controlled_three_arm_bundle",
                "complexity_route",
                "initial_retrieval",
                "ircot_step",
                "selected_strategy_output",
            ),
        )

    def test_rejects_event_output_cap_above_sent_per_call_control(self) -> None:
        def builder(request: Any) -> Mapping[str, Any]:
            payload = response_payload(request, route="A")
            evidence = payload["execution_evidence"]
            final = evidence["events"][-1]
            final["reservation"]["max_output_tokens"] = 2
            final["usage"]["output_tokens"] = 2
            unsigned = {key: value for key, value in final.items() if key != "event_sha256"}
            final["event_sha256"] = content_hash(unsigned)
            evidence["final_event_sha256"] = final["event_sha256"]
            evidence["event_log_sha256"] = content_hash(evidence["events"])
            return payload

        worker_usage = dict(usage(retrieval_calls=0))
        worker_usage["output_tokens"] = 2
        with self.assertRaisesRegex(AlgorithmOutputError, "output cap disagrees"):
            AdaptiveRAGControlledWorker(
                artifacts(),
                controls=AdaptiveRAGControls(max_output_tokens=1),
            ).run(
                FrozenAgentState(query="question"),
                InterventionContext(
                    external_runtime=FakeAdaptiveRAGWorker(
                        builder,
                        usage=worker_usage,
                    )
                ),
            )

    def test_rejects_event_reservation_schema_drift(self) -> None:
        def builder(request: Any) -> Mapping[str, Any]:
            payload = response_payload(request, route="A")
            evidence = payload["execution_evidence"]
            final = evidence["events"][-1]
            final["reservation"]["worker_note"] = "untrusted"
            unsigned = {key: value for key, value in final.items() if key != "event_sha256"}
            final["event_sha256"] = content_hash(unsigned)
            evidence["final_event_sha256"] = final["event_sha256"]
            evidence["event_log_sha256"] = content_hash(evidence["events"])
            return payload

        with self.assertRaisesRegex(AlgorithmOutputError, "reservation fields drifted"):
            AdaptiveRAGControlledWorker(artifacts()).run(
                FrozenAgentState(query="question"),
                InterventionContext(
                    external_runtime=FakeAdaptiveRAGWorker(
                        builder,
                        usage=usage(retrieval_calls=0),
                    )
                ),
            )

    def test_rejects_route_that_is_not_first_step_logit_argmax(self) -> None:
        def builder(request: Any) -> Mapping[str, Any]:
            payload = response_payload(request, route="B")
            payload["route_label"] = "A"
            payload["selected_strategy"] = "no_retrieval"
            return payload

        with self.assertRaisesRegex(AlgorithmOutputError, "logit argmax"):
            AdaptiveRAGControlledWorker(artifacts()).run(
                FrozenAgentState(query="question"),
                InterventionContext(
                    external_runtime=FakeAdaptiveRAGWorker(
                        builder,
                        usage=usage(retrieval_calls=0),
                    )
                ),
            )

    def test_rejects_unverified_official_parity_claim(self) -> None:
        def builder(request: Any) -> Mapping[str, Any]:
            payload = response_payload(request, route="B")
            payload["execution_evidence"]["official_parity_claimed"] = True
            return payload

        with self.assertRaisesRegex(
            AlgorithmOutputError,
            "official_parity_claimed",
        ):
            AdaptiveRAGControlledWorker(artifacts()).run(
                FrozenAgentState(query="question"),
                InterventionContext(
                    external_runtime=FakeAdaptiveRAGWorker(
                        builder,
                        usage=usage(),
                    )
                ),
            )

    def test_rejects_single_retrieval_query_drift(self) -> None:
        def builder(request: Any) -> Mapping[str, Any]:
            payload = response_payload(request, route="B")
            payload["initial_retrieval"] = {
                "query": "different question",
                "document_ids": ["d1"],
            }
            return payload

        with self.assertRaisesRegex(AlgorithmOutputError, "equal the question"):
            AdaptiveRAGControlledWorker(artifacts()).run(
                FrozenAgentState(query="question"),
                InterventionContext(
                    external_runtime=FakeAdaptiveRAGWorker(
                        builder,
                        usage=usage(),
                    )
                ),
            )

    def test_rejects_worker_artifact_bundle_drift(self) -> None:
        def builder(request: Any) -> Mapping[str, Any]:
            payload = response_payload(request, route="A")
            payload["artifacts"]["base_model_revision"] = "another-model-v1"
            return payload

        with self.assertRaisesRegex(AlgorithmOutputError, "artifact identity"):
            AdaptiveRAGControlledWorker(artifacts()).run(
                FrozenAgentState(query="question"),
                InterventionContext(
                    external_runtime=FakeAdaptiveRAGWorker(
                        builder,
                        usage=usage(retrieval_calls=0),
                    )
                ),
            )

    def test_host_rejects_non_allowlisted_document_metadata(self) -> None:
        for alias in (
            "answer_key",
            "reference_answer",
            "expected_answer",
            "correct_option",
        ):
            with self.subTest(alias=alias):

                def builder(request: Any, *, field: str = alias) -> Mapping[str, Any]:
                    payload = response_payload(request, route="B")
                    payload["documents"][0]["metadata"] = {field: "TOP-SECRET"}
                    return payload

                with self.assertRaisesRegex(
                    AlgorithmOutputError,
                    "non-allowlisted",
                ):
                    AdaptiveRAGControlledWorker(artifacts()).run(
                        FrozenAgentState(query="question"),
                        InterventionContext(
                            external_runtime=FakeAdaptiveRAGWorker(
                                builder,
                                usage=usage(),
                            )
                        ),
                    )

    def test_budget_fails_before_worker_when_two_model_calls_are_impossible(self) -> None:
        worker = FakeAdaptiveRAGWorker(
            lambda request: response_payload(request, route="A"),
            usage=usage(retrieval_calls=0),
        )
        state = FrozenAgentState(
            query="question",
            budget=ExecutionBudget(max_model_calls=1),
        )

        with self.assertRaisesRegex(BudgetExceededError, "classifier and answer"):
            AdaptiveRAGControlledWorker(artifacts()).run(
                state,
                InterventionContext(external_runtime=worker),
            )

        self.assertEqual(worker.calls, 0)

    def test_rejects_selected_arm_that_exceeds_sent_retrieval_budget(self) -> None:
        state = FrozenAgentState(
            query="question",
            budget=ExecutionBudget(max_retrieval_calls=0),
        )

        with self.assertRaisesRegex(AlgorithmOutputError, "call budget|retrieval-call budget"):
            AdaptiveRAGControlledWorker(artifacts()).run(
                state,
                InterventionContext(
                    external_runtime=FakeAdaptiveRAGWorker(
                        lambda request: response_payload(request, route="B"),
                        usage=usage(),
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
