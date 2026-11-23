from __future__ import annotations

import unittest

from agentinputlab.algorithms.reformulate.rrr import (
    RRRCommonSupport,
    RRRCommonSupportConfig,
    _common_support_rewrite_text,
)
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab.backends.llm import GenerationResponse, ModelCapability, Usage
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import AlgorithmOutputError, FrozenAgentState, InterventionContext
from agentinputlab.contracts.serialization import content_hash


class SequenceBackend:
    model_id = "rrr-test-model"
    capabilities = frozenset({ModelCapability.COMPLETION, ModelCapability.SEED})

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(
            text=self.outputs.pop(0),
            model=self.model_id,
            usage=Usage(input_tokens=10, output_tokens=3),
        )

    def fingerprint(self):
        return {"backend": "rrr-test", "model_id": self.model_id}


class StaticRetriever:
    retriever_id = "rrr-test-retriever"

    def __init__(self) -> None:
        self.queries = []

    def retrieve(self, query: str, *, top_k: int = 5):
        self.queries.append((query, top_k))
        return (
            RetrievalResult(
                document=Document(
                    doc_id="doc-1",
                    title="Dune",
                    text="Frank Herbert wrote Dune.",
                    corpus_revision="fixture-v1",
                ),
                score=1.0,
                rank=1,
            ),
        )

    def fingerprint(self):
        return {"retriever": "rrr-test", "revision": "fixture-v1"}


def config() -> RRRCommonSupportConfig:
    rewrite = "Provide a better search query. Question: {query}\nQuery:"
    reader = "Passage: {background}\nQuestion: {query}\nThe answer is"
    return RRRCommonSupportConfig(
        rewrite_prompt=rewrite,
        reader_prompt=reader,
        prompt_source_sha256="a" * 64,
        rewrite_prompt_sha256=content_hash(rewrite),
        reader_prompt_sha256=content_hash(reader),
        top_k=5,
    )


class RRRCommonSupportTests(unittest.TestCase):
    def test_recommended_profile_uses_public_factory_and_metered_core_without_checkpoint(
        self,
    ) -> None:
        backend = SequenceBackend(["Query: Dune author**", "Frank Herbert**"])
        method = create_intervention("rewrite.rrr.common_support_prompt", config=config())
        result = run_intervention(
            method,
            FrozenAgentState(query="Who wrote Dune?"),
            InterventionContext(backend=backend, retriever=StaticRetriever(), seed=17),
        )
        self.assertEqual(method.release_id, "rewrite.rrr.common_support_prompt")
        self.assertEqual(result["state"]["query"], "Dune author")
        self.assertFalse(result["scored"])
        self.assertIsNone(result["benchmark_score"])
        record = result["record"]
        self.assertEqual(record["usage"]["model_calls"], 2)
        self.assertEqual(record["usage"]["retrieval_calls"], 1)
        self.assertEqual(record["patch"]["profile_kind"], "controlled_adapter")
        self.assertFalse(record["patch"]["metadata"]["paper_checkpoint_used"])
        self.assertFalse(record["patch"]["metadata"]["ppo_training_used"])
        self.assertEqual([r.max_tokens for r in backend.requests], [100, 300])

    def test_source_locked_stage_order_materializes_query_and_real_evidence(self) -> None:
        backend = SequenceBackend(["Query: Dune author**", "Frank Herbert**"])
        retriever = StaticRetriever()
        state = FrozenAgentState(query="Who wrote Dune?", task_id="rrr-common")

        patch = RRRCommonSupport(config()).propose(
            state,
            InterventionContext(backend=backend, retriever=retriever, seed=17),
        )
        updated = patch.apply(state)

        self.assertEqual(updated.query, "Dune author")
        self.assertEqual(tuple(block.block_id for block in updated.context), ("doc-1",))
        self.assertEqual(retriever.queries, [("Dune author", 5)])
        self.assertEqual([request.seed for request in backend.requests], [17, 18])
        self.assertEqual(patch.trace.counters["model_calls"], 2)
        self.assertEqual(patch.trace.counters["retrieval_calls"], 1)
        self.assertFalse(patch.metadata["paper_checkpoint_used"])
        self.assertFalse(patch.metadata["ppo_training_used"])

    def test_rejects_multiple_queries_in_single_query_profile(self) -> None:
        backend = SequenceBackend(["first; second**", "unused"])
        with self.assertRaisesRegex(AlgorithmOutputError, "exactly one query"):
            RRRCommonSupport(config()).propose(
                FrozenAgentState(query="question"),
                InterventionContext(backend=backend, retriever=StaticRetriever(), seed=0),
            )

    def test_prompt_hashes_are_enforced(self) -> None:
        rewrite = "Question: {query}"
        reader = "{background}\n{query}"
        with self.assertRaisesRegex(Exception, "rewrite prompt hash"):
            RRRCommonSupportConfig(
                rewrite_prompt=rewrite,
                reader_prompt=reader,
                prompt_source_sha256="a" * 64,
                rewrite_prompt_sha256="b" * 64,
                reader_prompt_sha256=content_hash(reader),
            )

    def test_single_bold_wrapper_preserves_query_and_budget(self) -> None:
        for text in ("**Dune author**", "  **Dune author**  ", "Query: **Dune author**"):
            with self.subTest(text=text):
                backend = SequenceBackend([text, "Frank Herbert"])
                retriever = StaticRetriever()
                patch = RRRCommonSupport(config()).propose(
                    FrozenAgentState(query="Who wrote Dune?"),
                    InterventionContext(backend=backend, retriever=retriever, seed=17),
                )
                self.assertEqual(retriever.queries, [("Dune author", 5)])
                self.assertEqual([r.max_tokens for r in backend.requests], [100, 300])
                self.assertEqual([r.seed for r in backend.requests], [17, 18])
                event = next(e for e in patch.trace.events if e.name == "official_single_query_parser")
                self.assertEqual(event.payload["raw_rewrite_sha256"], content_hash(text))
                self.assertEqual(event.payload["format_normalization"], "single_bold_wrapper.v1")

    def test_bold_compatibility_does_not_choose_or_repair_ambiguous_output(self) -> None:
        for text in (
            "**first**\n**second**", "**first; second**", "**first** explanation",
            "**unfinished", "****", "**first\nsecond**", "", ";;**",
        ):
            with self.subTest(text=text):
                backend = SequenceBackend([text, "unused"])
                retriever = StaticRetriever()
                with self.assertRaisesRegex(AlgorithmOutputError, "exactly one query"):
                    RRRCommonSupport(config()).propose(
                        FrozenAgentState(query="question"),
                        InterventionContext(backend=backend, retriever=retriever, seed=17),
                    )
                self.assertEqual(len(backend.requests), 1)
                self.assertEqual(retriever.queries, [])

    def test_original_format_does_not_gain_normalization_metadata(self) -> None:
        patch = RRRCommonSupport(config()).propose(
            FrozenAgentState(query="Who wrote Dune?"),
            InterventionContext(backend=SequenceBackend(["Query: Dune author**", "Frank Herbert"]),
                                retriever=StaticRetriever(), seed=17),
        )
        event = next(e for e in patch.trace.events if e.name == "official_single_query_parser")
        self.assertNotIn("format_normalization", event.payload)

    def test_retained_final_bold_query_survives_logical_stop_adapter(self) -> None:
        from agentinputlab.backends import OpenAICompatibleBackend
        from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
        requests = []

        def transport(url, headers, body, timeout):
            requests.append(body)
            text = "**Dune author**" if len(requests) == 1 else "Frank Herbert"
            return {"model": "gpt-oss-20b", "choices": [{"message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}

        backend = GPTOSSFinalChannelStopBackend(OpenAICompatibleBackend(
            model_id="gpt-oss-20b", base_url="http://localhost:9/v1",
            deployment_identity={"server": "vllm", "revision": "fixture"}, transport=transport,
        ))
        result = run_intervention(
            create_intervention("rewrite.rrr.common_support_prompt", config=config()),
            FrozenAgentState(query="Who wrote Dune?"),
            InterventionContext(backend=backend, retriever=StaticRetriever(), seed=17),
        )
        self.assertEqual(result["state"]["query"], "Dune author")
        self.assertEqual(result["record"]["usage"]["output_tokens"], 6)
        self.assertEqual(len(requests), 2)
        self.assertEqual([r["max_tokens"] for r in requests], [100, 300])
        event = next(e for e in result["record"]["patch"]["trace"]["events"] if e["name"] == "official_single_query_parser")
        self.assertEqual(event["payload"]["retained_final_content_sha256"], content_hash("**Dune author**"))

    def test_logical_stop_recovery_never_promotes_reasoning_or_partial_content(self) -> None:
        marker = {"policy_id": "gpt-oss.final-channel-stop.v1", "requested_stop": ["**"], "matched_stop": "**"}
        for text in (None, "", "**first**\n**second**", "**partial", "ordinary text"):
            with self.subTest(text=text):
                response = GenerationResponse(text="", raw={"agentinputlab_final_channel_stop": marker,
                    "choices": [{"message": {"content": text, "reasoning": "**Dune author**"}}]})
                self.assertEqual(_common_support_rewrite_text(response), "")
        response = GenerationResponse(text="", raw={"choices": [{"message": {"content": "**Dune author**"}}]})
        self.assertEqual(_common_support_rewrite_text(response), "")
