from __future__ import annotations

import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agentinputlab.algorithms.reformulate.query2doc import (
    OfficialQuery2DocRow,
    PinnedQueryGymRuntime,
    Query2DocCache,
    Query2DocFixtureCached,
    Query2DocOfficialCached,
    QueryGymLiveBridge,
    QueryGymMode,
    QueryGymRawResult,
    compose_sparse_expansion,
    official_cached,
    querygym_runtime,
    select_demonstration_ids,
)
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, Usage
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    ExecutionBudget,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    PatchOp,
    ValidationError,
)
from agentinputlab.runner import InterventionRunner


class RecordingQueryGymRuntime:
    def __init__(self, result: QueryGymRawResult) -> None:
        self.result = result
        self.requests = []

    def reformulate(self, request):
        self.requests.append(request)
        return self.result

    def fingerprint(self):
        return {
            "implementation": "tests.recording_querygym.v1",
            "revision": "6d2d19213c7212c7940143e7b9b5f1d2af7ff47b",
        }


class Query2DocOfficialCachedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = Query2DocCache(
            (
                OfficialQuery2DocRow(
                    query_id="q1",
                    query="alpha beta",
                    pseudo_doc="alpha is connected to beta",
                    split="dev",
                ),
            ),
            artifact_hash="fixture-sha",
        )

    def test_official_sparse_expansion_is_five_queries_plus_pseudo_doc(self) -> None:
        self.assertEqual(
            compose_sparse_expansion("q", "doc"),
            "q q q q q doc",
        )

    def test_cached_profile_materializes_exact_query_and_surrogate_trace(self) -> None:
        state = FrozenAgentState(query="alpha beta", task_id="q1")
        patch = Query2DocFixtureCached(self.cache, split="dev").propose(
            state, InterventionContext()
        )
        self.assertEqual(patch.operations[0].op, PatchOp.REPLACE_QUERY)
        self.assertEqual(
            patch.apply(state).query,
            "alpha beta alpha beta alpha beta alpha beta alpha beta alpha is connected to beta",
        )
        event = patch.trace.events[1]
        self.assertEqual(event.payload["evidence_role"], "synthetic_query_surrogate")
        self.assertFalse(patch.metadata["pseudo_document_is_evidence"])
        self.assertEqual(patch.profile_kind, "fixture_unverified")

    def test_cached_profile_fails_closed_on_query_mismatch(self) -> None:
        state = FrozenAgentState(query="different", task_id="q1")
        with self.assertRaises(ArtifactError):
            Query2DocFixtureCached(self.cache, split="dev").propose(state, InterventionContext())

    def test_cache_fails_closed_on_missing_id(self) -> None:
        state = FrozenAgentState(query="alpha beta", task_id="missing")
        with self.assertRaises(ArtifactError):
            Query2DocFixtureCached(self.cache, split="dev").propose(state, InterventionContext())

    def test_official_profile_rejects_unverified_fixture_rows(self) -> None:
        with self.assertRaises(ArtifactError):
            Query2DocOfficialCached(self.cache, split="dev")

    def test_caller_checksum_does_not_claim_official_provenance(self) -> None:
        payload = (
            '{"query_id":"q1","query":"alpha beta",'
            '"pseudo_doc":"alpha is connected to beta","split":"dev"}\n'
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "query2doc.jsonl"
            path.write_text(payload, encoding="utf-8")
            expected = sha256(payload.encode("utf-8")).hexdigest()
            cache = Query2DocCache.from_jsonl(path, expected_sha256=expected)
        self.assertEqual(cache.verification, "caller_sha256_verified")
        with self.assertRaisesRegex(ArtifactError, "audited"):
            Query2DocOfficialCached(cache, split="dev")

    def test_official_loader_requires_terms_and_audited_split_before_reading(self) -> None:
        with self.assertRaisesRegex(ArtifactError, "terms"):
            Query2DocCache.from_official_jsonl("unused.jsonl", split="trec_dl2019")
        with self.assertRaisesRegex(ArtifactError, "only audited"):
            Query2DocCache.from_official_jsonl(
                "unused.jsonl", split="train", accept_license_review=True,
            )

    def test_native_split_binding_does_not_rewrite_source_text_or_claim_target_generation(self) -> None:
        # Synthetic parser test only: the production pins are not these bytes.
        raw = json.dumps({"query_id": "q1", "query": "alpha beta",
                          "pseudo_doc": "alpha is connected to beta"}).encode() + b"\n"
        spec = {"trec_dl2019": (sha256(raw).hexdigest(), len(raw), 1)}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "native.jsonl"
            path.write_bytes(raw)
            with patch.object(official_cached, "OFFICIAL_TREC_ARTIFACTS", spec):
                cache = Query2DocCache.from_official_jsonl(
                    path, split="trec_dl2019", accept_license_review=True,
                )
        self.assertEqual(cache.get("trec_dl2019", "q1").query, "alpha beta")
        method = Query2DocOfficialCached(cache, split="trec_dl2019")
        result = method.propose(FrozenAgentState(query="alpha beta", task_id="q1"), InterventionContext())
        self.assertEqual(result.trace.counters["model_calls"], 0)
        self.assertEqual(result.metadata["source_generation_model"], "text-davinci-003")
        self.assertEqual(result.metadata["historical_generation_cost"], "unknown_not_free")
        with self.assertRaisesRegex(ArtifactError, "exact source query"):
            Query2DocOfficialCached(cache, split="trec_dl2019", strict_query_match=False)

    def test_official_loader_rejects_changed_file_without_reading_unbounded_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.jsonl"
            path.write_bytes(b" " * 30000)
            with self.assertRaisesRegex(ArtifactError, "size/checksum"):
                Query2DocCache.from_official_jsonl(
                    path, split="trec_dl2019", accept_license_review=True,
                )

    def test_audited_trec_source_pins_are_split_specific_and_immutable(self) -> None:
        self.assertEqual(set(official_cached.OFFICIAL_TREC_ARTIFACTS), {"trec_dl2019", "trec_dl2020"})
        self.assertEqual(official_cached.OFFICIAL_TREC_ARTIFACTS["trec_dl2019"][1:], (21995, 43))
        self.assertEqual(official_cached.OFFICIAL_TREC_ARTIFACTS["trec_dl2020"][1:], (26973, 54))
        with self.assertRaises(TypeError):
            official_cached.OFFICIAL_TREC_ARTIFACTS["train"] = ("", 0, 0)

    def test_native_schema_rejects_gold_extra_fields_bad_values_and_duplicate_keys(self) -> None:
        ordinary = {"query_id": "q1", "query": "q", "pseudo_doc": "doc"}
        cases = [
            ([{**ordinary, "answer": "protected"}], "row fields"),
            ([{**ordinary, "query_id": 1}], "non-empty strings"),
            ([{**ordinary, "pseudo_doc": " "}], "non-empty strings"),
            ([ordinary, ordinary], "duplicate"),
        ]
        for rows, message in cases:
            with self.subTest(message=message), TemporaryDirectory() as directory:
                raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
                path = Path(directory) / "synthetic-native.jsonl"
                path.write_bytes(raw)
                fixture = {"trec_dl2019": (sha256(raw).hexdigest(), len(raw), len(rows))}
                with (
                    patch.object(official_cached, "OFFICIAL_TREC_ARTIFACTS", fixture),
                    self.assertRaisesRegex(ArtifactError, message),
                ):
                    Query2DocCache.from_official_jsonl(
                        path, split="trec_dl2019", accept_license_review=True,
                    )


class QueryGymBridgeTests(unittest.TestCase):
    def test_pinned_runtime_adapts_upstream_chat_without_changing_algorithm(self) -> None:
        class MethodConfig:
            def __init__(self, **kwargs):
                self.params = kwargs["params"]
                self.llm = kwargs["llm"]

        class PromptBank:
            def __init__(self, path):
                self.path = path

            def render(self, prompt_id, *, query):
                self.prompt_id = prompt_id
                return [{"role": "user", "content": f"Query: {query}\nPassage:"}]

        class QueryItem:
            def __init__(self, qid, text):
                self.qid = qid
                self.text = text

        class Query2Doc:
            def __init__(self, config, client, prompts):
                self.config = config
                self.client = client
                self.prompts = prompts

            def reformulate(self, item):
                pseudo_doc = self.client.chat(
                    self.prompts.render("query2doc.zeroshot.v1", query=item.text),
                    temperature=self.config.llm["temperature"],
                    max_tokens=self.config.llm["max_tokens"],
                )
                return type(
                    "Result",
                    (),
                    {
                        "reformulated": " ".join([item.text] * 5 + [pseudo_doc]),
                        "metadata": {"pseudo_doc": pseudo_doc, "mode": "zs"},
                    },
                )()

        symbols = querygym_runtime._UpstreamSymbols(
            method_config=MethodConfig,
            prompt_bank=PromptBank,
            query_item=QueryItem,
            query2doc=Query2Doc,
        )
        backend = CallableBackend(
            lambda request: GenerationResponse(
                text="generated passage",
                model="fixture-model",
                usage=Usage(input_tokens=8, output_tokens=2),
            ),
            model_id="fixture-model",
            revision="fixture-revision",
            callable_id="querygym-runtime-test",
        )
        identity = {
            "revision": "6d2d19213c7212c7940143e7b9b5f1d2af7ff47b",
            "source_closure_sha256": "source-closure",
        }
        with (
            patch.object(querygym_runtime, "_checkout_identity", return_value=identity),
            patch.object(querygym_runtime, "_load_upstream_symbols", return_value=symbols),
        ):
            runtime = PinnedQueryGymRuntime("/tmp/pinned-querygym", backend)
        raw = runtime.reformulate(
            querygym_runtime.QueryGymRequest(
                query_id="q1",
                query="what is the answer",
                mode=QueryGymMode.ZERO_SHOT,
                temperature=0.25,
                max_tokens=64,
                seed=19,
                backend_fingerprint=backend.fingerprint(),
            )
        )

        self.assertEqual(
            raw.reformulated,
            "what is the answer what is the answer what is the answer "
            "what is the answer what is the answer generated passage",
        )
        self.assertEqual(raw.usage["input_tokens"], 8)
        self.assertEqual(raw.usage["output_tokens"], 2)
        self.assertEqual(backend.calls[0].seed, 19)
        self.assertEqual(backend.calls[0].temperature, 0.25)
        self.assertEqual(backend.calls[0].max_tokens, 64)

    def test_seeded_demonstration_selection_is_stable(self) -> None:
        candidates = ("a", "b", "c", "d", "e")
        self.assertEqual(
            select_demonstration_ids(candidates, count=3, seed=17),
            select_demonstration_ids(candidates, count=3, seed=17),
        )

    def test_live_bridge_preserves_community_identity(self) -> None:
        runtime = RecordingQueryGymRuntime(
            QueryGymRawResult(
                reformulated="q q q q q generated",
                generated_text="generated",
                prompt_hash="prompt-sha",
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 0,
                    "input_tokens": 7,
                    "output_tokens": 4,
                },
            )
        )
        bridge = QueryGymLiveBridge(
            runtime,
            mode=QueryGymMode.FEW_SHOT,
            demonstration_ids=("d1", "d2", "d3", "d4"),
        )
        state = FrozenAgentState(query="q", task_id="case")
        patch = bridge.propose(state, InterventionContext(seed=11))
        self.assertEqual(patch.apply(state).query, "q q q q q generated")
        self.assertEqual(patch.metadata["implementation_origin"], "community")
        self.assertEqual(runtime.requests[0].demonstration_ids, ("d1", "d2", "d3", "d4"))
        self.assertEqual(runtime.requests[0].seed, 11)

    def test_live_bridge_exposes_explicit_native_or_controlled_profile(self) -> None:
        runtime = RecordingQueryGymRuntime(
            QueryGymRawResult(
                reformulated="rewritten",
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 0,
                    "input_tokens": 3,
                    "output_tokens": 2,
                },
            )
        )
        state = FrozenAgentState(query="q")
        controlled = QueryGymLiveBridge(runtime).propose(state, InterventionContext())
        native = QueryGymLiveBridge(runtime, profile_kind="official_native").propose(
            state,
            InterventionContext(),
        )
        self.assertEqual(controlled.profile_kind, "controlled_adapter")
        self.assertEqual(native.profile_kind, "official_native")

    def test_native_fallback_is_no_op_and_counted(self) -> None:
        runtime = RecordingQueryGymRuntime(
            QueryGymRawResult(
                reformulated="",
                fallback=True,
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 0,
                    "input_tokens": 3,
                    "output_tokens": 0,
                },
            )
        )
        state = FrozenAgentState(query="q")
        patch = QueryGymLiveBridge(runtime).propose(state, InterventionContext())
        self.assertEqual(patch.operations[0].op, PatchOp.NO_OP)
        self.assertTrue(patch.metadata["fallback"])
        self.assertEqual(patch.trace.counters["fallbacks"], 1)

    def test_missing_querygym_runtime_is_typed_error(self) -> None:
        with self.assertRaises(ExternalDependencyError):
            QueryGymLiveBridge(None).propose(FrozenAgentState(query="q"), InterventionContext())

    def test_querygym_runtime_revision_is_pinned(self) -> None:
        class WrongRevision(RecordingQueryGymRuntime):
            def fingerprint(self):
                return {"revision": "moving-main"}

        with self.assertRaisesRegex(ValueError, "pinned commit"):
            QueryGymLiveBridge(WrongRevision(QueryGymRawResult(reformulated="q")))

    def test_controlled_runner_rejects_hidden_querygym_runtime(self) -> None:
        runtime = RecordingQueryGymRuntime(
            QueryGymRawResult(
                reformulated="rewritten",
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 0,
                    "input_tokens": 3,
                    "output_tokens": 2,
                },
            )
        )
        with self.assertRaisesRegex(ValidationError, "runtime dependencies"):
            InterventionRunner().run(
                QueryGymLiveBridge(runtime),
                FrozenAgentState(query="q"),
                InterventionContext(),
            )
        self.assertEqual(runtime.requests, [])

    def test_context_runtime_is_monitored_and_unknown_cost_is_visible(self) -> None:
        runtime = RecordingQueryGymRuntime(
            QueryGymRawResult(
                reformulated="rewritten",
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 0,
                    "input_tokens": 3,
                    "output_tokens": 2,
                },
            )
        )
        record = InterventionRunner().run(
            QueryGymLiveBridge(None),
            FrozenAgentState(query="q"),
            InterventionContext(external_runtime=runtime),
        )
        self.assertEqual(record.usage.model_calls, 1)
        self.assertEqual(record.usage.input_tokens, 3)
        self.assertEqual(record.usage.output_tokens, 2)
        self.assertEqual(record.usage.unpriced_model_calls, 1)

    def test_context_runtime_unknown_cost_fails_closed_with_cost_budget(self) -> None:
        runtime = RecordingQueryGymRuntime(
            QueryGymRawResult(
                reformulated="rewritten",
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 0,
                    "input_tokens": 3,
                    "output_tokens": 2,
                },
            )
        )
        runner = InterventionRunner()
        with self.assertRaisesRegex(BudgetExceededError, "unknown_cost"):
            runner.run(
                QueryGymLiveBridge(None),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_cost_usd=1.0),
                ),
                InterventionContext(external_runtime=runtime),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 1)
        self.assertEqual(snapshot.usage.unpriced_model_calls, 1)

    def test_model_call_budget_rejects_before_querygym_side_effect(self) -> None:
        runtime = RecordingQueryGymRuntime(
            QueryGymRawResult(
                reformulated="must-not-run",
                usage={
                    "model_calls": 1,
                    "retrieval_calls": 0,
                    "input_tokens": 3,
                    "output_tokens": 2,
                },
            )
        )
        runner = InterventionRunner()
        with self.assertRaisesRegex(BudgetExceededError, "model_calls"):
            runner.run(
                QueryGymLiveBridge(None),
                FrozenAgentState(
                    query="q",
                    budget=ExecutionBudget(max_model_calls=0),
                ),
                InterventionContext(external_runtime=runtime),
            )
        self.assertEqual(runtime.requests, [])
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 0)

    def test_runtime_exception_keeps_reserved_attempt_and_unknown_cost(self) -> None:
        class RaisingRuntime(RecordingQueryGymRuntime):
            def reformulate(self, request):
                self.requests.append(request)
                raise RuntimeError("provider timeout after dispatch")

        runtime = RaisingRuntime(QueryGymRawResult(reformulated="unused"))
        runner = InterventionRunner()
        with self.assertRaisesRegex(RuntimeError, "timeout after dispatch"):
            runner.run(
                QueryGymLiveBridge(None),
                FrozenAgentState(query="q"),
                InterventionContext(external_runtime=runtime),
            )
        self.assertEqual(len(runtime.requests), 1)
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 1)
        self.assertEqual(snapshot.usage.unpriced_model_calls, 1)

    def test_missing_external_usage_is_rejected_after_preserving_known_minimum(self) -> None:
        runtime = RecordingQueryGymRuntime(QueryGymRawResult(reformulated="rewritten"))
        runner = InterventionRunner()
        with self.assertRaisesRegex(AlgorithmOutputError, "usage"):
            runner.run(
                QueryGymLiveBridge(None),
                FrozenAgentState(query="q"),
                InterventionContext(external_runtime=runtime),
            )
        snapshot = runner.monitor.list_snapshots()[0]
        self.assertEqual(snapshot.usage.model_calls, 1)
        self.assertEqual(snapshot.usage.unpriced_model_calls, 1)


if __name__ == "__main__":
    unittest.main()
