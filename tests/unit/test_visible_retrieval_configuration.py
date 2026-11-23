"""Per-case binding tests; synthetic fixtures, not real-model acceptance."""
import hashlib
import json
from dataclasses import replace

import pytest

from agentinputlab import FrozenAgentState, InterventionContext, RunConfig
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import BenchmarkExample
from agentinputlab.contracts import ContextBlock
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.visible_retrieval_configuration import VisibleContextBM25
from tests.unit.test_benchmark_configuration import Backend


def state(name="a"):
    return FrozenAgentState(query="apple", task_id=name, context=(
        ContextBlock("apple fruit " + name, block_id=name + "1", metadata={"title": "fruit"}),
        ContextBlock("pear " + name, block_id=name + "2"),
    ))


def test_validate_is_unbound_without_loading_or_inference():
    config = RunConfig("baseline.no_op", retriever={"kind": "visible_context_bm25"})
    assert config.validate()["runtime_ready"] is None
    with pytest.raises(ValidationError, match="bound to a case"):
        config._retriever().retrieve("apple")


def test_binding_is_identical_to_explicit_existing_bm25():
    s = state()
    explicit = InMemoryBM25Retriever(tuple(Document(
        b.block_id, b.text, title=b.metadata.get("title", b.block_id),
        corpus_revision=content_hash(s.context), metadata={k: v for k, v in b.metadata.items() if k != "title"}) for b in s.context),
        k1=1.5, b=0.6)
    bound = VisibleContextBM25({"kind": "visible_context_bm25", "k1": 1.5, "b": 0.6}).bind(s)
    assert bound.fingerprint() == explicit.fingerprint()
    assert bound.retrieve("apple", top_k=2) == explicit.retrieve("apple", top_k=2)


def test_old_retrieval_provenance_cannot_spoof_new_ranking():
    block = ContextBlock("apple", block_id="a", metadata={"rank": 99,
        "document_id": "old", "title": "apple", "official_support_id": "a"})
    result, = VisibleContextBM25({"kind": "visible_context_bm25"}).bind(
        replace(state(), context=(block,))).retrieve("apple")
    assert result.rank == 1 and result.document.doc_id == "a"
    assert result.document.metadata["official_support_id"] == "a"
    assert result.document.metadata["visible_context_provenance"] == {"rank": 99, "document_id": "old"}
    assert "title" not in result.document.metadata


def test_text_only_projection_is_explicit_and_matches_existing_bindings():
    s = state()
    binding = VisibleContextBM25({"kind": "visible_context_bm25",
                                 "document_projection": "text_only", "corpus_revision": "frozen-rev"})
    reference = InMemoryBM25Retriever(tuple(Document(b.block_id, b.text,
                                          corpus_revision="frozen-rev") for b in s.context))
    assert binding.bind(s).fingerprint() == reference.fingerprint()
    assert binding.bind(s).retrieve("apple") == reference.retrieve("apple")


def test_chunked_binding_indexes_full_document_without_modifying_state():
    original = "Chair:  open.\nA:  detail.\nB:  close.\n"
    s = replace(state(), context=(ContextBlock(original, block_id="meeting:1"),))
    binding = VisibleContextBM25({"kind": "visible_context_bm25", "chunk_words": 2})
    results = binding.bind(s).retrieve("detail", top_k=10)
    assert len(results) == 3
    assert "detail" in results[0].document.text
    chunks = sorted((r.document for r in results),
                    key=lambda d: d.metadata["visible_chunk_source"]["char_start"])
    assert "".join(c.text for c in chunks) == original
    for c in chunks:
        p = c.metadata["visible_chunk_source"]
        assert p["block_id"] == "meeting:1"
        assert p["block_sha256"] == hashlib.sha256(original.encode()).hexdigest()
        assert c.text == original[p["char_start"]:p["char_end"]]
    assert s.context[0].text == original


@pytest.mark.parametrize("value", [True, 0, -1, "350", None])
def test_invalid_chunk_words_is_rejected_before_execution(value):
    with pytest.raises(ValidationError, match="chunk_words"):
        VisibleContextBM25({"kind": "visible_context_bm25", "chunk_words": value})


@pytest.mark.parametrize("value", [{"k1": True}, {"k1": 0}, {"b": 1.1},
                                   {"b": float("inf")}, {"documents": []},
                                   {"document_projection": False}, {"corpus_revision": ""},
                                   {"corpus_revision": None}])
def test_bad_parameters_fail_without_execution(value):
    with pytest.raises(ValidationError):
        RunConfig("baseline.no_op", retriever={"kind": "visible_context_bm25", **value}).validate()


@pytest.mark.parametrize("blocks", [(), (ContextBlock("x"),),
    (ContextBlock("x", block_id="same"), ContextBlock("y", block_id="same")),
    (ContextBlock("x", block_id="x", metadata={"title": 1}),)])
def test_invalid_context_not_padded_or_repaired(blocks):
    with pytest.raises(ValidationError):
        VisibleContextBM25({"kind": "visible_context_bm25"}).bind(replace(state(), context=blocks))


def test_explicit_context_retriever_conflict_rejected():
    config = RunConfig("baseline.no_op", retriever={"kind": "visible_context_bm25"})
    with pytest.raises(ValidationError, match="retriever is configured twice"):
        config.run(state(), context=InterventionContext(retriever=InMemoryBM25Retriever(())))


def test_run_binds_per_case_and_keeps_original_state():
    config = RunConfig("baseline.no_op", retriever={"kind": "visible_context_bm25"})
    original = state()
    context = config._configured_context(None)
    first = config._bind_case_context(context, original)
    second = config._bind_case_context(context, state("b"))
    assert {r.document.doc_id for r in first.retriever.retrieve("apple")} == {"a1", "a2"}
    assert {r.document.doc_id for r in second.retriever.retrieve("apple")} == {"b1", "b2"}
    assert original == state()
    assert config.run(original)["record"]["input_state_hash"] == original.state_hash


def test_file_configuration_roundtrip_needs_no_python_retriever_injection(tmp_path):
    configured = RunConfig("baseline.no_op", retriever={"kind": "visible_context_bm25"})
    path = tmp_path / "run.json"
    path.write_text(json.dumps(configured.to_dict()))
    loaded = RunConfig.from_file(path)
    assert loaded.to_dict() == configured.to_dict()
    assert loaded.run(state())["record"]["input_state_hash"] == state().state_hash


def test_batch_downstream_uses_each_case_retriever_and_gold_stays_hidden():
    class Adapter:
        benchmark_id = "2wiki.multihop"
        pairing_mode = "stateless_score_twice"

        def examples(self, split):
            return tuple(BenchmarkExample(example_id=x, query="apple",
                visible_context=state(x).context, evaluator_gold={"secret": "DO_NOT_SEND_GOLD"})
                for x in ("a", "b"))

        def score(self, example, prediction):
            return {"answer_em": float(prediction.answer == "fruit")}

    backend = Backend([("The answer is fruit.", 2), ("fruit", 2),
                       ("The answer is fruit.", 2), ("fruit", 3)])
    config = RunConfig("acquire.ircot.common_support",
        parameters={"config": {"demonstrations": "Frozen example", "max_steps": 1,
            "prompt_sha256": hashlib.sha256(b"Frozen example").hexdigest()}},
        retriever={"kind": "visible_context_bm25"})
    result = config.evaluate("2wiki.multihop", split="dev", limit=2, adapter=Adapter(),
        context=InterventionContext(backend=backend),
        predictor_parameters={"schema_version": 1, "qa": {"output_format": "answer_only"},
                              "retrieval": {"source_mode": "retriever", "top_k": 2}})
    assert result["scored_count"] == 2
    assert all(r["usage"]["retrieval_calls"] == 1 for r in result["records"])
    assert result["records"][0]["case_retriever"]["fingerprint"] != result["records"][1]["case_retriever"]["fingerprint"]
    assert "DO_NOT_SEND_GOLD" not in str(backend.requests)
    assert "apple fruit a" in backend.requests[0].prompt and "apple fruit b" not in backend.requests[0].prompt
    assert "apple fruit b" in backend.requests[2].prompt and "apple fruit a" not in backend.requests[2].prompt


def test_ir_downstream_uses_bound_not_placeholder_retriever():
    class Adapter:
        benchmark_id = "beir.arguana"
        pairing_mode = "stateless_score_twice"

        def examples(self, split):
            return tuple(BenchmarkExample(example_id=x, query="apple",
                         visible_context=state(x).context) for x in ("a", "b"))

        def score(self, example, prediction):
            assert set(prediction.scores) == {example.example_id + "1", example.example_id + "2"}
            return {"test_score": 1.0}

    result = RunConfig("baseline.no_op", retriever={"kind": "visible_context_bm25"}).evaluate(
        "beir.arguana", split="test", limit=2, adapter=Adapter(),
        predictor_parameters={"schema_version": 1,
            "retrieval": {"source_mode": "retriever", "query_mode": "query_only", "top_k": 2}})
    assert result["scored_count"] == 2
    assert all(r["usage"]["retrieval_calls"] == 1 and not r["model_responses"] for r in result["records"])
