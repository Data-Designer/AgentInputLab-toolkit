"""Singleton fixtures exercise native exhaustion, not model acceptance."""
from unittest.mock import patch

import pytest

from agentinputlab import RunConfig
from agentinputlab.algorithms.acquire.qrag import (
    QRAGCommonSupport,
    QRAGCommonSupportConfig,
)
from agentinputlab.backends.retrieval import Document, RetrievalResult
from agentinputlab.contracts import (
    AlgorithmOutputError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from agentinputlab.contracts.serialization import content_hash
from tests.algorithms.unit.test_qrag_common_support import FixedRetriever, FixedRuntime


class SingletonRetriever(FixedRetriever):
    def retrieve(self, query, *, top_k=6):
        return (RetrievalResult(document=Document(doc_id="real-schema", text="original tool schema",
                                                  corpus_revision="visible-fixture-v1"),
                                rank=1, score=1.0),)


class SingletonRuntime(FixedRuntime):
    def select(self, request):
        self.requests.append(request)
        assert len(request.candidates) == 1 and request.max_steps == 2
        ids = (request.candidates[0].document.doc_id,)
        base = FixedRuntime().select(type(request)(
            question=request.question, candidates=FixedRetriever().retrieve(request.question),
            seed=request.seed,
        ))
        steps = ({"index": 0, "available_indices": (0,), "q_values": (0.5,),
                  "action_index": 0, "selected_q_value": 0.5,
                  "selected_indices": (0,), "selected_document_ids": ids},)
        return {**base, "trajectory": steps, "trajectory_sha256": content_hash(steps),
                "selected_indices": (0,), "selected_document_ids": ids,
                "candidate_order_sha256": content_hash(ids), "policy_steps": 1,
                "usage": {**base["usage"], "embedding_calls": 4}}


def test_opt_in_keeps_two_step_cap_and_one_real_document():
    runtime = SingletonRuntime()
    result = QRAGCommonSupport(QRAGCommonSupportConfig(allow_candidate_exhaustion=True)).propose(
        FrozenAgentState(query="visible task"),
        InterventionContext(retriever=SingletonRetriever(), external_runtime=runtime, seed=17),
    )
    assert runtime.requests[0].max_steps == 2
    assert result.trace.counters["policy_steps"] == 1
    assert result.trace.counters["embedding_calls"] == 4
    assert len(result.operations[0].value) == 1
    assert result.trace.events[0].payload["candidate_exhaustion_enabled"] is True


def test_default_does_not_change_or_call_policy():
    runtime = SingletonRuntime()
    with pytest.raises(AlgorithmOutputError, match="at least two"):
        QRAGCommonSupport().propose(FrozenAgentState(query="task"),
            InterventionContext(retriever=SingletonRetriever(), external_runtime=runtime))
    assert not runtime.requests


def test_exhaustion_does_not_allow_empty_pool_or_reduce_configured_cap():
    with (
        pytest.raises(AlgorithmOutputError, match="one real"),
        patch.object(SingletonRetriever, "retrieve", return_value=()),
    ):
        QRAGCommonSupport(QRAGCommonSupportConfig(allow_candidate_exhaustion=True)).propose(
            FrozenAgentState(query="task"),
            InterventionContext(retriever=SingletonRetriever(), external_runtime=SingletonRuntime()),
        )
    with pytest.raises(ValidationError, match="two selection"):
        QRAGCommonSupportConfig(allow_candidate_exhaustion=True, max_select_actions=1)


@pytest.mark.parametrize("value", [1, "true", None])
def test_exhaustion_flag_is_strict_boolean(value):
    with pytest.raises(ValidationError, match="must be boolean"):
        QRAGCommonSupportConfig(allow_candidate_exhaustion=value)


def test_run_config_accepts_explicit_exhaustion():
    assert RunConfig("acquire.qrag.common_support", parameters={"config": {
        "allow_candidate_exhaustion": True, "candidate_top_k": 1},
    }).validate()["valid"]


def test_runtime_builder_forwards_opt_in_without_changing_default():
    from agentinputlab.experiments.production_builders import (
        _build_external_runtime,
        _parse_external_runtime_block,
    )
    settings = {"kind": "qrag_common_support", "source_root": "/source",
        "checkpoint_root": "/weights", "embedder_root": "/embedder",
        "compatibility_manifest_path": "/manifest.json",
        "compatibility_manifest_sha256": "a" * 64, "embedder_manifest_sha256": "b" * 64}
    for enabled in (False, True):
        configured = _parse_external_runtime_block({**settings, "single_candidate_axis_compat": enabled})
        with patch("agentinputlab.algorithms.acquire.qrag.LocalQRAGFrozenPolicyRuntime") as ctor:
            _build_external_runtime(configured, backend=None)
        assert ctor.call_args.kwargs["single_candidate_axis_compat"] is enabled
    with pytest.raises(ValidationError, match="must be boolean"):
        _parse_external_runtime_block({**settings, "single_candidate_axis_compat": "true"})
