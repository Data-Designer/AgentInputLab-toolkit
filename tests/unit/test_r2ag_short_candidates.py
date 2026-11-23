"""Small deterministic compatibility fixtures, not real-model acceptance."""
from types import SimpleNamespace

import pytest

from agentinputlab import RunConfig
from agentinputlab.algorithms.acquire.r2ag import (
    LocalR2AGCommonSupportRuntime,
    R2AGCommonSupport,
    R2AGCommonSupportConfig,
    R2AGReaderResult,
)
from agentinputlab.algorithms.acquire.r2ag.bridge import compute_r2ag_retrieval_features
from agentinputlab.algorithms.acquire.r2ag.candidate_count import (
    R2AG_SHORT_CANDIDATE_COMPAT,
    _short_pool_forward,
    short_pool_features,
)
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ExternalDependencyError,
    FrozenAgentState,
    InterventionContext,
    ValidationError,
)
from tests.algorithms.unit.test_r2ag_common_support import (
    FixedEncoder,
    FixedReader,
    FixedRetriever,
)


def test_singleton_absent_neighbor_is_explicit_zero_not_a_duplicate():
    assert dict(short_pool_features((1, 0), [(0, 1)])[0]) == {
        "rerank": 0, "neighbor": 0, "precedent": 1,
    }
    with pytest.raises(ValidationError):
        compute_r2ag_retrieval_features((1, 0), [(0, 1)])


@pytest.mark.parametrize("count", [2, 6])
def test_multi_document_features_are_unchanged(count):
    docs = [(1.0, float(i + 1)) for i in range(count)]
    assert short_pool_features((1, 0), docs) == compute_r2ag_retrieval_features((1, 0), docs)


@pytest.mark.parametrize("query,docs", [([], [(1, 0)]), ((1, 0), []),
                                      ((0, 0), [(1, 0)]), ((1, 0), [(0, 0)])])
def test_invalid_feature_vectors_still_fail(query, docs):
    with pytest.raises((ValidationError, AlgorithmOutputError)):
        short_pool_features(query, docs)


class SmallRetriever(FixedRetriever):
    def retrieve(self, query, *, top_k=6):
        return super().retrieve(query, top_k=1)


class SmallReader(FixedReader):
    def run(self, *, query, documents, features, max_new_tokens, seed):
        assert len(documents) == len(features) == 1
        return R2AGReaderResult(anchor="Synthetic fixture hypothesis", input_tokens=10,
            output_tokens=3, marker_positions=(4,), rformer_output_sha256="a" * 64,
            projection_output_sha256="b" * 64)

    def fingerprint(self):
        return {**super().fingerprint(), "candidate_count_adapter": R2AG_SHORT_CANDIDATE_COMPAT}


def test_short_intervention_emits_one_real_document_and_one_anchor():
    runtime = LocalR2AGCommonSupportRuntime(reader=SmallReader(), feature_encoder=FixedEncoder(),
        allow_short_candidates=True)
    # Fixture encoder returns six vectors; bind the actual small input instead.
    runtime.feature_encoder = SimpleNamespace(encode=lambda **kw: ((1, 0), ((0, 1),)),
                                             fingerprint=lambda: {"fixture_only": True})
    state = FrozenAgentState(query="visible fixture")
    with pytest.raises(AlgorithmOutputError, match="exact checkpoint"):
        R2AGCommonSupport().propose(state, InterventionContext(retriever=SmallRetriever(), external_runtime=runtime))
    record = R2AGCommonSupport(R2AGCommonSupportConfig(allow_short_candidates=True)).propose(
        state, InterventionContext(retriever=SmallRetriever(), external_runtime=runtime))
    assert len(record.operations[0].value) == 2
    assert record.trace.events[0].payload["real_candidate_count"] == 1
    assert record.trace.counters["model_calls"] == 1


def test_run_config_validates_opt_in_without_loading_models():
    assert RunConfig("acquire.r2ag.common_support", parameters={
        "config": {"allow_short_candidates": True}}).validate()["valid"]
    with pytest.raises(ValidationError, match="boolean"):
        R2AGCommonSupportConfig(allow_short_candidates="true")


def test_file_runtime_forwards_opt_in_to_both_reader_and_runtime(monkeypatch):
    from agentinputlab.algorithms.acquire import r2ag as module
    from tests.unit.test_run_runtime_configuration import r2ag as runtime_config

    monkeypatch.delenv(module.R2AG_RUNTIME_SOCKET_ENV, raising=False)
    seen = {}
    def capture(name):
        def construct(**kwargs):
            seen[name] = kwargs
            return kwargs
        return construct
    monkeypatch.setattr(module, "LocalR2AGCommonSupportReader", capture("reader"))
    monkeypatch.setattr(module, "LocalSentenceTransformerR2AGFeatureEncoder", capture("encoder"))
    monkeypatch.setattr(module, "LocalR2AGCommonSupportRuntime", capture("runtime"))
    RunConfig("acquire.r2ag.common_support", runtime={
        **runtime_config(), "allow_short_candidates": True})._runtime(load=True)
    assert seen["reader"]["allow_short_candidates"] is True
    assert seen["runtime"]["allow_short_candidates"] is True
    assert "allow_short_candidates" not in seen["encoder"]


SOURCE = '''
class RFormer:
    def forward(self, x, label=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.input_layer(x)
        position_ids = torch.arange(self.num_k, device=x.device).expand(x.shape[0], self.num_k)
        pe = self.position_embeddings(position_ids)
        x = x + pe
        x = self.attention_layer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return x, None
'''


def test_tensor_prefix_preserves_full_six_document_output_and_weights():
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    scope = {"torch": torch}
    exec(compile(SOURCE, "<fixture-rformer>", "exec"), scope)
    model = SimpleNamespace(num_k=6, input_layer=torch.nn.Linear(3, 8),
        position_embeddings=torch.nn.Embedding(6, 8),
        attention_layer=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, dropout=0), num_layers=1).eval())
    old = scope["RFormer"].forward
    new = _short_pool_forward(SOURCE, {"torch": torch}, "<adapted-fixture>")
    data = torch.randn(6, 3)
    assert torch.equal(old(model, data)[0], new(model, data)[0])
    for count in (1, 2):
        assert new(model, data[:count])[0].shape == (1, count, 8)
    with pytest.raises(ValueError, match="capacity"):
        new(model, torch.randn(7, 3))


def test_unexpected_source_axis_expression_is_not_patched():
    with pytest.raises(ExternalDependencyError, match="positional-axis"):
        _short_pool_forward(SOURCE.replace("self.num_k)", "6)"), {}, "<unexpected-fixture>")
