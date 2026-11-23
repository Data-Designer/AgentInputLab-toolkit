"""BFCL cold-cache routing contracts; real source acceptance is separate."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from agentinputlab.artifacts import ArtifactStore
from agentinputlab.benchmarks import cache, describe_benchmark, load_benchmark, prepare_benchmark
from agentinputlab.contracts import ArtifactError, ValidationError

CELL = "bfcl.v3.simple_python"


@pytest.fixture
def source(monkeypatch):
    asset = cache.BenchmarkSmokeAssetRegistry.default().get("bfcl.v3.simple_python.v1_3")
    asset = replace(asset, size_bytes=7, sha256=sha256(b"fixture").hexdigest())
    monkeypatch.setattr(cache.BenchmarkSmokeAssetRegistry, "default",
                        lambda: SimpleNamespace(get=lambda key: asset))
    normalized = []
    monkeypatch.setattr(cache, "prepare_smoke_asset",
                        lambda a, path, root, **kw: normalized.append((a, path, root, kw)), raising=False)
    return asset, normalized


def test_discovery_exposes_locked_source_and_model_requirement():
    info = describe_benchmark(CELL)
    assert info["automatic_loader"] is True
    assert info["source_download_bytes"] == 44552456
    assert info["runtime_parameters"] == ["model_id"]
    assert info["split"] == "test"


def test_offline_missing_does_not_download(tmp_path, monkeypatch, source):
    monkeypatch.setattr(cache, "download_artifact_streamed", lambda *a, **kw: pytest.fail("network"))
    with pytest.raises(ArtifactError, match="offline"):
        prepare_benchmark(CELL, cache_dir=tmp_path, offline=True)
    assert not source[1]


def test_source_budget_rejected_before_download(tmp_path, monkeypatch, source):
    monkeypatch.setattr(cache, "download_artifact_streamed", lambda *a, **kw: pytest.fail("network"))
    with pytest.raises(ArtifactError, match="max_download_bytes"):
        prepare_benchmark(CELL, cache_dir=tmp_path, max_download_bytes=6)


def test_download_then_offline_reuse_includes_scorer_archive(tmp_path, monkeypatch, source):
    asset, normalized = source
    calls = []
    def download(manifest, store, *, policy, transport):
        calls.append(policy)
        return store.put_stream(manifest, [b"fixture"])
    monkeypatch.setattr(cache, "download_artifact_streamed", download)
    first = prepare_benchmark(CELL, cache_dir=tmp_path)
    second = prepare_benchmark(CELL, cache_dir=tmp_path, offline=True)
    assert first["acquired_from"] == "network" and second["acquired_from"] == "cache"
    assert first["source_archive"] == second["source_archive"]
    assert first["source_sha256"] == asset.sha256 and len(normalized) == 2
    assert len(calls) == 1 and calls[0].allowed_hosts == asset.allowed_hosts


def test_corrupt_cache_not_redownloaded_or_overwritten(tmp_path, monkeypatch, source):
    store = ArtifactStore(tmp_path / "sources")
    path = store.put_stream(source[0].artifact_manifest(), [b"fixture"])
    path.write_bytes(b"broken!")
    monkeypatch.setattr(cache, "download_artifact_streamed", lambda *a, **kw: pytest.fail("network"))
    with pytest.raises(ArtifactError, match="checksum"):
        prepare_benchmark(CELL, cache_dir=tmp_path)
    assert path.read_bytes() == b"broken!" and not source[1]


def test_mounted_archive_used_without_copy_or_network(tmp_path, monkeypatch, source):
    path = tmp_path / "source.tar.gz"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(cache, "download_artifact_streamed", lambda *a, **kw: pytest.fail("network"))
    result = prepare_benchmark(CELL, cache_dir=tmp_path / "cache", source_archive=path, offline=True)
    assert result["source_archive"] == str(path) and result["acquired_from"] == "mounted"
    assert source[1][0][1] == path
    assert not (tmp_path / "cache" / "sources").exists()


def test_source_option_not_silently_ignored_for_other_dataset(tmp_path):
    with pytest.raises(ValidationError, match="source_archive.*BFCL"):
        prepare_benchmark("beir.arguana", cache_dir=tmp_path, source_archive="wrong")


def test_prepared_data_alone_does_not_hide_missing_scorer_source(tmp_path, source):
    prepared = cache._prepared_path(source[0], tmp_path / "prepared")
    prepared.mkdir(parents=True)
    with pytest.raises(ArtifactError, match="scorer source.*offline"):
        prepare_benchmark(CELL, cache_dir=tmp_path, offline=True)
    assert not source[1]


def test_unverified_mounted_source_rejected_without_normalizing(tmp_path):
    path = tmp_path / "not-official.tar.gz"
    path.write_bytes(b"not official")
    with pytest.raises(ArtifactError, match="source size mismatch"):
        prepare_benchmark(CELL, cache_dir=tmp_path / "cache", source_archive=path, offline=True)
    assert not (tmp_path / "cache" / "prepared").exists()


@pytest.mark.parametrize("runtime", [{}, {"model_id": ""}, {"model_id": 1}, {"model_id": "q4", "typo": True}])
def test_invalid_model_runtime_rejected_before_download(monkeypatch, runtime):
    monkeypatch.setattr(cache, "prepare_benchmark", lambda *a, **kw: pytest.fail("prepare"))
    with pytest.raises(ValidationError):
        load_benchmark(CELL, **runtime)


def test_loader_constructs_original_native_bridge(monkeypatch, tmp_path):
    import agentinputlab.benchmarks.bfcl_native as native
    import agentinputlab.benchmarks.bfcl_v3 as adapter
    seen = []
    bridge, result = object(), object()
    monkeypatch.setattr(cache, "prepare_benchmark", lambda *a, **kw: {
        "kind": "bfcl_v3_simple", "source_archive": "pinned.tar.gz", "prepared_root": str(tmp_path)})
    monkeypatch.setattr(native, "SourceVerifiedBFCLV3ASTBridge",
                        lambda **kw: seen.append(kw) or bridge)
    monkeypatch.setattr(adapter, "load_bfcl_v3_simple_adapter",
                        lambda root, *, evaluation_bridge: result if root == str(tmp_path) and evaluation_bridge is bridge else None)
    assert load_benchmark(CELL, model_id="actual-q4") is result
    assert seen == [{"source_archive": "pinned.tar.gz", "model_id": "actual-q4"}]
