"""Cold/warm source preparation contracts; synthetic bytes are not benchmark evidence."""
from dataclasses import replace
from hashlib import sha256

import pytest

from agentinputlab.artifacts import ArtifactStore
from agentinputlab.benchmarks import cache, describe_benchmark, load_benchmark, prepare_benchmark
from agentinputlab.contracts import ArtifactError

CELL = "longbench.qasper"


def test_discovery_lists_native_sources_and_cold_size():
    info = describe_benchmark(CELL)
    assert info["automatic_loader"] is True and info["split"] == "test"
    assert info["source_download_bytes"] == 113943156
    assert len(info["sources"]) == 3 and info["license_review_required"]


@pytest.fixture
def sources(monkeypatch):
    from agentinputlab.benchmarks.longbench_sources import manifests

    real = manifests()
    values = {key: key.encode() for key in real}
    fake = {key: replace(m, sha256=sha256(values[key]).hexdigest(), size_bytes=len(values[key]))
            for key, m in real.items()}
    monkeypatch.setattr(cache, "_longbench_manifests", lambda: fake)
    calls = []
    def download(manifest, store, *, policy, transport):
        key = next(k for k, m in fake.items() if m == manifest)
        calls.append((key, policy))
        return store.put_stream(manifest, [values[key]])
    monkeypatch.setattr(cache, "download_artifact_streamed", download)
    return fake, values, calls


def test_missing_offline_and_licence_fail_before_download(tmp_path, sources):
    with pytest.raises(ArtifactError, match="terms"):
        prepare_benchmark(CELL, cache_dir=tmp_path)
    with pytest.raises(ArtifactError, match="offline"):
        prepare_benchmark(CELL, cache_dir=tmp_path, offline=True, accept_license_review=True)
    assert not sources[2]


def test_total_missing_source_cap_not_per_file(tmp_path, sources):
    sizes = [m.size_bytes for m in sources[0].values()]
    with pytest.raises(ArtifactError, match="max_download_bytes"):
        prepare_benchmark(CELL, cache_dir=tmp_path, accept_license_review=True, max_download_bytes=max(sizes))
    assert not sources[2]


def test_all_six_subsets_share_verified_cache_and_reuse(tmp_path, sources):
    first = prepare_benchmark(CELL, cache_dir=tmp_path, accept_license_review=True)
    second = prepare_benchmark("longbench.qmsum", cache_dir=tmp_path, accept_license_review=True, offline=True)
    assert first["longbench_sources"] == second["longbench_sources"]
    assert len(sources[2]) == 3
    assert set(second["acquired_from"].values()) == {"cache"}


def test_corrupt_cache_not_overwritten(tmp_path, sources):
    store = ArtifactStore(tmp_path / "sources")
    key = "scorer_path"
    path = store.put_stream(sources[0][key], [sources[1][key]])
    path.write_bytes(b"wrong")
    with pytest.raises(ArtifactError, match="checksum|size"):
        prepare_benchmark(CELL, cache_dir=tmp_path, accept_license_review=True)
    assert path.read_bytes() == b"wrong" and not sources[2]


def test_only_missing_files_count_against_cap(tmp_path, sources):
    store = ArtifactStore(tmp_path / "sources")
    missing = "prompts_path"
    for key, manifest in sources[0].items():
        if key != missing:
            store.put_stream(manifest, [sources[1][key]])
    report = prepare_benchmark(CELL, cache_dir=tmp_path, accept_license_review=True,
        max_download_bytes=sources[0][missing].size_bytes)
    assert [key for key, _ in sources[2]] == [missing]
    assert report["acquired_from"][missing] == "network"


def test_load_delegates_exact_sources_to_native_adapter(tmp_path, sources, monkeypatch):
    import agentinputlab.benchmarks.longbench as lb

    seen = []
    monkeypatch.setattr(lb, "LongBenchAdapter", lambda **kw: seen.append(kw) or "adapter")
    assert load_benchmark(CELL, cache_dir=tmp_path, accept_license_review=True) == "adapter"
    assert seen[0]["benchmark_id"] == CELL
    assert set(seen[0]) == {"benchmark_id", *sources[0]}
