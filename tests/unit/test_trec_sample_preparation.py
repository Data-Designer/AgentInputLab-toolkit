"""Synthetic source fixtures; no network/model or benchmark-quality claims."""

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

from agentinputlab.benchmarks import TRECDLSampleAdapter
from agentinputlab.cli.main import main
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.data import DownloadResponse, prepare_trec_dl_sample

REVISION = "a" * 40


class NativeSourceFixture:
    def __init__(self):
        self.calls = []
        self.fail_once = None
        self.peer = "93.184.216.34"
        self.truncated = False
        self.revision = REVISION

    def request(self, target, **kwargs):
        self.calls.append(target.url)
        if self.fail_once and self.fail_once in target.url:
            self.fail_once = None
            raise ArtifactError("SYNTHETIC network interruption")
        parts = urlsplit(target.url)
        if parts.path.startswith("/api/datasets/"):
            value = {"sha": self.revision}
        elif parts.path == "/splits":
            value = {"splits": [{"config": "corpus", "split": "corpus"}]}
        elif parts.path.endswith("qrels-pass.txt"):
            unused = "0" if "2020" in parts.path else "Q0"
            raw = b"".join(
                f"q{q:03d} {unused} {40 + q * 6 + d} {grade}\n".encode()
                for q in range(6)
                for d, grade in enumerate((0, 0, 1, 2, 2, 3))
            )
            return DownloadResponse(200, {}, raw, self.peer)
        elif parts.path.endswith(".tsv.gz"):
            raw = gzip.compress(
                b"".join(f"q{q:03d}\tNative question {q}\n".encode() for q in range(6))
            )
            return DownloadResponse(200, {}, raw, self.peer)
        elif parts.path == "/rows":
            query = parse_qs(parts.query)
            offset = int(query["offset"][0])
            length = int(query["length"][0])
            assert length <= 100 and query["config"] == ["corpus"]
            value = {
                "rows": [
                    {
                        "row": {"_id": str(i), "title": "", "text": f"Native passage {i}"},
                        "truncated_cells": ["text"] if self.truncated else [],
                    }
                    for i in range(offset, offset + length)
                ]
            }
        else:
            raise AssertionError(target.url)
        return DownloadResponse(200, {}, json.dumps(value).encode(), self.peer)


@pytest.fixture(autouse=True)
def fixed_dns():
    with (
        patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))]),
        patch("socket.socket.connect", side_effect=AssertionError("real network forbidden")),
    ):
        yield


def prepare(root, transport, **kwargs):
    return prepare_trec_dl_sample(
        root,
        year=2019,
        accept_license_review=True,
        expected_corpus_revision=REVISION,
        transport=transport,
        **kwargs,
    )


def test_real_adapter_verifies_both_years_and_disjoint_offsets(tmp_path):
    for year in (2019, 2020):
        results = []
        for offset in (0, 2):
            result = prepare_trec_dl_sample(
                tmp_path / f"{year}-{offset}",
                year=year,
                query_offset=offset,
                accept_license_review=True,
                transport=NativeSourceFixture(),
            )
            adapter = TRECDLSampleAdapter(
                result["root"], manifest_sha256=result["manifest_sha256"], license_acknowledged=True
            )
            assert result["counts"] == {"inputs": 2, "gold": 2, "corpus": 44}
            assert not result["full_dataset_score"]
            for e in adapter.examples("test"):
                assert any(x >= 2 for x in e.evaluator_gold["relevance"].values())
                assert "relevance" not in repr(e.visible_state())
            results.append({e.example_id for e in adapter.examples("test")})
        assert results[0].isdisjoint(results[1])


def test_offline_verified_cache_has_no_requests(tmp_path):
    transport = NativeSourceFixture()
    first = prepare(tmp_path / "sample", transport)
    before = len(transport.calls)
    second = prepare(tmp_path / "sample", transport, offline=True)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert second["acquired_from"] == "verified_cache" and second["downloaded_bytes"] == 0
    assert len(transport.calls) == before


def test_partial_acquisition_resumes_without_repeating_verified_components(tmp_path):
    transport = NativeSourceFixture()
    transport.fail_once = "/rows?"
    with pytest.raises(ArtifactError, match="interruption"):
        prepare(tmp_path / "sample", transport)
    before = list(transport.calls)
    prepare(tmp_path / "sample", transport)
    assert transport.calls.count(next(x for x in before if "qrels-pass.txt" in x)) == 1
    assert transport.calls.count(next(x for x in before if ".tsv.gz" in x)) == 1
    assert not (tmp_path / "sample/.prepare-lock").exists()


def test_reuse_locked_official_source_bundle_keeps_original_test_sample(tmp_path):
    transport = NativeSourceFixture()
    old = prepare(tmp_path / "test", transport)
    original = {
        p.relative_to(tmp_path / "test"): p.read_bytes()
        for p in (tmp_path / "test").rglob("*")
        if p.is_file()
    }
    second = NativeSourceFixture()
    result = prepare(
        tmp_path / "fit",
        second,
        query_offset=2,
        source_sample_root=old["root"],
        source_manifest_sha256=old["manifest_sha256"],
    )
    assert result["counts"]["inputs"] == 2
    assert not any("qrels-pass.txt" in x or ".tsv.gz" in x for x in second.calls)
    assert all((tmp_path / "test" / n).read_bytes() == b for n, b in original.items())


def test_corruption_and_changed_controls_do_not_overwrite_or_redownload(tmp_path):
    transport = NativeSourceFixture()
    prepare(tmp_path / "sample", transport)
    before = len(transport.calls)
    with pytest.raises(ArtifactError, match="not overwritten"):
        prepare(tmp_path / "sample", transport, query_offset=2)
    (tmp_path / "sample/corpus.jsonl").write_bytes(b"BROKEN")
    with pytest.raises(ArtifactError):
        prepare(tmp_path / "sample", transport)
    assert len(transport.calls) == before
    assert (tmp_path / "sample/corpus.jsonl").read_bytes() == b"BROKEN"


def test_cache_directories_cannot_redirect_writes(tmp_path):
    destination = tmp_path / "cache"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "resources").symlink_to(outside, target_is_directory=True)
    transport = NativeSourceFixture()
    with pytest.raises(ArtifactError, match="directories"):
        prepare(destination, transport)
    assert not transport.calls and not list(outside.iterdir())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_count": True},
        {"query_count": 9},
        {"query_offset": -1},
        {"background_count": 101},
        {"max_download_bytes": 0},
        {"expected_corpus_revision": True},
        {"source_sample_root": "unlocked"},
    ],
)
def test_invalid_controls_fail_before_network(tmp_path, kwargs):
    with pytest.raises(ValidationError):
        prepare_trec_dl_sample(tmp_path / "bad", year=2019, accept_license_review=True, **kwargs)
    assert not (tmp_path / "bad").exists()


def test_license_gate_and_offline_source_miss_are_explicit(tmp_path):
    with pytest.raises(ValidationError, match="terms"):
        prepare_trec_dl_sample(tmp_path / "no", year=2019)
    with pytest.raises(ArtifactError, match="offline"):
        prepare(tmp_path / "missing", NativeSourceFixture(), offline=True)


def test_wrong_revision_truncated_rows_and_foreign_peer_rejected(tmp_path):
    with pytest.raises(ArtifactError, match="revision"):
        prepare_trec_dl_sample(
            tmp_path / "rev",
            year=2019,
            accept_license_review=True,
            expected_corpus_revision="b" * 40,
            transport=NativeSourceFixture(),
        )
    t = NativeSourceFixture()
    t.truncated = True
    with pytest.raises(ArtifactError, match="truncated"):
        prepare(tmp_path / "truncated", t)
    t = NativeSourceFixture()
    t.peer = "1.1.1.1"
    with pytest.raises(ArtifactError, match="peer"):
        prepare(tmp_path / "peer", t)


def test_parallel_same_cache_prepares_once(tmp_path):
    transport = NativeSourceFixture()
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: prepare(tmp_path / "shared", transport), range(2)))
    assert results[0]["manifest_sha256"] == results[1]["manifest_sha256"]
    assert transport.calls.count("https://huggingface.co/api/datasets/BeIR/msmarco") == 1


def test_changed_corpus_commit_never_relabels_cached_rows(tmp_path):
    t = NativeSourceFixture()
    t.fail_once = "offset=40"
    kwargs = dict(year=2019, accept_license_review=True, transport=t)
    with pytest.raises(ArtifactError, match="interruption"):
        prepare_trec_dl_sample(tmp_path / "partial", **kwargs)
    t.revision = "b" * 40
    with pytest.raises(ArtifactError, match="integrity"):
        prepare_trec_dl_sample(tmp_path / "partial", **kwargs)
    assert not (tmp_path / "partial/manifest.json").exists()


def test_old_bundle_corpus_rows_not_reused_under_a_new_commit(tmp_path):
    old = prepare(tmp_path / "old", NativeSourceFixture())
    t = NativeSourceFixture()
    t.revision = "b" * 40
    new = prepare_trec_dl_sample(
        tmp_path / "new",
        year=2019,
        query_offset=2,
        expected_corpus_revision=t.revision,
        accept_license_review=True,
        transport=t,
        source_sample_root=old["root"],
        source_manifest_sha256=old["manifest_sha256"],
    )
    assert new["counts"]["inputs"] == 2
    assert any("offset=0&length=32" in url for url in t.calls)
    assert not any("qrels-pass.txt" in url for url in t.calls)


def test_declared_size_and_total_download_caps_are_enforced(tmp_path):
    class Oversized(NativeSourceFixture):
        def request(self, target, **kwargs):
            return DownloadResponse(200, {"Content-Length": str(2**30)}, b"{}", self.peer)

    with pytest.raises(ArtifactError, match="size"):
        prepare(tmp_path / "large", Oversized())
    with pytest.raises(ArtifactError, match="allowance"):
        prepare(tmp_path / "total", NativeSourceFixture(), max_download_bytes=64)


def test_cli_passes_native_sample_controls(tmp_path, capsys):
    with patch(
        "agentinputlab.data.prepare_trec_dl_sample", return_value={"full_dataset_score": False}
    ) as call:
        assert (
            main(
                [
                    "datasets",
                    "prepare-trec-sample",
                    str(tmp_path / "data"),
                    "--year",
                    "2020",
                    "--query-offset",
                    "2",
                    "--accept-license-review",
                    "--offline",
                ]
            )
            == 0
        )
    assert call.call_args.kwargs["query_offset"] == 2 and call.call_args.kwargs["offline"] is True
    assert json.loads(capsys.readouterr().out)["full_dataset_score"] is False
