"""Synthetic file-binding tests; not real-model acceptance results."""

import json
from unittest.mock import patch

import pytest

from agentinputlab import RunConfig
from agentinputlab.beir_sample_configuration import load, read_sources, sources
from agentinputlab.cli.main import main
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from tests.unit.test_beir_sample import write_bundle


def sample(tmp_path):
    root = tmp_path / "sample"
    root.mkdir()
    digest, _ = write_bundle(root)
    return dict(root=str(root), manifest_sha256=digest)


def test_file_relative_path_and_offline_adapter_preserve_sample_scope(tmp_path):
    value = sample(tmp_path)
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(dict(value, root="sample")))
    assert read_sources(path) == value
    with patch("socket.socket.connect", side_effect=AssertionError("no network")):
        adapter = load(value, benchmark_id="beir.fever", split="test", accept_license_review=True)
    assert adapter.fingerprint()["full_dataset_score"] is False
    assert adapter.fingerprint()["metrics_scope"] == "sampled_candidate_pool_only"


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"root": "a"},
        {"root": 1, "manifest_sha256": "a" * 64},
        {"root": "a", "manifest_sha256": "main"},
        {"root": "a", "manifest_sha256": "a" * 64, "guess": True},
    ],
)
def test_invalid_configuration_rejected(value):
    with pytest.raises(ValidationError):
        sources(value)


@pytest.mark.parametrize(
    "benchmark,split,ack",
    [
        ("bright.robotics", "test", True),
        ("beir.fever", "train", True),
        ("beir.fever", "test", False),
    ],
)
def test_wrong_scope_or_no_terms_ack_fails_before_loading(benchmark, split, ack):
    with pytest.raises(ValidationError):
        load(
            dict(root="/not-present", manifest_sha256="a" * 64),
            benchmark_id=benchmark,
            split=split,
            accept_license_review=ack,
        )


def test_corruption_and_benchmark_mismatch_rejected(tmp_path):
    value = sample(tmp_path)
    with pytest.raises(ValidationError, match="identity mismatch"):
        load(value, benchmark_id="beir.cqadupstack", split="test", accept_license_review=True)
    with pytest.raises(ArtifactError, match="checksum"):
        load(
            dict(value, manifest_sha256="f" * 64),
            benchmark_id="beir.fever",
            split="test",
            accept_license_review=True,
        )


@pytest.mark.parametrize(
    "other",
    [
        {"prepared_root": "data"},
        {"cache_dir": "cache"},
        {"adapter": object()},
        {"longbench_sources": {}},
        {"bfcl_source_archive": "archive"},
    ],
)
def test_conflicting_sources_fail_before_model_or_dataset_access(other):
    with pytest.raises(ValidationError, match="owns its dataset source"):
        RunConfig("baseline.no_op").evaluate(
            "beir.fever",
            split="test",
            example_ids=["q1"],
            beir_sample_sources=dict(root="/not-present", manifest_sha256="a" * 64),
            **other,
        )


def test_duplicate_keys_and_oversize_file_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"root":"a","root":"b"}')
    with pytest.raises(ValidationError):
        read_sources(path)
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValidationError, match="exceeds"):
        read_sources(path)


def test_actual_cli_forwards_explicit_sample_config(tmp_path):
    value = sample(tmp_path)
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(value))
    run = tmp_path / "run.json"
    run.write_text('{"release_id":"baseline.no_op"}')
    result = dict(records=[], failed_count=0, awaiting_clarification_count=0, scored_count=0)
    with patch.object(RunConfig, "evaluate", return_value=result) as evaluate:
        code = main(
            [
                "config",
                "evaluate",
                str(run),
                "--benchmark",
                "beir.fever",
                "--split",
                "test",
                "--case-id",
                "q1",
                "--beir-sample-config",
                str(path),
                "--accept-license-review",
                "--offline",
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )
    assert code == 0
    assert evaluate.call_args.kwargs["beir_sample_sources"] == value
    assert evaluate.call_args.kwargs["accept_license_review"] is True
