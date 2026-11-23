import tempfile
from pathlib import Path

import pytest

from agentinputlab.benchmarks import SourceVerifiedBFCLV3ASTBridge
from agentinputlab.benchmarks.bfcl_native import SourceVerifiedBFCLV3Result
from agentinputlab.contracts import ArtifactError, ValidationError


def test_unverified_archive_is_rejected_before_source_execution():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "untrusted.tar.gz"
        p.write_bytes(b"not the official source")
        with pytest.raises(ArtifactError, match="checksum"):
            SourceVerifiedBFCLV3ASTBridge(source_archive=p, model_id="configured-model")


def test_actual_model_identity_is_required():
    with pytest.raises(ValidationError, match="actual model"):
        SourceVerifiedBFCLV3ASTBridge(source_archive="unused", model_id="")


def test_case_level_source_verification_is_not_full_leaderboard_attestation():
    result = SourceVerifiedBFCLV3Result(0.0, diagnostic={"error_type": "wrong_count"})
    assert result.metrics == {"ast_accuracy": 0.0}
    assert result.provenance_status == "source_verified"
    assert not result.official_runtime_verified
