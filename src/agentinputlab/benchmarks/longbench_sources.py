"""Shared, byte-locked official LongBench v1 assets (not vendored data/code)."""
from agentinputlab.artifacts import AcquisitionPolicy, ArtifactKind, ArtifactManifest

from .longbench import (
    LONG_BENCH_DATA_REVISION,
    LONG_BENCH_DATA_SHA256,
    LONG_BENCH_PROMPTS_SHA256,
    LONG_BENCH_SCORER_SHA256,
    LONG_BENCH_SOURCE_REVISION,
)

ALLOWED_HOSTS = ("huggingface.co", "cas-bridge.xethub.hf.co", "cdn-lfs.huggingface.co",
                 "us.aws.cdn.hf.co", "raw.githubusercontent.com")


def manifests():
    source = f"https://raw.githubusercontent.com/THUDM/LongBench/{LONG_BENCH_SOURCE_REVISION}/LongBench"
    specs = {
        "data_zip": (LONG_BENCH_DATA_SHA256, 113932529,
            f"https://huggingface.co/datasets/zai-org/LongBench/resolve/{LONG_BENCH_DATA_REVISION}/data.zip"),
        "scorer_path": (LONG_BENCH_SCORER_SHA256, 5190, source + "/metrics.py"),
        "prompts_path": (LONG_BENCH_PROMPTS_SHA256, 5437, source + "/config/dataset2prompt.json"),
    }
    return {key: ArtifactManifest(artifact_id="longbench.v1." + key, kind=ArtifactKind.DATASET,
        revision=digest, sha256=digest, size_bytes=size, source_uri=url,
        acquisition=AcquisitionPolicy.DOWNLOAD, license_id="NOASSERTION")
        for key, (digest, size, url) in specs.items()}
