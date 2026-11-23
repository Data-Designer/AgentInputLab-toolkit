"""Explicit source-mounted bindings used by installed-package real acceptance."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import resources

from agentinputlab.api import create_intervention
from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker

EXTERNAL_REWRITERS = (
    "rewrite.reformer.repo_native_r_only",
    "rewrite.infocqr.common_support_rw_zsl",
)
REFORMER_RELEASES = (
    "rewrite.reformer.repo_native_r_only",
    "rewrite.reformer.paper_spec_q_concat_r",
    "rewrite.reformer.controlled_r_only",
    "rewrite.reformer.controlled_q_concat_r",
)
ASK_OR_ASSUME = "clarify.ask_or_assume.intent_monitor_component"


def bind_acon(release, *, upstream, backend, model, client_wrapper=None):
    """Bind real target, optimizer, native tokenizer and exact ACON prompts."""
    import importlib.metadata
    import os
    from types import SimpleNamespace

    import tiktoken

    from agentinputlab.algorithms.curate.acon.common import ACONArtifactIdentity
    from agentinputlab.contracts.serialization import content_hash
    from workers.acon.driver import OpenAICompatibleLLM, _tokenizer_fingerprint_hash

    package = resources.files("workers").joinpath("acon")
    policy = json.loads((package / "compatibility_manifest.json").read_text())
    profile_name = "smolagents.history.v1" if release == "curate.acon.history_runtime" else "smolagents.observation.v1"
    profile = policy["prompt_profiles"][profile_name]
    # This is the same exact native tokenizer constructed by pinned ACON base.py.
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    tok_hash = _tokenizer_fingerprint_hash(SimpleNamespace(tokenizer=tokenizer))
    tok_revision = (
        "tiktoken:" + importlib.metadata.version("tiktoken") + ":" + tokenizer.name + ":" + tok_hash
    )
    target = dict(backend.fingerprint())
    revision = target["model_revision"]
    optimizer = OpenAICompatibleLLM(
        base_url=model["base_url"],
        api_key="",
        model=model["id"],
        seed=17,
        default_temperature=0.0,
        timeout_seconds=180,
        max_tokens=512,
        extra_body=model.get("extra_body", {}),
    )
    artifacts = ACONArtifactIdentity(
        optimizer_model_revision=revision,
        target_model_revision=revision,
        tokenizer_revision=tok_revision,
        prompt_artifact_hash=profile["prompt_artifact_hash"],
        guideline_artifact_hash=profile["guideline_artifact_hash"],
        optimizer_deployment_fingerprint_hash=content_hash(optimizer.fingerprint()),
        target_backend_fingerprint_hash=content_hash(target),
        tokenizer_fingerprint_hash=tok_hash,
    )
    worker = JsonLineSubprocessWorker(
        command=(sys.executable, "-m", "workers.acon.driver"),
        worker_id="toolkit-native-acon",
        identity={
            "upstream_commit": policy["upstream"]["commit"],
            "prompt_profile": profile_name,
            "driver_sha256": hashlib.sha256((package / "driver.py").read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                (package / "compatibility_manifest.json").read_bytes()
            ).hexdigest(),
            "target_backend": target,
            "optimizer": dict(optimizer.fingerprint()),
            "tokenizer_revision": tok_revision,
        },
        timeout_seconds=400,
        environment={
            "PATH": os.environ["PATH"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "AGENTINPUTLAB_ACON_CHECKOUT": str(upstream / "upstream-paper/acon"),
            "AGENTINPUTLAB_ACON_OPENAI_BASE_URL": model["base_url"],
            "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL": model["id"],
            "AGENTINPUTLAB_ACON_OPTIMIZER_MODEL_REVISION": revision,
            "AGENTINPUTLAB_ACON_TARGET_MODEL_REVISION": revision,
            "AGENTINPUTLAB_ACON_TOKENIZER_REVISION": tok_revision,
            "AGENTINPUTLAB_ACON_MAX_TOKENS": "512",
            "AGENTINPUTLAB_ACON_TIMEOUT_SECONDS": "180",
            "AGENTINPUTLAB_ACON_EXTRA_BODY_JSON": json.dumps(model.get("extra_body", {})),
        },
    )
    if client_wrapper is not None:
        worker = client_wrapper(worker)
    return create_intervention(release, artifacts=artifacts), worker


def bind_ask_or_assume(*, upstream, backend, model):
    """Exercise the source-bound active intent branch, not turn-zero skip."""
    from agentinputlab.algorithms.clarify.ask_or_assume.monitor import AskOrAssumeControls

    package = resources.files("workers").joinpath("ask_or_assume")
    role = {
        "model": model["id"],
        "fingerprint": dict(backend.fingerprint()),
        "max_output_tokens": 512,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }
    config = {
        "base_url": model["base_url"],
        "api_key_env": "AIL_TOOLKIT_WORKER_LOCAL_KEY",
        "timeout_seconds": 180,
        "roles": {"intent": role, "coding": role},
    }
    worker = JsonLineSubprocessWorker(
        command=(
            sys.executable,
            str(package / "driver.py"),
            "--checkout",
            str(upstream / "upstream-paper/ask-or-assume-728464a"),
            "--backend-config-json",
            json.dumps(config),
        ),
        worker_id="toolkit-source-mounted-ask-or-assume",
        identity={
            "source_manifest_sha256": hashlib.sha256(
                (package / "compatibility_manifest.json").read_bytes()
            ).hexdigest(),
            "driver_sha256": hashlib.sha256((package / "driver.py").read_bytes()).hexdigest(),
            "backend": dict(backend.fingerprint()),
            "pricing": "local_unpriced_not_zero_compute_cost",
            "native_backend_defaults": True,
        },
        timeout_seconds=400,
        environment={
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "AIL_TOOLKIT_WORKER_LOCAL_KEY": "local-service-only",
        },
    )
    return create_intervention(
        ASK_OR_ASSUME,
        controls=AskOrAssumeControls(
            skip_first_turn=False, max_model_calls=2, max_tokens=2 * (28000 + 512)
        ),
    ), worker


def bind_external_rewriter(
    release, *, upstream, backend, model, dataset_revision, benchmark, manual_pattern_id=""
):
    """Mount only verified public-source workers; never paper result artifacts."""
    from agentinputlab.algorithms.reformulate.infocqr import InfoCQRExternalControls
    from agentinputlab.algorithms.reformulate.reformer import ReFormeRControls

    if release not in (*REFORMER_RELEASES, EXTERNAL_REWRITERS[1]):
        raise ValueError("no source-mounted rewriter binding for this exact release")
    manual = release in REFORMER_RELEASES[2:]
    if manual != bool(manual_pattern_id):
        raise ValueError("only manual ReFormeR profiles require a fixed pattern ID")
    kind = "reformer" if release in REFORMER_RELEASES else "infocqr"
    package = resources.files("workers").joinpath(kind)
    manifest = package / "compatibility_manifest.json"
    driver = package / "driver.py"
    policy = json.loads(manifest.read_text())
    checkout = (
        upstream / "upstream-paper" / ("ReFormeR-72e5245" if kind == "reformer" else "InfoCQR")
    )
    backend_config = {
        "base_url": model["base_url"],
        "api_key_env": "AIL_TOOLKIT_WORKER_LOCAL_KEY",
        "model": model["id"],
        "fingerprint": dict(backend.fingerprint()),
        "extra_body": model.get("extra_body", {}),
        "timeout_seconds": 180,
        "max_input_tokens_per_call": 28000,
        "max_output_tokens": 512,
        "temperature": 0.0,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }
    worker = JsonLineSubprocessWorker(
        command=(
            sys.executable,
            str(driver),
            "--checkout",
            str(checkout),
            "--compatibility-manifest" if kind == "reformer" else "--manifest",
            str(manifest),
            "--backend-config-json",
            json.dumps(backend_config),
            "--acknowledge-noassertion-license",
        ),
        worker_id="toolkit-source-mounted-" + kind,
        identity={
            "upstream_commit": policy["upstream"]["commit"],
            "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "backend": dict(backend.fingerprint()),
            "pricing": "local_unpriced_usage_not_zero_compute_cost",
        },
        timeout_seconds=400,
        environment={
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "AIL_TOOLKIT_WORKER_LOCAL_KEY": "local-service-only",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    if kind == "reformer":
        calls = 1 if manual else 2
        method = create_intervention(
            release,
            source_license_acknowledged=True,
            pattern_artifact_hash=policy["source_files"][
                "reformer/patterns/extracted_patterns.json"
            ],
            # This worker's max_tokens covers INPUT plus OUTPUT, unlike the
            # shared execution budget's max_output_tokens. Per-call output
            # remains 512 (one manual call or two automatic calls).
            controls=ReFormeRControls(max_model_calls=calls, max_tokens=calls * (28000 + 512)),
            **({"manual_pattern_id": manual_pattern_id} if manual else {}),
        )
    else:
        method = create_intervention(
            release,
            prompt_profile="rw_zsl",
            source_license_acknowledged=True,
            dataset_id=benchmark,
            dataset_revision=dataset_revision,
            split="test",
            retrieval_corpus_id=benchmark,
            controls=InfoCQRExternalControls(max_model_calls=1, max_tokens=28000 + 512),
        )
    return method, worker
