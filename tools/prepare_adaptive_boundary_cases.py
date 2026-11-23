"""Bounded real route-screening for controlled Adaptive-RAG boundary acceptance.

Freeze the bank/selection rule before inference, retain all screening costs, and
never use target-reader scores or test references to select a case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

from run_adaptive_controlled_acceptance import CELL, RELEASE, SPLIT, state_for
from run_static_acceptance import save

from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash
from workers.adaptive_rag.driver import RuntimeIdentity
from workers.adaptive_rag.runtime import LocalTransformersRouter

OFFSET, BANK_SIZE, TARGET_ROUTES = 150, 64, ("A", "C")


def select_boundaries(records):
    """First source-order case per required route; missing routes stay missing."""
    selected = {}
    for record in records:
        if record.get("status") != "router_response":
            continue
        logits = record["response"]["logits"]
        label = max(("A", "B", "C"), key=lambda key: logits[key])
        if label in TARGET_ROUTES and label not in selected:
            selected[label] = {
                "case_id": record["case_id"],
                "route": label,
                "screening_record_sha256": content_hash(record),
            }
    return [selected[label] for label in TARGET_ROUTES if label in selected]


def prepare(
    prepared_root,
    router_root,
    old_cohort,
    proof_path,
    proof_sha256,
    observed,
    observed_sha256,
    output,
):
    import torch

    if output.exists():
        raise ValueError("create-only screening; preserve all consumed screening attempts")
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != proof_sha256:
        raise ValueError("trusted old artifact proof mismatch")
    if hashlib.sha256(observed.read_bytes()).hexdigest() != observed_sha256:
        raise ValueError("frozen prior acceptance inventory mismatch")
    proof = json.loads(proof_path.read_text())
    observed_inventory = json.loads(observed.read_text())
    artifact_files = proof["router_artifact_files_exact"]
    actual = {}
    for name, expected in artifact_files.items():
        path = router_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("router asset closure missing/unsafe")
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        actual[name] = h.hexdigest()
        if actual[name] != expected:
            raise ValueError("old independent router artifact closure changed")
    if {
        p.relative_to(router_root).as_posix() for p in router_root.rglob("*") if p.is_file()
    } != set(actual):
        raise ValueError("router root has extra/unlisted files")
    adapter = load_production_benchmark_adapter(CELL, str(prepared_root))
    candidates = sorted(adapter.examples(SPLIT), key=lambda x: x.example_id)[
        OFFSET : OFFSET + BANK_SIZE
    ]
    if len(candidates) != BANK_SIZE:
        raise ValueError("fixed64 QASPER boundary bank unavailable")
    prior = {
        (r["release"], r["benchmark"], r["backbone"], e["case_id"])
        for r in observed_inventory["rows"]
        for e in r["evidence"]
    }
    models = json.loads((old_cohort / "manifest.json").read_text())["actual_model_bindings"]
    if any(
        (release, CELL, model, ex.example_id) in prior
        for model in models
        for release in ("baseline.no_op", RELEASE)
        for ex in candidates
    ):
        raise ValueError("bank intersects consumed reader cases; do not silently resample")
    # Reuse only the old independently verified router identity, not its routes.
    old_requests = [
        r
        for path in (old_cohort / "terminals").glob("*.json")
        for r in json.loads(path.read_text()).get("worker_requests", [])
    ]
    if not old_requests:
        raise ValueError("requires actual source-bound native router request identity")
    artifacts = old_requests[0]["payload"]["artifacts"]
    identity = RuntimeIdentity(
        **{
            f.name: artifacts[f.name]
            for f in fields(RuntimeIdentity)
            if f.name != "fixture_runtime"
        }
    )
    if identity.router_snapshot_sha256 != content_hash(actual):
        raise ValueError("screening source differs from independently verified native snapshot")
    output.mkdir(parents=True)
    (output / "claims").mkdir()
    (output / "terminals").mkdir()
    bank = [
        {
            "case_id": ex.example_id,
            "question": ex.visible_state().query,
            "question_sha256": content_hash(ex.visible_state().query),
            "question_only_state_hash": state_for(ex).state_hash,
        }
        for ex in candidates
    ]
    save(
        output / "manifest.json",
        {
            "schema": "agentinputlab.adaptive-route-boundary-bank.v1",
            "benchmark": CELL,
            "split": SPLIT,
            "source_offset": OFFSET,
            "bank_size": BANK_SIZE,
            "candidates": bank,
            "selection_rule": "first source-order case per A/C neural route, no replacement for missing route",
            "required_routes": TARGET_ROUTES,
            "seed": 17,
            "per_call_output_tokens": 1,
            "reader_quality_used_for_selection": False,
            "gold_or_training_labels_exposed_to_router": False,
            "reference_based_candidate_enrichment": False,
            "new_data_or_model_download_bytes": 0,
            "source_file_sha256": dict(adapter.prepared.file_sha256),
            "trusted_router_proof_sha256": proof_sha256,
            "router_artifact_files": actual,
            "prior_inventory_sha256": observed_sha256,
            "operator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "route-conditioned compatibility boundary sampling, not a benchmark quality estimate",
        },
    )
    save(
        output / "started.json",
        {
            "pid": os.getpid(),
            "uid": os.getuid(),
            "started": time.time(),
            "argv_sha256": hashlib.sha256(Path("/proc/self/cmdline").read_bytes()).hexdigest(),
            "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        },
    )
    torch.set_num_threads(8)
    router = LocalTransformersRouter(
        snapshot_root=router_root,
        identity=identity,
        weights_path=router_root / "model.safetensors",
        tokenizer_path=router_root / "tokenizer.json",
        device="cpu",
    )
    records = []
    for item in bank:
        save(
            output / "claims" / (item["case_id"] + ".json"), {"case": item, "started": time.time()}
        )
        record = {
            "case_id": item["case_id"],
            "question_sha256": item["question_sha256"],
            "seed": 17,
            "max_new_tokens": 1,
            "helper_role": "native_router_boundary_screen",
        }
        try:
            response = router.first_decoder_step(item["question"], seed=17)
            record.update(status="router_response", response=asdict(response))
        except Exception as exc:
            record.update(
                status="router_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                unknown_usage=True,
            )
        save(output / "terminals" / (item["case_id"] + ".json"), record)
        records.append(record)
    selected = select_boundaries(records)
    counts = Counter(
        max(("A", "B", "C"), key=lambda key: r["response"]["logits"][key])
        for r in records
        if r["status"] == "router_response"
    )
    selection = {
        "schema": "agentinputlab.adaptive-route-boundary-selection.v1",
        "benchmark": CELL,
        "split": SPLIT,
        "selected_cases": selected,
        "missing_routes": [
            route for route in TARGET_ROUTES if route not in {r["route"] for r in selected}
        ],
        "source_bank_manifest_sha256": hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
        "screening_attempts": len(records),
        "known_screening_responses": sum("response" in r for r in records),
        "screening_input_tokens": sum(
            r.get("response", {}).get("input_tokens", 0) for r in records
        ),
        "screening_output_tokens": sum(
            r.get("response", {}).get("output_tokens", 0) for r in records
        ),
        "unknown_usage_attempts": sum(r.get("unknown_usage", False) for r in records),
        "route_counts": dict(counts),
        "screening_cost_unpriced_not_free": True,
        "screening_is_extra_preparation_not_reader_budget_or_free_retry": True,
        "no_reader_model_calls": True,
        "publication_ready": False,
    }
    save(output / "selection.json", selection)
    print(json.dumps(selection), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "router", "old-cohort", "artifact-proof", "observed", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--artifact-proof-sha256", required=True)
    parser.add_argument("--observed-sha256", required=True)
    args = parser.parse_args()
    prepare(
        args.prepared_root,
        args.router,
        args.old_cohort,
        args.artifact_proof,
        args.artifact_proof_sha256,
        args.observed,
        args.observed_sha256,
        args.output,
    )


if __name__ == "__main__":
    main()
