"""Freeze disjoint real QA inputs and execute each independent trained router once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import tarfile
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from run_static_acceptance import QA_CELLS, save

from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash
from agentinputlab.recipes import AdaptiveRAGTrainedRouterArtifact
from agentinputlab.recipes.adaptive_rag import validate_adaptive_training_checkout
from workers.adaptive_rag.runtime import LocalTransformersRouter


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def query_key(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def choose_cases(examples, excluded, *, start_index=218, count=2):
    """Selection depends on source order and input overlap, never predictions."""
    selected, skipped = [], []
    for index, example in enumerate(sorted(examples, key=lambda x: x.example_id)):
        if index < start_index:
            continue
        if query_key(example.visible_state().query) in excluded:
            skipped.append(
                dict(index=index, case_id=example.example_id, reason="training_query_overlap")
            )
            continue
        selected.append((index, example))
        if len(selected) == count:
            return selected, skipped
    raise ValueError("insufficient disjoint real cases in the declared source suffix")


def warm_start_records(source, metadata):
    archive = source / "data.tar.gz"
    data = archive.read_bytes()
    expected = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD:data.tar.gz"], text=True
    ).strip()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if (
        blob != expected
        or hashlib.sha256(data).hexdigest() != metadata["official_data_archive_sha256"]
    ):
        raise ValueError("warm-start original source archive differs")
    records, bindings = {}, {}
    with tarfile.open(archive) as tar:
        members = {m.name.removeprefix("./"): m for m in tar.getmembers()}
        for split, field in (("train", "training"), ("dev", "validation")):
            name = metadata[field + "_data_relative_path"]
            member = members[name]
            if not member.isfile() or member.size > 16 * 1024 * 1024:
                raise ValueError("warm-start source member must be bounded regular data")
            payload = tar.extractfile(member).read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != metadata[field + "_data_sha256"]:
                raise ValueError("warm-start original train/dev member hash differs")
            records[split] = json.loads(payload)
            if len(records[split]) != metadata[field + "_record_count"]:
                raise ValueError("warm-start original train/dev row count differs")
            bindings[split] = dict(member=name, sha256=digest, rows=len(records[split]))
    return records, dict(archive_sha256=sha(archive), git_blob_sha1=blob, members=bindings)


def run(config, output):
    if output.exists():
        raise ValueError(
            "create-only preparation; inspect existing neural attempts before continuing"
        )
    artifact = AdaptiveRAGTrainedRouterArtifact.load(config["trained_artifact"])
    if artifact.artifact_id != config["trained_artifact_id"]:
        raise ValueError("trained artifact identity differs")
    source, source_binding = validate_adaptive_training_checkout(config["upstream"])
    warm_path = Path(config["warm_start_manifest"])
    if sha(warm_path) != config["warm_start_manifest_sha256"]:
        raise ValueError("warm-start provenance differs")
    warm, warm_binding = warm_start_records(source, json.loads(warm_path.read_text()))
    plan = json.loads((artifact.root / "fit-plan.json").read_text())
    fit_rows, fit_binding = {}, {}
    for split, filename in (("train", "train.json"), ("dev", "validation.json")):
        path = Path(config["native_data_root"]) / filename
        if sha(path) != plan["split_audit"]["files"][split]["sha256"]:
            raise ValueError("actual native fit data differs")
        fit_rows[split] = json.loads(path.read_text())
        fit_binding[split] = dict(path=str(path), sha256=sha(path), rows=len(fit_rows[split]))
    excluded = {
        query_key(row["question"])
        for group in (*warm.values(), *fit_rows.values())
        for row in group
    }
    cases, skipped, datasets = [], {}, {}
    for cell, split in QA_CELLS:
        adapter = load_production_benchmark_adapter(cell, config["prepared_root"])
        selected, excluded_cases = choose_cases(adapter.examples(split), excluded)
        skipped[cell] = excluded_cases
        datasets[cell] = dict(adapter.prepared.file_sha256)
        for index, example in selected:
            visible = example.visible_state()
            cases.append(
                dict(
                    benchmark=cell,
                    split=split,
                    case_id=example.example_id,
                    source_index=index,
                    question=visible.query,
                    question_sha256=content_hash(visible.query),
                    visible_state_hash=visible.state_hash,
                )
            )
    if len(cases) != 8 or any(query_key(row["question"]) in excluded for row in cases):
        raise ValueError("all selected queries must exclude native and warm-start train/dev")
    output.mkdir(parents=True)
    (output / "captures").mkdir()
    bindings = artifact.router_bindings()
    controls = artifact.router_configuration()
    import workers.adaptive_rag.runtime as runtime

    manifest = dict(
        schema="agentinputlab.adaptive-trained-router-preparation.v1",
        cases=cases,
        selection_rule="first two source-sorted cases at/after218 excluding native and warm-start train/dev; no outcomes",
        skipped_input_overlap=skipped,
        excluded_query_set_sha256=content_hash(sorted(excluded)),
        excluded_unique_queries=len(excluded),
        warm_start=warm_binding,
        native_fit=fit_binding,
        datasets=datasets,
        trained_artifact_id=artifact.artifact_id,
        router_bindings=bindings,
        router_configuration=controls,
        native_source=source_binding,
        dependency_versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "sentencepiece")
        },
        runtime_sha256=sha(Path(runtime.__file__)),
        operator_sha256=sha(Path(__file__)),
        expected_router_attempts=8,
        seed=17,
        router_output_tokens_per_attempt=1,
        target_backbone_outputs_not_used_for_selection=True,
        gold_not_exposed_to_router=True,
        training_repeated=False,
        no_target_calls_in_preparation=True,
        publication_ready=False,
    )
    save(output / "manifest.json", manifest)
    # The standalone helper has only real router identity fields. No fictitious
    # generator/retriever assets or unused dummy component hashes are supplied.
    load_start, load_cpu = time.monotonic(), time.process_time()
    native = LocalTransformersRouter(
        snapshot_root=artifact.checkpoint,
        identity=SimpleNamespace(fixture_runtime=False, **bindings),
        weights_path=artifact.checkpoint / "model.safetensors",
        tokenizer_path=artifact.checkpoint / "tokenizer.json",
        device="cpu",
        **controls,
    )
    load_wall, load_cpu_seconds = time.monotonic() - load_start, time.process_time() - load_cpu
    results = []
    for case in cases:
        key = content_hash(dict(benchmark=case["benchmark"], case_id=case["case_id"]))
        start, cpu = time.monotonic(), time.process_time()
        row = dict(
            case=case,
            seed=17,
            trained_artifact_id=artifact.artifact_id,
            router_bindings=bindings,
            router_configuration=controls,
            physical_attempts=1,
        )
        try:
            result = native.first_decoder_step(case["question"], seed=17)
            if (
                set(result.logits) != {"A", "B", "C"}
                or result.output_tokens != 1
                or not all(math.isfinite(v) for v in result.logits.values())
            ):
                raise ValueError("native router returned invalid first-step logits/usage")
            row.update(status="native_router_returned", result=asdict(result))
        except Exception as exc:
            row.update(status="failed_native_router", error_type=type(exc).__name__, error=str(exc))
        row.update(
            wall_seconds=time.monotonic() - start,
            cpu_seconds=time.process_time() - cpu,
            compute_separate_unpriced_not_free=True,
        )
        row["capture_id"] = content_hash(row)
        path = output / "captures" / (key + ".json")
        save(path, row)
        results.append(
            dict(
                path=str(path), sha256=sha(path), capture_id=row["capture_id"], status=row["status"]
            )
        )
        print(json.dumps(dict(case=case["case_id"], status=row["status"])), flush=True)
    save(
        output / "complete.json",
        dict(
            manifest_sha256=sha(output / "manifest.json"),
            captures=results,
            actual_physical_router_attempts=8,
            target_calls=0,
            automatic_retry=False,
            helper_loading_wall_seconds=load_wall,
            helper_loading_cpu_seconds=load_cpu_seconds,
            memory_and_accelerator_compute_not_fully_metered=True,
            original_failed_captures_retained=True,
            source_and_neural_costs_not_free=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
