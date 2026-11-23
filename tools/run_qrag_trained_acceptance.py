"""Real native-fitted Q-RAG preparation, followed by independent reader trials.

Eight fixed visible-input selections are shared by exact checksum across four
readers. Fit is never invoked here. Every new helper or reader attempt has a
create-only claim and retains its real failure and consumed cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ContextBlock, ExecutionBudget, InterventionContext
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.recipes import QRAGTrainedPolicyArtifact, QRAGTrainedPolicyIntervention
from agentinputlab.runner import InterventionRunner

RELEASE = "training.qrag.value_policy"
OFFSET = 226
BUDGET = dict(max_model_calls=8, max_output_tokens=4096, max_retrieval_calls=1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cases(config):
    adapters, examples, jobs = {}, {}, []
    for cell, split in QA_CELLS:
        adapter = load_production_benchmark_adapter(cell, config["prepared_root"])
        adapters[cell] = adapter
        values = sorted(adapter.examples(split), key=lambda e: e.example_id)[OFFSET : OFFSET + 2]
        if len(values) != 2:
            raise ValueError("fixed source cases unavailable; no outcome-based substitution")
        for index, example in enumerate(values, OFFSET):
            state = example.visible_state(budget=ExecutionBudget(**BUDGET))
            job = dict(
                benchmark=cell,
                split=split,
                case_id=example.example_id,
                source_index=index,
                state_hash=state.state_hash,
                query_sha256=content_hash(state.query),
                context_sha256=content_hash(state.context),
            )
            jobs.append(job)
            examples[cell, example.example_id] = (example, state)
    return adapters, examples, jobs


def prepare(config, output):
    if output.exists():
        raise ValueError("native preparation is create-only; no consumed-helper retry")
    import torch

    torch.set_num_threads(4)
    adapters, examples, jobs = cases(config)
    artifact = QRAGTrainedPolicyArtifact.load(config["trained_artifact"])
    if artifact.artifact_id != config["expected_artifact_id"]:
        raise ValueError("fitted artifact differs")
    fit_queries = set()
    plan = json.loads((artifact.root / "fit-plan.json").read_text())
    for spec in plan["split_audit"]["files"].values():
        path = Path(config["fit_protocol"]) / Path(spec["path"]).name
        if sha(path) != spec["sha256"]:
            raise ValueError("retained native train/development records differ")
        fit_queries.update(
            " ".join(r["question"].casefold().split()) for r in json.loads(path.read_text())
        )
    if any(" ".join(s.query.casefold().split()) in fit_queries for _, s in examples.values()):
        raise ValueError("fixed test input overlaps native fit/development input")
    output.mkdir()
    for name in ("claims", "preparations", "forward-captures"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        dict(
            release=RELEASE,
            jobs=jobs,
            trained_artifact_id=artifact.artifact_id,
            source_indices=[OFFSET, OFFSET + 1],
            dataset_files={c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
            config=config,
            seed=17,
            exact_fit_query_overlap=False,
            native_training_repeated=False,
            pretraining_data_independence_claimed=False,
            source_cases_fixed_before_outputs=True,
            automatic_retry=False,
            paper_results_changed=False,
        ),
    )
    start, cpu = time.monotonic(), time.process_time()
    runtime = artifact.runtime(**config["runtime"])
    native_torch, agent, env = runtime._load()
    assert native_torch is torch
    modules = {
        n: getattr(agent, n) for n in ("critic", "policy", "v_net_target", "action_embed_target")
    }
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in modules.values()):
        raise ValueError("requires native evaluation-only network states")
    save(
        output / "runtime-ready.json",
        dict(
            runtime=dict(runtime.fingerprint()),
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.monotonic() - start,
            new_optimizer_steps=0,
            new_teacher_calls=0,
            safe_weights_only=True,
            native_modules={n: len(m.state_dict()) for n, m in modules.items()},
        ),
    )
    from rl.bert_predictor import BertPredictor

    original = BertPredictor.forward
    events, active = [], {}

    @torch.compiler.disable
    def observed(module, *args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        event = dict(
            index=len(events),
            input_shape=list(ids.shape),
            completed=False,
            model_training=module.training,
            grad_enabled=torch.is_grad_enabled(),
        )
        events.append(event)
        value = original(module, *args, **kwargs)
        # Observation-only capture returns the exact native object unchanged.
        path = active["folder"] / f"forward-{event['index']:04d}.pt"
        torch.save(dict(input_ids=ids.detach().cpu(), output=value.detach().cpu()), path)
        event.update(
            completed=True,
            output_shape=list(value.shape),
            capture_path=str(path),
            capture_sha256=sha(path),
        )
        return value

    method, receipts = QRAGTrainedPolicyIntervention(runtime), []
    BertPredictor.forward = observed
    try:
        for job in jobs:
            example, state = examples[job["benchmark"], job["case_id"]]
            key = digest(job)
            save(
                output / "claims" / (key + ".json"),
                dict(job=job, physical_helper_attempt_once=True),
            )
            events.clear()
            active["folder"] = output / "forward-captures" / key
            active["folder"].mkdir()
            docs = tuple(
                Document(
                    doc_id=b.block_id or str(i),
                    text=b.text,
                    corpus_revision=adapters[job["benchmark"]].prepared.source_revision,
                )
                for i, b in enumerate(state.context)
            )
            retriever = InMemoryBM25Retriever(documents=docs)
            row = dict(
                job=job,
                trained_artifact_id=artifact.artifact_id,
                deployment_id=runtime.deployment_id,
                physical_helper_jobs=1,
                new_optimizer_steps=0,
                new_generative_model_calls=0,
                reference_exposed=False,
                automatic_retry=False,
            )
            start, cpu = time.monotonic(), time.process_time()
            try:
                record = InterventionRunner().run(
                    method, state, InterventionContext(seed=17, retriever=retriever)
                )
                materialized = record.patch.apply(state)
                row.update(
                    status="native_selected",
                    record=to_jsonable(record),
                    materialized_context=to_jsonable(materialized.context),
                    materialized_state_hash=materialized.state_hash,
                    retriever=dict(retriever.fingerprint()),
                )
            except Exception as exc:
                row.update(
                    status="failed_helper_terminal", error_type=type(exc).__name__, error=str(exc)
                )
            row.update(
                physical_embedding_forward_attempts=len(events),
                physical_embedding_forward_responses=sum(e["completed"] for e in events),
                native_forward_events=list(events),
                cpu_seconds=time.process_time() - cpu,
                wall_seconds=time.monotonic() - start,
            )
            path = output / "preparations" / (key + ".json")
            save(path, row)
            receipts.append(
                dict(
                    path=str(path),
                    sha256=sha(path),
                    job=job,
                    status=row["status"],
                    forwards=len(events),
                )
            )
    finally:
        BertPredictor.forward = original
    save(
        output / "complete.json",
        dict(
            manifest_sha256=sha(output / "manifest.json"),
            captures=receipts,
            benchmark_helper_jobs=8,
            physical_helper_jobs=8,
            physical_embedding_forward_attempts=sum(s["forwards"] for s in receipts),
            source_statuses=dict(Counter(s["status"] for s in receipts)),
            new_optimizer_steps=0,
            new_generative_model_calls=0,
            publication_ready=False,
        ),
    )
    print("8 fixed native trained-policy helper positions closed", flush=True)


def verified_capture(preparation, spec, original, artifact_id):
    path = Path(spec["path"])
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(preparation.resolve())
        or sha(path) != spec["sha256"]
    ):
        raise ValueError("native selection capture path/hash differs")
    row = json.loads(path.read_text())
    if (
        row["job"] != spec["job"]
        or row["job"]["state_hash"] != original.state_hash
        or row["trained_artifact_id"] != artifact_id
    ):
        raise ValueError("native capture source differs")
    if row["status"] != "native_selected":
        return row, None
    patch = row["record"]["patch"]
    if (
        patch["source_state_hash"] != original.state_hash
        or patch["release_id"] != RELEASE
        or patch["profile_kind"] != "controlled_adapter"
        or len(patch["operations"]) != 1
        or patch["operations"][0]["op"] != "add_context"
        or patch["metadata"]["trained_artifact_id"] != artifact_id
        or patch["metadata"]["deployment_id"] != row["deployment_id"]
    ):
        raise ValueError("requires the actual typed trained-policy patch")
    added = tuple(ContextBlock(**b) for b in patch["operations"][0]["value"])
    materialized = replace(original, context=original.context + added)
    if (
        materialized.state_hash != row["materialized_state_hash"]
        or to_jsonable(materialized.context) != row["materialized_context"]
    ):
        raise ValueError("native selected context materialization differs")
    return row, materialized


def run(config, output):
    if output.exists() or len(config["models"]) != 1:
        raise ValueError("one model/create-only reader lane required")
    preparation = Path(config["preparation_root"])
    if sha(preparation / "complete.json") != config["preparation_complete_sha256"]:
        raise ValueError("native preparation closure differs")
    complete = json.loads((preparation / "complete.json").read_text())
    manifest = json.loads((preparation / "manifest.json").read_text())
    adapters, examples, jobs = cases(config)
    if (
        complete["manifest_sha256"] != sha(preparation / "manifest.json")
        or jobs != manifest["jobs"]
        or complete["physical_helper_jobs"] != 8
        or {c: dict(a.prepared.file_sha256) for c, a in adapters.items()}
        != manifest["dataset_files"]
    ):
        raise ValueError("frozen source/data selection differs")
    model = config["models"][0]
    binding = verify_services([model])[model["id"]]
    output.mkdir()
    for name in ("claims", "terminals"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        dict(
            release=RELEASE,
            model=model,
            model_binding=binding,
            expected_positions=8,
            preparation_complete_sha256=sha(preparation / "complete.json"),
            trained_artifact_id=manifest["trained_artifact_id"],
            cases=jobs,
            seed=17,
            budget=BUDGET,
            per_call_tokens=512,
            new_helper_jobs=0,
            automatic_retry=False,
        ),
    )
    specs = {(s["job"]["benchmark"], s["job"]["case_id"]): s for s in complete["captures"]}
    if len(specs) != 8:
        raise ValueError("requires eight source-bound helper receipts")

    def execute(job):
        example, original = examples[job["benchmark"], job["case_id"]]
        spec = specs[job["benchmark"], job["case_id"]]
        capture, materialized = verified_capture(
            preparation, spec, original, manifest["trained_artifact_id"]
        )
        identifier = digest(
            dict(
                model=model["id"],
                case=job,
                artifact=manifest["trained_artifact_id"],
                release=RELEASE,
            )
        )
        save(
            output / "claims" / (identifier + ".json"),
            dict(model=model, case=job, scientific_attempt_once=True),
        )
        calls, row = (
            [],
            dict(
                model=model,
                case=job,
                release=RELEASE,
                helper_capture_sha256=spec["sha256"],
                helper_capture_path=spec["path"],
                trained_artifact_id=manifest["trained_artifact_id"],
                physical_new_helper_jobs=0,
                shared_actual_native_helper_not_synthetic=True,
                automatic_retry=False,
            ),
        )
        start = time.monotonic()
        if materialized is None:
            row.update(
                status="failed_helper_unreached_reader",
                error_type=capture["error_type"],
                error=capture["error"],
            )
        else:
            try:
                backend = RecordingBackend(
                    calls=calls,
                    model_id=model["id"],
                    base_url=model["base_url"],
                    extra_body=model["extra_body"],
                    transport=local_transport,
                    timeout_seconds=300,
                    deployment_identity=dict(
                        revision=digest(binding["model_json_hashes"]), server="vllm"
                    ),
                )
                predictor = ProductionPredictorBuilder().build(
                    StaticPredictorBuildRequest(
                        benchmark_id=job["benchmark"],
                        split=job["split"],
                        model_slot=model["id"],
                        seed=17,
                        builder_parameters=dict(
                            schema_version=1,
                            task="qa",
                            qa=dict(output_format="structured_json", max_tokens=512, use_seed=True),
                        ),
                        runtime=StaticRuntimeComponents(backend=backend),
                    )
                )
                prediction = predictor.predict(materialized)
                row.update(
                    status="scored_terminal",
                    prediction=to_jsonable(prediction),
                    metrics=dict(adapters[job["benchmark"]].score(example, prediction.prediction)),
                )
            except Exception as exc:
                row.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        if len(calls) > 1:
            raise ValueError("unexpected repeated reader generation")
        row.update(
            calls=calls,
            physical_new_generator_attempts=len(calls),
            wall_seconds=time.monotonic() - start,
            original_state_hash=original.state_hash,
            materialized_state_hash=materialized.state_hash if materialized else None,
        )
        save(output / "terminals" / (identifier + ".json"), row)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(execute, jobs))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    if len(rows) != 8:
        raise ValueError("fixed endpoint denominator is incomplete")
    save(
        output / "complete.json",
        dict(
            terminal_positions=8,
            expected_positions=8,
            statuses=dict(Counter(r["status"] for r in rows)),
            new_generator_attempts=sum(len(r["calls"]) for r in rows),
            new_helper_jobs=0,
            preserved_failed_denominators=True,
            automatic_retry=False,
            publication_ready=False,
        ),
    )
    save(output / "post-service-identities.json", verify_services([model]))
    print(model["id"], "8 trained-policy reader terminals closed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    (prepare if args.prepare else run)(config, args.output)
