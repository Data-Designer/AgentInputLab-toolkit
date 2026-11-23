"""Released SmartSearch helper loop, shared preparations, four real target models.

The original 8192-output-token frozen loop is retained, with a separately
declared native-budget cohort. A tiny union of fixed visible QA passages is the
explicit corpus; neither full Wikipedia parity nor a 4096-token run is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.recipes import SmartSearchArtifactIdentity, SmartSearchCommonSupportQueries
from agentinputlab.recipes.smartsearch import (
    SMARTSEARCH_CHECKPOINT_REVISION,
    SMARTSEARCH_MODEL_REPOSITORY,
    SMARTSEARCH_TOOL_SCHEMA,
    SMARTSEARCH_UPSTREAM_COMMIT,
)
from agentinputlab.runner import InterventionRunner
from workers.smartsearch.driver import (
    DEFAULT_MANIFEST,
    DriverError,
    LocalSnapshotOpenAIModel,
    SmartSearchDriver,
    _error_response,
)

RELEASE = "rewrite.smartsearch.common_support_queries"
BUDGET = {
    "max_model_calls": 11,
    "max_output_tokens": 82432,
    "max_retrieval_calls": 10,
    "max_context_tokens": 262144,
}


class RecordedHTTP:
    def __init__(self, delegate, directory):
        self.delegate, self.directory, self.count = delegate, directory, 0
        self.directory.mkdir()

    def post(self, endpoint, payload):
        self.count += 1
        root = self.directory / f"{self.count:04d}"
        root.mkdir()
        save(
            root / "request.json",
            {"endpoint": endpoint, "payload": payload, "started": time.time()},
        )
        try:
            response = self.delegate.post(endpoint, payload)
        except Exception as exc:
            save(
                root / "response.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "consumption": "unknown_if_generation_was_dispatched"
                    if endpoint.endswith("/completions")
                    else "tokenization_only",
                },
            )
            raise
        save(root / "response.json", response)
        return response


class LocalRetriever:
    def __init__(self, retriever, identity):
        self.retriever, self.identity = retriever, identity

    def fingerprint(self):
        return self.identity

    def search(self, query, *, top_n):
        return [
            {"id": r.document.doc_id, "contents": r.document.text, "score": r.score}
            for r in self.retriever.retrieve(query, top_k=top_n)
        ]


class DriverClient:
    worker_id = "smartsearch-pinned-inprocess-acceptance-worker"

    def __init__(self, driver):
        self.driver, self.current = driver, None

    def fingerprint(self):
        return {
            "worker_id": self.worker_id,
            "source_commit": SMARTSEARCH_UPSTREAM_COMMIT,
            "compatibility_sha256": self.driver.manifest_sha256,
            "runtime_sha256": content_hash(self.driver.runtime),
        }

    def call(self, payload):
        save(self.current / "worker-request.json", payload)
        try:
            response = self.driver.run(to_jsonable(payload))
        except DriverError as exc:
            response = _error_response(exc, to_jsonable(payload), self.driver.last_usage)
        save(self.current / "worker-response.json", response)
        return response


def execute(config, root):
    stop = threading.Event()

    def keepalive():
        while not stop.wait(30):
            (root.parent / "keep-services-alive").touch()

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()
    try:
        return execute_jobs(config, root)
    finally:
        stop.set()
        thread.join()
        Path(config["helper_service_directory"], "stop").touch(exist_ok=False)


def execute_jobs(config, root):
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    selected = {
        c: tuple(sorted(adapters[c].examples(s), key=lambda x: x.example_id)[28:30])
        for c, s in QA_CELLS
    }
    documents = []
    for c, values in selected.items():
        for example in values:
            for i, block in enumerate(
                example.visible_state(budget=ExecutionBudget(**BUDGET)).context
            ):
                documents.append(
                    Document(
                        doc_id=f"{c}:{example.example_id}:{i}",
                        text=block.text,
                        corpus_revision=adapters[c].prepared.source_revision,
                    )
                )
    bm25 = InMemoryBM25Retriever(
        documents=tuple(documents), retriever_id="fixed_visible_acceptance_pool"
    )
    corpus_hash = content_hash([to_jsonable(d) for d in documents])
    snapshot_path = Path(config["snapshot_manifest"])
    snapshot = json.loads(snapshot_path.read_text())
    helper_binding = json.loads(
        (Path(config["helper_service_directory"]) / "started.json").read_text()
    )
    admission = content_hash(
        {
            "max_model_len": 32768,
            "max_new_tokens": 8192,
            "implementation": "native_vllm_tokenizer_admission.v1",
            "helper_binding": helper_binding,
        }
    )
    model_identity = {
        "checkpoint_repository": SMARTSEARCH_MODEL_REPOSITORY,
        "checkpoint_revision": SMARTSEARCH_CHECKPOINT_REVISION,
        "tokenizer_revision": SMARTSEARCH_CHECKPOINT_REVISION,
        "model_snapshot_tree_sha256": snapshot["snapshot_tree_sha256"],
        "served_model_name": "smartsearch-3b-pinned",
        "context_admission_fingerprint_sha256": admission,
    }
    retrieval_config = content_hash(
        {"top_n": 5, "corpus_sha256": corpus_hash, "subset": "fixed_visible_pool_only"}
    )
    rid = {
        "transport": "acceptance_local_visible_pool_bm25.v1",
        "retriever_revision": "toolkit_bm25_visible_pool.v1",
        "corpus_sha256": corpus_hash,
        "index_sha256": content_hash(bm25.fingerprint()),
        "retrieval_config_sha256": retrieval_config,
    }
    model_fp = {
        "transport": "openai_completions_local_snapshot",
        "base_url": "http://127.0.0.1:8864",
        "tokenizer_base_url": "http://127.0.0.1:8864",
        "identity": model_identity,
    }
    artifacts = SmartSearchArtifactIdentity(
        model_snapshot_manifest_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        model_snapshot_tree_sha256=snapshot["snapshot_tree_sha256"],
        tokenizer_revision=SMARTSEARCH_CHECKPOINT_REVISION,
        served_model_name="smartsearch-3b-pinned",
        model_deployment_fingerprint_sha256=content_hash(model_fp),
        context_admission_fingerprint_sha256=admission,
        retriever_revision=rid["retriever_revision"],
        retriever_deployment_fingerprint_sha256=content_hash(rid),
        corpus_sha256=corpus_hash,
        index_sha256=rid["index_sha256"],
        retrieval_config_sha256=retrieval_config,
        compatibility_manifest_sha256=hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest(),
        model_transport="openai_completions_local_snapshot",
    )
    runtime = {
        **to_jsonable(artifacts.as_mapping()),
        "artifacts": to_jsonable(artifacts.as_mapping()),
        "model_snapshot_manifest_path": str(snapshot_path),
        "local_model_root": config["model_root"],
        "model_base_url": "http://127.0.0.1:8864",
        "tokenizer_base_url": "http://127.0.0.1:8864",
        "model_timeout_seconds": 300,
        "tool_schema": to_jsonable(SMARTSEARCH_TOOL_SCHEMA),
    }
    (root / "preparation").mkdir()
    (root / "claims").mkdir()
    (root / "terminals").mkdir()
    save(
        root / "manifest.json",
        {
            "models": config["models"],
            "expected_preparations": 8,
            "planned_downstream": 32 if config.get("reuse_baseline_root") else 64,
            "evaluation": {c: [e.example_id for e in es] for c, es in selected.items()},
            "budget": BUDGET,
            "target_per_call_output_cap": 512,
            "frozen_helper_per_call_cap": 8192,
            "helper_seed_applied": False,
            "target_seed": 17,
            "corpus_sha256": corpus_hash,
            "corpus_document_count": len(documents),
            "corpus_scope": "union_of_selected_visible_qa_contexts_not_full_benchmark_corpus",
            "artifacts": to_jsonable(artifacts),
            "runtime": runtime,
            "helper_binding": helper_binding,
            "data_bindings": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
            "shared_native_helper_cost_counted_once": True,
            "automatic_retry": False,
            "paper_results_changed": False,
            "wheel_sha256": hashlib.sha256(
                Path(config["installed_wheel"]).read_bytes()
            ).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    jobs = []
    for target_model in config["models"]:
        for c, s in QA_CELLS:
            for example in selected[c]:
                for release in (
                    (RELEASE,) if config.get("reuse_baseline_root") else ("baseline.no_op", RELEASE)
                ):
                    j = {
                        "model": target_model,
                        "benchmark": c,
                        "split": s,
                        "case_id": example.example_id,
                        "release": release,
                        "budget": BUDGET,
                        "seed": 17,
                        "per_call_output_cap": 512,
                    }
                    j["id"] = digest(j)
                    jobs.append(j)
    save(root / "jobs.json", {"jobs": jobs, "selected_before_any_outcome": True})
    if config.get("reuse_baseline_root"):
        old_root = Path(config["reuse_baseline_root"])
        if not (old_root / "complete.json").exists():
            raise ValueError("old baseline cohort is not terminal; do not redispatch")
        old_manifest = json.loads((old_root / "manifest.json").read_text())
        if (
            old_manifest["wheel_sha256"]
            != hashlib.sha256(Path(config["installed_wheel"]).read_bytes()).hexdigest()
        ):
            raise ValueError("baseline package bytes differ")
        prior = {}
        for path in old_root.glob("terminals/*.json"):
            row = json.loads(path.read_text())
            j = row["job"]
            if j["release"] != "baseline.no_op":
                continue
            key = (j["model"]["id"], j["benchmark"], j["case_id"])
            if key in prior:
                raise ValueError("ambiguous baseline; do not choose favourable attempt")
            prior[key] = (path, row)
        reuses = []
        for job in jobs:
            path, row = prior[(job["model"]["id"], job["benchmark"], job["case_id"])]
            old = row["job"]
            state = (
                adapters[job["benchmark"]]
                .example(job["split"], job["case_id"])
                .visible_state(budget=ExecutionBudget(**BUDGET))
            )
            binding = json.loads(
                (Path(job["model"]["service_directory"]) / "started.json").read_text()
            )
            if not (
                old["model"] == job["model"]
                and old["split"] == job["split"]
                and old["budget"] == BUDGET
                and old["seed"] == 17
                and old["per_call_output_cap"] == 512
                and row["input_state_hash"] == state.state_hash
                and row["model_binding"] == binding
                and row["status"] in {"scored_terminal", "failed_terminal"}
                and len(row["calls"]) == 1
                and row["calls"][0]["request"]["seed"] == 17
                and row["calls"][0]["request"]["max_tokens"] == 512
            ):
                raise ValueError("saved baseline execution identity mismatch")
            reuses.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": row["status"],
                    "new_baseline_attempts": 0,
                }
            )
        save(root / "baseline-reuses.json", {"reuses": reuses, "new_baseline_attempts": 0})
    model = LocalSnapshotOpenAIModel(runtime)
    model.http = RecordedHTTP(model.http, root / "helper-http")
    retriever = LocalRetriever(bm25, rid)
    driver = SmartSearchDriver(
        checkout=config["checkout"],
        runtime_config=runtime,
        model_factory=lambda cfg: model,
        retriever_factory=lambda cfg: retriever,
        environ={},
    )
    client = DriverClient(driver)
    method = SmartSearchCommonSupportQueries(artifacts=artifacts)
    preparation = {}

    def target(job):
        save(root / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        cell = job["benchmark"]
        example = adapters[cell].example(job["split"], job["case_id"])
        state = example.visible_state(budget=ExecutionBudget(**BUDGET))
        folder, record, error = preparation.get((cell, example.example_id), (None, None, None))
        calls = []
        target_model = job["model"]
        binding = json.loads((Path(target_model["service_directory"]) / "started.json").read_text())
        result = {
            "job": job,
            "model_binding": binding,
            "input_state_hash": state.state_hash,
            "shared_preparation": str(folder) if folder is not None else None,
            "package_location": __import__("agentinputlab").__file__,
        }
        try:
            if job["release"] != "baseline.no_op":
                if error:
                    raise RuntimeError("shared native helper failed; target inference unstarted")
                state = record.patch.apply(state)
                state = replace(
                    state,
                    budget=replace(
                        state.budget,
                        max_model_calls=11 - record.usage.model_calls,
                        max_output_tokens=82432 - record.usage.output_tokens,
                        max_context_tokens=262144 - record.usage.input_tokens,
                    ),
                )
            backend = RecordingBackend(
                calls=calls,
                model_id=target_model["id"],
                base_url=target_model["base_url"],
                deployment_identity={
                    "revision": digest(binding["model_json_hashes"]),
                    "server": "vllm",
                },
                extra_body=target_model.get("extra_body", {}),
                timeout_seconds=300,
                transport=local_transport,
            )
            predictor = ProductionPredictorBuilder().build(
                StaticPredictorBuildRequest(
                    benchmark_id=cell,
                    split=job["split"],
                    model_slot=target_model["id"],
                    seed=17,
                    builder_parameters={
                        "schema_version": 1,
                        "task": "qa",
                        "qa": {
                            "output_format": "structured_json",
                            "max_tokens": 512,
                            "use_seed": True,
                        },
                    },
                    runtime=StaticRuntimeComponents(backend=backend),
                )
            )
            prediction = predictor.predict(state)
            result.update(
                status="scored_terminal",
                prediction=to_jsonable(prediction),
                metrics=dict(adapters[cell].score(example, prediction.prediction)),
            )
        except Exception as exc:
            result.update(
                status="failed_helper_dependency"
                if error and job["release"] != "baseline.no_op"
                else "failed_terminal",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        save(
            root / "terminals" / (job["id"] + ".json"),
            {**result, "calls": calls, "automatic_retry": False},
        )

    pools = {m["id"]: ThreadPoolExecutor(2) for m in config["models"]}
    futures = [
        pools[j["model"]["id"]].submit(target, j) for j in jobs if j["release"] == "baseline.no_op"
    ]
    try:
        for c, _split in QA_CELLS:
            for example in selected[c]:
                folder = root / "preparation" / digest({"benchmark": c, "id": example.example_id})
                folder.mkdir()
                client.current = folder
                state = example.visible_state(budget=ExecutionBudget(**BUDGET))
                try:
                    record = InterventionRunner().run(
                        method, state, InterventionContext(seed=17, external_runtime=client)
                    )
                    result = {
                        "status": "prepared_actual_native_loop",
                        "intervention": to_jsonable(record),
                    }
                    preparation[(c, example.example_id)] = (folder, record, None)
                except Exception as exc:
                    result = {
                        "status": "failed_preparation",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "usage": driver.last_usage.as_mapping() if driver.last_usage else None,
                    }
                    preparation[(c, example.example_id)] = (folder, None, result)
                save(
                    folder / "terminal.json",
                    {
                        **result,
                        "question_only_source_state_hash": state.state_hash,
                        "helper_binding": helper_binding,
                        "automatic_retry": False,
                    },
                )
                for job in jobs:
                    if (job["benchmark"], job["case_id"], job["release"]) == (
                        c,
                        example.example_id,
                        RELEASE,
                    ):
                        futures.append(pools[job["model"]["id"]].submit(target, job))
        for future in futures:
            future.result()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    rows = [json.loads(p.read_text()) for p in (root / "terminals").glob("*.json")]
    save(
        root / "complete.json",
        {
            "preparations": 8,
            "successful_preparations": sum(x[1] is not None for x in preparation.values()),
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(x["status"] == "scored_terminal" for x in rows),
            "target_actual_calls": sum(len(x["calls"]) for x in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True)
    execute(json.loads(a.config.read_text()), a.output)


if __name__ == "__main__":
    main()
