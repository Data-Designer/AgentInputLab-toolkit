"""Real question-only Adaptive-RAG composite acceptance on bounded mounted QASPER cases.

This checks the complete controlled worker, not its common-support patch. Data,
weights, indexes and native source are never acquired by this operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.acquire.adaptive_rag import (
    AdaptiveRAGArtifactIdentity,
    AdaptiveRAGControlledWorker,
    AdaptiveRAGControls,
)
from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts import FrozenAgentState
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from workers.adaptive_rag.driver import (
    AdaptiveRAGWorkerDriver,
    RuntimeBundle,
    RuntimeIdentity,
    _path_closure_sha256,
)
from workers.adaptive_rag.runtime import (
    LocalTransformersRouter,
    OpenAICompatibleGenerator,
    ToolkitRetriever,
)

CELL, SPLIT, OFFSET = "qasper.v0_3_dev", "dev", 102
RELEASE = "acquire.adaptive_rag.controlled_reimplementation"
PROMPTS = {
    "no_retrieval": "Answer the question using your existing knowledge.",
    "single_retrieval": "Answer the question using the retrieved evidence.",
    "ircot": "Reason about the question using the retrieved evidence, retrieve additional evidence when useful, then answer.",
}


def select_cases(adapter, selection=None):
    if selection is not None:
        cases = selection.get("selected_cases")
        if (
            not isinstance(cases, list)
            or not 1 <= len(cases) <= 2
            or any(
                not isinstance(row, dict)
                or row.get("route") not in {"A", "C"}
                or not isinstance(row.get("case_id"), str)
                for row in cases
            )
            or len({row["case_id"] for row in cases}) != len(cases)
            or len({row["route"] for row in cases}) != len(cases)
        ):
            raise ValueError("boundary selection must have one case per real A/C route")
        by_id = {ex.example_id: ex for ex in adapter.examples(SPLIT)}
        if any(row["case_id"] not in by_id for row in cases):
            raise ValueError("boundary case is absent from the real mounted source")
        return [by_id[row["case_id"]] for row in cases]
    selected = sorted(adapter.examples(SPLIT), key=lambda x: x.example_id)[OFFSET : OFFSET + 2]
    if len(selected) != 2:
        raise ValueError("requires two fixed QASPER source cases at102:104")
    return selected


def load_boundary_selection(specification, adapter):
    if specification is None:
        return None, None
    if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
        raise ValueError("boundary selection requires an exact receipt path and trusted SHA")
    path = Path(specification["path"])
    selection = load_pinned_json_mapping(path, expected_sha256=specification["sha256"])
    if (
        selection.get("schema") != "agentinputlab.adaptive-route-boundary-selection.v1"
        or selection.get("benchmark") != CELL
        or selection.get("split") != SPLIT
        or selection.get("no_reader_model_calls") is not True
        or selection.get("screening_attempts") != 64
    ):
        raise ValueError("boundary selection does not match the frozen source-only protocol")
    bank = load_pinned_json_mapping(
        path.parent / "manifest.json", expected_sha256=selection["source_bank_manifest_sha256"]
    )
    if (
        bank.get("reader_quality_used_for_selection") is not False
        or bank.get("gold_or_training_labels_exposed_to_router") is not False
        or bank.get("source_file_sha256") != dict(adapter.prepared.file_sha256)
    ):
        raise ValueError("boundary source or non-oracle selection gate differs")
    selected = select_cases(adapter, selection)
    source_bank = {row["case_id"]: row for row in bank["candidates"]}
    if any(
        ex.example_id not in source_bank
        or content_hash(ex.visible_state().query) != source_bank[ex.example_id]["question_sha256"]
        for ex in selected
    ):
        raise ValueError("boundary query differs from the screened source")
    return selection, specification["sha256"]


def state_for(example):
    # Both arms share this declared question-only view. Evidence is exclusively
    # a separately bound retrieval corpus; no reference/support labels exposed.
    return FrozenAgentState(
        query=example.visible_state().query,
        task_id=example.example_id,
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096, max_retrieval_calls=8),
    )


def backend_for(model, revision, calls):
    return RecordingBackend(
        model_id=model["id"],
        base_url=model["base_url"],
        calls=calls,
        deployment_identity={"revision": revision},
        extra_body=model.get("extra_body", {}),
        api_key_env="AILAB_TOOLKIT_UNUSED_LOCAL_ONLY",
        transport=local_transport,
        timeout_seconds=240,
    )


class RecordedDriver:
    worker_id = "adaptive-rag-controlled-live"

    def __init__(self, driver, identity):
        self.driver, self.identity = driver, identity
        self.requests, self.responses = [], []

    def fingerprint(self):
        return self.identity

    def call(self, request):
        self.requests.append(to_jsonable(request))
        response = self.driver.handle(request)
        self.responses.append(to_jsonable(response))
        return response

    def call_controlled(self, request):
        return self.call(request.as_dict())


class CountedRouter:
    """Observe physical native calls; never infer them from successful job counts."""

    def __init__(self, native):
        self.native, self.lock, self.calls = native, threading.Lock(), []

    def component_manifest(self):
        return {
            **self.native.component_manifest(),
            "implementation_class": f"{type(self).__module__}.{type(self).__qualname__}",
        }

    def estimate_input_tokens(self, question):
        return self.native.estimate_input_tokens(question)

    def first_decoder_step(self, question, *, seed):
        with self.lock:
            row = {"question_sha256": content_hash(question), "seed": seed}
            self.calls.append(row)
            try:
                result = self.native.first_decoder_step(question, seed=seed)
                row["result"] = asdict(result)
                return result
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                raise


def run(config, output, upstream, router_root, observed):
    import torch

    import agentinputlab.backends.retrieval.memory as bm25_module
    import workers.adaptive_rag.driver as driver_module
    import workers.adaptive_rag.runtime as runtime_module

    if output.exists():
        raise ValueError("create-only cohort; do not repeat saved scientific attempts")
    torch.set_num_threads(8)
    bindings = verify_services(config["models"])
    adapter = load_production_benchmark_adapter(CELL, config["prepared_root"])
    selection, selection_sha = load_boundary_selection(config.get("boundary_selection"), adapter)
    selected = select_cases(adapter, selection)
    boundary_labels = (
        {row["case_id"]: row["route"] for row in selection["selected_cases"]} if selection else {}
    )
    inventory = json.loads(observed.read_text())
    prior = {
        (r["release"], r["benchmark"], r["backbone"], e["case_id"])
        for r in inventory["rows"]
        for e in r["evidence"]
    }
    jobs = []
    for model in config["models"]:
        for example in selected:
            for release in ("baseline.no_op", RELEASE):
                if (release, CELL, model["id"], example.example_id) in prior:
                    raise ValueError(
                        "fixed case has prior scientific attempt; review exact reuse first"
                    )
                job = {
                    "model": model,
                    "benchmark": CELL,
                    "split": SPLIT,
                    "case_id": example.example_id,
                    "release": release,
                    "seed": 17,
                    "budget": {
                        "max_model_calls": 8,
                        "max_output_tokens": 4096,
                        "max_retrieval_calls": 8,
                    },
                    "per_call_output_cap": 512,
                    "input_view": "question_only_both_arms",
                    "corpus_scope": "all_original_visible_paragraphs_of_this_case",
                    "parser": "final_text_only_QAPrediction_no_inferred_support",
                    "router": "existing_t5_base_flan_label_controlled_transfer",
                    "controlled_prompts": PROMPTS,
                    "input_state_hash": state_for(example).state_hash,
                }
                if selection:
                    job["boundary_target_route"] = boundary_labels[example.example_id]
                    job["boundary_selection_sha256"] = selection_sha
                job["id"] = digest(job)
                jobs.append(job)
    if config.get("expected_job_ids") is not None and sorted(j["id"] for j in jobs) != sorted(
        config["expected_job_ids"]
    ):
        raise ValueError("entry continuation differs from the frozen original jobs")
    # Check the actual installed constructor before claims or model dispatch.
    for model in config["models"]:
        backend_for(model, digest(bindings[model["id"]]["model_json_hashes"]), [])
    router_manifest = json.loads((router_root / "artifact-manifest.json").read_text())
    if (
        router_manifest["upstream_router_checkpoint_parity_claimed"] is not False
        or router_manifest["upstream_commit"] != "0c88670af8707667eb5c1163151bb5ce61b14acb"
    ):
        raise ValueError(
            "requires disclosed T5-base operator transfer artifact, not checkpoint parity"
        )
    # QASPER is not among this router's six declared native training datasets.
    if (
        router_manifest["training_data_relative_path"]
        != "classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/binary_silver/train.json"
    ):
        raise ValueError("router training-source scope changed; review before QASPER dispatch")
    output.mkdir(parents=True)
    for directory in ("claims", "terminals", "deployment", "prompts"):
        (output / directory).mkdir()
    prompt_paths = {}
    for name, text in PROMPTS.items():
        path = output / "prompts" / (name + ".txt")
        with path.open("x") as handle:
            handle.write(text)
        prompt_paths[name] = path
    common = dict(
        router_revision=router_manifest["artifact_closure_sha256"],
        router_snapshot_sha256=_path_closure_sha256(router_root),
        router_weights_sha256=_path_closure_sha256(router_root / "model.safetensors"),
        router_tokenizer_sha256=_path_closure_sha256(router_root / "tokenizer.json"),
        retriever_revision="toolkit.in_memory_bm25.v1",
        retriever_implementation_sha256=_path_closure_sha256(Path(bm25_module.__file__)),
    )
    contexts, deployments, shared_router = {}, {}, None
    for example in selected:
        visible = example.visible_state()
        documents = tuple(
            Document(
                block.block_id or f"p{i}",
                block.text,
                title=str(block.metadata.get("title", "")),
                corpus_revision=content_hash(visible.context),
            )
            for i, block in enumerate(visible.context)
        )
        if not documents or len({d.doc_id for d in documents}) != len(documents):
            raise ValueError("requires nonempty unambiguous real visible corpus")
        retriever = InMemoryBM25Retriever(documents)
        corpus_path = output / "deployment" / (example.example_id + "-corpus.json")
        index_path = output / "deployment" / (example.example_id + "-index.json")
        save(
            corpus_path,
            {
                "documents": documents,
                "source_state_hash": visible.state_hash,
                "selection": "all_visible_context_no_gold_enrichment",
            },
        )
        save(index_path, retriever.fingerprint())
        contexts[example.example_id] = (example, retriever, corpus_path, index_path)
        for model in config["models"]:
            key = model["id"], example.example_id
            prefix = output / "deployment" / digest(key)
            artifact_path = prefix.with_suffix(".model.json")
            generator_path = prefix.with_suffix(".generator.json")
            retriever_path = prefix.with_suffix(".retriever.json")
            save(artifact_path, bindings[model["id"]])
            save(
                generator_path,
                {
                    "model": model,
                    "temperature": 0,
                    "input_token_upper_bound": 65536,
                    "final_text_only": True,
                    "implicit_retry": False,
                    "runtime_sha256": _path_closure_sha256(Path(runtime_module.__file__)),
                },
            )
            save(
                retriever_path,
                {"retriever": retriever.fingerprint(), "top_k_controlled_by_driver": True},
            )
            identity = RuntimeIdentity(
                **common,
                base_model_revision=digest(bindings[model["id"]]["model_json_hashes"]),
                generator_artifact_sha256=_path_closure_sha256(artifact_path),
                generator_config_sha256=_path_closure_sha256(generator_path),
                retriever_config_sha256=_path_closure_sha256(retriever_path),
                corpus_revision=content_hash(visible.context),
                corpus_manifest_sha256=_path_closure_sha256(corpus_path),
                index_revision=content_hash(retriever.fingerprint()),
                index_manifest_sha256=_path_closure_sha256(index_path),
            )
            if shared_router is None:
                shared_router = CountedRouter(
                    LocalTransformersRouter(
                        snapshot_root=router_root,
                        identity=identity,
                        weights_path=router_root / "model.safetensors",
                        tokenizer_path=router_root / "tokenizer.json",
                        device="cpu",
                    )
                )
            deployments[key] = (identity, artifact_path, generator_path, retriever_path)
    source_sha = _path_closure_sha256(Path(driver_module.__file__))
    source_records = {}
    for key, (identity, artifact_path, generator_path, retriever_path) in deployments.items():
        if key[0] in source_records:
            continue
        example, retriever, corpus_path, index_path = contexts[key[1]]
        # Startup verification invokes no model. All runtime closure gates stay on.
        generator = OpenAICompatibleGenerator(
            backend=None, identity=identity, input_token_upper_bound=65536
        )
        driver = AdaptiveRAGWorkerDriver(
            source_root=upstream,
            router_weights_path=router_root / "model.safetensors",
            prompt_root=output / "prompts",
            runtime=RuntimeBundle(
                identity,
                shared_router,
                generator,
                ToolkitRetriever(retriever=retriever, identity=identity),
            ),
            router_snapshot_path=router_root,
            router_tokenizer_path=router_root / "tokenizer.json",
            generator_artifact_manifest_path=artifact_path,
            generator_config_path=generator_path,
            retriever_implementation_path=Path(bm25_module.__file__),
            retriever_config_path=retriever_path,
            corpus_manifest_path=corpus_path,
            index_manifest_path=index_path,
        )
        source_records[key[0]] = driver.source_fingerprint()
    save(
        output / "manifest.json",
        {
            "schema": "agentinputlab.adaptive-controlled-acceptance.v1",
            "jobs": jobs,
            "dataset": dict(adapter.prepared.file_sha256),
            "router": router_manifest,
            "source": source_records,
            "driver_sha256": source_sha,
            "runtime_sha256": _path_closure_sha256(Path(runtime_module.__file__)),
            "operator_sha256": _path_closure_sha256(Path(__file__)),
            "prior_inventory_sha256": _path_closure_sha256(observed),
            "actual_model_bindings": bindings,
            "controlled_arms_not_upstream_arm_parity": True,
            "checkpoint_parity_claimed": False,
            "physical_router_inference_per_treatment": True,
            "automatic_retry": False,
            "paper_results_changed": False,
            "boundary_selection": selection,
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
    stop = threading.Event()

    def keepalive():
        while not stop.wait(30):
            (output.parent / "keep-services-alive").touch()

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()

    def one(job):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model = job["model"]
        example, retriever, corpus_path, index_path = contexts[job["case_id"]]
        identity, artifact_path, generator_path, retriever_path = deployments[
            (model["id"], example.example_id)
        ]
        calls = []
        backend = backend_for(model, identity.base_model_revision, calls)
        generator = OpenAICompatibleGenerator(
            backend=backend, identity=identity, input_token_upper_bound=65536
        )
        state = state_for(example)
        row = {
            "job": job,
            "input_state_hash": state.state_hash,
            "package_location": __import__("agentinputlab").__file__,
            "model_binding": bindings[model["id"]],
            "calls": calls,
        }
        try:
            if job["release"] == "baseline.no_op":
                request = generator.request(
                    operation="answer.no_retrieval",
                    question=state.query,
                    documents=(),
                    reasoning=(),
                    prompt=PROMPTS["no_retrieval"],
                    max_output_tokens=512,
                    seed=17,
                )
                response = backend.generate(request)
                if not response.text.strip():
                    raise ValueError("model produced empty final answer within fixed output budget")
                answer = response.text
            else:
                driver = AdaptiveRAGWorkerDriver(
                    source_root=upstream,
                    router_weights_path=router_root / "model.safetensors",
                    prompt_root=output / "prompts",
                    runtime=RuntimeBundle(
                        identity,
                        shared_router,
                        generator,
                        ToolkitRetriever(retriever=retriever, identity=identity),
                    ),
                    router_snapshot_path=router_root,
                    router_tokenizer_path=router_root / "tokenizer.json",
                    generator_artifact_manifest_path=artifact_path,
                    generator_config_path=generator_path,
                    retriever_implementation_path=Path(bm25_module.__file__),
                    retriever_config_path=retriever_path,
                    corpus_manifest_path=corpus_path,
                    index_manifest_path=index_path,
                )
                worker_identity = {
                    "worker_id": "adaptive-controlled-inprocess",
                    "driver_sha256": source_sha,
                    "runtime_sha256": _path_closure_sha256(Path(runtime_module.__file__)),
                    "identity_sha256": content_hash(asdict(identity)),
                    "fixture_mode": False,
                }
                client = RecordedDriver(driver, worker_identity)
                row["worker_requests"], row["worker_responses"] = client.requests, client.responses
                artifacts = AdaptiveRAGArtifactIdentity(
                    **{k: v for k, v in asdict(identity).items() if k != "fixture_runtime"},
                    router_checkpoint="operator/t5-base-flan-xl-15e-transfer",
                    worker_fingerprint_sha256=content_hash(worker_identity),
                    **{
                        k + "_prompt_sha256": _path_closure_sha256(v)
                        for k, v in prompt_paths.items()
                    },
                )
                result = AdaptiveRAGControlledWorker(
                    artifacts=artifacts,
                    client=client,
                    controls=AdaptiveRAGControls(max_output_tokens=512),
                ).run(state, InterventionContext(seed=17))
                row["composite_result"] = to_jsonable(result)
                answer = result.final_answer
            row.update(
                status="scored_terminal",
                prediction={"answer": answer},
                metrics=dict(adapter.score(example, QAPrediction(answer=answer))),
            )
        except Exception as exc:
            row.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        if selection and job["release"] != "baseline.no_op":
            events = [
                event
                for response in row.get("worker_responses", [])
                for event in (
                    response.get("payload", {}).get("execution_evidence", {}).get("events", [])
                    if response.get("status") == "ok"
                    else response.get("partial_events", [])
                )
                if event["operation"] == "router.first_decoder_step"
            ]
            actual_labels = [
                max(("A", "B", "C"), key=lambda k: event["result"]["route_logits"][k])
                for event in events
            ]
            row["boundary_route_verified"] = actual_labels == [job["boundary_target_route"]]
        row["known_generation_responses"] = sum("response" in c for c in calls)
        row["generation_attempts"] = len(calls)
        save(output / "terminals" / (job["id"] + ".json"), row)
        print(
            json.dumps(
                {
                    "model": model["id"],
                    "release": job["release"],
                    "status": row["status"],
                    "generation_attempts": len(calls),
                }
            ),
            flush=True,
        )

    try:

        def lane(model):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(one, (j for j in jobs if j["model"]["id"] == model["id"])))

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lane, config["models"]))
    finally:
        stop.set()
        thread.join()
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    assert len(rows) == len(jobs) == len(config["models"]) * len(selected) * 2
    save(
        output / "complete.json",
        {
            "expected": len(jobs),
            "terminal_count": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "automatic_retry": False,
            "physical_router_calls": len(shared_router.calls),
            "physical_router_responses": sum("result" in r for r in shared_router.calls),
            "router_call_records": shared_router.calls,
            "actual_generation_attempts": sum(r["generation_attempts"] for r in rows),
            "publication_ready": False,
        },
    )
    save(output / "post-service-identities.json", verify_services(config["models"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "output", "upstream", "router", "observed"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.upstream, args.router, args.observed)


if __name__ == "__main__":
    main()
