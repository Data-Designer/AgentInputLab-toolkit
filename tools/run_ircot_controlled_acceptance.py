"""Fixed real HotpotQA acceptance through the installed native IRCoT controller.

No asset acquisition or retry. Each task has a fresh process/private HOME. The
native evidence-acquisition stage and downstream answer share 8 calls/4096 output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.acquire.ircot import IRCoTControlledWorker, IRCoTControls
from agentinputlab.backends import (
    Document,
    FinalContentStopBackend,
    GenerationRequest,
    InMemoryBM25Retriever,
)
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts import ContextBlock, FrozenAgentState
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner
from workers.ircot import PinnedIRCoTControlledRuntime

CELL, SPLIT, OFFSET = "hotpotqa.distractor", "validation", 200
RELEASES = ("baseline.no_op", "acquire.ircot.controlled")
CONTROLS = {
    "per_step_top_k": 5,
    "max_steps": 7,
    "max_total_documents": 15,
    "max_output_tokens": 3584,
    "deduplicate": True,
    "reset_policy": "upstream_native",
}


def select_cases(adapter):
    examples = sorted(adapter.examples(SPLIT), key=lambda e: e.example_id)[OFFSET : OFFSET + 2]
    if len(examples) != 2:
        raise ValueError("requires two fixed real HotpotQA cases at200:202")
    return examples


def state_for(example):
    return FrozenAgentState(
        query=example.visible_state().query,
        task_id=example.example_id,
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096, max_retrieval_calls=8),
    )


def source_documents(example):
    visible = example.visible_state()
    documents = tuple(
        Document(
            b.block_id or f"p{i}",
            b.text,
            title=str(b.metadata.get("title", b.block_id or f"p{i}")),
            corpus_revision=content_hash(visible.context),
        )
        for i, b in enumerate(visible.context)
    )
    if not documents or len({d.doc_id for d in documents}) != len(documents):
        raise ValueError("requires a nonempty unique real visible paragraph corpus")
    return documents


def build_jobs(models, cases, source_proof):
    jobs = []
    for model in models:
        for example in cases:
            for release in RELEASES:
                job = {
                    "model": model,
                    "release": release,
                    "benchmark": CELL,
                    "split": SPLIT,
                    "case_id": example.example_id,
                    "seed": 17,
                    "budget": {
                        "max_model_calls": 8,
                        "max_output_tokens": 4096,
                        "max_retrieval_calls": 8,
                    },
                    "per_call_output_cap": 512,
                    "native_controls": CONTROLS,
                    "native_seed_policy": "17 + native call index; downstream answer seed17",
                    "native_input_limit": "original GPT2 prompt fitting8000",
                    "input_view": "question_only_both_arms",
                    "corpus_scope": "all_original_visible_distractor_paragraphs_of_this_case",
                    "final_parser": "raw_final_text_only_no_inferred_support",
                    "final_content_stop_policy": "native logical stops on final content; all provider usage charged",
                    "input_state_hash": state_for(example).state_hash,
                    "source_proof": source_proof,
                }
                job["id"] = digest(job)
                jobs.append(job)
    return jobs


def source_proof(config, assets):
    import agentinputlab

    wheel = Path(config["distribution"]["path"])
    expected = config["distribution"]["sha256"]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
        raise ValueError("trusted installed distribution checksum differs")
    installed = Path(agentinputlab.__file__).parent.parent
    count = 0
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.startswith(("agentinputlab/", "workers/")) and not name.endswith("/"):
                if (installed / name).read_bytes() != z.read(name):
                    raise ValueError("installed package source differs from the selected wheel")
                count += 1
    tokenizer_path = assets / "gpt2-tokenizer-closure-v2.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    for name, sha in tokenizer["file_sha256"].items():
        if hashlib.sha256((assets / "gpt2-tokenizer" / name).read_bytes()).hexdigest() != sha:
            raise ValueError("native GPT2 tokenizer asset changed")
    parser = assets / "en_core_web_sm-3.4.1-py3-none-any.whl"
    if (
        hashlib.sha256(parser.read_bytes()).hexdigest()
        != "83872781dc64893d45d9dbe940e05f80df7e7196e169ea29e2e9742fed079549"
    ):
        raise ValueError("official native sentence-parser checksum differs")
    model_files = {}
    with zipfile.ZipFile(parser) as z:
        for name in z.namelist():
            if name.startswith("en_core_web_sm/") and not name.endswith("/"):
                data = z.read(name)
                if (installed / name).read_bytes() != data:
                    raise ValueError("installed native sentence parser differs")
                model_files[name] = hashlib.sha256(data).hexdigest()
    versions = {
        n: importlib.metadata.version(n)
        for n in ("spacy", "en-core-web-sm", "transformers", "jsonnet", "nltk", "numpy", "openai")
    }
    return {
        "wheel_sha256": expected,
        "packaged_files_verified": count,
        "tokenizer_manifest_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokenizer": tokenizer,
        "native_dependencies": versions,
        "native_parser_installed_closure_sha256": content_hash(model_files),
    }


def one(specification):
    bundle = json.loads(specification.read_text())
    job = bundle["job"]
    paths = bundle["paths"]
    output = Path(paths["output"])
    directory = output / "workers" / job["id"]
    directory.mkdir()
    home = directory / "home"
    home.mkdir()
    os.environ.update(
        HOME=str(home),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(Path(paths["assets"]) / "hf-home"),
        NLTK_DATA=str(Path(paths["assets"]) / "nltk_data"),
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    save(
        output / "claims" / (job["id"] + ".json"),
        {"job": job, "started": time.time(), "pid": os.getpid()},
    )
    calls = []
    record = None
    runtime = None
    monitor = RunMonitor()
    row = {
        "job": job,
        "calls": calls,
        "scientific_retries": 0,
        "additional_attempts": 1,
        "package_location": __import__("agentinputlab").__file__,
        "input_state_hash": job["input_state_hash"],
    }
    stage = "source_and_native_initialization"
    try:
        adapter = load_production_benchmark_adapter(CELL, paths["prepared_root"])
        if dict(adapter.prepared.file_sha256) != bundle["dataset_file_sha256"]:
            raise ValueError("mounted source file differs from frozen source")
        example = next(e for e in adapter.examples(SPLIT) if e.example_id == job["case_id"])
        state = state_for(example)
        if state.state_hash != job["input_state_hash"]:
            raise ValueError("fixed input view changed")
        documents = source_documents(example)
        corpus = content_hash(documents)
        retriever = InMemoryBM25Retriever(documents)
        model = job["model"]
        inner = RecordingBackend(
            model_id=model["id"],
            base_url=model["base_url"],
            deployment_identity={
                "revision": digest(bundle["model_binding"]["model_json_hashes"]),
                "server": "vllm",
            },
            calls=calls,
            timeout_seconds=240,
            transport=local_transport,
            extra_body=model.get("extra_body", {}),
        )
        backend = FinalContentStopBackend(inner)
        row["backend_fingerprint"] = to_jsonable(backend.fingerprint())
        row["retriever_fingerprint"] = to_jsonable(retriever.fingerprint())
        row["visible_corpus_sha256"] = corpus
        if job["release"] != "baseline.no_op":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(Path(paths["assets"]) / "gpt2-tokenizer"), local_files_only=True
            )
            runtime = PinnedIRCoTControlledRuntime(
                source_root=Path(paths["upstream"]),
                backend=backend,
                retriever=retriever,
                tokenizer=tokenizer,
                tokenizer_fingerprint=job["source_proof"]["tokenizer"],
                dependency_fingerprint={
                    "versions": job["source_proof"]["native_dependencies"],
                    "parser_closure": job["source_proof"]["native_parser_installed_closure_sha256"],
                },
                corpus_revision=corpus,
                index_revision=content_hash(retriever.fingerprint()),
                cache_home=home,
                max_model_calls=7,
                max_output_tokens=3584,
                max_retrieval_calls=8,
                per_call_output_tokens=512,
                seed=17,
            )
            row["worker_fingerprint"] = to_jsonable(runtime.fingerprint())
            method = IRCoTControlledWorker(
                corpus_revision=corpus,
                index_revision=content_hash(retriever.fingerprint()),
                prompt_hash=runtime.fingerprint()["prompt_sha256"],
                controls=IRCoTControls(**CONTROLS),
            )
            stage = "native_controller"
            record = InterventionRunner(monitor=monitor).run(
                method,
                state,
                InterventionContext(
                    backend=backend, retriever=retriever, external_runtime=runtime, seed=17
                ),
            )
            state = record.patch.apply(state)
            native_usage = runtime.last_usage
            used_calls = native_usage["model_calls"]
            used_output = native_usage["output_tokens"]
        else:
            hits = retriever.retrieve(state.query, top_k=5)
            state = FrozenAgentState(
                query=state.query,
                task_id=state.task_id,
                budget=state.budget,
                context=tuple(
                    ContextBlock(
                        h.document.text,
                        block_id=h.document.doc_id,
                        source=corpus,
                        metadata={"title": h.document.title},
                    )
                    for h in hits
                ),
            )
            used_calls = used_output = 0
        if used_calls >= 8 or used_output >= 4096:
            raise ValueError("no reserved downstream budget remains")
        stage = "downstream_answer"
        context = "\n\n".join(
            f"[{b.block_id}] {b.metadata.get('title', '')}\n{b.text}" for b in state.context
        )
        prompt = (
            "Use the paragraphs to answer the question. Return the answer itself.\n\n"
            + context
            + "\n\nQuestion: "
            + state.query
            + "\nAnswer:"
        )
        answer = backend.generate(
            GenerationRequest(
                prompt=prompt, max_tokens=min(512, 4096 - used_output), temperature=0, seed=17
            )
        )
        if not answer.text.strip():
            raise ValueError("downstream final content is empty under the fixed generation budget")
        row["prediction"] = {"answer": answer.text}
        row["metrics"] = dict(adapter.score(example, QAPrediction(answer=answer.text)))
        row["status"] = "scored_terminal"
    except Exception as exc:
        row.update(
            status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
        )
    row["intervention"] = to_jsonable(record) if record else None
    row["worker_events"] = to_jsonable(runtime.last_events) if runtime else []
    row["worker_usage"] = dict(runtime.last_usage) if runtime else {}
    row["monitor_snapshots"] = to_jsonable(monitor.list_snapshots())
    row["generation_attempts"] = len(calls)
    row["known_generation_responses"] = sum("response" in c for c in calls)
    assert len(calls) <= 8
    assert sum(c["response"]["usage"]["output_tokens"] for c in calls if "response" in c) <= 4096
    save(output / "terminals" / (job["id"] + ".json"), row)
    (output.parent / "keep-services-alive").touch()


def run(config, output, upstream, assets, observed):
    if output.exists():
        raise ValueError("create-only batch; preserve all previous attempts")
    bindings = verify_services(config["models"])
    adapter = load_production_benchmark_adapter(CELL, config["prepared_root"])
    cases = select_cases(adapter)
    raw_inventory = observed.read_bytes()
    if hashlib.sha256(raw_inventory).hexdigest() != config.get("observed_sha256"):
        raise ValueError("the frozen prior receipt inventory checksum differs")
    inventory = json.loads(raw_inventory)
    prior = {
        (r["release"], r["benchmark"], r["backbone"], e["case_id"])
        for r in inventory["rows"]
        for e in r["evidence"]
    }
    if any(
        (release, CELL, m["id"], e.example_id) in prior
        for m in config["models"]
        for e in cases
        for release in RELEASES
    ):
        raise ValueError("fixed source cases intersect prior attempts; do not resample silently")
    demo = (
        upstream / "prompts/hotpotqa/gold_with_2_distractors_context_cot_qa_codex.txt"
    ).read_text()
    if any(e.example_id in demo for e in cases):
        raise ValueError("fixed evaluation case appears among native source demonstrations")
    proof = source_proof(config, assets)
    jobs = build_jobs(config["models"], cases, proof)
    output.mkdir()
    for name in ("claims", "terminals", "jobs", "workers"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        {
            "jobs": jobs,
            "actual_model_bindings": bindings,
            "dataset_file_sha256": dict(adapter.prepared.file_sha256),
            "source_proof": proof,
            "native_source_is_original_controller": True,
            "historical_Codex_or_full_corpus_parity": False,
            "all_visible_paragraphs_retained_no_gold_enrichment": True,
            "retrieval_baseline": "one original-query BM25 retrieval",
            "shared_method_plus_answer_budget": {"calls": 8, "output_tokens": 4096},
            "unknown_transport_usage_retained_not_zero": True,
            "CPU_GPU_compute_not_measured": True,
            "paper_results_changed": False,
            "automatic_retry": False,
            "observed_inventory_sha256": config["observed_sha256"],
        },
    )
    for job in jobs:
        save(
            output / "jobs" / (job["id"] + ".json"),
            {
                "job": job,
                "paths": {
                    "output": str(output),
                    "prepared_root": config["prepared_root"],
                    "assets": str(assets),
                    "upstream": str(upstream),
                },
                "dataset_file_sha256": dict(adapter.prepared.file_sha256),
                "model_binding": bindings[job["model"]["id"]],
            },
        )
    stop = threading.Event()

    def keepalive():
        while not stop.wait(30):
            (output.parent / "keep-services-alive").touch()

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()

    def launch(job):
        with (output / "workers" / (job["id"] + ".log")).open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "one",
                    "--specification",
                    str(output / "jobs" / (job["id"] + ".json")),
                ],
                cwd="/tmp",
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
                check=True,
            )

    def lane(model):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(launch, [j for j in jobs if j["model"]["id"] == model["id"]]))

    try:
        with ThreadPoolExecutor(len(config["models"])) as pool:
            list(pool.map(lane, config["models"]))
    finally:
        stop.set()
        thread.join()
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    assert len(rows) == len(jobs)
    save(output / "post-service-identities.json", verify_services(config["models"]))
    save(
        output / "complete.json",
        {
            "expected": len(jobs),
            "terminal_count": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "failed": sum(r["status"] == "failed_terminal" for r in rows),
            "generation_attempts": sum(r["generation_attempts"] for r in rows),
            "scientific_retries": 0,
            "publication_ready": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="operation", required=True)
    single = subs.add_parser("one")
    single.add_argument("--specification", type=Path, required=True)
    batch = subs.add_parser("run")
    for name in ("config", "output", "upstream", "assets", "observed"):
        batch.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "one":
        one(args.specification)
    else:
        run(
            json.loads(args.config.read_text()),
            args.output,
            args.upstream,
            args.assets,
            args.observed,
        )


if __name__ == "__main__":
    main()
