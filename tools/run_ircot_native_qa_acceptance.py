"""Two source-fixed HotpotQA cases through the entire mounted native QA graph.

One source-isolated subprocess/private HOME per case. Raw model/native outcomes
are durable before independent QA scoring; failed consumed cases are not retried.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.api import create_intervention, run_composite_intervention
from agentinputlab.backends import Document, FinalContentStopBackend, InMemoryBM25Retriever
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.observability import RunMonitor

RELEASE = "acquire.ircot.native_like_substituted"
CONTROLS = dict(
    per_step_top_k=6,
    max_steps=7,
    max_total_documents=15,
    max_output_tokens=4096,
    deduplicate=True,
    reset_policy="upstream_native",
)


def state_for(example):
    return FrozenAgentState(
        query=example.query,
        task_id=example.example_id,
        budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096, max_retrieval_calls=8),
    )


def documents_for(example):
    source = example.visible_state()
    corpus = content_hash(source.context)
    return tuple(
        Document(
            b.block_id,
            b.text,
            title=str(b.metadata.get("title", b.block_id)),
            corpus_revision=corpus,
        )
        for b in source.context
    )


def one(specification):
    spec = json.loads(specification.read_text())
    job = spec["job"]
    config = spec["config"]
    output = Path(spec["output"])
    directory = output / "workers" / job["id"]
    directory.mkdir()
    home = directory / "home"
    home.mkdir()
    assets = Path(config["native_assets"])
    # Set cache scope before importing Transformers or any native source.
    os.environ.update(
        HOME=str(home),
        HF_HOME=str(assets / "hf-home"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NLTK_DATA=str(assets / "nltk_data"),
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
    )
    save(output / "claims" / (job["id"] + ".json"), dict(job=job, pid=os.getpid()))
    calls = []
    runtime = record = raw = None
    monitor = RunMonitor()
    row = dict(job=job, additional_attempts=1, automatic_retry=False)
    stage = "native_source_initialization"
    try:
        from transformers import AutoTokenizer

        from agentinputlab.algorithms.acquire import write_ircot_substitution_manifest
        from workers.ircot import PinnedIRCoTNativeQARuntime

        adapter = load_production_benchmark_adapter("hotpotqa.distractor", config["prepared_root"])
        assert dict(adapter.prepared.file_sha256) == spec["dataset_files"]
        example = next(e for e in adapter.examples("validation") if e.example_id == job["case_id"])
        state = state_for(example)
        assert state.state_hash == job["input_state_hash"]
        documents = documents_for(example)
        assert content_hash(documents) == job["visible_source_corpus_sha256"]
        corpus = documents[0].corpus_revision
        retriever = InMemoryBM25Retriever(documents)
        model = job["model"]
        binding = spec["model_binding"]
        backend = RecordingBackend(
            calls=calls,
            model_id=model["id"],
            base_url=model["base_url"],
            deployment_identity={
                "revision": digest(binding["model_json_hashes"]),
                "server": "vllm",
            },
            extra_body=model["extra_body"],
            transport=local_transport,
            timeout_seconds=240,
        )
        if job["final_content_stop_binding"]:
            backend = FinalContentStopBackend(backend)
        tokenizer_receipt = json.loads((assets / "gpt2-tokenizer-closure-v2.json").read_text())
        for name, pin in tokenizer_receipt["file_sha256"].items():
            assert (
                hashlib.sha256((assets / "gpt2-tokenizer" / name).read_bytes()).hexdigest() == pin
            )
        tokenizer = AutoTokenizer.from_pretrained(
            str(assets / "gpt2-tokenizer"), local_files_only=True
        )
        prerequisite = json.loads(Path(config["prerequisite"]).read_text())["package_source_proof"]
        assert {
            name: importlib.metadata.version(name) for name in prerequisite["native_dependencies"]
        } == prerequisite["native_dependencies"]
        runtime = PinnedIRCoTNativeQARuntime(
            source_root=Path(config["source_root"]),
            backend=backend,
            retriever=retriever,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_receipt,
            dependency_fingerprint={
                "versions": prerequisite["native_dependencies"],
                "parser_closure": prerequisite["native_parser_installed_closure_sha256"],
            },
            corpus_revision=corpus,
            index_revision=content_hash(retriever.fingerprint()),
            cache_home=home,
            seed=17,
            max_model_calls=8,
            max_output_tokens=4096,
            per_call_output_tokens=512,
            max_retrieval_calls=8,
        )
        native_run = runtime.run

        def capture(request):
            nonlocal raw
            raw = native_run(request)
            return raw

        runtime.run = capture
        receipt = write_ircot_substitution_manifest(
            runtime, directory / "substitutions.json", accept_substitutions=True
        )
        row["runtime_fingerprint"] = to_jsonable(runtime.fingerprint())
        row["substitution_manifest"] = receipt
        stage = "native_full_QA"
        record = run_composite_intervention(
            create_intervention(
                RELEASE,
                manifest_path=receipt["path"],
                expected_manifest_sha256=receipt["sha256"],
                controls=CONTROLS,
            ),
            state,
            InterventionContext(
                backend=backend,
                retriever=retriever,
                external_runtime=runtime,
                monitor=monitor,
                seed=17,
            ),
        )
        row.update(
            status="answer_returned",
            execution=record,
            prediction={"answer": record["final_answer"]},
        )
    except Exception as exc:
        row.update(
            status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
        )
    row.update(
        calls=calls,
        native_response=to_jsonable(raw),
        native_usage=runtime.last_usage if runtime else {},
        monitor_snapshots=to_jsonable(monitor.list_snapshots()),
        provider_requests=len(calls),
    )
    save(
        output / "model-outcomes" / (job["id"] + ".json"), {**row, "native_scoring_started": False}
    )
    if record is not None:
        try:
            row.update(
                status="scored_terminal",
                metrics=dict(adapter.score(example, QAPrediction(**row["prediction"]))),
            )
        except Exception as exc:
            row.update(
                status="failed_terminal",
                stage="native_QA_scoring",
                error_type=type(exc).__name__,
                error=str(exc),
            )
    assert (
        len(calls) <= 8
        and sum(c.get("response", {}).get("usage", {}).get("output_tokens", 0) for c in calls)
        <= 4096
    )
    save(output / "terminals" / (job["id"] + ".json"), row)


def run(config, output):
    assert len(config["models"]) == 1 and not output.exists()
    model = config["models"][0]
    binding = verify_services([model])[model["id"]]
    adapter = load_production_benchmark_adapter("hotpotqa.distractor", config["prepared_root"])
    examples = sorted(adapter.examples("validation"), key=lambda e: e.example_id)[202:204]
    assert len(examples) == 2
    prompt = (
        Path(config["source_root"])
        / "prompts/hotpotqa/gold_with_2_distractors_context_cot_qa_codex.txt"
    ).read_text()
    assert all(e.example_id not in prompt for e in examples)
    jobs = []
    for e in examples:
        job = dict(
            model=model,
            release=RELEASE,
            benchmark="hotpotqa.distractor",
            case_id=e.example_id,
            seed=17,
            input_state_hash=state_for(e).state_hash,
            visible_source_corpus_sha256=content_hash(documents_for(e)),
            input_projection="question_only_with_external_original_visible_paragraph_index",
            native_controls=CONTROLS,
            common_budget={
                "max_model_calls": 8,
                "max_output_tokens": 4096,
                "max_retrieval_calls": 8,
            },
            per_call_ceiling=512,
            native_generator_default_cap=300,
            final_content_stop_binding=model["id"] == "gpt-oss-20b",
            no_historical_Codex_full_ES_or_published_score_parity=True,
        )
        job["id"] = digest(job)
        jobs.append(job)
    output.mkdir()
    for folder in ("claims", "terminals", "model-outcomes", "jobs", "workers"):
        (output / folder).mkdir()
    save(
        output / "manifest.json",
        dict(
            jobs=jobs,
            dataset_files=dict(adapter.prepared.file_sha256),
            model_binding=binding,
            source_indices=[202, 203],
            native_profile="upstream.hotpotqa.codex_qa.v1",
            sampling_parameters=model["extra_body"],
            input_is_question_only_and_corpus_is_original_visible_distractors=True,
            automatic_retry=False,
            paper_results_changed=False,
        ),
    )
    for job in jobs:
        save(
            output / "jobs" / (job["id"] + ".json"),
            dict(
                job=job,
                config=config,
                output=str(output),
                dataset_files=dict(adapter.prepared.file_sha256),
                model_binding=binding,
            ),
        )

    def execute(job):
        with (output / "workers" / (job["id"] + ".log")).open("x") as log:
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--job",
                    str(output / "jobs" / (job["id"] + ".json")),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            save(output / "workers" / (job["id"] + "-process.json"), {"pid": p.pid})
            code = p.wait()
        assert code == 0, (job["id"], code)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(execute, jobs))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    assert len(rows) == 2
    save(
        output / "complete.json",
        dict(
            expected_positions=2,
            terminal_positions=2,
            scored=sum(r["status"] == "scored_terminal" for r in rows),
            failed=sum(r["status"] == "failed_terminal" for r in rows),
            provider_requests=sum(r["provider_requests"] for r in rows),
            automatic_retry=False,
            publication_ready=False,
        ),
    )
    print(model["id"], "two full native QA cases closed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.job:
        one(args.job)
    elif args.config and args.output:
        run(json.loads(args.config.read_text()), args.output)
    else:
        parser.error("use --job or --config plus --output")
