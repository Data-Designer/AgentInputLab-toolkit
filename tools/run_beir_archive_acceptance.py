"""Test a pinned BEIR archive or bounded BEIR/TREC source-side sample.

The index is a separately named SQLite lexical profile. It is not claimed to
match another BM25 engine's ranking. Official test qrels remain scoring-only.
TREC passage samples retain graded nDCG@10 and threshold-2 AP/Recall@100.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from acceptance_method_bindings import EXTERNAL_REWRITERS, bind_external_rewriter
from run_clariq_acceptance import verify_services
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.reformulate.query2doc import PinnedQueryGymRuntime
from agentinputlab.algorithms.reformulate.rrr import RRRCommonSupportConfig
from agentinputlab.api import create_intervention
from agentinputlab.backends import SQLiteFTS5Retriever
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from agentinputlab.benchmarks import BEIRArchiveAdapter, BEIRSampleAdapter, TRECDLSampleAdapter
from agentinputlab.contracts import ContextBlock
from agentinputlab.contracts.errors import ExternalDependencyError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASES = (
    "baseline.no_op",
    "rewrite.query2doc.querygym_live",
    "rewrite.rrr.common_support_prompt",
    *EXTERNAL_REWRITERS,
)


def preflight_dependencies(releases):
    """Reject missing declared extras before dispatching any case claims."""
    needed = {}
    if "rewrite.query2doc.querygym_live" in releases:
        needed["yaml"] = "querygym"
    if "rewrite.reformer.repo_native_r_only" in releases:
        needed["pandas"] = "reformer"
    for module, extra in needed.items():
        if importlib.util.find_spec(module) is None:
            raise ExternalDependencyError(f"install agentinputlab[{extra}] before dispatch")


class RecordingWorker:
    """Keep the unchanged source-worker request/response and failed usage."""

    def __init__(self, inner, exchanges):
        self.inner, self.exchanges, self.worker_id = inner, exchanges, inner.worker_id

    def call(self, payload):
        row = {"request": to_jsonable(payload), "started": time.time()}
        self.exchanges.append(row)
        try:
            result = self.inner.call(payload)
            row["response"] = to_jsonable(result)
            return result
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            row["resource_receipt"] = to_jsonable(self.inner.last_resource_receipt)

    @property
    def last_resource_receipt(self):
        return self.inner.last_resource_receipt

    def fingerprint(self):
        return self.inner.fingerprint()


def execute(
    config,
    output,
    benchmark,
    archive,
    index_root,
    upstream,
    *,
    sample_root=None,
    sample_manifest_sha256=None,
    releases=RELEASES,
):
    if not releases or len(set(releases)) != len(releases) or not set(releases) <= set(RELEASES):
        raise ValueError("choose unique reviewed retrieval release identities")
    preflight_dependencies(releases)
    bindings = verify_services(config["models"])
    if sample_root is not None:
        adapter_class = TRECDLSampleAdapter if benchmark.startswith("trec_dl.") else BEIRSampleAdapter
        adapter = adapter_class(
            sample_root, manifest_sha256=sample_manifest_sha256, license_acknowledged=True
        )
        if adapter.benchmark_id != benchmark:
            raise ValueError("sample belongs to a different benchmark")
        dataset_revision = adapter.manifest_sha256
    else:
        adapter = BEIRArchiveAdapter(
            benchmark_id=benchmark, source_archive=archive, license_acknowledged=True
        )
        dataset_revision = adapter.lock["sha256"]
    group_names = sorted(
        {example.visible_metadata["forum"] for example in adapter.examples("test")}
    )
    examples = tuple(
        e
        for group in group_names
        for e in sorted(
            (e for e in adapter.examples("test") if e.visible_metadata["forum"] == group),
            key=lambda x: x.example_id,
        )[:2]
    )
    score_preflight = [dict(adapter.score(example, {})) for example in examples]
    retrievers = {}
    for example in examples:
        forum = example.visible_metadata["forum"]
        if forum in retrievers:
            continue
        binding = {**adapter.fingerprint(), "forum": forum}
        path = index_root / (digest(binding) + ".sqlite")
        retrievers[forum] = SQLiteFTS5Retriever.build(
            path, adapter.iter_documents(forum=forum), corpus_identity=binding
        )
    prompt_path = upstream / "upstream-paper/RAG-query-rewriting/generate/inprompts/myprompt.jsonl"
    import hashlib

    raw = prompt_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "1483b80e5b81a26b97641cc2a5c4535c045eb39394d2b88651aade0d4afa5fb4"
    )
    prompts = [json.loads(line) for line in raw.splitlines()]
    rewrite = next(p["prompt"] for p in prompts if p.get("task") == "rewrite" and p.get("pid") == 2)
    reader = next(p["prompt"] for p in prompts if p.get("task") == "step2" and p.get("pid") == 1)
    rrr = RRRCommonSupportConfig(
        rewrite_prompt=rewrite,
        reader_prompt=reader,
        prompt_source_sha256=hashlib.sha256(raw).hexdigest(),
        rewrite_prompt_sha256=content_hash(rewrite),
        reader_prompt_sha256=content_hash(reader),
    )
    for name in ("claims", "terminals"):
        (output / name).mkdir()
    jobs = []
    for model in config["models"]:
        for example in examples:
            for release in releases:
                job = {
                    "model": model,
                    "benchmark": benchmark,
                    "case_id": example.example_id,
                    "release": release,
                    "seed": 17,
                    "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                    "per_call_output_cap": 512,
                    "retriever": retrievers[example.visible_metadata["forum"]].fingerprint(),
                    "method_context_max_characters_per_document": 10000,
                }
                job["id"] = digest(job)
                jobs.append(job)
    expected = config.get("expected_job_ids")
    if expected is not None and set(expected) != {j["id"] for j in jobs}:
        raise ValueError("fixed job identities differ from approved entry-only continuation")
    save(
        output / "manifest.json",
        {
            "jobs": jobs,
            "dataset": adapter.fingerprint(),
            "model_bindings": bindings,
            "score_provider_empty_run_preflight": score_preflight,
            "empty_run_preflight_is_model_performance_evidence": False,
            "automatic_retry": False,
            "paper_results_changed": False,
        },
    )

    def one(job):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model = job["model"]
        release = job["release"]
        example = adapter.example("test", job["case_id"])
        retriever = retrievers[example.visible_metadata["forum"]]
        source = bindings[model["id"]]
        calls, exchanges = [], []
        record = None
        monitor = RunMonitor()
        stage = "intervention"
        started = time.monotonic()
        inner = RecordingBackend(
            calls=calls,
            model_id=model["id"],
            base_url=model["base_url"],
            deployment_identity={"revision": digest(source["model_json_hashes"]), "server": "vllm"},
            timeout_seconds=180,
            extra_body=model.get("extra_body", {}),
            transport=local_transport,
        )
        backend = GPTOSSFinalChannelStopBackend(inner) if model["id"] == "gpt-oss-20b" else inner
        state = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        result = {
            "job": job,
            "input_state_hash": state.state_hash,
            "model_binding": source,
            "package_location": __import__("agentinputlab").__file__,
            "additional_attempts": 1,
            "automatic_retry": False,
        }
        try:
            if release != "baseline.no_op":
                if release in EXTERNAL_REWRITERS:
                    backend = inner
                    method, runtime = bind_external_rewriter(
                        release,
                        upstream=upstream,
                        backend=backend,
                        model=model,
                        dataset_revision=dataset_revision,
                        benchmark=benchmark,
                    )
                    runtime = RecordingWorker(runtime, exchanges)
                    state = replace(
                        state,
                        context=tuple(
                            ContextBlock(
                                text=r.document.text[:10000],
                                block_id=r.document.doc_id,
                                source=r.document.corpus_revision,
                            )
                            for r in retriever.retrieve(state.query, top_k=3)
                        ),
                    )
                elif release == "rewrite.query2doc.querygym_live":
                    method = create_intervention(release)
                    runtime = PinnedQueryGymRuntime(upstream / "upstream/QueryGym", backend)
                else:
                    method = create_intervention(release, config=rrr)
                    runtime = None
                record = InterventionRunner(monitor=monitor).run(
                    method,
                    state,
                    InterventionContext(
                        backend=backend, retriever=retriever, external_runtime=runtime, seed=17
                    ),
                )
                state = record.patch.apply(state)
            stage = "retrieval_and_official_scoring"
            ranked = retriever.retrieve(state.query, top_k=101)
            result.update(
                status="scored_terminal",
                metrics=dict(adapter.score(example, {r.document.doc_id: r.score for r in ranked})),
                ranked_ids=[r.document.doc_id for r in ranked],
            )
        except Exception as exc:
            result.update(
                status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
            )
        result.update(
            seconds=time.monotonic() - started,
            calls=calls,
            worker_exchanges=exchanges,
            monitor_snapshots=to_jsonable(monitor.list_snapshots()),
            intervention=to_jsonable(record) if record else None,
        )
        save(output / "terminals" / (job["id"] + ".json"), result)
        (output.parent / "keep-services-alive").touch()

    def lane(model):
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(one, [j for j in jobs if j["model"]["id"] == model["id"]]))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lane, config["models"]))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    save(
        output / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--archive", type=Path)
    sources.add_argument("--sample-root", type=Path)
    p.add_argument("--sample-manifest-sha256")
    p.add_argument("--index-root", type=Path, required=True)
    p.add_argument("--release", action="append", choices=RELEASES)
    a = p.parse_args()
    a.output.mkdir(parents=True)
    execute(
        json.loads(a.config.read_text()),
        a.output,
        a.benchmark,
        a.archive,
        a.index_root,
        a.upstream_root,
        sample_root=a.sample_root,
        sample_manifest_sha256=a.sample_manifest_sha256,
        releases=tuple(a.release) if a.release else RELEASES,
    )


if __name__ == "__main__":
    main()
