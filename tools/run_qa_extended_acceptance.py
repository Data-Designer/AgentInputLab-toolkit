"""Real QA compatibility cases for additional rewrite/acquire/curate profiles.

Visible passages are the only retrieval corpus. Embedding and deterministic
compression preparation can be shared; each target backbone executes its own
method calls and structured downstream prediction. No reference labels enter
the helper models or target prompts.
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

from acceptance_method_bindings import EXTERNAL_REWRITERS, bind_external_rewriter
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.acquire.hyde import HyDEConfig
from agentinputlab.api import create_intervention
from agentinputlab.backends import (
    Document,
    InMemoryBM25Retriever,
    SentenceTransformerDenseRetriever,
)
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from agentinputlab.backends.retrieval.dense import RetrieverProfile
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

HYDE = "acquire.hyde.paper_reimplementation"
COMPRESSOR = "curate.llmlingua2.xlm_roberta_official"
RELEASES = (*EXTERNAL_REWRITERS, HYDE, COMPRESSOR)
_ENCODER = None
_ENCODER_LOCK = threading.Lock()


class MountedEncoder:
    def __init__(self, identity):
        self.identity = identity

    def fingerprint(self):
        return self.identity

    def encode(self, texts, **kwargs):
        with _ENCODER_LOCK:
            return _ENCODER.encode(texts, **kwargs)


def execute(config, output, upstream):
    import torch
    from sentence_transformers import SentenceTransformer

    global _ENCODER
    torch.set_num_threads(2)
    mount = Path(config["embedder_path"])
    manifest = json.loads((mount / "agentinputlab-manifest.json").read_text())
    # Verify exactly the files loaded by SentenceTransformers; no remote load.
    loaded = {
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "1_Pooling/config.json",
    }
    files = {r["path"]: r for r in manifest["files"]}
    for name in loaded:
        f = mount / name
        assert f.stat().st_size == files[name]["size"]
        with f.open("rb") as h:
            observed = hashlib.file_digest(h, "sha256").hexdigest()
        assert observed == files[name]["sha256"], name
    helper_identity = {
        "model": manifest["repository"],
        "revision": manifest["revision"],
        "files": {n: files[n]["sha256"] for n in sorted(loaded)},
        "device": "cpu",
        "loader": "sentence_transformers",
    }
    _ENCODER = SentenceTransformer(
        str(mount), device="cpu", local_files_only=True, trust_remote_code=False
    )
    encoder = MountedEncoder(helper_identity)
    profile = RetrieverProfile(
        profile_id="operator.e5.multilingual.visible_passages",
        backend="sentence_transformers_dense",
        model_id=manifest["repository"],
        model_revision=manifest["revision"],
        query_prefix="query: ",
        document_prefix="passage: ",
        expected_dimensions=1024,
        license_spdx="MIT",
    )
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    examples = {
        c: tuple(sorted(a.examples(s), key=lambda x: x.example_id)[8:10])
        for (c, s), a in zip(QA_CELLS, adapters.values(), strict=False)
    }
    dense = {}
    sparse = {}
    compressed = {}
    for name in ("claims", "terminals", "preparation"):
        (output / name).mkdir()
    for cell, _split in QA_CELLS:
        for example in examples[cell]:
            state = example.visible_state(
                budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
            )
            docs = tuple(
                Document(
                    doc_id=b.block_id or str(i),
                    text=b.text,
                    corpus_revision=adapters[cell].prepared.source_revision,
                )
                for i, b in enumerate(state.context)
            )
            key = (cell, example.example_id)
            sparse[key] = InMemoryBM25Retriever(documents=docs)
            dense[key] = SentenceTransformerDenseRetriever(
                docs, profile=profile, encoder=encoder, device="cpu"
            )
            dense[key].build_index()
    jobs = []
    for model in config["models"]:
        for cell, split in QA_CELLS:
            for example in examples[cell]:
                for release in RELEASES:
                    j = {
                        "model": model,
                        "benchmark": cell,
                        "split": split,
                        "case_id": example.example_id,
                        "release": release,
                        "seed": 17,
                        "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                        "per_call_output": 512,
                        "scope": "visible_passage_corpus_controlled_compatibility",
                        "hyde_hypotheses": 4,
                    }
                    j["id"] = digest(j)
                    jobs.append(j)
    save(
        output / "manifest.json",
        {
            "jobs": jobs,
            "helper": helper_identity,
            "automatic_retry": False,
            "paper_results_changed": False,
            "reference_exposed": False,
        },
    )
    helper = create_intervention(
        COMPRESSOR,
        model_name=config["compressor_path"],
        device_map="cpu",
        model_config={"local_files_only": True},
    )

    def one(job):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model = job["model"]
        cell = job["benchmark"]
        key = (cell, job["case_id"])
        release = job["release"]
        a = adapters[cell]
        ex = next(e for e in examples[cell] if e.example_id == job["case_id"])
        state = ex.visible_state(budget=ExecutionBudget(**job["budget"]))
        source = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        calls = []
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
        result = {
            "job": job,
            "package_location": __import__("agentinputlab").__file__,
            "model_binding": source,
            "input_state_hash": state.state_hash,
            "prepared_file_hashes": dict(a.prepared.file_sha256),
            "automatic_retry": False,
            "additional_attempts": 1,
        }
        try:
            runtime = None
            if release in EXTERNAL_REWRITERS:
                backend = inner
                method, runtime = bind_external_rewriter(
                    release,
                    upstream=upstream,
                    backend=backend,
                    model=model,
                    dataset_revision=a.prepared.source_revision,
                    benchmark=cell,
                )
            elif release == HYDE:
                method = create_intervention(
                    release, config=HyDEConfig(num_hypotheses=4, max_tokens=512)
                )
            else:
                method = helper
            if release == COMPRESSOR:
                prepared = compressed[key]
                if isinstance(prepared, Exception):
                    raise prepared
                record = prepared
                result["shared_preparation_receipt"] = str(
                    output / "preparation" / (digest(key) + ".json")
                )
            else:
                record = InterventionRunner(monitor=monitor).run(
                    method,
                    state,
                    InterventionContext(
                        backend=backend,
                        retriever=sparse[key],
                        dense_embedder=dense[key],
                        vector_searcher=dense[key],
                        external_runtime=runtime,
                        seed=17,
                    ),
                )
            state = record.patch.apply(state)
            stage = "downstream_structured_generation"
            if record.usage.model_calls >= 8 or record.usage.output_tokens >= 4096:
                raise RuntimeError("shared budget exhausted")
            predictor = ProductionPredictorBuilder().build(
                StaticPredictorBuildRequest(
                    benchmark_id=cell,
                    split=job["split"],
                    model_slot=model["id"],
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
            prediction = predictor.predict(
                replace(
                    state,
                    budget=replace(
                        state.budget,
                        max_model_calls=8 - record.usage.model_calls,
                        max_output_tokens=4096 - record.usage.output_tokens,
                    ),
                )
            )
            stage = "official_scoring"
            result.update(
                status="scored_terminal",
                metrics=dict(a.score(ex, prediction.prediction)),
                prediction=to_jsonable(prediction),
            )
        except Exception as exc:
            result.update(
                status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
            )
        result.update(
            seconds=time.monotonic() - started,
            calls=calls,
            monitor_snapshots=to_jsonable(monitor.list_snapshots()),
            intervention=to_jsonable(record) if record else None,
        )
        save(output / "terminals" / (job["id"] + ".json"), result)
        (output.parent / "keep-services-alive").touch()

    pools = {m["id"]: ThreadPoolExecutor(3) for m in config["models"]}
    futures = []
    try:
        for job in jobs:
            if job["release"] != COMPRESSOR:
                futures.append(pools[job["model"]["id"]].submit(one, job))
        # CPU preparation overlaps genuine target generation already queued.
        for cell, _split in QA_CELLS:
            for ex in examples[cell]:
                key = (cell, ex.example_id)
                state = ex.visible_state(
                    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
                )
                try:
                    prepared = InterventionRunner().run(helper, state, InterventionContext(seed=17))
                except Exception as exc:
                    prepared = exc
                compressed[key] = prepared
                save(
                    output / "preparation" / (digest(key) + ".json"),
                    {
                        "key": key,
                        "reference_exposed": False,
                        "record": to_jsonable(prepared)
                        if not isinstance(prepared, Exception)
                        else None,
                        "error": str(prepared) if isinstance(prepared, Exception) else None,
                    },
                )
                for job in jobs:
                    if job["release"] == COMPRESSOR and (job["benchmark"], job["case_id"]) == key:
                        futures.append(pools[job["model"]["id"]].submit(one, job))
        for f in futures:
            f.result()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
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
    a = p.parse_args()
    a.output.mkdir(parents=True)
    execute(json.loads(a.config.read_text()), a.output, a.upstream_root)


if __name__ == "__main__":
    main()
