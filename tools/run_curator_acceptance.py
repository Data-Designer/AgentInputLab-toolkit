"""Pinned curator preparation plus independent four-backbone QA/LongBench runs.

Preparation is shared only when it is independent of the evaluated backbone
and bound to an identical visible state. Helper inference and target inference
costs have separate receipts. No reference answer is given to either model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.api import create_intervention
from agentinputlab.backends import GenerationRequest
from agentinputlab.benchmarks import LongBenchAdapter
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.runner import InterventionRunner

LONG = (
    "longbench.qasper",
    "longbench.multifieldqa_en",
    "longbench.govreport",
    "longbench.qmsum",
    "longbench.passage_retrieval_en",
    "longbench.repobench_p",
)
RELEASE = "curate.longllmlingua.official"


def execute(config, output):
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(config.get("helper_cpu_threads", 2))
    torch.manual_seed(17)
    mount = Path(config["compressor_snapshot"])
    source_lock = Path(config["compressor_source_lock"])
    lock = json.loads(source_lock.read_text())
    assert lock["revision"] == "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
    for name, expected in lock["configuration_files"].items():
        assert hashlib.sha256((mount / name).read_bytes()).hexdigest() == expected
    # Large immutable shared weights were fully hashed once immediately before
    # this phase. Recheck resolved paths/size, and retain that source receipt.
    for row in lock["weights"]:
        p = mount / row["name"]
        assert (
            str(p.resolve()) == row["shared_read_only_source"] and p.stat().st_size == row["size"]
        )
    source_hash = hashlib.sha256(source_lock.read_bytes()).hexdigest()
    helper_device = config.get("helper_device", "cuda:0")
    if helper_device not in ("cpu", "cuda:0"):
        raise ValueError("helper_device must be cpu or explicitly authorized cuda:0")
    free = total = None
    if helper_device == "cuda:0":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one explicitly authorized helper GPU is required")
        free, total = torch.cuda.mem_get_info()
        if free < 18 * 1024**3:
            raise RuntimeError(
                "helper requires18GiB measured free capacity; no other process is signaled"
            )
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    splits = dict(QA_CELLS)
    assets = Path(config["longbench_source_root"])
    for cell in LONG:
        adapters[cell] = LongBenchAdapter(
            benchmark_id=cell,
            data_zip=assets / "data.zip",
            scorer_path=assets / "metrics.py",
            prompts_path=assets / "dataset2prompt.json",
        )
        splits[cell] = "test"
    examples = {
        c: tuple(
            sorted(a.examples(splits[c]), key=lambda e: e.example_id)[
                config.get("case_offset", 10) : config.get("case_offset", 10) + 2
            ]
        )
        if c not in LONG
        else tuple(sorted(a.examples("test"), key=lambda e: e.example_id)[:2])
        for c, a in adapters.items()
    }
    jobs = []
    for model in config["models"]:
        for cell, values in examples.items():
            for example in values:
                j = {
                    "model": model,
                    "benchmark": cell,
                    "split": splits[cell],
                    "case_id": example.example_id,
                    "release": RELEASE,
                    "helper_source_lock_sha256": source_hash,
                    "helper_device": helper_device,
                    "helper_physical_gpu": config.get("helper_physical_gpu"),
                    "helper_dtype": "float16",
                    "seed": 17,
                    "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                    "per_call_output_cap": 512,
                    "longbench_max_input_tokens": 24000,
                }
                j["id"] = digest(j)
                jobs.append(j)
    for n in ("claims", "terminals", "preparation"):
        (output / n).mkdir()
    save(
        output / "manifest.json",
        {
            "jobs": jobs,
            "automatic_retry": False,
            "paper_results_changed": False,
            "helper_source_lock": lock,
            "helper_source_lock_sha256": source_hash,
            "measured_helper_free_bytes": free,
            "helper_total_bytes": total,
            "reference_exposed": False,
            "preparation_reuse": "exact_visible_state_backbone_independent_native_compression",
            "target_and_helper_usage_separate": True,
        },
    )
    helper = create_intervention(
        RELEASE,
        model_name=str(mount),
        model_revision=lock["revision"],
        device_map=helper_device,
        model_config={"local_files_only": True, "torch_dtype": "float16"},
    )
    tokenizers = {
        m["id"]: AutoTokenizer.from_pretrained(
            m["tokenizer_path"], local_files_only=True, trust_remote_code=False
        )
        for m in config["models"]
    }
    mutex = {m["id"]: threading.Lock() for m in config["models"]}

    def one(job, record, receipt):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model = job["model"]
        cell = job["benchmark"]
        adapter = adapters[cell]
        ex = next(e for e in examples[cell] if e.example_id == job["case_id"])
        state = ex.visible_state(budget=ExecutionBudget(**job["budget"]))
        calls = []
        started = time.monotonic()
        stage = "apply_native_curator_patch"
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        result = {
            "job": job,
            "model_binding": binding,
            "package_location": __import__("agentinputlab").__file__,
            "shared_preparation_receipt": str(receipt),
            "input_state_hash": state.state_hash,
            "additional_attempts": 1,
            "automatic_retry": False,
        }
        backend = RecordingBackend(
            calls=calls,
            model_id=model["id"],
            base_url=model["base_url"],
            timeout_seconds=300,
            deployment_identity={
                "revision": digest(binding["model_json_hashes"]),
                "server": "vllm",
            },
            extra_body=model.get("extra_body", {}),
            transport=local_transport,
        )
        try:
            state = record.patch.apply(state)
            if cell in LONG:
                stage = "native_prompt_and_middle_truncation"
                prompt = adapter.render(state)
                with mutex[model["id"]]:
                    tokenizer = tokenizers[model["id"]]
                    tokens = tokenizer.encode(prompt, add_special_tokens=False)
                    before = len(tokens)
                    if before > 24000:
                        prompt = tokenizer.decode(
                            tokens[:12000], skip_special_tokens=True
                        ) + tokenizer.decode(tokens[-12000:], skip_special_tokens=True)
                result["target_prompt"] = {
                    "tokens_before": before,
                    "max_input_tokens": 24000,
                    "middle_truncation": before > 24000,
                    "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
                stage = "target_generation"
                response = backend.generate(
                    GenerationRequest(prompt=prompt, max_tokens=512, seed=17)
                )
                stage = "native_scoring"
                metrics = dict(adapter.score(ex, response.text))
                result["prediction"] = response.text
            else:
                stage = "structured_target_generation"
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
                prediction = predictor.predict(state)
                stage = "native_scoring"
                metrics = dict(adapter.score(ex, prediction.prediction))
                result["prediction"] = to_jsonable(prediction)
            result.update(status="scored_terminal", metrics=metrics)
        except Exception as exc:
            result.update(
                status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
            )
        result.update(seconds=time.monotonic() - started, calls=calls, model_calls=len(calls))
        save(output / "terminals" / (job["id"] + ".json"), result)
        (output.parent / "keep-services-alive").touch()

    pools = {m["id"]: ThreadPoolExecutor(3) for m in config["models"]}
    futures = []
    prepared = 0
    try:
        for cell, values in examples.items():
            for ex in values:
                state = ex.visible_state(
                    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
                )
                receipt = (
                    output / "preparation" / (digest({"cell": cell, "id": ex.example_id}) + ".json")
                )
                started = time.monotonic()
                if helper_device != "cpu":
                    torch.cuda.reset_peak_memory_stats()
                try:
                    record = InterventionRunner().run(helper, state, InterventionContext(seed=17))
                except Exception as exc:
                    save(
                        receipt,
                        {
                            "status": "failed_preparation",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "seconds": time.monotonic() - started,
                            "peak_allocated_bytes": (
                                torch.cuda.max_memory_allocated() if helper_device != "cpu" else 0
                            ),
                            "helper_source_lock_sha256": source_hash,
                            "automatic_retry": False,
                            "reference_exposed": False,
                        },
                    )
                    save(
                        output / "attention.json",
                        {
                            "failed_preparation": str(receipt),
                            "completed_preparations": prepared,
                            "other_cases_unstarted": True,
                            "reason": "inspect native helper stage before any repeat",
                        },
                    )
                    return
                save(
                    receipt,
                    {
                        "status": "native_compressed",
                        "record": to_jsonable(record),
                        "seconds": time.monotonic() - started,
                        "peak_allocated_bytes": (
                            torch.cuda.max_memory_allocated() if helper_device != "cpu" else 0
                        ),
                        "helper_source_lock_sha256": source_hash,
                        "reference_exposed": False,
                    },
                )
                prepared += 1
                for job in jobs:
                    if (job["benchmark"], job["case_id"]) == (cell, ex.example_id):
                        futures.append(pools[job["model"]["id"]].submit(one, job, record, receipt))
                (output.parent / "keep-services-alive").touch()
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
            "native_preparation_count": prepared,
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
