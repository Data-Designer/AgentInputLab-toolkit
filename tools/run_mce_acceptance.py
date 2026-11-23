"""Four isolated native MCE text fits and fixed real paired QA evaluations.

Training uses a small, disjoint official QASPER train/dev mapping. All actual
fit requests are saved before transport. No generated code or hidden test
references enter either optimizer. Scientific failures are never retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.benchmarks import BenchmarkSmokeAssetRegistry
from agentinputlab.benchmarks.preprocess import _qasper_answer, _qasper_context_blocks
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.recipes import (
    MCEExample,
    MCETextArtifact,
    MCETextArtifactCommonSupport,
    MCETextFitControls,
    MCETextOfflineRecipe,
    validate_mce_text_checkout,
)
from agentinputlab.runner import InterventionRunner


class DurableFitBackend(RecordingBackend):
    def __init__(self, *, receipt_root, **kwargs):
        super().__init__(**kwargs)
        self.receipt_root = receipt_root
        self.receipt_root.mkdir()

    def generate(self, request):
        path = self.receipt_root / f"{len(self.calls) + 1:04d}"
        path.mkdir()
        save(path / "request.json", {"request": to_jsonable(request), "started": time.time()})
        try:
            return super().generate(request)
        finally:
            save(path / "response.json", self.calls[-1])


def fit_process(path):
    job = json.loads(path.read_text())
    root, calls = Path(job["root"]), []
    model = job["model"]
    backend = DurableFitBackend(
        receipt_root=root / "provider-attempts",
        calls=calls,
        model_id=model["id"],
        base_url=model["base_url"],
        deployment_identity={
            "revision": digest(job["binding"]["model_json_hashes"]),
            "server": "vllm",
        },
        extra_body=model.get("extra_body", {}),
        transport=local_transport,
        timeout_seconds=180,
    )
    started = time.monotonic()
    try:
        artifact = MCETextOfflineRecipe(controls=MCETextFitControls()).fit(
            checkout=job["checkout"],
            output_root=root / "native-fit",
            backend=backend,
            train=[MCEExample(**value) for value in job["train"]],
            validation=[MCEExample(**value) for value in job["validation"]],
        )
        save(root / "artifact.json", artifact.as_dict())
        result = {
            "status": "fit_completed",
            "artifact_id": artifact.artifact_id,
            "selected_iteration": artifact.selected_iteration,
            "iteration_scores": artifact.iteration_scores,
        }
    except Exception as exc:
        result = {"status": "failed_fit", "error_type": type(exc).__name__, "error": str(exc)}
    save(
        root / "fit-terminal.json",
        {
            **result,
            "calls": calls,
            "seconds": time.monotonic() - started,
            "additional_fit_attempts": 1,
            "automatic_retry": False,
            "seed_mode": "17_plus_logical_call_index_v1",
        },
    )


def prepare_split(source, excluded_papers):
    asset = BenchmarkSmokeAssetRegistry.default().get("qasper.dev.official")
    if source.stat().st_size != asset.size_bytes:
        raise ValueError("QASPER native archive size differs")
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != asset.sha256:
            raise ValueError("QASPER native archive checksum differs")
    with tarfile.open(source, "r:gz") as archive:
        data = {}
        for split in ("train", "dev"):
            member = archive.getmember("qasper-" + split + "-v0.3.json")
            if not member.isfile() or member.size > 64 * 1024**2:
                raise ValueError("QASPER member type/size differs")
            data[split] = json.load(archive.extractfile(member))
    used = set(excluded_papers)

    def choose(source_split, split, count):
        rows = []
        for pid, paper in sorted(data[source_split].items()):
            if pid in used:
                continue
            question = sorted(paper["qas"], key=lambda q: q["question_id"])[0]
            answer, _ = _qasper_answer(question["answers"][0]["answer"])
            rows.append(
                to_jsonable(
                    MCEExample(
                        example_id=question["question_id"],
                        split=split,
                        expected_output=answer,
                        input={
                            "question": question["question"],
                            "context": "\n\n".join(
                                b["text"] for b in _qasper_context_blocks(paper, paper_id=pid)
                            )[:20000],
                        },
                        metadata={
                            "paper_id": pid,
                            "official_split": source_split,
                            "source_archive_sha256": asset.sha256,
                        },
                    )
                )
            )
            used.add(pid)
            if len(rows) == count:
                return rows
        raise ValueError("insufficient disjoint native QASPER examples")

    return choose("train", "train", 2), choose("dev", "validation", 1), asset.sha256


def execute(config, output, checkout):
    checkout, source_hashes = validate_mce_text_checkout(checkout)
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    selected = {
        c: tuple(sorted(adapters[c].examples(s), key=lambda e: e.example_id)[24:26])
        for c, s in QA_CELLS
    }
    train, validation, archive_sha = prepare_split(
        Path(config["training_archive"]),
        {e.visible_metadata["paper_id"] for e in selected["qasper.v0_3_dev"]},
    )
    save(
        output / "manifest.json",
        {
            "models": config["models"],
            "train": train,
            "validation": validation,
            "source_hashes": source_hashes,
            "source_archive_sha256": archive_sha,
            "evaluation_fixed_indices": [24, 25],
            "evaluation": {c: [e.example_id for e in es] for c, es in selected.items()},
            "data_bindings": {c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
            "expected_fits": 4,
            "planned_downstream": 64,
            "downstream_releases": ["baseline.no_op", "recipe.mce.text_artifact_eval"],
            "fit_controls": to_jsonable(MCETextFitControls()),
            "fit_cost_separate_from_downstream": True,
            "seed": 17,
            "per_call_output_cap": 512,
            "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
            "validation_reference_never_in_model_prompt": True,
            "test_gold_used_for_fit": False,
            "generated_code_executed": False,
            "paper_results_changed": False,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "wheel_sha256": hashlib.sha256(
                Path(config["installed_wheel"]).read_bytes()
            ).hexdigest(),
            "automatic_retry": False,
        },
    )
    stop = threading.Event()

    def keepalive():
        while not stop.wait(30):
            (output.parent / "keep-services-alive").touch()

    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()

    def lane(model):
        root = output / model["id"]
        root.mkdir()
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        request = root / "fit-job.json"
        save(
            request,
            {
                "root": str(root),
                "checkout": str(checkout),
                "model": model,
                "binding": binding,
                "train": train,
                "validation": validation,
            },
        )
        # No real credentials or inherited provider/SDK settings. The native
        # client constructed but unused by the four-seam text profile cannot
        # accidentally contact any serving model or external provider.
        env = {
            "PATH": os.environ["PATH"],
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
            "CUDA_VISIBLE_DEVICES": "",
            "OPENAI_API_KEY": "dummy-unused-native-client",
            "OPENAI_API_BASE": "http://127.0.0.1:1/v1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
        }
        with (root / "fit-process.log").open("x") as log:
            try:
                process = subprocess.run(
                    (sys.executable, str(Path(__file__).resolve()), "--fit-job", str(request)),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=960,
                    check=False,
                )
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                exit_code = "deadline_consumption_preserved_in_provider_receipts"
        save(root / "fit-process-exit.json", {"exit_code": exit_code})
        if not (root / "artifact.json").exists():
            save(
                root / "attention.json",
                {"fit_not_completed": True, "evaluation_unstarted": True, "automatic_retry": False},
            )
            return
        artifact = MCETextArtifact(**json.loads((root / "artifact.json").read_text()))
        method = MCETextArtifactCommonSupport(artifact)
        fp = {"revision": digest(binding["model_json_hashes"]), "server": "vllm"}

        def evaluate(item):
            cell, split, example, release = item
            job = {
                "model": model,
                "benchmark": cell,
                "split": split,
                "case_id": example.example_id,
                "release": release,
                "seed": 17,
                "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                "per_call_output_cap": 512,
                "artifact_id": artifact.artifact_id if release != "baseline.no_op" else None,
            }
            target = root / digest(job)
            target.mkdir()
            save(target / "claim.json", {"job": job, "started": time.time()})
            state = example.visible_state(budget=ExecutionBudget(**job["budget"]))
            calls, record = [], None
            result = {
                "job": job,
                "input_state_hash": state.state_hash,
                "model_binding": binding,
                "package_location": __import__("agentinputlab").__file__,
            }
            started = time.monotonic()
            try:
                if release != "baseline.no_op":
                    record = InterventionRunner().run(method, state, InterventionContext(seed=17))
                    state = record.patch.apply(state)
                backend = RecordingBackend(
                    calls=calls,
                    model_id=model["id"],
                    base_url=model["base_url"],
                    deployment_identity=fp,
                    extra_body=model.get("extra_body", {}),
                    transport=local_transport,
                    timeout_seconds=300,
                )
                predictor = ProductionPredictorBuilder().build(
                    StaticPredictorBuildRequest(
                        benchmark_id=cell,
                        split=split,
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
                result.update(
                    status="scored_terminal",
                    prediction=to_jsonable(prediction),
                    metrics=dict(adapters[cell].score(example, prediction.prediction)),
                )
            except Exception as exc:
                result.update(
                    status="failed_terminal", error_type=type(exc).__name__, error=str(exc)
                )
            save(
                target / "terminal.json",
                {
                    **result,
                    "calls": calls,
                    "intervention": to_jsonable(record) if record else None,
                    "seconds": time.monotonic() - started,
                    "automatic_retry": False,
                },
            )

        work = [
            (c, s, e, release)
            for c, s in QA_CELLS
            for e in selected[c]
            for release in ("baseline.no_op", method.release_id)
        ]
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(evaluate, work))

    try:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lane, config["models"]))
    finally:
        stop.set()
        thread.join()
    fits = [json.loads(p.read_text()) for p in output.glob("*/fit-terminal.json")]
    rows = [json.loads(p.read_text()) for p in output.glob("*/*/terminal.json")]
    save(
        output / "complete.json",
        {
            "expected_fits": 4,
            "fit_terminals": len(fits),
            "fit_completed": sum(r["status"] == "fit_completed" for r in fits),
            "planned_downstream": 64,
            "downstream_terminal": len(rows),
            "downstream_scored": sum(r["status"] == "scored_terminal" for r in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-job", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args()
    if args.fit_job:
        fit_process(args.fit_job)
    elif args.config and args.output and args.checkout:
        args.output.mkdir(parents=True)
        execute(json.loads(args.config.read_text()), args.output, args.checkout)
    else:
        parser.error("supply --config --output --checkout, or --fit-job")


if __name__ == "__main__":
    main()
