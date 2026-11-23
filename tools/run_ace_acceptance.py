"""Fit native ACE on fixed QASPER train/dev cases, then evaluate held-out QA.

Each backbone fits its own real artifact through the installed native worker.
Training and evaluation have separate usage receipts. Nothing is copied from a
paper artifact, and no ToolTalk/global-test example is used for fitting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.artifacts import ArtifactStore
from agentinputlab.benchmarks import BenchmarkSmokeAssetRegistry, prepare_benchmark
from agentinputlab.benchmarks.preprocess import _qasper_answer, _qasper_context_blocks
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers.base import JsonLineSubprocessWorker
from agentinputlab.recipes import ACEControls, ACEOfflineRecipe, ACEPlaybookCommonSupport
from agentinputlab.runner import InterventionRunner


class RecordedWorker:
    worker_id = "ace-native-real-acceptance"

    def __init__(self, delegate, root):
        self.delegate, self.root = delegate, root

    def fingerprint(self):
        return self.delegate.fingerprint()

    def call(self, payload):
        save(self.root / "worker-request.json", payload)
        try:
            response = self.delegate.call(payload)
        except Exception as exc:
            save(
                self.root / "worker-transport-error.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "consumption": "unknown_if_process_failed_after_start",
                    "resource_receipt": to_jsonable(self.delegate._last_resource_receipt),
                },
            )
            raise
        save(self.root / "worker-response.json", response)
        save(self.root / "worker-resource.json", to_jsonable(self.delegate._last_resource_receipt))
        return response


def execute(config, output):
    done = threading.Event()

    def heartbeat():
        while not done.wait(30):
            (output.parent / "keep-services-alive").touch()

    keepalive = threading.Thread(target=heartbeat, daemon=True)
    keepalive.start()
    try:
        return execute_jobs(config, output)
    finally:
        done.set()
        keepalive.join()


def execute_jobs(config, output):
    # This 10.8 MB pinned archive includes official train/dev/test. Only train/dev
    # members are read for fitting; no test member is loaded into training.
    cache = Path(config["training_cache_dir"])
    acquisition = prepare_benchmark("qasper.v0_3_dev", cache_dir=cache, accept_license_review=True)
    asset = BenchmarkSmokeAssetRegistry.default().get("qasper.dev.official")
    archive = ArtifactStore(cache / "sources").verify(asset.artifact_manifest())
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    selected = {
        c: tuple(sorted(a.examples(s), key=lambda e: e.example_id)[12:14])
        for (c, s), a in zip(QA_CELLS, adapters.values(), strict=True)
    }
    eval_papers = {e.visible_metadata["paper_id"] for e in selected["qasper.v0_3_dev"]}
    with tarfile.open(archive, "r:gz") as source:
        raw = {}
        for split in ("train", "dev"):
            member = source.getmember("qasper-" + split + "-v0.3.json")
            if not member.isfile() or member.size > 64 * 1024**2:
                raise ValueError("QASPER training member size/type differs")
            raw[split] = json.load(source.extractfile(member))

    def first(split):
        for pid, paper in sorted(raw[split].items()):
            if pid in eval_papers:
                continue
            qa = sorted(paper["qas"], key=lambda x: x["question_id"])[0]
            answer, _ = _qasper_answer(qa["answers"][0]["answer"])
            context = "\n\n".join(b["text"] for b in _qasper_context_blocks(paper, paper_id=pid))[
                :20000
            ]
            return {
                "example_id": qa["question_id"],
                "input": {"question": qa["question"], "context": context},
                "expected_output": answer,
                "metadata": {
                    "paper_id": pid,
                    "official_split": split,
                    "context_prefix_characters": 20000,
                },
            }
        raise ValueError("no disjoint native training paper")

    train, valid = first("train"), first("dev")
    assert train["metadata"]["paper_id"] != valid["metadata"]["paper_id"]
    save(
        output / "manifest.json",
        {
            "models": config["models"],
            "training": [train],
            "validation": [valid],
            "evaluation": {c: [e.example_id for e in es] for c, es in selected.items()},
            "training_archive_sha256": asset.sha256,
            "source_acquisition": acquisition,
            "controls": {
                "max_rounds": 1,
                "max_model_calls": 16,
                "max_tokens": 200000,
                "max_cost_usd": 0.0,
                "seed": 17,
            },
            "per_call_output_cap": 512,
            "automatic_retry": False,
            "fit_count": 4,
            "planned_downstream_count": 32,
            "test_gold_used_for_fit": False,
            "paper_results_changed": False,
            "empty_selected_native_artifact": "explicit_no_op_not_fake_learned_content",
        },
    )

    def model_lane(model):
        root = output / model["id"]
        root.mkdir()
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        fp = {
            "model_id": model["id"],
            "revision": digest(binding["model_json_hashes"]),
            "server": "vllm",
        }
        roles = {r: dict(fp) for r in ("target", "reflector", "curator")}
        startup = {
            "api_provider": "openai",
            "max_tokens_per_call": 512,
            "json_mode": True,
            "base_url": model["base_url"],
            "api_key_env": "OPENAI_API_KEY",
            "extra_body": model.get("extra_body", {}),
            "roles": {
                r: {
                    "model": model["id"],
                    "fingerprint": roles[r],
                    "input_cost_per_million": 0.0,
                    "output_cost_per_million": 0.0,
                }
                for r in roles
            },
        }
        worker = RecordedWorker(
            JsonLineSubprocessWorker(
                command=(
                    sys.executable,
                    "-m",
                    "workers.ace.driver",
                    "--checkout",
                    config["ace_checkout"],
                    "--llm-config-json",
                    json.dumps(startup),
                ),
                worker_id="ace-native",
                identity={
                    "upstream_commit": "bcb7cea0504afad6f55fec4845dd4864c9f9eee7",
                    "package_location": __import__("agentinputlab").__file__,
                    "deployment": startup,
                },
                environment={
                    "PATH": os.environ["PATH"],
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "OPENAI_API_KEY": "toolkit-local-not-a-secret",
                },
                timeout_seconds=1200,
                max_stderr_bytes=4 * 1024**2,
            ),
            root,
        )
        started = time.monotonic()
        save(
            root / "fit-claim.json",
            {"started": time.time(), "model_binding": binding, "model_fingerprints": roles},
        )
        try:
            artifact = ACEOfflineRecipe(
                client=worker,
                controls=ACEControls(
                    max_rounds=1, max_model_calls=16, max_tokens=200000, max_cost_usd=0.0, seed=17
                ),
            ).fit(
                (train,),
                validation_examples=(valid,),
                context=InterventionContext(seed=17, run_config={"ace_model_fingerprints": roles}),
            )
            save(root / "artifact.json", artifact.as_dict())
            method = ACEPlaybookCommonSupport(artifact)
            save(
                root / "fit-terminal.json",
                {
                    "status": "native_fit_completed",
                    "usage": dict(artifact.usage),
                    "seconds": time.monotonic() - started,
                    "empty_playbook": not artifact.playbook.bullets,
                },
            )
        except Exception as exc:
            save(
                root / "fit-terminal.json",
                {
                    "status": "failed_terminal",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "seconds": time.monotonic() - started,
                    "downstream_not_started": True,
                    "automatic_retry": False,
                },
            )
            return

        def evaluate(item):
            cell, split, example = item
            case = root / digest({"cell": cell, "case": example.example_id})
            case.mkdir()
            save(
                case / "claim.json",
                {
                    "cell": cell,
                    "case_id": example.example_id,
                    "seed": 17,
                    "artifact_id": artifact.artifact_id,
                },
            )
            state = example.visible_state(
                budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
            )
            calls = []
            backend = RecordingBackend(
                calls=calls,
                model_id=model["id"],
                base_url=model["base_url"],
                deployment_identity=fp,
                extra_body=model.get("extra_body", {}),
                transport=local_transport,
                timeout_seconds=300,
            )
            result = {
                "benchmark": cell,
                "case_id": example.example_id,
                "model": model["id"],
                "release": method.release_id,
                "artifact_id": artifact.artifact_id,
                "fit_usage_charged_once": True,
                "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                "per_call_output_cap": 512,
                "seed": 17,
            }
            try:
                record = InterventionRunner().run(method, state, InterventionContext(seed=17))
                patched = record.patch.apply(state)
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
                prediction = predictor.predict(patched)
                result.update(
                    status="scored_terminal",
                    metrics=dict(adapters[cell].score(example, prediction.prediction)),
                    prediction=to_jsonable(prediction),
                    intervention=to_jsonable(record),
                )
            except Exception as exc:
                result.update(
                    status="failed_terminal", error_type=type(exc).__name__, error=str(exc)
                )
            result["calls"] = calls
            save(case / "terminal.json", result)

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(evaluate, [(c, s, e) for c, s in QA_CELLS for e in selected[c]]))
        (output.parent / "keep-services-alive").touch()

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(model_lane, config["models"]))
    fits = [json.loads(p.read_text()) for p in output.glob("*/fit-terminal.json")]
    rows = [json.loads(p.read_text()) for p in output.glob("*/*/terminal.json")]
    save(
        output / "complete.json",
        {
            "fit_terminal": len(fits),
            "fit_completed": sum(x["status"] == "native_fit_completed" for x in fits),
            "downstream_terminal": len(rows),
            "downstream_scored": sum(x["status"] == "scored_terminal" for x in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    execute(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
