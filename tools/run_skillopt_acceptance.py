"""Native SkillOpt fitting on disjoint small QASPER splits and frozen QA checks.

Four isolated native CLI processes fit independent model-bound artifacts.
Every actual provider attempt, including upstream/SDK retries, passes a local
metered relay. Fitting cost is separate from the downstream 4096/8 budgets.
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

from acceptance_fit_transport import BoundedFitRelay
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
    SkillOptArtifact,
    SkillOptControls,
    SkillOptOfflineRecipe,
    SkillOptSkillCommonSupport,
)
from agentinputlab.recipes.skillopt import validate_skillopt_checkout
from agentinputlab.runner import InterventionRunner


def write_config(root, checkout, split_root, model):
    """Use native SearchQA and validation gate; no algorithm replacements."""
    config = root / "skillopt.yaml"
    text = f"""_base_: {checkout / "configs/_base_/default.yaml"}
model:
  backend: openai_compatible
  optimizer_backend: openai_compatible
  target_backend: openai_compatible
  optimizer: {model}
  target: {model}
  reasoning_effort: ''
  rewrite_reasoning_effort: ''
  rewrite_max_completion_tokens: 512
train:
  num_epochs: 1
  train_size: 2
  batch_size: 2
  accumulation: 1
  seed: 17
gradient:
  minibatch_size: 2
  merge_batch_size: 1
  analyst_workers: 1
  max_analyst_rounds: 1
  failure_only: false
optimizer:
  learning_rate: 4
  min_learning_rate: 1
  lr_scheduler: constant
  lr_control_mode: fixed
  skill_update_mode: patch
  use_slow_update: false
  use_meta_skill: false
evaluation:
  use_gate: true
  sel_env_num: 1
  test_env_num: 0
  eval_test: false
env:
  name: searchqa
  skill_init: {checkout / "skillopt/envs/searchqa/skills/initial.md"}
  split_mode: split_dir
  split_dir: {split_root}
  max_turns: 1
  max_completion_tokens: 512
  exec_timeout: 180
  workers: 1
  limit: 0
"""
    with config.open("x") as f:
        f.write(text)
    return config


def fit_process(job_path):
    job = json.loads(job_path.read_text())
    root = Path(job["root"])
    config = write_config(root, Path(job["checkout"]), Path(job["split_root"]), job["model"]["id"])
    controls = SkillOptControls(
        num_epochs=1, batch_size=2, learning_rate=4, seed=17, timeout_seconds=900
    )
    environment = {
        "OPENAI_COMPATIBLE_BASE_URL": job["relay_base_url"],
        "OPENAI_COMPATIBLE_API_KEY": "dummy-local-not-a-secret",
        "OPENAI_COMPATIBLE_MODEL": job["model"]["id"],
        "TARGET_OPENAI_COMPATIBLE_TEMPERATURE": "0",
        "TARGET_OPENAI_COMPATIBLE_MAX_TOKENS": "512",
        "OPTIMIZER_OPENAI_COMPATIBLE_TEMPERATURE": "0",
        "OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS": "512",
        "OPENAI_COMPATIBLE_TIMEOUT_SECONDS": "180",
    }
    started = time.monotonic()
    try:
        artifact = SkillOptOfflineRecipe(controls=controls).fit(
            checkout=job["checkout"],
            config_path=config,
            output_root=root / "native-fit",
            context=InterventionContext(
                seed=17,
                run_config={
                    "skillopt_model_fingerprints": {
                        "target": job["model_fingerprint"],
                        "optimizer": job["model_fingerprint"],
                    }
                },
            ),
            environment=environment,
        )
        save(root / "artifact.json", artifact.as_dict())
        terminal = {
            "status": "fit_completed",
            "artifact_id": artifact.artifact_id,
            "history": json.loads((root / "native-fit/history.json").read_text()),
            "selected_skill_may_equal_initial": True,
        }
    except Exception as exc:
        terminal = {"status": "failed_fit", "error_type": type(exc).__name__, "error": str(exc)}
    save(
        root / "fit-terminal.json",
        {
            **terminal,
            "seconds": time.monotonic() - started,
            "additional_fit_attempts": 1,
            "automatic_retry": False,
        },
    )


def execute(config, output, checkout):
    from transformers import AutoTokenizer

    checkout = validate_skillopt_checkout(checkout)
    adapters = {
        c: load_production_benchmark_adapter(c, config["prepared_root"]) for c, _ in QA_CELLS
    }
    selected = {
        c: tuple(sorted(adapters[c].examples(s), key=lambda x: x.example_id)[18:20])
        for c, s in QA_CELLS
    }
    eval_papers = {e.visible_metadata["paper_id"] for e in selected["qasper.v0_3_dev"]}
    source = Path(config["training_archive"])
    asset = BenchmarkSmokeAssetRegistry.default().get("qasper.dev.official")
    with source.open("rb") as handle:
        observed = hashlib.file_digest(handle, "sha256").hexdigest()
    if source.stat().st_size != asset.size_bytes or observed != asset.sha256:
        raise ValueError("fixed QASPER training archive source differs")
    with tarfile.open(source, "r:gz") as archive:
        official = {}
        for split in ("train", "dev"):
            member = archive.getmember("qasper-" + split + "-v0.3.json")
            if not member.isfile() or member.size > 64 * 1024**2:
                raise ValueError("native QASPER member type/size differs")
            official[split] = json.load(archive.extractfile(member))
    seen = set(eval_papers)

    def choose(split, count):
        values = []
        for pid, paper in sorted(official[split].items()):
            if pid in seen:
                continue
            question = sorted(paper["qas"], key=lambda q: q["question_id"])[0]
            answer, _ = _qasper_answer(question["answers"][0]["answer"])
            context = "\n\n".join(b["text"] for b in _qasper_context_blocks(paper, paper_id=pid))[
                :20000
            ]
            values.append(
                {
                    "id": question["question_id"],
                    "question": question["question"],
                    "context": context,
                    "answers": [answer],
                    "paper_id": pid,
                    "official_split": split,
                }
            )
            seen.add(pid)
            if len(values) == count:
                return values
        raise ValueError("not enough paper-disjoint native development examples")

    train, valid = choose("train", 2), choose("dev", 1)
    split_root = output / "native-split"
    split_root.mkdir()
    hashes = {}
    for name, values in (("train", train), ("val", valid), ("test", valid)):
        folder = split_root / name
        folder.mkdir()
        save(folder / "items.json", values)
        hashes[name] = hashlib.sha256((folder / "items.json").read_bytes()).hexdigest()
    save(
        output / "manifest.json",
        {
            "models": config["models"],
            "training": train,
            "validation": valid,
            "evaluation": {c: [e.example_id for e in values] for c, values in selected.items()},
            "source_archive_sha256": observed,
            "native_split_sha256": hashes,
            "profile": "native_skillopt_searchqa_on_official_qasper_development_mapping",
            "source_context_prefix_characters": 20000,
            "native_rollout_context_character_cap": 6000,
            "test_tree_mirrors_validation_for_native_loader_only": True,
            "native_test_evaluation_disabled": True,
            "global_test_tooltalk_used_for_training": False,
            "paper_disjoint": True,
            "actual_fit_count": 4,
            "planned_downstream_count": 32,
            "offline_fit_max_physical_calls_each": 32,
            "offline_fit_max_total_tokens_each": 200000,
            "seed": 17,
            "per_call_output_cap": 512,
            "automatic_retry": False,
            "native_internal_retries_observed_and_share_aggregate_caps": True,
            "paper_results_changed": False,
            "acceptance_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fit_transport_sha256": hashlib.sha256(
                (Path(__file__).parent / "acceptance_fit_transport.py").read_bytes()
            ).hexdigest(),
            "installation_receipt": config["installation_receipt"],
            "installation_receipt_sha256": hashlib.sha256(
                Path(config["installation_receipt"]).read_bytes()
            ).hexdigest(),
        },
    )
    stop = threading.Event()

    def keepalive():
        while not stop.wait(30):
            (output.parent / "keep-services-alive").touch()

    keep = threading.Thread(target=keepalive, daemon=True)
    keep.start()

    def lane(model):
        root = output / model["id"]
        root.mkdir()
        binding = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        fingerprint = {
            "model_id": model["id"],
            "revision": digest(binding["model_json_hashes"]),
            "server": "vllm",
        }
        tokenizer = AutoTokenizer.from_pretrained(
            model["tokenizer_path"], local_files_only=True, trust_remote_code=False
        )
        save(root / "fit-claim.json", {"started": time.time(), "model_binding": binding})
        with BoundedFitRelay(
            model=model, root=root / "provider-attempts", tokenizer=tokenizer
        ) as relay:
            request = root / "fit-job.json"
            save(
                request,
                {
                    "root": str(root),
                    "checkout": str(checkout),
                    "split_root": str(split_root),
                    "model": model,
                    "model_fingerprint": fingerprint,
                    "relay_base_url": relay.base_url,
                },
            )
            environment = {
                "PATH": os.environ["PATH"],
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHON_DOTENV_DISABLED": "1",
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "LANG": "C.UTF-8",
                "OMP_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
            }
            with (root / "fit-process.log").open("x") as log:
                try:
                    process = subprocess.run(
                        (sys.executable, str(Path(__file__).resolve()), "--fit-job", str(request)),
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=960,
                        check=False,
                    )
                    exit_code = process.returncode
                except subprocess.TimeoutExpired:
                    exit_code = "timeout_consumption_preserved_in_provider_receipts"
            save(
                root / "fit-transport.json",
                {
                    "child_exit": exit_code,
                    "summary": relay.summary(),
                    "actual_model_identity": fingerprint,
                    "paper_artifact_reused": False,
                },
            )
        if not (root / "artifact.json").exists():
            save(
                root / "attention.json",
                {
                    "native_fit_not_completed": True,
                    "evaluation_unstarted": True,
                    "provider_attempts": relay.summary(),
                    "child_exit": exit_code,
                },
            )
            return
        artifact = SkillOptArtifact(**json.loads((root / "artifact.json").read_text()))
        method = SkillOptSkillCommonSupport(artifact=artifact)

        def evaluate(value):
            cell, split, example = value
            job = {
                "model": model,
                "benchmark": cell,
                "split": split,
                "case_id": example.example_id,
                "release": method.release_id,
                "artifact_id": artifact.artifact_id,
                "seed": 17,
                "budget": {"max_model_calls": 8, "max_output_tokens": 4096},
                "per_call_output_cap": 512,
            }
            target = root / digest(job)
            target.mkdir()
            save(target / "claim.json", {"job": job, "started": time.time()})
            state = example.visible_state(budget=ExecutionBudget(**job["budget"]))
            calls, record = [], None
            started = time.monotonic()
            result = {
                "job": job,
                "model_binding": binding,
                "input_state_hash": state.state_hash,
                "package_location": __import__("agentinputlab").__file__,
            }
            try:
                record = InterventionRunner().run(method, state, InterventionContext(seed=17))
                backend = RecordingBackend(
                    calls=calls,
                    model_id=model["id"],
                    base_url=model["base_url"],
                    deployment_identity=fingerprint,
                    timeout_seconds=300,
                    extra_body=model.get("extra_body", {}),
                    transport=local_transport,
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
                prediction = predictor.predict(record.patch.apply(state))
                result.update(
                    status="scored_terminal",
                    metrics=dict(adapters[cell].score(example, prediction.prediction)),
                    prediction=to_jsonable(prediction),
                )
            except Exception as exc:
                result.update(
                    status="failed_terminal", error_type=type(exc).__name__, error=str(exc)
                )
            save(
                target / "terminal.json",
                {
                    **result,
                    "seconds": time.monotonic() - started,
                    "calls": calls,
                    "intervention": to_jsonable(record) if record else None,
                    "automatic_retry": False,
                },
            )

        work = [(c, s, e) for c, s in QA_CELLS for e in selected[c]]
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(evaluate, work))

    try:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lane, config["models"]))
    finally:
        stop.set()
        keep.join()
    fits = [json.loads(f.read_text()) for f in output.glob("*/fit-terminal.json")]
    rows = [json.loads(f.read_text()) for f in output.glob("*/*/terminal.json")]
    save(
        output / "complete.json",
        {
            "expected_fits": 4,
            "fit_terminals": len(fits),
            "fit_completed": sum(f["status"] == "fit_completed" for f in fits),
            "planned_downstream": 32,
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
        parser.error("supply --config --output --checkout, or one fixed --fit-job")


if __name__ == "__main__":
    main()
