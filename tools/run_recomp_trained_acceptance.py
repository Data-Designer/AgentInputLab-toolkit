"""Native trained encoder/seq2seq preparation once, then fixed reader closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_recomp_acceptance import CapturedContextSource, RecordedWorker
from run_recomp_selective_acceptance import ExactCapturedCompressor
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.curate.recomp import (
    RECOMP_COMMON_SUPPORT_SPLITTER,
    prepare_recomp_sentence_state,
)
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker
from agentinputlab.recipes import RECOMPTrainedEncoderArtifact, RECOMPTrainedSeq2SeqArtifact


def protocol(config):
    mode = config.get("training_mode", "extractive")
    if mode not in {"extractive", "abstractive"}:
        raise ValueError("trained RECOMP acceptance needs extractive or abstractive mode")
    return mode, "curate.recomp." + mode, 208 if mode == "extractive" else 212


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cases(config):
    mode, _, offset = protocol(config)
    adapters, selected = {}, {}
    for cell, split in QA_CELLS:
        adapter = load_production_benchmark_adapter(cell, config["prepared_root"])
        adapters[cell] = adapter
        examples = sorted(adapter.examples(split), key=lambda e: e.example_id)[offset : offset + 2]
        assert len(examples) == 2
        for example in examples:
            original = example.visible_state(
                budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
            )
            selected[cell, example.example_id] = (
                example,
                original,
                prepare_recomp_sentence_state(original) if mode == "extractive" else original,
            )
    return adapters, selected


def parameters(worker):
    p = worker.requests[0]["payload"]
    specific = (
        dict(input_template_hash=p["input_template_hash"])
        if "input_template_hash" in p
        else dict(sentence_splitter_revision=p["sentence_splitter_revision"])
    )
    return dict(
        checkpoint_revision=p["checkpoint_revision"],
        tokenizer_revision=p["tokenizer_revision"],
        asset_manifest_hash=p["asset_manifest_hash"],
        asset_license_acknowledged=True,
        **specific,
    )


def prepare(config, output):
    assert not output.exists()
    mode, release, offset = protocol(config)
    artifact_class = (
        RECOMPTrainedEncoderArtifact if mode == "extractive" else RECOMPTrainedSeq2SeqArtifact
    )
    candidate = artifact_class.load(config["trained_candidate"])
    assert candidate.artifact_id == config["expected_artifact_id"]
    plan = json.loads((candidate.root / "fit-plan.json").read_text())
    # Prove the selected test captures did not become labels in this fit.
    train = Path(plan["split_audit"]["files"]["train"]["path"])
    assert sha(train) == plan["split_audit"]["files"]["train"]["sha256"]
    if mode == "extractive":
        train_queries = {
            " ".join(r["dpr_instance"]["query"].casefold().split())
            for r in json.loads(train.read_text())
        }
    else:
        from agentinputlab.recipes.recomp_abstractive import _records

        train_queries = {" ".join(r["question"].casefold().split()) for r in _records(train)}
    adapters, selected = cases(config)
    assert all(
        " ".join(original.query.casefold().split()) not in train_queries
        for _, original, _ in selected.values()
    )
    import workers.recomp.driver as driver

    directory = Path(driver.__file__).parent
    environment = dict(
        PATH=os.environ["PATH"],
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=config["worker_addons"],
    )
    command = (
        config["worker_python"],
        str(directory / "driver.py"),
        "--checkout",
        config["source_root"],
        "--device",
        "cpu",
    )
    smoke = subprocess.run(
        (*command, "--smoke-import", mode),
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=True,
    )
    native = json.loads(smoke.stdout)
    assert native["status"] == "ok" and native["execution_evidence"] == "pinned_upstream_source"
    identity = dict(
        driver_sha256=sha(directory / "driver.py"),
        compatibility_manifest_sha256=sha(directory / "compatibility_manifest.json"),
        runtime_fingerprint_hash=native["runtime_fingerprint_hash"],
        checkpoint_manifest_sha256=candidate.manifest_sha256,
        trained_artifact_id=candidate.artifact_id,
        trained_weight_sha256=sha(candidate.checkpoint / "model.safetensors"),
        native_fit_terminal_sha256=sha(candidate.root / "fit-terminal.json"),
        execution_evidence="pinned_upstream_source",
    )
    output.mkdir()
    save(
        output / "manifest.json",
        dict(
            trained_artifact_id=candidate.artifact_id,
            training_mode=mode,
            native_import=native,
            candidate_manifest_sha256=candidate.manifest_sha256,
            identity=identity,
            source_indices=[offset, offset + 1],
            cases=[
                dict(
                    benchmark=cell,
                    case_id=case,
                    original_state_hash=original.state_hash,
                    prepared_state_hash=state.state_hash,
                )
                for (cell, case), (_, original, state) in selected.items()
            ],
            dataset_files={cell: dict(a.prepared.file_sha256) for cell, a in adapters.items()},
            training_query_overlap=False,
            preparation_shared_across_readers=True,
            automatic_retry=False,
        ),
    )
    for (cell, case), (_, original, state) in selected.items():
        folder = output / digest({"cell": cell, "case_id": case})
        folder.mkdir()
        worker = RecordedWorker(
            JsonLineSubprocessWorker(
                command=(
                    *command,
                    "--checkpoint",
                    str(candidate.checkpoint),
                    "--checkpoint-manifest",
                    str(candidate.checkpoint_manifest),
                    "--acknowledge-noassertion-checkpoint",
                ),
                worker_id="trained-native-recomp-candidate",
                identity=identity,
                timeout_seconds=900,
                environment=environment,
            ),
            folder,
        )
        component = create_intervention(
            release,
            **candidate.inference_parameters(
                **(
                    dict(sentence_splitter_revision=RECOMP_COMMON_SUPPORT_SPLITTER)
                    if mode == "extractive"
                    else {}
                )
            ),
        )
        record = run_intervention(
            component,
            state,
            InterventionContext(
                seed=17,
                external_runtime=worker,
                retriever=CapturedContextSource(cell, case, content_hash(original.context)),
            ),
        )["record"]
        assert record["usage"]["model_calls"] == 1 and 0 <= record["usage"]["output_tokens"] <= (
            0 if mode == "extractive" else 512
        )
        save(
            folder / "terminal.json",
            dict(
                status="native_compressed",
                record=record,
                worker_responses=worker.responses,
                physical_helper_model_calls=1,
                reference_exposed=False,
                original_state_hash=original.state_hash,
                automatic_retry=False,
            ),
        )
    terminals = list(output.glob("*/terminal.json"))
    assert len(terminals) == 8
    save(
        output / "complete.json",
        dict(
            expected_captures=8,
            terminal_captures=8,
            physical_new_native_helper_calls=8,
            actual_native_scorer_with_trained_candidate=True,
            training_mode=mode,
            pretrained_artifact_substitution=False,
            terminals={str(p.relative_to(output)): sha(p) for p in terminals},
            automatic_retry=False,
        ),
    )


def run(config, output):
    assert len(config["models"]) == 1 and not output.exists()
    mode, release, offset = protocol(config)
    model = config["models"][0]
    binding = verify_services([model])[model["id"]]
    preparation = Path(config["preparation_root"])
    complete = json.loads((preparation / "complete.json").read_text())
    assert complete["physical_new_native_helper_calls"] == 8 and complete["terminal_captures"] == 8
    source = json.loads((preparation / "manifest.json").read_text())
    adapters, selected = cases(config)
    assert {cell: dict(a.prepared.file_sha256) for cell, a in adapters.items()} == source[
        "dataset_files"
    ]
    jobs = []
    for (cell, case), (_, original, state) in selected.items():
        folder = preparation / digest({"cell": cell, "case_id": case})
        assert (
            sha(folder / "terminal.json")
            == complete["terminals"][str((folder / "terminal.json").relative_to(preparation))]
        )
        job = dict(
            release=release,
            model=model,
            benchmark=cell,
            case_id=case,
            seed=17,
            trained_artifact_id=source["trained_artifact_id"],
            preparation_terminal_sha256=sha(folder / "terminal.json"),
            source_original_state_hash=original.state_hash,
            prepared_state_hash=state.state_hash,
            per_call_max_tokens=512,
            budget={"max_model_calls": 8, "max_output_tokens": 4096},
            source_indices=[offset, offset + 1],
            shared_native_helper=True,
            assembly="controlled_top5_source_order",
        )
        job["id"] = digest(job)
        jobs.append(job)
    output.mkdir()
    for name in ("claims", "terminals", "model-outcomes"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        dict(
            jobs=jobs,
            model_binding=binding,
            source_preparation_manifest_sha256=sha(preparation / "manifest.json"),
            source_preparation_complete_sha256=sha(preparation / "complete.json"),
            dataset_files=source["dataset_files"],
            new_target_attempts=8,
            physical_new_helper_calls_in_this_reader_lane=0,
            logical_helper_cost_charged_to_each_case=True,
            automatic_retry=False,
        ),
    )

    def execute(job):
        example, original, state = selected[job["benchmark"], job["case_id"]]
        folder = preparation / digest({"cell": job["benchmark"], "case_id": job["case_id"]})
        worker = ExactCapturedCompressor(folder)
        calls = []
        save(output / "claims" / (job["id"] + ".json"), {"job": job})
        row = dict(
            job=job,
            additional_reader_attempts=1,
            additional_helper_attempts=0,
            automatic_retry=False,
        )
        prediction = None
        try:
            helper = run_intervention(
                create_intervention(release, **parameters(worker)),
                state,
                InterventionContext(
                    seed=17,
                    external_runtime=worker,
                    retriever=CapturedContextSource(
                        job["benchmark"], job["case_id"], content_hash(original.context)
                    ),
                ),
            )
            assert helper["record"]["patch"] == worker.record["patch"]
            from agentinputlab.contracts import ContextBlock, FrozenAgentState

            # Apply the actual component through the normal typed runner output.
            data = helper["state"]
            compressed = FrozenAgentState(
                **{
                    **data,
                    "context": tuple(ContextBlock(**b) for b in data["context"]),
                    "messages": (),
                    "budget": replace(
                        state.budget,
                        max_model_calls=8 - helper["record"]["usage"]["model_calls"],
                        max_output_tokens=4096 - helper["record"]["usage"]["output_tokens"],
                    ),
                }
            )
            row["helper"] = helper["record"]
            backend = RecordingBackend(
                calls=calls,
                model_id=model["id"],
                base_url=model["base_url"],
                extra_body=model["extra_body"],
                transport=local_transport,
                timeout_seconds=240,
                deployment_identity={
                    "revision": digest(binding["model_json_hashes"]),
                    "server": "vllm",
                },
            )
            predictor = ProductionPredictorBuilder().build(
                StaticPredictorBuildRequest(
                    benchmark_id=job["benchmark"],
                    split=next(s for c, s in QA_CELLS if c == job["benchmark"]),
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
            prediction = predictor.predict(compressed)
            row.update(status="prediction_returned", prediction=to_jsonable(prediction))
        except Exception as exc:
            row.update(
                status="failed_terminal",
                stage="trained_component_and_reader",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        row.update(calls=calls, logical_helper_reuse=worker.proof())
        save(
            output / "model-outcomes" / (job["id"] + ".json"),
            {**row, "native_scoring_started": False},
        )
        if prediction is not None:
            try:
                row.update(
                    status="scored_terminal",
                    metrics=dict(adapters[job["benchmark"]].score(example, prediction.prediction)),
                )
            except Exception as exc:
                row.update(
                    status="failed_terminal",
                    stage="native_QA_scoring",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        assert len(calls) <= 1 and len(worker.used) == 2
        save(output / "terminals" / (job["id"] + ".json"), row)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(execute, jobs))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    assert len(rows) == 8
    save(
        output / "complete.json",
        dict(
            expected_positions=8,
            terminal_positions=8,
            scored=sum(r["status"] == "scored_terminal" for r in rows),
            failed=sum(r["status"] == "failed_terminal" for r in rows),
            physical_new_reader_calls=sum(len(r["calls"]) for r in rows),
            physical_new_helper_calls=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    a = p.parse_args()
    (prepare if a.prepare else run)(json.loads(a.config.read_text()), a.output)
