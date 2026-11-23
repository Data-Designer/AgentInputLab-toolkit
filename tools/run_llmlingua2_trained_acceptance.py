"""Real trained-classifier preparation once, then distinct four-reader trials.

The two multilingual helper probes are synthetic inputs, not benchmark rows.
Every saved context comes from the actual fitted native classifier. Downstream
reader failures, unconsumed helper failures and their costs remain explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ContextBlock, ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.recipes import LLMLingua2TrainedArtifact
from agentinputlab.runner import InterventionRunner

RELEASE = "training.llmlingua2.distillation"
OFFSET = 224
BUDGET = dict(max_model_calls=8, max_output_tokens=4096)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cases(config):
    adapters, examples, jobs = {}, {}, []
    for cell, split in QA_CELLS:
        adapter = load_production_benchmark_adapter(cell, config["prepared_root"])
        adapters[cell] = adapter
        values = sorted(adapter.examples(split), key=lambda e: e.example_id)[OFFSET : OFFSET + 2]
        if len(values) != 2:
            raise ValueError("fixed source cases unavailable; no outcome-based substitution")
        for index, example in enumerate(values, OFFSET):
            state = example.visible_state(budget=ExecutionBudget(**BUDGET))
            job = dict(
                benchmark=cell,
                split=split,
                case_id=example.example_id,
                source_index=index,
                state_hash=state.state_hash,
                query_sha256=content_hash(state.query),
                context_sha256=content_hash(state.context),
            )
            jobs.append(job)
            examples[cell, example.example_id] = (example, state)
    return adapters, examples, jobs


def assert_loaded_weights(model, artifact):
    """No forward/gradient; prove the runtime did not resize/substitute weights."""
    import torch
    from safetensors.torch import load_file

    if type(model).__name__ not in {
        "BertForTokenClassification",
        "XLMRobertaForTokenClassification",
    }:
        raise ValueError("requires the real native token-classifier runtime")
    tensors = load_file(str(artifact.checkpoint / "model.safetensors"), device="cpu")
    actual = model.state_dict()
    if set(actual) != set(tensors) or any(
        actual[k].shape != tensors[k].shape
        or actual[k].dtype != tensors[k].dtype
        or not torch.equal(actual[k].detach().cpu(), tensors[k])
        for k in tensors
    ):
        raise ValueError("loaded runtime differs from fitted safe tensors")
    return dict(
        parameter_keys=len(tensors),
        exact_keys_shapes_dtypes_values=True,
        safe_weight_sha256=sha(artifact.checkpoint / "model.safetensors"),
        new_forwards=0,
    )


def prepare(config, output):
    if output.exists():
        raise ValueError("create-only native preparations; no consumed-helper retry")
    import llmlingua.prompt_compressor as native_module
    import torch

    torch.set_num_threads(4)
    artifact = LLMLingua2TrainedArtifact.load(config["trained_artifact"])
    if artifact.artifact_id != config["expected_artifact_id"]:
        raise ValueError("fitted artifact differs")
    plan = json.loads((artifact.root / "fit-plan.json").read_text())
    native_input = artifact.root / "protocol/data/fit-data.json"
    if sha(native_input) != plan["data_audit"]["sha256"]:
        raise ValueError("native training input copy differs")
    training = json.loads(native_input.read_text())
    fit_texts = {" ".join(text.casefold().split()) for text in training["origin"]}
    adapters, examples, jobs = cases(config)
    if any(" ".join(state.query.casefold().split()) in fit_texts for _, state in examples.values()):
        raise ValueError("fixed test query exactly overlaps a native fit input")
    output.mkdir()
    for name in ("claims", "preparation", "boundaries", "classifier_tensors"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        dict(
            release=RELEASE,
            trained_artifact_id=artifact.artifact_id,
            jobs=jobs,
            source_indices=[OFFSET, OFFSET + 1],
            seed=17,
            helper_native_initialization_seed=42,
            helper_device="cpu",
            helper_dtype="float32",
            config={"rate": 0.5},
            dataset_files={c: dict(a.prepared.file_sha256) for c, a in adapters.items()},
            native_fit_plan_sha256=sha(artifact.root / "fit-plan.json"),
            native_fit_terminal_sha256=sha(artifact.root / "fit-terminal.json"),
            exact_fit_query_overlap=False,
            pretraining_or_warmstart_data_independence_claimed=False,
            source_cases_fixed_before_helper_or_reader_output=True,
            synthetic_boundary_inputs_not_benchmark=True,
            automatic_retry=False,
            paper_results_changed=False,
        ),
    )
    helper = artifact.intervention(config={"rate": 0.5}, device_map="cpu")
    start, cpu = time.monotonic(), time.process_time()
    compressor = helper._load(use_llmlingua2=True)
    model = compressor.model
    if model.training or model.config.num_labels != 2:
        raise ValueError("native fitted classifier must be evaluation-mode/two-label")
    weights_before = assert_loaded_weights(model, artifact)
    save(
        output / "native-runtime-ready.json",
        dict(
            model_class=type(model).__name__,
            tokenizer_class=type(compressor.tokenizer).__name__,
            vocabulary_size=len(compressor.tokenizer),
            native_runtime_sha256=sha(Path(native_module.__file__)),
            weights=weights_before,
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.monotonic() - start,
            dependencies={
                n: importlib.metadata.version(n)
                for n in ("llmlingua", "torch", "transformers", "numpy", "safetensors")
            },
            new_optimizer_steps=0,
            new_teacher_calls=0,
        ),
    )
    events = []
    active = {}

    def before(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        active["inputs"] = {
            "input__" + k: v.detach().cpu().contiguous()
            for k, v in kwargs.items()
            if isinstance(v, torch.Tensor)
        }
        if "input__input_ids" not in active["inputs"]:
            active["inputs"]["input__input_ids"] = ids.detach().cpu().contiguous()
        events.append(
            dict(
                shape=list(ids.shape),
                input_slots=int(ids.numel()),
                nonpadding_input_tokens=int(mask.sum()) if mask is not None else int(ids.numel()),
                model_training=bool(module.training),
                grad_enabled=torch.is_grad_enabled(),
                completed=False,
            )
        )

    def after(module, args, kwargs, result):
        from safetensors.torch import save_file

        path = active["folder"] / (f"forward-{len(events) - 1:04d}.safetensors")
        save_file(
            {**active["inputs"], "logits": result.logits.detach().cpu().contiguous()}, str(path)
        )
        events[-1].update(
            completed=True,
            logits_shape=list(result.logits.shape),
            tensor_capture_path=str(path),
            tensor_capture_sha256=sha(path),
        )

    hooks = [
        model.register_forward_pre_hook(before, with_kwargs=True),
        model.register_forward_hook(after, with_kwargs=True),
    ]
    receipts = []
    try:

        def one(job, state, folder):
            key = digest(job)
            save(
                output / "claims" / (key + ".json"),
                dict(job=job, physical_helper_attempt_once=True),
            )
            events.clear()
            active["folder"] = output / "classifier_tensors" / key
            active["folder"].mkdir()
            start, cpu = time.monotonic(), time.process_time()
            row = dict(
                job=job,
                trained_artifact_id=artifact.artifact_id,
                physical_helper_jobs=1,
                new_generative_model_calls=0,
                new_optimizer_steps=0,
                reference_exposed=False,
                automatic_retry=False,
            )
            try:
                record = InterventionRunner().run(helper, state, InterventionContext(seed=17))
                materialized = record.patch.apply(state)
                if record.usage.model_calls != 0 or record.usage.output_tokens != 0:
                    raise ValueError("token classifier unexpectedly charged generative calls")
                row.update(
                    status="native_compressed",
                    record=to_jsonable(record),
                    materialized_context=to_jsonable(materialized.context),
                    materialized_state_hash=materialized.state_hash,
                )
            except Exception as exc:
                row.update(
                    status="failed_native_helper", error_type=type(exc).__name__, error=str(exc)
                )
            row.update(
                native_forward_events=list(events),
                physical_classifier_forward_attempts=len(events),
                cpu_seconds=time.process_time() - cpu,
                wall_seconds=time.monotonic() - start,
            )
            path = folder / (key + ".json")
            save(path, row)
            return dict(
                path=str(path),
                sha256=sha(path),
                job=job,
                status=row["status"],
                physical_classifier_forward_attempts=len(events),
            )

        for job in jobs:
            receipts.append(
                one(job, examples[job["benchmark"], job["case_id"]][1], output / "preparation")
            )
        boundaries = []
        for language, text in (
            ("zh", "工程团队约定周四核验接口，并在周六完成部署。"),  # noqa: RUF001
            (
                "es",
                "La prueba de integración empieza el jueves y el despliegue se revisa el sábado.",
            ),
        ):
            state = FrozenAgentState(
                query="Multilingual compression boundary probe.", context=(ContextBlock(text=text),)
            )
            job = dict(
                language=language,
                state_hash=state.state_hash,
                synthetic_boundary_not_benchmark=True,
            )
            boundaries.append(one(job, state, output / "boundaries"))
    finally:
        for hook in hooks:
            hook.remove()
    weights_after = assert_loaded_weights(model, artifact)
    if weights_before != weights_after:
        raise ValueError("native inference changed fitted weights")
    save(
        output / "complete.json",
        dict(
            manifest_sha256=sha(output / "manifest.json"),
            captures=receipts,
            boundaries=boundaries,
            benchmark_helper_jobs=8,
            synthetic_helper_jobs=2,
            physical_helper_jobs=10,
            physical_classifier_forward_attempts=sum(
                r["physical_classifier_forward_attempts"] for r in [*receipts, *boundaries]
            ),
            source_statuses=dict(Counter(r["status"] for r in receipts)),
            weights_before_after_equal=True,
            safe_weight_sha256=weights_before["safe_weight_sha256"],
            new_generative_model_calls=0,
            new_optimizer_steps=0,
            publication_ready=False,
        ),
    )
    print("8 native fitted compressions +2 labeled synthetic boundary jobs closed", flush=True)


def verified_capture(preparation, spec, original, expected_artifact_id):
    path = Path(spec["path"])
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(preparation.resolve())
        or sha(path) != spec["sha256"]
    ):
        raise ValueError("native capture path/hash differs")
    row = json.loads(path.read_text())
    if (
        row["job"] != spec["job"]
        or row["job"]["state_hash"] != original.state_hash
        or row["trained_artifact_id"] != expected_artifact_id
    ):
        raise ValueError("native capture source differs; no permissive reuse")
    if row["status"] != "native_compressed":
        return row, None
    patch = row["record"]["patch"]
    if (
        patch["source_state_hash"] != original.state_hash
        or patch["release_id"] != RELEASE
        or patch["profile_kind"] != "controlled_adapter"
        or len(patch["operations"]) != 1
        or patch["operations"][0]["op"] != "replace_context"
        or patch["metadata"]["trained_artifact_id"] != expected_artifact_id
        or patch["metadata"]["model_revision"] != "native-fit-" + expected_artifact_id
    ):
        raise ValueError("requires the actual typed trained-compressor context patch")
    context = tuple(ContextBlock(**b) for b in patch["operations"][0]["value"])
    materialized = replace(original, context=context)
    if (
        materialized.state_hash != row["materialized_state_hash"]
        or to_jsonable(context) != row["materialized_context"]
    ):
        raise ValueError("native patch materialization differs")
    return row, materialized


def run(config, output):
    if output.exists() or len(config["models"]) != 1:
        raise ValueError("one model/create-only reader lane required")
    preparation = Path(config["preparation_root"])
    if sha(preparation / "complete.json") != config["preparation_complete_sha256"]:
        raise ValueError("native preparation closure differs")
    complete = json.loads((preparation / "complete.json").read_text())
    manifest = json.loads((preparation / "manifest.json").read_text())
    if (
        complete["manifest_sha256"] != sha(preparation / "manifest.json")
        or complete["physical_helper_jobs"] != 10
        or complete["new_generative_model_calls"] != 0
    ):
        raise ValueError("requires actual complete native fitted-classifier preparations")
    adapters, examples, jobs = cases(config)
    if (
        jobs != manifest["jobs"]
        or {c: dict(a.prepared.file_sha256) for c, a in adapters.items()}
        != manifest["dataset_files"]
    ):
        raise ValueError("frozen data/source selection differs")
    model = config["models"][0]
    binding = verify_services([model])[model["id"]]
    output.mkdir()
    for name in ("claims", "terminals"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        dict(
            release=RELEASE,
            model=model,
            model_binding=binding,
            expected_positions=8,
            preparation_complete_sha256=sha(preparation / "complete.json"),
            trained_artifact_id=manifest["trained_artifact_id"],
            cases=jobs,
            seed=17,
            budget=BUDGET,
            per_call_tokens=512,
            new_helper_jobs=0,
            automatic_retry=False,
        ),
    )
    specs = {(s["job"]["benchmark"], s["job"]["case_id"]): s for s in complete["captures"]}
    if len(specs) != 8:
        raise ValueError("requires eight unique source-bound helper receipts")

    def execute(job):
        example, original = examples[job["benchmark"], job["case_id"]]
        spec = specs[job["benchmark"], job["case_id"]]
        capture, materialized = verified_capture(
            preparation, spec, original, manifest["trained_artifact_id"]
        )
        identifier = digest(
            dict(
                model=model["id"],
                case=job,
                artifact=manifest["trained_artifact_id"],
                release=RELEASE,
            )
        )
        save(
            output / "claims" / (identifier + ".json"),
            dict(model=model, case=job, scientific_attempt_once=True),
        )
        calls = []
        row = dict(
            model=model,
            case=job,
            release=RELEASE,
            helper_capture_sha256=spec["sha256"],
            helper_capture_path=spec["path"],
            trained_artifact_id=manifest["trained_artifact_id"],
            physical_new_helper_jobs=0,
            shared_actual_native_helper_not_synthetic=True,
            automatic_retry=False,
        )
        start = time.monotonic()
        if materialized is None:
            row.update(
                status="failed_helper_unreached_reader",
                error_type=capture["error_type"],
                error=capture["error"],
            )
        else:
            try:
                backend = RecordingBackend(
                    calls=calls,
                    model_id=model["id"],
                    base_url=model["base_url"],
                    extra_body=model["extra_body"],
                    transport=local_transport,
                    timeout_seconds=300,
                    deployment_identity={
                        "revision": digest(binding["model_json_hashes"]),
                        "server": "vllm",
                    },
                )
                predictor = ProductionPredictorBuilder().build(
                    StaticPredictorBuildRequest(
                        benchmark_id=job["benchmark"],
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
                prediction = predictor.predict(materialized)
                row.update(
                    status="scored_terminal",
                    prediction=to_jsonable(prediction),
                    metrics=dict(adapters[job["benchmark"]].score(example, prediction.prediction)),
                )
            except Exception as exc:
                row.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        if len(calls) > 1:
            raise ValueError("unexpected repeated reader generation")
        row.update(
            calls=calls,
            physical_new_generator_attempts=len(calls),
            wall_seconds=time.monotonic() - start,
            original_state_hash=original.state_hash,
            materialized_state_hash=materialized.state_hash if materialized else None,
        )
        save(output / "terminals" / (identifier + ".json"), row)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(execute, jobs))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    if len(rows) != 8:
        raise ValueError("fixed endpoint denominator is incomplete")
    save(
        output / "complete.json",
        dict(
            terminal_positions=8,
            expected_positions=8,
            statuses=dict(Counter(r["status"] for r in rows)),
            new_generator_attempts=sum(len(r["calls"]) for r in rows),
            new_helper_jobs=0,
            preserved_failed_denominators=True,
            automatic_retry=False,
            publication_ready=False,
        ),
    )
    save(output / "post-service-identities.json", verify_services([model]))
    print(model["id"], "8 trained-compressor reader terminals closed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    (prepare if args.prepare else run)(config, args.output)
