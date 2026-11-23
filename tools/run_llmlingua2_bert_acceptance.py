"""Bounded compatibility acceptance, not a full benchmark or paper experiment.

Compress source-visible inputs once with the fixed official BERT helper, then
evaluate the identical materialized patches independently for each reader.
All model calls are single attempts; references enter only the original scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ContextBlock, ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.curate.llmlingua.multilingual import (
    BERT_MULTILINGUAL_FILES,
    BERT_MULTILINGUAL_REVISION,
)
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks.production import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.runner import InterventionRunner

RELEASE = "curate.llmlingua2.bert_multilingual_official"
BUDGET = dict(max_model_calls=8, max_output_tokens=4096)


def boundary_states():
    for language, text in (("zh", "项目会议决定下周二提交计划。预算审批在周五完成。"),
                           ("es", "El comité aprobó el proyecto el martes. El presupuesto se revisa el viernes.")):
        yield language, FrozenAgentState(query="Multilingual compression boundary probe.",
            context=(ContextBlock(text=text),))


def cases(config):
    result = []
    for cell, split in QA_CELLS:
        adapter = load_production_benchmark_adapter(cell, config["prepared_root"])
        values = sorted(adapter.examples(split), key=lambda e: e.example_id)[80:82]
        assert len(values) == 2
        for example in values:
            state = example.visible_state(budget=ExecutionBudget(**BUDGET))
            result.append((cell, split, adapter, example, state))
    return result


def prepare(config, output):
    import torch

    torch.set_num_threads(4)
    values = cases(config)
    jobs = [dict(benchmark=cell, split=split, case_id=ex.example_id, state_hash=state.state_hash)
            for cell, split, _, ex, state in values]
    output.mkdir()
    (output / "preparation").mkdir()
    (output / "claims").mkdir()
    save(output / "manifest.json", dict(release=RELEASE, jobs=jobs, source_indices=[80, 81],
        fixed_before_model_outputs=True, seed=17, helper_native_initialization_seed=42,
        source_checkpoint_revision=BERT_MULTILINGUAL_REVISION, helper_device="cpu",
        helper_dtype="float32", automatic_retry=False, paper_results_changed=False,
        reference_exposed=False, four_backbone_certification_claimed=False))
    helper = create_intervention(RELEASE, model_name=config["snapshot"], device_map="cpu",
        model_config={"local_files_only": True, "torch_dtype": "float32"})
    preparations = []
    for job, (_, _, _, _, state) in zip(jobs, values, strict=True):
        key = digest(job)
        save(output / "claims" / (key + ".json"), dict(job=job, started=time.time()))
        start = time.monotonic()
        try:
            record = InterventionRunner().run(helper, state, InterventionContext(seed=17))
            materialized = record.patch.apply(state)
            row = dict(job=job, status="native_compressed", record=to_jsonable(record),
                materialized_context=to_jsonable(materialized.context),
                materialized_state_hash=materialized.state_hash,
                seconds=time.monotonic()-start, classifier_calls=1,
                generative_model_calls=0, reference_exposed=False)
        except Exception as exc:
            save(output / "preparation" / (key + ".json"), dict(job=job,
                status="failed_preparation", error_type=type(exc).__name__, error=str(exc),
                seconds=time.monotonic()-start, automatic_retry=False))
            raise
        save(output / "preparation" / (key + ".json"), row)
        preparations.append(key)
    # Fixed multilingual boundary probes are real helper invocations, but are
    # synthetic inputs, not new benchmark outcomes or four-reader trials.
    boundaries = []
    for language, state in boundary_states():
        start = time.monotonic()
        record = InterventionRunner().run(helper, state, InterventionContext(seed=17))
        boundaries.append(dict(language=language, record=to_jsonable(record),
            seconds=time.monotonic()-start, synthetic_boundary_not_benchmark=True))
    save(output / "multilingual-boundaries.json", boundaries)
    compressor = helper._load(use_llmlingua2=True)
    assert type(compressor.model).__name__ == "BertForTokenClassification"
    assert "BertTokenizer" in type(compressor.tokenizer).__name__
    assert compressor.model.config.model_type == "bert"
    assert len(compressor.tokenizer) == compressor.model.config.vocab_size == 119647
    from agentinputlab.artifacts import sha256_file
    assets = {name: sha256_file(Path(config["snapshot"]) / name) for name, _ in BERT_MULTILINGUAL_FILES}
    assert assets == dict(BERT_MULTILINGUAL_FILES)
    save(output / "complete.json", dict(preparations=preparations, dataset_classifier_calls=8,
        synthetic_multilingual_classifier_calls=2, generative_model_calls=0,
        model_class=type(compressor.model).__name__, tokenizer_class=type(compressor.tokenizer).__name__,
        vocabulary_size=len(compressor.tokenizer), helper_asset_sha256=assets,
        dependencies={n: importlib.metadata.version(n) for n in
            ("llmlingua", "torch", "transformers", "accelerate", "tiktoken", "numpy")},
        publication_ready=False, benchmark_quality_claimed=False))
    print("8 source-visible native compressions and 2 synthetic multilingual boundary probes saved", flush=True)


def evaluate(config, output):
    prepared = Path(config["preparation"])
    assert (prepared / "complete.json").is_file()
    values = cases(config)
    output.mkdir()
    (output / "claims").mkdir()
    (output / "terminals").mkdir()
    model = config["model"]
    jobs = [dict(benchmark=cell, split=split, case_id=ex.example_id, state_hash=state.state_hash,
        model=model, release=RELEASE, seed=17, per_call_output_tokens=512, budget=BUDGET)
        for cell, split, _, ex, state in values]
    save(output / "manifest.json", dict(jobs=jobs, automatic_retry=False,
        fixed_before_outputs=True, paper_results_changed=False,
        shared_helper_complete_sha256=hashlib.sha256((prepared/"complete.json").read_bytes()).hexdigest()))

    def one(item):
        job, (_, _, adapter, example, state) = item
        key = digest(job)
        save(output / "claims" / (key + ".json"), dict(job=job, started=time.time()))
        helper_key = digest({k: job[k] for k in ("benchmark", "split", "case_id", "state_hash")})
        receipt = prepared / "preparation" / (helper_key + ".json")
        row = json.loads(receipt.read_text())
        assert row["job"]["state_hash"] == state.state_hash and row["status"] == "native_compressed"
        state = replace(state, context=tuple(ContextBlock(**r) for r in row["materialized_context"]))
        assert state.state_hash == row["materialized_state_hash"]
        service = Path(model["service_directory"])
        binding = json.loads((service/"started.json").read_text())
        calls = []
        backend = RecordingBackend(calls=calls, model_id=model["id"], base_url=model["base_url"],
            timeout_seconds=300, deployment_identity={"revision": digest(binding["model_json_hashes"])},
            extra_body=model.get("extra_body", {}), transport=local_transport)
        result = dict(job=job, shared_preparation_receipt=str(receipt),
            shared_preparation_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            model_binding=binding, automatic_retry=False, additional_attempts=1,
            helper_compression_repeated=False, reference_exposed=False)
        start = time.monotonic()
        try:
            predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                benchmark_id=job["benchmark"], split=job["split"], model_slot=model["id"], seed=17,
                builder_parameters={"schema_version":1, "task":"qa", "qa":{
                    "output_format":"structured_json", "max_tokens":512, "use_seed":True}},
                runtime=StaticRuntimeComponents(backend=backend)))
            prediction = predictor.predict(state)
            metrics = dict(adapter.score(example, prediction.prediction))
            result.update(status="scored_terminal", prediction=to_jsonable(prediction), metrics=metrics)
        except Exception as exc:
            result.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        result.update(calls=calls, physical_generation_calls=len(calls), seconds=time.monotonic()-start)
        save(output / "terminals" / (key+".json"), result)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one, zip(jobs, values, strict=True)))
    terminals = [json.loads(p.read_text()) for p in sorted((output/"terminals").glob("*.json"))]
    assert len(terminals) == 8
    save(output / "complete.json", dict(expected_positions=8, terminal_positions=len(terminals),
        scored=sum(r["status"]=="scored_terminal" for r in terminals),
        failed=sum(r["status"]=="failed_terminal" for r in terminals),
        physical_generation_calls=sum(r["physical_generation_calls"] for r in terminals),
        automatic_retry=False, quality_success_claimed=False, publication_ready=False))
    print(model["id"], "8 terminal positions saved", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("preflight", "prepare", "evaluate"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == "preflight":
        assert len(cases(config)) == 8
        print("four source-backed QA adapters, eight fixed source states; zero model calls")
    elif args.output is None:
        parser.error("--output is required for execution")
    elif args.mode == "prepare":
        prepare(config, args.output)
    else:
        evaluate(config, args.output)


if __name__ == "__main__":
    main()
