"""Fixed real sentence-input RECOMP pairs; safe existing assets, no retries/downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_recomp_acceptance import CapturedContextSource, RecordedWorker
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.curate.recomp import (
    RECOMP_COMMON_SUPPORT_SPLITTER,
    prepare_recomp_sentence_state,
)
from agentinputlab.api import create_intervention
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.integrations.external_workers import JsonLineSubprocessWorker
from agentinputlab.runner import InterventionRunner

RELEASE = 'curate.recomp.extractive'
SOURCE_OFFSET = 98


def select_cases(adapter, split):
    rows = sorted(adapter.examples(split), key=lambda e: e.example_id)[SOURCE_OFFSET:SOURCE_OFFSET + 2]
    if len(rows) != 2:
        raise ValueError('requires two fixed source cases at98:100')
    return rows


def run(config, output, upstream, worker_python, checkpoint, asset_manifest, conversion, addons, observed):
    if output.exists():
        raise ValueError('create-only cohort already exists; inspect saved attempts')
    bindings = verify_services(config['models'])
    asset = json.loads(asset_manifest.read_text())
    proof = json.loads(conversion.read_text())
    asset_sha = hashlib.sha256(asset_manifest.read_bytes()).hexdigest()
    if (asset['mode'] != 'extractive' or proof['checkpoint_manifest_sha256'] != asset_sha
            or proof['all_tensor_keys_shapes_dtypes_values_equal'] is not True
            or proof['quantization_or_parameter_change'] is not False):
        raise ValueError('requires exact safe-format conversion proof')
    import workers.recomp.driver as driver_module
    worker_directory = Path(driver_module.__file__).parent
    environment = {'PATH': os.environ['PATH'], 'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
        'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'TOKENIZERS_PARALLELISM': 'false',
        'PYTHONPATH': str(addons)}
    command = (str(worker_python), str(worker_directory / 'driver.py'), '--checkout',
               str(upstream / 'upstream-paper/recomp'), '--device', 'cpu')
    smoke = subprocess.run((*command, '--smoke-import', 'extractive'), env=environment,
        capture_output=True, text=True, check=True, timeout=120)
    native = json.loads(smoke.stdout)
    if native.get('status') != 'ok' or native.get('execution_evidence') != 'pinned_upstream_source':
        raise ValueError('native import failed before any inference')
    identity = {'driver_sha256': hashlib.sha256((worker_directory / 'driver.py').read_bytes()).hexdigest(),
        'compatibility_manifest_sha256': hashlib.sha256((worker_directory / 'compatibility_manifest.json').read_bytes()).hexdigest(),
        'runtime_fingerprint_hash': native['runtime_fingerprint_hash'],
        'checkpoint_manifest_sha256': asset_sha, 'conversion_proof_sha256': hashlib.sha256(conversion.read_bytes()).hexdigest(),
        'source_weight_sha256': proof['source_weight_sha256'], 'execution_evidence': native['execution_evidence']}
    assert identity['compatibility_manifest_sha256'] == native['compatibility_manifest_sha256']
    adapters = {c: load_production_benchmark_adapter(c, config['prepared_root']) for c, _ in QA_CELLS}
    selected = {c: select_cases(adapters[c], s) for c, s in QA_CELLS}
    captures = {}
    for cell, examples in selected.items():
        for example in examples:
            original = example.visible_state(budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096))
            captures[(cell, example.example_id)] = (original, prepare_recomp_sentence_state(original, max_candidates=4096))
    inventory = json.loads(observed.read_text())
    old = {(r['release'], r['benchmark'], r['backbone'], e['case_id'])
           for r in inventory['rows'] for e in r['evidence']}
    jobs = []
    for model in config['models']:
        for cell, split in QA_CELLS:
            for example in selected[cell]:
                for release in ('baseline.no_op', RELEASE):
                    if (release, cell, model['id'], example.example_id) in old:
                        raise ValueError('selected case already has a saved attempt; review exact reuse before dispatch')
                    job = {'model': model, 'benchmark': cell, 'split': split, 'case_id': example.example_id,
                        'release': release, 'seed': 17, 'budget': {'max_model_calls': 8, 'max_output_tokens': 4096},
                        'per_call_output_cap': 512, 'predictor': 'structured_json',
                        'input_preparation': RECOMP_COMMON_SUPPORT_SPLITTER, 'preparation_on_both_arms': True,
                        'candidate_limit': 4096, 'compressor': identity,
                        'source_captured_state_hash': captures[(cell, example.example_id)][0].state_hash,
                        'prepared_input_state_hash': captures[(cell, example.example_id)][1].state_hash,
                        'assembly': 'controlled_top5_source_order_not_paper_native'}
                    job['id'] = digest(job)
                    jobs.append(job)
    output.mkdir(parents=True)
    for name in ('claims', 'terminals', 'preparation'):
        (output / name).mkdir()
    save(output / 'manifest.json', {'jobs': jobs, 'model_bindings': bindings, 'native_import': native,
        'case_selection': 'fixed_sorted_source_indices98:100_before_outcomes',
        'observed_inventory_sha256': hashlib.sha256(observed.read_bytes()).hexdigest(),
        'source_context_is_visible_capture_not_live_retrieval': True, 'automatic_retry': False,
        'both_arms_use_same_prepared_input': True, 'paper_results_changed': False,
        'helper_cost_physical_once_per_capture_logical_in_each_treatment': True})

    def target(job, state, original_hash, record=None, prep=None):
        save(output / 'claims' / (job['id'] + '.json'), {'job': job, 'started': time.time()})
        calls = []
        row = {'job': job, 'package_location': __import__('agentinputlab').__file__,
            'model_binding': bindings[job['model']['id']], 'input_state_hash': state.state_hash,
            'source_captured_state_hash': original_hash, 'intervention': to_jsonable(record) if record else None,
            'shared_preparation_receipt': str(prep) if prep else None, 'additional_target_attempts': 1,
            'automatic_retry': False}
        try:
            if record is not None:
                state = replace(record.patch.apply(state), budget=replace(state.budget,
                    max_model_calls=state.budget.max_model_calls - record.usage.model_calls,
                    max_output_tokens=state.budget.max_output_tokens - record.usage.output_tokens))
            model = job['model']
            backend = RecordingBackend(calls=calls, model_id=model['id'], base_url=model['base_url'],
                api_key_env='AILAB_TOOLKIT_UNUSED_LOCAL_ONLY',
                timeout_seconds=180, extra_body=model.get('extra_body', {}), transport=local_transport,
                deployment_identity={'revision': digest(bindings[model['id']]['model_json_hashes']), 'server': 'vllm'})
            row['target_backend_fingerprint'] = dict(backend.fingerprint())
            predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                benchmark_id=job['benchmark'], split=job['split'], model_slot=model['id'], seed=17,
                builder_parameters={'schema_version': 1, 'task': 'qa', 'qa': {
                    'output_format': 'structured_json', 'max_tokens': 512, 'use_seed': True}},
                runtime=StaticRuntimeComponents(backend=backend)))
            prediction = predictor.predict(state)
            example = adapters[job['benchmark']].example(job['split'], job['case_id'])
            row.update(status='scored_terminal', metrics=dict(adapters[job['benchmark']].score(example, prediction.prediction)),
                       prediction=to_jsonable(prediction))
        except Exception as exc:
            row.update(status='failed_terminal', error_type=type(exc).__name__, error=str(exc))
        row['calls'] = calls
        save(output / 'terminals' / (job['id'] + '.json'), row)
        (output.parent / 'keep-services-alive').touch()

    pools = {m['id']: ThreadPoolExecutor(2) for m in config['models']}
    futures = []
    try:
        for cell, _split in QA_CELLS:
            for example in selected[cell]:
                original, state = captures[(cell, example.example_id)]
                matching = [j for j in jobs if (j['benchmark'], j['case_id']) == (cell, example.example_id)]
                for job in matching:
                    if job['release'] == 'baseline.no_op':
                        futures.append(pools[job['model']['id']].submit(target, job, state, original.state_hash))
                prep = output / 'preparation' / digest({'cell': cell, 'case_id': example.example_id})
                prep.mkdir()
                worker = RecordedWorker(JsonLineSubprocessWorker(command=(*command, '--checkpoint', str(checkpoint),
                    '--checkpoint-manifest', str(asset_manifest), '--acknowledge-noassertion-checkpoint'),
                    worker_id='toolkit-native-recomp-sentence-extractive', identity=identity,
                    timeout_seconds=900, environment=environment), prep)
                record = None
                try:
                    method = create_intervention(RELEASE, checkpoint_revision=asset['checkpoint_revision'],
                        tokenizer_revision=asset['tokenizer_revision'], sentence_splitter_revision=RECOMP_COMMON_SUPPORT_SPLITTER,
                        asset_manifest_hash=asset_sha, asset_license_acknowledged=True)
                    record = InterventionRunner().run(method, state, InterventionContext(seed=17,
                        retriever=CapturedContextSource(cell, example.example_id, content_hash(original.context)),
                        external_runtime=worker))
                    save(prep / 'terminal.json', {'status': 'native_compressed', 'record': to_jsonable(record),
                        'worker_responses': worker.responses, 'source_captured_state_hash': original.state_hash,
                        'prepared_state_hash': state.state_hash, 'physical_helper_model_calls': record.usage.model_calls,
                        'reference_exposed': False, 'automatic_retry': False})
                except Exception as exc:
                    save(prep / 'terminal.json', {'status': 'failed_preparation', 'worker_responses': worker.responses,
                        'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False})
                for job in matching:
                    if job['release'] == 'baseline.no_op':
                        continue
                    if record is None:
                        save(output / 'terminals' / (job['id'] + '.json'), {'job': job,
                            'status': 'capture_dependency_failed', 'additional_target_attempts': 0, 'calls': [],
                            'preparation_receipt': str(prep / 'terminal.json'), 'automatic_retry': False})
                    else:
                        futures.append(pools[job['model']['id']].submit(target, job, state, original.state_hash, record, prep / 'terminal.json'))
        for future in futures:
            future.result()
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    rows = [json.loads(p.read_text()) for p in (output / 'terminals').glob('*.json')]
    save(output / 'complete.json', {'expected': len(jobs), 'terminal': len(rows),
        'statuses': dict(Counter(r['status'] for r in rows)), 'full_toolkit_acceptance_complete': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'upstream', 'worker-python', 'checkpoint', 'asset-manifest', 'conversion', 'addons', 'observed'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    run(json.loads(a.config.read_text()), a.output, a.upstream, a.worker_python, a.checkpoint,
        a.asset_manifest, a.conversion, a.addons, a.observed)


if __name__ == '__main__':
    main()
