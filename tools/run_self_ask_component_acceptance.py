"""Source-pinned first-stage Self-Ask and separate reader on fixed real QA cases."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_plan_budget_component_acceptance import remaining_budget
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.decompose.self_ask import SelfAskConfig
from agentinputlab.api import create_intervention
from agentinputlab.backends import FINAL_CONTENT_STOP_POLICY_ID, FinalContentStopBackend
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.drivers.static_production import (
    StaticPredictorBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.experiments.production_builders import ProductionPredictorBuilder
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASE = 'plan.self_ask.planner_component'
NOTEBOOK_SHA256 = '9645cde930a24d943d6fd0237c2c158dc37dc04a17a30aca29017896d7e1c3c2'


def selected_qa_cells(config):
    names = config.get('qa_cells', [cell for cell, _ in QA_CELLS])
    known = {cell for cell, _ in QA_CELLS}
    if (not isinstance(names, list) or not names or any(not isinstance(v, str) for v in names)
            or len(set(names)) != len(names) or not set(names) <= known):
        raise ValueError('choose a nonempty unique subset of the four prepared QA cells')
    return tuple((cell, split) for cell, split in QA_CELLS if cell in names)


def selected_final_stop_models(config):
    names = config.get('final_content_stop_model_ids', [])
    known = {model['id'] for model in config['models']}
    if (not isinstance(names, list) or any(not isinstance(v, str) for v in names)
            or len(set(names)) != len(names) or not set(names) <= known):
        raise ValueError('final-content stop policy requires explicit configured model IDs')
    return frozenset(names)


def native_demonstrations(upstream):
    path = upstream / 'upstream-paper/self-ask/self-ask_plus_search-engine_demo.ipynb'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValidationError('source Self-Ask notebook checksum differs')
    values = []
    for cell in json.loads(raw)['cells']:
        if cell.get('cell_type') != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'prompt'):
                values.append(ast.literal_eval(node.value))
    suffix = '\n\nQuestion: '
    if len(values) != 1 or not values[0][0].endswith(suffix):
        raise ValidationError('source Self-Ask demonstration serialization differs')
    return values[0][0][:-len(suffix)]


def run(config, output):
    qa_cells, final_stop_models = selected_qa_cells(config), selected_final_stop_models(config)
    bindings = verify_services(config['models'])
    demonstrations = native_demonstrations(Path(config['upstream_root']))
    method_config = SelfAskConfig(demonstrations=demonstrations, max_tokens_per_call=512)
    offset = config.get('component_case_offset', 92)
    if type(offset) is not int or offset < 0:
        raise ValueError('source case offset must be non-negative')
    adapters = {cell: load_production_benchmark_adapter(cell, config['prepared_root']) for cell, _ in qa_cells}
    cases = {cell: tuple(sorted(adapters[cell].examples(split), key=lambda e: e.example_id)[offset:offset + 2])
             for cell, split in qa_cells}
    if len(config['models']) != 4 or any(len(rows) != 2 for rows in cases.values()):
        raise ValueError('four original models and two fixed cases per QA required')
    jobs = []
    for model in config['models']:
        for cell, split in qa_cells:
            for example in cases[cell]:
                for release in ('baseline.no_op', RELEASE):
                    job = {'model': model, 'benchmark': cell, 'split': split,
                        'case_id': example.example_id, 'release': release, 'seed': 17,
                        'budget': {'max_model_calls': 8, 'max_output_tokens': 4096},
                        'per_call_output_cap': 512, 'demonstration_sha256': content_hash(demonstrations),
                        'first_step_only': release == RELEASE,
                        'optimizer_input_projection': 'query_only' if release == RELEASE else 'none',
                        'search_executed': False, 'iterative_solver_executed': False,
                        'separate_reader_protocol': 'original_visible_state_plus_initial_continuation',
                        'support_evidence_labels_inferred_from_plan': False}
                    job['stop_policy'] = (FINAL_CONTENT_STOP_POLICY_ID if model['id'] in final_stop_models
                        else 'gpt-oss.final-channel-stop.v1' if model['id'] == 'gpt-oss-20b' else 'native')
                    job['id'] = digest(job)
                    jobs.append(job)
    if config.get('previous_observed_inventory'):
        previous = json.loads(Path(config['previous_observed_inventory']).read_text())
        if previous.get('schema') != 'agentinputlab.acceptance-observed-evidence.v1':
            raise ValidationError('previous inventory schema differs')
        old = {(row['release'], row['benchmark'], row['backbone'], evidence['case_id'])
               for row in previous['rows'] for evidence in row['evidence']}
        if any((j['release'], j['benchmark'], j['model']['id'], j['case_id']) in old for j in jobs):
            raise ValidationError('frozen cohort overlaps previously attempted case/profile/model identities')
    output.mkdir(exist_ok=False)
    for name in ('claims', 'terminals'):
        (output / name).mkdir()
    save(output / 'manifest.json', {'jobs': jobs, 'model_bindings': bindings,
        'selection': f'fixed_source_lexicographic_indices{offset}:{offset + 2}_before_outputs',
        'source_hashes': {cell: dict(a.prepared.file_sha256) for cell, a in adapters.items()},
        'source_revisions': {cell: a.prepared.source_revision for cell, a in adapters.items()},
        'native_notebook_sha256': NOTEBOOK_SHA256, 'demonstration_sha256': content_hash(demonstrations),
        'eligibility': 'controlled_SelfAsk_initial_continuation_component_not_full_loop',
        'automatic_retry': False, 'new_fit': 0, 'asset_acquisition_bytes': 0, 'paper_results_changed': False})

    def one(job):
        save(output / 'claims' / (job['id'] + '.json'), {'job': job, 'pid': os.getpid(), 'started': time.time()})
        model, adapter = job['model'], adapters[job['benchmark']]
        example = adapter.example(job['split'], job['case_id'])
        original = example.visible_state(budget=ExecutionBudget(**job['budget']))
        calls, record = [], None
        inner = RecordingBackend(calls=calls, model_id=model['id'], base_url=model['base_url'],
            extra_body=model.get('extra_body', {}), timeout_seconds=180, transport=local_transport,
            deployment_identity={'revision': digest(bindings[model['id']]['model_json_hashes']), 'server': 'vllm'})
        backend = (FinalContentStopBackend(inner) if model['id'] in final_stop_models
            else GPTOSSFinalChannelStopBackend(inner) if model['id'] == 'gpt-oss-20b' else inner)
        monitor, started, stage = RunMonitor(), time.monotonic(), 'initial_self_ask_component'
        row = {'job': job, 'input_state_hash': original.state_hash,
            'model_binding': bindings[model['id']], 'automatic_retry': False,
            'package_location': __import__('agentinputlab').__file__, 'additional_attempts': 1}
        try:
            state = original
            if job['release'] == RELEASE:
                record = InterventionRunner(monitor=monitor).run(
                    create_intervention(RELEASE, config=method_config), original,
                    InterventionContext(backend=backend, seed=17))
                state = record.patch.apply(original)
                if (state.query, state.context, state.messages) != (original.query, original.context, original.messages):
                    raise ValidationError('planner component changed reader query/context/history')
                if len(calls) != 1:
                    raise ValidationError('initial planner did not use exactly one generation')
            stage = 'separate_downstream_reader'
            state = replace(state, budget=remaining_budget(original.budget, calls))
            predictor = ProductionPredictorBuilder().build(StaticPredictorBuildRequest(
                benchmark_id=job['benchmark'], split=job['split'], model_slot=model['id'], seed=17,
                builder_parameters={'schema_version': 1, 'task': 'qa', 'qa': {
                    'output_format': 'answer_only', 'max_tokens': 512, 'use_seed': True}},
                runtime=StaticRuntimeComponents(backend=backend)))
            prediction = predictor.predict(state)
            stage = 'native_scoring'
            row.update(status='scored_terminal', prediction=to_jsonable(prediction),
                       metrics=dict(adapter.score(example, prediction.prediction)))
        except Exception as exc:
            row.update(status='failed_terminal', stage=stage, error_type=type(exc).__name__, error=str(exc))
        row.update(calls=calls, model_calls=len(calls), intervention=to_jsonable(record) if record else None,
                   monitor_snapshots=to_jsonable(monitor.list_snapshots()), seconds=time.monotonic() - started,
                   physical_tool_or_retrieval_calls=0)
        save(output / 'terminals' / (job['id'] + '.json'), row)
        (output.parent / 'keep-services-alive').touch()

    def lane(model):
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(one, [j for j in jobs if j['model']['id'] == model['id']]))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lane, config['models']))
    rows = [json.loads(p.read_text()) for p in (output / 'terminals').glob('*.json')]
    save(output / 'complete.json', {'expected': len(jobs), 'terminal': len(rows),
        'statuses': dict(Counter(row['status'] for row in rows)),
        'full_toolkit_acceptance_complete': False, 'paper_results_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
