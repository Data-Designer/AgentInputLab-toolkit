"""Native function-calling BFCL environment for SkillOpt's unchanged optimizer.

The source loader is shared with ACE, not its text-answer adapter or optimizer.
BFCL's official test inventory is subdivided explicitly by the caller; this is
compatibility fitting, not a standard leaderboard training split.
"""
from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.backends import GenerationRequest
from agentinputlab.benchmarks.bfcl_v3 import BFCL_V3_OFFICIAL_SOURCE_REVISION
from agentinputlab.contracts import ContextBlock, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.recipes.ace_bfcl import create_bfcl_feedback


def create_bfcl_environment(*, cfg, configuration, target_backend):
    try:
        from skillopt.envs.base import EnvAdapter
    except ImportError as exc:
        raise ExternalDependencyError('mount the pinned SkillOpt checkout before creating its BFCL environment') from exc
    pin = configuration['source_input']
    source = load_pinned_json_mapping(pin['path'], expected_sha256=pin['sha256'], max_bytes=8 * 1024 * 1024)
    prepared = create_bfcl_feedback(source_input=pin,
        model_id=configuration['backend']['model_id'],
        partition_label=configuration['partition_label'], output_root=None)
    adapter = prepared.adapter
    groups = dict(train=prepared.train, validation=prepared.validation)
    cap, seed = configuration['per_call_output_tokens'], configuration['seed']

    def binding():
        adapter.verify_integrity()
        return to_jsonable(dict(prepared_files=dict(adapter.prepared.file_sha256),
            scorer=adapter.evaluation_provenance))

    original_binding = binding()
    owned = {}
    for role, examples in groups.items():
        for example in examples:
            functions = adapter.function_tools(example)
            state = example.visible_state()
            state = replace(state, tool_names=tuple(f['function']['name'] for f in functions),
                context=(*state.context, ContextBlock(
                    block_id='native_function_catalog',
                    text=json.dumps(to_jsonable(functions), sort_keys=True),
                    source=BFCL_V3_OFFICIAL_SOURCE_REVISION)))
            owned[content_hash(example.example_id)] = (example, state, functions, role)
    # Old saved sources may include their exact visible states. Verify these
    # against the native adapter instead of trusting a caller-supplied prompt.
    if 'rows' in source:
        rows = [r for r in source['rows'] if r.get('role') in groups]
        if len(rows) != len(owned) or len({r['case_id'] for r in rows}) != len(rows):
            raise ValidationError('BFCL captured fit rows differ from declared groups')
        from agentinputlab.api import _state

        for row in rows:
            item = owned.get(content_hash(row['case_id']))
            if item is None:
                raise ValidationError('BFCL captured fit row is outside declared groups')
            example, state, functions, role = item
            captured = _state(row['state'])
            # The source fixes the experiment budget; the bare dataset adapter
            # has no deployment context-window limit. Preserve that explicit
            # budget without accepting changed task content or tool schemas.
            state = replace(state, budget=captured.budget)
            if (row['role'] != role or row['state_hash'] != state.state_hash
                    or captured.state_hash != state.state_hash
                    or row['functions'] != to_jsonable(functions)):
                raise ValidationError('BFCL captured visible state or tools changed')
            owned[content_hash(row['case_id'])] = (example, state, functions, role)
    if any(cap > state.budget.max_output_tokens or state.budget.max_model_calls < 1
           for _, state, _, _ in owned.values()):
        raise ValidationError('BFCL rollout cap exceeds the declared state budget')

    def visible_item(example):
        return dict(id=content_hash(example.example_id), question=example.query,
                    task_type=adapter.benchmark_id)

    class BFCLEnvAdapter(EnvAdapter):
        def __init__(self):
            self.analyst_workers = cfg.get('analyst_workers', 1)
            self.failure_only = cfg.get('failure_only', False)
            self.minibatch_size = cfg.get('minibatch_size', 1)
            self.edit_budget = cfg.get('edit_budget', 4)

        def fingerprint(self):
            return dict(implementation='agentinputlab.skillopt.native_BFCL_environment.v1',
                source_input=to_jsonable(pin), native_binding=original_binding,
                backend=to_jsonable(target_backend.fingerprint()),
                groups={role:[dict(case_id=e.example_id,
                    state_hash=owned[content_hash(e.example_id)][1].state_hash) for e in examples]
                    for role, examples in groups.items()}, metric='ast_accuracy',
                per_call_output_tokens=cap, seed=seed, heldout_not_exposed=True,
                partition_label=configuration['partition_label'],
                native_tools_and_AST_feedback=True, automatic_retry=False,
                native_reflection_patch_merge_gate_preserved=True,
                caller_declared_disjoint_compatibility_groups_NOT_standard_benchmark_quality=True)

        def _batch(self, role, count, batch_seed):
            if type(count) is not int or count < 1 or count > len(groups[role]):
                raise ValidationError('BFCL native fit batch exceeds declared group')
            return [visible_item(e) for e in random.Random(batch_seed).sample(list(groups[role]), count)]

        def build_train_env(self, batch_size, seed, **kwargs):
            return self._batch('train', batch_size, seed)

        def build_eval_env(self, env_num, split, seed, **kwargs):
            if split not in ('val', 'validation', 'valid_seen'):
                raise ValidationError('BFCL fit exposes validation only; heldout remains separate')
            return self._batch('validation', env_num, seed)

        def get_task_types(self):
            return [adapter.benchmark_id]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            if not isinstance(skill_content, str) or not skill_content.strip():
                raise ValidationError('BFCL current skill must be nonempty')
            if any(kwargs.get(k) for k in ('diagnostic_mode', 'diagnostic_instruction', 'diagnostic_trace_context_by_id')):
                raise ValidationError('BFCL extra diagnostic attempts are not admitted')
            if binding() != original_binding:
                raise ValidationError('BFCL source/scorer binding changed')
            items = list(env_manager)
            if any(not isinstance(item, dict) or item.get('id') not in owned for item in items):
                raise ValidationError('Undeclared BFCL fit case')
            if len({item['id'] for item in items}) != len(items):
                raise ValidationError('Duplicate BFCL fit case')
            if any(item != visible_item(owned[item['id']][0]) for item in items):
                raise ValidationError('BFCL fit input changed')
            root = Path(out_dir)
            root.mkdir(parents=True, exist_ok=True)
            results = []
            for item in items:
                example, state, functions, _ = owned[item['id']]
                state = replace(state, context=(*state.context, ContextBlock(
                    text='SKILLOPT CURRENT FITTING SKILL\n' + skill_content,
                    block_id='skillopt-fit-skill:' + content_hash(skill_content),
                    source='SkillOpt native fitting candidate')))
                request = GenerationRequest(prompt=state.query,
                    system='\n\n'.join(x for x in [state.instruction, *[b.text for b in state.context]] if x),
                    tools=functions, max_tokens=cap, temperature=0, seed=seed)
                folder = root / 'predictions' / item['id']
                folder.mkdir(parents=True, exist_ok=False)
                try:
                    response = target_backend.generate(request)
                    if response.usage.output_tokens > cap:
                        raise ValidationError('BFCL provider exceeded declared per-call cap')
                    message = response.raw['choices'][0]['message']
                    prediction = [{c['function']['name']: c['function']['arguments']}
                        for c in (message.get('tool_calls') or ())]
                    metrics = dict(adapter.score(example, prediction))
                    score = metrics.get('ast_accuracy')
                    if type(score) not in (float, int) or score not in (0.0, 1.0):
                        raise ValidationError('BFCL native ast_accuracy must be zero or one')
                    score = float(score)
                except Exception as exc:
                    with (folder / 'consumed-execution-error.json').open('x') as f:
                        json.dump(dict(error_type=type(exc).__name__, automatic_retry=False), f)
                    raise
                trace = json.dumps(to_jsonable(message), sort_keys=True, ensure_ascii=False)
                result = dict(id=item['id'], source_case_id=example.example_id, question=state.query,
                    task_type=adapter.benchmark_id, hard=int(score), soft=score, response=trace,
                    agent_ok=True, n_turns=1, fail_reason='' if score else 'native BFCL AST mismatch',
                    native_prediction=prediction, native_metrics=metrics,
                    generation_request=to_jsonable(request), generation_response=to_jsonable(response),
                    AST_scoring_provenance=to_jsonable(adapter.evaluation_provenance),
                    native_tools_unchanged=True, usage=dict(model_calls=1,
                        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens))
                conversation = [dict(type='message', turn=1, content=trace),
                    dict(role='system', content='[NATIVE BFCL AST EVALUATION]\n' +
                        json.dumps(dict(metric='ast_accuracy', score=score)))]
                for name, value in (('result.json', result), ('conversation.json', conversation)):
                    with (folder / name).open('x') as f:
                        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
                results.append(result)
            with (root / 'results.jsonl').open('x') as f:
                for row in results:
                    f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            return results

    return BFCLEnvAdapter()
