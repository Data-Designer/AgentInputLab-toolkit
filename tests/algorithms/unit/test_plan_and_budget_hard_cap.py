"""Hard-cap extension fixtures; not real benchmark or model evidence."""
from dataclasses import replace

import pytest

from agentinputlab import ContextBlock, ExecutionBudget, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.decompose.plan_and_budget import (
    PAB_HARD_CAP_RELEASE,
    CachedPlanBudgetRecord,
    PlanAndBudgetHardCapExtension,
    allocate_hard_token_caps,
)
from agentinputlab.api import create_intervention, run_composite_intervention
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, ModelCapability, Usage
from agentinputlab.contracts import (
    AlgorithmOutputError,
    ArtifactError,
    BudgetExceededError,
    ValidationError,
)
from agentinputlab.runner import CompositeInterventionRunner


def model(*texts):
    values=iter(texts)
    return CallableBackend(lambda r: GenerationResponse(next(values),usage=Usage(3,min(2,r.max_tokens))),
        revision='fixture',callable_id='hard-cap-fixture',capabilities=frozenset({
            ModelCapability.CHAT,ModelCapability.COMPLETION,ModelCapability.JSON_OBJECT,ModelCapability.SEED}))


def cached():
    return {'id':CachedPlanBudgetRecord('id','Question?',('First','Second'),(80,20),3)}


def state(**kwargs):
    return FrozenAgentState(query='Question?',task_id='id',**kwargs)


@pytest.mark.parametrize('credits,total,limit,expected',[
    ((80,20),100,512,(80,20)),((99,1),1000,512,(512,488)),
    ((0,100),1024,512,(1,512)),((1,1,1),5,512,(2,2,1)),
    ((1,1),100,1,(1,1)),((1,1),2,512,(1,1))])
def test_minimum_weighted_extras_saturation_and_unused_pool(credits,total,limit,expected):
    caps=allocate_hard_token_caps(credits,total,limit)
    assert caps==expected and sum(caps)<=total and max(caps)<=limit and min(caps)>=1


@pytest.mark.parametrize('schedule',['constant','linear','polynomial','exponential','cosine'])
def test_every_source_schedule_preserves_global_and_step_caps(schedule):
    for total in range(5,101):
        caps=allocate_hard_token_caps((1,2,3,4,5),total,17,schedule)
        assert sum(caps)==min(total,85) and all(1<=v<=17 for v in caps)
    assert allocate_hard_token_caps((1,1,1),60,512,schedule)[0]>=allocate_hard_token_caps((1,1,1),60,512,schedule)[-1]


@pytest.mark.parametrize('credits,total,limit',[((),10,5),((1,),True,5),((True,),10,5),
    ((float('nan'),),10,5),((0,0),10,5),((float('inf'),),10,5),((1e308,1e308),10,5),((1,),10,False)])
def test_invalid_allocator_parameters_fail_closed(credits,total,limit):
    with pytest.raises(ValidationError):
        allocate_hard_token_caps(credits,total,limit)


def test_cached_full_workflow_actual_decode_caps_seed_history_and_no_hidden_gold():
    backend=model('First answer','Second answer','Final answer')
    original=state(context=(ContextBlock('Visible reference',block_id='p',metadata={'gold':'HIDDEN'}),),
        metadata={'answer':'HIDDEN'},budget=ExecutionBudget(max_output_tokens=30,max_model_calls=3))
    policy=create_intervention(PAB_HARD_CAP_RELEASE,records=cached(),output_format='Answer text',
        per_step_max_tokens=10,final_max_tokens=8)
    result=CompositeInterventionRunner().run(policy,original,InterventionContext(backend=backend,seed=17))
    assert result.result.final_answer=='Final answer' and result.usage.model_calls==3
    assert result.usage.output_tokens==6 and result.result.evidence==original.context
    assert [r.seed for r in backend.calls]==[17,18,19]
    assert [r.max_tokens for r in backend.calls]==[10,10,8]
    assert 'First answer' in backend.calls[1].prompt and 'Second answer' in backend.calls[-1].prompt
    assert 'HIDDEN' not in repr(backend.calls)
    assert result.result.metadata['per_step_hard_token_cap']


def test_live_native_preparation_plus_steps_plus_final_share_budget():
    backend=model('1. First\nHint: A\n2. Second\nHint: B',
        '{"problem":{"evaluated_level":3},"1":{"evaluated_level":2,"credit":80},'
        '"2":{"evaluated_level":2,"credit":20}}','A','B','Final')
    policy=PlanAndBudgetHardCapExtension(output_format='Answer text')
    result=CompositeInterventionRunner().run(policy,state(),InterventionContext(backend=backend,seed=17))
    assert result.usage.model_calls==5 and result.usage.output_tokens==10
    assert [r.seed for r in backend.calls]==[17,18,19,20,21]
    assert all(r.max_tokens<=512 for r in backend.calls)
    assert all(r.metadata['release_id']==PAB_HARD_CAP_RELEASE for r in backend.calls)
    assert result.result.metadata['step_caps']==(409,103)


def test_live_plan_cannot_silently_drop_steps_when_call_budget_insufficient():
    backend=model('1. First\nHint: A\n2. Second\nHint: B',
        '{"problem":{"evaluated_level":3},"1":{"evaluated_level":2,"credit":80},'
        '"2":{"evaluated_level":2,"credit":20}}')
    runner=CompositeInterventionRunner()
    with pytest.raises(BudgetExceededError,match='no steps silently dropped'):
        runner.run(PlanAndBudgetHardCapExtension(output_format='Answer text'),
            state(budget=ExecutionBudget(max_model_calls=4)),InterventionContext(backend=backend))
    assert len(backend.calls)==2 and runner.monitor.list_snapshots()[0].usage.model_calls==2


@pytest.mark.parametrize('budget',[ExecutionBudget(max_model_calls=2),ExecutionBudget(max_output_tokens=2)])
def test_cached_insufficient_budget_rejected_before_call(budget):
    backend=model()
    with pytest.raises(BudgetExceededError):
        PlanAndBudgetHardCapExtension(output_format='text',records=cached()).run(
            state(budget=budget),InterventionContext(backend=backend))
    assert not backend.calls


def test_empty_intermediate_is_paid_failure_without_final_fallback_or_retry():
    backend=model('')
    runner=CompositeInterventionRunner()
    with pytest.raises(AlgorithmOutputError,match='empty text; no retry'):
        runner.run(PlanAndBudgetHardCapExtension(output_format='text',records=cached()),
            state(),InterventionContext(backend=backend))
    assert len(backend.calls)==1 and runner.monitor.list_snapshots()[0].usage.model_calls==1


def test_reported_over_cap_preparation_cannot_continue_even_without_observed_runner():
    backend=CallableBackend(lambda r:GenerationResponse('1. A',usage=Usage(1,r.max_tokens+1)),
        revision='fixture',callable_id='bad-provider',capabilities=frozenset({
            ModelCapability.CHAT,ModelCapability.COMPLETION,ModelCapability.JSON_OBJECT}))
    with pytest.raises(BudgetExceededError,match='preparation exceeded'):
        PlanAndBudgetHardCapExtension(output_format='text').run(state(),InterventionContext(backend=backend))
    assert len(backend.calls)==1


def test_cache_mismatch_is_not_live_fallback():
    with pytest.raises(ArtifactError,match='ID/query mismatch'):
        PlanAndBudgetHardCapExtension(output_format='text',records=cached()).run(
            replace(state(),task_id='other'),InterventionContext(backend=model()))


def test_python_composite_api_keeps_answer_and_observed_record_without_fabricating_score():
    result=run_composite_intervention(
        create_intervention(PAB_HARD_CAP_RELEASE,records=cached(),output_format='text'),
        state(),InterventionContext(backend=model('A','B','Final'),seed=17))
    assert result['final_answer']=='Final' and result['record']['usage']['model_calls']==3
    assert result['scored'] is False and result['benchmark_score'] is None
    assert 'state' not in result


@pytest.mark.parametrize('kwargs',[{'final_max_tokens':True},{'per_step_max_tokens':0},{'total_step_tokens':0},
    {'initial_level':True},{'temperature':float('nan')},{'records':[]},{'profile_kind':'official_native'}])
def test_invalid_constructor_controls(kwargs):
    with pytest.raises((ValidationError,ArtifactError)):
        PlanAndBudgetHardCapExtension(output_format='text',**kwargs)
