"""Paper Algorithm1 boundary/history fixtures, not model-quality measurements."""
from dataclasses import replace

import pytest

from agentinputlab import ContextBlock, FrozenAgentState, InterventionContext
from agentinputlab.algorithms.clarify import (
    UDCS_DETERMINISTIC_RELEASE,
    UDCS_HISTORY_KEY,
    UDCSPaperSpecDeterministicPolicy,
    UDCSPromptNativePolicy,
    parse_udcs_response,
    route_udcs_decision,
)
from agentinputlab.api import ToolkitService, create_intervention
from agentinputlab.backends import GenerationResponse, ReplayBackend
from agentinputlab.backends.llm import Usage
from agentinputlab.contracts import AlgorithmOutputError, PatchOp, ValidationError
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.runner import InterventionRunner


def response(u=0.75, action="search[shoes]", confidence=0.9):
    return (f"<think>Consider preferences.</think><u_request>{u}</u_request>"
        "<u_request_explanation>Which color do you want?</u_request_explanation>"
        f"<action>{action}</action><confidence>{confidence}</confidence><explanation>Proceed carefully.</explanation>")


@pytest.mark.parametrize("u,threshold,triggered", [(0.49,0.5,False),(0.5,0.5,True),
    (0.51,0.5,True),(0,0,True),(1,1,True),(0.99,1,False)])
def test_algorithm1_inclusive_threshold_boundaries(u,threshold,triggered):
    route=route_udcs_decision(parse_udcs_response(response(u)),threshold)
    assert route.threshold_triggered is triggered
    assert route.routed_action==("request_clarification" if triggered else "search[shoes]")
    assert route.requests_clarification is triggered


def test_below_threshold_preserves_model_clarification_action_instead_of_forcing_continue():
    decision=parse_udcs_response(response(0.1,"request_clarification: What size?"))
    route=route_udcs_decision(decision)
    assert route.requests_clarification and not route.threshold_triggered
    assert route.routed_action==decision.action and route.question=="What size?"


def test_confidence_is_not_the_request_uncertainty_threshold():
    assert route_udcs_decision(parse_udcs_response(response(0.9,confidence=1))).requests_clarification
    assert not route_udcs_decision(parse_udcs_response(response(0.1,confidence=0))).requests_clarification


def test_forced_route_does_not_treat_tool_arguments_as_a_question():
    route=route_udcs_decision(parse_udcs_response(response(0.9,"send_email: recipient@example.test")))
    assert route.question=="Which color do you want?" and route.action_overridden


def test_first_generation_payload_matches_native_prompt_except_local_routing_metadata():
    requests=[]
    from agentinputlab.backends.llm import CallableBackend
    backend=CallableBackend(lambda r: (requests.append(r) or GenerationResponse(
        text=response(),model='fixture',usage=Usage(input_tokens=10,output_tokens=20))),
        revision='fixture',callable_id='synthetic-capture')
    state=FrozenAgentState(query="Buy shoes",context=(ContextBlock(text="A catalog"),))
    context=InterventionContext(backend=backend,seed=17)
    native=UDCSPromptNativePolicy(max_tokens=512).propose(state,context)
    paper=UDCSPaperSpecDeterministicPolicy().propose(state,context)
    a,b=map(to_jsonable,requests)
    a.pop('metadata')
    b.pop('metadata')
    assert a==b
    assert native.operations[0].op is PatchOp.NO_OP
    assert paper.operations[0].op is PatchOp.REQUEST_CLARIFICATION
    assert paper.metadata['threshold_enforced'] and not native.metadata['threshold_enforced']


def test_history_propagates_all_signals_and_clears_old_request_without_fabricated_observation():
    requests=[]
    from agentinputlab.backends.llm import CallableBackend
    values=iter((response(0.8),response(0.1,"continue_task")))
    backend=CallableBackend(lambda r: (requests.append(r) or GenerationResponse(
        text=next(values),model='fixture',usage=Usage(input_tokens=10,output_tokens=20))),
        revision='fixture',callable_id='synthetic-two-steps')
    state=FrozenAgentState(query="Buy shoes",context=(ContextBlock(text="visible observation"),))
    policy=UDCSPaperSpecDeterministicPolicy()
    first=InterventionRunner().run(policy,state,InterventionContext(backend=backend,seed=17))
    next_state=first.patch.apply(state)
    second=InterventionRunner().run(policy,next_state,InterventionContext(backend=backend,seed=17))
    final=second.patch.apply(next_state)
    assert len(final.metadata[UDCS_HISTORY_KEY])==2
    assert final.metadata['clarification_request'] is None
    assert final.metadata[UDCS_HISTORY_KEY][0]['observation']==('visible observation',)
    assert '<u_request>0.8</u_request>' in requests[1].prompt
    assert '<action>request_clarification</action>' in requests[1].prompt
    assert requests[1].prompt.index('<u_request>0.8')<requests[1].prompt.index('<action>request_clarification')
    assert not second.patch.metadata['environment_action_executed']
    assert not second.patch.metadata['user_reply_generated']
    assert first.usage.model_calls==second.usage.model_calls==1


def test_history_guard_refuses_before_generation_without_silent_truncation():
    policy=UDCSPaperSpecDeterministicPolicy(max_history_entries=1)
    state=FrozenAgentState(query="Buy shoes")
    backend=ReplayBackend([response()])
    first=policy.propose(state,InterventionContext(backend=backend))
    with pytest.raises(ValidationError,match='safety bound'):
        policy.propose(first.apply(state),InterventionContext(backend=backend))


@pytest.mark.parametrize('kwargs',[{'clarification_threshold':float('nan')}, {'clarification_threshold':True},
    {'max_tokens':0},{'max_history_entries':False},{'profile_kind':'official_native'}, {'release_id':'other'}])
def test_invalid_constructor_configuration_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        UDCSPaperSpecDeterministicPolicy(**kwargs)


def test_repository_codec_clamps_remain_explicit_and_invalid_probability_is_not_retried():
    policy=UDCSPaperSpecDeterministicPolicy()
    state=FrozenAgentState(query='Buy shoes')
    patch=policy.propose(state,InterventionContext(backend=ReplayBackend([response(1.4)])))
    assert 'clamped_u_request' in patch.trace.warnings
    with pytest.raises(AlgorithmOutputError):
        policy.propose(state,InterventionContext(backend=ReplayBackend(['unparseable'])))
    decision=replace(parse_udcs_response(response()),u_request=float('nan'))
    with pytest.raises(AlgorithmOutputError):
        route_udcs_decision(decision)


def test_registry_and_operator_sdk_surface_preserve_the_distinct_release():
    policy=create_intervention(UDCS_DETERMINISTIC_RELEASE)
    assert isinstance(policy,UDCSPaperSpecDeterministicPolicy)
    service=ToolkitService(factories={UDCS_DETERMINISTIC_RELEASE:lambda: policy},
        context_factory=lambda seed:InterventionContext(backend=ReplayBackend([response()]),seed=seed))
    result=service.execute({'release_id':UDCS_DETERMINISTIC_RELEASE,'seed':17,'state':{'query':'Buy shoes'}})
    assert result['scored'] is False
    assert result['record']['patch']['release_id']==UDCS_DETERMINISTIC_RELEASE
    assert result['state']['metadata']['clarification_request']=='Which color do you want?'
