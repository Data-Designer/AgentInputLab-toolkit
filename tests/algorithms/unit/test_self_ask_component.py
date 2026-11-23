"""Synthetic first-stage conformance tests, not live-model result evidence."""
from dataclasses import replace

import pytest

from agentinputlab.algorithms.decompose.self_ask import (
    SelfAskConfig,
    SelfAskLoop,
    SelfAskPlannerComponent,
)
from agentinputlab.backends.llm import CallableBackend, GenerationResponse, ModelCapability, Usage
from agentinputlab.backends.retrieval.memory import InMemoryBM25Retriever
from agentinputlab.contracts import (
    AlgorithmOutputError,
    BudgetExceededError,
    ContextBlock,
    ExecutionBudget,
    FrozenAgentState,
    InterventionContext,
    Message,
    ValidationError,
)
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

DEMO = 'Question: Synthetic demo?\nAre follow up questions needed here: No.\nSo the final answer is: A.'


def backend(text):
    return CallableBackend(lambda _: GenerationResponse(text=text, model='synthetic',
        usage=Usage(input_tokens=31, output_tokens=9)), model_id='synthetic',
        revision='synthetic-self-ask-first-stage-v1', callable_id='unit.self_ask_component',
        capabilities=frozenset({ModelCapability.COMPLETION, ModelCapability.SEED}))


@pytest.mark.parametrize('cap,seed', [(512, 17), (64, 11), (7, 2)])
def test_all_model_visible_first_request_fields_match_existing_full_loop(cap, seed):
    state = FrozenAgentState(query='Synthetic target?', budget=ExecutionBudget(max_output_tokens=cap))
    cfg = SelfAskConfig(demonstrations=DEMO, max_tokens_per_call=512)
    full, component = backend('No.\nSo the final answer is: answer'), backend('No.\nSo the final answer is: answer')
    SelfAskLoop(cfg).run(state, InterventionContext(backend=full, retriever=InMemoryBM25Retriever(()), seed=seed))
    SelfAskPlannerComponent(cfg).propose(state, InterventionContext(backend=component, seed=seed))
    assert len(full.calls) == len(component.calls) == 1
    # Only release audit identity differs; prompt/system/messages/stops/control fields are identical.
    assert replace(full.calls[0], metadata={}) == replace(component.calls[0], metadata={})
    assert component.calls[0].max_tokens == cap
    assert component.calls[0].stop == ('\nIntermediate answer:',)


def test_followup_materializes_initial_continuation_without_search_or_reader_changes():
    state = FrozenAgentState(query='Synthetic query?', instruction='answer only',
        context=(ContextBlock(text='original context'),), messages=(Message(role='user', content='history'),))
    model = backend('Yes.\nFollow up: first question?')
    patch = SelfAskPlannerComponent(SelfAskConfig(demonstrations=DEMO)).propose(
        state, InterventionContext(backend=model, seed=17))
    updated = patch.apply(state)
    assert len(model.calls) == 1
    assert (updated.query, updated.context, updated.messages) == (state.query, state.context, state.messages)
    assert 'Follow up: first question?' in updated.instruction
    assert not patch.metadata['search_executed'] and not patch.metadata['full_loop']
    assert patch.metadata['planner_decision'] == 'first_followup'
    assert patch.trace.counters['retrieval_calls'] == 0
    assert 'original context' not in model.calls[0].prompt and 'history' not in model.calls[0].prompt


@pytest.mark.parametrize('text', ['No.', 'No', 'Are follow up questions needed here: No.',
                                '\nAre follow up questions needed here:\nNo.  '])
def test_native_no_decision_is_a_paid_noop_not_a_claimed_plan(text):
    state = FrozenAgentState(query='Synthetic query?')
    patch = SelfAskPlannerComponent(SelfAskConfig(demonstrations=DEMO)).propose(
        state, InterventionContext(backend=backend(text)))
    assert patch.apply(state) == state
    assert patch.trace.counters['model_calls'] == 1
    assert patch.metadata['planner_decision'] == 'no_followup'


@pytest.mark.parametrize('text', ['', 'not a valid initial continuation', 'No. Extra explanation',
                                'Are follow up questions needed here: Yes.', 'Nope'])
def test_paid_empty_or_unparseable_continuation_keeps_known_usage(text):
    monitor = RunMonitor()
    with pytest.raises(AlgorithmOutputError):
        InterventionRunner(monitor=monitor).run(
            SelfAskPlannerComponent(SelfAskConfig(demonstrations=DEMO)),
            FrozenAgentState(query='Synthetic target?'), InterventionContext(backend=backend(text)))
    usage = monitor.list_snapshots()[-1].usage
    assert usage.model_calls == 1 and usage.input_tokens == 31 and usage.output_tokens == 9


def test_unavailable_generation_budget_refused_before_any_call():
    model = backend('No.')
    with pytest.raises(BudgetExceededError):
        SelfAskPlannerComponent(SelfAskConfig(demonstrations=DEMO)).propose(
            FrozenAgentState(query='q', budget=ExecutionBudget(max_model_calls=0)),
            InterventionContext(backend=model))
    assert model.calls == []


def test_component_constructor_rejects_ambiguous_configs():
    with pytest.raises(ValidationError):
        SelfAskPlannerComponent('not config')


@pytest.mark.parametrize('text', [
    'No.\n\nExplanation: This fact can be checked directly.',
    'No.  \nThis is a command, not a question.',
    'Are follow up questions needed here: No.\n**Reasoning:** no extra query.',
])
def test_explicit_no_explanation_policy_reuses_one_paid_response(text):
    state = FrozenAgentState(query='Synthetic target?')
    cfg = SelfAskConfig(demonstrations=DEMO)
    strict, compatible = backend(text), backend(text)
    with pytest.raises(AlgorithmOutputError):
        SelfAskPlannerComponent(cfg).propose(state, InterventionContext(backend=strict))
    patch = SelfAskPlannerComponent(cfg, allow_no_decision_explanation=True).propose(
        state, InterventionContext(backend=compatible))
    assert strict.calls == compatible.calls and len(compatible.calls) == 1
    assert patch.apply(state) == state
    assert patch.trace.counters['output_tokens'] == 9
    assert patch.metadata['compatibility_applied'] is True
    assert patch.metadata['response_compatibility'] == 'standalone_no_with_explanation.v1'


@pytest.mark.parametrize('text', [
    'Yes.\nExplanation without a follow-up.',
    'No. Inline explanation is not a standalone decision.',
    'No follow-up questions are needed.',
    'No.\nFollow up:', 'No.\nfollow up:',
    'No.\nSO THE FINAL ANSWER IS:', 'No.\nYes.',
    'No.\nAre follow up questions needed here: Yes.',
])
def test_no_explanation_policy_does_not_guess_or_drop_conflicting_markers(text):
    with pytest.raises(AlgorithmOutputError):
        SelfAskPlannerComponent(SelfAskConfig(demonstrations=DEMO),
            allow_no_decision_explanation=True).propose(
                FrozenAgentState(query='q'), InterventionContext(backend=backend(text)))


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_no_explanation_policy_requires_explicit_boolean(value):
    with pytest.raises(ValidationError, match='must be boolean'):
        SelfAskPlannerComponent(SelfAskConfig(demonstrations=DEMO),
                                allow_no_decision_explanation=value)


def test_no_explanation_policy_is_available_in_public_run_configuration():
    from agentinputlab.configuration import RunConfig
    config = RunConfig.from_dict({'release_id': 'plan.self_ask.planner_component',
        'parameters': {'config': {'demonstrations': DEMO},
                       'allow_no_decision_explanation': True}})
    assert config.create_intervention().allow_no_decision_explanation is True
