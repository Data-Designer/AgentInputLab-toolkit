"""Projection fixtures only; no source rows, model or evaluator are mocked as real results."""
import run_udcs_deterministic_acceptance as worker

from agentinputlab import ContextBlock, ExecutionBudget
from agentinputlab.benchmarks.base import BenchmarkExample


def test_policy_projection_preserves_source_query_context_and_hides_evaluator_labels():
    example=BenchmarkExample(example_id='synthetic',query='A source initial request',
        instruction='Return a benchmark-predictor JSON object',
        visible_context=(ContextBlock(text='Visible source content'),),
        visible_metadata={'source_revision':'synthetic'},
        evaluator_gold={'clarification_need':4,'forbidden':'hidden label'})
    budget=ExecutionBudget(max_model_calls=8,max_output_tokens=4096)
    source,state=worker.project(example,budget)
    assert state.query==source.query and state.context==source.context
    assert state.budget==source.budget==budget
    assert state.instruction==worker.ROUTING_INSTRUCTION
    assert state.tool_names==('continue_task',)
    assert 'clarification_need' not in state.metadata and 'forbidden' not in state.metadata
    assert source.state_hash!=state.state_hash
