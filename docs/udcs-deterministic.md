# UDCS: deterministic paper-step routing

`clarify.udcs.paper_spec_deterministic_route` implements Algorithm 1 of
[UDCS, arXiv:2606.19559v1](https://arxiv.org/html/2606.19559v1#S4).
The numerical guard routes to clarification when `u_request >= threshold`;
otherwise it retains the emitted action. Consequently, a model-proposed
clarification below the threshold is retained too. Action confidence is recorded
separately and does not replace request uncertainty in this guard.

```python
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab import FrozenAgentState, InterventionContext

policy = create_intervention(
    "clarify.udcs.paper_spec_deterministic_route",
    clarification_threshold=0.5,
    max_tokens=512,
)
state = FrozenAgentState(query="Find a suitable pair of shoes.")
# backend is an operator-configured chat backend with fingerprint() and generate().
result = run_intervention(policy, state, InterventionContext(backend=backend, seed=17))
```

This entrypoint performs one model call and returns an input patch. It does not
execute tool actions, invent user answers, or stand in for an entire benchmark
agent. A clarification request appears in `state.metadata.clarification_request`.
The observed step, reasoning, request uncertainty, explanations, routed action
and confidence are preserved under `udcs.paper_spec.history.v1`. Applying the
patch and passing the resulting visible state to the next step includes those
fields in its history prompt. Supply real subsequent observations/replies from
the benchmark; do not synthesize them to make a trajectory succeed. A continued
non-clarification route clears a stale active request but retains its historical
step. The default 64-entry safety guard rejects before another model call;
it never silently discards history. This guard is not a paper hyperparameter.

The existing `clarify.udcs.repo_native_prompt_route` remains unchanged: it leaves
the threshold decision to the model's emitted action. The new profile preserves
the same six-field prompt and repository-compatible parser, including explicit
clamp warnings and action fallback. It changes routing and history propagation;
therefore old end-task scores cannot simply be relabeled as new-profile scores.
For an empty UDCS history, generation inputs match the prompt-native component;
release/routing metadata remains distinct. Controlled and factorial modes do
not claim historical author environment or complete native-runner parity.

The bounded ClariQ acceptance operator separates the benchmark predictor's JSON
instruction from the clarification policy's XML output contract. It preserves
the original source query/context, hidden labels and native scorer, and records
both visible-state hashes. Its binary route maps to native need labels 1 (no
request) and 2 (any request); it does not infer higher ordinal need grades.
The same initial-query BM25 ranking over the complete small question bank is
shared across readers for RQ2 scoring, not presented as a UDCS ranking gain.
Failed model cases remain missing in the native selected-topic denominator;
their raw responses and consumed budgets are saved before scoring. This is
fixed-subset compatibility evidence, not full-dev performance or an LLM judge.
