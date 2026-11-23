# Native DecomP planning component

`plan.decomp.planner_component` executes exactly one transition of the pinned
DecomP controller, using its original planner participant, demonstrations,
completion prompt, output parser and branch scores. It stops before routing
the generated subquestion to an executor or QA participant. This is an
unexecuted next-subquestion component, not a whole plan, full solver, answer
extractor, or the separate common-support ODQA implementation.

```python
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab import InterventionContext
from agentinputlab.backends import DecompCompletionBackend
from agentinputlab.integrations.external_workers import DecompPlannerLiveRuntime

# The backend must preserve the source's raw completion parameters. The legacy
# worker environment and external checkout are the same pinned prerequisites
# described for DecompLiveRuntime, including the exact dependency versions.
runtime = DecompPlannerLiveRuntime(
    python_executable="/path/to/legacy-worker/bin/python",
    checkout="/path/to/DecomP",
    nltk_data="/path/to/nltk_data",
    backend=backend,  # explicit DecompCompletionBackend; never inferred from state
    per_call_output_cap=512,
    # Optional: use your provider's enforced context-window upper bound when
    # the task has a finite cumulative max_context_tokens budget.
    # per_call_input_token_bound=40960,
)
method = create_intervention(
    "plan.decomp.planner_component",
    config_artifact_id="decomp.commaqa.coarse.p1",
)
result = run_intervention(method, state,
    InterventionContext(backend=backend, external_runtime=runtime, seed=17))
```

The component has its own request schema and runtime identity. A full-system
request cannot run through its dedicated runtime; component responses cannot
satisfy `plan.decomp.full`. The worker loads the declared source config, checks
that its starting participant is a native planner, and uses the original
best-first frontier with exactly one controller execution. It retains all
first-step candidates and selects the same minimum-score frontier state as
the native search. Router/QA execution and final answers are rejected at this
boundary. An empty or immediate `[EOQ]` next question remains a costed unusable
component output, with no synthesized plan or automatic retry.

The native request contains the visible query and exact context paragraph IDs
and text; each source config's `add_context` flag still determines whether its
planner prompt consumes those paragraphs. Reader instruction and history are
not sent to the native planner, but remain unchanged for the downstream reader.
Gold labels and scorers are not exposed. The unexecuted next question is
appended to the original instruction. Continue that patched state in the
original reader/environment and score its real endpoint with the remaining
shared budget; returning a planning patch alone is not an endpoint score.

The actual native cap is preserved (for example, a source config requesting
100 tokens is not expanded to the 512-token safety ceiling). All native
sampling, stop strings and log-probability inputs remain explicit. The runtime
requires source/dependency/import-origin attestation before callbacks and
retains actual token usage, unknown consumption, and local unpriced costs.
Both explicitly chosen controlled and factorial component modes are available;
no historical GPT-3 checkpoint or cross-domain task-performance parity follows
from transferring a source config to a new reader or dataset.
