# Plan-and-Budget: complete answer workflow

`plan.plan_and_budget.full` returns a `CompositeResult` containing the answer,
visible evidence, all preparation/solver costs and a source-bound trace. It is
not the existing decomposition-only or soft-budget input-patch profile.

Install `agentinputlab[plan-and-budget]` and mount the upstream P-and-B checkout
at revision `2ca2d90c6d3d630ec2b11cc22e0f56751f7f2b7e`. The adapter verifies both
source files by SHA-256 before use; it never imports the upstream dotenv, API
client setup or gold-aware evaluator. NumPy 1.26.4 preserves the native
`argsort` allocation behavior, including ties.

```python
from agentinputlab.api import create_intervention
from agentinputlab import FrozenAgentState
from agentinputlab.contracts import InterventionContext
from agentinputlab.integrations.full_architecture.plan_and_budget import (
    PlanAndBudgetNativePromptRuntime,
)
from agentinputlab.runner import CompositeInterventionRunner

method = create_intervention(
    "plan.plan_and_budget.full",
    output_format="A concise final answer in plain text.",
    decomposition_max_tokens=512,
    assessment_max_tokens=512,
    max_solver_tokens=512,
)
context = InterventionContext(
    backend=backend,  # Your explicitly configured, observed LLMBackend.
    external_runtime=PlanAndBudgetNativePromptRuntime(source_root="upstream/P-and-B"),
    seed=17,
)
record = CompositeInterventionRunner().run(
    method, FrozenAgentState(query="Your question", task_id="case-1"), context,
)
print(record.result.final_answer)
```

Live mode first regenerates decomposition and credit fields from the visible
query, without labels (two calls). The native reasoning function then assembles
the weighted prompt using its original level-to-word map: 200, 250, 350, 450 or
600 soft words. One observed solver generation produces the final answer.
There are **three calls**, not one call per step. Empty final output or failed
preparation remains a costed failure; the adapter does not automatically retry.

For an existing cache, pass `records=PlanAndBudgetCache.from_native_json(...)`.
The task ID and query must match exactly; cache misses never fall back to live
generation. Cache mode uses one solver call. The cache loader discards reference,
answer, solution and gold fields. Solver references come only from the caller's
visible `state.context`, not from cached reference/solution fields. The source
MATH/NI/TravelPlanner caches are not caches for other benchmarks.

This is a **controlled native-source solver binding**, not a claim of historical
paper-result or native-harness parity. The original prompt and NumPy allocator
execute unchanged up to the native provider boundary. The caller's chat backend,
temperature, seed and decoder cap replace the source provider/model sampling
defaults (including its default `top_p`); these choices and backend defaults
must be recorded in the experiment configuration. The source retry decorator is
explicitly removed. No synthetic provider response is used to build the prompt.
Official benchmark scoring remains a separate, caller-owned step.

The per-step hints are **soft word budgets**, not decoder token caps. The single
solver call has the explicit decoder output cap and shares the total run budget
with preparation. `hard_cap_extension` remains a separate release, not an alias
of this workflow. Tool/history-driven agent loops are rejected rather than
silently converted to static QA. The Python composite runner is the execution
entrypoint; the patch-only HTTP SDK and static intervention builder intentionally
do not expose this composite as a patch.
