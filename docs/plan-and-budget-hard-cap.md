# Plan-and-Budget: explicit decode-cap extension

`plan.plan_and_budget.hard_cap_extension` is an answer-producing composite
workflow, distinct from the native single-call soft-word solver. It preserves
the existing visible-input decomposition and credit-assessment prompts and
parsers, then executes each sub-question sequentially under an actual decoder
`max_tokens` cap. A separate capped call synthesizes the final answer.

```python
from agentinputlab.api import create_intervention, run_composite_intervention
from agentinputlab import ExecutionBudget, FrozenAgentState, InterventionContext

method = create_intervention(
    "plan.plan_and_budget.hard_cap_extension",
    output_format="Return the concise final answer.",
    decomposition_max_tokens=512,
    assessment_max_tokens=512,
    total_step_tokens=512,
    per_step_max_tokens=512,
    final_max_tokens=512,
)
state = FrozenAgentState(
    query="Your question",
    budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096),
)
# backend is your configured chat/completion backend with JSON support.
result = run_composite_intervention(method, state, InterventionContext(backend=backend, seed=17))
answer = result["final_answer"]
```

Live preparation uses two calls. One to five emitted sub-questions each use one
call, followed by one final call: at most eight calls for a five-step plan.
All preparation, intermediate and final generated tokens count against the same
run budget. Seeds increment by the actual call index. Insufficient budget stops
the workflow; it does not silently drop steps. Empty responses and provider
overruns are failures, without an implicit retry or previous-answer fallback.

For an exact cached plan, pass the existing solver-only `PlanAndBudgetCache` or
ID-keyed `CachedPlanBudgetRecord` mapping through `records`. ID and query must
match. Cache provenance and historical preparation costs remain the caller's
responsibility; a missing record never triggers hidden regeneration.

The subproblem output pool is explicitly `total_step_tokens` (default 512),
separate from the 512-per-step ceiling and the shared 4096 run ceiling. This
preserves credit weighting instead of automatically saturating every step cap.
Allocation reserves the final-call budget first, then gives every step one
token and distributes the remainder by credits and the chosen original decay
schedule. Saturated steps pass excess allocation to unsaturated positive-credit
steps. Zero-credit steps retain their one-token minimum. Stable source-index
ties are explicit. Step caps are fixed before execution; unused step allocation
does not create additional steps or retries. Final synthesis may use unspent
total budget up to `final_max_tokens`. Unallocated or unused budget is not
reported as consumed.

This is a controlled/factorial extension of the planning idea, not the released
author runner or a reproduction of its reported results. The original
`full`, cached-soft and regenerated-soft profiles remain unchanged. The
registry labels this entrypoint `composite`: it returns an observed trajectory
and final answer, not an input patch. Python's composite runner/API handles it;
the loopback SDK transport remains explicitly patch-only.

Keep the benchmark's native scorer independent. The bounded acceptance driver
uses source-backed QA and six LongBench entries, answer-only evaluation, fixed
source indices and pinned reader tokenizers. Overlength visible reference text
is middle-truncated at a declared context-token window before the method runs;
this is recorded as an input projection, not native full-prompt parity. No
dataset labels, fabricated responses or LLM judge are used by the optimizer.
