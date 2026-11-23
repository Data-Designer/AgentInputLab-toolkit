# ACE native Tau feedback

`ACETauFeedback` runs one generated JSON tool plan against Tau's registered
environment, tools and reward evaluator. Tool calls execute in order without
another agent generation or user turn. The final response ends this explicit
noninteractive plan. Use the benchmark's native interactive agent for heldout
inference.

Install `agentinputlab[ace]` and the dependencies of the pinned `tau2-bench`
checkout in a dedicated worker process. The supported revision is
`cf71a8070269883e38a365ffa85f78f46844c1f4`. Keep the checkout clean and use its
own `data` directory. Capture manifests and run outputs belong outside the
checkout. No benchmark data, weights or private acceptance records are shipped.

## Prepare existing captures

Supported domains are `retail` and `banking_knowledge`. Retail fitting and
validation IDs must both belong to the official training split. This Banking
revision has no official training split: explicitly declare disjoint fitting
groups and heldout exclusions; the API records `source_official_split=None`.
ToolTalk tasks remain globally test-only and are not feedback inputs.

```python
from agentinputlab.recipes import ACETauFeedback, build_ace_tau_feedback_input

source_input = build_ace_tau_feedback_input(
    checkout="/opt/upstreams/tau2-bench",
    domain="retail",
    captures=existing_provider_capture_records,
    native_configurations={
        task_id: original_provider_configuration_for_task[task_id]
        for task_id in train_ids + validation_ids
    },
    train_ids=train_ids,
    validation_ids=validation_ids,
    heldout_ids=reserved_heldout_ids,
    partition_label="declared-subdivision-of-official-train",
    target_model=target_model_fingerprint,
    output_path="/data/ace/tau-input.json",
)
feedback = ACETauFeedback(
    checkout="/opt/upstreams/tau2-bench",
    source_input=source_input,
    output_root="/data/ace/new-tau-feedback",
)
train = feedback.fit_examples("train")
validation = feedback.fit_examples("validation")
```

The builder consumes saved captures; it never regenerates initial user messages.
Keep their original simulator-call receipts and costs. Each configuration binds
the native domain, split, seed, budgets, endpoints and dependency hashes. The
user and judge follow the evaluated backbone. The fingerprint's `model_id`
must match that explicit deployment. These are declared model bindings, not
independent remote-weight checks.

Only visible user messages, policy, context and native function schemas enter
the model input. Protected task criteria stay with the native environment and
evaluator. The builder checks the native task identity, initial message, policy
and schemas and pins relative source/data checksums. It does not generate
labels, import private acceptance helpers or execute generated Python.

## Worker and online stream

Use `feedback` with `ACEWorkerRuntime` and its examples with
`ACEOfflineRecipe.fit`. The subprocess worker accepts constructor fields as
flat JSON through:

```bash
python -m workers.ace.driver \
  --checkout /opt/upstreams/ace \
  --llm-config-json "$ACE_LLM_CONFIG" \
  --task-processor-factory agentinputlab.recipes.ace_tau:create_tau_feedback \
  --task-processor-config-json "$TAU_FEEDBACK_CONFIG"
```

For a transductive stream, construct the processor with `mode="online"` and
pass `feedback.stream_examples()` to `ACEOnlineRecipe.run_stream` with explicit
test-feedback and source/asset acknowledgement. ACE's stream label does not
change the original split or make adaptation independent heldout evidence.

The adapter uses native tool argument handling, initialization, retrieval and
`evaluate_simulation(..., strict_replay=True)`. It makes no new user-simulator
calls. Tasks with natural-language reward assertions can invoke the original
backbone-bound judge; retain its actual usage separately from the evaluated
agent's configured budget. Unpriced local `cost_usd=0` is not zero compute.

Malformed plans receive internal feedback zero with `native_metrics=None`.
Valid native reward zero stays zero, and native execution/judge errors
propagate without a repair call. Exact repeated valid plans reuse their
feedback with a reuse receipt. Keep original failures, denominators and costs.
