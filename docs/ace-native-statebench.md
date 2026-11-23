# ACE native STATE-Bench feedback

`ACEStateBenchFeedback` connects ACE's unchanged Generator / Reflector / Curator
loop to STATE-Bench's registered domain tools, state-requirements scorer and
original task-requirements judge. ACE generates one JSON tool plan; its actions
execute in order against a fresh native environment without another generation
or access to tool results while planning. This is an explicit noninteractive
fitting adapter. Use the original benchmark agent for closed-loop heldout runs.

Install `agentinputlab[ace]` in a dedicated worker process and install the native
dependencies from the STATE-Bench checkout. The supported revision is
`4efcbf2d4fe60df04878859b692d9391f3d5b33a`. The checkout must be clean; put input
manifests and run outputs **outside** it. No upstream source, dataset, model or
acceptance record is bundled in the package.

## Freeze existing training captures

Use the existing native provider to capture the official training split, with
your explicit deployment, source/dependency hashes and budgets. STATE capture
loads task opening messages; it does not call a model. Captured data can remain
on the execution machine. Select train/validation IDs before inspecting model
outcomes. Both groups must come from the official training split, with disjoint
IDs and inputs; heldout IDs are exclusions, not examples supplied to ACE.

```python
from pathlib import Path
from workers.official_arenas.providers.statebench import StateBenchProvider
from agentinputlab.recipes import (
    ACEStateBenchFeedback, build_ace_statebench_feedback_input,
)

# native_configuration is the provider's explicit configuration object.
# Use split="train" and provide declared target/judge model fingerprints.
captures = StateBenchProvider().capture(
    native_configuration, output_root=Path("/data/captures"),
)
source_input = build_ace_statebench_feedback_input(
    checkout="/opt/upstreams/STATE-Bench",
    captures=captures,
    native_configuration=native_configuration,
    train_ids=selected_train_checkpoint_ids,
    validation_ids=selected_validation_checkpoint_ids,
    heldout_ids=reserved_heldout_checkpoint_ids,
    target_model=target_model_fingerprint,
    environment_model=judge_model_fingerprint,
    output_path="/data/ace/state-fit-input.json",
)
feedback = ACEStateBenchFeedback(
    checkout="/opt/upstreams/STATE-Bench",
    source_input=source_input,
    output_root="/data/ace/new-feedback-run",
)
train = feedback.fit_examples("train")
validation = feedback.fit_examples("validation")
```

Checkpoint IDs use `domain:task_id`. Fingerprint mappings include `model_id`
matching `agent_model` or `evaluator_model` in the native configuration. Their
remaining fields can bind your provider, model revision and verified assets.
These mappings record the operator's declared deployment; the builder does not
independently hash a remote model's weights. Credentials belong in the named
environment variables, not fingerprints or committed configuration files.

The builder rechecks actual protocol train membership, opening messages, task
identities, user IDs, visible policy and function names. It stores relative
native source/task/prompt checksums, so the verified checkout can be mounted at
another location. A new machine still needs access to the pinned input manifest
and its trusted checksum. The manifest is data-only: it never imports an
acceptance helper or executes a supplied source-file path.

## Offline worker or online recipe

Pass `feedback` as `data_processor` to `ACEWorkerRuntime`, and use `train` /
`validation` with `ACEOfflineRecipe.fit`. For the subprocess worker, the shipped
startup factory accepts the same constructor fields as flat JSON:

```bash
python -m workers.ace.driver \
  --checkout /opt/upstreams/ace \
  --llm-config-json "$ACE_LLM_CONFIG" \
  --task-processor-factory agentinputlab.recipes.ace_statebench:create_statebench_feedback \
  --task-processor-config-json "$STATE_FEEDBACK_CONFIG"
```

Use a fresh feedback output directory for each process. Read the
[ACE worker deployment guide](../workers/ace/README.md) for explicit models,
fingerprints and recipe controls. The native judge uses its original deployment
and controls, separate from the evaluated agent's budget; retain its actual
request/usage receipts separately. No judge cost is represented as zero merely
because the plan itself makes no further evaluated-agent calls.

For an explicitly transductive stream over the same training captures, construct
the processor with `mode="online"`:

```python
from agentinputlab.recipes import ACEOnlineRecipe

feedback = ACEStateBenchFeedback(
    checkout="/opt/upstreams/STATE-Bench",
    source_input=source_input,
    output_root="/data/ace/new-online-feedback",
    mode="online",
)
result = ACEOnlineRecipe().run_stream(
    feedback.stream_examples(),
    checkout="/opt/upstreams/ace",
    llm_config=explicit_three_role_deployment,
    output_root="/data/ace/new-online-run",
    allow_test_feedback=True,
    source_and_asset_acknowledged=True,
    data_processor=feedback,
)
```

The native ACE API labels stream examples `test`; their recorded official split
remains `train`. Initial evaluation, pre-window evaluation and adaptation share
each case's configured budget. This stream is not independent heldout evidence.
ToolTalk or another globally test-only set must never be used for fitting or
online feedback.

Visible query, policy, context, messages and native tool schemas go to the model.
Protected task requirements stay in the environment/scorer; feedback targets
are opaque case IDs. Invalid plan JSON gets internal training feedback zero
with `native_metrics=None`, not a fabricated official score. Native tool or
judge errors propagate without a repair/retry call. Exact repeated valid plans
reuse the recorded feedback and mark that reuse rather than redraw the judge.
Keep real native zeros, failures, original denominators and all costs.
