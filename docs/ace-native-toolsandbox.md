# ACE native ToolSandbox feedback

`ACEToolSandboxFeedback` connects ACE's Generator / Reflector / Curator to
ToolSandbox's registered Agent-role tools and native state-based scorer. It
accepts a single JSON plan: the actions execute in order, without returning tool
results to the planner for another generation. Use ToolSandbox's original agent
for closed-loop heldout evaluation.

Install `agentinputlab[ace]` and the dependencies of the native ToolSandbox
checkout in a dedicated worker process. The supported native revision is
`165848b9a78cead7ca7fe7c89c688b58e6501219`. Keep that checkout clean and put
capture manifests and outputs outside it. Upstream code, data and model weights
are external dependencies, not package contents.

## Prepare disjoint fitting groups

ToolSandbox has no official training split. Declare disjoint fitting and
validation task families and a separate heldout exclusion list before examining
model outcomes. The builder preserves this caller-defined provenance; it does
not turn the selected groups into an official benchmark split. ToolTalk's
globally test-only tasks must not be used for fitting or online feedback.

```python
from agentinputlab.recipes import (
    ACEToolSandboxFeedback,
    build_ace_toolsandbox_feedback_input,
)

# captures are existing ToolSandboxProvider.capture(...) records.
source_input = build_ace_toolsandbox_feedback_input(
    checkout="/opt/upstreams/ToolSandbox",
    captures=captures,
    native_configuration=native_configuration,
    train_ids=selected_train_checkpoint_ids,
    validation_ids=selected_validation_checkpoint_ids,
    heldout_ids=reserved_heldout_checkpoint_ids,
    partition_label="caller-declared-disjoint-task-families",
    target_model=target_model_fingerprint,
    environment_model=environment_model_fingerprint,
    output_path="/data/ace/toolsandbox-input.json",
)
feedback = ACEToolSandboxFeedback(
    checkout="/opt/upstreams/ToolSandbox",
    source_input=source_input,
    output_root="/data/ace/new-toolsandbox-feedback",
)
train = feedback.fit_examples("train")
validation = feedback.fit_examples("validation")
```

The native configuration includes explicit scenario names, target/user model
IDs, endpoint URLs, named credential environment variables, dependency hashes
and budgets. Set those environment variables before preparing input; the builder
does not supply credentials or call a model. Fingerprints must include matching
`model_id` values and can record deployment revisions and asset checksums.
They are caller declarations, not an independent remote-weight verification.

The model receives the original visible query, policy, messages and context.
Only the tool catalog is projected through ToolSandbox's native Agent role;
simulated-user tools such as `end_conversation` are not advertised. Hidden task
requirements remain scorer-side. Relative source checksums, frozen starting
contexts and a trusted manifest checksum bind the prepared data to its native
runtime. No acceptance helper or generated Python is executed by this adapter.

## Offline worker and online stream

Pass `feedback` to `ACEWorkerRuntime` as `data_processor`, with its train and
validation examples passed to `ACEOfflineRecipe.fit`. The subprocess worker
accepts the same constructor fields as flat JSON:

```bash
python -m workers.ace.driver \
  --checkout /opt/upstreams/ace \
  --llm-config-json "$ACE_LLM_CONFIG" \
  --task-processor-factory agentinputlab.recipes.ace_toolsandbox:create_toolsandbox_feedback \
  --task-processor-config-json "$TOOLSANDBOX_FEEDBACK_CONFIG"
```

Use a new output directory for each process. See the
[ACE deployment guide](../workers/ace/README.md) for role models and controls.
For a transductive stream, construct the processor with `mode="online"` and
pass `feedback.stream_examples()` to `ACEOnlineRecipe.run_stream`, with explicit
`allow_test_feedback=True` and source/asset acknowledgement. ACE's stream label
does not change the caller-defined origin or make it independent heldout data.

The plan feedback invokes native tools and `scenario.play_and_evaluate`, with
native similarity, milestone, minefield and turn-count metrics. It makes no
user-simulator or judge model calls. Malformed plans receive internal feedback
zero and `native_metrics=None`, not a fabricated official score. Native tool
errors propagate without repair calls. Exact repeated valid plans reuse their
feedback with an explicit reuse receipt; real native zeros and failures remain
in the recorded outcomes and costs.
