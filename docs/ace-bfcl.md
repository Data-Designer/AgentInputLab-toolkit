# ACE with BFCL Python-simple feedback

`ACEBFCLFeedback` connects ACE's native offline loop to the source-verified
BFCL Python AST checker. The native ACE text generator puts a JSON array of
function calls in its `final_answer` string. The adapter decodes that array and
uses `ast_accuracy` for training feedback and validation selection. This is an
explicit text-to-function-call task adapter, not native BFCL handler prompt
parity or a leaderboard training protocol.

```python
from agentinputlab.benchmarks import (
    SourceVerifiedBFCLV3ASTBridge,
    load_bfcl_v3_simple_adapter,
)
from agentinputlab.recipes import ACEBFCLFeedback

bridge = SourceVerifiedBFCLV3ASTBridge(
    source_archive="/data/bfcl/official-v1_3.tar.gz",
    model_id="your-deployed-model-id",
)
adapter = load_bfcl_v3_simple_adapter(
    "/data/normalized", evaluation_bridge=bridge,
)
feedback = ACEBFCLFeedback(
    adapter=adapter,
    train=caller_training_examples,
    validation=caller_validation_examples,
    partition_label="my-disjoint-compatibility-groups",
)
train = feedback.fit_examples("train")
validation = feedback.fit_examples("validation")
```

Acquire and verify the benchmark assets before fitting. Choose disjoint groups
before inspecting model outputs and keep held-out examples out of both groups.
The original BFCL release exposes a test split: caller-defined subdivisions
inside that split are compatibility samples, not an official training split.
Do not reuse globally test-only tasks for fitting.

Pass `feedback` as `data_processor` to `ACEWorkerRuntime`, or construct it in an
operator-owned startup factory using `--task-processor-factory` and
`--task-processor-config-json` (see [the ACE worker guide](../workers/ace/README.md)).
Use the resulting `train` and `validation` examples with `ACEOfflineRecipe.fit`.
Keep the recipe's model fingerprints identical to the worker deployment.

For the public file interface, no startup factory is needed. Create a source
JSON with `source_archive`, `normalized_root`, `prepared_file_hashes`, and
explicit `train_ids`, `validation_ids`, `heldout_ids`. Asset paths are relative
to that source file; retain the hashes returned by verified dataset preparation.
Reference its file SHA in a preparation configuration:

```json
{
  "fit": {"release_id": "recipe.ace.offline", "checkout": "./upstream/ace",
          "python_executable": "./worker-env/bin/python",
          "timeout_seconds": 1200, "controls": {}, "deployment": {}},
  "feedback": {
    "kind": "bfcl", "source_input": {"path": "./bfcl-source.json", "sha256": "REPLACE_WITH_FILE_SHA256"},
    "model_id": "your-deployed-model-id",
    "partition_label": "my-disjoint-compatibility-groups"
  },
  "timeout_seconds": 120
}
```

Fill `controls` and `deployment` using the complete ACE example in
[fit configuration](fit-configuration.md); empty objects above are placeholders,
not executable defaults. The scorer's `model_id` must equal the target model.

```sh
agentinputlab config prepare-fit bfcl-preparation.json --output-dir ./bfcl-prepared
agentinputlab config fit ./bfcl-prepared/fit.json
agentinputlab config fit ./bfcl-prepared/fit.json --execute --output-dir ./bfcl-fit
```

Preparation verifies the pinned AST assets and creates the native fit examples
without scoring, generating responses or fitting. Only the final explicit
`--execute` starts optimization. The same `feedback` object can be used directly
in a fit file containing the matching native examples. The file interface is
offline only; it never infers groups or silently starts online adaptation.

Inputs retain the visible question, task instruction, conversation, context,
tool names and function schema. The visible `native_function_tools`
catalog uses the bound BFCL handler's name conversion (for example,
`math.factorial` becomes `math_factorial`). Model outputs are not renamed or
repaired after generation. Evaluator targets are opaque case IDs; hidden
reference arguments stay in the scorer. Malformed call JSON receives zero
training feedback without repair, execution or another model call. Dataset
drift and scorer errors remain errors. No generated Python is executed.

The native Generator/Reflector/Curator loop, budget tracker and playbook
selection remain unchanged. Record fit costs separately from held-out
inference. Evaluate frozen artifacts with the declared downstream tool-calling
interface and the original AST scorer; a successful fit alone does not prove
held-out compatibility or improved task accuracy.

## Explicit online feedback

`ACEOnlineBFCLFeedback(adapter=adapter, stream=ordered_examples)` exposes
`stream_examples()` and `validate_online_examples()`. Pass the examples and
`data_processor=feedback` to `ACEOnlineRecipe.run_stream`, acknowledging its
transductive test feedback explicitly. The original native online loop keeps
its window order; primary outcomes are measured before that window's update.
Initial evaluation, feedback and adaptation share each case's configured budget.
These results are online adaptation evidence, not independent held-out results.
Do not use this mode on a globally test-only arena such as this project's ToolTalk.
