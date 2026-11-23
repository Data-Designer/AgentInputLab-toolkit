# Prepare ACE and SkillOpt artifacts from a file

For Tau captures produced by `config evaluate-arena --capture-only`, a
preparation file may provide top-level `capture_bundle: {"path":
"./capture/capture-bundle.json", "sha256": "<trusted bundle file SHA-256>"}`
instead of manually constructing `captures`, `native_configurations`, `domain`
and `target_model`. The bundle exports and pins these original inputs.
Still declare `groups.train`, `groups.validation`, `groups.heldout`, the native
environment/feedback, optimization controls, and `partition_label` explicitly.
Only successful train/validation captures are selected for fitting. Missing or
failed captures are rejected without generating replacements. Native split,
group, model and source checks still run; a capture bundle is not permission to
fit on test data. Explicit fields conflicting with the bundle are rejected.

Use `FitConfig` for the existing offline text-space optimization recipes, and
`RunConfig` to evaluate their frozen output. Fitting does not update model
weights and does not implicitly run when an evaluation artifact is absent.

For an explicitly transductive ACE stream, use the separate
[`config online` entrypoint](online-configuration.md); `config fit` remains offline.

```sh
# Check the file without opening a model connection or starting a fit.
agentinputlab config fit fit.json
# Execute once in a new directory; native model calls can incur costs.
agentinputlab config fit fit.json --execute --output-dir ./fit-001
```

The same API is `FitConfig.from_file("fit.json").execute("./fit-001")`.
Paths follow the configuration file, including the explicitly selected worker
Python. The worker must have the same toolkit build and its native dependencies
installed. A virtual-environment Python symlink remains in that environment.

The output contains `started.json`, `terminal.json` and, on success,
`artifact.json`. A failed or completed output directory is never executed again.
Keep it for diagnosis and cost accounting. For evaluation, bind the returned
artifact path and SHA to `recipe.ace.eval_only` or `recipe.skillopt.eval_only`
as shown in [configuration](configuration.md). Do not charge fitting costs as
zero or repeatedly add the same fitting cost to each test case.

## ACE generic text feedback

This entry uses the pinned `workers.ace.driver` and the original
`ACEOfflineRecipe.fit`. Its feedback is generic text exact match, **not** the
native scorer for an interactive arena. Native STATE, ToolSandbox and Tau
feedback adapters can be selected explicitly as described below; generic text
scoring is never substituted for native arena feedback.

```json
{
  "release_id": "recipe.ace.offline",
  "checkout": "./upstream/ace",
  "python_executable": "./worker-env/bin/python",
  "timeout_seconds": 1200,
  "controls": {
    "max_rounds": 1, "max_model_calls": 8, "max_tokens": 4096,
    "max_cost_usd": 0, "seed": 17
  },
  "deployment": {
    "api_provider": "openai", "max_tokens_per_call": 512, "json_mode": true,
    "base_url": "http://127.0.0.1:8000/v1", "api_key_env": "OPENAI_API_KEY",
    "roles": {
      "target": {"model": "your-model", "fingerprint": {"model": "your-model", "revision": "your-revision"}, "input_cost_per_million": 0, "output_cost_per_million": 0},
      "reflector": {"model": "your-model", "fingerprint": {"model": "your-model", "revision": "your-revision"}, "input_cost_per_million": 0, "output_cost_per_million": 0},
      "curator": {"model": "your-model", "fingerprint": {"model": "your-model", "revision": "your-revision"}, "input_cost_per_million": 0, "output_cost_per_million": 0}
    }
  },
  "examples": {
    "train": [{"example_id": "train-1", "split": "train", "input": "Your development question", "expected_output": "Its answer"}],
    "validation": [{"example_id": "dev-1", "split": "validation", "input": "A different validation question", "expected_output": "Its answer"}]
  }
}
```

Replace all example data and deployment identities. Zero pricing represents an
explicitly unpriced local deployment, not free compute; set real provider prices
and a corresponding cost cap when applicable. The native `max_tokens` fitting
accounting remains unchanged; it is not a renamed test-time output-only budget.
All optimization controls are required, so missing fields cannot silently select
a large default fit. Train/validation labels, IDs and full inputs must be
disjoint; test-labeled records are rejected rather than relabeled.

## ACE native arena feedback

Add a `feedback` object to the ACE fit file to use the existing native feedback
factory in the selected worker environment:

```json
{
  "kind": "statebench",
  "checkout": "./upstream/STATE-Bench",
  "source_input": {"path": "./state-fit-input.json", "sha256": "REPLACE_WITH_FILE_SHA256"},
  "environment_base_url": "http://127.0.0.1:8001/v1",
  "environment_names": ["STATE_JUDGE_API_KEY"]
}
```

Supported kinds are `statebench`, `toolsandbox`, and `tau`; ToolSandbox does not
accept an environment endpoint because its plan feedback uses native tools and
state scoring without a judge model. Environment names refer to credentials
already selected in the frozen native configuration; values stay outside JSON.
These calls have their own native accounting and are not represented as free
ACE target-model calls.

Use the existing [STATE](ace-native-statebench.md),
[ToolSandbox](ace-native-toolsandbox.md), or [Tau](ace-native-tau.md) input builder
and its `fit_examples("train")` / `fit_examples("validation")` output for the
file's `examples`. The worker checks those examples against the frozen source
before fitting. This entry does not invent development partitions, regenerate
captures, or fit on ToolTalk. It fixes the feedback mode to offline and writes
fresh native feedback under the fit output directory. Unknown factories,
caller-selected feedback output paths, and online mode are rejected.

The worker uses the current installed toolkit code with the explicitly chosen
Python and its native dependencies. A missing native dependency or mismatched
capture is an error, not a reason to fall back to generic exact-match scoring.
These adapters evaluate noninteractive tool plans during fitting, not a
closed-loop heldout agent. The source builders remain explicit preparation
steps; selecting a feedback kind alone does not create the required inputs.
You can now call those builders through the file-based preparation command below.

### Prepare native feedback without writing a Python adapter

```sh
agentinputlab config prepare-fit preparation.json --output-dir ./prepared-fit
agentinputlab config fit ./prepared-fit/fit.json
# Only this explicit execution step starts optimization/model calls:
agentinputlab config fit ./prepared-fit/fit.json --execute --output-dir ./fit-001
```

The preparation file has these fields:

| Field | Content |
| --- | --- |
| `fit` | The ACE configuration above, excluding `examples` and `feedback`; keep the explicit deployment, controls, timeout, ACE checkout and worker Python. |
| `feedback` | `kind`, native benchmark `checkout`, and optional `environment_names` / environment endpoint as above; omit `source_input`, which this command builds. |
| `captures` | `{"path":"captures.json","sha256":"<trusted file SHA-256>"}` pointing to the existing native inventory JSON array. |
| `groups` | Explicit `train`, `validation`, and `heldout` arrays of native checkpoint IDs. Train and validation must be nonempty; all groups must be disjoint. |
| `native_configuration` | The original STATE/ToolSandbox provider configuration that produced the captures, unchanged. |
| `target_model`, `environment_model` | The original declared deployment fingerprints, including the matching `model_id`. |
| `timeout_seconds` | An explicit preparation-process time limit; it does not alter the fitting or inference budget. |

For ToolSandbox also supply `partition_label`: these are caller-declared
development groups, not an official training split. Tau uses `domain`,
`partition_label`, and a per-case `native_configurations` object instead of
`native_configuration` and `environment_model`; its original same-backbone
rules remain enforced. There is no ToolTalk fitting mode.

For STATE/ToolSandbox, existing captures can be produced by
`config evaluate-arena ... --list-cases --output-dir ./inventory` without agent
execution; use its `capture/inventory.json`. STATE fitting requires the official
`train` split. A capture alone is not permission to treat a held-out task as
development data. Tau preparation consumes existing user captures rather than
silently generating new simulated conversations.

Paths are relative to the preparation file. The command invokes the original
builder and feedback validator inside the selected native Python environment,
generates `source-input.json` and the complete `fit.json`, and records the source
inventory hash and selected groups. It does not call the optimizer, judge, user
simulator or model. Completed output directories are not overwritten: reuse
their `fit.json`. Failures leave a terminal record and do not trigger retries.
The native checkout/dependencies and capture inventory must already be present;
use the arena preparation command for supported environment installation.

## SkillOpt native configuration

This entry calls the original `SkillOptOfflineRecipe.fit` and pinned upstream
trainer with a checksum-bound native YAML file. Supply its real development
splits, initial skill, model/provider configuration, and evaluation settings in
that YAML. It is not automatically constructed from held-out benchmark data.
For development-only fitting, use the upstream settings `evaluation.eval_test:
false` and `evaluation.test_env_num: 0`. The native configuration controls those
splits; the wrapper does not claim a split audit it has not performed.

```json
{
  "release_id": "recipe.skillopt.offline",
  "checkout": "./upstream/SkillOpt",
  "python_executable": "./worker-env/bin/python",
  "native_config": {"path": "./skillopt.yaml", "sha256": "REPLACE_WITH_FILE_SHA256"},
  "controls": {"num_epochs": 1, "batch_size": 1, "learning_rate": 4,
               "seed": 17, "max_steps": 1, "timeout_seconds": 1200},
  "model_fingerprints": {
    "target": {"model": "your-model", "revision": "your-revision"},
    "optimizer": {"model": "your-model", "revision": "your-revision"}
  },
  "environment_names": ["OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL"]
}
```

Export environment values outside the file. Only names are recorded. Native
per-call and other optimization limits belong in the upstream configuration and
provider settings; the wrapper does not silently replace them with test budgets.
SkillOpt usage is not synthesized: the terminal's aggregate optimization usage
is `null` where the native artifact does not expose that aggregate. Retain the
native runtime history and provider metering for complete cost accounting.

### Built-in retrieval environment (no custom Python factory)

For the native IR rollout, add `environment` to the SkillOpt fit file:

```json
{
  "kind": "retrieval",
  "source_input": {"path": "./retrieval-source.json", "sha256": "REPLACE_WITH_FILE_SHA256"},
  "backend": {
    "model_id": "your-model",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "deployment_identity": {"model_id": "your-model", "revision": "your-revision", "server": "vllm"}
  },
  "per_call_output_tokens": 512,
  "seed": 17,
  "stop_policy": "provider",
  "partition_label": "caller_declared_development_groups"
}
```

Use the retrieval source format below: it binds the dataset, retriever, metric
and explicit train/validation/heldout IDs. Optional captured contexts use the
same source checks. Only train and validation enter the SkillOpt environment.
The original SkillOpt query rollout and native scoring remain unchanged;
sharing the input loader does not substitute ACE's optimizer or reflection.

The environment seed must match `controls.seed`, and its deployment identity
must equal `model_fingerprints.target`. Include the backend's `api_key_env` in
`environment_names`; credentials remain environment variables, never JSON.
Point the native YAML optimizer and the target backend at the intended metered
services. This binding does not invent a combined optimization token cap:
native controls and any provider-side total accounting still apply.

Run the existing `agentinputlab config fit fit.json --execute --output-dir NEW`
command. It writes a checksummed built-in environment binding inside `NEW`;
you do not supply a Python factory or `allow_custom_environment` for this path.
Do not combine it with `environment_binding`. The optional
`stop_policy: "gpt_oss_final_channel"` explicitly selects the existing GPT-OSS
vLLM compatibility adapter; it is never inferred from a model name. This does
not expand token limits or treat reasoning as final output.

### Built-in STATE environment

For complete native STATE episodes, set `environment.kind` to `statebench`,
add the pinned STATE `checkout`, and supply `environment_backend` alongside
the target `backend`. Both use the backend JSON format above; the environment
backend supplies the original user simulator and judge. List both credential
environment-variable names in `environment_names`. Use `stop_policy: provider`:
STATE keeps its native Responses API parser rather than a text stop adapter.

`source_input` binds the frozen original training captures, disjoint train and
validation IDs, held-out IDs, native configuration, model identities and source
files. The public `build_ace_statebench_feedback_input` builder can produce this
shared **input description**; selecting SkillOpt does not execute ACE feedback
or replace the native agent with ACE's noninteractive plans.

For file-only preparation, use `config prepare-fit preparation.json --output-dir
./prepared-fit`. Supply `fit` as the SkillOpt template above (without its
`environment`), and put the STATE `environment` object at the top level without
`source_input`. Also supply `captures`, explicit `groups`, `native_configuration`,
`target_model`, `environment_model`, and `timeout_seconds` in the preparation
format above. The command freezes the selected official training captures and
produces `./prepared-fit/fit.json`; it does not execute ACE feedback, fit a skill
or call any model. Then run `config fit ./prepared-fit/fit.json --execute
--output-dir ./skill-fit-001`. Native YAML and credential environment values
remain explicit, and test cases are never silently used as development cases.

The SkillOpt environment injects the current skill into each captured state,
runs the original agent/tool/user/judge episode, and feeds its observable
conversation and native completion score to the original SkillOpt optimizer.
Held-out episodes and extra diagnostic rollouts are not executed during fitting.
The explicit per-call cap must match the frozen native agent budget. Agent,
simulator and judge requests are recorded separately in `native-call-*.json`;
missing provider usage remains unknown rather than becoming zero. Failed calls
leave evidence and are not automatically retried. Put fit outputs outside both
native source checkouts.

### Built-in ToolSandbox environment

Use `environment.kind: toolsandbox` with `checkout`, `backend`,
`environment_backend`, `worker_python`, and an explicit
`episode_timeout_seconds`. Keep `per_call_output_tokens`, `seed`,
`stop_policy: provider`, `partition_label`, and the pinned `source_input`.
The environment backend is the native user simulator; the native similarity
scorer does not use an LLM judge. Simulator usage is recorded separately from
the evaluated agent's budget.

Install the same toolkit wheel and the ToolSandbox native dependencies in
`worker_python`. Each episode runs in a new process to isolate mutable native
state and dependency versions from the SkillOpt trainer. The worker checks the
toolkit implementation digest before executing an episode. It does not import
a caller's helper script or fall back to another interpreter. A timeout stops
the owned child gracefully, retains its log, and does not retry the episode.
This isolated fitting worker requires POSIX. A private lifetime pipe also
terminates the worker if its trainer exits unexpectedly; a lost parent is not
reported as a completed or scored episode. Inspect the saved attempt before
continuing, rather than automatically repeating model calls.

The file-only preparation format is the same as SkillOpt STATE above, using a
ToolSandbox `environment` and adding a top-level `partition_label`. Groups are
explicit caller-declared development groups, not an official training split.
The frozen starting context, clocks, tools, original budget and native scoring
are preserved. Credentials referenced by the original native configuration must
also be available during native preparation; keep their values outside JSON.
Only agent-visible messages and the native scalar score enter reflection.
Zero similarity and malformed-agent terminal results are retained, not retried
or silently converted to successful episodes.

### Built-in Tau environment

Use `environment.kind: tau` for complete native Retail or Banking Knowledge
episodes. Set `checkout`, `worker_python`, `episode_timeout_seconds`, target
`backend`, and `environment_backend` explicitly, along with `seed`,
`per_call_output_tokens`, `stop_policy: provider`, and `partition_label`.
Install the same toolkit wheel and the pinned Tau native dependencies in the
POSIX worker interpreter. The worker uses the same process-lifetime and timeout
handling as ToolSandbox above.

For `config prepare-fit`, provide the SkillOpt `fit` template and `environment`
without `source_input`, plus `captures`, explicit `groups`, `domain` (`retail`
or `banking_knowledge`), per-case `native_configurations`, `target_model`,
`partition_label`, and `timeout_seconds`. Retail development captures must be
from its official training split. Banking Knowledge uses explicitly declared,
disjoint development groups; its loader does not provide an official training
partition. The preparation command verifies the original tasks, tool schemas,
initial messages and source files without generating new user messages.

Run the generated `fit.json` through `config fit --execute`. Each candidate skill
is evaluated with the original interactive agent, tools, user simulator and
native reward. This is not ACE's noninteractive plan feedback. The frozen
same-backbone role identities, seeds and per-role caps are retained. The native
profile requires local endpoints; target requests and simulator/judge requests
may use separate local routes so their costs remain separate. Declare credential
environment-variable names, never secret values, in configuration. Only visible
dialogue and the native scalar reward enter SkillOpt reflection; held-out cases
do not enter fitting. Failures are saved without automatic retries.

### BFCL and custom environment extension

For BFCL Python-simple native function calling, use the same `environment`
object with `"kind": "bfcl"`. Its pinned `source_input` uses the BFCL source
schema described in [ACE BFCL preparation](ace-bfcl.md): source archive,
normalized assets, prepared-file hashes, and explicit disjoint `train_ids`,
`validation_ids`, and `heldout_ids`. No caller Python factory is needed.
The SkillOpt environment sends the native function schemas and current skill
to the configured backend, then scores its tool calls with the original BFCL
AST checker. It does not use ACE's text-answer protocol. Optional saved fit
`rows` are checked against the reconstructed native visible states and tools.
BFCL's source inventory is an official test split: these explicit subdivisions
are compatibility-fitting groups, not an official leaderboard training split.
Held-out cases cannot enter the fitting rollout. Scorer/transport errors retain
failure evidence and propagate without automatic retries or fabricated scores.

An existing `SkillOptEnvironmentBinding` can be serialized under
`environment_binding` using `factory_path`, `factory_sha256`,
`configuration_path`, `configuration_sha256` and optional `factory_name`.
This executes caller-owned Python code, not sandboxed data, and requires the
explicit `allow_custom_environment: true` setting. Both files are verified by
the original loader before execution. Do not enable this for untrusted files.

Neither entry automatically downloads weights, enables SFT/RL, retries a failed
fit, or proves that the resulting artifact improves task scores. Use the frozen
artifact with the original held-out execution and scorer for that evaluation.
# ACE retrieval feedback from files

Use `feedback.kind="retrieval"` to connect the existing ACE optimizer to a
verified IR adapter and a configured retriever. No Python factory is required.
Use the same `fit` template (checkout, Python, deployment and explicit controls)
as above, with this preparation file:

```json
{
  "fit": {"release_id": "recipe.ace.offline", "checkout": "...", "python_executable": "...", "controls": {}, "deployment": {}, "timeout_seconds": 1200},
  "feedback": {"kind": "retrieval", "source_input": {"path": "ir-source.json", "sha256": "<SHA-256 of ir-source.json>"}},
  "timeout_seconds": 180
}
```

The `...` and empty controls/deployment above are placeholders: retain the full
controls and deployment from your fit template, not empty values. For example,
`ir-source.json` has the following shape (replace case IDs with your explicit
development partition; these names do not select examples automatically):

```json
{
  "dataset": {"kind": "production", "benchmark_id": "beir.arguana", "prepared_root": "prepared"},
  "retriever_config": {"path": "retriever.json", "sha256": "<SHA-256 of retriever.json>"},
  "groups": {
    "train": {"split": "test", "ids": ["<train-source-id>"]},
    "validation": {"split": "test", "ids": ["<validation-source-id>"]},
    "heldout": {"split": "test", "ids": ["<heldout-source-id>"]}
  },
  "metric": "ndcg_at_10",
  "top_k": 15,
  "allow_json_fence": false,
  "partition_label": "caller_declared_compatibility_partition_not_official_training"
}
```

`retriever.json` uses the existing run-configuration retriever schema and contains
only `release_id: "baseline.no_op"`, optional `seed`, and `retriever`. This marker
selects no intervention: only its BM25 or prepared-corpus retriever is loaded;
no baseline/model run occurs. Relative paths follow their own containing JSON
file. A prepared-corpus retriever must name the same benchmark as the feedback.

For existing source-verified samples, use dataset kind `trec_dl_sample` or
`beir_sample`, with `benchmark_id`, `root`, `manifest_sha256`, and explicit
`license_acknowledged: true` instead of `prepared_root`. Each loader retains its
native sample scope, exclusions and scorer. No full-corpus download is triggered.
An already-mounted complete archive uses `kind: "beir_archive"`, `benchmark_id`,
`source_archive`, and `license_acknowledged: true`; the existing archive loader
checks its built-in source checksum. This does not download the archive.

When development and heldout queries use separate verified corpora, add
`heldout_dataset` alongside `dataset`, using the same descriptor format above.
`dataset` supplies train/validation; `heldout_dataset` supplies the explicitly
listed, nonempty heldout group. Both must identify the same benchmark and pairing
mode. IDs and visible inputs must be disjoint across all three groups. Each
example is scored by its original dataset adapter; corpora are never merged.
`retriever_config` remains the development retriever, and heldout examples never
enter fit feedback or its empty-ranking preflight. Configure heldout evaluation
and its retriever separately. This applies to both ACE and SkillOpt IR fitting.

If the original fit used previously retrieved context, explicitly add
`captured_states: {"path": "captured.json", "sha256": "<trusted SHA-256>"}` to
the IR source file. The capture file is a JSON object with `dataset_sha256`
(the canonical hash of the native adapter fingerprint, or the role-bound
fingerprint containing `fit_dataset`, `heldout_dataset`, `groups`, and
`native_scorer_uses_original_owned_examples: true`) and `records`, keyed by
every declared group ID. Each record contains `source_state_sha256` (the native
example's `visible_state().state_hash`), `state` (the saved visible state), and
`state_sha256` (that captured `FrozenAgentState`'s hash). Use trusted capture
receipts, not evaluator answers, to supply this optional input.

Captures can replace visible retrieval context and add the nonnegative integer
metadata field `initial_capture_retrieval_calls`. All original metadata, task,
query, instruction, messages and tool names must stay unchanged; the captured
budget may only deduct that recorded number of retrieval calls. The native
scorer still receives its original owned example and qrels, not the capture.
Captures are not silently reconstructed or re-retrieved, and omitting this field
uses the source adapter's original context rather than claiming equivalence to
a context-enriched fit. The separate historical retrieval cost stays in the
capture record and is not counted as newly consumed fit calls.

```sh
agentinputlab config prepare-fit prepare-ir.json --output-dir prepared-ir-fit
agentinputlab config fit prepared-ir-fit/fit.json
agentinputlab config fit prepared-ir-fit/fit.json --execute --output-dir ir-fit-attempt
```

Preparation checks source identity, disjoint groups, and the declared native
metric using empty rankings. It does not call a model, retrieve documents or fit
a playbook. The generated inputs contain visible query/context only, with opaque
example IDs as feedback targets; qrels remain in the native scorer. Partitions
drawn inside an official test split are explicitly compatibility partitions, not
official training or independent held-out benchmark evidence. ToolTalk remains
test-only and is not supported as a retrieval fit source.
