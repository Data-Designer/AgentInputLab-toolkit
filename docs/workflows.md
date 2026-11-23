# Automatic workflows (0.1.0a12)

This entrypoint is available from **0.1.0a12**; it is not part of 0.1.0a11.
It connects configuration, bounded data acquisition, native environments,
method preparation, execution and native scoring. `workflow launch` runs those
steps together. Each exact profile retains its native inputs and budgets.

## First complete path, without a model

```sh
agentinputlab workflow init --query "Find a suitable flight" > workflow.json
agentinputlab workflow inspect workflow.json
agentinputlab workflow prepare workflow.json --output-dir prepared-workflow
agentinputlab workflow run prepared-workflow/workflow.json --execute --output-dir workflow-run
agentinputlab workflow status workflow-run
```

The default Plan-and-Solve profile produces an input patch, not a model answer or
benchmark score. These commands make no model calls. New directories are used
for preparation and execution; existing evidence is not overwritten.

`workflow run` without `--execute` only inspects prerequisites, even if an output
directory is supplied. It neither downloads assets nor contacts a model.

## Reuse an already configured real method

Use the existing [run configuration](configuration.md); do not replace native
prompts, budgets, artifacts or parameters to fit a template:

```sh
agentinputlab workflow init --run-config run.json --benchmark bfcl.v3.simple_python --limit 1 --cache-dir ./toolkit-cache > workflow.json
agentinputlab workflow inspect workflow.json
agentinputlab workflow launch workflow.json --execute --allow-network --output-dir workflow-run
```

Review the selected source terms and explicitly set
`target.accept_license_review` if required. Network acquisition is off by default.
Preparation reuses verified data and completed preparation receipts. It resolves
a limit into fixed case IDs and binds their optimizer-visible state hashes.
The default download and normalized-data limits are 512 MiB. Large or gated
assets, credentials, demonstrations and learned artifacts are named in the
prerequisite report when they require user input.

The run phase uses already prepared data in offline mode. Static cases go
through the existing `RunConfig.evaluate()` and original scorer. An official
score of zero is retained as a scored outcome. An input patch is not relabelled
as a scored benchmark result.

## Generate a model and dataset configuration

```sh
agentinputlab workflow init --release-id plan.plan_and_solve.ps_prompt \
  --benchmark bfcl.v3.simple_python --case-id simple_2 \
  --model-id YOUR_SERVED_MODEL --base-url http://127.0.0.1:8000/v1 \
  --deployment-revision YOUR_MODEL_REVISION --cache-dir ./toolkit-cache > workflow.json
agentinputlab workflow launch workflow.json --execute --allow-network --output-dir workflow-run
```

Use `--parameters-config`, `--retriever-config`, `--artifacts-config`,
`--runtime-config` and `--preparation-config` for native JSON settings. A complete
`--run-config` preserves its explicit values. The initializer derives the task
output format for Plan-and-Budget; it can also derive file-worker bindings and
a visible-context BM25 configuration for supported context-based tasks.
Corpus retrieval is explicit for IR datasets. Defaults and derived settings
are visible in the generated JSON before execution.
`--budget-config` supplies explicit cumulative limits. Use `--predictor-config`
to preserve an existing downstream protocol; a paid IR query reader is not
silently added to an unconfigured ranking pipeline. Reuse data directly with
`--prepared-root`, `--bfcl-source-archive`, `--longbench-config`,
`--beir-sample-config` or `--trec-sample-config`. Nested source paths are resolved
relative to their source configuration file. `--accept-license-review` records
your explicit review for static datasets that require it.

`workflow recipes` lists every exact profile's preparation route, required
constructor fields and runtime kinds. Context/skill preparation (including ACE
and SkillOpt) is distinguished from SFT/RL weight training. Evaluation can reuse
their existing artifact; creating a new learned/context artifact uses its native
fit or online configuration and separately recorded budget.
Profiles requiring a native Python runtime/WorkerClient are identified as
`native_python_api`; a file workflow does not silently replace them with a
different method variant. This includes native R2AG same-candidate inference,
IRCoT controlled, patched external HyDE, and InfoCQR's native teacher/student
bindings. Their existing Python interfaces remain available.

## Native arenas

Selecting `--benchmark tooltalk.full`, `state_bench.common_support`,
`toolsandbox.common_support`, or a Tau domain generates native arena settings.
Supply `--python /path/to/a/compatible/python` to prepare an isolated environment.
STATE, ToolSandbox and Tau also require `--environment-backend-config`, naming
the simulator/judge deployment explicitly. ToolTalk requires no environment LLM.
Preparation lists the native inventory without agent calls and freezes the
requested case IDs or limit automatically. An existing `--arena-config` is also
accepted. Native Python-version and platform requirements are checked by the
existing environment recipes.

## Method preparation and pinned assets

`preparation.sources=true` acquires the selected method's registry-pinned source.
Existing source trees are checked and left unchanged. `preparation.python` and
an optional `preparation.environment` enable an owned process-worker environment;
the active Python environment is never modified.

Worker environments are shared through the toolkit cache and keyed by the
interpreter, selected recipe and toolkit identity. Changing an output directory
reuses a previously bound worker. If the calling Python has no pip, preparation
creates an owned installer environment. NumPy for Plan-and-Budget, cold Parquet
reading, native IR measures, and LongBench scoring extras can also be installed in an isolated
execution environment. `preparation.python` can point to an existing environment
with those dependencies. Model weights and GPU services remain explicit resources.
Delegated processes and their native descendants use the same toolkit code without
adding the caller's binary site-packages to the selected interpreter. An explicitly
configured native `worker_python` must independently contain the same implementation;
an older installation is reported before an episode starts.

### Reuse fitted artifacts without repeating preparation calls

Use `--artifact-catalog /path/to/catalog.json` with `workflow init` to select an
existing ACE/SkillOpt artifact or another supported artifact binding automatically.
The catalog is caller-owned data, not a download of our experimental results.
For example:

```json
{
  "schema_version": 1,
  "bindings": [{
    "identity": {
      "release_id": "recipe.skillopt.eval_only",
      "benchmark_id": "tooltalk.full",
      "model_id": "my-served-model",
      "deployment_identity": {"revision": "my-exact-model-revision"}
    },
    "artifacts": {
      "artifact": {"path": "artifacts/skill.json", "sha256": "<64-hex-sha256>"}
    }
  }]
}
```

The equivalent workflow setting is `preparation.artifact_catalog`. Artifact paths
are relative to the catalog. Selection requires exactly one matching method,
benchmark, model ID and deployment identity; it never selects a different model's
artifact or guesses between duplicates. Explicit `run.artifacts` takes precedence.
The catalog and selected artifact are bound into preparation/resume identities.
JSON artifact contents and checksums are checked before generation. Missing
private artifacts require user input; a missing artifact does not implicitly
start a fit, fit on the benchmark test split, or reuse another backbone's asset.

Method preparation calls the existing `worker`, `adaptive`, `qrag`,
`longllmlingua`, `recomp`, `hyde`, `query2doc` or `ircot` helper. A helper can be
selected explicitly in `preparation.method.kind`, with its native keyword
arguments in `preparation.method.options`. Deterministic worker/binding helpers
are selected automatically when the matching unbound configuration is supplied.
Arguments are checked before downloading. No method is fitted implicitly.

InfoCQR and ReFormeR worker `controls.max_tokens` count input **plus** output.
Their generated default guard is derived from the declared per-call input/output
ceilings. `run.budget.max_output_tokens` still controls the cumulative output,
and the model's per-call output cap remains unchanged. Explicit worker controls
are preserved; a conflicting guard is reported before generation.

Missing public prompt/checkpoint files can be declared in `preparation.assets`.
Each entry contains an `ArtifactManifest`, `destination`, and optional
`allowed_hosts`. SHA-256 and byte size are mandatory. Missing files share
`preparation.max_download_bytes` (default 512 MiB); existing files are verified
and reused. Differing existing files are preserved and reported. Downloads use
the package's streamed, checksum-verified HTTPS transport. Private/gated files
must be supplied through their documented access route.

## One prerequisite report

`inspect` (also named `plan`) checks configuration, method runtime requirements,
retrieval/artifact bindings, selected data, scorer settings, worker dependency
availability and declared model capabilities. It reports preparation operations
and an action for each blocked check. Inspection only reads existing inputs;
it never normalizes raw data, installs packages or generates text.
`can_attempt` reports local preflight; `ready_for_execution` additionally requires
the endpoint check for model-using workflows. Model-output quality remains an
execution result. Native runtime verification details retain their actual scope.
Coarse catalog family tags are informational: they do not veto an existing
configured transfer. Its input, runtime and scorer contract is still validated
by the original native executor, not inferred from the tags.

`--probe-endpoint` makes one authenticated `GET /models` request to
the configured API root. Execution makes this check before dispatching a new
case. Completed runs can be resumed/read after their service has stopped.
The check makes no generation request, does not follow
redirects and does not claim that model-list access verifies tool calling or
JSON-schema generation. Keep API keys in environment variables. If you use a
local endpoint alongside an external HTTP proxy, configure `NO_PROXY` for the
intended loopback destination; the workflow does not change global proxy settings.
Providers without a model-list route can still be contacted; the report records
that the exact model ID was not checked by listing.

## Resume without repeating consumed attempts

```sh
agentinputlab workflow run prepared-workflow/workflow.json --execute --output-dir workflow-run --resume
```

For an all-in-one launch, use the original workflow and
`workflow launch workflow.json --execute --output-dir workflow-run --resume`.
Completed preparation stages are reused. Failure and interruption evidence is
kept with that launch.

The wrapper journals a started record before execution and fsyncs each terminal
record. Resume validates the workflow/implementation identities and the journal:

- Completed scored cases and real failures are not rerun.
- Started cases without terminal evidence stay `needs_reconciliation`; they
  are not treated as free, unstarted work.
- Other never-started cases can continue after an ordinary interrupted call.
- A truncated journal or changed binding blocks reuse instead of causing a retry.
- A hard-killed process can leave a `.running` guard. Confirm that no process
  owns that run and reconcile its journal before manually removing the guard;
  the library never guesses that a process is dead or signals it.

`all_attempts_terminal` and `all_succeeded` are separate fields. `all_succeeded`
describes execution terminals, not task accuracy: a natively scored zero is a
completed evaluation and its metric remains zero. Clarification
requests remain `awaiting_user`; no fake reply is generated. Native failures
retain their recorded costs. Missing usage is unknown, not zero.

## Python

```python
from agentinputlab.workflows import (
    WorkflowConfig, inspect_workflow, launch_workflow,
)

workflow = WorkflowConfig.from_file("workflow.json")
print(inspect_workflow(workflow))  # no generation or downloads
print(launch_workflow(workflow, execute=True, allow_network=True, output_dir="workflow-run"))
```
