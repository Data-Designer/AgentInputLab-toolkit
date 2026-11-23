# Configure and run workflows in Studio

The installed Studio workflow builder uses the same preparation, execution,
budget accounting and native scoring code as `agentinputlab workflow`.
It is included in the published wheel and runs locally in your environment.

From an environment containing this updated source installation:

```sh
agentinputlab-studio --workflow-root ./studio-workflows
```

Open the printed `/workflows` URL. Without `--workflow-root`, configuration
generation and export remain available, but inspection and execution are disabled.
Keep this trusted-operator service bound to localhost. For a server installation,
use an SSH tunnel rather than exposing the service publicly.

## A normal run

1. Choose an editable starting preset, then the exact method profile and benchmark.
   The catalogue includes all registered entries; selection is not a compatibility
   guarantee. Preflight explains the requirements of the selected combination.
2. Enter your served model ID, API base URL and deployment revision. Credentials
   remain in the server environment; enter an environment-variable **name**, not
   an API key. The form does not automatically deploy model weights.
3. Set the per-case budget, seed and concurrency (1–4 cases). Expand advanced
   settings for the method's native parameters, retriever, artifacts, runtime and
   preparation. Temperature, per-call token caps and `max_length` use the names
   shown in the selected method's parameter reference; they are not interchangeable
   across algorithms. The complete JSON also exposes vector runtimes, local model
   configuration, target/scorer options and case selection.
4. Click **Generate configuration**. Review or edit the complete `workflow.json`.
   Subsequent form changes must be generated again; they do not silently replace
   an existing JSON draft.
5. Click **Check prerequisites**. The optional endpoint probe checks availability
   without generating a response. Missing inputs and unsupported routes are listed
   before execution. Dependency downloads require the explicit download checkbox.
6. Click **Prepare, execute and score**. Model-backed runs send requests to your
   configured service. Saved runs show terminal outcomes, native metrics, usage,
   failures and the output directory. Prompt-only runs are explicitly unscored;
   generating a prompt is not benchmark success.

## Reusable configuration files

**Download run.json** exports the algorithm, backend, budget and execution
settings for `RunConfig` / `agentinputlab run`. **Download workflow.json** also
contains the benchmark or query target, preparation and concurrency settings.
Both can be imported. When importing only a `run.json`, the currently selected
benchmark/query supplies its target; the imported method, parameters, seed and
budget are preserved. Relative paths resolve against the server's workflow root.

Presets are starting configurations, not thousands of pre-generated Cartesian
product files. A profile needing a checkpoint, fitted artifact, native runtime,
retriever or dataset-specific input still needs that resource. The builder does
not invent it or replace the selected method with another algorithm.

## Saved runs and safe continuation

Runs persist under `<workflow-root>/<run-id>/`. Reopening Studio with the same
root restores the list. **Resume saved run** uses the immutable saved configuration
and the native journal; completed attempts, including failures, are not repeated.
Editing the draft and launching creates a separate run instead.

Only one workflow is active per Studio manager; its cases may run concurrently.
After an unclean process termination, an existing `.studio-running` guard requires
operator reconciliation of the original process and journal before continuation.
Do not remove that guard while a worker might still be running. Uncertain attempts
are not automatically retried. Large outputs remain in the native journal; the
webpage shows a bounded preview, not a replacement evidence file.
