# First-run diagnostics

All installed guides can be read without GitHub, a browser or network access:

```sh
agentinputlab docs --list
agentinputlab docs getting-started
agentinputlab docs configuration
agentinputlab docs acceptance-coverage
```

The same topics are available through
`agentinputlab.documentation.list_guides()` and `read_guide(topic)`.
Use `--json` for machine-readable CLI output. These commands do not execute
models, download datasets or write files.

## Locate the layer that failed

| Symptom | Check | Meaning / next step |
| --- | --- | --- |
| Cannot import the package or command not found | `python -m pip show agentinputlab`; `python -m agentinputlab --version` | Use the same Python environment for installation and execution. |
| Registry doctor passes but a profile cannot run | `agentinputlab show EXACT_RELEASE --parameters` | Doctor checks the catalog, not native workers, models, fitted assets or task eligibility. Inspect the profile's prerequisites. |
| Dataset listed but no automatic loader | `agentinputlab datasets show BENCHMARK_ID` | Follow its native setup route. Discovery does not imply every runtime can be downloaded and launched automatically. |
| Config rejected before execution | `agentinputlab config validate run.json` | Correct unknown keys, native constructor arguments or conflicting generation settings. Validation makes no model calls. |
| HTTP auth / model / connection error | Check your endpoint's advertised model ID, key environment variable and model availability | Keep keys out of JSON and logs. Do not assume a failed connection consumed zero tokens or retry an uncertain request automatically. |
| Unsupported model architecture | Inspect the native worker's pinned Transformers/runtime version | A generic HTTP backend and an author's local worker support different architectures. Use a compatible, explicitly named profile. |
| Missing cache or fitted artifact | Check exact task, split, model and artifact identities | Do not substitute another task's cache or another backbone's weights. |
| Output is a patch with `scored: false` | Inspect the profile kind | A prompt intervention is not a downstream model answer. Execute the downstream agent and benchmark-native scorer separately. |
| Valid output but score is zero | Read native scorer diagnostics and budget usage | This can be a genuine unsuccessful answer. Keep it in the denominator; don't reinterpret it as a transport failure. |
| Model output empty, malformed or truncated | Inspect actual usage and terminal error | Preserve the failed attempt and costs. Changing caps/parser/profile needs a separately identified run or explicitly labeled saved-response replay. |

## What a successful user check should record

First run one fixed real case with your actual endpoint, dataset split and
native scorer. Record the release ID, model revision, dependency versions,
budget, actual usage, terminal status and score. Distinguish these questions:

1. Could the library instantiate and call the configured dependency?
2. Did the model produce output accepted by the native protocol?
3. Did the native scorer judge the task successful?

Only the third establishes task success for that case. A CLI exit code, an
offline smoke test or a replay of a saved response does not establish universal
model × dataset × algorithm compatibility.

## Reporting a reproducible issue

Include your package/Python versions, exact release ID, dataset ID and split,
model family and revision, a minimal redacted config, terminal error type and
the step that failed. State whether any model request was sent and its known
usage. Do not attach API keys, authorization headers, private prompts or full
research archives. This makes it possible to repair the adapter without
accidentally repeating a consumed scientific attempt.
