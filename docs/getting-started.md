# First run and troubleshooting

Use a virtual environment and the same Python interpreter for installation and
execution. The dependency-free core works on Python 3.10+; native model extras
can require a particular Python version, platform, upstream checkout or weights.

```sh
python -m venv .venv
# macOS/Linux; Windows PowerShell: .venv\Scripts\Activate.ps1
source .venv/bin/activate
python -m pip install agentinputlab==0.1.0a15
python -m agentinputlab --version
python -m agentinputlab doctor --offline --json
python -m agentinputlab smoke --json
```

`smoke` uses labeled synthetic fixtures, makes no model calls and downloads no
datasets or weights. `doctor` checks registry integrity, not GPU readiness or
universal method/benchmark compatibility. Neither command is a benchmark score.

Version 0.1.0a2 supports both `python -m agentinputlab` and the
`agentinputlab` console command, including `--version`. The module form uses
the selected Python interpreter even when its scripts directory is not on PATH.

## Choose a profile before installing heavy dependencies

```sh
agentinputlab list --releases --search plan_and_solve --json
agentinputlab show plan.plan_and_solve.ps_prompt
agentinputlab show plan.plan_and_solve.ps_prompt --parameters
```

Use the exact release ID and its prerequisite documentation. An available
entrypoint is not a configured runtime. Install only the extras you need, for
example `python -m pip install 'agentinputlab[benchmarks,evaluation]==0.1.0a15'`.
Do not install all extras into one environment: some native methods deliberately
pin incompatible upstream versions. Keep those runtimes in separate environments.

The search and parameter-inspection commands require 0.1.0a3+. They expose the
actual constructor signature without creating the algorithm or making a model
call. In Python use `describe_intervention(release_id)` from `agentinputlab.api`.
Required constructor arguments are listed separately from runtime requirements
in the profile's help and prerequisites; an optional `client=None` may still
require an external worker at execution time.

For editable JSON, use `agentinputlab config init > run.json`, then
`agentinputlab config validate run.json`. See the [configuration guide](configuration.md)
for budget, temperature, output length and retriever settings. Validation checks
the configuration, not endpoint availability or native runtime readiness.

For datasets, start with `agentinputlab datasets list --search beir` and
`agentinputlab datasets show beir.arguana`. See the
[dataset preparation walkthrough](benchmark-setup.md) for cache configuration,
source sizes, offline loading and native-runtime prerequisites.

## Make one real model call

No OpenAI SDK or local GPU package is required for the HTTP backend. Configure
an existing OpenAI-compatible endpoint and its actual model revision. This
example makes **one real request**, which can consume billed tokens; it does not
start an experiment, download a model, or compute a benchmark score.

```python
import os
from agentinputlab.backends.llm import GenerationRequest, OpenAICompatibleBackend

backend = OpenAICompatibleBackend(
    model_id=os.environ["AIL_MODEL"],  # Exact served model ID, not a paper nickname.
    base_url=os.environ["AIL_BASE_URL"],  # e.g. http://127.0.0.1:8000/v1
    deployment_identity={"revision": os.environ["AIL_MODEL_REVISION"]},
    api_key_env="AIL_API_KEY",  # Set only if your endpoint requires authentication.
    timeout_seconds=120,
)
response = backend.generate(GenerationRequest(
    prompt="What is the capital of France?", max_tokens=64, temperature=0,
))
print(response.text)
print(response.usage)
```

The revision is operator-declared provenance, not an independent weight check.
`base_url` is the API root; the backend appends `/chat/completions`. Do not pass
the full completion URL. Configure secrets in your environment, not source files.
Responses require the compatible chat schema and actual token usage. To apply an
intervention, supply this backend through `InterventionContext(backend=backend)`;
see [real execution and profile-specific dependencies](real-execution.md).

## Fix common setup problems

| Symptom | Check |
| --- | --- |
| `agentinputlab: command not found` | Activate the environment where pip installed the package, or use `python -m agentinputlab` (0.1.0a2+). |
| `ModuleNotFoundError` for an optional dependency | Install the selected profile's extra in its required Python/runtime environment. |
| HTTP 401 / 403 | API key environment variable, endpoint access and model permissions. |
| HTTP 404 | API root (often `/v1`) and exact served model ID. |
| HTTP 429 | Provider quota and concurrent request limits. |
| HTTP 5xx / transport failure | Local server/provider health and network reachability. |
| Profile unavailable | Check the profile's required worker, source revision and artifacts; installing the core does not provision them. |
| Unknown profile / wrong constructor parameters | `list --releases --search NAME`, then `show EXACT_RELEASE --parameters`. Suggestions never silently change the selected profile. |
| Dataset cannot be prepared | `datasets show ID` distinguishes automatic loaders from native adapters and lists source sizes and runtime requirements. |
| Cannot create artifact cache directory | Choose a writable `--cache-dir` / `cache_dir` or set `AGENTINPUTLAB_CACHE`; do not change shared research directories' permissions. |
| Response schema is invalid / missing usage | The endpoint must return explicit integer `usage.prompt_tokens` and `usage.completion_tokens`. Missing counts are unknown, not zero; fix the provider or gateway before another request. |

Version 0.1.0a2 exposes a sanitized `OpenAICompatibleTransportError`
with `http_status` for the built-in HTTP transport. Status hints contain no
provider response body, credentials or request URL. Custom transport failures
remain generic unless they use the internal sanitized status path.

There are **no automatic model retries**. A failed request may already have
consumed tokens: retain the failure and unknown cost rather than treating it as
zero usage. Decide any retry explicitly within your own declared budget.

Starting in 0.1.0a4, incomplete token-usage fields are rejected even when the
response contains an answer. Explicit zero counts remain valid. This prevents
an endpoint that omits accounting fields from silently bypassing token budgets.
