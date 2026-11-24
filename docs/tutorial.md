# Tutorial: from installation to a scored task

[中文](tutorial.zh-CN.md) · [README](../README.md) · [Parameter reference](configuration.md)

This tutorial separates three outputs: a changed input, a model-generated
answer, and a native benchmark score. Complete them in that order. Commands
assume Python 3.10+ and a new working directory; keep API keys out of files.

## 1. Install and try a zero-call intervention

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install agentinputlab==0.1.0a19
agentinputlab --version
agentinputlab doctor --offline --json
agentinputlab smoke --json
agentinputlab config init > run.json
agentinputlab config validate run.json
agentinputlab config run run.json --query "Find a suitable flight"
```

On Windows PowerShell activate with `.venv\Scripts\Activate.ps1`. The final
command applies `plan.plan_and_solve.ps_prompt`; inspect `state.instruction`.
It is a real prompt transformation, not a generated flight recommendation.
`validate` checks configuration; `doctor` checks local dependencies. Neither
proves that a remote model or dataset is ready.

The same operation in Python:

```python
from agentinputlab import RunConfig
config = RunConfig.from_file("run.json")
result = config.run("Find a suitable flight")
print(result["state"]["instruction"])
```

Discover exact profiles and their parameters before changing algorithms:

```sh
agentinputlab list --releases --search rrr
agentinputlab show rewrite.rrr.common_support_prompt --parameters
agentinputlab datasets list --search longbench
agentinputlab datasets show longbench.multifieldqa_en
```

## 2. Run with your model API

Clone the public repository to obtain [examples/rrr_api.py](../examples/rrr_api.py).
Install the package first; the example works with the published a15 package.
Set `AIL_API_KEY` in your terminal or secret manager without committing it.
Set these non-secret environment variables to your provider's actual values:

```sh
export AIL_BASE_URL="https://your-provider.example/v1"
export AIL_MODEL="your-served-model-id"
export AIL_MODEL_REVISION="your-deployment-revision"
python examples/rrr_api.py > rrr.json
agentinputlab config validate rrr.json
agentinputlab config run rrr.json --query "What is the capital of France?"
```

Replace the placeholder URL before running. Configuration generation makes no
model call. The last command uses the prompt-adapted RRR profile: rewrite the
question, retrieve one document from the included tiny BM25 corpus, then read
it to produce an answer. It allows two model calls, one retrieval, 512 output
tokens per generation and 1024 output tokens in total. Inspect the returned
answer, usage and execution status. This is an API demonstration, not a native
benchmark result or a reproduction of trained PPO weights.

This patch-oriented profile stores its reader answer in the execution trace:
`result["record"]["patch"]["trace"]["events"]`, in the event named
`frozen_reader_output`, under `payload.answer`. The changed query is
`result["state"]["query"]`; usage is `result["record"]["usage"]`.
`unpriced_model_calls > 0` means the provider's monetary cost is unknown,
not that the API calls were free.

Alternatively `python examples/rrr_api.py --execute` runs the same example.
For hosted models that return an alias or need provider-specific thinking /
structured-output settings, edit `backend.response_model_aliases` and
`backend.extra_body` explicitly as described in [configuration](configuration.md).
Do not put request-owned fields such as `temperature` into `extra_body`.

### Customize without guessing field names

| Goal | Edit in `rrr.json` |
| --- | --- |
| Total generated tokens / calls | `budget.max_output_tokens` / `budget.max_model_calls` |
| Stage output lengths | `parameters.config.max_rewriter_tokens` / `max_reader_tokens` |
| Sampling | `parameters.config.temperature` |
| Retrieved documents | `parameters.config.top_k` and `retriever.documents` |
| BM25 behavior | `retriever.k1` / `retriever.b` |
| API model / endpoint | `backend.model_id` / `backend.base_url` |

For other profiles, inspect `show ... --parameters`; not all share these
field names. `max_length` may include input tokens or denote compression
length. Changing a total budget does not automatically change a stage cap.

## 3. Evaluate a native benchmark

Use LongBench MultiFieldQA-en for a bounded end-to-end scoring example:

```sh
python -m pip install 'agentinputlab[longbench]==0.1.0a19'
```

Prepare the original data archive, native scorer and prompt templates at the
pinned revisions in [benchmark setup](benchmark-setup.md). These assets are
not bundled in the wheel. Review their licenses. Reuse an existing matching
download; do not substitute arbitrary files with the same name. Save their
paths in `longbench.json`:

```json
{"data_zip":"/path/to/data.zip","scorer_path":"/path/to/metrics.py","prompts_path":"/path/to/dataset2prompt.json"}
```

Create `native.json` in Python using the same API identity as above:

```python
import json, os
config = {
    "release_id": "baseline.no_op", "seed": 17,
    "budget": {"max_output_tokens": 512, "max_model_calls": 1},
    "backend": {
        "model_id": os.environ["AIL_MODEL"],
        "base_url": os.environ["AIL_BASE_URL"],
        "deployment_identity": {"revision": os.environ["AIL_MODEL_REVISION"]},
        "api_key_env": "AIL_API_KEY"
    }
}
with open("native.json", "x", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
with open("predictor.json", "x", encoding="utf-8") as f:
    json.dump({"schema_version": 1, "native_text": {
        "max_output_tokens": 512, "temperature": 0.0}}, f)
```

Here `baseline.no_op` leaves the input unchanged; the evaluator still calls
the downstream model. It is not a claim to reproduce any paper's ReAct profile.

```sh
agentinputlab config evaluate native.json \
  --benchmark longbench.multifieldqa_en --split test --limit 1 \
  --longbench-config longbench.json --predictor-config predictor.json \
  --output native-evaluation.jsonl
```

The output file records started and terminal events. A completed record includes
the prediction, native metrics and usage. `scored_count` tells you how many cases
were scored; retain failed rows and their denominator. A low score is a model
outcome, not an installation failure. Do not call a one-case check a benchmark
leaderboard result. To try Plan-and-Solve, copy the configuration to a new file,
set `release_id` to `plan.plan_and_solve.ps_prompt`, and choose a new output path.

For a first-time download, replace `--longbench-config longbench.json` with
`--cache-dir ./longbench-cache --accept-license-review` after reviewing the
upstream licenses. This cache-first route verifies and reuses the same pinned
sources; its initial download is about 114 MB shared by the six subsets.

The equivalent Python entry point is `RunConfig.evaluate(...,
longbench_sources=..., predictor_parameters=...)`; see [configuration](configuration.md).

## 4. One workflow and browser controls

Try preparation, execution and persisted status without a model first:

```sh
agentinputlab workflow init --query "Find a suitable flight" > workflow.json
agentinputlab workflow inspect workflow.json
agentinputlab workflow prepare workflow.json --output-dir prepared-workflow
agentinputlab workflow run prepared-workflow/workflow.json --execute --output-dir workflow-run
agentinputlab workflow status workflow-run
agentinputlab-studio --workflow-root ./studio-workflows
```

Open the printed `/workflows` URL. Edit the preset, method, backend, budget and
advanced JSON; generate the configuration, check prerequisites, then explicitly
execute. Both CLI and Studio use the same engine. Missing checkpoints, source
files or unsupported routes must be resolved before model calls. Use the
[workflow guide](workflows.md) for benchmark preparation, download authorization,
concurrency and continuation. Resume saved workflows; do not blindly relaunch
an interrupted command with a new output file.

For a remote host, use an SSH tunnel and keep Studio on localhost. The UI is
for a trusted operator, not a public multi-user service. Credentials belong in
the server environment, not form fields or screenshots.

## 5. Troubleshooting and acceptance

| Symptom | What to check |
| --- | --- |
| Command not found | Activate the environment; try `python -m agentinputlab` |
| GitHub download times out | Install from official PyPI; keep TLS verification enabled |
| Missing key / 401 / 403 | Environment-variable name, provider account and permissions |
| 429 / provider exhausted | Wait for quota or adjust concurrency; do not rerun consumed tasks blindly |
| Missing runtime / artifact | The selected profile's `show` output and setup guide |
| Wrong response model / unsupported schema | Explicit provider identity and transport configuration |
| Configuration accepted but no answer | Check whether the profile only proposes an input patch |
| Output exists | Inspect the journal; resume supported workflows, preserve previous attempts |

See [validation](validation.md) for what is tested offline versus with real
models. A passing install check covers packaging; a native scored run additionally
tests that particular method, dataset and model configuration. Keep the version,
configuration, source hashes, case IDs and terminal outcomes with your receipt.
