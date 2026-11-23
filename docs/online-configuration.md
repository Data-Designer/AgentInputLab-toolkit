# Run an explicit ACE online stream

`config online` calls the existing `ACEOnlineRecipe.run_stream` protocol. It
evaluates each primary prediction before adapting on that window's feedback.
This is transductive online evaluation, not an offline artifact fit or an
independent held-out result. ToolTalk remains test-only and is rejected here.

```sh
agentinputlab config online online.json
agentinputlab config online online.json --execute --output-dir ./online-attempt-001
```

The first command validates without model calls or downloads. The second runs
once, using the current Python environment's toolkit and native dependencies.
Use a new output directory, and supply the configured API-key environment
variable through your usual secret manager; do not put a secret in the file.
The Python equivalent is
`OnlineConfig.from_file("online.json").execute("./online-attempt-001")`, imported
from `agentinputlab.online_configuration`.

An explicit generic-text stream configuration:

```json
{
  "release_id": "recipe.ace.online_transductive",
  "benchmark": "hotpotqa.distractor",
  "checkout": "./upstream/ACE",
  "allow_test_feedback": true,
  "source_and_asset_acknowledged": true,
  "controls": {
    "max_rounds": 1,
    "per_case_model_calls": 8,
    "per_case_output_tokens": 4096,
    "per_call_output_tokens": 512,
    "window_size": 1,
    "max_stream_examples": 128,
    "playbook_token_budget": 4096,
    "timeout_seconds": 3600,
    "seed": 17
  },
  "deployment": {
    "api_provider": "openai",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "max_tokens_per_call": 512,
    "json_mode": true,
    "roles": {
      "target": {"model": "YOUR_MODEL", "fingerprint": {"model_id": "YOUR_MODEL", "revision": "YOUR_REVISION"}, "input_cost_per_million": 0, "output_cost_per_million": 0},
      "reflector": {"model": "YOUR_MODEL", "fingerprint": {"model_id": "YOUR_MODEL", "revision": "YOUR_REVISION"}, "input_cost_per_million": 0, "output_cost_per_million": 0},
      "curator": {"model": "YOUR_MODEL", "fingerprint": {"model_id": "YOUR_MODEL", "revision": "YOUR_REVISION"}, "input_cost_per_million": 0, "output_cost_per_million": 0}
    }
  },
  "stream": [
    {"example_id": "your-fixed-id", "input": {"question": "Your visible question", "context": "Your visible evidence"}, "expected_output": "Your acknowledged feedback label", "split": "test"}
  ]
}
```

Replace the example with your actual ordered stream, source revision, deployment
identity and prices. `split: test` is the native online role and does not grant
permission to adapt on a held-out dataset. Generic text uses the existing ACE
text feedback; setting `benchmark` alone does not select an official scorer.
All initial, pre-update, adaptation and post-update model calls count toward the
original per-case limits. Zero monetary prices mean unconfigured prices, not
zero token or call usage.

For prepared STATE, ToolSandbox or Tau feedback, replace `stream` with:

```json
{
  "feedback": {
    "kind": "tau",
    "checkout": "./upstream/tau2-bench",
    "source_input": {"path": "./prepared/public-feedback-input.json", "sha256": "TRUSTED_SHA256"},
    "environment_base_url": "http://127.0.0.1:8000/v1",
    "environment_names": ["OPENAI_API_KEY"]
  }
}
```

Set the matching `benchmark` (`tau3.retail` in this example); keep the prepared
source's original model and partition binding. The native processor derives the
same ordered stream from the pinned inputs. Tau/STATE/ToolSandbox feedback uses
their declared noninteractive tool-plan protocol, not an interactive rollout.
BFCL and retrieval use their existing online processors through the same command.
They require a separately pinned **online** source, not an offline fitting file.
For BFCL, use `feedback.kind: "bfcl"`, `feedback.model_id` equal to the target
deployment, and `feedback.source_input`. That source contains:

```json
{
  "benchmark": "bfcl.v3.simple_python",
  "source_archive": "./official-source.tar.gz",
  "normalized_root": "./prepared-bfcl",
  "prepared_file_hashes": {"ACTUAL_FILENAME": "ACTUAL_SHA256"},
  "stream": {"split": "test", "ids": ["YOUR_FIRST_ID", "YOUR_SECOND_ID"]}
}
```

Use the prepared dataset's actual complete hash mapping. The original AST scorer
is retained; selecting official test IDs for transductive feedback does not make
the adapted results independent held-out evidence.

For retrieval, use `feedback.kind: "retrieval"` and `feedback.source_input`.
Its pinned source contains `benchmark`, `dataset`, `retriever_config`, `metric`,
`top_k`, `allow_json_fence`, `retrieval_limit_per_case` and
`stream: {"split": "SOURCE_SPLIT", "ids": [...]}`. The retrieval limit is explicit
and counts uncached attempts before IO, including failed attempts. Supply the
remaining allowance after any initial capture (for example, 7 if one of 8
retrieval calls was already used); it is separate from model-call/token limits.
`dataset` and `retriever_config` use the same verified local bindings described
in [fitting configuration](fit-configuration.md); an optional `captured_states`
reference preserves saved visible contexts and their original retrieval costs.
The native IR scorer retains its qrels. The source must specify the exact stream
order; offline `groups`, train/validation IDs and automatic split inference are
not accepted. Every source and retriever path is relative to its containing file.

Results include the original native terminal and manifest references, all
primary outcomes and per-case usage. Native feedback events are saved separately;
`primary_native_feedback` links only an exact primary prediction hash. Later
adapted answers never replace failed primaries. A returned prediction does not
mean a successful task. Native failures return CLI exit code 2 and retain the
original evidence without automatic retry. A completed stream may legitimately
contain failed primaries and zero rewards.
