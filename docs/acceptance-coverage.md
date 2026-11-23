# What acceptance establishes

The public catalog has **27 method identities, 87 profiles and 31 benchmark
entries**. These are discovery counts, not counts of successfully tested
combinations. The historical review considered four backbone identities;
87 × 31 × 4 = 10,788 review positions, not 10,788 successful real-model runs.

Real model/data acceptance was performed within an explicitly approved scope.
It includes actual endpoint responses and native scoring where the adapter and
scorer were available. That scope also retained scientific failures, reused
saved responses without extra model calls, and recorded setup/protocol failures.
It did not turn a failed attempt into a pass or repeatedly retry until success.
Research evidence remains separate from this distribution.

## Read statuses literally

| Evidence or status | Meaning | Not established |
| --- | --- | --- |
| Imported / discoverable profile | Entrypoint and metadata can be inspected | Dependencies provisioned, a model called, or benchmark eligibility |
| Offline tests / installed consumer checks | Contracts, configuration, API/CLI and failure handling verified | Real-model execution or task accuracy |
| Saved-response replay | Existing response reused under a named adapter/scorer | A new model attempt or all current provider versions tested |
| Real terminal result | A particular profile, model revision, case, budget and scorer executed | All cases/backbones work, or the model answered correctly |
| Paused / deferred / blocked / inapplicable | Work intentionally excluded or prerequisites unsatisfied | Acceptance passed |

The historical approved scope deferred additional SFT/RL and unavailable
sandbox-dependent execution, paused selected methods/benchmarks, and kept
optional author-weight profiles and runtime-version conflicts explicit. Those
catalog entries remain available for discovery or appropriately provisioned
integrations; they are not advertised as universally accepted. The current
profile's metadata and prerequisites take precedence over an older summary.

## Concrete limits of the reviewed release scope

The October 2026 release review did **not** certify the full Cartesian product.
The following are configuration positions, not unique model requests or failed
tasks. They describe a historical review, not an automated scan of your machine.

| Recorded disposition | Positions | Why it is not a pass |
| --- | ---: | --- |
| Owner-paused methods or benchmarks | 1,276 | Execution deliberately paused |
| Restricted profile variants | 270 | Outside the approved execution scope |
| Optional RRR author-weight variants | 112 | Exact warm-up/PPO assets not certified in this release scope |
| R²AG pinned-reader runtime conflicts | 88 | Model architectures not recognized by the upstream pinned runtime |
| Owner-reviewed profile inapplicability | 24 | Profile not applicable under the fixed native input contract |

These five dispositions partition 1,770 unrun positions in the pending ledger.
The wider row-status index separately records 868 additional-SFT/RL positions
and 372 isolated-execution positions as deferred. Neither those counts nor the
remaining catalog positions are a count of successful real-model runs. Other
positions retain native input/cache inapplicability, unsuccessful fitted
dependencies, malformed or truncated model responses, and actual task failures.
There is no supported conversion from "reviewed/closed" to "passed".

### What can be repaired versus what needs a different setup

| Profile or dependency | Current boundary | Practical path |
| --- | --- | --- |
| R²AG upstream pinned reader | Transformers 4.40.2 does not recognize the reviewed Qwen3, Qwen3.5, GPT-OSS and Gemma4 architectures | Use the documented reader-aligned route with its own compatible runtime and validated artifact. A dependency upgrade alone is not evidence of upstream parity. |
| RRR author checkpoints | Separate external weight profiles | Start with `rewrite.rrr.common_support_prompt` and your configured model/retriever; it is a controlled prompt adapter, not PPO reproduction. |
| HyDE native default | Eight hypotheses plus a downstream reader need at least nine model calls | Configure a sufficient budget for a new run or retain the explicit protocol conflict. The toolkit does not silently drop hypotheses or raise your cap. |
| ACON / ReFormeR / SkillOpt and fitted variants | Validated fit artifacts or native development data may be required | Provision the exact artifact or run an explicitly budgeted fit. A failed fit and a cross-model artifact are not valid substitutes. |
| Cached profiles | Exact query/task/split identity must be present | Provide a matching cache or choose a separately identified live profile. Cache misses are not zero-cost successful runs. |
| OpenHands / generated-code MCE paths | Isolated execution was unavailable; `clarify.ask_or_assume.openhands_full` has no execution entrypoint | These paths remain outside the accepted scope. Do not execute generated code unsandboxed to bypass the prerequisite. |
| Additional SFT/RL variants | Training was deferred | Inference with already validated assets is distinct from testing a new training recipe. |

Adapter defects can be fixed and retested. A valid, budget-exhausted response or
an incorrect task answer is a recorded outcome, not a reason to automatically
retry until a passing answer appears. See `agentinputlab docs troubleshooting`
for first-run diagnostics.

## Check your configuration

1. Inspect `agentinputlab show EXACT_RELEASE --parameters` and
   `agentinputlab datasets show BENCHMARK_ID`.
2. Provision the stated runtime, dataset split, model revision and scorer.
   Some extras require separate environments; do not install every extra together.
3. Validate your [run configuration](configuration.md), then execute a fixed real
   case with your chosen budget. Record the exact source and dependency versions.
4. Check actual usage, terminal status and native score separately. A zero score
   can be a valid outcome; `scored: false` is not a benchmark success.

No package can guarantee that every provider will stay available, every model
will produce valid output, or every task will succeed. AgentInputLab's contract
is explicit configuration, recorded costs and typed failures—not guaranteed
scientific success. Changes to a profile, model, corpus, runtime or budget need
their own acceptance check.
