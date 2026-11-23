# IRCoT prediction archives

`acquire.ircot.prediction_replay` reads the upstream answer prediction format:
a JSON object mapping exact question IDs to answer strings. It returns an
archived answer as a composite result, not a newly generated answer or an
evidence paragraph. Retrieval-ID lists are a different output type and are
rejected by this answer interface.

The [upstream download script](https://github.com/StonyBrookNLP/ircot/blob/3c1820f698eea5eeddb4fba3c56b64c961e063e4/download/predictions.sh)
links the authors' prediction archive. Keep downloaded files, configurations,
provenance and results outside the toolkit source tree. The package neither
downloads this archive implicitly nor redistributes its contents.

An operator manifest with schema `agentinputlab.ircot.prediction_replay.v1`
contains these exact fields:

- `origin_kind`: `author_release` or `controlled_capture`, never mixed silently.
- `run_revision`: the actual immutable Git revision of the generating run.
- `benchmark`, `split`, `selection_rule`: the declared evaluation scope and an
  ID-based selection rule. A subset must not be selected by answer quality.
- `generator_fingerprint`, `dependency_fingerprint`: the original model and
  corpus/index/prompt identities, not the identities of a desired replacement.
- `provenance`: `source_uri`, `source_artifact_sha256`, `run_config_sha256`, and
  `query_source_sha256`. Hashes establish byte identity, not authorship or a
  redistribution grant; review source provenance independently.
- `predictions`, `inputs`: relative `path` and `sha256` pairs. The original
  prediction file is unchanged. `inputs` maps selected IDs to label-free records
  with `query`, `input_state_hash`, `historical_usage`, `origin_receipt_sha256`.

The selected input IDs must exist in the source prediction file. Queries must
match exactly. Known original state hashes are checked; a null state hash
means only ID/query identity is established, not context or budget parity.
Captured toolkit runs require their original receipt checksum. Published
archives with unavailable token telemetry use `historical_usage: null`, not
invented zero costs. Recorded historical usage is inspectable and is not
charged again as new inference. Empty archived answers remain in the inventory
and fail on execution; they are not retried, filled or silently omitted.

```python
from agentinputlab import FrozenAgentState, InterventionContext
from agentinputlab.api import create_intervention, run_composite_intervention

method = create_intervention(
    "acquire.ircot.prediction_replay",
    manifest_path="/your/private/archive/manifest.json",
    expected_manifest_sha256=trusted_manifest_sha256,
    expected_generator_fingerprint_sha256=trusted_original_generator_hash,
    benchmark="hotpotqa.distractor", split="validation",
)
result = run_composite_intervention(
    method, FrozenAgentState(task_id=source_id, query=source_question),
    InterventionContext(),
)
answer = result["final_answer"]
```

An optional live backend must have the same fingerprint as the archive model;
it is never called. Run the benchmark's independent scorer on the returned
answer. `scored=False` from the toolkit execution API is not a benchmark score.
The archive's full prediction count and selected record count are both traced.
Replay does not demonstrate fresh inference under the current seed, budget,
model or deployment, and does not unlock other benchmark/model configurations.

This interface is separate from the native retrieval-only IRCoT worker and
the complete upstream QA configuration. It does not relabel a retrieval-chain
trace as a full upstream QA run.

## Score saved answers with the public CLI

Save a `RunConfig` containing the same release and pinned parameters shown
above to `run.json`. No `backend` is required for answer-only replay. For QA
datasets, put the following reader settings in `predictor.json`:

```json
{"schema_version": 1, "qa": {"output_format": "answer_only"}}
```

```sh
agentinputlab config evaluate run.json --benchmark hotpotqa.distractor \
  --split validation --case-id YOUR_SOURCE_ID --prepared-root /your/prepared/data \
  --predictor-config predictor.json --offline --output new-results.jsonl
```

The independent dataset adapter supplies the original gold and scorer. The
result records distinguish new usage (zero model calls) from the archive's
historical usage, which remains unknown when the source did not record it.
Select the complete intended ID set before scoring; empty saved answers remain
failed endpoints in that denominator.

Some controlled captures used a question-only state rather than the dataset
adapter's instruction and visible context. For those captures only, explicitly
add `"archived_input": {"projection": "ircot_question_only.v1"}` to
`predictor.json`. The original task ID, exact question and unchanged execution
budget must reproduce the capture's non-null state hash. A mismatch fails
before replay; this option does not relax identity checks or change the scorer.
It cannot be combined with other input preparation or used for author archives.
Leave it unset when the original state already matches.

For a controlled LongBench capture that moved visible documents into an external
index, use `"archived_input": {"projection": "ircot_longbench_task.v1",
"context_marker": "EXACT_MARKER_FROM_YOUR_CAPTURE"}` instead. This reconstructs
the native dataset task template (including its instruction), with the original
marker in place of document context. Substituting the visible documents back must
recover the native prompt exactly; the projected task and unchanged budget must
also reproduce the pinned capture's exact state hash. A different marker,
template, task, or budget is rejected. Scoring still uses the independent native
dataset adapter. This is saved-answer scoring, not a fresh IRCoT retrieval run;
it neither reconstructs missing retrieval evidence nor generates new answers.

## Score original retrieval rankings

Answer archives alone do not contain document rankings. When your controlled
run also retained its original ranked document IDs, `config evaluate` can score
that separately pinned receipt using the native IR dataset scorer:

```json
{
  "schema_version": 1,
  "archived_input": {"projection": "ircot_question_only.v1"},
  "archived_retrieval": {"path": "ranking-receipt.json", "sha256": "TRUSTED_SHA256"}
}
```

Use the existing `config evaluate` command with the matching BEIR, BRIGHT or
TREC-DL benchmark, its original test IDs and dataset source configuration. The
CLI resolves the receipt path relative to this predictor JSON; Python callers
supply an absolute path. No live backend or retriever is required or allowed in
this mode. The dataset source still needs to be installed/prepared independently.

The receipt schema `agentinputlab.ircot.archived_retrieval.v1` has exactly these
fields: `schema`, `benchmark`, `split`, `prediction_manifest_sha256`,
`generator_fingerprint_sha256`, `dataset_fingerprint_sha256`, and `records`.
Each task-ID record has `query`, `input_state_hash`, `origin_receipt_sha256`,
`answer_sha256`, and `ranked_ids`. Bind it to the same original inference receipt
as the answer archive. Use `agentinputlab.contracts.serialization.content_hash`
for the answer string and native adapter fingerprint; the receipt's outer
`sha256` is the hash of its file bytes. Export only saved original IDs in their
original order, including empty rankings. Do not derive rankings from answers,
qrels or desired scores. Duplicate IDs and embedded metrics/gold are rejected.

The scorer receives descending rank scores; this mode is for runs whose original
scoring protocol used that order, not original numeric retrieval scores. The
answer result remains unchanged (`evidence=[]`); `downstream_metadata` separately
identifies archived-rank scoring, the original receipt and dataset fingerprint.
Empty or missing answers remain failed endpoints even if a rank receipt exists.
Historical costs remain in the answer metadata, with zero new model/retrieval
calls. This proves reproducible saved-result scoring, not fresh inference or
independent authorship of operator-supplied artifacts.
