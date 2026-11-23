# Fit and reload a ReFormeR pattern library

`training.reformer.pattern_induction` calls the native
`IterativePatternExtractor.extract_patterns_from_batch` at upstream commit
`72e52450a922bc0051b5b39c3a6307186a555137`. It executes the native prompt,
JSON parser, and iterative replacement of the consolidated library. It does
not vendor the upstream repository or its training pairs. Install optional
dependencies with `pip install 'agentinputlab[reformer-fit]'`, and mount your
own pinned checkout and permitted training data.

```python
from agentinputlab.recipes.reformer import (
    ReFormeRInductionControls, ReFormeRPatternArtifact,
    ReFormeRPatternInduction, ReFormeRQueryPair,
)
from workers.reformer.driver import BackendConfig, OpenAICompatibleBackend

# backend.json contains endpoint/model identity, bounded per-call token limits
# and the NAME of an environment variable holding a credential, never its value.
config = BackendConfig.from_json(open("backend.json", encoding="utf-8").read())
fit = ReFormeRPatternInduction(ReFormeRInductionControls(batch_size=2, seed=17))
artifact = fit.fit(
    checkout="/path/to/ReFormeR",
    training_pairs=train_pairs,       # sequence of ReFormeRQueryPair(split="train")
    held_out_pairs=held_out_pairs,    # declared validation/test pairs, never prompted
    output_root="/path/to/new-fit",  # must not exist; no implicit resume or retry
    backend_config=config,
    backend=OpenAICompatibleBackend(config),
    source_license_acknowledged=True,
)
reloaded = ReFormeRPatternArtifact.load(artifact.root)
print(reloaded.pattern_path, reloaded.pattern_sha256)
```

For the same API from a shell, provide a JSON list of query-pair records:

```bash
python -m workers.reformer.induction --checkout /path/to/ReFormeR \
  --pairs split-pairs.json --backend-config backend.json --output /path/to/new-fit \
  --acknowledge-external-source --batch-size 2 --max-patterns 25 --seed 17
```

Each record contains `query_id`, `original_query`, and `split` (`train`,
`validation`, or `test`). Training records also require a nonempty
`reformulated_query`. Validation and test records may omit that field: their
IDs and original queries support the split audit and never enter fit prompts.
All supplied IDs and normalized original queries must be unique and disjoint.
Declare the full evaluation inventory to audit its separation from training;
the audit covers the records you supply. Pair order is explicit, with no
implicit sampling, shuffle, benchmark access, or test-label selection.

The adapter repairs two pinned-source import bindings and replaces the native
Ollama transport with the explicit bounded backend. Native Ollama construction
is blocked. Generation temperature and the per-call cap come from backend
configuration; total calls, prompt-plus-output tokens and price ceilings come
from induction controls. Batch `i` uses `seed+i` (zero-based). `max_patterns`
is the native prompt instruction, not a newly imposed truncation rule.
These choices are recorded as a controlled transport profile, not the original
paper's model or unrestricted Ollama training configuration.

Fit directories contain immutable start/terminal records, per-batch model
requests/responses, native parse results, and a hash-bound pattern artifact.
Invalid or empty final libraries remain paid failed fits, with no automatic
retry or deployment. Candidate generation alone is not held-out evaluation.
Provider token costs are retained; unknown provider usage is not treated as
free. Local unpriced inference should be reported as unpriced separately.

To evaluate a successful fitted candidate, pass `reloaded.pattern_path` as
`pattern_path` to `workers.reformer.driver.ReFormeRRuntime`; bind
`reloaded.pattern_sha256` in the corresponding inference profile. The inference
worker verifies the library and executes the unchanged native selection and
application. Evaluate on disjoint held-out queries and retain real failures
before claiming a deployment or quality improvement. A candidate artifact
does not automatically replace the bundled upstream library.
