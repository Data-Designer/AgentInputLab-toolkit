# Patched external-source HyDE

`acquire.hyde.official_external_patched` executes the mounted upstream
`HyDE.e2e_search` and task-specific `Promptor` at the registered revision. The
toolkit does not distribute these source files, original provider clients,
weights or an index. Review your mounted source and its applicable terms.

The reviewed patch flattens each encoder result before the source computes its
mean. This supports encoders returning a `(1, dimension)` singleton batch as
well as flat vectors. The mean still includes the original query and **every**
hypothesis, including empty strings; it does not adopt the clean-room profile's
empty-hypothesis filtering. Only actual corpus hits become evidence.

```python
from agentinputlab.api import create_intervention, run_intervention
from agentinputlab import FrozenAgentState, InterventionContext
from agentinputlab.integrations.full_architecture.hyde import (
    HyDEPatchedSourceRuntime, write_hyde_patch_manifest,
)

# These two small files remain in your external, revision-pinned source checkout.
source = "/your/mounted/hyde-checkout"
receipt = write_hyde_patch_manifest(
    source, "/your/private/receipts/hyde-patch.json",
    accept_source_review=True,
)
runtime = HyDEPatchedSourceRuntime(
    source_root=source, patch_manifest=receipt["path"],
    expected_manifest_sha256=receipt["sha256"],
)
method = create_intervention(
    "acquire.hyde.official_external_patched",
    config={"task": "arguana", "num_hypotheses": 8, "top_k": 100},
)
# Supply your configured, fingerprinted backend/embedder/corpus index.
context = InterventionContext(
    backend=backend, dense_embedder=embedder, vector_searcher=searcher,
    external_runtime=runtime, seed=17,
)
result = run_intervention(method, FrozenAgentState(query="Your source query"), context)
```

Install `agentinputlab[hyde-native]` for the small NumPy runtime dependency.
The manifest is create-only and locks both original source files, the exact
patch, patched-source checksum and declared dependency substitutions. Execution
applies the patch in memory; the checkout stays untouched. Keep this receipt
outside your public package or result-free source branch.

Generation uses the runner's observed backend: separate seeded draws, the
configured 512-token maximum, temperature 0.7 and triple-newline stop by default.
The original generator client module is not imported. Serial draws, backend
sampling defaults and omitted log-probability ordering are explicit transport
differences; this entrypoint does not claim original batch-sampling or published
score parity. Embedding and search use the already-observed context objects.
Their calls/tokens are charged once. Calling-thread CPU time is a diagnostic
lower bound, excluding native worker threads; it is not charged again on top of
the enclosing monitor's process CPU accounting. Whole-process CPU windows must
not be summed across concurrent requests. An insufficient shared budget rejects before the first draw, or stops
later without silently omitting the remaining draws. No implicit retries occur.

The expected interfaces are `LLMBackend`, `DenseEmbedder` and `VectorSearcher`.
They may use caller-selected local models/services; bind and verify the exact
model, tokenizer, corpus, metric and preprocessing in your experiment recipe.
The native prompt task is explicit (`web search`, `arguana`, `scifact`, etc.).
Retrieval scores still come from the independent benchmark adapter. An input
patch is not itself a score, and hypothetical passages are not corpus evidence.

The registry keeps the profile externally gated: an importable entrypoint is
not a source-license grant, an asset bundle or whole-benchmark certification.
The other clean-room and controlled HyDE profiles remain unchanged.
