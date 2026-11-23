# Installed-user acceptance — 23 November 2026

This check exercises the public product, separately from paper experiments.
The live cases below are fixed user tests, not a full algorithm × benchmark ×
model sweep. No research result is changed.

## Installation and interface checks

| Installation | Checks | Result |
| --- | --- | --- |
| Official PyPI a15, fresh Linux Python 3.13 environment | Installed consumer contracts | 147 passed, 0 failed |
| Public GitHub a15 source, separate fresh environment | Same installed contracts | 147 passed, 0 failed |
| a16 candidate wheel | Same installed contracts | 147 passed, 0 failed |
| a15 and a16 user journeys | CLI, Python, configuration, actual Studio HTTP, restart/resume | 20 checks passed per tested installation |

The contracts cover 27 method identities, 87 profiles, 86 executable entry
points and 31 benchmark entries (21 with automatic loaders). Discovery checks
inspect every registered entry; they do not execute every model-dependent
profile. Server-side GitHub download timed out, so source installation used
an archive of the verified public commit. Official PyPI installation worked
directly. Studio used actual localhost HTTP requests, including configuration,
preflight, execution and cross-origin rejection. The resumed journal was
unchanged. This is not a browser-rendering audit.

## Fixed live model and native-scoring checks

Provider: official DeepSeek API; requested and returned model ID
`deepseek-flash`, operator-declared deployment revision `1017`, seed 17,
temperature 0, thinking disabled, at most 512 output tokens per generation.
This records the tested identity, not an immutable provider-alias guarantee.

| Scenario | Physical model requests | Outcome |
| --- | --- | --- |
| RRR rewrite → inline BM25 → reader | 2 across initial attempt and repaired continuation | Completed after provenance repair; first rewrite reused unchanged |
| LongBench MultiFieldQA-en, unchanged-input baseline | 1 | Native answer F1 = 0.75 |
| Same fixed case, Plan-and-Solve prompt | 1 | Native answer F1 = 0.75 |
| LongBench QMSum, unchanged-input baseline, CLI evaluation | 1 | Native ROUGE = 0.2639593860 |

LongBench case IDs were frozen before calls:
`c29e95ab6195406aceecf3874186150cb1b8b26db5bcd0e4` and
`fdd371de2668a6f1e7914fe9a67aef33927ecc392fdc2606`. The adapter verified the
original data, prompt and scorer hashes; no replacement scoring was used.
Each native case had a 512-output-token / one-call budget. The RRR demo had
1024 output tokens / two calls / one retrieval and is not a benchmark score.

Saved responses were reused only when endpoint, declared model identity and
the entire request body matched. Rechecking installed distributions and repaired
code did not regenerate those answers. Five physical requests were made in
total; logical calls during replay are reported separately.

## Problems found and repaired

1. The previous RRR tutorial omitted inline-corpus provenance. Its first real
   rewrite succeeded, but the method then rejected the retrieved document.
   a16 derives missing inline revisions from content; it does not relax the
   RRR parser or remove provenance validation. The original failure is retained.
2. Studio's unified-workflow form offered 2–4 workers although that executor
   currently accepts one. The form and documentation now match the executor.
   Separate campaign concurrency is unchanged.
3. Guides still called published API options development-only, and one setup
   command pinned an older release. Those instructions are corrected.

Synthetic regression covers the repaired paths, content-sensitive revision
changes, explicit-revision preservation, typo rejection and tutorial links.
Live results above are recorded separately; no synthetic response is presented
as a provider response. Profiles requiring checkpoints, training products or
native environments still require those resources, as listed in the catalog.

Release CI and installed-artifact scans remain required in addition to this
bounded live acceptance. See [validation](validation.md) for the acceptance levels.
