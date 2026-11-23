# Third-party source and asset boundaries

AgentInputLab standardizes interfaces and supplies clean-room control logic,
thin adapters, replay schemas, and external-worker protocols. It does not
vendor the audited upstream repositories, model weights, or benchmark data.
Small worker-side compatibility paths may follow a permissively licensed
upstream control flow; those paths name the source symbol and exact revision in
their local manifest and preserve the upstream attribution.

The authoritative, commit-pinned inventory is
`manifests/algorithms/registry.yaml`. The following summary is a release
safeguard, not legal advice.

| Method | Source license disposition | Core distribution rule |
|---|---|---|
| Query2doc official generations | CC-BY-4.0 asset | Download by manifest; preserve attribution |
| QueryGym | Apache-2.0 | Optional community adapter; source not vendored |
| Rewrite-Retrieve-Read | NOASSERTION | External checkout only |
| ReFormeR | NOASSERTION | External checkout only |
| InfoCQR | NOASSERTION | External checkout only |
| Plan-and-Solve | NOASSERTION | Clean-room paper-spec or external checkout |
| Decomposed Prompting | Apache-2.0 | Pinned external worker; bounded best-first compatibility path follows `BestFirstDecomposer.find_answer_decomp` and preserves attribution |
| Self-Ask | MIT | Minimal faithful engineering; preserve notice |
| LLMCompiler | MIT | Optional external/full-system integration |
| Plan-and-Budget | MIT | Minimal logic port requires preserved notice |
| HyDE | NOASSERTION | Clean-room paper reimplementation only |
| IRCoT | Apache-2.0 | Isolated external worker |
| Adaptive-RAG | Apache-2.0 | Rebuild recipe; assets separately licensed |
| R²AG | MIT | Native/factorial integration; weights separately gated |
| Q-RAG | NOASSERTION | User-provided checkout/checkpoints only |
| Ask or Assume? | MIT root; PolyForm enterprise subtree | Enterprise subtree excluded |
| UDCS | Apache-2.0 | Prompt/component integration; preserve notice |
| LongLLMLingua and LLMLingua-2 code | MIT | Lazy official adapter; weights/data separately governed |
| LLMLingua-2 MeetingBank training data | CC-BY-NC-SA-4.0 | Never bundled by runtime package |
| RECOMP code | MIT | Weights and training data remain NOASSERTION |
| ACON | MIT | Runtime/recipe integration; paper/code drift disclosed |
| SmartSearch | MIT | Optional isolated inference/training environment |
| RQ-RAG | NOASSERTION | External checkout only |
| ACE | Apache-2.0 | External worker/replay protocol |
| MCE | MIT | Sandboxed external recipe only |

The Plan-and-Budget allocation/schedule port follows
`junhongmit/P-and-B@2ca2d90c6d3d630ec2b11cc22e0f56751f7f2b7e`.
Copyright (c) 2024 Junhong Lin. Its upstream MIT permission notice is reproduced
in `NOTICE`; cache datasets and their underlying terms are separate and are not
bundled. Solver references/answers are excluded from the public cache view.

The Query2doc cached loader audits only the author's two native TREC split files
at `intfloat/query2doc_msmarco@3019b760cd849830aec6d0654dbe8a4a03e791d6`.
Attribution: Liang Wang, Nan Yang and Furu Wei, *Query2doc: Query Expansion with
Large Language Models* (EMNLP 2023). These are `text-davinci-003` generations;
the package contains checksums and loading code, not the generations. Use also
requires review of the underlying MS MARCO data terms.

## Optional retrieval model profiles

AgentInputLab distributes only model identifiers, immutable repository
revisions, and preprocessing contracts for the profiles below. It does not
bundle, mirror, or redistribute model weights. A configured runtime obtains
weights from the user's local cache or, only when explicitly enabled, from the
named upstream repository.

| Profile asset | License disposition | Core distribution rule |
|---|---|---|
| BAAI BGE base English v1.5 | MIT | Pinned downloader/local-cache profile; no bundled weights |
| intfloat E5 base v2 | MIT | Pinned downloader/local-cache profile; no bundled weights |
| Qwen3 Embedding 0.6B | Apache-2.0 | Pinned downloader/local-cache profile; no bundled weights |
| facebook Contriever MS MARCO | CC-BY-NC-4.0 | Included for the current academic research scope; pinned downloader/local-cache profile; no bundled weights |

## Benchmark and metric dependencies

| Asset or package | License disposition | Distribution rule |
|---|---|---|
| BEIR ArguAna official archive | CC-BY-SA-4.0 | Checksum-pinned downloader only; data is not included in wheels, sdists, containers, or Git |
| BRIGHT Earth Science and Economics | CC-BY-4.0 | Official query/qrel and corpus Parquet components are independently checksum-pinned; downloader only |
| MuSiQue v1.0 | CC-BY-4.0 | Official archive is checksum-pinned; only answerable dev and its evaluator-only overlap audit are normalized locally |
| 2WikiMultiHopQA `data_ids_april7` | Apache-2.0 | Official corrected archive and alias table are checksum-pinned; only dev is normalized locally |
| HotpotQA distractor validation | CC-BY-SA-4.0 | Official Hugging Face organization mirror is checksum-pinned; source annotation defects remain auditable |
| QASPER v0.3 train/dev archive | CC-BY-4.0 | Official archive is checksum-pinned; only the caption-augmented dev setting is normalized locally |
| tau2 1.0.1 Banking Knowledge source | MIT | Exact-commit archive is checksum-pinned and downloader-only; hidden simulator, evaluator, and database state remain evaluator-side in local prepared caches |
| AppWorld / AppWorld-UL | Apache-2.0 code repository; benchmark assets separately governed upstream | Exact-commit external runtime bridge; no runtime, simulator state, evaluator gold, or task rows are vendored |
| Microsoft STATE-Bench 0.8.1 | MIT | Exact-commit external runtime bridge; only a project-authored hash-locked task-ID membership index is packaged |
| Apple ToolSandbox | Apple source-code license in the repository `LICENSE` | Exact-commit external runtime bridge for academic evaluation; upstream runtime, scenarios, and evaluator implementation are not redistributed |
| ClariQ official dev TSV | NOASSERTION | Explicit local license-review gate; no rows or derived cache are redistributed |
| `ir-measures` | MIT | Optional `evaluation` dependency; resolved version is recorded in each evaluation environment |
| `pyarrow` | Apache-2.0 | Optional `benchmarks` dependency used to read pinned upstream Parquet assets; no benchmark rows are bundled |
| `rouge-score` | Apache-2.0 | Optional `evaluation` dependency; resolved version is recorded in each evaluation environment |

The byte locks and source provenance for audited smoke data are maintained in
`manifests/benchmarks/smoke_assets.json`. Synthetic unit-test fixtures are
project-authored and contain no copied benchmark rows.

For all assets, source, model, and data licenses are evaluated independently.
An open-source code license never propagates automatically to weights or data.
