# Retriever profiles and controlled-comparison boundary

AgentInputLab exposes eight selectable retriever profiles through
`build_retriever(profile_id, documents, **options)`: one dependency-free BM25
control and seven dense recipes. The dense implementation standardizes input
formatting, bounded batch indexing, L2 normalization, cosine top-k, deterministic
tie-breaking, and runtime fingerprints. It does **not** ship model weights or
claim reproduction of published scores.

## Built-in catalog

| Profile ID | Runtime recipe | Pinned model revision | Dimensions | Query / document contract | License |
|---|---|---|---:|---|---|
| `sparse.bm25.memory.v1` | AgentInputLab in-memory BM25 | Not applicable | — | Default Unicode word tokenizer; title and text indexed together | MIT |
| `dense.bge.base_en_v1_5` | SentenceTransformers-compatible dense encoder | `BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | 768 | Query prefix `Represent this sentence for searching relevant passages: `; no document prefix | MIT |
| `dense.bge.m3` | SentenceTransformers-compatible dense encoder | `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181` | 1024 | Symmetric dense representation used by the paper RQ4 factorial | MIT |
| `dense.e5.base_v2` | SentenceTransformers-compatible dense encoder | `intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | 768 | Query prefix `query: `; document prefix `passage: ` | MIT |
| `dense.e5.large_v2` | SentenceTransformers-compatible dense encoder | `intfloat/e5-large-v2@f169b11e22de13617baa190a028a32f3493550b6` | 1024 | Query prefix `query: `; document prefix `passage: `; paper RQ4 profile | MIT |
| `dense.e5.multilingual_large` | SentenceTransformers-compatible dense encoder | `intfloat/multilingual-e5-large@3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | 1024 | Query prefix `query: `; document prefix `passage: `; distinct from English E5-large-v2 | NOASSERTION (not verified by this catalog) |
| `dense.qwen3.embedding_0_6b` | SentenceTransformers-compatible dense encoder | `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 1024 | Pinned SentenceTransformers web-search prompt ending in `Query:` (no trailing space); no document prefix; left padding | Apache-2.0 |
| `dense.contriever.msmarco` | Generic SentenceTransformers loading path | `facebook/contriever-msmarco@abe8c1493371369031bcb1e02acb754cf4e162fa` | 768 | No query or document prefix | CC-BY-NC-4.0 |

The model repository pages are the provenance sources for
[BGE](https://huggingface.co/BAAI/bge-base-en-v1.5),
[E5](https://huggingface.co/intfloat/e5-base-v2),
[Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), and
[Contriever](https://huggingface.co/facebook/contriever-msmarco). The pinned
repository revision identifies a recipe, not the bytes in a user's cache.
Every built-in dense fingerprint therefore reports `weights_bundled=False` and
`weight_files_verified=False`.

Contriever has the strictest provenance boundary: the default path asks
SentenceTransformers to load a Transformers checkpoint and use its generic
pooling compatibility behavior. AgentInputLab has not yet compared that path
against the upstream Contriever tokenizer, pooling code, index, or published
results. It is an executable controlled profile, **not** an official Contriever
parity profile. It is included because AgentInputLab is currently an academic
research project; the catalog still records the upstream CC-BY-NC-4.0 terms so
users do not mistake the project MIT license for a model-license grant.

## One public factory

```python
from agentinputlab.backends.retrieval import Document, build_retriever

documents = [
    Document("d1", "Paris is the capital of France", corpus_revision="wiki-v1"),
    Document("d2", "Berlin is the capital of Germany", corpus_revision="wiki-v1"),
]

retriever = build_retriever(
    "sparse.bm25.memory.v1",
    documents,
    k1=1.2,
    b=0.75,
)
results = retriever.retrieve("capital of France", top_k=2)
```

The factory fails closed on cross-family settings: a BM25 profile rejects
`encoder`, `batch_size`, and other dense-only options; a dense profile rejects
`k1`, `b`, and other BM25-only options. An unknown profile ID is also an error.

## Local SentenceTransformers path

Install the optional runtime without downloading a checkpoint:

```bash
python -m pip install -e '.[dense-retrieval]'
```

Then use a model already present in the local Hugging Face cache:

```python
retriever = build_retriever(
    "dense.e5.base_v2",
    documents,
    batch_size=32,
    device="cuda:0",
    # This is the default and prevents an implicit network download.
    local_files_only=True,
)
retriever.build_index()
```

Set `local_files_only=False` only when downloading the exact pinned revision is
an intended, authorized action. Construction and `fingerprint()` remain lazy;
the import and checkpoint lookup occur at `build_index()` or `retrieve()`.
Corpus documents, profile, batching, device, offline policy, and retriever ID are
read-only after construction, so a cached index cannot be silently reused after
a configuration mutation.

## API, RPC, or self-hosted embedding path

`EmbeddingBackendEncoder` bridges the existing `EmbeddingBackend` contract to
the retriever. Consequently the same profile can use a local function, an SDK,
an internal RPC service, or an OpenAI-compatible/self-hosted embedding endpoint.
The transport must return an `EmbeddingResponse` or ordered vectors accepted by
`CallableEmbeddingBackend`, and the backend must expose a stable, secret-free
fingerprint.

```python
from agentinputlab.backends.embedding import CallableEmbeddingBackend
from agentinputlab.backends.retrieval import build_retriever


def embedding_transport(request):
    # Send request.texts to the institution's approved API or local server.
    # Return EmbeddingResponse, or vectors accepted by CallableEmbeddingBackend.
    return my_embedding_client(request)


backend = CallableEmbeddingBackend(
    client=embedding_transport,
    model_id="BAAI/bge-base-en-v1.5",
    revision="deployment-2026-07-23",
    dimensions=768,
    identity={
        "protocol": "openai-compatible-embeddings",
        "deployment": "research-cluster-a",
    },
    network=True,
)
retriever = build_retriever(
    "dense.bge.base_en_v1_5",
    documents,
    embedding_backend=backend,
    embedding_request_metadata={"experiment": "controlled-study-v1"},
)
```

The adapter fingerprint contains the backend fingerprint, dimensions, and a
hash of request metadata. The retriever fingerprint incorporates the adapter
identity hash in addition to profile, corpus, batching, device, prefixes, and
normalization. API credentials must stay in the transport and must never enter
either identity mapping.

An injected encoder whose identity is derived from its own `fingerprint()` is
checked before and after every encode operation and whenever the retriever
fingerprint is requested. An explicitly supplied `encoder_identity` cannot be
independently re-queried and is therefore marked `explicit_identity_unverified`.
In either case, AgentInputLab cannot infer that a deployment alias really serves
the selected checkpoint. The experiment owner must verify the served model and
record its immutable weight/deployment revision. Results from an unverified
service must not be labeled model-card or paper parity.

## What must be locked for a fair comparison

For a controlled comparison, keep these fields identical across algorithm
arms unless the field is an explicitly crossed experimental factor:

- profile and actual encoder/backend identity;
- corpus content/revision and document formatting;
- model repository/deployment revision and local weight-file identity;
- query/document prefixes, tokenizer behavior, pooling, dimensions, and
  normalization;
- index contents, similarity metric, top-k, and tie policy;
- batching/device policy when it can change numerical behavior.

The retriever fingerprint covers the standardized runtime recipe and corpus.
It does not hash local checkpoint files. A future weight-verification layer is
therefore still required before any weight-level or published-result parity
claim.

## Verified scope as of this alpha

| Claim | Status |
|---|---|
| BM25 construction and deterministic retrieval | Offline conformance tested |
| Five-profile catalog and fail-closed factory | Offline conformance tested |
| Dense prefixes, bounded batching, dimensions, L2/cosine ranking, stable ties | Offline fixture tested |
| Lazy pinned loader arguments and no-download default | Offline mock tested |
| `EmbeddingBackend` / API injection and identity propagation | Offline fixture tested |
| Real pinned BGE, E5, Qwen3, and Contriever checkpoint load/index/query | CPU-smoke-tested offline on 2026-07-23; all four passed; full revisions, scores, vector hashes, and environment are recorded in [`qa_logs/retriever_checkpoint_smoke_2026-07-23.json`](qa_logs/retriever_checkpoint_smoke_2026-07-23.json) |
| Real BGE, E5, Qwen3, or Contriever checkpoint downloaded by CI | **No**; CI remains offline and deterministic |
| Local checkpoint byte hashes verified | **No** |
| Upstream numerical or published benchmark parity | **No** |

Reproduce the real checkpoint smoke from an existing local cache with:

```bash
python scripts/check_dense_retriever_checkpoints.py
```

The script is offline by default. Passing `--allow-download` is the only path
that permits lookup or acquisition of the exact pinned revisions.
