# AgentInputLab

A Python toolkit and typed JavaScript client for composing and evaluating input interventions for LLM agents.

This public repository contains reusable library code, worker adapters, synthetic tests and documentation. It contains no paper results, experiment logs, datasets, trained weights or private execution configuration.

[Website](https://data-designer.github.io/AgentInputLab/) · [PyPI](https://pypi.org/project/agentinputlab/) · [Source](https://github.com/Data-Designer/AgentInputLab-toolkit)

## Install

The commands below work without GitHub access. All Markdown guides are bundled
in the installed package, including a real HTTP backend example and troubleshooting.

Python 3.10+; optional model dependencies have their own Python/platform constraints.

```sh
python -m pip install agentinputlab==0.1.0a15
python -m agentinputlab --version
python -m agentinputlab doctor --offline --json
python -m agentinputlab smoke --json
python -m agentinputlab docs getting-started
python -m agentinputlab docs configuration
python -m agentinputlab docs acceptance-coverage
```

Use `agentinputlab docs --list` and `agentinputlab docs TOPIC` to read the guides
locally, with no repository account, source extraction or optional dependencies.

To obtain the public source and included documentation from PyPI:

```sh
python -m pip download --no-deps --no-binary=agentinputlab agentinputlab==0.1.0a15
tar -xzf agentinputlab-0.1.0a15.tar.gz
cd agentinputlab-0.1.0a15
python -m pip install .
```

Alternatively, use the public Git checkout:

```sh
git clone https://github.com/Data-Designer/AgentInputLab-toolkit.git
cd AgentInputLab-toolkit
python -m pip install .
agentinputlab doctor --offline
agentinputlab smoke --json
```

The Python distribution is an initial alpha. The JavaScript SDK remains source-only while npm publication is deferred. To build local Python and SDK artifacts:

```sh
python -m pip install build
python -m build
python -m pip install dist/agentinputlab-0.1.0a15-py3-none-any.whl
cd packages/sdk
npm ci
npm test
npm pack
# From a JavaScript application:
npm install /absolute/path/to/agentinputlab-sdk-0.1.0-alpha.2.tgz
```

## Python: apply a real intervention

### Discover before running (0.1.0a3+)

```sh
agentinputlab list --releases --search query2doc
agentinputlab show plan.plan_and_solve.ps_prompt --parameters
agentinputlab datasets list --search beir
agentinputlab datasets show beir.arguana
```

These commands inspect metadata and constructor parameters without running an
algorithm or downloading data. Dataset details include source sizes and native
runtime prerequisites. Start with the [first-run guide](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/getting-started.md)
and [dataset walkthrough](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/benchmark-setup.md).

### Edit a run configuration (0.1.0a6+)

```sh
agentinputlab config init > run.json
# Edit run.json: budget, seed and the selected method's parameters.
agentinputlab config validate run.json
agentinputlab config run run.json --query "Find a suitable flight"
```

The starter is a prompt-only Plan-and-Solve intervention, not a model answer.
`RunConfig.from_file("run.json").run(...)` provides the same interface in Python.
The [configuration guide](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/configuration.md) explains model endpoints,
temperature, per-call output caps, cumulative budgets, BM25 and custom/dense
retrievers. Parameter inspection includes nested configuration fields. Unknown
fields and conflicting generation overrides are rejected instead of ignored.

### Prepare, execute and score together (0.1.0a12)

Version 0.1.0a12 adds a unified
[workflow entrypoint](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/workflows.md) over the existing native executors:

```sh
agentinputlab workflow init --run-config run.json --benchmark bfcl.v3.simple_python --limit 1 > workflow.json
agentinputlab workflow inspect workflow.json
agentinputlab workflow launch workflow.json --execute --allow-network --output-dir workflow-run
```

Missing inputs, incompatible budgets and unavailable resources are reported
before generation. Preparation acquires bounded pinned assets and uses owned
environments. Completed attempts, including real failures, are not retried by
`--resume`. Exact profiles that require a native Python runtime or explicit
training configuration are identified by `workflow recipes`; they are not
silently replaced with file-configured variants.

The installed package includes an editable [Studio workflow builder](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/studio-workflows.md):
start `agentinputlab-studio --workflow-root ./studio-workflows` and open
`/workflows`. It generates/imports/exports `workflow.json` and `run.json`, checks
prerequisites, executes the native pipeline and resumes saved runs. Studio runs
locally; the public website does not receive credentials or execute models.

### Apply a prompt intervention

```python
from agentinputlab import FrozenAgentState, InterventionContext
from agentinputlab.api import create_intervention, run_intervention

method = create_intervention("plan.plan_and_solve.ps_prompt")
result = run_intervention(method, FrozenAgentState(query="Find a suitable flight"),
                          InterventionContext(seed=17))
print(result["state"]["instruction"])
```

This executes the Plan-and-Solve prompt intervention. It is not an LLM response or a scored benchmark result. For model-driven methods, supply an explicit backend, retriever or external worker in `InterventionContext`; constructors retain their documented method-specific parameters. See [real execution](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/real-execution.md).

```python
from agentinputlab.api import catalog
from agentinputlab.registry import AlgorithmRegistry

print(catalog()["benchmarks"])
registry = AlgorithmRegistry.default()
for method, profile in registry.list_releases():
    print(method.method_id, profile.id, profile.status.value, profile.blocking_reasons)
```

The catalog exposes 27 method identities and 31 benchmark entries, including ToolTalk and datasets outside the paper's reporting subset. Choose a release profile and provision its declared code, checkpoints, environment and native scorer. Profile registration and real execution are separate compatibility dimensions; see [validation and profile prerequisites](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/validation.md). Composite agents retain their native interfaces.

For Rewrite-Retrieve-Read (RRR), start with
[`rewrite.rrr.common_support_prompt`](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/rrr-controlled.md): the controlled
prompt adapter uses your configured backbone and retriever for the
rewrite → retrieve → reader chain. Author warm-up/PPO checkpoints are separate,
optional external profiles.

For offline optimization, see [Q-RAG value-policy training](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/qrag-training.md)
and [R²AG reader-aligned training and generation](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/r2ag-training.md). These
recipes save fitted artifact identities and separate training costs from
subsequent inference.

## JavaScript / TypeScript

The SDK connects to the same Python execution core; it does not reimplement algorithms. In Python, configure a server with explicit profile factories and dependency contexts. The CLI exposes only NO_OP and Plan-and-Solve by default:

```sh
# Set a newly generated local token of at least 24 characters in your shell.
export AGENTINPUTLAB_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
agentinputlab-serve --port 8765
```

```typescript
import { AgentInputLab } from '@agentinputlab/sdk';
const client = new AgentInputLab({ token: process.env.AGENTINPUTLAB_API_TOKEN! });
const inventory = await client.catalog();
const result = await client.intervene({
  release_id: 'plan.plan_and_solve.ps_prompt',
  state: { query: 'Find a suitable flight', budget: { max_model_calls: 8, max_output_tokens: 4096 } },
  seed: 17,
});
console.log(result.state);
```

The bundled HTTP server is authenticated and loopback-only. It is a development bridge, not a public multi-tenant hosting service. Browser CORS is intentionally disabled. Do not expose its token in browser bundles. See [API contract](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/api.md).

## Tests versus real measurements

From the unpacked PyPI source distribution, run the dependency-free consumer
contract audit against the installed package (not against a source-path override):

```sh
python tools/check_consumer_contracts.py
```

It checks every profile's discovery interface, every dataset's setup metadata,
the five command-line entrypoints, Python prompt execution, authenticated HTTP
transport and malformed requests. It reports missing runtime prerequisites
separately; it does not download data, call models or certify task accuracy.
See [feature guarantees and their checks](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/consumer-contracts.md).

The full development test suite below is for a Git checkout; `tests/` is not
included in the Python sdist. The sdist does include `packages/sdk` with its
TypeScript source and protocol tests, so the SDK build instructions above work
without access to the private repository.

```sh
python -m pip install '.[dev]'
pytest
python tools/check_distribution.py dist/*.whl dist/*.tar.gz
python tools/compatibility_inventory.py --output /tmp/agentinputlab-compatibility.json
```

Offline tests exercise algorithm behavior with explicit synthetic fixtures, transport, budgets, typed failures and worker contracts. The compatibility inventory checks every registered method, benchmark and requested backbone identity; it does not run models and labels untested cells as untested. Model names are user-configured, not hardcoded as a supported-model allowlist.

Real acceptance requires licensed data, pinned assets, an actual serving endpoint and the benchmark's native scorer. A patch, a synthetic fixture, a mock HTTP reply or an imported adapter is never an official score. See the staged [validation policy](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/validation.md).

The catalog is not an all-combinations support guarantee. Real acceptance was
run within a declared, owner-approved scope; paused, training-dependent,
environment-blocked and failed configurations remain distinct. See
[coverage boundaries](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/docs/acceptance-coverage.md) before choosing a profile.

## Credits

AgentInputLab toolkit: **Chuang Zhao**. The preferred research citation and
upstream copyright/license notices are separate from software authorship.

## License

MIT for this repository's code. External dependencies retain their own licenses; consult `THIRD_PARTY_NOTICES.md` before downloading or distributing them. Registry publication follows the [acceptance and ownership checks](https://github.com/Data-Designer/AgentInputLab-toolkit/blob/main/PUBLISHING.md).
