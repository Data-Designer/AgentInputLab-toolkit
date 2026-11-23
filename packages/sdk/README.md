# @agentinputlab/sdk

Typed ESM client (Node 20+) for the AgentInputLab Python core. No algorithm forks, Python spawning, hidden retries or fabricated benchmark scores.

This package is not yet published to npm. Run `npm ci && npm test && npm pack` here, then `npm install /path/to/agentinputlab-sdk-0.1.0-alpha.2.tgz` in your application. The package is intentionally private until npm publication is approved.

```typescript
import { AgentInputLab } from '@agentinputlab/sdk';
const client = new AgentInputLab({token: process.env.AGENTINPUTLAB_API_TOKEN!});
console.log(await client.catalog());
const patch = await client.intervene({release_id: 'plan.plan_and_solve.ps_prompt', state: {query: 'Plan this task'}, seed: 17});
console.log(patch.state);
```

Start the Python service separately with `agentinputlab-serve`; configure real dependencies and additional profiles in trusted Python. The default service supplies only NO_OP and the Plan-and-Solve prompt adapter. Model-driven algorithms require an actual backend and often additional assets.

`AgentInputLabError` exposes `status` and `code`. Request timeout defaults to 120 seconds and is configurable. Aborting the client does not cancel an already-running server call; do not automatically retry. Use HTTPS for non-loopback gateways. Never expose server tokens in browser bundles.

Successful responses are validated against the requested endpoint's minimum
contract; empty/wrong-shaped payloads raise `invalid_response`, not typed success.
NaN/Infinity request values are rejected before network dispatch instead of
silently becoming JSON `null`. `timeoutMs` must be an integer from 1 to
2,147,483,647; an already-aborted signal submits no request. Tokens must be
printable ASCII without whitespace. Bodies larger than 1 MiB are rejected before
dispatch. An external gateway must preserve the
`agentinputlab.v1` envelope and result semantics.
