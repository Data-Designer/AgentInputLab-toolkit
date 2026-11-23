import test from 'node:test';
import assert from 'node:assert/strict';
import { AgentInputLab } from '../dist/index.js';

test('reject malformed successful endpoint payloads instead of claiming typed success', async () => {
  for (const data of [null, [], 42, {}, {status: 'ok'}, {methods: []}, {scored: true, benchmark_score: 1}]) {
    for (const call of ['health', 'catalog', 'intervene']) {
      const client = new AgentInputLab({token: 'fixture', fetch: async () => new Response(JSON.stringify({data}))});
      const operation = call === 'intervene'
        ? client.intervene({release_id:'baseline.no_op',state:{query:'x'}}) : client[call]();
      await assert.rejects(operation,
        error => error.code === 'invalid_response', `${call}: ${JSON.stringify(data)}`);
    }
  }
});

test('nonfinite request budgets must not be silently serialized as null', async () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    let calls=0;
    const client=new AgentInputLab({token:'fixture',fetch:async()=>{calls++; return new Response('{}');}});
    await assert.rejects(client.intervene({release_id:'baseline.no_op',state:{query:'x',budget:{max_cost_usd:value}}}), TypeError);
    assert.equal(calls,0);
  }
});

test('already aborted requests never dispatch even with a custom transport', async () => {
  const controller=new AbortController(); controller.abort(new Error('cancelled'));
  let calls=0;
  const client=new AgentInputLab({token:'fixture',fetch:async()=>{calls++;return new Response('{}');}});
  await assert.rejects(client.health(controller.signal), /cancelled/);
  assert.equal(calls,0);
});

test('reject invalid token and timer configuration before network use', () => {
  for (const token of [' ', 'abc\txyz', 'abc\x7f', 123])
    assert.throws(()=>new AgentInputLab({token}), TypeError);
  for (const timeoutMs of [0.5, 2147483648])
    assert.throws(()=>new AgentInputLab({token:'fixture',timeoutMs}), TypeError);
});

test('oversized UTF-8 input is rejected without opening a connection', async () => {
  let calls=0;
  const client=new AgentInputLab({token:'fixture',fetch:async()=>{calls++;return new Response('{}');}});
  await assert.rejects(client.intervene({release_id:'baseline.no_op',state:{query:'字'.repeat(350000)}}), /1 MiB/);
  assert.equal(calls,0);
});
