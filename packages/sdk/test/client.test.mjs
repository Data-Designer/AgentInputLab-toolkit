import test from 'node:test';
import assert from 'node:assert/strict';
import { AgentInputLab, AgentInputLabError } from '../dist/index.js';
test('wire contract forwards explicit seed and budget without retry', async () => {
  let calls=0;
  const client=new AgentInputLab({token:'test-token',fetch:async(url,opts)=>{
    calls++; assert.equal(url,'http://127.0.0.1:8765/v1/interventions');
    assert.equal(opts.redirect,'error'); assert.equal(opts.headers.Authorization,'Bearer test-token');
    const body=JSON.parse(opts.body); assert.equal(body.seed,17); assert.equal(body.state.budget.max_output_tokens,512);
    return new Response(JSON.stringify({data:{scored:false,benchmark_score:null,state:{query:'test'},
      record:{release_id:'baseline.no_op',input_state_hash:'fixture-in',output_state_hash:'fixture-out',status:'no_op',patch:{},usage:{}}}}));
  }});
  const result=await client.intervene({release_id:'baseline.no_op',state:{query:'test',budget:{max_output_tokens:512}},seed:17});
  assert.equal(result.scored,false); assert.equal(calls,1);
});
test('HTTP failures are typed and never retried',async()=>{
  let calls=0;const c=new AgentInputLab({token:'test-token',fetch:async()=>{calls++;return new Response(JSON.stringify({error:{code:'budget_exhausted',message:'Budget exhausted'}}),{status:422});}});
  await assert.rejects(c.health(),e=>e instanceof AgentInputLabError&&e.status===422&&e.code==='budget_exhausted');assert.equal(calls,1);
});
test('reject insecure remote and credential-bearing origins',()=>{
  for(const baseUrl of ['http://example.com','https://user:pass@example.com','https://example.com/path','https://example.com/?token=bad'])
    assert.throws(()=>new AgentInputLab({token:'test',baseUrl}));
});
test('reject malformed successful responses',async()=>{
  const c=new AgentInputLab({token:'test',fetch:async()=>new Response('{}')});
  await assert.rejects(c.catalog(),e=>e.code==='invalid_response');
});
test('abort request is not retried',async()=>{
  let calls=0;const c=new AgentInputLab({token:'test',timeoutMs:5,fetch:async(_u,o)=>{
    calls++;return new Promise((_r,j)=>o.signal.addEventListener('abort',()=>j(o.signal.reason),{once:true}));
  }});await assert.rejects(c.health());assert.equal(calls,1);
});
