# Ask-or-Assume standalone intent component worker

This directory implements only
`clarify.ask_or_assume.intent_monitor_component`. It is a pinned external
worker for the decision and forced-question portion of
[`nedwards99/ask-or-assume`](https://github.com/nedwards99/ask-or-assume/tree/728464a755f15fd448aff61a5cb32dce97fd9c1d).
It is **not** the paper's complete UA-Multi system, an OpenHands runtime, a
coding agent, or a paper-result reproduction.

## Source and license boundary

- Commit: `728464a755f15fd448aff61a5cb32dce97fd9c1d`
- Root license: MIT, SHA-256
  `90bd960a6d24cce8f64f3f2b9be5923d5b33e2650afd6fb683ced4be83e5bb94`
- Excluded source tree: `enterprise/`, which has a separate PolyForm Free
  Trial license

Users mount their own checkout. `compatibility_manifest.json` pins the root
license plus the official IntentAgent class, IntentAgent system prompt,
`ClarifyDecisionTool`, ClarifyAgent controller, `ClarifyTool`, both function
calling paths, and tool-name constants. The driver verifies the exact Git HEAD,
every listed byte hash, a clean checkout, and the absence of `enterprise`
imports before serving. It never imports or reads a source path below
`enterprise/`.

## Why this is a controlled component harness

The official `IntentAgent` constructor requires OpenHands `AgentConfig`,
`LLMRegistry`, `State`, `ConversationMemory`, `Condenser`, controller events,
and runtime wiring. Starting that class faithfully would start the surrounding
OpenHands system and would no longer be a standalone component test.

The worker instead performs the minimum separable path supported by the pinned
source:

1. statically loads the exact official IntentAgent prompt and
   `ClarifyDecisionTool` schema without executing checkout code;
2. forces an `intent` model call to `clarify_decision`;
3. when the decision is true, forces a separate `coding` model call through the
   official `ClarifyTool` schema;
4. renders its user message with the pinned MIT ClarifyAgent function-calling
   control flow;
5. returns the decision, reasons, both tool messages, structured questions, and
   a two-stage trace, including the official `wait_for_response` flag.

The response is labeled `controlled_component_harness`, `component_only=true`,
`openhands_full_system=false`, and `ua_multi_parity_claimed=false`.

## Backend configuration

Serving uses an operator-supplied OpenAI-compatible endpoint. Requests cannot
set an endpoint, API-key environment name, model, prompt, tool schema, or
checkout path. Both role fingerprints must match the fingerprints sent by
`AskOrAssumeIntentMonitor`.

```json
{
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key_env": "OPENAI_API_KEY",
  "timeout_seconds": 60,
  "roles": {
    "intent": {
      "model": "intent-model",
      "fingerprint": {
        "provider": "openai-compatible",
        "model": "intent-model",
        "revision": "deployment-revision"
      },
      "max_output_tokens": 512,
      "input_cost_per_million": 0,
      "output_cost_per_million": 0
    },
    "coding": {
      "model": "coding-model",
      "fingerprint": {
        "provider": "openai-compatible",
        "model": "coding-model",
        "revision": "deployment-revision"
      },
      "max_output_tokens": 1024,
      "input_cost_per_million": 0,
      "output_cost_per_million": 0
    }
  }
}
```

Verify the checkout and extract the official schemas without making a model
call:

```bash
python workers/ask_or_assume/driver.py \
  --checkout /opt/upstreams/ask-or-assume \
  --verify-only
```

Start JSONL service:

```bash
python workers/ask_or_assume/driver.py \
  --checkout /opt/upstreams/ask-or-assume \
  --backend-config-json "$ASK_OR_ASSUME_BACKEND_CONFIG"
```

## Budget semantics

`max_coding_turns` counts completed main-agent coding turns, not raw OpenHands
controller iterations. `max_user_turns` counts clarification interactions.
This avoids the paper setup's roughly three controller iterations per effective
UA-Multi coding turn becoming an accidental compute advantage.

`max_model_calls` is checked before each provider attempt, so a one-call budget
can run the verdict-only path but cannot start the forced-question path.
Provider-reported token and explicitly priced cost totals are checked after a
successful response because this API has no exact preflight token count.

## Remaining parity limits

- No OpenHands environment, coding actions, condenser, simulated user,
  Interactive-SWE/SWE-bench harness, or UA-Multi controller loop is executed.
- The compatibility history is a canonical AgentInputLab state projection,
  not OpenHands' full event-to-message conversion.
- No paper score may be attributed to this release. Full-system claims remain
  attached only to the blocked `clarify.ask_or_assume.openhands_full` release.
