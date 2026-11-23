# Official stateful arena workers

These workers preserve upstream benchmark ownership. AgentInputLab runs the
selected registry intervention on optimizer-visible state first; this worker
package then restores two fresh upstream runtime instances, injects the two
materialized states, continues the official interaction, and calls the
official scorer.

Providers:

- `tau3.py`: pinned τ³/tau2 `banking_knowledge`;
- `statebench.py`: pinned STATE-Bench v0.8.1;
- `toolsandbox.py`: pinned ToolSandbox;
- `appworld_ul.py`: fail-closed official UL plugin boundary; and
- `fixture.py`: deterministic transport test only.

`driver.py` accepts exactly one JSON request on stdin and emits exactly one
JSON response on stdout. Protected replay data are supplied only through
checksum-locked environment bindings. Do not add gold, evaluator state, user
simulator prompts, or credentials to a host-visible payload.

See
[`doc/production_one_click_and_official_arenas.md`](../../doc/production_one_click_and_official_arenas.md)
for capture, multi-model dependency locks, config generation, and the
one-command production workflow.
