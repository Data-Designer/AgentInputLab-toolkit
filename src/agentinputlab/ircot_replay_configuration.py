"""Explicit original input binding for captured IRCoT answers.

This is not a general input override. The projection must reproduce a non-null
state hash in the pinned capture, including the caller's unchanged budget.
"""

from collections.abc import Mapping

from agentinputlab.algorithms.acquire.ircot_replay import IRCoTPredictionReplay
from agentinputlab.contracts import ArtifactError, FrozenAgentState, ValidationError


def validate(value, method, *, benchmark_id, split):
    if (not isinstance(value, Mapping)
            or not isinstance(method, IRCoTPredictionReplay)
            or method.benchmark != benchmark_id or method.split != split):
        raise ValidationError("archived_input requires a matching IRCoT replay scope")
    if value.get("projection") == "ircot_longbench_task.v1":
        marker = value.get("context_marker")
        if (set(value) != {"projection", "context_marker"}
                or not isinstance(marker, str) or not marker.strip() or len(marker) > 4096
                or not benchmark_id.startswith("longbench.") or split != "test"):
            raise ValidationError("ircot_longbench_task.v1 requires a LongBench test scope and original context_marker")
    elif dict(value) != {"projection": "ircot_question_only.v1"}:
        raise ValidationError("archived_input requires an explicit supported IRCoT projection")
    if method.archive.manifest["origin_kind"] != "controlled_capture":
        raise ValidationError("input projection requires a state-bound controlled capture, not an author archive")


def prepare(state, method, *, options=None, adapter=None):
    options = {"projection": "ircot_question_only.v1"} if options is None else options
    validate(options, method, benchmark_id=method.benchmark, split=method.split)
    record = method.archive.records.get(state.task_id)
    query = state.query
    if options["projection"] == "ircot_longbench_task.v1":
        from agentinputlab.benchmarks.longbench import LongBenchAdapter

        if (not isinstance(adapter, LongBenchAdapter) or adapter.benchmark_id != method.benchmark
                or state.messages or state.tool_names):
            raise ArtifactError("LongBench task projection requires its native adapter and a static visible state")
        marker = options["context_marker"]
        query = adapter.template.format(context=marker, input=state.query)
        if state.instruction:
            query += "\n\n" + state.instruction
        if (query.count(marker) != 1
                or query.replace(marker, "\n\n".join(block.text for block in state.context)) != adapter.render(state)):
            raise ArtifactError("LongBench task projection must losslessly reconstruct the native prompt")
    projected = FrozenAgentState(query=query, task_id=state.task_id, budget=state.budget)
    if (record is None or record["query"] != query
            or record["input_state_hash"] is None
            or record["input_state_hash"] != projected.state_hash):
        raise ArtifactError("input projection does not reproduce the original captured state hash")
    return projected, {
        "projection": options["projection"],
        "source_state_sha256": state.state_hash,
        "captured_state_sha256": projected.state_hash,
        "manifest_sha256": method.expected_manifest_sha256,
        "original_state_hash_verified": True,
        "new_model_generation": False,
    }
