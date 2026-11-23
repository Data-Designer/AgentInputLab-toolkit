"""Explicit paid query reader after an input patch; no hidden model calls."""
import json
import math
from collections.abc import Mapping

from agentinputlab.backends.llm import GenerationRequest
from agentinputlab.benchmarks.ir_query import IR_QUERY_ENDPOINT_ID, parse_ir_query
from agentinputlab.contracts import BudgetExceededError, ValidationError
from agentinputlab.contracts.serialization import content_hash

PROTOCOL = "fixed-source-blind-IR-query-reader-from-complete-patched-state.v1"


def options(value):
    allowed = {"output_format", "max_tokens", "temperature", "allow_json_fence", "context_prefill", "protocol"}
    if (not isinstance(value, Mapping) or set(value) - allowed
            or value.get("output_format") != "json_query"
            or type(value.get("max_tokens", 512)) is not int or value.get("max_tokens", 512) < 1
            or type(value.get("temperature", 0.0)) not in (int, float)
            or not math.isfinite(value.get("temperature", 0.0)) or value.get("temperature", 0.0) < 0
            or type(value.get("allow_json_fence", False)) is not bool):
        raise ValidationError("patch_retrieval requires output_format=json_query, positive max_tokens, "
                              "nonnegative finite temperature, and optional boolean allow_json_fence")
    result = {"max_tokens": value.get("max_tokens", 512), "temperature": value.get("temperature", 0.0),
              "allow_json_fence": value.get("allow_json_fence", False)}
    protocol = value.get("protocol", PROTOCOL)
    if protocol not in (PROTOCOL, IR_QUERY_ENDPOINT_ID):
        raise ValidationError("unknown patch_retrieval protocol; select an explicit registered reader")
    if protocol == IR_QUERY_ENDPOINT_ID:
        if result["temperature"] != 0:
            raise ValidationError("context_ir_query_endpoint.v1 has a fixed temperature of zero")
        result["protocol"] = protocol
    if "context_prefill" in value:
        from agentinputlab.retrieval_prefill_configuration import options as prefill_options

        result["context_prefill"] = prefill_options(value["context_prefill"])
    return result


def reader_prompt(state):
    # This is the frozen acceptance reader, not a task-ID/gold-dependent prompt.
    return ('You are the retrieval-query stage of an information retrieval agent. '
        'Use the original request, task instructions, visible context and any advisory '
        'plan below to produce one useful search query for the indexed corpus. '
        'Advisory plans and context can be mistaken. Do not execute tools, answer the '
        'question, or invent document IDs. Return exactly a JSON object with one '
        'nonempty string field "query".\n\n' + json.dumps(dict(
            original_request=state.query, task_instruction=state.instruction,
            visible_context=[dict(block_id=b.block_id, text=b.text, source=b.source) for b in state.context],
            messages=[dict(m) for m in state.messages]), ensure_ascii=False, sort_keys=True))


def generate_query(state, *, backend, settings, seed):
    """Use an observed backend: even rejected outputs remain captured and charged."""
    if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
        raise BudgetExceededError("patch retrieval reader has no remaining generation budget")
    if state.budget.max_retrieval_calls < 1:
        raise BudgetExceededError("patch retrieval reader has no remaining ranking budget")
    prompt = reader_prompt(state)
    response = backend.generate(GenerationRequest(prompt=prompt,
        max_tokens=min(settings["max_tokens"], state.budget.max_output_tokens),
        temperature=settings["temperature"], seed=seed, response_format={"type": "json_object"}))
    query = parse_ir_query(response.text, allow_json_fence=settings["allow_json_fence"])
    return query, {"protocol": PROTOCOL, "reader_prompt_sha256": content_hash(prompt),
        "reader_response_sha256": content_hash(response.text), "paid_query_reader": True,
        "decoder_policy": "single_json_fence.v1" if settings["allow_json_fence"] else "strict_json_query.v1"}
