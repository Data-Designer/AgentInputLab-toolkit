"""BFCL file-evaluation binding to the existing native FC/AST implementation."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace

from agentinputlab.backends.llm import GenerationRequest, ModelCapability
from agentinputlab.benchmarks.execution import DownstreamPrediction
from agentinputlab.configuration import _known, _object
from agentinputlab.connectors.tools import SchemaOnlyToolConnector, ToolSpec
from agentinputlab.contracts import ContextBlock
from agentinputlab.contracts.errors import CapabilityError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable


def options(value, *, split, backend):
    if split != "test":
        raise ValidationError("BFCL Python-simple exposes only the test split")
    data = _object(value, "BFCL predictor")
    _known(data, {"schema_version", "function_calling"}, "BFCL predictor")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValidationError("BFCL predictor schema_version must be 1")
    settings = _object(data.get("function_calling", {}), "function_calling")
    _known(settings, {"max_output_tokens", "temperature"}, "function_calling")
    # Reuse the public request's numeric validation without sending a request.
    request = GenerationRequest(prompt="validation", max_tokens=settings.get("max_output_tokens", 512),
                                temperature=settings.get("temperature", 0.0))
    if backend is None or ModelCapability.TOOL_CALLING not in backend.capabilities:
        raise CapabilityError("BFCL evaluation requires a tool-calling backend")
    return {"max_output_tokens": request.max_tokens, "temperature": request.temperature}


def visible_state(adapter, example, budget):
    from agentinputlab.benchmarks.bfcl_v3 import BFCL_V3_OFFICIAL_SOURCE_REVISION

    functions = adapter.function_tools(example)
    connector = SchemaOnlyToolConnector(tuple(ToolSpec(name=t["function"]["name"],
        description=t["function"]["description"], input_schema=t["function"]["parameters"],
        read_only=True) for t in functions))
    state = example.visible_state(budget=budget)
    state = replace(state, tool_names=tuple(t.name for t in connector.list_tools()),
        context=(*state.context, ContextBlock(block_id="native_function_catalog",
            text=json.dumps(to_jsonable(functions), sort_keys=True), source=BFCL_V3_OFFICIAL_SOURCE_REVISION)))
    return state, functions, connector


def predict(state, *, backend, functions, settings, seed):
    if state.budget.max_model_calls < 1 or state.budget.max_output_tokens < 1:
        raise CapabilityError("BFCL function-call generation has no remaining model budget")
    known = {item["function"]["name"] for item in functions}
    if not set(state.tool_names) <= known:
        raise ValidationError("BFCL transformed state introduced an unknown tool")
    selected = tuple(item for item in functions if item["function"]["name"] in state.tool_names)
    request = GenerationRequest(prompt=state.query,
        system="\n\n".join(x for x in [state.instruction, *[b.text for b in state.context]] if x),
        tools=selected, max_tokens=min(settings["max_output_tokens"], state.budget.max_output_tokens),
        temperature=settings["temperature"], seed=seed)
    response = backend.generate(request)
    if response.usage.output_tokens > request.max_tokens:
        raise ValidationError("BFCL provider exceeded the per-call output cap")
    # Native FC decoding/AST validation remains in the source-verified adapter.
    # Do not interpret a free-text answer as an invented tool call.
    try:
        message = response.raw["choices"][0]["message"]
        if not isinstance(message, Mapping):
            raise TypeError
        calls = message.get("tool_calls") or ()
        if not isinstance(calls, (list, tuple)):
            raise TypeError
        prediction = [{call["function"]["name"]: call["function"]["arguments"]} for call in calls]
    except (KeyError, IndexError, TypeError):
        raise ValidationError("BFCL requires an OpenAI-compatible assistant tool_calls response") from None
    return DownstreamPrediction(prediction=prediction, metadata={
        "prediction_kind": "native_bfcl_function_calling", "generated_code_executed": False,
        "official_aggregate_leaderboard_claimed": False})
