"""File binding for the existing native compiler and visible-capture search.

The search contract matches the controlled acceptance connector. It is not a
live Wikipedia tool and never exposes evaluator gold or executes user code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.connectors.tools import ToolConnector, ToolResult, ToolSpec
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable


def validate_runtime(release_id, value):
    from agentinputlab.configuration import _known, _object

    if release_id not in {"plan.llm_compiler.full", "plan.llm_compiler.planner_component"}:
        raise ValidationError("llmcompiler_live is supported only by LLMCompiler full/planner_component")
    _known(value, {"kind", "checkout", "python_executable", "prompt_bundle", "prompt_bundle_sha256",
                   "per_call_output_cap", "timeout_seconds", "argument_order", "tools"}, "runtime")
    for key in ("checkout", "python_executable", "prompt_bundle"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValidationError(f"runtime.{key} must be a non-empty path")
    digest = value.get("prompt_bundle_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValidationError("runtime.prompt_bundle_sha256 must be a SHA-256 digest")
    cap = value.get("per_call_output_cap", 512)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValidationError("runtime.per_call_output_cap must be a positive integer")
    timeout = value.get("timeout_seconds", 400)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValidationError("runtime.timeout_seconds must be positive and finite")
    try:
        valid_timeout = math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ValidationError("runtime.timeout_seconds must be positive and finite")
    order = _object(value.get("argument_order", {}), "runtime.argument_order")
    for name, keys in order.items():
        if (not name or not isinstance(keys, (tuple, list))
                or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValidationError("runtime.argument_order requires unique argument-name arrays")
    if "tools" in value:
        tools = _object(value["tools"], "runtime.tools")
        _known(tools, {"kind", "source_revision"}, "runtime.tools")
        if (tools.get("kind") != "visible_context_search"
                or not isinstance(tools.get("source_revision"), str) or not tools["source_revision"].strip()):
            raise ValidationError("runtime.tools requires visible_context_search and source_revision")
        if order and order != {"search": ("query",)} and order != {"search": ["query"]}:
            raise ValidationError("visible_context_search accepts only the query argument")


class VisibleContextSearch:
    """The original read-only top-five BM25 search over a case's visible blocks."""

    def __init__(self, state, source_revision):
        self.state, self.calls, self.selected = state, [], []
        self.retriever = InMemoryBM25Retriever(tuple(Document(block.block_id, block.text,
            corpus_revision=source_revision) for block in state.context))

    def list_tools(self):
        return (ToolSpec("search", "search(query: string) -> string: search the supplied source-visible document capture; returns up to five passages",
            {"type": "object", "properties": {"query": {"type": "string"}},
             "required": ["query"], "additionalProperties": False}),)

    def fingerprint(self):
        return {"kind": "visible_capture_bm25_search", "capture_sha256": content_hash(self.state.context),
            "retriever": dict(self.retriever.fingerprint()), "top_k": 5,
            "full_corpus_or_live_wikipedia": False}

    def call(self, call):
        if call.name != "search" or set(call.arguments) != {"query"} or not isinstance(call.arguments["query"], str):
            raise ValidationError("visible_context_search requires search(query: string)")
        rows = self.retriever.retrieve(call.arguments["query"], top_k=5)
        self.calls.append({"query": call.arguments["query"], "results": to_jsonable(rows)})
        self.selected.extend(row.document.doc_id for row in rows)
        return ToolResult(call.call_id, output="\n\n".join(f"[{row.document.doc_id}] {row.document.text}" for row in rows))


def bind_case(config, context, state):
    """Bind fresh case tools and the same captured backend; no model call here."""
    from agentinputlab.integrations.external_workers import LLMCompilerLiveWorker

    value = config.runtime
    validate_runtime(config.release_id, value)
    if context.backend is None:
        raise ValidationError("llmcompiler_live requires a configured backend or context.backend")
    if context.external_runtime is not None:
        raise ValidationError("external runtime is configured twice")
    tools = context.tools
    if "tools" in value:
        if tools is not None:
            raise ValidationError("tools are configured twice; choose runtime.tools or context.tools")
        tools = VisibleContextSearch(state, value["tools"]["source_revision"])
    if not isinstance(tools, ToolConnector):
        raise ValidationError("llmcompiler_live requires runtime.tools or context.tools")
    interpreter = Path(value["python_executable"])
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValidationError("LLMCompiler worker Python is missing or not executable; prepare the isolated worker environment")
    if not Path(value["checkout"]).is_dir():
        raise ValidationError("LLMCompiler checkout directory is missing; prepare the pinned upstream source")
    try:
        with Path(value["prompt_bundle"]).open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != value["prompt_bundle_sha256"]:
            raise ArtifactError("LLMCompiler prompt bundle checksum mismatch or oversized bundle")
        bundle = json.loads(raw)
        if (not isinstance(bundle, Mapping) or not isinstance(bundle.get("prompts"), Mapping)
                or content_hash(bundle["prompts"]) != config.parameters.get("prompt_hash")):
            raise ArtifactError("LLMCompiler prompt_hash must match the configured prompt bundle")
    except (OSError, ValueError) as exc:
        raise ArtifactError("cannot read LLMCompiler prompt bundle") from exc
    worker = LLMCompilerLiveWorker(
        python_executable=value["python_executable"], checkout=Path(value["checkout"]),
        prompt_bundle=Path(value["prompt_bundle"]), backend=context.backend, tools=tools,
        per_call_output_cap=value.get("per_call_output_cap", 512),
        timeout_seconds=value.get("timeout_seconds", 400),
        argument_order=dict(value.get("argument_order", {})))
    return replace(context, tools=tools, external_runtime=worker)
