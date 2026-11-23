"""Real four-backbone BFCL simple-Python FC/AST compatibility cases."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from acceptance_method_bindings import EXTERNAL_REWRITERS, bind_external_rewriter
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.reformulate.query2doc import PinnedQueryGymRuntime
from agentinputlab.api import create_intervention
from agentinputlab.backends import Document, GenerationRequest, InMemoryBM25Retriever
from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
from agentinputlab.benchmarks import SourceVerifiedBFCLV3ASTBridge, load_bfcl_v3_simple_adapter
from agentinputlab.connectors.tools import SchemaOnlyToolConnector, ToolSpec
from agentinputlab.contracts import ContextBlock
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.observability import RunMonitor
from agentinputlab.runner import InterventionRunner

RELEASES = (
    "baseline.no_op",
    "plan.plan_and_solve.ps_prompt",
    "plan.plan_and_budget.native_regenerated_soft_budget",
    "plan.llm_compiler.common_support_planner",
    "rewrite.query2doc.querygym_live",
    *EXTERNAL_REWRITERS,
)


def execute(config, output, archive, upstream):
    bridges = {
        m["id"]: SourceVerifiedBFCLV3ASTBridge(source_archive=archive, model_id=m["id"])
        for m in config["models"]
    }
    adapters = {
        m["id"]: load_bfcl_v3_simple_adapter(
            config["prepared_root"], evaluation_bridge=bridges[m["id"]]
        )
        for m in config["models"]
    }
    jobs = []
    for model in config["models"]:
        adapter = adapters[model["id"]]
        for example in adapter.examples("test")[:2]:
            for release in RELEASES:
                job = {
                    "model": model,
                    "benchmark": adapter.benchmark_id,
                    "case_id": example.example_id,
                    "release": release,
                    "seed": 17,
                    "mode": "openai_compatible_function_calling",
                    "budget": {
                        "max_model_calls": 8,
                        "max_output_tokens": 4096,
                        "max_context_tokens": 262144,
                    },
                    "per_call_output_cap": 512,
                    "scorer_binding": bridges[model["id"]].fingerprint(),
                }
                job["id"] = digest(job)
                jobs.append(job)
    for name in ("claims", "terminals"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        {
            "jobs": jobs,
            "automatic_retry": False,
            "paper_results_changed": False,
            "aggregate_leaderboard_claimed": False,
            "scoring_scope": "source_verified_Python_simple_AST",
            "native_model_registry": "actual_model_alias_with_OpenAI_FC_transport_flags",
        },
    )

    def one(job):
        save(output / "claims" / (job["id"] + ".json"), {"job": job, "started": time.time()})
        model = job["model"]
        adapter = adapters[model["id"]]
        bridge = bridges[model["id"]]
        example = adapter.example("test", job["case_id"])
        native_tools = bridge.tools(example.visible_metadata["function"])
        tools = SchemaOnlyToolConnector(
            tuple(
                ToolSpec(
                    name=t["function"]["name"],
                    description=t["function"]["description"],
                    input_schema=t["function"]["parameters"],
                    read_only=True,
                )
                for t in native_tools
            )
        )
        visible = example.visible_state(budget=ExecutionBudget(**job["budget"]))
        schema = json.dumps(to_jsonable(native_tools), sort_keys=True)
        state = replace(
            visible,
            tool_names=tuple(t.name for t in tools.list_tools()),
            context=(
                ContextBlock(
                    text=schema, block_id="native_function_catalog", source=bridge.source_revision
                ),
            ),
        )
        retriever = InMemoryBM25Retriever(
            documents=(
                Document(
                    doc_id="native_function_catalog",
                    text=schema,
                    corpus_revision=bridge.source_revision,
                ),
            )
        )
        source = json.loads((Path(model["service_directory"]) / "started.json").read_text())
        calls = []
        monitor = RunMonitor()
        record = None
        started = time.monotonic()
        stage = "intervention"
        inner = RecordingBackend(
            calls=calls,
            model_id=model["id"],
            base_url=model["base_url"],
            deployment_identity={"revision": digest(source["model_json_hashes"]), "server": "vllm"},
            timeout_seconds=180,
            extra_body=model.get("extra_body", {}),
            transport=local_transport,
        )
        backend = GPTOSSFinalChannelStopBackend(inner) if model["id"] == "gpt-oss-20b" else inner
        result = {
            "job": job,
            "package_location": __import__("agentinputlab").__file__,
            "model_binding": source,
            "benchmark_visible_state_hash": visible.state_hash,
            "runtime_state_hash": state.state_hash,
            "prepared_file_hashes": dict(adapter.prepared.file_sha256),
            "additional_attempts": 1,
            "automatic_retry": False,
        }
        try:
            runtime = None
            kwargs = {}
            if job["release"] in EXTERNAL_REWRITERS:
                backend = inner
                method, runtime = bind_external_rewriter(
                    job["release"],
                    upstream=upstream,
                    backend=backend,
                    model=model,
                    dataset_revision=bridge.source_revision,
                    benchmark=adapter.benchmark_id,
                )
            else:
                if job["release"] == "rewrite.query2doc.querygym_live":
                    runtime = PinnedQueryGymRuntime(upstream / "upstream/QueryGym", backend)
                if job["release"] == "plan.plan_and_budget.native_regenerated_soft_budget":
                    kwargs = {"decomposition_max_tokens": 512, "assessment_max_tokens": 512}
                method = create_intervention(job["release"], **kwargs)
            record = InterventionRunner(monitor=monitor).run(
                method,
                state,
                InterventionContext(
                    backend=backend,
                    tools=tools,
                    retriever=retriever,
                    external_runtime=runtime,
                    seed=17,
                ),
            )
            state = record.patch.apply(state)
            stage = "native_function_call_generation"
            if record.usage.model_calls >= 8 or record.usage.output_tokens >= 4096:
                raise RuntimeError("shared budget exhausted before function call generation")
            prompt = state.query
            additions = [state.instruction, *[b.text for b in state.context]]
            system = "\n\n".join(x for x in additions if x)
            response = backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    system=system,
                    tools=native_tools,
                    max_tokens=min(512, 4096 - record.usage.output_tokens),
                    seed=17,
                )
            )
            stage = "native_fc_decoder_and_ast_scoring"
            raw_calls = response.raw["choices"][0]["message"].get("tool_calls") or ()
            prediction = [{c["function"]["name"]: c["function"]["arguments"]} for c in raw_calls]
            metrics = dict(adapter.score(example, prediction))
            result.update(
                status="scored_terminal",
                metrics=metrics,
                native_function_calls=to_jsonable(prediction),
                finish_reason=response.finish_reason,
                ast_scoring_provenance=to_jsonable(adapter.evaluation_provenance),
            )
        except Exception as exc:
            result.update(
                status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
            )
        result.update(
            seconds=time.monotonic() - started,
            calls=calls,
            monitor_snapshots=to_jsonable(monitor.list_snapshots()),
            intervention=to_jsonable(record) if record else None,
        )
        save(output / "terminals" / (job["id"] + ".json"), result)
        (output.parent / "keep-services-alive").touch()

    # A real baseline exercises the source decoder/checker before large fan-out;
    # it is part of the fixed manifest and is never called a second time.
    probe = jobs[0]
    one(probe)
    observed = json.loads((output / "terminals" / (probe["id"] + ".json")).read_text())
    if observed["status"] != "scored_terminal":
        save(output / "attention.json", {"probe": probe["id"], "others_unstarted": len(jobs) - 1})
        return

    def lane(model):
        with ThreadPoolExecutor(3) as pool:
            list(
                pool.map(
                    one,
                    [j for j in jobs if j["model"]["id"] == model["id"] and j["id"] != probe["id"]],
                )
            )

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lane, config["models"]))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    save(
        output / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True)
    execute(json.loads(a.config.read_text()), a.output, a.source_archive, a.upstream_root)


if __name__ == "__main__":
    main()
