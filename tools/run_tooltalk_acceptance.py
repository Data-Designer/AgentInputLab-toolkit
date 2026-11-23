"""Run installed interventions through fresh official ToolTalk executions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acceptance_method_bindings import (
    ASK_OR_ASSUME,
    EXTERNAL_REWRITERS,
    bind_ask_or_assume,
    bind_external_rewriter,
)
from run_static_acceptance import RecordingBackend, digest, local_transport, save

RELEASES = (
    "baseline.no_op",
    "plan.plan_and_solve.ps_prompt",
    "plan.plan_and_budget.native_regenerated_soft_budget",
    "plan.llm_compiler.common_support_planner",
    "rewrite.query2doc.querygym_live",
    "clarify.udcs.repo_native_prompt_route",
    "clarify.future_turns.future_outcome_match",
    "clarify.active_task_disambiguation.eig_uniform",
)


def worker(job, output, checkout, upstream):
    from agentinputlab import InterventionContext
    from agentinputlab.algorithms.clarify import ActiveDisambiguationControls, FutureTurnControls
    from agentinputlab.algorithms.reformulate.query2doc import PinnedQueryGymRuntime
    from agentinputlab.api import _state, create_intervention
    from agentinputlab.backends import Document, InMemoryBM25Retriever
    from agentinputlab.backends.llm.gpt_oss_final_channel import GPTOSSFinalChannelStopBackend
    from agentinputlab.connectors.tools import SchemaOnlyToolConnector, ToolSpec
    from agentinputlab.contracts.serialization import to_jsonable
    from agentinputlab.runner import InterventionRunner
    from workers.official_arenas.protocol import (
        EXECUTABLE_PAYLOAD_PROTOCOL,
        content_hash,
        merge_usage,
    )
    from workers.official_arenas.providers.tooltalk import ToolTalkProvider, _tool_schemas

    model = job["model"]
    release = job["release"]
    calls = []
    os.chdir(checkout)
    sys.path.insert(0, str(checkout / "src"))
    os.environ["AIL_TOOLKIT_ACCEPTANCE_LOCAL_KEY"] = "local-toolkit-process-key"
    provider = ToolTalkProvider()
    source = json.loads((Path(model["service_directory"]) / "started.json").read_text())
    scorer = checkout / "src/tooltalk/evaluation/tool_executor.py"
    deps = {
        "model_configuration": digest(source["model_json_hashes"]),
        "official_scorer_source": hashlib.sha256(scorer.read_bytes()).hexdigest(),
    }
    budget = {
        "profile_id": "toolkit-acceptance-4096-v1",
        "max_model_calls": 8,
        "max_retrieval_calls": 8,
        "max_tool_calls": 8,
        "max_output_tokens": 4096,
        "max_context_tokens": 262144,
        "max_output_tokens_per_call": 512,
    }
    config = {
        "dataset_parts": ["easy", "tooltalk"],
        "limit": 78,
        "api_mode": "all",
        "agent_model": model["id"],
        "agent_base_url": model["base_url"],
        "agent_key_env": "AIL_TOOLKIT_ACCEPTANCE_LOCAL_KEY",
        "locked_dependency_hashes": deps,
        "budget_override": budget,
    }
    result = {
        "job": job,
        "additional_attempts": 1,
        "automatic_retry": False,
        "package_location": __import__("agentinputlab").__file__,
        "model_binding": source,
        "provider_source_revision": provider.source_revision,
        "scorer_id": provider.scorer_id,
        "scorer_revision": provider.scorer_revision,
    }
    started = time.monotonic()
    stage = "native_provider_preflight"
    record = None
    try:
        blockers = provider.doctor(config)
        if blockers:
            raise RuntimeError("; ".join(blockers))
        rows = provider.capture(config, output_root=output)
        selected = next(r for r in rows if r["checkpoint_id"] == job["checkpoint_id"])
        state = _state(dict(selected["visible_state"]))
        original = state.state_hash
        inner = RecordingBackend(
            calls=calls,
            model_id=model["id"],
            base_url=model["base_url"],
            deployment_identity={"revision": deps["model_configuration"], "server": "vllm"},
            timeout_seconds=180,
            extra_body=model.get("extra_body", {}),
            transport=local_transport,
        )
        backend = GPTOSSFinalChannelStopBackend(inner) if model["id"] == "gpt-oss-20b" else inner
        specs = tuple(
            ToolSpec(
                name=s["function"]["name"],
                description=s["function"]["description"],
                input_schema=s["function"]["parameters"],
                read_only=False,
            )
            for s in _tool_schemas()
        )
        kwargs = {}
        runtime = None
        if release == "plan.plan_and_budget.native_regenerated_soft_budget":
            kwargs = {"decomposition_max_tokens": 512, "assessment_max_tokens": 512}
        elif release == "clarify.udcs.repo_native_prompt_route":
            kwargs = {"max_tokens": 512}
        elif release == "clarify.future_turns.future_outcome_match":
            kwargs = {"controls": FutureTurnControls(max_tokens=512)}
        elif release == "clarify.active_task_disambiguation.eig_uniform":
            kwargs = {"controls": ActiveDisambiguationControls(max_tokens=512)}
        elif release == "rewrite.query2doc.querygym_live":
            runtime = PinnedQueryGymRuntime(upstream / "upstream/QueryGym", backend)
        stage = "intervention"
        retriever = InMemoryBM25Retriever(
            documents=tuple(
                Document(
                    doc_id=b.block_id or str(i),
                    text=b.text,
                    corpus_revision=provider.source_revision,
                )
                for i, b in enumerate(state.context)
            )
        )
        if release in EXTERNAL_REWRITERS:
            backend = inner
            method, runtime = bind_external_rewriter(
                release,
                upstream=upstream,
                backend=backend,
                model=model,
                dataset_revision=provider.source_revision,
                benchmark=provider.benchmark_id,
            )
        elif release == ASK_OR_ASSUME:
            backend = inner
            method, runtime = bind_ask_or_assume(upstream=upstream, backend=backend, model=model)
        else:
            method = create_intervention(release, **kwargs)
        record = InterventionRunner().run(
            method,
            state,
            InterventionContext(
                backend=backend,
                seed=17,
                tools=SchemaOnlyToolConnector(specs),
                retriever=retriever,
                external_runtime=runtime,
            ),
        )
        materialized = to_jsonable(record.patch.apply(state))
        payload = {
            "schema_version": 1,
            "protocol": EXECUTABLE_PAYLOAD_PROTOCOL,
            "visible_state_hash": original,
            "release_id": release,
            "intervention_execution": {
                "input_state_hash": original,
                "output_state_hash": content_hash(materialized),
                "materialized_state": materialized,
                "usage": to_jsonable(record.usage),
            },
        }
        stage = "native_episode_and_official_scorer"
        downstream = provider.run_arm(
            config,
            replay_record={
                "checkpoint_id": selected["checkpoint_id"],
                "protected": selected["protected"],
            },
            arm_payload=payload,
            seed=17,
            role="baseline" if release == "baseline.no_op" else "treatment",
            work_dir=output,
        )
        result.update(
            status="scored_terminal",
            metrics=to_jsonable(downstream.metrics),
            official_trace=to_jsonable(downstream.trace),
            usage=merge_usage(to_jsonable(record.usage), to_jsonable(downstream.usage)),
        )
    except Exception as exc:
        result.update(
            status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
        )
    result.update(
        seconds=time.monotonic() - started,
        optimizer_calls=calls,
        intervention=to_jsonable(record) if record else None,
    )
    save(output / "terminal.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    upstream = args.upstream_root.resolve()
    if args.job:
        worker(json.loads(args.job.read_text()), args.output.resolve(), checkout, upstream)
        return
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True)
    cases = []
    for part in ("easy", "tooltalk"):
        first = sorted((checkout / "data" / part).glob("*.json"))[config.get("case_offset", 0)]
        row = json.loads(first.read_bytes())
        cases.append(
            {
                "checkpoint_id": part + ":" + str(row["conversation_id"]),
                "source_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            }
        )
    jobs = []
    for model in config["models"]:
        for case in cases:
            for release in config.get("releases", RELEASES):
                if release not in (*RELEASES, *EXTERNAL_REWRITERS, ASK_OR_ASSUME):
                    raise ValueError("unknown native acceptance release")
                job = {
                    "model": model,
                    "checkpoint_id": case["checkpoint_id"],
                    "case_source_sha256": case["source_sha256"],
                    "benchmark": "tooltalk.full",
                    "release": release,
                    "seed": 17,
                    "budget": {"model_calls": 8, "output_tokens": 4096, "per_call": 512},
                    "ask_or_assume_active_branch": release == ASK_OR_ASSUME,
                }
                job["id"] = digest(job)
                jobs.append(job)
    save(
        output / "manifest.json",
        {"jobs": jobs, "paper_results_changed": False, "automatic_retry": False},
    )

    def one(job):
        directory = output / job["id"]
        directory.mkdir()
        save(directory / "job.json", job)
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        for name, value in config.get("runtime_environment", {}).items():
            if name not in {"HF_HOME", "HF_HUB_CACHE"}:
                raise ValueError("unsupported runtime environment setting")
            environment[name] = value
        environment.pop("PYTHONPATH", None)
        with (directory / "worker.log").open("x") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(args.config.resolve()),
                    "--output",
                    str(directory),
                    "--checkout",
                    str(checkout),
                    "--upstream-root",
                    str(upstream),
                    "--job",
                    str(directory / "job.json"),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        if not (directory / "terminal.json").exists():
            save(
                directory / "terminal.json",
                {
                    "job": job,
                    "status": "entry_failure_terminal",
                    "returncode": process.returncode,
                    "model_calls": None,
                    "automatic_retry": False,
                    "log_sha256": hashlib.sha256(
                        (directory / "worker.log").read_bytes()
                    ).hexdigest(),
                },
            )
        (output.parent / "keep-services-alive").touch()

    def per_model(model):
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(one, [j for j in jobs if j["model"]["id"] == model["id"]]))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(per_model, config["models"]))
    terms = [json.loads(p.read_text()) for p in output.glob("*/terminal.json")]
    save(
        output / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(terms),
            "scored": sum(r["status"] == "scored_terminal" for r in terms),
            "full_toolkit_acceptance_complete": False,
        },
    )


if __name__ == "__main__":
    main()
