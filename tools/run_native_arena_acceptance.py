"""Real installed-package STATE-Bench / ToolSandbox compatibility cases.

Every episode receives a separate process and fresh native environment. The
user simulator and (where required) native judge are explicitly bound to the
operator-selected local model and their costs stay in the native trace.
"""

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
from run_tooltalk_acceptance import RELEASES


def provider_config(arena, model, environment, checkout):
    if arena in {"tau3", "retail"}:
        return tau_provider_config(arena, model, checkout)
    if arena == "statebench":
        from workers.official_arenas.providers.statebench import StateBenchProvider

        provider = StateBenchProvider()
        source = checkout / "state_bench/scoring.py"
        config = {
            "domains": ["customer_support"],
            "split": "test",
            "limit": 2,
            "agent_provider": "openai",
            "agent_api_key_var": "AIL_TOOLKIT_LOCAL_KEY",
            "score_reasoning_effort": "low",
            "agent_model": model["id"],
            "agent_base_url": model["base_url"],
            "evaluator_model": environment["id"],
            "evaluator_base_url": environment["base_url"],
            "evaluator_api_key_var": "AIL_TOOLKIT_LOCAL_KEY",
        }
    else:
        from workers.official_arenas.providers.toolsandbox import ToolSandboxProvider

        provider = ToolSandboxProvider()
        source = checkout / "tool_sandbox/common/evaluation.py"
        config = {
            "scenario_names": None,
            "limit": 2,
            "preferred_tool_backend": "DEFAULT",
            "agent_model": model["id"],
            "agent_base_url": model["base_url"],
            "agent_key_env": "AIL_TOOLKIT_LOCAL_KEY",
            "user_model": environment["id"],
            "user_base_url": environment["base_url"],
            "user_key_env": "AIL_TOOLKIT_LOCAL_KEY",
        }
    started = json.loads((Path(model["service_directory"]) / "started.json").read_text())
    env_started = json.loads((Path(environment["service_directory"]) / "started.json").read_text())
    config["locked_dependency_hashes"] = {
        "evaluated_model": digest(started["model_json_hashes"]),
        "environment_model": digest(env_started["model_json_hashes"]),
        "official_scorer_source": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    config["budget_override"] = {
        "profile_id": "toolkit-acceptance-4096-v1",
        "max_model_calls": 8,
        "max_retrieval_calls": 8,
        "max_tool_calls": 8,
        "max_output_tokens": 4096,
        "max_context_tokens": 262144,
        "max_output_tokens_per_call": 512,
    }
    return provider, config, started, env_started


def tau_provider_config(arena, model, checkout):
    """Local, backbone-bound native tau roles; no paper configuration reuse."""
    from workers.official_arenas.providers.tau3 import Tau3Provider
    from workers.official_arenas.providers.tau_retail import TauRetailProvider

    provider = TauRetailProvider() if arena == "retail" else Tau3Provider()
    source = json.loads((Path(model["service_directory"]) / "started.json").read_text())
    model_hash = digest(source["model_json_hashes"])
    native_scorer = checkout / "src/tau2/evaluator/evaluator.py"
    scorer_hash = hashlib.sha256(native_scorer.read_bytes()).hexdigest()
    budget = {
        "profile_id": "toolkit-tau-compatibility-4096-v1",
        "max_model_calls": 8,
        "max_retrieval_calls": 8,
        "max_tool_calls": 8,
        "max_output_tokens": 4096,
        "max_context_tokens": 262144,
        "max_output_tokens_per_call": 512,
    }

    def args(tokens):
        return {
            "api_base": model["base_url"],
            "api_key": "toolkit-local-no-auth",
            "temperature": 0,
            "seed": 17,
            "num_retries": 0,
            "max_tokens": tokens,
            "timeout": 180,
        }

    config = {
        "task_set_name": "retail" if arena == "retail" else "banking_knowledge",
        "task_split_name": "test",
        "limit": 2,
        "llm_agent": "openai/" + model["id"],
        "llm_user": "openai/" + model["id"],
        "llm_args_agent": args(512),
        "llm_args_user": args(1024),
        "max_steps": 8,
        "max_errors": 3,
        "timeout": 600,
        "budget_override": budget,
        "locked_dependency_hashes": {
            "agent_model": model_hash,
            "user_simulator": model_hash,
            "nl_assertion_judge": model_hash,
            "official_runtime": digest(provider.source_revision),
            "scorer": scorer_hash,
        },
    }
    if arena == "retail":
        config.update(
            capture_seed=17,
            llm_judge="openai/" + model["id"],
            llm_args_judge={**args(2048), "response_format": {"type": "json_object"}},
        )
    else:
        config.update(retrieval_config="bm25", retrieval_config_kwargs={},
                      capture_max_attempts=1, capture_text_only_fallback=False)
    return provider, config, source, source


def load_captured_checkpoint(job):
    path = Path(job["checkpoint_file"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != job["checkpoint_sha256"]:
        raise ValueError("saved native checkpoint checksum differs")
    value = json.loads(raw)
    if value["checkpoint_id"] != job["checkpoint_id"]:
        raise ValueError("saved native checkpoint ID differs")
    return value


def setup(checkout):
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(checkout / "src"))
    os.environ["AIL_TOOLKIT_LOCAL_KEY"] = "isolated-local-service-key"


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

    setup(checkout)
    model = job["model"]
    release = job["release"]
    calls = []
    record = None
    started = time.monotonic()
    stage = "native_preflight"
    result = {
        "job": job,
        "package_location": __import__("agentinputlab").__file__,
        "additional_attempts": 1,
        "automatic_retry": False,
    }
    try:
        provider, config, source, env_source = provider_config(
            job["arena"], model, job["environment_model"], checkout
        )
        config["limit"] = job.get("capture_limit", 2)
        blockers = provider.doctor(config)
        if blockers:
            raise RuntimeError("; ".join(blockers))
        result.update(
            model_binding=source,
            environment_model_binding=env_source,
            provider_source_revision=provider.source_revision,
            scorer_id=provider.scorer_id,
            scorer_revision=provider.scorer_revision,
        )
        if "checkpoint_file" in job:
            selected = load_captured_checkpoint(job)
            if (
                selected["protected"]["locked_dependency_hashes"]
                != config["locked_dependency_hashes"]
            ):
                raise ValueError("saved native checkpoint model/source locks differ")
            result["capture_source_sha256"] = job["checkpoint_sha256"]
        else:
            selected = next(
                r
                for r in provider.capture(config, output_root=output)
                if r["checkpoint_id"] == job["checkpoint_id"]
            )
        state = _state(dict(selected["visible_state"]))
        original = state.state_hash
        schema_block = next(b for b in state.context if b.block_id == "official_tool_catalog")
        raw = json.loads(schema_block.text)
        schemas = raw["tool_schemas"] if isinstance(raw, dict) else raw
        specs = []
        for schema in schemas:
            function = schema.get("function", schema)
            specs.append(
                ToolSpec(
                    name=function["name"],
                    description=function.get("description", ""),
                    input_schema=function["parameters"],
                    read_only=False,
                )
            )
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
                tools=SchemaOnlyToolConnector(tuple(specs)),
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
        response = provider.run_arm(
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
            metrics=to_jsonable(response.metrics),
            official_trace=to_jsonable(response.trace),
            usage=merge_usage(to_jsonable(record.usage), to_jsonable(response.usage)),
        )
    except Exception as exc:
        import traceback

        result.update(
            status="failed_terminal",
            stage=stage,
            error_type=type(exc).__name__,
            error=str(exc),
            diagnostic_traceback=traceback.format_exc(),
        )
    result.update(
        seconds=time.monotonic() - started,
        optimizer_calls=calls,
        intervention=to_jsonable(record) if record else None,
    )
    save(output / "terminal.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--arena", choices=("statebench", "toolsandbox"), required=True)
    p.add_argument("--job", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--case-offset", type=int, default=0)
    a = p.parse_args()
    checkout = a.checkout.resolve()
    upstream = a.upstream_root.resolve()
    output = a.output.resolve()
    if a.job:
        worker(json.loads(a.job.read_text()), output, checkout, upstream)
        return
    config = json.loads(a.config.read_text())
    setup(checkout)
    environment = config["models"][0]
    provider, native, _, _ = provider_config(a.arena, environment, environment, checkout)
    native["limit"] = a.case_offset + 2
    blockers = provider.doctor(native)
    if blockers:
        raise RuntimeError("; ".join(blockers))
    captures = provider.capture(native, output_root=output)[a.case_offset : a.case_offset + 2]
    if a.preflight_only:
        print(json.dumps({"arena": a.arena, "checkpoints": [r["checkpoint_id"] for r in captures]}))
        return
    output.mkdir(parents=True)
    jobs = []
    for model in config["models"]:
        for row in captures:
            for release in config.get("releases", RELEASES):
                if release not in (*RELEASES, *EXTERNAL_REWRITERS, ASK_OR_ASSUME):
                    raise ValueError("unknown native acceptance release")
                job = {
                    "model": model,
                    "environment_model": environment,
                    "arena": a.arena,
                    "benchmark": provider.benchmark_id,
                    "checkpoint_id": row["checkpoint_id"],
                    "release": release,
                    "seed": 17,
                    "capture_limit": a.case_offset + 2,
                    "ask_or_assume_active_branch": release == ASK_OR_ASSUME,
                    "budget": native["budget_override"],
                    "environment_policy": "explicit_local_qwen3_4b_user_and_native_judge",
                }
                job["id"] = digest(job)
                jobs.append(job)
    save(
        output / "manifest.json",
        {"jobs": jobs, "automatic_retry": False, "paper_results_changed": False},
    )

    def one(job):
        directory = output / job["id"]
        directory.mkdir()
        save(directory / "job.json", job)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        env.pop("PYTHONPATH", None)
        code = None
        with (directory / "worker.log").open("x") as log:
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--config",
                        str(a.config.resolve()),
                        "--output",
                        str(directory),
                        "--checkout",
                        str(checkout),
                        "--upstream-root",
                        str(upstream),
                        "--arena",
                        a.arena,
                        "--job",
                        str(directory / "job.json"),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = "own_worker_timeout_consumption_unknown"
        if not (directory / "terminal.json").exists():
            save(
                directory / "terminal.json",
                {
                    "job": job,
                    "status": "entry_failure_terminal",
                    "returncode": code,
                    "model_calls": None,
                    "automatic_retry": False,
                    "log_sha256": hashlib.sha256(
                        (directory / "worker.log").read_bytes()
                    ).hexdigest(),
                },
            )
        (output.parent / "keep-services-alive").touch()

    # One real native episode catches harness/dependency faults before fan-out.
    # It stays in the same fixed manifest and is not repeated by the pool.
    probe = jobs[0]
    one(probe)
    probe_result = json.loads((output / probe["id"] / "terminal.json").read_text())
    if probe_result.get("status") != "scored_terminal":
        save(
            output / "attention.json",
            {
                "reason": "first native end-to-end probe did not reach scorer",
                "job_id": probe["id"],
                "other_jobs_unstarted": len(jobs) - 1,
                "automatic_retry": False,
            },
        )
        return

    def lane(model):
        with ThreadPoolExecutor(2) as pool:
            list(
                pool.map(
                    one,
                    [j for j in jobs if j["model"]["id"] == model["id"] and j["id"] != probe["id"]],
                )
            )

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lane, config["models"]))
    rows = [json.loads(x.read_text()) for x in output.glob("*/terminal.json")]
    save(
        output / "complete.json",
        {
            "expected": len(jobs),
            "terminal": len(rows),
            "scored": sum(r["status"] == "scored_terminal" for r in rows),
            "full_toolkit_acceptance_complete": False,
        },
    )


if __name__ == "__main__":
    main()
