"""Preparation contracts for the existing, exact release profiles.

Discovery describes requirements; it never promotes catalog membership to a
successful model/benchmark test. Native and controlled profiles keep their IDs.
"""
from __future__ import annotations

import inspect
from dataclasses import replace

from agentinputlab.api import describe_intervention
from agentinputlab.registry import AlgorithmRegistry

# These adapters already exist in the package. Preparation calls their public
# file entrypoints rather than reimplementing algorithms or native bindings.
PREPARERS = {
    "worker": ("agentinputlab.worker_preparation", "prepare"),
    "adaptive": ("agentinputlab.adaptive_configuration", "prepare"),
    "qrag": ("agentinputlab.qrag_configuration", "prepare"),
    "longllmlingua": ("agentinputlab.longllmlingua_configuration", "prepare"),
    "recomp": ("agentinputlab.recomp_configuration", "prepare"),
    "hyde": ("agentinputlab.hyde_configuration", "prepare"),
    "query2doc": ("agentinputlab.query2doc_configuration", "prepare"),
    "ircot": ("agentinputlab.ircot_configuration", "prepare"),
}
AUTO_PREPARERS = {
    "rewrite.query2doc.controlled_live": "query2doc",
    "curate.longllmlingua.controlled": "longllmlingua",
    "acquire.hyde.controlled": "hyde",
    "acquire.adaptive_rag.controlled_reimplementation": "adaptive",
    "acquire.qrag.released_checkpoint_native": "qrag",
    "acquire.qrag.reader_transfer_factorial": "qrag",
}
WORKER_KINDS = {
    "reformer_worker": ("reformer", ("pandas",)),
    "infocqr_worker": (None, ()),
    "ask_or_assume_worker": (None, ()),
    "acon_observation_worker": ("acon", ("requests", "jinja2", "tiktoken")),
    "acon_history_worker": ("acon", ("requests", "jinja2", "tiktoken")),
}
# Runtime requirements cannot be inferred from semantic-family discovery tags.
# Keep these tied to exact existing configuration adapters.
_RUNTIME = {
    "rewrite.query2doc.querygym_live": ("querygym_query2doc",),
    "rewrite.query2doc.controlled_live": ("querygym_query2doc",),
    "rewrite.rrr.prompt_external": ("rrr_checkpoint",),
    "rewrite.rrr.warmup_external": ("rrr_checkpoint",),
    "rewrite.rrr.ppo_external": ("rrr_checkpoint",),
    **{name: ("reformer_worker",) for name in (
        "rewrite.reformer.repo_native_r_only", "rewrite.reformer.paper_spec_q_concat_r",
        "rewrite.reformer.controlled_r_only", "rewrite.reformer.controlled_q_concat_r")},
    "rewrite.infocqr.common_support_rw_zsl": ("infocqr_worker",),
    "plan.decomp.full": ("decomp_live",),
    "plan.decomp.planner_component": ("decomp_live",),
    "plan.llm_compiler.full": ("llmcompiler_live",),
    "plan.llm_compiler.planner_component": ("llmcompiler_live",),
    "plan.plan_and_budget.full": ("plan_and_budget_native",),
    "acquire.ircot.native_like_substituted": ("ircot_native",),
    "acquire.adaptive_rag.controlled_reimplementation": ("adaptive_rag_controlled",),
    "acquire.qrag.released_checkpoint_native": ("qrag_native",),
    "acquire.qrag.reader_transfer_factorial": ("qrag_native",),
    "acquire.qrag.common_support": ("qrag_common_support", "qrag_common_support_replay"),
    "acquire.r2ag.common_support": ("r2ag_common_support", "r2ag_common_support_replay"),
    "clarify.ask_or_assume.intent_monitor_component": ("ask_or_assume_worker",),
    "curate.longllmlingua.controlled": ("longllmlingua_native",),
    "curate.recomp.extractive": ("recomp_extractive_worker",),
    "curate.recomp.common_support_extractive": ("recomp_extractive_worker",),
    "curate.recomp.abstractive": ("recomp_abstractive_worker",),
    "curate.acon.history_runtime": ("acon_history_worker",),
    "curate.acon.observation_runtime": ("acon_observation_worker",),
    "curate.acon.common_support_observation": ("acon_observation_worker",),
}
_RETRIEVAL = {
    "rewrite.query2doc.querygym_live", "rewrite.query2doc.controlled_live",
    "rewrite.rrr.common_support_prompt", "rewrite.rrr.prompt_external",
    "rewrite.rrr.warmup_external", "rewrite.rrr.ppo_external",
    "plan.decomp.common_support_odqa", "acquire.ircot.common_support",
    "acquire.qrag.common_support", "acquire.r2ag.common_support",
    "acquire.adaptive_rag.common_support", "acquire.adaptive_rag.controlled_reimplementation",
    "rewrite.reformer.repo_native_r_only", "rewrite.reformer.paper_spec_q_concat_r",
    "rewrite.infocqr.common_support_rw_zsl",
}
_CONTEXT_PREPARATION = {
    "recipe.ace.offline", "recipe.ace.online_transductive",
    "recipe.skillopt.offline", "training.reformer.pattern_induction",
    "training.acon.guideline_optimization", "recipe.mce.sandboxed_offline",
}
_NATIVE_PYTHON = {
    "acquire.r2ag.controlled_same_candidates", "acquire.ircot.controlled",
    "acquire.hyde.official_external_patched", "rewrite.infocqr.t5_student_external",
    "rewrite.infocqr.live_teacher_external",
}


def recipe(release_id):
    info = describe_intervention(release_id)
    kind = info["execution_kind"]
    runtime = _RUNTIME.get(release_id, ())
    if info.get("blocker"):
        route = "unavailable"
    elif kind == "training_recipe":
        route = "context_preparation" if release_id in _CONTEXT_PREPARATION else "weight_training"
    elif release_id in _NATIVE_PYTHON:
        route = "native_python_api"
    else:
        route = "native_arena_or_static_evaluation"
    return {"release_id": release_id, "execution_kind": kind, "route": route,
            "runtime_kinds": list(runtime), "retriever_required": release_id in _RETRIEVAL,
            "vector_runtime_required": release_id.startswith("acquire.hyde."),
            "preparer": AUTO_PREPARERS.get(release_id),
            "required_parameters": [p["name"] for p in info["parameters"] if p["required"]],
            "configuration_is_acceptance_evidence": False}


def list_recipes():
    return [recipe("baseline.no_op"), *[recipe(r.id) for _, r in AlgorithmRegistry.default().list_releases()]]


def complete_configuration(run, *, benchmark_id=None, split=None):
    """Derive only task-format fields and constructor defaults, not identities.

    User-supplied values always win. Demonstrations, learned artifacts, prompts,
    checkpoints and native configuration IDs are never invented.
    """
    parameters = dict(run.parameters)
    changes = []
    if (run.release_id == "rewrite.infocqr.common_support_rw_zsl" or run.release_id.startswith("rewrite.reformer.")):
        # Native worker max_tokens counts INPUT + output. ExecutionBudget's
        # max_output_tokens is a separate, unchanged global output budget.
        controls = dict(parameters.get("controls", {}))
        calls = controls.get("max_model_calls", 2 if "infocqr" in run.release_id else 3)
        ceiling = run.runtime.get("max_input_tokens_per_call", 16384) + run.runtime.get("max_output_tokens", 512)
        if "max_tokens" not in controls:
            controls["max_tokens"] = calls * ceiling
            parameters["controls"] = controls
            changes.append("parameters.controls.max_tokens (input+output guard)")
    if run.release_id == "rewrite.infocqr.common_support_rw_zsl" and benchmark_id:
        from agentinputlab.benchmarks import BenchmarkRegistry
        revision = BenchmarkRegistry.default().get(benchmark_id).source_revision
        for name, value in {"dataset_id": benchmark_id, "dataset_revision": revision,
                            "split": split, "retrieval_corpus_id": benchmark_id + ":visible-context"}.items():
            if name not in parameters and value:
                parameters[name] = value
                changes.append("parameters." + name)
        if not run.retriever and not benchmark_id.startswith(("beir.", "bright.", "trec_dl.")):
            run = replace(run, retriever={"kind": "visible_context_bm25"})
            changes.append("retriever")
    if run.release_id.startswith("rewrite.reformer.") and "pattern_artifact_hash" not in parameters:
        import json
        from importlib.resources import files
        manifest = json.loads(files("workers").joinpath("reformer/compatibility_manifest.json").read_text())
        parameters["pattern_artifact_hash"] = manifest["source_files"]["reformer/patterns/extracted_patterns.json"]
        changes.append("parameters.pattern_artifact_hash")
    if (run.release_id in {"plan.plan_and_budget.full", "plan.plan_and_budget.hard_cap_extension"}
            and "output_format" not in parameters):
        if benchmark_id == "bfcl.v3.simple_python":
            output = "Return the Python function call required by the visible function schema."
        elif benchmark_id and (benchmark_id.startswith(("beir.", "bright.", "trec_dl."))):
            output = 'Return exactly one JSON object with one string field "query".'
        else:
            output = 'Return JSON with answer, support_ids, evidence_ids and evidence_texts, using the visible evidence IDs.'
        parameters["output_format"] = output
        changes.append("parameters.output_format")
    if "split" in recipe(run.release_id)["required_parameters"] and "split" not in parameters and split:
        parameters["split"] = split
        changes.append("parameters.split")
    if run.release_id != "baseline.no_op" and recipe(run.release_id)["route"] != "unavailable":
        factory = AlgorithmRegistry.default().resolve_entrypoint(run.release_id)
        # Required config objects can have their own native no-argument defaults.
        from dataclasses import is_dataclass

        from agentinputlab.configuration import _hints
        from agentinputlab.contracts.serialization import to_jsonable
        for name, parameter in inspect.signature(factory).parameters.items():
            annotation = _hints(factory).get(name)
            if (name not in parameters and name not in run.artifacts and parameter.default is inspect.Parameter.empty
                    and inspect.isclass(annotation) and is_dataclass(annotation)):
                try:
                    parameters[name] = to_jsonable(annotation())
                    changes.append("parameters." + name)
                except (TypeError, ValueError):
                    pass
    return replace(run, parameters=parameters), changes
