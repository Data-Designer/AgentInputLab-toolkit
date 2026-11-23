"""Local preflight checks which never generate text or fit method assets."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from agentinputlab.contracts.errors import ValidationError

from .configuration import file_sha
from .recipes import WORKER_KINDS, recipe

_IMPORTS = {
    "plan_and_budget_native": ("numpy",),
    "llmcompiler_live": ("langchain", "langchain_community", "openai", "tiktoken"),
    "decomp_live": ("numpy", "jsonnet", "nltk", "openai", "diskcache"),
    "ircot_native": ("numpy", "jsonnet", "transformers", "nltk", "requests"),
    "qrag_native": ("torch", "transformers", "numpy"),
    "qrag_common_support": ("torch", "transformers", "numpy"),
    "r2ag_common_support": ("torch", "transformers", "numpy"),
    "recomp_abstractive_worker": ("torch", "transformers", "numpy"),
    "recomp_extractive_worker": ("torch", "transformers", "numpy"),
    "longllmlingua_native": ("torch", "llmlingua"),
    "adaptive_rag_controlled": ("torch", "transformers", "numpy"),
    "rrr_checkpoint": ("torch", "transformers"),
    **{kind: value[1] for kind, value in WORKER_KINDS.items()},
}


def interpreter_readiness(python, modules, *, source_paths=()):
    """Probe imports in the selected environment, with no CUDA/model loading."""
    python = str(python)
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValidationError("selected worker Python is missing or not executable")
    code = (
        "import importlib.util,json,sys; "
        f"sys.path[:0]={list(map(str, source_paths))!r}; "
        f"names={tuple(modules)!r}; "
        "missing=[name for name in names if importlib.util.find_spec(name) is None]; "
        "print(json.dumps(dict(python=list(sys.version_info[:3]),missing=missing)))"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    try:
        result = subprocess.run([python, "-I", "-c", code], env=env, text=True,
                                capture_output=True, timeout=30, check=False)
        if result.returncode:
            raise ValidationError("worker environment cannot inspect its installed dependencies")
        observed = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        raise ValidationError("worker dependency probe did not complete") from None
    if observed["missing"]:
        raise ValidationError("missing worker dependencies: " + ", ".join(observed["missing"]))
    return {**observed, "modules_checked": list(modules), "source_paths": list(map(str, source_paths)), "model_calls": 0,
            "verification": "interpreter_and_import_availability"}


def method_readiness(workflow, *, dependencies=True):
    run = workflow.run
    contract = recipe(run.release_id)
    required = contract["runtime_kinds"]
    if required and run.runtime.get("kind") not in required:
        raise ValidationError("this exact profile requires runtime.kind " + " or ".join(required))
    if contract["retriever_required"] and not run.retriever:
        raise ValidationError("method requires retriever configuration; provide its native corpus or visible-context binding")
    if contract["vector_runtime_required"] and not run.vector_runtime:
        raise ValidationError("HyDE requires a dense vector_runtime and its verified corpus/model; sparse BM25 is not a substitute")
    if contract["route"] == "unavailable":
        raise ValidationError("profile has no executable entrypoint")
    if contract["route"] == "native_python_api":
        raise ValidationError("this exact native profile requires its Python runtime/WorkerClient binding; use its native Python API or an explicitly selected file-configured profile")
    if contract["route"] in {"context_preparation", "weight_training"}:
        action = ("use its native fit/online configuration, then bind the resulting artifact to the evaluation profile"
                  if contract["route"] == "context_preparation" else "use its explicit SFT/RL weight-training recipe")
        raise ValidationError(action)
    from agentinputlab.api import describe_intervention
    fields = {value["name"]: value for value in describe_intervention(run.release_id)["parameters"]}
    for name in ("source_license_acknowledged", "asset_license_acknowledged"):
        if name in fields and run.parameters.get(name, fields[name].get("default")) is not True:
            raise ValidationError(f"this exact profile requires parameters.{name}=true after reviewing its source/assets")
    if not dependencies:
        return {"contract": contract, "model_calls": 0}
    # Resolving constructor parameters also verifies artifact contents using
    # the existing loaders at execution. Here validate file identities first.
    for name, reference in run.artifacts.items():
        if isinstance(reference, Mapping) and reference.get("path"):
            path = Path(reference["path"])
            if not path.exists():
                raise ValidationError(f"artifacts.{name}.path is missing")
            digest = reference.get("sha256")
            # A router is a manifest-verified directory, not a scalar file.
            if digest and path.is_file() and file_sha(path) != digest:
                raise ValidationError(f"artifacts.{name} checksum differs")
    if run.runtime:
        modules = _IMPORTS.get(run.runtime.get("kind"), ())
        result = interpreter_readiness(run.runtime.get("python_executable", sys.executable), modules)
    else:
        result = {"model_calls": 0, "modules_checked": []}
    if run.release_id in {"curate.longllmlingua.official", "curate.llmlingua2.xlm_roberta_official",
                          "curate.llmlingua2.bert_multilingual_official"}:
        interpreter_readiness(sys.executable, ("torch", "llmlingua", "transformers"))
    return {**result, "contract": contract}


def local_references(run):
    """Check nested file bindings, including prompts/replays/retriever files."""
    checked = []
    directory_keys = {"checkout", "source_root", "checkpoint_root", "embedder_root", "snapshot",
                      "artifact_root", "base_model_root", "feature_model_root", "tokenizer_root", "nltk_data"}
    path_keys = directory_keys | {"path", "python_executable", "prompt_bundle", "tokenizer_manifest", "corpus_path"}

    def visit(mapping, prefix):
        for name, value in mapping.items():
            location = prefix + "." + name
            if isinstance(value, Mapping):
                visit(value, location)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Mapping):
                        visit(item, f"{location}[{index}]")
            elif isinstance(value, str) and (name in path_keys or name.endswith("_path")):
                target = Path(value)
                if not target.exists():
                    raise ValidationError(f"{location} is missing: {target}")
                if name in directory_keys and not target.is_dir():
                    raise ValidationError(f"{location} must be a directory")
                expected = mapping.get(name.removesuffix("_path") + "_sha256")
                if name == "path":
                    expected = mapping.get("sha256")
                if expected and target.is_file() and file_sha(target) != expected:
                    raise ValidationError(f"{location} checksum differs")
                checked.append(location)
    for name in ("runtime", "retriever", "vector_runtime", "artifacts"):
        visit(getattr(run, name), name)
    return {"checked": checked, "model_calls": 0}


def predictor_readiness(workflow, *, dependencies=True):
    """Use the same task-format parsers as execution before any generation."""
    run, target = workflow.run, workflow.target
    parameters = dict(target.get("predictor_parameters", {"schema_version": 1}))
    backend = run._backend()
    benchmark = target["benchmark_id"]
    is_ir = benchmark.startswith(("beir.", "bright.", "trec_dl."))
    if "trace_ranking" in parameters:
        from agentinputlab.benchmarks.trace_retrieval import validate_trace_ranking

        trace_options = parameters.pop("trace_ranking")
        validate_trace_ranking(trace_options, parameters, benchmark_id=benchmark,
                               release_id=run.release_id, has_retriever=bool(run.retriever))
        if target.get("qrag_reader"):
            raise ValidationError("trace_ranking cannot be combined with a Q-RAG reader")
    if is_ir and dependencies:
        interpreter_readiness(sys.executable, ("ir_measures",))
    if benchmark == "bfcl.v3.simple_python":
        from agentinputlab.bfcl_configuration import options
        options(parameters, split=target["split"], backend=backend)
    elif benchmark.startswith("longbench."):
        from agentinputlab.longbench_configuration import options
        options(parameters, split=target["split"], backend=backend)
        if dependencies:
            interpreter_readiness(sys.executable, ("rouge", "jieba", "fuzzywuzzy"))
    else:
        from agentinputlab.experiments.production_builders import _parse_predictor_parameters
        # These are separately parsed by existing method-specific bindings.
        special = {}
        for name in ("ircot_input", "archived_retrieval", "archived_input", "retriever_capability", "history",
                     "retrieval_prefill", "patch_retrieval", "composite_retrieval"):
            if name in parameters:
                special[name] = parameters.pop(name)
        for name, module in (("patch_retrieval", "agentinputlab.patch_retrieval_configuration"),
                             ("retrieval_prefill", "agentinputlab.retrieval_prefill_configuration")):
            if name in special:
                import importlib
                importlib.import_module(module).options(special[name])
        parsed = _parse_predictor_parameters(parameters)
        task = "retrieval" if is_ir else "qa"
        if parsed["task"] not in {"auto", task}:
            raise ValidationError("predictor task does not match the selected benchmark protocol")
        if is_ir and not run.retriever and "archived_retrieval" not in special:
            raise ValidationError("IR scoring requires an explicit corpus retriever or a separately selected archived-ranking protocol")
        if not is_ir:
            from agentinputlab.backends.llm import ModelCapability, require_capabilities
            required = set(parsed["qa"]["required_capabilities"])
            if parsed["qa"]["use_seed"]:
                required.add(ModelCapability.SEED)
            if backend is not None:
                require_capabilities(backend, required)
    return {"native_predictor_parameters_valid": True, "model_calls": 0}


def budget_and_input_readiness(workflow, adapter, examples):
    run, target = workflow.run, workflow.target
    minimum = 1
    if run.release_id.startswith("acquire.hyde."):
        settings = run.parameters.get("config", run.parameters.get("bindings", {}).get("config", {}))
        minimum += settings.get("num_hypotheses", 8)
    elif run.release_id in {"plan.plan_and_budget.full", "plan.plan_and_budget.hard_cap_extension"}:
        minimum = 1 if run.artifacts.get("records") or run.parameters.get("records") else 3
    elif (run.release_id == "acquire.qrag.common_support"
          and "trace_ranking" in target.get("predictor_parameters", {})):
        minimum = 0  # the frozen local ranker does not require a text-generation call
    elif (run.release_id == "baseline.no_op" and target["benchmark_id"] == "beir.fiqa_2018"
          and run.retriever.get("kind") == "beir_archive"):
        minimum = 0
    if run.budget.max_model_calls < minimum or run.budget.max_output_tokens < minimum:
        raise ValidationError(f"configured method plus downstream scoring needs at least {minimum} model calls/output tokens; budget unchanged")
    if run.runtime.get("kind") in {"infocqr_worker", "reformer_worker"}:
        controls = run.parameters.get("controls", {})
        guard = controls.get("max_tokens", 8192)
        ceiling = run.runtime.get("max_input_tokens_per_call", 16384) + run.runtime.get("max_output_tokens", 512)
        if guard < ceiling:
            raise ValidationError(f"native controls.max_tokens counts input+output and is below the declared per-call ceiling {ceiling}; global output budget is a separate limit")
    if (target["benchmark_id"] == "bfcl.v3.simple_python"
            and run.release_id in {"plan.plan_and_budget.full", "plan.plan_and_budget.hard_cap_extension"}):
        raise ValidationError("native Plan-and-Budget full accepts QA states, while BFCL supplies tool-call states; choose an applicable exact profile")
    return {"minimum_model_calls": minimum, "selected_cases": len(examples), "model_calls": 0}


def native_arena_readiness(native):
    modules = {"tooltalk": ("numpy", "transformers", "openai", "torch"),
               "statebench": ("state_bench",), "toolsandbox": ("tool_sandbox",),
               "tau_banking": ("tau2",), "tau_retail": ("tau2",)}[native.arena]
    # The native executor adds these exact checkout paths before importing its
    # provider. A valid source checkout need not also be installed globally.
    observed = interpreter_readiness(native.worker_python or sys.executable, modules,
                                     source_paths=(Path(native.checkout) / "src", native.checkout))
    if native.worker_python:
        from agentinputlab.arena_configuration import _implementation_digest
        code = "from agentinputlab.arena_configuration import _implementation_digest; print(_implementation_digest())"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        try:
            child = subprocess.run([native.worker_python, "-I", "-c", code], env=env,
                                   capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise ValidationError("native worker toolkit identity probe did not complete") from None
        if child.returncode or child.stdout.strip() != _implementation_digest():
            raise ValidationError("native worker has different toolkit code; prepare an owned environment with the same installed wheel")
    source = {"tooltalk": "src/tooltalk/evaluation/tool_executor.py",
              "statebench": "state_bench/scoring.py", "toolsandbox": "tool_sandbox/common/evaluation.py",
              "tau_banking": "src/tau2/evaluator/evaluator.py", "tau_retail": "src/tau2/evaluator/evaluator.py"}[native.arena]
    if not (Path(native.checkout) / source).is_file():
        raise ValidationError("native scorer/source entrypoint is missing")
    return {**observed, "arena": native.arena, "native_scorer_source_present": True}
