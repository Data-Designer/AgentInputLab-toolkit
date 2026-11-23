"""Tau-specific public execution: explicit roles, selected captures, no retries."""
from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import urlsplit

from agentinputlab.contracts import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

PARAMETERS = {
    "tau_banking": {"split": "train", "limit": 1000, "max_steps": 16, "max_errors": 3,
                    "timeout_seconds": 600, "user_max_tokens": 1024,
                    "retrieval_config": "bm25", "retrieval_config_kwargs": {},
                    "agent_extra_body": None},
    "tau_retail": {"split": "test", "limit": 40, "max_steps": 16, "max_errors": 3,
                   "timeout_seconds": 600, "user_max_tokens": 1024, "judge_max_tokens": 512,
                   "agent_extra_body": None},
}


def provider_config(native, run, override):
    from agentinputlab.arena_configuration import _provider, _sha

    model, environment = run.backend, native.environment_backend
    if (model["model_id"] != environment["model_id"]
            or model["deployment_identity"] != environment["deployment_identity"]):
        raise ValidationError("Tau user and judge must use the declared evaluated backbone and identity")
    for backend in (model, environment):
        if urlsplit(backend["base_url"]).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValidationError("This Tau profile requires explicit local endpoints; no remote judge fallback")
    params = {**PARAMETERS[native.arena], **to_jsonable(native.parameters)}
    for key in ("limit", "max_steps", "max_errors", "user_max_tokens", "judge_max_tokens"):
        if key in params and (type(params[key]) is not int or params[key] < 1):
            raise ValidationError(f"Tau {key} must be a positive integer")
    timeout = params["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValidationError("Tau timeout_seconds must be finite and positive")
    if params["split"] not in {"train", "test"}:
        raise ValidationError("Tau split must be train or test")
    if native.arena == "tau_retail" and run.seed != 17:
        raise ValidationError("The native Retail profile requires seed 17")
    provider = _provider(native)
    identity = content_hash(model["deployment_identity"])
    locks = {key: identity for key in ("agent_model", "user_simulator", "nl_assertion_judge")}
    locks.update(official_runtime=content_hash({"source_revision": provider.source_revision}),
                 scorer=_sha(Path(native.checkout) / "src/tau2/evaluator/evaluator.py"))

    def arguments(backend, cap):
        value = {"api_base": model["base_url"], "api_key": "agentinputlab-native-role-routing",
                 "seed": run.seed, "temperature": 0, "num_retries": 0, "max_tokens": cap,
                 "timeout": backend.get("timeout_seconds", timeout)}
        if "extra_body" in backend:
            from workers.official_arenas.protocol import native_extra_body
            value["extra_body"] = native_extra_body(backend, "extra_body")
        return value

    # Input optimizers and the downstream agent can use different explicit
    # transport extensions on the same model. Omission retains legacy inheritance;
    # {} explicitly removes helper-only extensions from native agent calls.
    agent_arguments = arguments(model, native.max_output_tokens_per_call)
    if params["agent_extra_body"] is not None:
        from workers.official_arenas.protocol import native_extra_body
        extensions = native_extra_body(params, "agent_extra_body")
        agent_arguments.pop("extra_body", None)
        if extensions:
            agent_arguments["extra_body"] = extensions

    # LiteLLM uses an explicit provider prefix; wire model IDs stay unprefixed.
    model_id = model["model_id"]
    if model_id.startswith("openai/"):
        raise ValidationError("Tau backend.model_id must be the wire model ID, without openai/ prefix")
    config = dict(task_set_name="retail" if native.arena == "tau_retail" else "banking_knowledge",
                  task_split_name=params["split"], limit=params["limit"],
                  llm_agent="openai/" + model_id, llm_user="openai/" + model_id,
                  llm_args_agent=agent_arguments,
                  llm_args_user=arguments(environment, params["user_max_tokens"]),
                  max_steps=params["max_steps"], max_errors=params["max_errors"], timeout=timeout,
                  budget_override=override, locked_dependency_hashes=locks)
    if native.arena == "tau_retail":
        config.update(capture_seed=17, llm_judge="openai/" + model_id,
                      llm_args_judge={**arguments(environment, params["judge_max_tokens"]),
                                      "response_format": {"type": "json_object"}})
    else:
        config.update(retrieval_config=params["retrieval_config"],
                      retrieval_config_kwargs=params["retrieval_config_kwargs"],
                      capture_max_attempts=1, capture_text_only_fallback=False)
    return dict(provider._validate_config(config))


def role_calls(native, run, config, work, *, initial_capture=False):
    # Reuse the scoped role-aware HTTP accounting used by complete native Tau
    # episodes. It does not run a fitter or change request bodies/scoring.
    from agentinputlab.recipes.skillopt_tau import _account_calls
    return _account_calls(work, {"backend": run.backend,
                                "environment_backend": native.environment_backend,
                                "seed": 0 if initial_capture and native.arena == "tau_banking" else run.seed}, config)


def capture(native, run, provider, config, job, work):
    from tau2.runner.helpers import get_tasks

    from agentinputlab.arena_configuration import _save, _sha
    from agentinputlab.observability.monitor import _safe_error_message

    tasks = get_tasks(config["task_set_name"], task_split_name=config["task_split_name"],
                      num_tasks=config["limit"])
    available = [str(task.id) for task in tasks]
    if not available or len(set(available)) != len(available):
        raise ValidationError("Tau inventory is empty or has duplicate IDs")
    if job.get("inventory_only"):
        # Loading task IDs does not simulate a user or instantiate an agent.
        _save(work / "terminal.json", {"status": "captured", "case_ids": available,
                                      "inventory_kind": "task_ids_only", "model_calls": 0})
        return 0
    chosen = job.get("case_ids") or available[:job["limit"]]
    if not chosen or set(chosen) - set(available):
        raise ValidationError("requested Tau cases are absent from the selected native split")
    if native.capture_bundle is not None:
        from agentinputlab.tau_capture_bundle import reuse
        return reuse(native, config, chosen, available, work)
    rows, failures, configurations, cases = [], {}, {}, {}
    for i, case_id in enumerate(chosen):
        output = work / f"capture-{i:04d}"
        output.mkdir()
        selected_config = {**config, "limit": 1, "capture_task_ids": [case_id]}
        configurations[case_id] = selected_config
        receipt = {"case_id": case_id, "status": "started", "automatic_retry": False,
                   "outside_agent_budget": True}
        _save(output / "started.json", receipt)
        events = []
        try:
            with role_calls(native, run, selected_config, output, initial_capture=True) as events:
                captured = provider.capture(selected_config, output_root=output)
            if len(captured) != 1 or captured[0]["checkpoint_id"] != case_id:
                raise ValidationError("Tau capture returned an unexpected case")
            rows.append(captured[0])
            receipt.update(status="captured", capture_seed=captured[0]["protected"]["capture_seed"])
        except Exception as exc:
            receipt.update(status="failed", stage="initial_user_capture", error_type=type(exc).__name__,
                           error=_safe_error_message(exc))
            failures[case_id] = {**receipt, "receipt": str(output / "terminal.json")}
        finally:
            receipt["role_model_calls"] = to_jsonable(events)
            _save(output / "terminal.json", receipt)
            cases[case_id] = {"status": receipt["status"], "receipt": {
                "path": str((output / "terminal.json").relative_to(work)),
                "sha256": _sha(output / "terminal.json")}}
    _save(work / "inventory.json", rows)
    _save(work / "native-configurations.json", configurations)
    _save(work / "capture-bundle.json", {
        "schema": "agentinputlab.tau.capture_bundle.v1", "arena": native.arena,
        "provider_configuration": config, "cases": cases,
        "target_model": {"model_id": run.backend["model_id"],
                         "deployment_identity": run.backend["deployment_identity"]},
        "inventory": {"path": "inventory.json", "sha256": _sha(work / "inventory.json")},
        "native_configurations": {"path": "native-configurations.json",
                                  "sha256": _sha(work / "native-configurations.json")},
        "capture_cost_scope": "original_capture_once_outside_agent_budget",
    })
    _save(work / "terminal.json", {"status": "captured", "case_ids": available,
          "capture_failures": failures, "captured_case_ids": [r["checkpoint_id"] for r in rows],
          "capture_bundle": {"path": str(work / "capture-bundle.json"),
                             "sha256": _sha(work / "capture-bundle.json")},
          "inventory_sha256": _sha(work / "inventory.json"), "automatic_retry": False})
    return 0
