"""Configured single-arm native arena execution in fresh worker processes.

Protected task records stay with the native provider, never in method context.
This is not a paired experiment, an automatic retry loop, or a paper exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agentinputlab.configuration import RunConfig, _json_mapping, _known, _unique
from agentinputlab.contracts import InterventionContext
from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.observability.monitor import _safe_error_message
from agentinputlab.tau_arena_configuration import PARAMETERS as _TAU_PARAMETERS

_PROVIDERS = {
    "tooltalk": ("tooltalk", "ToolTalkProvider", "src/tooltalk/evaluation/tool_executor.py"),
    "statebench": ("statebench", "StateBenchProvider", "state_bench/scoring.py"),
    "toolsandbox": ("toolsandbox", "ToolSandboxProvider", "tool_sandbox/common/evaluation.py"),
    "tau_banking": ("tau3", "Tau3Provider", "src/tau2/evaluator/evaluator.py"),
    "tau_retail": ("tau_retail", "TauRetailProvider", "src/tau2/evaluator/evaluator.py"),
}
_PARAMETERS = {
    "tooltalk": {"limit": 78},
    "statebench": {"domains": ["customer_support"], "split": "test", "limit": 1000,
                   "score_reasoning_effort": "low"},
    "toolsandbox": {"scenario_names": None, "limit": 1000, "preferred_tool_backend": "DEFAULT"},
    **_TAU_PARAMETERS,
}


def _read(path: Path, *, limit: int = 1024 * 1024) -> Any:
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValidationError("arena configuration exceeds its size limit")
        return json.loads(raw, object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValidationError("cannot read arena JSON file") from None


def _save(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(to_jsonable(value), stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeArenaConfig:
    arena: str
    checkout: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    environment_backend: Mapping[str, Any] = field(default_factory=dict)
    max_output_tokens_per_call: int = 512
    worker_python: str | None = None
    scorer_cache: str | None = None
    capture_bundle: Mapping[str, str] | None = None
    history: Mapping[str, str] | None = None
    prior_intervention_usage: Mapping[str, str] | None = None
    agent_extra_body_mode: str = "merge"
    environment_extra_body_mode: str = "merge"
    agent_responses_reasoning_disabled: bool = False

    def __post_init__(self):
        if not isinstance(self.arena, str) or self.arena not in _PROVIDERS:
            raise ValidationError("arena must be tooltalk, statebench, toolsandbox, tau_banking, or tau_retail")
        if type(self.agent_responses_reasoning_disabled) is not bool:
            raise ValidationError("agent_responses_reasoning_disabled must be boolean")
        if self.agent_responses_reasoning_disabled and self.arena != "statebench":
            raise ValidationError("agent_responses_reasoning_disabled is only supported for STATE-Bench")
        for name in ("agent_extra_body_mode", "environment_extra_body_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in {"merge", "replace"}:
                raise ValidationError(f"{name} must be merge or replace")
            if value != "merge" and (self.arena in _TAU_PARAMETERS or
                    (name == "environment_extra_body_mode" and self.arena == "tooltalk")):
                raise ValidationError(f"{name} is unsupported for this native arena role")
        if self.prior_intervention_usage is not None:
            from agentinputlab.fit_configuration import _reference
            object.__setattr__(self, "prior_intervention_usage",
                               _reference(self.prior_intervention_usage, "prior_intervention_usage"))
            if self.history is not None:
                raise ValidationError("prior_intervention_usage cannot be combined with live history")
        if not isinstance(self.checkout, str) or not self.checkout.strip():
            raise ValidationError("arena checkout must be an explicit native source directory")
        object.__setattr__(self, "checkout", str(Path(self.checkout).expanduser().absolute()))
        if self.history is not None:
            history = _json_mapping(self.history, "history")
            _known(history, {"kind", "prefix_policy"}, "history")
            if history.get("kind") != "acon":
                raise ValidationError("history.kind must be acon")
            policy = history.get("prefix_policy", "strict_system_user")
            if not isinstance(policy, str) or policy not in {
                    "strict_system_user", "preserve_pre_user_assistant"}:
                raise ValidationError("history.prefix_policy is unsupported")
            object.__setattr__(self, "history", history)
        if self.capture_bundle is not None:
            from agentinputlab.fit_configuration import _reference
            if self.arena not in _TAU_PARAMETERS:
                raise ValidationError("capture_bundle currently applies only to Tau")
            pin = _reference(self.capture_bundle, "capture_bundle")
            object.__setattr__(self, "capture_bundle", pin)
        if self.scorer_cache is not None:
            if self.arena != "tooltalk":
                raise ValidationError("scorer_cache currently applies only to ToolTalk")
            if not isinstance(self.scorer_cache, str) or not self.scorer_cache.strip():
                raise ValidationError("scorer_cache must be an explicit Hugging Face hub cache directory")
            object.__setattr__(self, "scorer_cache", str(Path(self.scorer_cache).expanduser().absolute()))
        if self.worker_python is not None:
            if not isinstance(self.worker_python, str) or not self.worker_python.strip():
                raise ValidationError("worker_python must be an explicit Python executable path")
            object.__setattr__(self, "worker_python", str(Path(self.worker_python).expanduser().absolute()))
        parameters = _json_mapping(self.parameters, "arena parameters")
        _known(parameters, set(_PARAMETERS[self.arena]), "arena parameters")
        environment = _json_mapping(self.environment_backend, "environment_backend")
        _known(environment, {"model_id", "base_url", "api_key_env", "deployment_identity", "timeout_seconds", "extra_body"},
               "environment_backend (use api_key_env, never a plaintext key)")
        if self.arena == "tooltalk" and environment:
            raise ValidationError("ToolTalk has no separate environment model")
        if self.arena != "tooltalk":
            if not environment:
                raise ValidationError("STATE/ToolSandbox/Tau require an explicit environment_backend")
            RunConfig("baseline.no_op", backend=environment)._backend()
        cap = self.max_output_tokens_per_call
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValidationError("max_output_tokens_per_call must be a positive integer")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "environment_backend", environment)

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path).absolute()
        value = _read(path)
        if not isinstance(value, dict):
            raise ValidationError("arena configuration must be an object")
        _known(value, {"arena", "checkout", "parameters", "environment_backend", "max_output_tokens_per_call", "worker_python", "scorer_cache", "capture_bundle", "history", "prior_intervention_usage", "agent_extra_body_mode", "environment_extra_body_mode", "agent_responses_reasoning_disabled"},
               "arena configuration")
        if value.get("prior_intervention_usage") is not None:
            from agentinputlab.fit_configuration import _reference
            pin = _reference(value["prior_intervention_usage"], "prior_intervention_usage")
            source = Path(pin["path"]).expanduser()
            value["prior_intervention_usage"] = {**pin, "path": str(source if source.is_absolute() else path.parent / source)}
        if value.get("capture_bundle") is not None:
            from agentinputlab.fit_configuration import _reference
            pin = _reference(value["capture_bundle"], "capture_bundle")
            source = Path(pin["path"]).expanduser()
            value["capture_bundle"] = {**pin, "path": str(source if source.is_absolute() else path.parent / source)}
        if isinstance(value.get("scorer_cache"), str):
            if not value["scorer_cache"].strip():
                raise ValidationError("scorer_cache must be an explicit cache directory")
            cache = Path(value["scorer_cache"]).expanduser()
            value["scorer_cache"] = str(cache if cache.is_absolute() else path.parent / cache)
        if isinstance(value.get("checkout"), str):
            root = Path(value["checkout"]).expanduser()
            value["checkout"] = str(root if root.is_absolute() else path.parent / root)
        if isinstance(value.get("worker_python"), str):
            if not value["worker_python"].strip():
                raise ValidationError("worker_python must be an explicit Python executable path")
            executable = Path(value["worker_python"]).expanduser()
            value["worker_python"] = str(executable if executable.is_absolute() else path.parent / executable)
        try:
            return cls(**value)
        except TypeError:
            raise ValidationError("arena configuration requires arena and checkout") from None

    def provider_config(self, run: RunConfig) -> dict[str, Any]:
        """Bind transport and one shared budget; do not infer scientific settings."""
        self.history_policy(run)
        run.validate()
        if self.prior_intervention_usage is not None:
            from agentinputlab.arena_prior_usage import load
            load(self.prior_intervention_usage, run)
        if run.retriever.get("kind") == "visible_context_bm25":
            raise ValidationError(
                "visible_context_bm25 supports run --state and static evaluate only; "
                "native arenas require an explicit arena-compatible retriever")
        if not run.backend:
            raise ValidationError("native arena execution requires backend configuration")
        # _run_case builds the intervention backend via RunConfig._backend,
        # including its explicit stop policy. Native continuation calls have
        # no logical stop strings; retain their own tool/JSON decoding.
        budget = run.budget
        if budget.max_context_tokens is None or budget.max_cost_usd is not None:
            raise ValidationError("native arenas require explicit max_context_tokens and do not support max_cost_usd")
        override = {"profile_id": "configured-native-arena.v1", **to_jsonable(budget),
                    "max_output_tokens_per_call": self.max_output_tokens_per_call}
        override.pop("max_cost_usd")
        from workers.official_arenas.protocol import native_extra_body, validate_shared_budget
        validate_shared_budget(override)
        model = run.backend
        for item in (model, self.environment_backend):
            if item:
                key = item.get("api_key_env", "OPENAI_API_KEY")
                if not isinstance(key, str) or not key.isidentifier():
                    raise ValidationError("native api_key_env must be an environment variable name")
        if self.arena in _TAU_PARAMETERS:
            from agentinputlab.tau_arena_configuration import provider_config
            return provider_config(self, run, override)
        params = {**_PARAMETERS[self.arena], **to_jsonable(self.parameters)}
        config = {**params, "agent_model": model["model_id"], "agent_base_url": model["base_url"],
                  "budget_override": override,
                  "locked_dependency_hashes": {
                      "evaluated_model": content_hash(model["deployment_identity"]),
                      "official_scorer_source": _sha(Path(self.checkout) / _PROVIDERS[self.arena][2]),
                  }}
        key = model.get("api_key_env", "OPENAI_API_KEY")
        if "timeout_seconds" in model:
            config["agent_timeout_seconds"] = model["timeout_seconds"]
        if "timeout_seconds" in self.environment_backend:
            config["environment_timeout_seconds"] = self.environment_backend["timeout_seconds"]
        for prefix, backend in (("agent", model), ("environment", self.environment_backend)):
            if "extra_body" in backend:
                config[prefix + "_extra_body"] = native_extra_body(backend, "extra_body")
            mode = getattr(self, prefix + "_extra_body_mode")
            if mode != "merge":
                config[prefix + "_extra_body_mode"] = mode
        if self.arena == "tooltalk":
            config.update(dataset_parts=["easy", "tooltalk"], api_mode="all", agent_key_env=key)
        else:
            environment = self.environment_backend
            config["locked_dependency_hashes"]["environment_model"] = content_hash(environment["deployment_identity"])
            if self.arena == "statebench":
                config.update(agent_provider="openai", agent_api_key_var=key, request_attempts=1,
                    evaluator_model=environment["model_id"], evaluator_base_url=environment["base_url"],
                    evaluator_api_key_var=environment.get("api_key_env", "OPENAI_API_KEY"))
                if self.agent_responses_reasoning_disabled:
                    config["agent_responses_reasoning_disabled"] = True
            else:
                config.update(agent_key_env=key, user_model=environment["model_id"],
                    user_base_url=environment["base_url"], user_key_env=environment.get("api_key_env", "OPENAI_API_KEY"))
        return config

    def history_policy(self, run: RunConfig) -> str | None:
        if self.history is None:
            return None
        if run.release_id != "curate.acon.history_runtime":
            raise ValidationError("live history requires curate.acon.history_runtime, not an initial patch profile")
        return self.history.get("prefix_policy", "strict_system_user")


def _provider(native: NativeArenaConfig, *, history_transform=None):
    module, factory, _ = _PROVIDERS[native.arena]
    factory = getattr(importlib.import_module("workers.official_arenas.providers." + module), factory)
    return factory() if history_transform is None else factory(history_transform=history_transform)


def _prepare_scoring_runtime(native: NativeArenaConfig) -> None:
    if native.arena == "tooltalk":
        # The pinned official scorer initializes its semantic matcher lazily.
        # Load it before spending agent calls, and retain that same instance for
        # scoring. This uses the original scorer, weights and HF cache settings.
        utils = importlib.import_module("tooltalk.apis.utils")
        if utils._vectorize_text is None:
            utils._vectorize_text = utils._TextVectorizer()


def _schema_tools(state):
    from agentinputlab.connectors.tools import SchemaOnlyToolConnector, ToolSpec
    block = next((b for b in state.context if b.block_id == "official_tool_catalog"), None)
    if block is None:
        raise ValidationError("native visible state has no official tool catalog")
    data = json.loads(block.text)
    schemas = data["tool_schemas"] if isinstance(data, dict) else data
    specs = []
    for schema in schemas:
        function = schema.get("function", schema)
        specs.append(ToolSpec(name=function["name"], description=function.get("description", ""),
                              input_schema=function["parameters"], read_only=False))
    return SchemaOnlyToolConnector(tuple(specs))


def _run_case(run: RunConfig, provider, provider_config, selected, work: Path,
              native: NativeArenaConfig | None = None) -> dict[str, Any]:
    from agentinputlab.api import _state
    from agentinputlab.benchmark_configuration import _CapturedBackend
    from agentinputlab.observability import BudgetLedger, RunMonitor
    from agentinputlab.runner import InterventionRunner
    from workers.official_arenas.protocol import EXECUTABLE_PAYLOAD_PROTOCOL, merge_usage

    state = _state(dict(selected["visible_state"]))
    if state.budget != run.budget:
        raise ValidationError("native captured budget differs from the run configuration")
    if selected["protected"]["locked_dependency_hashes"] != provider_config["locked_dependency_hashes"]:
        raise ValidationError("native captured dependency binding differs")
    prior = None
    if native is not None and native.prior_intervention_usage is not None:
        from agentinputlab.arena_prior_usage import load, validate_case
        prior = validate_case(load(native.prior_intervention_usage, run), selected["checkpoint_id"], state)
    method = run.create_intervention()
    if not callable(getattr(method, "propose", None)):
        raise ValidationError("this native continuation requires an input-patch profile; use the composite's native runner")
    # Runtime construction uses the capture backend, so saved responses include
    # external runtime calls as well. There is no second budget wrapper here.
    captured = _CapturedBackend(run._backend())
    context = replace(run, backend={})._configured_context(
        InterventionContext(backend=captured, tools=_schema_tools(state)))
    ledger, monitor = BudgetLedger(run.budget), RunMonitor()
    history_policy = native.history_policy(run) if native is not None else None
    live_history = None
    if history_policy is not None:
        from agentinputlab.integrations.live_history import ACONLiveHistory

        live_history = ACONLiveHistory(method, context,
            per_call_output_tokens=native.max_output_tokens_per_call, prefix_policy=history_policy)
        provider = _provider(native, history_transform=live_history)
    else:
        context = replace(context, budget_ledger=ledger, monitor=monitor)
    row = {"case_id": selected["checkpoint_id"], "input_state_hash": state.state_hash,
           "status": "started", "automatic_retry": False}
    _save(work / "started.json", row)
    stage = "intervention"
    record = None
    role_events = None
    try:
        if live_history is None:
            record = InterventionRunner().run(method, state, context)
        materialized = to_jsonable(record.patch.apply(state) if record is not None else state)
        initial_usage = to_jsonable(record.usage if record is not None else ledger.usage)
        if prior is not None:
            from agentinputlab.arena_prior_usage import debit
            initial_usage = debit(prior, record)
            row["prior_intervention"] = {"receipt": to_jsonable(native.prior_intervention_usage),
                "source": prior["source"], "usage": prior["usage"],
                "historical_not_new_model_calls": True}
        payload = {"schema_version": 1, "protocol": EXECUTABLE_PAYLOAD_PROTOCOL,
                   "visible_state_hash": state.state_hash, "release_id": run.release_id,
                   "intervention_execution": {"input_state_hash": state.state_hash,
                       "output_state_hash": content_hash(materialized), "materialized_state": materialized,
                       "usage": initial_usage}}
        # Persist the completed intervention before any downstream call.
        _save(work / "intervention.json", {"record": record, "model_responses": captured.calls})
        stage = "native_episode_and_official_scorer"
        scope = nullcontext()
        if native is not None and native.arena in _TAU_PARAMETERS:
            from agentinputlab.tau_arena_configuration import role_calls
            scope = role_calls(native, run, provider_config, work)
        with scope as role_events:
            downstream = provider.run_arm(provider_config, replay_record={
                "checkpoint_id": selected["checkpoint_id"], "protected": selected["protected"]},
                arm_payload=payload, seed=run.seed,
                role="baseline" if run.release_id == "baseline.no_op" else "treatment", work_dir=work)
        row.update(status="scored", metrics=to_jsonable(downstream.metrics),
                   usage=merge_usage(initial_usage, to_jsonable(downstream.usage)),
                   native_trace=to_jsonable(downstream.trace), runtime_instance_id=downstream.runtime_instance_id)
        if prior is not None:
            row["current_run_usage"] = merge_usage(to_jsonable(record.usage), to_jsonable(downstream.usage))
    except Exception as exc:
        row.update(status="failed", stage=stage, error_type=type(exc).__name__,
                   error=_safe_error_message(exc))
    row.update(intervention=to_jsonable(record) if record is not None else None,
               intervention_usage=to_jsonable(ledger.usage),
               intervention_monitor=to_jsonable(monitor.list_snapshots()),
               model_responses=captured.calls, scorer_id=provider.scorer_id,
               scorer_revision=provider.scorer_revision, source_revision=provider.source_revision)
    if live_history is not None:
        # Providers already include dynamic history costs in downstream usage.
        # Retain decisions on failures too, without adding those costs twice.
        row.update(execution_mode="native_live_history",
                   live_history_records=to_jsonable(live_history.records))
    if role_events is not None:
        row["native_role_model_calls"] = to_jsonable(role_events)
        row["environment_usage_outside_agent_budget"] = True
    _save(work / "terminal.json", row)
    return row


def _implementation_digest() -> str:
    """Bind isolated workers to packaged code/resources, not a version label."""
    from workers.official_arenas import protocol

    digest = hashlib.sha256()
    for name, root, suffixes in (
        ("agentinputlab", Path(__file__).parent, {".py", ".typed", ".json", ".html", ".css", ".js"}),
        ("workers", Path(protocol.__file__).parents[1], {".py", ".md", ".txt", ".yaml", ".yml", ".json"}),
    ):
        for file in sorted(root.rglob("*")):
            if not file.is_file() or file.suffix not in suffixes or "__pycache__" in file.parts:
                continue
            digest.update((name + "/" + file.relative_to(root).as_posix()).encode())
            digest.update(b"\0" + hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def _worker(job_path: Path) -> int:
    job = _read(job_path)
    work = Path(job["work_dir"])
    try:
        if "worker_code_sha256" in job:
            identity = {"python_executable": sys.executable, "python_version": sys.version,
                        "implementation_sha256": _implementation_digest()}
            _save(work / "worker-runtime.json", identity)
            if identity["implementation_sha256"] != job["worker_code_sha256"]:
                raise ValidationError("isolated worker toolkit implementation differs; install the same wheel")
        run = RunConfig.from_file(job["run_config"])
        native = NativeArenaConfig.from_file(job["arena_config"])
        if native.scorer_cache is not None:
            # Preparation is the explicit download step. Evaluation must not
            # fetch a scorer after consuming agent calls, or use another cache.
            os.environ.update(HF_HUB_CACHE=native.scorer_cache, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        checkout = Path(native.checkout)
        if not checkout.is_dir():
            raise ValidationError("native checkout directory is missing")
        os.chdir(checkout)
        sys.path[:0] = [str(checkout / "src"), str(checkout)]
        provider = _provider(native)
        config = native.provider_config(run)
        blockers = provider.doctor(config)
        if blockers:
            raise ValidationError("native provider preflight: " + "; ".join(blockers))
        if job["mode"] == "capture":
            if native.arena in _TAU_PARAMETERS:
                from agentinputlab.tau_arena_configuration import capture
                return capture(native, run, provider, config, job, work)
            rows = provider.capture(config, output_root=work)
            if len({r["checkpoint_id"] for r in rows}) != len(rows):
                raise ValidationError("native inventory contains duplicate case IDs")
            _save(work / "inventory.json", rows)
            _save(work / "terminal.json", {"status": "captured", "case_ids": [r["checkpoint_id"] for r in rows],
                                          "inventory_sha256": _sha(work / "inventory.json")})
            return 0
        if job["mode"] != "case":
            raise ValidationError("unknown native worker mode")
        inventory = Path(job["inventory"])
        if _sha(inventory) != job["inventory_sha256"]:
            raise ValidationError("native inventory checksum differs")
        selected = next(r for r in _read(inventory, limit=256 * 1024 * 1024) if r["checkpoint_id"] == job["case_id"])
        _prepare_scoring_runtime(native)
        return 0 if _run_case(run, provider, config, selected, work, native=native)["status"] == "scored" else 1
    except Exception as exc:
        if not (work / "terminal.json").exists():
            _save(work / "terminal.json", {"status": "failed", "stage": "native_preflight",
                  "error_type": type(exc).__name__, "error": _safe_error_message(exc),
                  "automatic_retry": False})
        return 1


def _launch(job: dict[str, Any], path: Path) -> dict[str, Any]:
    options: dict[str, Any] = {}
    command = [sys.executable, "-m", "agentinputlab.arena_configuration", str(path)]
    if job.get("worker_python") is not None:
        python = job["worker_python"]
        if not Path(python).is_file() or not os.access(python, os.X_OK):
            raise ValidationError("worker_python is missing or not executable; no fallback was attempted")
        job = {**job, "worker_code_sha256": job.get("worker_code_sha256", _implementation_digest())}
        # A fixed bootstrap rejects older/different installed code before any
        # native episode. Never inject the host's packages into the worker env.
        command = [python, "-c",
            "import sys; from pathlib import Path\n"
            "from agentinputlab.arena_configuration import _worker, _implementation_digest\n"
            "if _implementation_digest() != sys.argv[2]:\n"
            "    raise SystemExit('worker toolkit implementation differs; install the same wheel')\n"
            "raise SystemExit(_worker(Path(sys.argv[1])))", str(path), job["worker_code_sha256"]]
        environment = dict(os.environ, PYTHONNOUSERSITE="1")
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        options["env"] = environment
    _save(path, job)
    work = Path(job["work_dir"])
    # Fresh processes isolate native imports, cwd and mutable environment state.
    with (work / "worker.log").open("xb") as log:
        result = subprocess.run(command, cwd=path.parent, stdout=log,
                                stderr=subprocess.STDOUT, check=False, **options)
    terminal = work / "terminal.json"
    if not terminal.exists():
        _save(terminal, {"status": "failed", "stage": "worker_exit", "returncode": result.returncode,
                         "automatic_retry": False})
    row = _read(terminal, limit=256 * 1024 * 1024)
    if result.returncode and row.get("status") in {"scored", "captured"}:
        raise ValidationError("native worker exit contradicts its terminal receipt")
    return row


def evaluate_arena(config: RunConfig, native: NativeArenaConfig, *, output_dir: str | Path,
                   case_ids: Sequence[str] | None = None, limit: int | None = None,
                   inventory_only: bool = False, capture_only: bool = False) -> dict[str, Any]:
    """Execute explicitly selected cases once, or list native cases without model calls."""
    if not isinstance(inventory_only, bool):
        raise ValidationError("inventory_only must be boolean")
    if not isinstance(capture_only, bool) or (capture_only and inventory_only):
        raise ValidationError("capture_only must be boolean and cannot be combined with inventory_only")
    if inventory_only:
        if case_ids is not None or limit is not None:
            raise ValidationError("inventory-only mode does not execute a selection")
    elif ((case_ids is None) == (limit is None)
          or (limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1))):
        raise ValidationError("choose case_ids or a positive limit, not both")
    if case_ids is not None and (isinstance(case_ids, (str, bytes)) or not isinstance(case_ids, Sequence)
            or not case_ids or any(not isinstance(x, str) or not x.strip() for x in case_ids)
            or len(set(case_ids)) != len(case_ids)):
        raise ValidationError("case_ids must be unique non-empty strings")
    config.validate()
    native.history_policy(config)
    root = Path(output_dir).absolute()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        raise ValidationError("output_dir already exists; inspect its receipts instead of rerunning") from None
    _save(root / "run.json", config.to_dict())
    native_value = to_jsonable(native)
    if native.prior_intervention_usage is not None:
        native_value["prior_intervention_usage"]["path"] = str(Path(native.prior_intervention_usage["path"]).expanduser().absolute())
    if native.capture_bundle is not None:
        native_value["capture_bundle"]["path"] = str(Path(native.capture_bundle["path"]).expanduser().absolute())
    _save(root / "arena.json", native_value)
    capture = root / "capture"
    capture.mkdir()
    common = {"run_config": str(root / "run.json"), "arena_config": str(root / "arena.json")}
    if native.worker_python is not None:
        common.update(worker_python=native.worker_python, worker_code_sha256=_implementation_digest())
    inventory = _launch({**common, "mode": "capture", "work_dir": str(capture),
                         "inventory_only": inventory_only, "case_ids": list(case_ids) if case_ids else None,
                         "limit": limit}, root / "capture-job.json")
    if inventory["status"] != "captured":
        raise ValidationError(f"native preflight failed; inspect {capture / 'terminal.json'}")
    available = inventory["case_ids"]
    chosen = list(case_ids) if case_ids is not None else available[:limit]
    if not inventory_only and (not chosen or set(chosen) - set(available)):
        raise ValidationError("requested cases are absent from the native inventory; inspect capture/terminal.json")
    terminals = []
    if capture_only:
        failures = inventory.get("capture_failures", {})
        terminals = [failures[x] if x in failures else {"case_id": x, "status": "captured",
                     "receipt": str(capture / "terminal.json")} for x in chosen]
    elif not inventory_only:
        for i, case_id in enumerate(chosen):
            if case_id in inventory.get("capture_failures", {}):
                terminals.append(inventory["capture_failures"][case_id])
                continue
            work = root / f"case-{i:04d}"
            work.mkdir()
            row = _launch({**common, "mode": "case", "work_dir": str(work), "case_id": case_id,
                           "inventory": inventory.get("inventory_path", str(capture / "inventory.json")),
                           "inventory_sha256": inventory["inventory_sha256"]}, root / f"job-{i:04d}.json")
            terminals.append({"case_id": case_id, "status": row["status"],
                              "metrics": row.get("metrics"), "receipt": str(work / "terminal.json")})
    summary = {"arena": native.arena, "release_id": config.release_id, "inventory_only": inventory_only,
               "capture_only": capture_only,
               "available_case_ids": available, "selected_count": len(terminals),
               "scored_count": sum(x["status"] == "scored" for x in terminals),
               "captured_count": sum(x["status"] == "captured" for x in terminals),
               "failed_count": sum(x["status"] not in {"scored", "captured"} for x in terminals),
               "cases": terminals, "automatic_retry": False, "paired_comparison": False}
    _save(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated native arena worker (internal).")
    parser.add_argument("job", type=Path)
    raise SystemExit(_worker(parser.parse_args().job.absolute()))
